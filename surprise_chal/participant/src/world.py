"""Persistent world model for the Surprise agent (AE-style memory).

The observation is **stateless with zero fog memory**: a tile that leaves vision
vanishes next turn, and the obs never carries production queues (not even ours).
So we carry our own state across turns. The asymmetry that makes this cheap:
**terrain never changes mid-game**, so every tile we ever see stays valid forever.

`WorldModel` is pure state + parsing. The planner reads it; the actuator does the
hex math. Nothing here emits actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.constants import BUILDING_STATS
from engine.hex_grid import HexCoord, HexGrid

# building types whose completion lets us produce units onto adjacent tiles
PRODUCTION_BUILDINGS = ("Barracks", "Factory", "Airbase")
AIR_TYPES = ("Bomber", "Fighter", "Airbase")


def _is_building(entity: dict) -> bool:
    return entity.get("type") in BUILDING_STATS


@dataclass
class Enemy:
    """A last-known sighting of an enemy entity (persists through fog)."""

    eid: str
    owner_id: str
    type: str
    q: int
    r: int
    hp: int
    is_building: bool
    last_seen: int

    @property
    def coord(self) -> HexCoord:
        return HexCoord(self.q, self.r)


@dataclass
class PendingOrder:
    """A unit we believe we queued (the obs won't tell us). Tracked conservatively
    — only recorded after our own affordability/validity check passed, so silent
    engine no-ops don't desync the model. Decremented each turn; dropped at 0."""

    unit_type: str
    turns_left: int
    target: tuple[int, int]


class WorldModel:
    """All cross-turn memory. `update(obs)` refreshes it each turn."""

    def __init__(self) -> None:
        self.pid: str = ""
        self.grid: HexGrid = HexGrid(35, 30)
        self.turn: int = -1
        self.gold: int = 0
        self.max_turns: int = 300

        # ── persistent memory ──
        self.terrain: dict[tuple[int, int], str] = {}  # (q,r) -> terrain name
        self.enemies: dict[str, Enemy] = {}  # eid -> last-known sighting
        self.pending: dict[str, list[PendingOrder]] = {}  # building_id -> orders

        # ── per-turn (recomputed every update) ──
        self.own_units: list[dict] = []
        self.own_buildings: list[dict] = []
        self.visible_enemies: list[dict] = []
        self.occupied: set[HexCoord] = set()
        self.visible: set[HexCoord] = set()
        self.visible_free: set[HexCoord] = set()
        self.known_players: list[str] = []
        self.treaties: list[dict] = []
        self.incoming: list[dict] = []
        self.peace_partners: set[str] = set()
        self.breaking_partners: set[str] = set()  # active break countdown → war now
        self.global_chat: list[dict] = []
        self.private_chat: list[dict] = []

    # ── update ────────────────────────────────────────────────────────────────

    def update(self, obs: dict) -> None:
        new_turn = int(obs.get("turn_number", 0))
        # a fresh game (container reuse / restart): wipe persistent memory
        if new_turn < self.turn:
            self.terrain.clear()
            self.enemies.clear()
            self.pending.clear()

        self.pid = obs["player_id"]
        self.turn = new_turn
        self.gold = int(obs.get("resources", {}).get("gold", 0))
        self.max_turns = int(obs.get("max_turns", 300))
        self.grid = HexGrid(int(obs.get("map_width", 35)), int(obs.get("map_height", 30)))

        self.own_units = []
        self.own_buildings = []
        self.visible_enemies = []
        self.occupied = set()
        self.visible = set()
        self.visible_free = set()

        for tile in obs.get("visible_tiles", []):
            c = HexCoord(tile["q"], tile["r"])
            self.visible.add(c)
            self.terrain[(c.q, c.r)] = tile.get("terrain", "normal")
            ents = tile.get("entities", [])
            if not ents:
                self.visible_free.add(c)
            for e in ents:
                self.occupied.add(HexCoord(e["q"], e["r"]))
                if e.get("owner_id") == self.pid:
                    (self.own_buildings if _is_building(e) else self.own_units).append(e)
                else:
                    self.visible_enemies.append(e)

        # refresh last-known enemy sightings
        for e in self.visible_enemies:
            self.enemies[e["id"]] = Enemy(
                eid=e["id"],
                owner_id=e.get("owner_id", "?"),
                type=e["type"],
                q=e["q"],
                r=e["r"],
                hp=int(e.get("hp", 0)),
                is_building=_is_building(e),
                last_seen=self.turn,
            )
        # forget very stale enemy *units* (they move); keep buildings far longer
        # (terrain-static, but bases can be destroyed by others — still useful intel)
        for eid in list(self.enemies):
            en = self.enemies[eid]
            age = self.turn - en.last_seen
            if (not en.is_building and age > 15) or age > 80:
                del self.enemies[eid]

        self.known_players = list(obs.get("known_players", []))
        self.treaties = list(obs.get("treaties", []))
        self.incoming = list(obs.get("incoming_treaty_proposals", []))
        self.global_chat = list(obs.get("global_chat", []))
        self.private_chat = list(obs.get("private_chat", []))

        # ACTIVE peace blocks attacks; a BREAKING treaty (break_in_turns set) is
        # war THIS turn per the verified quirk — do NOT count it as protection.
        self.peace_partners = set()
        self.breaking_partners = set()
        for t in self.treaties:
            partner = t.get("partner_id")
            if partner is None:
                continue
            if t.get("breaking_in_turns") is None:
                self.peace_partners.add(partner)
            else:
                self.breaking_partners.add(partner)

        self._tick_pending()

    def _tick_pending(self) -> None:
        """Advance our tracked production orders one turn; drop completed/orphaned."""
        live_ids = {b["id"] for b in self.own_buildings}
        for bid in list(self.pending):
            if bid not in live_ids:
                del self.pending[bid]
                continue
            remaining: list[PendingOrder] = []
            for o in self.pending[bid]:
                o.turns_left -= 1
                if o.turns_left > 0:
                    remaining.append(o)
            if remaining:
                self.pending[bid] = remaining
            else:
                del self.pending[bid]

    def record_order(self, building_id: str, unit_type: str, turns: int, target: HexCoord) -> None:
        self.pending.setdefault(building_id, []).append(
            PendingOrder(unit_type=unit_type, turns_left=turns, target=(target.q, target.r))
        )

    def pending_count(self, unit_type: str | None = None) -> int:
        return sum(
            1
            for orders in self.pending.values()
            for o in orders
            if unit_type is None or o.unit_type == unit_type
        )

    # ── convenience accessors ───────────────────────────────────────────────────

    @property
    def my_bases(self) -> list[dict]:
        """All own Bases (complete or under construction)."""
        return [b for b in self.own_buildings if b["type"] == "Base"]

    @property
    def my_complete_bases(self) -> list[dict]:
        """Only complete Bases self-spot and keep us alive."""
        return [b for b in self.own_buildings if b["type"] == "Base" and b.get("is_complete", False)]

    @property
    def complete_buildings(self) -> list[dict]:
        return [b for b in self.own_buildings if b.get("is_complete", False)]

    def producers(self, kinds: tuple[str, ...]) -> list[dict]:
        return [b for b in self.complete_buildings if b["type"] in kinds]

    def count_buildings(self, btype: str) -> int:
        return sum(1 for b in self.own_buildings if b["type"] == btype)

    @property
    def air_threat(self) -> bool:
        """Any enemy air capability seen recently (drives Fighter/Airbase rush)."""
        return any(en.type in AIR_TYPES for en in self.enemies.values())

    def is_free(self, c: HexCoord) -> bool:
        """Visible and empty (so we actually know it's placeable)."""
        return c in self.visible and c not in self.occupied

    def terrain_at(self, c: HexCoord) -> str:
        return self.terrain.get((c.q, c.r), "normal")

    def entry_cost(self, c: HexCoord) -> int:
        return 2 if self.terrain_at(c) == "difficult" else 1
