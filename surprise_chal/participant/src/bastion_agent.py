"""Bastion — survival-first defensive agent (new candidate; old agents untouched).

Built from what the evaluate_agent.py baselines showed (hard opponents):
  * algo   — eliminated ~turn 30. Its "deploy near the nearest enemy" logic pulls
    the garrison off the bases, and the opening is too slow for the grinder.
  * shadow — eliminated ~turn 56 sitting on 3500 unspent gold. The standing-ring
    idea is right, but it chronically under-builds (1 mine, 2 bases, no army).
  * fast_expand — the only sparring personality that survives (50%): mass base
    redundancy + mass mines + continuous production.

Bastion combines the three: shadow's standing denial rings + persistent memory,
algo's scout-teleport remote base founding, and fast_expand's spending aggression.
Primary objective is SURVIVAL — keep ≥3 complete Bases alive, spread far apart,
each ringed by cheap bodies. Everything else (economy, buildings, kills) exists
only to fund and protect that.

Engine facts this leans on (verified in turn_processor.py):
  * Elimination = zero COMPLETE Bases; under-construction Bases don't save you,
    and complete Bases always self-report in our obs (vision_bonus 3).
  * Teleport quirk: a 1-hop move [here, dest] is legal for any unit if the dest's
    entry cost ≤ movement_range — instant ring refills and remote founding.
  * One entity per tile + range-1 attacks ⇒ a fully occupied ring physically
    denies Bombers/Tanks/Infantry the adjacency they need to hit the Base.
  * Attacks resolve before moves, from the pre-move tile ⇒ rings must be standing.
  * Production queues tick in parallel; gold is spent at enqueue; a completed unit
    with no free spawn tile next to its building is LOST — cap enqueues by space.
  * Artillery splash ignores ownership/treaties — never splash our own ring.
  * A BREAKING treaty is war THIS turn (is_peace only returns True for ACTIVE).
"""

from __future__ import annotations

from agent_base import PlayerAgent
from engine.actions import (
    ActionPayload,
    AttackAction,
    ConstructBuildingAction,
    MoveAction,
    ProduceUnitAction,
    ProposeTreatyAction,
    RespondTreatyAction,
)
from engine.constants import BUILDING_STATS, TREATY_CUTOFF_TURN, UNIT_STATS
from engine.hex_grid import HexCoord, HexGrid

# how dangerous an enemy entity is TO OUR BASES (drives threat scores + focus fire)
_THREAT_WEIGHT = {
    "Bomber": 5.0,      # ×4 vs buildings: two delete a Base in one turn
    "Artillery": 4.0,   # range 3 + treaty-bypassing splash
    "Tank": 3.0,
    "Fighter": 2.0,
    "Infantry": 1.0,
    "Scout": 0.4,
    "Medic": 0.3,
}
_ATTACK_PRIORITY = {
    "Bomber": 0,
    "Artillery": 1,
    "Fighter": 2,
    "Tank": 3,
    "Infantry": 4,
    "Scout": 5,
    "Medic": 6,
    "Airbase": 7,
    "Base": 8,
    "Factory": 9,
    "Barracks": 10,
    "Mine": 11,
}
_AIR_TYPES = ("Bomber", "Fighter", "Airbase")
_PRODUCTION_BUILDINGS = ("Barracks", "Factory", "Airbase")
_BASE_COST = BUILDING_STATS["Base"].gold_cost


def _is_building(entity: dict) -> bool:
    return entity.get("type") in BUILDING_STATS


def _c(entity: dict) -> HexCoord:
    return HexCoord(entity["q"], entity["r"])


class _Sighting:
    __slots__ = ("eid", "owner", "type", "coord", "hp", "building", "seen")

    def __init__(self, eid, owner, type_, coord, hp, building, seen):
        self.eid = eid
        self.owner = owner
        self.type = type_
        self.coord = coord
        self.hp = hp
        self.building = building
        self.seen = seen


class _Memory:
    """Cross-turn memory. Terrain never changes, so every seen tile stays valid."""

    def __init__(self) -> None:
        self.turn = -1
        self.terrain: dict[HexCoord, str] = {}
        self.sightings: dict[str, _Sighting] = {}
        # building_id -> list of [unit_type, turns_left] we believe we queued
        self.pending: dict[str, list[list]] = {}
        # (q,r) -> turn we ordered a Base there (covers the invisible 5-turn build)
        self.base_orders: dict[HexCoord, int] = {}
        # (q,r) -> turn a base of ours was lost there (kill zones: do not refound)
        self.lost_bases: dict[HexCoord, int] = {}
        self.known_base_coords: set[HexCoord] = set()
        self.last_proposed: dict[str, int] = {}

    def reset(self) -> None:
        self.terrain.clear()
        self.sightings.clear()
        self.pending.clear()
        self.base_orders.clear()
        self.lost_bases.clear()
        self.known_base_coords.clear()
        self.last_proposed.clear()


class AlgoAgent(PlayerAgent):
    def __init__(self) -> None:
        self.mem = _Memory()

    async def decide(self, observation: dict) -> ActionPayload:
        turn = int(observation.get("turn_number", 0))
        if turn < self.mem.turn:  # fresh game in a reused container
            self.mem.reset()
        t = _Turn(self.mem, observation)
        actions = t.run()
        self.mem.turn = turn
        return ActionPayload(player_id=t.pid, turn_number=turn, actions=actions)


class _Turn:
    """All per-turn computation. Bounded: no whole-map pathfinding, ~ms total."""

    def __init__(self, mem: _Memory, obs: dict) -> None:
        self.mem = mem
        self.obs = obs
        self.pid: str = obs["player_id"]
        self.turn: int = int(obs.get("turn_number", 0))
        self.max_turns: int = int(obs.get("max_turns", 300))
        self.budget: int = int(obs.get("resources", {}).get("gold", 0))
        self.grid = HexGrid(int(obs.get("map_width", 35)), int(obs.get("map_height", 30)))

        self.own_units: list[dict] = []
        self.own_buildings: list[dict] = []
        self.visible_enemies: list[dict] = []
        self.visible: set[HexCoord] = set()
        self.occupied: set[HexCoord] = set()
        self._parse_tiles()
        self._update_memory()

        # tiles spoken for this turn (moves land / builds rise / units spawn here)
        self.reserved: set[HexCoord] = set(self.occupied)
        self.moved_units: set[str] = set()
        self.actions: list = []

        self.bases = [b for b in self.own_buildings if b["type"] == "Base"]
        self.complete_bases = [b for b in self.bases if b.get("is_complete", False)]
        self.pending_bases = [b for b in self.bases if not b.get("is_complete", False)]
        self.complete_buildings = [
            b for b in self.own_buildings if b.get("is_complete", False)
        ]
        self.ring_tiles = self._ring_tiles()
        self.base_threat = self._base_threats()

        # treaty bookkeeping: ACTIVE peace blocks attacks; BREAKING is war now
        self.peace_partners: set[str] = set()
        for treaty in obs.get("treaties", []):
            partner = treaty.get("partner_id")
            if partner and treaty.get("breaking_in_turns") is None:
                self.peace_partners.add(partner)

    # ── observation parsing + memory ────────────────────────────────────────────

    def _parse_tiles(self) -> None:
        for tile in self.obs.get("visible_tiles", []):
            c = HexCoord(tile["q"], tile["r"])
            self.visible.add(c)
            self.mem.terrain[c] = tile.get("terrain", "normal")
            for e in tile.get("entities", []):
                self.occupied.add(HexCoord(e["q"], e["r"]))
                if e.get("owner_id") == self.pid:
                    (self.own_buildings if _is_building(e) else self.own_units).append(e)
                else:
                    self.visible_enemies.append(e)

    def _update_memory(self) -> None:
        mem = self.mem
        seen_ids = set()
        for e in self.visible_enemies:
            seen_ids.add(e["id"])
            mem.sightings[e["id"]] = _Sighting(
                e["id"], e.get("owner_id", "?"), e["type"], _c(e),
                int(e.get("hp", 0)), _is_building(e), self.turn,
            )
        for eid in list(mem.sightings):
            s = mem.sightings[eid]
            age = self.turn - s.seen
            # a sighting whose tile we can see but the entity isn't there → gone
            if s.coord in self.visible and eid not in seen_ids:
                del mem.sightings[eid]
            elif (not s.building and age > 12) or age > 90:
                del mem.sightings[eid]

        # purge Base orders that completed (the Base now self-reports) or failed
        own_base_coords = {
            _c(b) for b in self.own_buildings if b["type"] == "Base"
        }
        for coord in list(mem.base_orders):
            ordered = mem.base_orders[coord]
            if coord in own_base_coords:
                del mem.base_orders[coord]
            elif self.turn - ordered > 7:  # build takes 5; allow slack, then distrust
                del mem.base_orders[coord]

        # kill-zone memory: a base of ours confirmed gone (tile visible, base
        # absent) was destroyed there — refounding into the same grinder just
        # feeds it 300g at a time. Out-of-vision sites are left unjudged.
        for coord in list(mem.known_base_coords):
            if coord in own_base_coords:
                continue
            if coord in self.visible:
                mem.lost_bases[coord] = self.turn
                mem.known_base_coords.discard(coord)
        mem.known_base_coords.update(own_base_coords)
        for coord in list(mem.lost_bases):
            if self.turn - mem.lost_bases[coord] > 25:
                del mem.lost_bases[coord]

        # tick our believed production queue
        live = {b["id"] for b in self.own_buildings}
        for bid in list(mem.pending):
            if bid not in live:
                del mem.pending[bid]
                continue
            keep = []
            for order in mem.pending[bid]:
                order[1] -= 1
                if order[1] > 0:
                    keep.append(order)
            if keep:
                mem.pending[bid] = keep
            else:
                del mem.pending[bid]

    # ── shared helpers ───────────────────────────────────────────────────────────

    def _dist(self, a: HexCoord, b: HexCoord) -> int:
        return self.grid.distance(a, b)

    def _terrain(self, c: HexCoord) -> str:
        return self.mem.terrain.get(c, "unknown")

    def _entry_cost(self, c: HexCoord) -> int:
        t = self._terrain(c)
        if t == "difficult":
            return 2
        if t == "unknown":
            return 2  # be conservative about tiles we've never seen
        return 1

    def _free_now(self, c: HexCoord) -> bool:
        return c in self.visible and c not in self.reserved

    def _believed_free(self, c: HexCoord) -> bool:
        """For tiles outside current vision: free unless something was last seen there."""
        if c in self.visible:
            return c not in self.reserved
        if c in self.reserved:
            return False
        if c not in self.mem.terrain:
            return False  # never seen: terrain unknown, don't gamble moves on it
        return all(s.coord != c for s in self.mem.sightings.values())

    def _ring_tiles(self) -> set[HexCoord]:
        out: set[HexCoord] = set()
        for b in self.bases:  # pending bases need their rings held too
            out.update(self.grid.neighbors(_c(b)))
        return out

    def _fresh_enemies(self, max_age: int = 4) -> list[_Sighting]:
        return [
            s for s in self.mem.sightings.values()
            if self.turn - s.seen <= max_age
        ]

    def _base_threats(self) -> dict[str, float]:
        """Threat score per own base (complete or pending) from fresh sightings."""
        scores: dict[str, float] = {}
        fresh = self._fresh_enemies()
        for b in self.bases:
            bc = _c(b)
            score = 0.0
            for s in fresh:
                if s.building:
                    continue
                d = self._dist(bc, s.coord)
                if d > 7:
                    continue
                w = _THREAT_WEIGHT.get(s.type, 0.5)
                score += w * (8 - d)
            scores[b["id"]] = score
        return scores

    def _count_buildings(self, kind: str) -> int:
        return sum(1 for b in self.own_buildings if b["type"] == kind)

    def _unit_count(self, kind: str) -> int:
        have = sum(1 for u in self.own_units if u["type"] == kind)
        have += sum(
            1 for orders in self.mem.pending.values() for o in orders if o[0] == kind
        )
        return have

    def _total_base_sites(self) -> int:
        coords = {_c(b) for b in self.bases}
        coords.update(self.mem.base_orders)
        return len(coords)

    # ── main flow ────────────────────────────────────────────────────────────────

    def run(self) -> list:
        self._diplomacy()
        self._attacks()
        self._found_bases()
        # under pressure, ring bodies and counter-battery outrank new buildings
        # (the seed-68 death bought Mine #4 while its last two bases sat at 20 hp)
        if self._under_pressure():
            self._production()
            self._constructs()
        else:
            self._constructs()
            self._production()
        self._movements()
        return self.actions

    def _under_pressure(self) -> bool:
        if any(score >= 8.0 for score in self.base_threat.values()):
            return True
        return any(
            b.get("hp", 300) < 180 for b in self.complete_bases
        )

    # ── diplomacy: universal peace, free upside for a turtle ────────────────────

    def _diplomacy(self) -> None:
        if self.turn >= TREATY_CUTOFF_TURN:
            return
        bound = {t.get("partner_id") for t in self.obs.get("treaties", [])}
        for prop in self.obs.get("incoming_treaty_proposals", []):
            proposer = prop.get("proposer_id")
            if not proposer:
                continue
            self.actions.append(
                RespondTreatyAction(
                    proposing_player_id=proposer,
                    treaty_type=prop.get("treaty_type", "peace"),
                    accept=True,
                )
            )
            bound.add(proposer)
        for other in self.obs.get("known_players", []):
            if other in bound:
                continue
            if self.turn - self.mem.last_proposed.get(other, -1000) < 10:
                continue
            self.actions.append(ProposeTreatyAction(target_player_id=other))
            self.mem.last_proposed[other] = self.turn

    # ── attacks: focus-fire what can hurt the bases ──────────────────────────────

    def _attacks(self) -> None:
        targets = [
            e for e in self.visible_enemies
            if e.get("owner_id") not in self.peace_partners
        ]
        if not targets:
            return
        own_coords = {_c(e) for e in self.own_units} | {
            _c(b) for b in self.own_buildings
        }
        for u in self.own_units:
            rng = u.get("attack_range", 0)
            if rng < 1 or u.get("attack_power", 0) < 1:
                continue
            here = _c(u)
            in_range = [
                t for t in targets if 0 < self._dist(here, _c(t)) <= rng
            ]
            if u["type"] == "Artillery":
                # splash ignores ownership: never fire where our own ring would eat it
                in_range = [
                    t for t in in_range
                    if not any(
                        c in own_coords
                        for c in self.grid.ring(_c(t), 1)
                    )
                ]
                if not in_range:
                    continue
                enemy_coords = {_c(t) for t in targets}
                # kill the healers first (60 atk one-shots a 60 hp Medic — enemy
                # medics are what make the ring attrition war unwinnable), then
                # counter-battery, then whatever the splash hits most of
                def arty_score(t: dict) -> tuple:
                    splash = sum(
                        1 for c in self.grid.ring(_c(t), 1) if c in enemy_coords
                    )
                    prio = {"Medic": 0, "Artillery": 1, "Bomber": 2}.get(
                        t["type"], _ATTACK_PRIORITY.get(t["type"], 12)
                    )
                    return (prio, -splash, t.get("hp", 999))

                best = min(in_range, key=arty_score)
            else:
                if not in_range:
                    continue
                best = min(
                    in_range,
                    key=lambda t: (
                        _ATTACK_PRIORITY.get(t["type"], 12),
                        t.get("hp", 999),
                        self._dist(here, _c(t)),
                    ),
                )
            self.actions.append(AttackAction(unit_id=u["id"], target=_c(best)))

    # ── base founding: the primary survival lever ────────────────────────────────

    def _base_target(self) -> int:
        want = 3
        if self.turn >= 30:
            want = 4
        if self.turn >= 70:
            want = 5
        if self.turn >= 120:
            want = 6
        if self.turn >= 170:
            want = 7
        if self._air_threat():
            # bombers two-shot a base and bases cannot be repaired: under an air
            # threat redundancy is a consumable — stock more of it
            want += 1
        # surplus gold converts to redundancy — never die rich
        return min(9, want + self.budget // 2000)

    def _found_bases(self) -> None:
        # A complete base under sustained fire is already half-lost: count only
        # healthy bases when judging the crisis, and refound BEFORE one dies.
        healthy = [b for b in self.complete_bases if b.get("hp", 300) >= 120]
        emergency = len(healthy) < 2 and self.turn >= 8
        sites_now = self._total_base_sites()
        want = self._base_target()
        if not emergency and sites_now >= want:
            return
        # economy bootstraps before redundancy: a Mine must at least be ordered
        if not emergency and self._count_buildings("Mine") < 1:
            return

        foundings = 2 if (emergency or self.budget >= 2 * _BASE_COST + 400) else 1
        foundings = min(foundings, max(0, want - sites_now) if not emergency else 2)
        scouts = [
            u for u in self.own_units
            if u["type"] == "Scout" and u["id"] not in self.moved_units
        ]
        for _ in range(foundings):
            if self.budget < _BASE_COST:
                return
            # remote founding first when a scout is free — distance is the best
            # armour a new base can have
            if scouts:
                scout = scouts[-1]
                placed = self._remote_base_site(scout)
                if placed is not None:
                    scouts.pop()
                    landing, site = placed
                    here = _c(scout)
                    self.actions.append(
                        MoveAction(unit_id=scout["id"], path=[here, landing])
                    )
                    self.moved_units.add(scout["id"])
                    self.reserved.discard(here)
                    self.reserved.add(landing)
                    self._emit_base(site)
                    continue
            # otherwise take the best visible site, relaxing standards rather
            # than founding nothing (seed-68 death: clearance 5 never passed and
            # the agent sat on 2 dying bases for 25 turns without founding)
            spot = None
            for min_sep, clearance in ((4, 5), (3, 3), (3, 1)):
                spot = self._visible_base_site(min_sep, clearance)
                if spot is not None:
                    break
            if spot is None and not healthy:
                # truly desperate: any free visible tile beats elimination
                spot = next(
                    (c for c in self.visible if self._free_now(c) and c not in self.ring_tiles),
                    None,
                )
            if spot is None:
                return
            self._emit_base(spot)

    def _emit_base(self, coord: HexCoord) -> None:
        self.actions.append(ConstructBuildingAction(building_type="Base", coord=coord))
        self.budget -= _BASE_COST
        self.reserved.add(coord)
        self.mem.base_orders[coord] = self.turn

    def _site_score(self, c: HexCoord) -> tuple:
        base_coords = [_c(b) for b in self.bases] + list(self.mem.base_orders)
        enemy_coords = [s.coord for s in self.mem.sightings.values()]
        enemy_d = min((self._dist(c, e) for e in enemy_coords), default=12)
        own_d = min((self._dist(c, b) for b in base_coords), default=8)
        terr = self._terrain(c)
        # a Base on rich yields 50/turn with 300 hp — economy AND redundancy.
        # But distance from known enemies comes first: survival buys economy time.
        bonus = 3 if terr == "rich_resource" else (1 if terr == "concealment" else 0)
        return (min(enemy_d, 10), bonus, min(own_d, 8))

    def _enemy_clearance(self, c: HexCoord, max_age: int = 8) -> int:
        fresh = [s.coord for s in self._fresh_enemies(max_age=max_age)]
        return min((self._dist(c, e) for e in fresh), default=99)

    def _in_kill_zone(self, c: HexCoord) -> bool:
        return any(
            self._dist(c, lost) <= 5 for lost in self.mem.lost_bases
        )

    def _visible_base_site(self, min_sep: int, clearance: int) -> HexCoord | None:
        base_coords = [_c(b) for b in self.bases] + list(self.mem.base_orders)
        cands = [
            c for c in self.visible
            if self._free_now(c)
            and c not in self.ring_tiles
            and all(self._dist(c, b) >= min_sep for b in base_coords)
            and self._enemy_clearance(c) >= clearance
            and not self._in_kill_zone(c)
        ]
        if not cands:
            return None
        return max(cands, key=self._site_score)

    def _remote_base_site(self, scout: dict) -> tuple[HexCoord, HexCoord] | None:
        """Pick (scout landing, base site) from remembered terrain, far from
        everything. The scout teleports adjacent → the site is in its vision when
        the construct is validated (moves resolve in phase 1, builds in phase 2)."""
        base_coords = [_c(b) for b in self.bases] + list(self.mem.base_orders)
        cands: list[HexCoord] = []
        for clearance in (6, 4):
            # sample remembered tiles outside current vision (bounded scan)
            for c in self.mem.terrain:
                if c in self.visible or not self._believed_free(c):
                    continue
                if any(self._dist(c, b) < 5 for b in base_coords):
                    continue
                if self._enemy_clearance(c) < clearance:
                    continue
                if self._in_kill_zone(c):
                    continue
                cands.append(c)
            if cands:
                break
        if not cands:
            return None
        site = max(cands, key=self._site_score)
        # land on a believed-free neighbour; prefer concealment (hides the scout)
        neighbours = [
            n for n in self.grid.neighbors(site)
            if self._believed_free(n) and self._entry_cost(n) <= 3
        ]
        if not neighbours:
            return None
        landing = max(
            neighbours, key=lambda n: (self._terrain(n) == "concealment",)
        )
        return landing, site

    # ── constructs: economy + production scaffolding ─────────────────────────────

    def _build_targets(self) -> list[str]:
        """Ordered queue of non-Base buildings still wanted (most urgent first).

        Lesson from the seed-68 trace: with Mines ahead of everything, the budget
        never reached a Factory and the defense stayed melee-only infantry — which
        loses the ring attrition war against infantry+medic blobs. Combined arms
        (artillery behind the rings) comes before economy width."""
        count = self._count_buildings
        queue: list[str] = []
        if count("Barracks") < 1:
            queue.append("Barracks")
        if count("Mine") < 1:
            queue.append("Mine")
        # Factory early: artillery is what breaks the infantry grind
        if count("Factory") < 1 and (self.turn >= 10 or count("Mine") >= 1):
            queue.append("Factory")
        if count("Mine") < 2 and self.turn >= 8:
            queue.append("Mine")

        mine_target = min(12, min(8, 2 + self.turn // 12) + self.budget // 1500)
        factory_target = 1
        if self.turn >= 60 or self.budget >= 1200:
            factory_target = 2
        factory_target = min(4, factory_target + self.budget // 3000)

        airbase_target = 0
        if self.turn >= 25 or self._air_threat() or self.budget >= 900:
            airbase_target = 1
        if self.turn >= 80 or self.budget >= 1800:
            airbase_target = 2
        airbase_target = min(4, airbase_target + self.budget // 4000)

        barracks_target = 2 if (self.turn >= 25 or self.budget >= 600) else 1
        if self.budget >= 1500:
            barracks_target = 3

        for kind, target in (
            ("Airbase", airbase_target if self._air_threat() else 0),
            ("Mine", min(mine_target, 2 + count("Mine"))),  # ≤2 new mines per turn
            ("Barracks", barracks_target),
            ("Factory", factory_target),
            ("Airbase", airbase_target),
        ):
            need = target - count(kind) - queue.count(kind)
            queue.extend([kind] * max(0, need))
        return queue

    def _air_threat(self) -> bool:
        return any(s.type in _AIR_TYPES for s in self.mem.sightings.values())

    def _constructs(self) -> None:
        if not self.complete_buildings:
            return
        # only hold an emergency Base fund in a true crisis; otherwise economy
        # IS survival — shadow died hoarding 3500 gold behind a reserve like this
        reserve = _BASE_COST if len(self.complete_bases) < 2 else 0
        built = 0
        for kind in self._build_targets():
            if built >= 4:
                break
            cost = BUILDING_STATS[kind].gold_cost
            # the first Barracks/Mine must never be starved by the base reserve —
            # the seed-69 crash saved 300 for a base for 60 turns while the 100g
            # Barracks rebuild (the actual recovery) stayed unaffordable
            floor = (
                0
                if kind == "Mine"
                or (kind == "Barracks" and self._count_buildings("Barracks") == 0)
                else reserve
            )
            if self.budget < cost + floor:
                continue
            spot = self._anchored_site(kind)
            if spot is None:
                continue
            self.actions.append(ConstructBuildingAction(building_type=kind, coord=spot))
            self.budget -= cost
            self.reserved.add(spot)
            self.own_buildings.append(
                {"type": kind, "q": spot.q, "r": spot.r, "is_complete": False, "id": f"planned-{kind}-{built}"}
            )
            built += 1

    def _anchored_site(self, kind: str) -> HexCoord | None:
        """Tile adjacent to a completed building. Prefer off-ring slots (rings are
        for replaceable bodies) and rich tiles for Mines. If every anchor tile is
        occupied by one of our own idle units, vacate it: the unit teleports out in
        phase 1 and the construct validates against the emptied tile in phase 2."""
        off_ring: list[HexCoord] = []
        on_ring: list[HexCoord] = []
        squat: list[tuple[HexCoord, dict]] = []
        squatters = {
            _c(u): u
            for u in self.own_units
            if u["id"] not in self.moved_units
            and _c(u) not in self.ring_tiles
            and u.get("movement_range", 0) >= 1
        }
        seen: set[HexCoord] = set()
        for b in self.complete_buildings:
            for n in self.grid.neighbors(_c(b)):
                if n in seen:
                    continue
                seen.add(n)
                if self._free_now(n):
                    (on_ring if n in self.ring_tiles else off_ring).append(n)
                elif n in squatters and n not in self.reserved - self.occupied:
                    squat.append((n, squatters[n]))
        enemy_coords = [s.coord for s in self._fresh_enemies(max_age=8)]

        # spread infrastructure across colonies: the seed-69 collapse lost every
        # production building in one push because they all hugged the home base
        base_coords = [_c(b) for b in self.complete_bases]
        density = {
            bc: sum(1 for b in self.own_buildings if self._dist(_c(b), bc) <= 3)
            for bc in base_coords
        }

        def sparseness(c: HexCoord) -> int:
            if not base_coords:
                return 0
            home = min(base_coords, key=lambda bc: self._dist(c, bc))
            return -density[home]

        def score(c: HexCoord) -> tuple:
            rich = self._terrain(c) == "rich_resource"
            enemy_d = min((self._dist(c, e) for e in enemy_coords), default=9)
            if kind == "Mine":
                return (rich, min(enemy_d, 9), sparseness(c))
            return (sparseness(c), min(enemy_d, 9), not rich)

        # NEVER fall back to ring tiles: a building there dies to one bomber hit
        # (×4 vs buildings) and the hole it leaves is adjacency straight to the
        # Base. Early game may use a ring tile only while we have no ring to hold.
        pool = off_ring or (on_ring if len(self.own_units) < 4 else [])
        if pool:
            return max(pool, key=score)
        if not squat:
            return None
        spot, unit = max(squat, key=lambda item: score(item[0]))
        out = self._evacuation_tile(spot, unit)
        if out is None:
            return None
        self._teleport(unit, out)
        return spot

    def _evacuation_tile(self, spot: HexCoord, unit: dict) -> HexCoord | None:
        move = unit.get("movement_range", 0)
        for r in (1, 2):
            for c in self.grid.ring(spot, r):
                if self._free_now(c) and self._entry_cost(c) <= move:
                    return c
        return None

    # ── production: bodies for the rings, teeth for the garrison ─────────────────

    def _ring_gaps(self) -> tuple[list[HexCoord], list[HexCoord]]:
        """(cost-1 gaps, difficult gaps) on base rings, threat-sorted. Pending
        bases are included: a construction site has no ring and dies to any
        passing grinder — bodies around it are what buy it the 5 build turns."""
        ranked = sorted(
            self.bases,
            key=lambda b: -self.base_threat.get(b["id"], 0.0),
        )
        easy: list[HexCoord] = []
        hard: list[HexCoord] = []
        for b in ranked:
            for c in self.grid.neighbors(_c(b)):
                if not self._free_now(c):
                    continue
                (hard if self._entry_cost(c) >= 2 else easy).append(c)
        return easy, hard

    def _unit_targets(self) -> dict[str, int]:
        nb = max(1, len(self.complete_bases))
        easy, hard = self._ring_gaps()
        on_ring = sum(1 for u in self.own_units if _c(u) in self.ring_tiles)
        infantry_pool = sum(
            1 for u in self.own_units
            if u["type"] in ("Infantry", "Tank") and _c(u) not in self.ring_tiles
        )
        # surplus gold scales every cap: the v2 agent died at T213/229 holding
        # 16k gold — banked income must become standing defense instead
        rich = self.budget
        # a standing-army floor independent of ring gaps: the seed-68 death sat
        # at 4-9 infantry all game because demand only tracked open ring tiles
        floor = min(8 + self.turn // 6, 26) - self._unit_count("Infantry")
        want = {
            "Infantry": max(
                0,
                len(easy) + 2 - infantry_pool - self._pending("Infantry"),
                floor,
            ),
            "Tank": min(8, max(len(hard), nb) + rich // 3000),
            "Fighter": min(
                14,
                max(2, nb)
                + min(5, self._known_bombers())  # interceptors per known bomber
                + (2 if self._air_threat() else 0)
                + rich // 2000,
            ),
            "Artillery": min(10, min(nb + 1, 4) + (1 if self._enemy_artillery_near() else 0) + rich // 2500),
            "Medic": min(nb, 4),
            "Scout": 2 + (1 if self.turn >= 80 else 0),
        }
        # war-prep: treaties void at the cutoff — harden before open war resumes
        if self.max_turns > TREATY_CUTOFF_TURN and self.turn >= TREATY_CUTOFF_TURN - 40:
            want["Fighter"] += 3
            want["Artillery"] += 2
            want["Tank"] += 1
            want["Infantry"] += 4
        return want

    def _pending(self, unit_type: str) -> int:
        return sum(
            1 for orders in self.mem.pending.values() for o in orders if o[0] == unit_type
        )

    def _known_bombers(self) -> int:
        return sum(
            1 for s in self._fresh_enemies(max_age=10) if s.type == "Bomber"
        )

    def _enemy_artillery_near(self) -> bool:
        base_coords = [_c(b) for b in self.bases]
        return any(
            s.type == "Artillery"
            and any(self._dist(s.coord, b) <= 6 for b in base_coords)
            for s in self._fresh_enemies(max_age=6)
        )

    def _production(self) -> None:
        want = self._unit_targets()
        producers: dict[str, list[dict]] = {"Barracks": [], "Factory": [], "Airbase": []}
        for b in self.complete_buildings:
            if b["type"] in producers:
                producers[b["type"]].append(b)
        # spawn capacity per building this turn: free neighbours minus queued orders
        capacity = {
            b["id"]: max(
                0,
                sum(1 for n in self.grid.neighbors(_c(b)) if self._free_now(n))
                - len(self.mem.pending.get(b["id"], [])),
            )
            for blist in producers.values()
            for b in blist
        }

        # priority order: a first Scout unlocks remote founding; ring bodies plus
        # artillery (kills the enemy medics) and our medics win the attrition war
        plan: list[tuple[str, str]] = []
        if self._unit_count("Scout") < 1 and self.turn >= 2:
            plan.append(("Scout", "Barracks"))
        if self._unit_count("Scout") < 2 and self.turn >= 8:
            plan.append(("Scout", "Barracks"))
        plan += [("Infantry", "Barracks")] * want.get("Infantry", 0)
        plan += [("Artillery", "Factory")] * max(0, want["Artillery"] - self._unit_count("Artillery"))
        plan += [("Medic", "Barracks")] * max(0, want["Medic"] - self._unit_count("Medic"))
        if self._air_threat():
            plan += [("Fighter", "Airbase")] * max(0, want["Fighter"] - self._unit_count("Fighter"))
        plan += [("Tank", "Factory")] * max(0, want["Tank"] - self._unit_count("Tank"))
        if not self._air_threat():
            plan += [("Fighter", "Airbase")] * max(0, want["Fighter"] - self._unit_count("Fighter"))
        plan += [("Scout", "Barracks")] * max(0, want["Scout"] - self._unit_count("Scout"))

        reserve = _BASE_COST if len(self.complete_bases) < 2 else 0
        for unit_type, producer_kind in plan:
            cost = UNIT_STATS[unit_type].gold_cost
            floor = 0 if unit_type in ("Infantry", "Medic") else reserve
            if self.budget < cost + floor:
                if unit_type == "Scout" and self._unit_count("Scout") < 1:
                    # stall one turn rather than letting 50g infantry buys starve
                    # the 100g scout forever (it never appeared before T29 once)
                    break
                continue
            building = next(
                (b for b in producers[producer_kind] if capacity.get(b["id"], 0) > 0),
                None,
            )
            if building is None:
                continue
            target = self._spawn_tile(building)
            if target is None:
                capacity[building["id"]] = 0
                continue
            self.actions.append(
                ProduceUnitAction(
                    building_id=building["id"], unit_type=unit_type, target=target
                )
            )
            self.budget -= cost
            self.reserved.add(target)
            capacity[building["id"]] -= 1
            self.mem.pending.setdefault(building["id"], []).append(
                [unit_type, UNIT_STATS[unit_type].build_turns]
            )

    def _spawn_tile(self, building: dict) -> HexCoord | None:
        bc = _c(building)
        # drop straight onto a ring gap when the producer touches one
        for n in self.grid.neighbors(bc):
            if n in self.ring_tiles and self._free_now(n):
                return n
        for n in self.grid.neighbors(bc):
            if self._free_now(n):
                return n
        return None

    # ── movement: standing rings, teleport refills, sentinels ────────────────────

    def _movements(self) -> None:
        committed = {
            u["id"] for u in self.own_units if _c(u) in self.ring_tiles
        }
        idle = [
            u for u in self.own_units
            if u["id"] not in committed
            and u["id"] not in self.moved_units
            and u.get("movement_range", 0) >= 1
        ]

        easy, hard = self._ring_gaps()
        used: set[str] = set()

        # 1) refill ring gaps — most threatened bases come first in the gap lists
        fillers = [u for u in idle if u["type"] in ("Infantry", "Tank", "Medic")]
        for gap in easy + hard:
            if gap in self.reserved:
                continue
            cost = self._entry_cost(gap)
            best = None
            for u in fillers:
                if u["id"] in used or u.get("movement_range", 0) < cost:
                    continue
                d = self._dist(_c(u), gap)
                # prefer infantry for rings; keep tanks free unless needed
                rank = (0 if u["type"] == "Infantry" else 1, d)
                if best is None or rank < best[0]:
                    best = (rank, u)
            if best is None:
                continue
            self._teleport(best[1], gap)
            used.add(best[1]["id"])

        # 2) active defense: swarm enemies that get within reach of a base.
        # Passive rings lose the attrition war — kill the grinders instead.
        self._engage(idle, used)

        # 3) fighters: one interceptor parked within 2 of each base, threat-first
        self._garrison(
            [u for u in idle if u["type"] == "Fighter" and u["id"] not in used],
            used, max_dist=2,
        )
        # 3) artillery: counter-battery platforms within 2 of each base (elevated if possible)
        self._garrison(
            [u for u in idle if u["type"] == "Artillery" and u["id"] not in used],
            used, max_dist=2, prefer_elevated=True,
        )
        # 4) leftover tanks/medics hug the weakest base
        self._garrison(
            [u for u in idle if u["type"] in ("Tank", "Medic") and u["id"] not in used],
            used, max_dist=1,
        )
        # 5) scouts explore: hop to the frontier of remembered terrain so the
        # memory (and with it the remote-founding candidate pool) keeps growing.
        # Exploration comes first — remote bases are the actual survival lever;
        # only a SECOND idle scout lingers near home as a hidden sentinel.
        explorer_done = False
        for u in idle:
            if u["type"] != "Scout" or u["id"] in used:
                continue
            if not explorer_done:
                explorer_done = True
                dest = self._frontier_tile() or self._sentinel_tile(_c(u))
            else:
                dest = self._sentinel_tile(_c(u))
            if dest is not None:
                self._teleport(u, dest)
                used.add(u["id"])

    def _frontier_tile(self) -> HexCoord | None:
        """A believed-free remembered tile with unknown neighbours, far from our
        bases — landing there reveals a fresh vision-5 disk every turn."""
        base_coords = [_c(b) for b in self.bases]
        if not base_coords:
            return None
        best = None
        best_key = None
        for c, terr in self.mem.terrain.items():
            if not self._believed_free(c) or self._entry_cost(c) > 3:
                continue
            unknown = sum(
                1 for n in self.grid.neighbors(c) if n not in self.mem.terrain
            )
            if unknown == 0:
                continue
            d = min(self._dist(c, b) for b in base_coords)
            conceal = 1 if terr == "concealment" else 0
            key = (unknown, conceal, min(d, 12))
            if best_key is None or key > best_key:
                best, best_key = c, key
        return best

    def _engage(self, idle: list[dict], used: set[str]) -> None:
        """Teleport spare combat units adjacent to enemies threatening a base so
        they attack next turn (attacks fire pre-move, so this is pre-positioning).
        """
        base_coords = [_c(b) for b in self.bases]
        if not base_coords:
            return
        hostiles = [
            e for e in self.visible_enemies
            if e.get("owner_id") not in self.peace_partners
            and e.get("type") in _THREAT_WEIGHT
            and any(self._dist(_c(e), b) <= 5 for b in base_coords)
        ]
        if not hostiles:
            return
        hostiles.sort(key=lambda e: -_THREAT_WEIGHT.get(e["type"], 0.5))
        # never commit everything: keep a couple of bodies back as ring reserve
        soldiers = [
            u for u in idle
            if u["id"] not in used
            and u["type"] in ("Infantry", "Tank", "Fighter")
        ]
        reserve_keep = 2
        budget_units = max(0, len(soldiers) - reserve_keep)
        for enemy in hostiles:
            if budget_units <= 0:
                return
            ec = _c(enemy)
            assigned = 0
            for u in sorted(soldiers, key=lambda u: self._dist(_c(u), ec)):
                if budget_units <= 0 or assigned >= 2:
                    break
                if u["id"] in used:
                    continue
                here = _c(u)
                if self._dist(here, ec) <= 1:
                    used.add(u["id"])  # already in attack position: hold
                    assigned += 1
                    budget_units -= 1
                    continue
                move = u.get("movement_range", 0)
                spot = next(
                    (
                        c for c in self.grid.neighbors(ec)
                        if self._free_now(c) and self._entry_cost(c) <= move
                    ),
                    None,
                )
                if spot is None:
                    break
                self._teleport(u, spot)
                used.add(u["id"])
                assigned += 1
                budget_units -= 1

    def _teleport(self, unit: dict, dest: HexCoord) -> None:
        here = _c(unit)
        if dest == here or self._entry_cost(dest) > unit.get("movement_range", 0):
            return
        self.actions.append(MoveAction(unit_id=unit["id"], path=[here, dest]))
        self.moved_units.add(unit["id"])
        self.reserved.discard(here)
        self.reserved.add(dest)

    def _garrison(
        self,
        units: list[dict],
        used: set[str],
        max_dist: int,
        prefer_elevated: bool = False,
    ) -> None:
        if not units or not self.complete_bases:
            return
        ranked_bases = sorted(
            self.complete_bases, key=lambda b: -self.base_threat.get(b["id"], 0.0)
        )
        # which bases already have this unit type nearby?
        assigned: list[HexCoord] = []
        per_base: dict[str, int] = {}
        kinds = {u["type"] for u in units}
        for u in self.own_units:
            if u["type"] not in kinds:
                continue
            for b in self.complete_bases:
                if self._dist(_c(u), _c(b)) <= max_dist:
                    per_base[b["id"]] = per_base.get(b["id"], 0) + 1
                    break
        for u in units:
            here = _c(u)
            target_base = next(
                (b for b in ranked_bases if per_base.get(b["id"], 0) < 1), None
            )
            if target_base is None:
                return
            bc = _c(target_base)
            if self._dist(here, bc) <= max_dist:
                per_base[target_base["id"]] = per_base.get(target_base["id"], 0) + 1
                continue
            move = u.get("movement_range", 0)
            cands = [
                c
                for r in range(1, max_dist + 1)
                for c in self.grid.ring(bc, r)
                if self._free_now(c) and self._entry_cost(c) <= move
            ]
            if not cands:
                continue
            if prefer_elevated:
                dest = max(cands, key=lambda c: (self._terrain(c) == "elevated", -self._dist(c, bc)))
            else:
                dest = min(cands, key=lambda c: self._dist(c, bc))
            self._teleport(u, dest)
            used.add(u["id"])
            per_base[target_base["id"]] = per_base.get(target_base["id"], 0) + 1
            assigned.append(dest)

    def _sentinel_tile(self, here: HexCoord) -> HexCoord | None:
        if not self.bases:
            return None
        hot = max(self.bases, key=lambda b: self.base_threat.get(b["id"], 0.0))
        bc = _c(hot)
        cands = [
            c
            for r in (4, 5)
            for c in self.grid.ring(bc, r)
            if self._free_now(c) and self._entry_cost(c) <= 3
        ]
        if not cands:
            return None
        conceal = [c for c in cands if self._terrain(c) == "concealment"]
        pool = conceal or cands
        dest = pool[(self.turn * 7) % len(pool)]
        return None if dest == here else dest
