"""Actuator: turn a Plan + WorldModel into a validated ActionPayload.

ALL hex/range math lives here and is deterministic — the LLM never touches it.
Everything is bounded per-turn (no whole-map pathfinding) to stay well under the
~10s deadline and the 1 CPU / 1 GiB cap.

Survival levers, in the priority the plan sets out:
  1. base redundancy   — found a 2nd/3rd Base early
  2. universal peace    — accept all, propose selectively
  3. per-Base denial rings — keep all 6 neighbours of every complete Base occupied
     by our own units (air units are blocked from occupied tiles, so a full ring
     physically denies a Bomber the adjacency it needs)
  4. economy           — Mines (rich tiles first) to fund the above
  5. stay hidden        — don't over-extend

Two engine facts shape the mechanics here (verified):
  * Teleport quirk: a move is validated only by path[0]==current, hop-count ≤
    movement_range, and Σ(entry-cost of listed tiles) ≤ movement_range. A single
    hop `[here, dest]` to any non-difficult free tile is always legal for a move-1
    unit → one reserve can refill any ring gap. Used for **pre-positioning only**.
  * Attacks resolve BEFORE movement and fire from the pre-move tile, so a reactive
    teleport can't intercept this turn — the ring must be *standing*, not scrambled.
"""

from __future__ import annotations

from engine.actions import (
    AttackAction,
    BreakTreatyAction,
    ConstructBuildingAction,
    MoveAction,
    ProduceUnitAction,
    ProposeTreatyAction,
    RespondTreatyAction,
)
from engine.constants import BUILDING_STATS, UNIT_STATS
from engine.hex_grid import HexCoord

from planner import Plan
from world import WorldModel

# attack target priority: kill the thing most able to hurt our Bases first.
_THREAT_PRIORITY = {
    "Bomber": 0,
    "Artillery": 1,
    "Fighter": 2,
    "Tank": 3,
    "Scout": 4,
    "Infantry": 5,
    "Medic": 6,
}
_RING_FILL_UNIT = "Infantry"  # cheapest body, eats only 50/hit on a ring tile


class Actuator:
    def __init__(self, world: WorldModel, plan: Plan) -> None:
        self.w = world
        self.p = plan
        self.grid = world.grid
        self.budget = world.gold
        # tiles spoken-for this turn: every occupied tile, plus anything we build /
        # produce-onto / move-onto, so our own actions never collide with each other.
        self.reserved: set[HexCoord] = set(world.occupied)
        self.busy_units: set[str] = set()  # units that already have an attack this turn
        self.actions: list = []

    # ── public entry point ──────────────────────────────────────────────────────

    def act(self) -> list:
        self._diplomacy()
        # attacks first: they fire from the pre-move tile and mark a unit busy so it
        # isn't also yanked to refill a ring (it stays put, holding its tile).
        self._attacks()
        # constructs are funded by the engine BEFORE produces, so decide them first
        # against the same running budget to stay in sync.
        self._constructs()
        self._produces()
        # movement is free; do it last, using only idle (non-attacking) units.
        self._movements()
        return self.actions

    # ── helpers ─────────────────────────────────────────────────────────────────

    def _free(self, c: HexCoord) -> bool:
        return c in self.w.visible and c not in self.reserved

    def _ring(self, base: dict) -> list[HexCoord]:
        return self.grid.neighbors(HexCoord(base["q"], base["r"]))

    def _free_neighbour(self, coord: HexCoord) -> HexCoord | None:
        for n in self.grid.neighbors(coord):
            if self._free(n):
                return n
        return None

    def _dist(self, a: HexCoord, b: HexCoord) -> int:
        return self.grid.distance(a, b)

    def _all_ring_coords(self) -> set[HexCoord]:
        out: set[HexCoord] = set()
        for b in self.w.my_complete_bases:
            out.update(self._ring(b))
        return out

    # ── diplomacy ────────────────────────────────────────────────────────────────

    def _diplomacy(self) -> None:
        # accept EVERY incoming proposal — it's free (they already know us) and an
        # active peace blocks all direct attacks on our Bases.
        pending = {t.get("partner_id") for t in self.w.treaties}
        for prop in self.w.incoming:
            proposer = prop.get("proposer_id")
            if proposer is None:
                continue
            self.actions.append(
                RespondTreatyAction(proposing_player_id=proposer, treaty_type="peace", accept=True)
            )
            pending.add(proposer)

        if not self.p.propose_peace:
            return
        # propose selectively to players we've MET but aren't yet bound to. A
        # delivered proposal leaks our id (not position) to them, but universal
        # peace is our #2 survival lever — worth it. Skip anyone we're hunting.
        for other in self.w.known_players:
            if other in pending or other in self.p.hunter_targets:
                continue
            self.actions.append(ProposeTreatyAction(target_player_id=other, treaty_type="peace"))
            pending.add(other)

        # If a partner is BREAKING peace with us, that's war THIS turn (quirk). We
        # don't pre-emptively break treaties (turtle), but we drop them as hunt
        # candidates handled by the planner.

    # ── attacks ──────────────────────────────────────────────────────────────────

    def _attacks(self) -> None:
        # candidate targets: visible enemies we are allowed to hit (not peace
        # partners; breaking partners ARE fair game — protection is already gone).
        targets = [
            e
            for e in self.w.visible_enemies
            if e.get("owner_id") not in self.w.peace_partners
        ]
        if not targets:
            return
        for u in self.w.own_units:
            rng = u.get("attack_range", 0)
            if rng < 1:
                continue
            here = HexCoord(u["q"], u["r"])
            in_range = [t for t in targets if 0 < self._dist(here, HexCoord(t["q"], t["r"])) <= rng]
            if not in_range:
                continue
            tgt = min(in_range, key=lambda t: _THREAT_PRIORITY.get(t["type"], 8))
            tc = HexCoord(tgt["q"], tgt["r"])
            # artillery splash ignores ownership — don't splash our own bodies.
            if u["type"] == "Artillery" and self._friendly_in_splash(tc):
                continue
            self.actions.append(AttackAction(unit_id=u["id"], target=tc))
            self.busy_units.add(u["id"])

    def _friendly_in_splash(self, center: HexCoord) -> bool:
        own = {(HexCoord(e["q"], e["r"])) for e in self.w.own_units + self.w.own_buildings}
        for c in [center, *self.grid.ring(center, 1)]:
            if c in own:
                return True
        return False

    # ── constructs ───────────────────────────────────────────────────────────────

    def _afford(self, cost: int) -> bool:
        return self.budget >= cost

    def _emit_construct(self, btype: str, coord: HexCoord) -> None:
        self.actions.append(ConstructBuildingAction(building_type=btype, coord=coord))
        self.budget -= BUILDING_STATS[btype].gold_cost
        self.reserved.add(coord)

    def _anchor_tiles(self) -> list[HexCoord]:
        """Free tiles adjacent to a COMPLETED own building — where non-Base
        buildings may be placed. Prefer slots NOT on a base ring (keep rings for
        cheap, replaceable Infantry rather than 200/hit buildings)."""
        ring = self._all_ring_coords()
        off_ring: list[HexCoord] = []
        on_ring: list[HexCoord] = []
        for b in self.w.complete_buildings:
            bc = HexCoord(b["q"], b["r"])
            for n in self.grid.neighbors(bc):
                if not self._free(n):
                    continue
                (on_ring if n in ring else off_ring).append(n)
        return off_ring + on_ring

    def _constructs(self) -> None:
        w, p = self.w, self.p

        # 1) Barracks — needed for Infantry to seed the rings. Highest priority.
        if w.count_buildings("Barracks") == 0 and self._afford(BUILDING_STATS["Barracks"].gold_cost):
            spot = self._first_anchor()
            if spot is not None:
                self._emit_construct("Barracks", spot)

        # 2) Bootstrap economy with a FIRST Mine before anything else — without an
        #    income building the whole opening stalls. (T0 = Barracks + Mine.)
        if w.count_buildings("Mine") == 0 and self._afford(BUILDING_STATS["Mine"].gold_cost):
            spot = self._mine_site()
            if spot is not None:
                self._emit_construct("Mine", spot)

        # 3) Base redundancy — the top survival lever, founded EARLY but only once a
        #    first income building exists (don't blow the whole opening on two Bases
        #    with zero economy). With Mine income this lands ~T8–15.
        if (
            w.count_buildings("Mine") >= 1
            and len(w.my_bases) < p.base_target
            and self._afford(BUILDING_STATS["Base"].gold_cost)
        ):
            site = self._base_site()
            if site is not None:
                self._emit_construct("Base", site)

        # 4) Scale economy — more Mines (rich tiles first) to fund the army.
        if w.count_buildings("Mine") < p.mine_target and self._afford(BUILDING_STATS["Mine"].gold_cost):
            spot = self._mine_site()
            if spot is not None:
                self._emit_construct("Mine", spot)

        # 4) Military building — Airbase (air threat) or Factory (default ground).
        want_air = p.want_airbase and w.count_buildings("Airbase") == 0
        want_fac = (not p.want_airbase) and p.want_factory and w.count_buildings("Factory") == 0
        # only once a basic economy + a redundancy Base exist (don't starve survival)
        economy_ready = w.count_buildings("Mine") >= 1 and len(w.my_bases) >= min(2, p.base_target)
        if economy_ready:
            if want_air and self._afford(BUILDING_STATS["Airbase"].gold_cost):
                spot = self._first_anchor()
                if spot is not None:
                    self._emit_construct("Airbase", spot)
            elif want_fac and self._afford(BUILDING_STATS["Factory"].gold_cost):
                spot = self._first_anchor()
                if spot is not None:
                    self._emit_construct("Factory", spot)

    def _first_anchor(self) -> HexCoord | None:
        tiles = self._anchor_tiles()
        return tiles[0] if tiles else None

    def _mine_site(self) -> HexCoord | None:
        # prefer an anchor tile that is RICH (flat 50/turn vs 20).
        rich = [c for c in self._anchor_tiles() if self.w.terrain_at(c) == "rich_resource"]
        if rich:
            return rich[0]
        tiles = self._anchor_tiles()
        return tiles[0] if tiles else None

    def _base_site(self) -> HexCoord | None:
        """Pick a visible free tile to found a redundancy Base — spread as far from
        existing Bases as our current vision allows. A Base can be founded on any
        tile we can SEE (no anchor needed)."""
        bases = [HexCoord(b["q"], b["r"]) for b in self.w.my_bases]
        if not bases:
            return None
        ring = self._all_ring_coords()
        # don't plant a 2nd Base on an existing base's own ring (defeats redundancy)
        cands = [
            c
            for c in self.w.visible_free
            if c not in self.reserved and c not in ring
        ]
        if not cands:
            return None
        # separation requirement, relaxed late so we never go without redundancy
        min_sep = 3
        if self.w.turn > self.w.max_turns // 3 or len(self.w.my_bases) == 1 and self.w.turn > 25:
            min_sep = 2
        spread = [c for c in cands if min(self._dist(c, b) for b in bases) >= min_sep]
        pool = spread or cands
        # maximise the minimum distance to any existing Base (best spread)
        return max(pool, key=lambda c: min(self._dist(c, b) for b in bases))

    # ── production ───────────────────────────────────────────────────────────────

    def _emit_produce(self, building: dict, unit_type: str, target: HexCoord) -> None:
        self.actions.append(
            ProduceUnitAction(building_id=building["id"], unit_type=unit_type, target=target)
        )
        self.budget -= UNIT_STATS[unit_type].gold_cost
        self.reserved.add(target)
        self.w.record_order(building["id"], unit_type, UNIT_STATS[unit_type].build_turns, target)

    def _produces(self) -> None:
        w, p = self.w, self.p

        # ── ring fill: Infantry from Barracks, one per Barracks per turn, capped at
        #    the real gap count (current gaps minus units already in flight). ──
        gaps = self._ring_gap_count()
        in_flight = w.pending_count(_RING_FILL_UNIT)
        need = max(0, gaps - in_flight)
        if need > 0:
            for bar in w.producers(("Barracks",)):
                if need <= 0:
                    break
                if not self._afford(UNIT_STATS[_RING_FILL_UNIT].gold_cost):
                    break
                tgt = self._produce_target(bar)
                if tgt is None:
                    continue
                self._emit_produce(bar, _RING_FILL_UNIT, tgt)
                need -= 1

        # ── garrison: punishment muscle to one-shot an intruding Bomber ──
        self._produce_garrison("Factory", "Tank", p.garrison_tanks)
        self._produce_garrison("Airbase", "Fighter", p.garrison_fighters)

    def _produce_target(self, building: dict) -> HexCoord | None:
        """A free tile ≤1 from the building. Prefer a base-ring gap the building
        already touches so the Infantry lands directly on the ring."""
        bc = HexCoord(building["q"], building["r"])
        ring = self._all_ring_coords()
        ring_adj = [n for n in self.grid.neighbors(bc) if n in ring and self._free(n)]
        if ring_adj:
            return ring_adj[0]
        return self._free_neighbour(bc)

    def _produce_garrison(self, producer_type: str, unit_type: str, target_count: int) -> None:
        if target_count <= 0:
            return
        have = sum(1 for u in self.w.own_units if u["type"] == unit_type)
        have += self.w.pending_count(unit_type)
        for prod in self.w.producers((producer_type,)):
            if have >= target_count:
                break
            if not self._afford(UNIT_STATS[unit_type].gold_cost):
                break
            tgt = self._free_neighbour(HexCoord(prod["q"], prod["r"]))
            if tgt is None:
                continue
            self._emit_produce(prod, unit_type, tgt)
            have += 1

    def _ring_gap_count(self) -> int:
        return sum(1 for b in self.w.my_complete_bases for c in self._ring(b) if self._free(c))

    # ── movement (teleport-based pre-positioning; never interception) ────────────

    def _movements(self) -> None:
        ring_coords = self._all_ring_coords()
        # units committed to a standing ring stay put — never strip a ring.
        committed = {
            u["id"] for u in self.w.own_units if HexCoord(u["q"], u["r"]) in ring_coords
        }
        idle = [
            u
            for u in self.w.own_units
            if u["id"] not in committed
            and u["id"] not in self.busy_units
            and u.get("movement_range", 0) >= 1
        ]

        # 1) refill ring gaps with the nearest idle body (Infantry/Tank, not Scout).
        gaps = [c for b in self.w.my_complete_bases for c in self._ring(b) if self._free(c)]
        fillers = [u for u in idle if u["type"] in ("Infantry", "Tank")]
        used: set[str] = set()
        for gap in gaps:
            cost = self.w.entry_cost(gap)
            best = None
            for u in fillers:
                if u["id"] in used:
                    continue
                if u.get("movement_range", 0) < cost:
                    continue  # can't pay entry (e.g. Infantry → difficult tile)
                here = HexCoord(u["q"], u["r"])
                d = self._dist(here, gap)
                if best is None or d < best[0]:
                    best = (d, u)
            if best is None:
                continue
            u = best[1]
            here = HexCoord(u["q"], u["r"])
            self.actions.append(MoveAction(unit_id=u["id"], path=[here, gap]))
            self.reserved.add(gap)
            used.add(u["id"])

        # 2) roam a Scout outward to grow the WorldModel (find hidden Base sites).
        for u in idle:
            if u["type"] != "Scout" or u["id"] in used:
                continue
            dest = self._scout_frontier(HexCoord(u["q"], u["r"]))
            if dest is not None:
                self.actions.append(MoveAction(unit_id=u["id"], path=[HexCoord(u["q"], u["r"]), dest]))
                self.reserved.add(dest)
                used.add(u["id"])

    def _scout_frontier(self, here: HexCoord) -> HexCoord | None:
        """A free visible tile far from our Bases (and within the scout's hop), to
        push the vision boundary outward over successive turns."""
        bases = [HexCoord(b["q"], b["r"]) for b in self.w.my_bases]
        if not bases:
            return None
        cands = [
            c
            for c in self.w.visible_free
            if c not in self.reserved and self.w.entry_cost(c) <= 3  # Scout move = 3
        ]
        if not cands:
            return None
        far = max(cands, key=lambda c: min(self._dist(c, b) for b in bases))
        # only bother if it actually extends our reach
        if min(self._dist(far, b) for b in bases) <= 1:
            return None
        return far
