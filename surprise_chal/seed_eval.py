"""Multi-seed evaluator for the Surprise challenge.

Runs whole 20-player games FULLY IN-PROCESS (every seat is a local agent, no HTTP)
across a list of map seeds, and reports survival/win statistics — both for the
subject in seat 0 and per-archetype across the whole field. Because the engine and
all agents are deterministic given a seed, results are reproducible.

It reuses the real competition turn loop (`game_runner.GameRunner`) and only
overrides the action-collection seam, exactly like `server/src/eval_harness.py`,
so what you measure is the real engine — not a re-implementation.

Run (from anywhere; deps are stdlib + the engine, but httpx is imported by
game_runner at module load, so use uv):

    uv run --no-project --with httpx python seed_eval.py
    uv run --no-project --with httpx python seed_eval.py --seeds 67,1,2,3,4 --turns 150
    uv run --no-project --with httpx python seed_eval.py --subject turtle --field aggressor

The bots here are SPARRING partners: simple but functional, one per strategic
archetype (see PLAN.md "Archetype roster"). They are not the Phase-1 agent.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", "src"))

from agent_base import PlayerAgent  # noqa: E402
from baseline_random import RandomAgent  # noqa: E402
from engine.actions import (  # noqa: E402
    ActionPayload,
    AttackAction,
    BreakTreatyAction,
    ConstructBuildingAction,
    MoveAction,
    ProduceUnitAction,
    ProposeTreatyAction,
    RespondTreatyAction,
)
from engine.constants import BUILDING_STATS, UNIT_STATS  # noqa: E402
from engine.hex_grid import HexCoord, HexGrid  # noqa: E402
import game_runner  # noqa: E402
from game_runner import GameConfig, GameRunner, PlayerRegistration  # noqa: E402
from schemas.observation import build_observation  # noqa: E402


# ── shared observation parsing ────────────────────────────────────────────────


@dataclass
class View:
    """A parsed observation: just the fields the sparring bots need."""

    pid: str
    turn: int
    gold: int
    grid: HexGrid
    my_units: list[dict] = field(default_factory=list)
    my_buildings: list[dict] = field(default_factory=list)
    enemy_units: list[dict] = field(default_factory=list)
    enemy_buildings: list[dict] = field(default_factory=list)
    occupied: set[tuple[int, int]] = field(default_factory=set)
    visible_free: list[HexCoord] = field(default_factory=list)
    known_players: list[str] = field(default_factory=list)
    incoming: list[dict] = field(default_factory=list)
    treaties: list[dict] = field(default_factory=list)

    @property
    def my_bases(self) -> list[dict]:
        return [b for b in self.my_buildings if b["type"] == "Base"]

    @property
    def enemy_bases(self) -> list[dict]:
        return [b for b in self.enemy_buildings if b["type"] == "Base"]

    def complete_producers(self, kinds: tuple[str, ...]) -> list[dict]:
        return [
            b for b in self.my_buildings
            if b["type"] in kinds and b.get("is_complete", True)
        ]

    def free_neighbors(self, coord: HexCoord) -> list[HexCoord]:
        return [n for n in self.grid.neighbors(coord) if (n.q, n.r) not in self.occupied]


def parse(obs: dict) -> View:
    pid = obs["player_id"]
    grid = HexGrid(obs.get("map_width", 35), obs.get("map_height", 30))
    v = View(pid=pid, turn=obs.get("turn_number", 0),
             gold=obs.get("resources", {}).get("gold", 0), grid=grid,
             known_players=list(obs.get("known_players", [])),
             incoming=list(obs.get("incoming_treaty_proposals", [])),
             treaties=list(obs.get("treaties", [])))
    for tile in obs.get("visible_tiles", []):
        c = (tile["q"], tile["r"])
        ents = tile.get("entities", [])
        if not ents:
            v.visible_free.append(HexCoord(tile["q"], tile["r"]))
        for e in ents:
            v.occupied.add((e["q"], e["r"]))
            is_building = e["type"] in BUILDING_STATS
            if e.get("owner_id") == pid:
                (v.my_buildings if is_building else v.my_units).append(e)
            else:
                (v.enemy_buildings if is_building else v.enemy_units).append(e)
    return v


def nearest(grid: HexGrid, here: HexCoord, targets: list[dict]) -> dict | None:
    if not targets:
        return None
    return min(targets, key=lambda e: grid.distance(here, HexCoord(e["q"], e["r"])))


def peace_actions(v: View, propose: bool = True) -> list:
    """Accept every incoming proposal; optionally propose peace to met players."""
    out: list = []
    have_or_pending = {t["partner_id"] for t in v.treaties}
    for prop in v.incoming:
        out.append(RespondTreatyAction(proposing_player_id=prop["proposer_id"],
                                       treaty_type="peace", accept=True))
        have_or_pending.add(prop["proposer_id"])
    if propose:
        for other in v.known_players:
            if other not in have_or_pending:
                out.append(ProposeTreatyAction(target_player_id=other, treaty_type="peace"))
    return out


def economy_actions(v: View, want_mines: int = 3) -> list:
    """Build the opening Barracks, then ramp Mines around complete buildings."""
    out: list = []
    have = {b["type"] for b in v.my_buildings}
    anchor = v.my_bases[0] if v.my_bases else (v.my_buildings[0] if v.my_buildings else None)
    if anchor is None:
        return out
    acoord = HexCoord(anchor["q"], anchor["r"])
    if "Barracks" not in have and v.gold >= BUILDING_STATS["Barracks"].gold_cost:
        for n in v.free_neighbors(acoord):
            out.append(ConstructBuildingAction(building_type="Barracks", coord=n))
            return out
    n_mines = sum(1 for b in v.my_buildings if b["type"] == "Mine")
    if n_mines < want_mines and v.gold >= BUILDING_STATS["Mine"].gold_cost:
        for b in v.my_buildings:
            if not b.get("is_complete", True):
                continue
            for n in v.free_neighbors(HexCoord(b["q"], b["r"])):
                out.append(ConstructBuildingAction(building_type="Mine", coord=n))
                return out
    return out


def found_second_base(v: View, min_dist: int = 3) -> list:
    """Found a redundancy Base on a free visible tile away from the first."""
    if len(v.my_bases) >= 2 or v.gold < BUILDING_STATS["Base"].gold_cost:
        return []
    if not v.my_bases:
        return []
    home = HexCoord(v.my_bases[0]["q"], v.my_bases[0]["r"])
    cands = [c for c in v.visible_free if v.grid.distance(home, c) >= min_dist]
    if cands:
        spot = min(cands, key=lambda c: v.grid.distance(home, c))
        return [ConstructBuildingAction(building_type="Base", coord=spot)]
    return []


def defensive_attacks(v: View) -> list:
    """Every unit fires at an enemy already in range (focus Bombers/Artillery)."""
    out: list = []
    threats = v.enemy_units + v.enemy_buildings
    pri = {"Bomber": 0, "Artillery": 1, "Fighter": 2, "Tank": 3}
    for u in v.my_units:
        rng = u.get("attack_range", 0)
        if rng < 1:
            continue
        here = HexCoord(u["q"], u["r"])
        in_range = [e for e in threats if 0 < v.grid.distance(here, HexCoord(e["q"], e["r"])) <= rng]
        if not in_range:
            continue
        tgt = min(in_range, key=lambda e: pri.get(e["type"], 9))
        out.append(AttackAction(unit_id=u["id"], target=HexCoord(tgt["q"], tgt["r"])))
    return out


def teleport_to(v: View, unit: dict, dest: HexCoord) -> MoveAction:
    """A single non-adjacent hop (engine accepts it — no waypoint-adjacency check)."""
    return MoveAction(unit_id=unit["id"], path=[HexCoord(unit["q"], unit["r"]), dest])


# ── archetype sparring bots ───────────────────────────────────────────────────


class TurtleBot(PlayerAgent):
    """Economy + denial rings + 2nd Base + universal peace. Baseline-to-beat."""

    async def decide(self, obs: dict) -> ActionPayload:
        v = parse(obs)
        acts: list = []
        acts += peace_actions(v)
        acts += economy_actions(v)
        acts += found_second_base(v)
        # ring fill: produce Infantry onto free neighbours of each base
        for b in v.complete_producers(("Barracks",)):
            if v.gold < UNIT_STATS["Infantry"].gold_cost:
                break
            base = nearest(v.grid, HexCoord(b["q"], b["r"]), v.my_bases) or b
            ring_gaps = v.free_neighbors(HexCoord(base["q"], base["r"]))
            tgt = next((n for n in v.grid.neighbors(HexCoord(b["q"], b["r"]))
                        if (n.q, n.r) not in v.occupied), None)
            if ring_gaps and tgt is not None:
                acts.append(ProduceUnitAction(building_id=b["id"], unit_type="Infantry", target=tgt))
        # pull idle infantry toward the nearest base ring (teleport)
        for u in v.my_units:
            if u["type"] != "Infantry" or u.get("movement_range", 0) < 1:
                continue
            base = nearest(v.grid, HexCoord(u["q"], u["r"]), v.my_bases)
            if base is None:
                continue
            gaps = v.free_neighbors(HexCoord(base["q"], base["r"]))
            if gaps and v.grid.distance(HexCoord(u["q"], u["r"]), gaps[0]) > 1:
                acts.append(teleport_to(v, u, gaps[0]))
        acts += defensive_attacks(v)
        return ActionPayload(player_id=v.pid, turn_number=v.turn, actions=acts)


class AggressorBot(PlayerAgent):
    """A genuine base-killer (so the evaluator can actually punish weak defense):
    rush Airbase → mass Bombers, roam a Scout to FIND hidden enemy Bases (kept in
    cross-turn memory), then teleport Bombers onto DISTINCT free ring tiles of a
    target Base and alpha-strike — 2 Bombers = 400 dmg = a dead 300-hp Base in one
    turn. A full denial ring (no free ring tile) denies the adjacency and stops it
    cold — which is exactly the defense we want to regression-test."""

    def __init__(self) -> None:
        self.known_bases: dict[tuple[int, int], int] = {}  # (q,r) -> last_seen turn

    async def decide(self, obs: dict) -> ActionPayload:
        v = parse(obs)
        acts: list = []
        # remember every enemy Base we glimpse (fog drops them; memory doesn't)
        for eb in v.enemy_bases:
            self.known_bases[(eb["q"], eb["r"])] = v.turn
        # forget very old sightings (a base may have been destroyed by someone else)
        self.known_bases = {c: t for c, t in self.known_bases.items() if v.turn - t < 60}

        have = {b["type"] for b in v.my_buildings}
        anchor = v.my_bases[0] if v.my_bases else None
        if anchor:
            acoord = HexCoord(anchor["q"], anchor["r"])
            if "Barracks" not in have and v.gold >= BUILDING_STATS["Barracks"].gold_cost:
                for n in v.free_neighbors(acoord):
                    acts.append(ConstructBuildingAction(building_type="Barracks", coord=n)); break
            elif "Airbase" not in have and v.gold >= BUILDING_STATS["Airbase"].gold_cost:
                for n in v.free_neighbors(acoord):
                    acts.append(ConstructBuildingAction(building_type="Airbase", coord=n)); break
        # a Scout to find hidden Bases
        for b in v.complete_producers(("Barracks",)):
            if not any(u["type"] == "Scout" for u in v.my_units) and v.gold >= UNIT_STATS["Scout"].gold_cost:
                tgt = next((n for n in v.grid.neighbors(HexCoord(b["q"], b["r"]))
                            if (n.q, n.r) not in v.occupied), None)
                if tgt is not None:
                    acts.append(ProduceUnitAction(building_id=b["id"], unit_type="Scout", target=tgt))
        # mass Bombers
        for ab in v.complete_producers(("Airbase",)):
            if v.gold >= UNIT_STATS["Bomber"].gold_cost:
                tgt = next((n for n in v.grid.neighbors(HexCoord(ab["q"], ab["r"]))
                            if (n.q, n.r) not in v.occupied), None)
                if tgt is not None:
                    acts.append(ProduceUnitAction(building_id=ab["id"], unit_type="Bomber", target=tgt))

        claimed: set[tuple[int, int]] = set()  # ring tiles assigned to a bomber this turn

        # roam the Scout toward the nearest known/unexplored Base to keep eyes on it
        for u in v.my_units:
            if u["type"] != "Scout":
                continue
            here = HexCoord(u["q"], u["r"])
            tgt_c = self._nearest_known(v, here)
            if tgt_c is not None and v.grid.distance(here, tgt_c) > 2:
                acts.append(teleport_to(v, u, tgt_c))
            else:  # wander outward to discover new ones
                far = max(v.visible_free, key=lambda c: v.grid.distance(here, c), default=None)
                if far is not None and (far.q, far.r) not in claimed:
                    acts.append(teleport_to(v, u, far)); claimed.add((far.q, far.r))

        # Bombers: position onto distinct free ring tiles of a target, then bomb.
        for u in v.my_units:
            if u["type"] != "Bomber":
                continue
            here = HexCoord(u["q"], u["r"])
            tc = self._nearest_known(v, here)
            if tc is None:
                continue
            if v.grid.distance(here, tc) <= u.get("attack_range", 1):
                acts.append(AttackAction(unit_id=u["id"], target=tc))  # adjacent → strike
                continue
            # not adjacent: claim a distinct free ring tile and teleport onto it
            gaps = [n for n in v.grid.neighbors(tc)
                    if (n.q, n.r) not in v.occupied and (n.q, n.r) not in claimed
                    and v.grid.distance(here, n) <= u.get("movement_range", 2)]
            if gaps:
                g = min(gaps, key=lambda n: v.grid.distance(here, n))
                acts.append(teleport_to(v, u, g)); claimed.add((g.q, g.r))
        acts += peace_actions(v, propose=False)  # accept peace, never offer
        return ActionPayload(player_id=v.pid, turn_number=v.turn, actions=acts)

    def _nearest_known(self, v: View, here: HexCoord) -> HexCoord | None:
        cands = [HexCoord(*c) for c in self.known_bases]
        if not cands:
            return None
        return min(cands, key=lambda c: v.grid.distance(here, c))


class TreatyAmbusherBot(PlayerAgent):
    """Propose peace, lull for `fuse` turns, then break + strike the same turn."""

    def __init__(self, fuse: int = 8) -> None:
        self.fuse = fuse
        self.peace_since: dict[str, int] = {}

    async def decide(self, obs: dict) -> ActionPayload:
        v = parse(obs)
        acts: list = []
        acts += peace_actions(v)
        active = {t["partner_id"] for t in v.treaties}
        for p in active:
            self.peace_since.setdefault(p, v.turn)
        # a partner whose peace is old enough and who is visible → betray
        visible_owners = {e["owner_id"] for e in v.enemy_units + v.enemy_buildings}
        for p in list(active):
            if v.turn - self.peace_since.get(p, v.turn) >= self.fuse and p in visible_owners:
                acts.append(BreakTreatyAction(partner_player_id=p, treaty_type="peace"))
        # strike anything in range this turn (protection drops immediately on break)
        acts += defensive_attacks(v)
        # produce some infantry to have a striking force
        for b in v.complete_producers(("Barracks",)):
            if v.gold >= UNIT_STATS["Infantry"].gold_cost:
                tgt = next((n for n in v.grid.neighbors(HexCoord(b["q"], b["r"]))
                            if (n.q, n.r) not in v.occupied), None)
                if tgt is not None:
                    acts.append(ProduceUnitAction(building_id=b["id"], unit_type="Infantry", target=tgt))
        acts += economy_actions(v, want_mines=1)
        return ActionPayload(player_id=v.pid, turn_number=v.turn, actions=acts)


class SplashRaiderBot(PlayerAgent):
    """Make peace, then artillery-splash a peace partner's Base from an empty tile."""

    async def decide(self, obs: dict) -> ActionPayload:
        v = parse(obs)
        acts: list = []
        acts += peace_actions(v)
        have = {b["type"] for b in v.my_buildings}
        anchor = v.my_bases[0] if v.my_bases else None
        if anchor:
            acoord = HexCoord(anchor["q"], anchor["r"])
            if "Factory" not in have and v.gold >= BUILDING_STATS["Factory"].gold_cost:
                for n in v.free_neighbors(acoord):
                    acts.append(ConstructBuildingAction(building_type="Factory", coord=n)); break
        for f in v.complete_producers(("Factory",)):
            if v.gold >= UNIT_STATS["Artillery"].gold_cost:
                tgt = next((n for n in v.grid.neighbors(HexCoord(f["q"], f["r"]))
                            if (n.q, n.r) not in v.occupied), None)
                if tgt is not None:
                    acts.append(ProduceUnitAction(building_id=f["id"], unit_type="Artillery", target=tgt))
        # splash any visible enemy Base from an EMPTY adjacent tile (peace-proof)
        for u in v.my_units:
            if u["type"] != "Artillery":
                continue
            here = HexCoord(u["q"], u["r"])
            eb = nearest(v.grid, here, v.enemy_bases)
            if eb is None:
                continue
            ec = HexCoord(eb["q"], eb["r"])
            empty_adj = [n for n in v.grid.neighbors(ec)
                         if (n.q, n.r) not in v.occupied
                         and 0 < v.grid.distance(here, n) <= u.get("attack_range", 3)]
            if empty_adj:
                acts.append(AttackAction(unit_id=u["id"], target=empty_adj[0]))
            elif v.grid.distance(here, ec) > u.get("attack_range", 3):
                step = min(v.grid.neighbors(here), key=lambda n: v.grid.distance(n, ec))
                if (step.q, step.r) not in v.occupied:
                    acts.append(MoveAction(unit_id=u["id"], path=[here, step]))
        acts += economy_actions(v, want_mines=2)
        return ActionPayload(player_id=v.pid, turn_number=v.turn, actions=acts)


class EconomistBot(PlayerAgent):
    """Pure economy + Base-spam, minimal military. Passive survival baseline."""

    async def decide(self, obs: dict) -> ActionPayload:
        v = parse(obs)
        acts: list = []
        acts += peace_actions(v)
        acts += economy_actions(v, want_mines=5)
        acts += found_second_base(v, min_dist=2)
        # one cheap unit for vision if a barracks exists
        for b in v.complete_producers(("Barracks",)):
            if v.gold >= UNIT_STATS["Infantry"].gold_cost and len(v.my_units) < 3:
                tgt = next((n for n in v.grid.neighbors(HexCoord(b["q"], b["r"]))
                            if (n.q, n.r) not in v.occupied), None)
                if tgt is not None:
                    acts.append(ProduceUnitAction(building_id=b["id"], unit_type="Infantry", target=tgt))
        acts += defensive_attacks(v)
        return ActionPayload(player_id=v.pid, turn_number=v.turn, actions=acts)


def _make_real_agent() -> PlayerAgent:
    """Load the actual Phase-1 submission agent (participant/src/algo_agent.py).

    Imported lazily so the sparring bots don't depend on the participant package.
    participant/src is appended AFTER server/src on the path, so the shared engine
    (byte-identical mirror) resolves to the server copy game_runner already uses;
    only world/planner/actuator/algo_agent are unique to the participant tree.
    """
    psrc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "participant", "src")
    if psrc not in sys.path:
        sys.path.append(psrc)
    from algo_agent import AlgoAgent  # noqa: E402

    return AlgoAgent()


ARCHETYPES: dict[str, callable] = {
    "agent": _make_real_agent,  # the real Phase-1 submission (turtle)
    "turtle": TurtleBot,
    "aggressor": AggressorBot,
    "ambusher": TreatyAmbusherBot,
    "splash": SplashRaiderBot,
    "economist": EconomistBot,
    "random": RandomAgent,
}


# ── in-process multi-seed runner ──────────────────────────────────────────────


class EvalRunner(GameRunner):
    """All seats in-process; records each player's elimination turn."""

    def __init__(self, regs, config, actors: dict) -> None:
        super().__init__(regs, config)
        self.actors = actors
        self.elim_turn: dict[str, int] = {}

    async def _collect_actions(self, player_urls):  # type: ignore[override]
        state = self.state
        for pid in player_urls:
            if not state.players[pid].alive and pid not in self.elim_turn:
                self.elim_turn[pid] = state.turn_number
        alive = [pid for pid in player_urls if state.players[pid].alive]

        async def one(pid: str):
            obs = build_observation(state, pid, self.diplomacy, self.chat_log,
                                    self.config.max_turns)
            try:
                payload = await self.actors[pid].decide(obs)
            except Exception:  # a buggy bot must not crash the tournament
                payload = None
            if payload is None:
                payload = ActionPayload(player_id=pid, turn_number=state.turn_number, actions=[])
            return pid, payload

        return dict(await asyncio.gather(*[one(pid) for pid in alive]))


@dataclass
class GameResult:
    seed: int
    survived: dict[str, bool]
    elim_turn: dict[str, int]
    turns: int
    n_survivors: int


def run_one_game(seed: int, roster: list[str], cfg: dict) -> GameResult:
    random.seed(seed)  # make any random-using bot reproducible per seed
    n = len(roster)
    ids = [f"player-{i}" for i in range(n)]
    regs = [PlayerRegistration(pid, f"{roster[i]}:{pid}", "local://x") for i, pid in enumerate(ids)]
    actors = {pid: ARCHETYPES[roster[i]]() for i, pid in enumerate(ids)}
    config = GameConfig(seed=seed, map_width=cfg["w"], map_height=cfg["h"],
                        max_turns=cfg["turns"])
    runner = EvalRunner(regs, config, actors)
    runner.initialise()
    asyncio.run(runner.run())
    survived = {pid: bool(runner.state.players[pid].alive) for pid in ids}
    return GameResult(seed=seed, survived=survived, elim_turn=dict(runner.elim_turn),
                      turns=runner.state.turn_number,
                      n_survivors=sum(survived.values()))


def build_roster(subject: str, field_spec: str, n: int) -> list[str]:
    """Seat 0 = subject; seats 1..n-1 cycle through the field spec."""
    field_names = [s.strip() for s in field_spec.split(",") if s.strip()]
    roster = [subject]
    i = 0
    while len(roster) < n:
        roster.append(field_names[i % len(field_names)])
        i += 1
    return roster[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="67,1,2,3,4,5,6,7",
                    help="comma-separated map seeds")
    ap.add_argument("--players", type=int, default=20)
    ap.add_argument("--turns", type=int, default=150,
                    help="max turns/game (real eval undisclosed; Discord=50, local=300)")
    ap.add_argument("--width", type=int, default=35)
    ap.add_argument("--height", type=int, default=30)
    ap.add_argument("--subject", default="turtle", choices=list(ARCHETYPES),
                    help="archetype in seat player-0 (the one the summary highlights)")
    ap.add_argument("--field", default="turtle,aggressor,ambusher,splash,economist,random",
                    help="comma-separated archetypes cycled through seats 1..n-1")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    roster = build_roster(args.subject, args.field, args.players)
    cfg = {"w": args.width, "h": args.height, "turns": args.turns}

    # silence game logging + replay file writes for batch runs
    import logging
    logging.disable(logging.CRITICAL)
    game_runner.ReplayRecorder = _NullRecorder

    print(f"seeds={seeds}  players={args.players}  turns={args.turns}  map={args.width}x{args.height}")
    print(f"subject(seat0)={args.subject}   field={args.field}\n")
    print(f"  {'seed':>5} | {'subject':>8} | {'elim@':>6} | {'survivors':>9} | {'len':>4}")
    print("  " + "-" * 46)

    results: list[GameResult] = []
    for seed in seeds:
        r = run_one_game(seed, roster, cfg)
        results.append(r)
        s0 = "player-0"
        alive0 = r.survived[s0]
        elim = "-" if alive0 else str(r.elim_turn.get(s0, r.turns))
        tag = "SURVIVE" if alive0 else "  dead "
        print(f"  {seed:>5} | {tag:>8} | {elim:>6} | {r.n_survivors:>9} | {r.turns:>4}")

    # ── aggregate: subject (seat 0) ──
    ng = len(results)
    subj_alive = sum(r.survived["player-0"] for r in results)
    sole = sum(1 for r in results if r.survived["player-0"] and r.n_survivors == 1)
    deaths = [r.elim_turn.get("player-0", r.turns) for r in results if not r.survived["player-0"]]
    mean_surv = sum(r.n_survivors for r in results) / ng
    mean_len = sum(r.turns for r in results) / ng
    print("\n  subject (player-0 = %s):" % args.subject)
    print(f"    survival rate : {subj_alive}/{ng}  ({100*subj_alive/ng:.0f}%)")
    print(f"    sole-win rate : {sole}/{ng}  ({100*sole/ng:.0f}%)")
    if deaths:
        print(f"    mean elim turn (when it died): {sum(deaths)/len(deaths):.1f}")
    print(f"    mean #survivors/game : {mean_surv:.1f}   mean game length : {mean_len:.0f}")

    # ── aggregate: per-archetype survival across ALL seats & seeds ──
    by_arch: dict[str, list[int]] = {}
    for r in results:
        for i, name in enumerate(roster):
            by_arch.setdefault(name, []).append(int(r.survived[f"player-{i}"]))
    print("\n  per-archetype survival (all seats, all seeds):")
    for name in sorted(by_arch, key=lambda k: -sum(by_arch[k]) / len(by_arch[k])):
        vals = by_arch[name]
        seats = sum(1 for x in roster if x == name)
        print(f"    {name:>10}: {100*sum(vals)/len(vals):5.1f}%  "
              f"({sum(vals)}/{len(vals)} seat-games, {seats} seat(s)/game)")


class _NullRecorder:
    """Drop-in for ReplayRecorder that writes nothing (batch eval)."""

    def __init__(self, *a, **k) -> None: ...
    def record_initial(self, *a, **k) -> None: ...
    def record_turn(self, *a, **k) -> None: ...
    def close(self, *a, **k) -> None: ...


if __name__ == "__main__":
    main()
