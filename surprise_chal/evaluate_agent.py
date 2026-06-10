#!/usr/bin/env python3
"""Multi-seed evaluator for Surprise agents.

This is not the official scorer. It is a local strategy evaluator built around
the rules in RULES.md: survival is the primary objective, with elimination of
opponents as a secondary signal. It runs the real engine in-process so you can
iterate quickly without Docker or Vertex.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import types
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PARTICIPANT_SRC = ROOT / "participant" / "src"
SERVER_SRC = ROOT / "server" / "src"

# Prefer participant first so the submitted agent imports the same local mirror
# it will use in-container. The engine mirrors are intended to be identical.
sys.path.insert(0, str(PARTICIPANT_SRC))
sys.path.insert(1, str(SERVER_SRC))

# game_runner imports httpx for the official HTTP path. This evaluator overrides
# action collection and never uses httpx, so do not require it locally.
sys.modules.setdefault("httpx", types.SimpleNamespace(AsyncClient=object))

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
from engine.constants import (  # noqa: E402
    BUILDING_STATS,
    DEFAULT_MAP_HEIGHT,
    DEFAULT_MAP_WIDTH,
    MAX_TURNS,
    UNIT_STATS,
)
from engine.hex_grid import HexCoord, HexGrid  # noqa: E402
from engine.resources import ResourceType  # noqa: E402
from game_runner import GameConfig, GameRunner, PlayerRegistration  # noqa: E402
from schemas.observation import build_observation  # noqa: E402

PLAYER_ID = "player-0"
PRODUCTION_BUILDINGS = ("Barracks", "Factory", "Airbase")
HARD_PERSONALITIES = (
    "bomber_rush",
    "artillery_siege",
    "base_hunter",
    "fast_expand",
    "treaty_ambush",
)


def coord(entity: dict[str, Any]) -> HexCoord:
    return HexCoord(entity["q"], entity["r"])


def is_building(entity: dict[str, Any]) -> bool:
    return entity.get("type") in BUILDING_STATS


@dataclass
class FlatObservation:
    own_units: list[dict[str, Any]]
    own_buildings: list[dict[str, Any]]
    enemies: list[dict[str, Any]]
    occupied: set[tuple[int, int]]
    terrain: dict[tuple[int, int], str]
    visible: list[tuple[int, int]]


def flatten_observation(observation: dict[str, Any], player_id: str) -> FlatObservation:
    own_units, own_buildings, enemies = [], [], []
    occupied: set[tuple[int, int]] = set()
    terrain: dict[tuple[int, int], str] = {}
    visible: list[tuple[int, int]] = []
    for tile in observation.get("visible_tiles", []):
        pos = (tile["q"], tile["r"])
        visible.append(pos)
        terrain[pos] = tile.get("terrain", "normal")
        for entity in tile.get("entities", []):
            occupied.add((entity["q"], entity["r"]))
            if entity.get("owner_id") == player_id:
                (own_buildings if is_building(entity) else own_units).append(entity)
            else:
                enemies.append(entity)
    return FlatObservation(own_units, own_buildings, enemies, occupied, terrain, visible)


class HardOpponent:
    """A legal but sharper local sparring bot.

    The personalities deliberately pressure the survival-first strategy:
    air raids, artillery sieges, direct Base hunting, greedy expansion, and
    treaty ambushes. They only use their observations plus local memory.
    """

    def __init__(self, personality: str, slot: int) -> None:
        self.personality = personality
        self.slot = slot
        self.known_enemy_bases: dict[str, HexCoord] = {}
        self.last_proposed: dict[str, int] = {}

    async def decide(self, observation: dict[str, Any]) -> ActionPayload:
        pid = observation["player_id"]
        turn = observation.get("turn_number", 0)
        gold = observation.get("resources", {}).get("gold", 0)
        grid = HexGrid(
            observation.get("map_width", DEFAULT_MAP_WIDTH),
            observation.get("map_height", DEFAULT_MAP_HEIGHT),
        )
        flat = flatten_observation(observation, pid)
        self._remember_enemy_bases(flat.enemies)

        actions: list[Any] = []
        actions.extend(self._diplomacy(observation, flat, turn))
        complete = [b for b in flat.own_buildings if b.get("is_complete", True)]
        gold = self._build(grid, turn, gold, flat, complete, actions)
        gold = self._produce(grid, turn, gold, flat, complete, actions)
        self._command_units(grid, turn, flat, complete, actions)
        return ActionPayload(player_id=pid, turn_number=turn, actions=actions)

    def _remember_enemy_bases(self, enemies: list[dict[str, Any]]) -> None:
        for enemy in enemies:
            if enemy["type"] == "Base":
                self.known_enemy_bases[enemy["owner_id"]] = coord(enemy)

    def _diplomacy(
        self, observation: dict[str, Any], flat: FlatObservation, turn: int
    ) -> list[Any]:
        actions: list[Any] = []
        treaties = observation.get("treaties", [])
        ambush = self.personality == "treaty_ambush"

        for proposal in observation.get("incoming_treaty_proposals", []):
            proposer = proposal.get("proposer_id")
            if not proposer:
                continue
            actions.append(
                RespondTreatyAction(
                    proposing_player_id=proposer,
                    treaty_type=proposal.get("treaty_type", "peace"),
                    accept=ambush,
                )
            )

        if not ambush:
            return actions

        known = observation.get("known_players", [])
        active_or_breaking = {t.get("partner_id") for t in treaties}
        for player_id in known:
            if player_id in active_or_breaking:
                continue
            if turn - self.last_proposed.get(player_id, -1000) < 20:
                continue
            actions.append(ProposeTreatyAction(target_player_id=player_id))
            self.last_proposed[player_id] = turn

        army_ready = len(flat.own_units) >= 10 or turn >= 70
        for treaty in treaties:
            partner = treaty.get("partner_id")
            if partner and treaty.get("breaking_in_turns") is None and army_ready:
                actions.append(BreakTreatyAction(partner_player_id=partner))
        return actions

    def _build(
        self,
        grid: HexGrid,
        turn: int,
        gold: int,
        flat: FlatObservation,
        complete: list[dict[str, Any]],
        actions: list[Any],
    ) -> int:
        if not complete:
            return gold

        counts = self._building_counts(flat.own_buildings)
        planned = counts.copy()
        for want in self._build_order(turn, gold, counts):
            if gold < BUILDING_STATS[want].gold_cost + self._build_reserve(want):
                continue
            if planned.get(want, 0) >= self._building_target(want, turn, gold):
                continue
            spot = (
                self._base_site(grid, flat, complete)
                if want == "Base"
                else self._anchored_site(grid, want, flat, complete)
            )
            if spot is None:
                continue
            actions.append(ConstructBuildingAction(building_type=want, coord=spot))
            flat.occupied.add((spot.q, spot.r))
            planned[want] = planned.get(want, 0) + 1
            gold -= BUILDING_STATS[want].gold_cost
        return gold

    def _build_order(self, turn: int, gold: int, counts: dict[str, int]) -> list[str]:
        order: list[str] = []
        if counts.get("Barracks", 0) < 1:
            order.append("Barracks")
        if counts.get("Mine", 0) < 1:
            order.append("Mine")

        if self.personality == "bomber_rush":
            order += ["Mine", "Airbase", "Airbase", "Factory", "Base", "Mine"]
        elif self.personality == "artillery_siege":
            order += ["Factory", "Factory", "Mine", "Base", "Airbase", "Mine"]
        elif self.personality == "base_hunter":
            order += ["Factory", "Airbase", "Mine", "Base", "Factory", "Mine"]
        elif self.personality == "fast_expand":
            order += ["Mine", "Base", "Mine", "Mine", "Barracks", "Factory", "Airbase"]
        else:
            order += ["Mine", "Factory", "Base", "Airbase", "Factory", "Mine"]

        if gold >= 1200:
            order += ["Factory", "Airbase", "Base", "Mine"]
        if turn >= 150:
            order += ["Base", "Factory", "Airbase"]
        return order

    def _building_target(self, kind: str, turn: int, gold: int) -> int:
        targets = {
            "bomber_rush": {
                "Mine": 3,
                "Barracks": 1,
                "Factory": 1,
                "Airbase": 3,
                "Base": 2,
            },
            "artillery_siege": {
                "Mine": 3,
                "Barracks": 1,
                "Factory": 3,
                "Airbase": 1,
                "Base": 2,
            },
            "base_hunter": {
                "Mine": 3,
                "Barracks": 1,
                "Factory": 2,
                "Airbase": 2,
                "Base": 2,
            },
            "fast_expand": {
                "Mine": 7,
                "Barracks": 2,
                "Factory": 2,
                "Airbase": 1,
                "Base": 4,
            },
            "treaty_ambush": {
                "Mine": 4,
                "Barracks": 1,
                "Factory": 2,
                "Airbase": 2,
                "Base": 3,
            },
        }[self.personality]
        target = targets.get(kind, 0)
        if gold >= 2000 and kind in ("Factory", "Airbase", "Base"):
            target += 1
        if turn >= 200 and kind == "Base":
            target += 1
        return target

    def _build_reserve(self, kind: str) -> int:
        if self.personality in ("bomber_rush", "base_hunter") and kind == "Airbase":
            return 0
        if self.personality == "fast_expand" and kind in ("Mine", "Base"):
            return 0
        return 50

    def _produce(
        self,
        grid: HexGrid,
        turn: int,
        gold: int,
        flat: FlatObservation,
        complete: list[dict[str, Any]],
        actions: list[Any],
    ) -> int:
        counts = self._unit_counts(flat.own_units)
        producers = [b for b in complete if b["type"] in PRODUCTION_BUILDINGS]
        producers.sort(key=lambda b: ("Airbase", "Factory", "Barracks").index(b["type"]))

        for building in producers:
            free = self._free_neighbors(grid, coord(building), flat.occupied)
            if not free:
                continue
            limit = 2 if gold >= 700 else 1
            for unit_type in self._unit_order(building["type"], counts, turn, gold)[:limit]:
                if not free or gold < UNIT_STATS[unit_type].gold_cost:
                    continue
                target = free.pop(0)
                actions.append(
                    ProduceUnitAction(
                        building_id=building["id"],
                        unit_type=unit_type,
                        target=target,
                    )
                )
                flat.occupied.add((target.q, target.r))
                counts[unit_type] = counts.get(unit_type, 0) + 1
                gold -= UNIT_STATS[unit_type].gold_cost
        return gold

    def _unit_order(
        self, building_type: str, counts: dict[str, int], turn: int, gold: int
    ) -> list[str]:
        if building_type == "Barracks":
            if counts.get("Scout", 0) < 3:
                return ["Scout", "Infantry"]
            if counts.get("Medic", 0) * 4 < counts.get("Infantry", 0) + counts.get("Tank", 0):
                return ["Medic", "Infantry"]
            return ["Infantry", "Scout"]

        if building_type == "Factory":
            if self.personality == "artillery_siege":
                return ["Artillery", "Tank"]
            if counts.get("Tank", 0) < 4:
                return ["Tank", "Artillery"]
            return ["Artillery", "Tank"]

        if building_type == "Airbase":
            if self.personality in ("bomber_rush", "base_hunter"):
                if counts.get("Fighter", 0) < max(1, counts.get("Bomber", 0) // 2):
                    return ["Fighter", "Bomber"]
                return ["Bomber", "Fighter"]
            if counts.get("Fighter", 0) < 2:
                return ["Fighter", "Bomber"]
            return ["Bomber", "Fighter"]

        return []

    def _command_units(
        self,
        grid: HexGrid,
        turn: int,
        flat: FlatObservation,
        complete: list[dict[str, Any]],
        actions: list[Any],
    ) -> None:
        reserved = set(flat.occupied)
        anchors = [b for b in complete if b["type"] == "Base"] or complete
        for unit in sorted(flat.own_units, key=lambda u: self._unit_priority(u["type"])):
            here = coord(unit)
            target = self._best_attack_target(grid, unit, flat.enemies)
            if target is not None:
                actions.append(AttackAction(unit_id=unit["id"], target=coord(target)))

            destination = self._movement_destination(grid, turn, unit, flat, anchors)
            if destination is None:
                continue
            path = self._greedy_path(
                grid,
                here,
                destination,
                unit.get("movement_range", 0),
                reserved,
                flat.terrain,
            )
            if path and len(path) > 1:
                actions.append(MoveAction(unit_id=unit["id"], path=path))
                reserved.discard((here.q, here.r))
                reserved.add((path[-1].q, path[-1].r))

    def _best_attack_target(
        self, grid: HexGrid, unit: dict[str, Any], enemies: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        attack_range = unit.get("attack_range", 0)
        if attack_range < 1:
            return None
        here = coord(unit)
        candidates = [
            enemy
            for enemy in enemies
            if 0 < grid.distance(here, coord(enemy)) <= attack_range
        ]
        if not candidates:
            return None

        def score(enemy: dict[str, Any]) -> tuple[int, int, int]:
            priority = {
                "Base": 0,
                "Bomber": 1,
                "Airbase": 2,
                "Artillery": 3,
                "Factory": 4,
                "Fighter": 5,
                "Tank": 6,
                "Barracks": 7,
                "Mine": 8,
                "Scout": 9,
                "Infantry": 10,
                "Medic": 11,
            }.get(enemy["type"], 12)
            if unit["type"] == "Fighter" and enemy["type"] in ("Bomber", "Fighter"):
                priority -= 3
            return (priority, enemy.get("hp", 999), grid.distance(here, coord(enemy)))

        return min(candidates, key=score)

    def _movement_destination(
        self,
        grid: HexGrid,
        turn: int,
        unit: dict[str, Any],
        flat: FlatObservation,
        anchors: list[dict[str, Any]],
    ) -> HexCoord | None:
        here = coord(unit)
        unit_type = unit["type"]
        enemy_goal = self._enemy_goal(grid, here, flat.enemies)
        anchor_goal = self._nearest_coord(grid, here, [coord(a) for a in anchors])

        if unit_type == "Scout":
            if enemy_goal is not None:
                return enemy_goal
            if anchor_goal is not None:
                return self._explore_away(grid, here, anchor_goal)
            return None

        if unit_type in ("Medic", "Artillery"):
            if enemy_goal is not None:
                return self._range_goal(grid, here, enemy_goal, 2 if unit_type == "Medic" else 3)
            return anchor_goal

        if enemy_goal is not None:
            desired = 1
            if unit_type == "Fighter":
                desired = 2
            elif unit_type == "Artillery":
                desired = 3
            return self._range_goal(grid, here, enemy_goal, desired)

        if unit_type in ("Tank", "Infantry", "Fighter", "Bomber") and anchor_goal is not None:
            if grid.distance(here, anchor_goal) > 3:
                return anchor_goal
        return None

    def _enemy_goal(
        self, grid: HexGrid, here: HexCoord, enemies: list[dict[str, Any]]
    ) -> HexCoord | None:
        visible_bases = [coord(e) for e in enemies if e["type"] == "Base"]
        if visible_bases:
            return self._nearest_coord(grid, here, visible_bases)
        if self.known_enemy_bases:
            return self._nearest_coord(grid, here, list(self.known_enemy_bases.values()))
        visible_high_value = [
            coord(e)
            for e in enemies
            if e["type"] in ("Airbase", "Factory", "Bomber", "Artillery", "Tank")
        ]
        if visible_high_value:
            return self._nearest_coord(grid, here, visible_high_value)
        if enemies:
            return self._nearest_coord(grid, here, [coord(e) for e in enemies])
        return None

    @staticmethod
    def _range_goal(
        grid: HexGrid, here: HexCoord, target: HexCoord, desired: int
    ) -> HexCoord | None:
        distance = grid.distance(here, target)
        if distance == desired:
            return None
        if distance > desired:
            return target
        best = here
        best_distance = distance
        for nb in grid.neighbors(here):
            nb_distance = grid.distance(nb, target)
            if nb_distance > best_distance:
                best = nb
                best_distance = nb_distance
        return best if best != here else None

    @staticmethod
    def _explore_away(grid: HexGrid, here: HexCoord, anchor: HexCoord) -> HexCoord:
        best = here
        best_distance = grid.distance(here, anchor)
        for nb in grid.neighbors(here):
            distance = grid.distance(nb, anchor)
            if distance > best_distance:
                best = nb
                best_distance = distance
        return best

    @staticmethod
    def _greedy_path(
        grid: HexGrid,
        start: HexCoord,
        goal: HexCoord,
        movement: int,
        occupied: set[tuple[int, int]],
        terrain: dict[tuple[int, int], str],
    ) -> list[HexCoord] | None:
        if movement <= 0 or start == goal:
            return None
        path = [start]
        current = start
        budget = movement
        reserved = set(occupied)
        reserved.discard((start.q, start.r))
        while budget > 0 and current != goal:
            candidates = []
            for nb in grid.neighbors(current):
                if (nb.q, nb.r) in reserved:
                    continue
                cost = 2 if terrain.get((nb.q, nb.r)) == "difficult" else 1
                if cost <= budget:
                    candidates.append((grid.distance(nb, goal), cost, nb))
            if not candidates:
                break
            _, cost, step = min(candidates)
            if grid.distance(step, goal) >= grid.distance(current, goal) and len(path) > 1:
                break
            path.append(step)
            reserved.add((step.q, step.r))
            current = step
            budget -= cost
        return path if len(path) > 1 else None

    @staticmethod
    def _nearest_coord(
        grid: HexGrid, here: HexCoord, coords: list[HexCoord]
    ) -> HexCoord | None:
        if not coords:
            return None
        return min(coords, key=lambda c: grid.distance(here, c))

    @staticmethod
    def _free_neighbors(
        grid: HexGrid, here: HexCoord, occupied: set[tuple[int, int]]
    ) -> list[HexCoord]:
        return [nb for nb in grid.neighbors(here) if (nb.q, nb.r) not in occupied]

    @staticmethod
    def _building_counts(buildings: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for building in buildings:
            counts[building["type"]] = counts.get(building["type"], 0) + 1
        return counts

    @staticmethod
    def _unit_counts(units: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for unit in units:
            counts[unit["type"]] = counts.get(unit["type"], 0) + 1
        return counts

    @staticmethod
    def _unit_priority(unit_type: str) -> int:
        return {
            "Scout": 0,
            "Bomber": 1,
            "Fighter": 2,
            "Tank": 3,
            "Infantry": 4,
            "Medic": 5,
            "Artillery": 6,
        }.get(unit_type, 9)

    def _base_site(
        self,
        grid: HexGrid,
        flat: FlatObservation,
        complete: list[dict[str, Any]],
    ) -> HexCoord | None:
        candidates = [
            HexCoord(q, r) for q, r in flat.visible if (q, r) not in flat.occupied
        ]
        if not candidates:
            return None
        bases = [coord(b) for b in complete if b["type"] == "Base"] or [
            coord(b) for b in complete
        ]

        def score(candidate: HexCoord) -> tuple[int, int, int]:
            rich = 1 if flat.terrain.get((candidate.q, candidate.r)) == "rich_resource" else 0
            base_dist = min((grid.distance(candidate, b) for b in bases), default=0)
            enemy_dist = min(
                (grid.distance(candidate, coord(e)) for e in flat.enemies), default=8
            )
            return (rich, min(base_dist, 10), min(enemy_dist, 8))

        return max(candidates, key=score)

    def _anchored_site(
        self,
        grid: HexGrid,
        building_type: str,
        flat: FlatObservation,
        complete: list[dict[str, Any]],
    ) -> HexCoord | None:
        candidates: list[tuple[dict[str, Any], HexCoord]] = []
        for building in complete:
            for nb in grid.neighbors(coord(building)):
                if (nb.q, nb.r) not in flat.occupied:
                    candidates.append((building, nb))
        if not candidates:
            return None

        def score(item: tuple[dict[str, Any], HexCoord]) -> tuple[int, int, int]:
            anchor, candidate = item
            rich = 1 if flat.terrain.get((candidate.q, candidate.r)) == "rich_resource" else 0
            enemy_dist = min(
                (grid.distance(candidate, coord(e)) for e in flat.enemies), default=8
            )
            base_bias = 1 if anchor["type"] == "Base" else 0
            if building_type == "Mine":
                return (rich, min(enemy_dist, 8), base_bias)
            return (base_bias, min(enemy_dist, 8), rich)

        return max(candidates, key=score)[1]


@dataclass
class EvalStats:
    player_id: str
    calls: int = 0
    errors: int = 0
    actions: int = 0
    invalid_core_actions: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    min_completed_bases: int = 999
    min_total_bases: int = 999
    max_units: int = 0
    max_buildings: int = 0
    elimination_turn: int | None = None
    gold_samples: list[int] = field(default_factory=list)

    def observe_state(self, runner: GameRunner) -> None:
        assert runner.state is not None
        player = runner.state.players[self.player_id]
        if not player.alive and self.elimination_turn is None:
            self.elimination_turn = runner.state.turn_number
        complete_bases = runner.state.count_bases(self.player_id)
        total_bases = sum(
            1
            for b in runner.state.buildings_for(self.player_id)
            if b.__class__.__name__ == "Base"
        )
        self.min_completed_bases = min(self.min_completed_bases, complete_bases)
        self.min_total_bases = min(self.min_total_bases, total_bases)
        self.max_units = max(self.max_units, len(runner.state.units_for(self.player_id)))
        self.max_buildings = max(
            self.max_buildings, len(runner.state.buildings_for(self.player_id))
        )
        self.gold_samples.append(player.resources.get(ResourceType.GOLD))

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


class EvaluatorRunner(GameRunner):
    def __init__(
        self,
        registrations,
        config,
        actors: dict[str, Any],
        stats: dict[str, EvalStats],
    ):
        super().__init__(registrations, config)
        self.actors = actors
        self.stats = stats

    async def _collect_actions(self, player_urls):  # type: ignore[override]
        assert self.state is not None
        for stat in self.stats.values():
            stat.observe_state(self)
        alive = [pid for pid in player_urls if self.state.players[pid].alive]

        async def one(pid: str):
            obs = build_observation(
                self.state, pid, self.diplomacy, self.chat_log, self.config.max_turns
            )
            start = time.perf_counter()
            try:
                payload = await self.actors[pid].decide(obs)
            except Exception as exc:  # noqa: BLE001
                if pid in self.stats:
                    self.stats[pid].errors += 1
                    print(f"agent error on turn {self.state.turn_number}: {exc}")
                payload = ActionPayload(
                    player_id=pid, turn_number=self.state.turn_number, actions=[]
                )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if pid in self.stats:
                stat = self.stats[pid]
                stat.calls += 1
                stat.total_ms += elapsed_ms
                stat.max_ms = max(stat.max_ms, elapsed_ms)
                stat.actions += len(payload.actions)
                stat.invalid_core_actions += self._count_invalid_core(payload, pid)
            return pid, payload

        return dict(await asyncio.gather(*(one(pid) for pid in alive)))

    def _count_invalid_core(self, payload: ActionPayload, player_id: str) -> int:
        """Approximate invalid move/attack/build/produce actions before resolution."""
        assert self.state is not None
        from engine.actions import ActionValidator

        validator = ActionValidator(self.state, player_id)
        invalid = 0
        for action in payload.actions:
            if isinstance(action, MoveAction) and not validator.validate_move(action):
                invalid += 1
            elif isinstance(action, AttackAction) and not validator.validate_attack(action):
                invalid += 1
            elif (
                isinstance(action, ConstructBuildingAction)
                and action.building_type != "Base"
                and not validator.validate_construct(action)
            ):
                invalid += 1
            elif isinstance(action, ProduceUnitAction) and not validator.validate_produce(action):
                invalid += 1
        return invalid


def load_agent(kind: str):
    if kind == "llm":
        from llm_agent import LLMAgent

        return LLMAgent()
    # any other name loads <kind>_agent.py and instantiates its AlgoAgent class
    # (algo → algo_agent, shadow → shadow_agent, bastion → bastion_agent, ...)
    module_name = kind if kind.endswith("_agent") else f"{kind}_agent"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(f"unknown agent {kind!r}: {exc}") from exc
    return module.AlgoAgent()


def opponent_label(slot: int, mode: str) -> str:
    if mode == "random":
        return "random"
    if mode == "mixed" and slot % 3 == 0:
        return "random"
    return HARD_PERSONALITIES[(slot - 1) % len(HARD_PERSONALITIES)]


def make_opponent(slot: int, mode: str):
    from baseline_random import RandomAgent

    label = opponent_label(slot, mode)
    if label == "random":
        return RandomAgent()
    return HardOpponent(label, slot)


def resource_gold(player) -> int:
    return player.resources.get(ResourceType.GOLD)


def unique_label(kind: str, used: set[str]) -> str:
    label = kind
    if label not in used:
        used.add(label)
        return label
    suffix = 2
    while f"{label}_{suffix}" in used:
        suffix += 1
    label = f"{label}_{suffix}"
    used.add(label)
    return label


def parse_agent_kinds(args: argparse.Namespace) -> list[str]:
    if args.agents:
        kinds = [part.strip() for part in args.agents.split(",") if part.strip()]
        if not kinds:
            raise SystemExit("--agents must include at least one agent name")
        return kinds

    kinds = []
    for kind in ("bastion", args.agent):
        if kind not in kinds:
            kinds.append(kind)
    if args.include_shadow and "shadow" not in kinds:
        kinds.append("shadow")
    return kinds


def tracked_agents(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    kinds = parse_agent_kinds(args)
    if len(kinds) > args.players:
        raise SystemExit(
            f"--agents lists {len(kinds)} agent(s), but --players is only {args.players}"
        )

    used_labels: set[str] = set()
    tracked: list[tuple[str, str, str]] = []
    for slot, kind in enumerate(kinds):
        tracked.append((f"player-{slot}", unique_label(kind, used_labels), kind))
    return tracked


async def run_one(seed: int, args: argparse.Namespace) -> dict[str, Any]:
    registrations = [PlayerRegistration(PLAYER_ID, PLAYER_ID, "local://agent")]
    registrations += [
        PlayerRegistration(f"player-{i}", f"player-{i}", "local://opponent")
        for i in range(1, args.players)
    ]

    tracked = tracked_agents(args)
    labels_by_pid = {pid: label for pid, label, _ in tracked}
    stats = {pid: EvalStats(pid) for pid, _, _ in tracked}
    actors = {pid: load_agent(kind) for pid, _, kind in tracked}
    tracked_ids = set(actors)
    for i in range(1, args.players):
        pid = f"player-{i}"
        if pid in tracked_ids:
            continue
        label = opponent_label(i, args.opponents)
        actors[pid] = make_opponent(i, args.opponents)
        if args.benchmark_opponents and label in HARD_PERSONALITIES:
            labels_by_pid[pid] = label
            stats[pid] = EvalStats(pid)

    runner = EvaluatorRunner(
        registrations,
        GameConfig(
            seed=seed,
            map_width=args.width,
            map_height=args.height,
            max_turns=args.turns,
            replay_path=str(ROOT / "replays" / f"eval_seed_{seed}.jsonl"),
        ),
        actors,
        stats,
    )
    runner.initialise()
    if runner.recorder:
        runner.recorder.close()
        runner.recorder = None

    await runner.run()
    for stat in stats.values():
        stat.observe_state(runner)

    assert runner.state is not None
    results: dict[str, list[dict[str, Any]]] = {}
    for pid, label in labels_by_pid.items():
        stat = stats[pid]
        player = runner.state.players[pid]
        survived = player.alive
        survival_turn = args.turns if survived else stat.elimination_turn or runner.state.turn_number
        alive_opponents = sum(
            1 for other_pid, p in runner.state.players.items() if other_pid != pid and p.alive
        )
        eliminated_opponents = args.players - 1 - alive_opponents
        final_units = len(runner.state.units_for(pid))
        final_buildings = len(runner.state.buildings_for(pid))
        final_bases = runner.state.count_bases(pid)
        total_bases = sum(
            1
            for b in runner.state.buildings_for(pid)
            if b.__class__.__name__ == "Base"
        )

        row = {
            "agent": label,
            "player_id": pid,
            "seed": seed,
            "survived": survived,
            "turn": survival_turn,
            "game_turn": runner.state.turn_number,
            "final_bases": final_bases,
            "total_bases": total_bases,
            "units": final_units,
            "buildings": final_buildings,
            "gold": resource_gold(player),
            "eliminated_opponents": eliminated_opponents,
            "alive_opponents": alive_opponents,
            "calls": stat.calls,
            "errors": stat.errors,
            "avg_ms": stat.avg_ms,
            "max_ms": stat.max_ms,
            "avg_actions": stat.actions / stat.calls if stat.calls else 0.0,
            "invalid_core_actions": stat.invalid_core_actions,
            "min_completed_bases": 0
            if stat.min_completed_bases == 999
            else stat.min_completed_bases,
            "max_units": stat.max_units,
            "max_buildings": stat.max_buildings,
            "avg_gold": statistics.mean(stat.gold_samples) if stat.gold_samples else 0,
        }
        results.setdefault(label, []).append(row)
    return results


def score(results: list[dict[str, Any]], turns: int) -> tuple[float, list[str]]:
    survival_rate = statistics.mean(1.0 if r["survived"] else 0.0 for r in results)
    turn_ratio = statistics.mean(r["turn"] / turns for r in results)
    avg_bases = statistics.mean(r["final_bases"] for r in results)
    avg_units = statistics.mean(r["units"] for r in results)
    avg_buildings = statistics.mean(r["buildings"] for r in results)
    avg_elims = statistics.mean(r["eliminated_opponents"] for r in results)
    error_rate = sum(r["errors"] for r in results) / max(1, sum(r["calls"] for r in results))
    invalid_rate = sum(r["invalid_core_actions"] for r in results) / max(
        1, sum(r["calls"] for r in results)
    )
    max_latency = max(r["max_ms"] for r in results)

    value = 0.0
    value += 45.0 * survival_rate
    value += 10.0 * turn_ratio
    value += 15.0 * min(avg_bases / 3.0, 1.0)
    value += 12.0 * min(avg_units / 35.0, 1.0)
    value += 8.0 * min(avg_buildings / 12.0, 1.0)
    value += 5.0 * min(avg_elims / 3.0, 1.0)
    value += 5.0 * max(0.0, 1.0 - error_rate * 10.0 - invalid_rate * 0.5)
    if max_latency > 9000:
        value -= 10.0
    elif max_latency > 5000:
        value -= 4.0

    notes: list[str] = []
    if survival_rate >= 0.95:
        notes.append("survival: excellent")
    elif survival_rate >= 0.75:
        notes.append("survival: decent, but brittle on some maps")
    else:
        notes.append("survival: weak; fix Base redundancy/defense first")

    if avg_bases < 2:
        notes.append("base redundancy is low; hidden maps may punish this")
    elif avg_bases >= 3:
        notes.append("base redundancy is healthy")

    if avg_units < 20:
        notes.append("army size is light; likely vulnerable after treaty cutoff")
    if avg_buildings < 8:
        notes.append("economy/production footprint is small")
    if avg_elims < 1:
        notes.append("offense is conservative; acceptable if survival stays high")
    if invalid_rate > 1:
        notes.append("many core actions look invalid before resolution")
    if max_latency > 9000:
        notes.append("latency is near/over the 10s turn deadline")

    return max(0.0, min(100.0, value)), notes


def print_one_summary(label: str, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    grade, notes = score(results, args.turns)
    survival = statistics.mean(1.0 if r["survived"] else 0.0 for r in results)
    print(f"\nSummary: {label}")
    print(f"survival rate:       {survival * 100:.1f}%")
    print(f"avg finish turn:     {statistics.mean(r['turn'] for r in results):.1f}/{args.turns}")
    print(f"avg final bases:     {statistics.mean(r['final_bases'] for r in results):.2f}")
    print(f"avg final units:     {statistics.mean(r['units'] for r in results):.1f}")
    print(f"avg final buildings: {statistics.mean(r['buildings'] for r in results):.1f}")
    print(f"avg opp eliminations:{statistics.mean(r['eliminated_opponents'] for r in results):.1f}")
    print(f"worst max latency:   {max(r['max_ms'] for r in results):.1f} ms")
    print(f"survival-first score:{grade:.1f}/100")
    for note in notes:
        print(f"- {note}")


def aggregate_rows(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) == 1:
        return rows[0] | {"result_text": "PASS" if rows[0]["survived"] else "FAIL"}
    survival = statistics.mean(1.0 if r["survived"] else 0.0 for r in rows)
    return {
        "agent": label,
        "seed": rows[0]["seed"],
        "result_text": f"{survival * 100:.0f}%",
        "turn": statistics.mean(r["turn"] for r in rows),
        "final_bases": statistics.mean(r["final_bases"] for r in rows),
        "units": statistics.mean(r["units"] for r in rows),
        "buildings": statistics.mean(r["buildings"] for r in rows),
        "gold": statistics.mean(r["gold"] for r in rows),
        "eliminated_opponents": statistics.mean(r["eliminated_opponents"] for r in rows),
        "avg_ms": statistics.mean(r["avg_ms"] for r in rows),
        "max_ms": max(r["max_ms"] for r in rows),
        "invalid_core_actions": sum(r["invalid_core_actions"] for r in rows),
    }


def print_report(results_by_agent: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> None:
    print("\nPer-seed results")
    print(
        "agent            seed  result  turn  bases  units  bldgs  gold   elim  avg_ms  max_ms  invalid"
    )
    labels = [label for label, rows in results_by_agent.items() if rows]
    seeds = sorted({r["seed"] for rows in results_by_agent.values() for r in rows})
    for seed in seeds:
        for label in labels:
            seed_rows = [r for r in results_by_agent[label] if r["seed"] == seed]
            if not seed_rows:
                continue
            row = aggregate_rows(label, seed_rows)
            print(
                f"{label:<15} {row['seed']:>4}  {row['result_text']:<5}  {row['turn']:>4.0f}  "
                f"{row['final_bases']:>5.1f}  {row['units']:>5.1f}  {row['buildings']:>5.1f}  "
                f"{row['gold']:>5.0f}  {row['eliminated_opponents']:>4.1f}  "
                f"{row['avg_ms']:>6.1f}  {row['max_ms']:>6.1f}  "
                f"{row['invalid_core_actions']:>7}"
            )

    print("\nScoreboard")
    scored = []
    for label, rows in results_by_agent.items():
        if not rows:
            continue
        grade, _ = score(rows, args.turns)
        survival = statistics.mean(1.0 if r["survived"] else 0.0 for r in rows)
        scored.append((grade, survival, label))
    for grade, survival, label in sorted(scored, reverse=True):
        print(f"{label:<7} score={grade:5.1f}/100  survival={survival * 100:5.1f}%")

    for label, rows in results_by_agent.items():
        if not rows:
            continue
        print_one_summary(label, rows, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Surprise agent across multiple engine seeds."
    )
    parser.add_argument(
        "--agent",
        default="algo",
        help="primary agent to evaluate with bastion/shadow when --agents is not set",
    )
    parser.add_argument(
        "--agents",
        help=(
            "comma-separated participant agents to evaluate in the same game "
            "(for example: bastion,algo,shadow)"
        ),
    )
    parser.add_argument(
        "--no-shadow",
        dest="include_shadow",
        action="store_false",
        help="disable shadow_agent in the default roster when using --agent",
    )
    parser.set_defaults(include_shadow=True)
    parser.add_argument("--seeds", default="67,68,69,70,71")
    parser.add_argument("--turns", type=int, default=MAX_TURNS)
    parser.add_argument("--players", type=int, default=20)
    parser.add_argument("--width", type=int, default=DEFAULT_MAP_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_MAP_HEIGHT)
    parser.add_argument(
        "--opponents",
        default="hard",
        choices=("hard", "mixed", "random"),
        help="hard = specialized sparring bots, mixed = hard bots plus randoms, random = original baseline",
    )
    parser.add_argument(
        "--no-benchmarks",
        dest="benchmark_opponents",
        action="store_false",
        help="do not print aggregate benchmark scores for the hard-bot personalities",
    )
    parser.set_defaults(benchmark_opponents=True)
    return parser.parse_args()


async def amain() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    tracked = [label for _, label, _ in tracked_agents(args)]
    if args.benchmark_opponents and args.opponents != "random":
        tracked.extend(p for p in HARD_PERSONALITIES if p not in tracked)
    print(
        f"Evaluating agents={','.join(tracked)} over {len(seeds)} seed(s), "
        f"{args.players} players, {args.turns} turns, opponents={args.opponents}"
    )
    results_by_agent: dict[str, list[dict[str, Any]]] = {label: [] for label in tracked}
    for seed in seeds:
        print(f"running seed {seed}...")
        seed_results = await run_one(seed, args)
        for label, rows in seed_results.items():
            results_by_agent.setdefault(label, []).extend(rows)
    print_report(results_by_agent, args)


if __name__ == "__main__":
    asyncio.run(amain())
