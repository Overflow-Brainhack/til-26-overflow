"""Policy interface and the heuristic implementation.

The Policy abstract class lets us swap in a learned policy later without
touching the manager or server. The heuristic policy implements a priority
decision tree:
    1. Frozen → STAY
    2. Imminent enemy-blast danger → dodge to nearest safe cell
    3. Attack opportunity (enemy in our bomb's blast OR predicted to be) → PLACE_BOMB
    4. Defend (enemy near our base) → intercept
    5. Collect highest value-per-distance tile (optionally through walls)
    6. Explore frontier
    7. STAY (final fallback)

Toggles (`HeuristicPolicy(**kwargs)` / CLI flags in auto_play.py):
    predictive_bomb              — bomb based on expected hits, not just current overlap
    predictive_bomb_threshold    — minimum expected hits to bomb predictively
    wall_breaking                — pathfinding may route through destructible walls
    wall_break_cost              — extra cost (≈ ticks lost) to traverse a destructible wall
"""

from abc import ABC, abstractmethod
from typing import Optional

from constants import (
    Action,
    BOMB_TIMER,
    DIR_VECTOR,
    Direction,
)
from map_memory import MapMemory
from observation import ParsedObs
from pathfinding import EdgeCost, first_action_to, next_pos_after, reachable_cells
from threat import (
    cells_in_blast,
    cells_safe_for_at_least,
    expected_blast_hits,
    imminent_danger,
    project_danger,
)


# Search-space tunables (independent of feature toggles).
DEFEND_RADIUS = 4                  # enemy within this many cells of base = threat
EXPLORE_BUDGET = 60.0              # max Dijkstra cost when looking for frontier


class Policy(ABC):
    @abstractmethod
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int: ...


class HeuristicPolicy(Policy):
    """Rule-based agent. All feature toggles are constructor args so callers
    can run baseline-vs-feature comparisons without forking the policy."""

    def __init__(
        self,
        *,
        predictive_bomb: bool = True,
        predictive_bomb_threshold: float = 0.25,
        wall_breaking: bool = True,
        wall_break_cost: float = 5.0,
    ) -> None:
        self.predictive_bomb = predictive_bomb
        self.predictive_bomb_threshold = predictive_bomb_threshold
        self.wall_breaking = wall_breaking
        self.wall_break_cost = wall_break_cost

    # ── main entrypoint ─────────────────────────────────────────────────────
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.frozen_ticks > 0:
            return int(Action.STAY)

        timeline = project_danger(memory)
        danger_now: set[tuple[int, int]] = set()
        for tick, cells in timeline.items():
            if tick <= 1:
                danger_now.update(cells)

        my_blast_tick = imminent_danger(memory, obs.location)
        if my_blast_tick is not None and my_blast_tick <= BOMB_TIMER:
            chosen = self._dodge(obs, memory, timeline)
            if chosen is not None:
                return self._mask_check(chosen, obs)

        attack = self._try_attack(obs, memory)
        if attack is not None:
            return self._mask_check(attack, obs)

        defend = self._try_defend(obs, memory, danger_now)
        if defend is not None:
            return self._mask_check(defend, obs)

        collect = self._try_collect(obs, memory, danger_now)
        if collect is not None:
            return self._mask_check(collect, obs)

        explore = self._try_explore(obs, memory, danger_now)
        if explore is not None:
            return self._mask_check(explore, obs)

        return int(Action.STAY)

    # ── edge cost builder (incorporates wall-breaking flag) ────────────────
    def _edge_cost(
        self,
        memory: MapMemory,
        *,
        danger_avoid: Optional[set[tuple[int, int]]] = None,
        allow_walls: Optional[bool] = None,
    ) -> EdgeCost:
        """Build an EdgeCost reflecting current toggles + per-call overrides.

        allow_walls defaults to self.wall_breaking. Pass False explicitly when
        wall-breaking would be unsafe (e.g. dodging — no time to wait for a
        bomb to clear a wall).
        """
        if allow_walls is None:
            allow_walls = self.wall_breaking
        wall_cost = self.wall_break_cost

        def cost(a: tuple[int, int], b: tuple[int, int]) -> Optional[float]:
            if not memory.in_bounds(b):
                return None
            if memory.passable(a, b):
                if danger_avoid is not None and b in danger_avoid:
                    return None
                return 1.0
            if allow_walls and memory.edge_is_destructible_wall(a, b):
                return wall_cost
            return None

        return cost

    # ── sub-strategies ──────────────────────────────────────────────────────

    def _dodge(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        timeline: dict[int, set[tuple[int, int]]],
    ) -> Optional[int]:
        immediate = timeline.get(0, set()) | timeline.get(1, set())

        # Dodging never breaks walls — no time for a bomb fuse during evacuation.
        edge = self._edge_cost(memory, allow_walls=False)

        def dodge_cost(a: tuple[int, int], b: tuple[int, int]) -> Optional[float]:
            base = edge(a, b)
            if base is None:
                return None
            if b in immediate:
                return None
            return base

        safe = cells_safe_for_at_least(memory, BOMB_TIMER + 1)
        if not safe:
            return self._panic_move(obs, memory, timeline)

        action = first_action_to(obs.location, obs.direction, safe, dodge_cost)
        if action is not None:
            return int(action)
        return self._panic_move(obs, memory, timeline)

    def _panic_move(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        timeline: dict[int, set[tuple[int, int]]],
    ) -> int:
        best_action = int(Action.STAY)
        best_tick = imminent_danger(memory, obs.location) or 99

        for action, dest in self._immediate_neighbors(obs, memory):
            tick = imminent_danger(memory, dest) if dest != obs.location else best_tick
            tick = tick if tick is not None else 99
            if tick > best_tick:
                best_tick = tick
                best_action = int(action)
        return best_action

    def _try_attack(self, obs: ParsedObs, memory: MapMemory) -> Optional[int]:
        if obs.action_mask[Action.PLACE_BOMB] != 1:
            return None
        if obs.team_bombs <= 0:
            return None
        # Don't double-bomb the same cell — friendly-fire is off so this isn't
        # unsafe, just wasteful.
        sitting_bomb = memory.bombs.get(obs.location)
        if sitting_bomb is not None and sitting_bomb.ally:
            return None

        targets = cells_in_blast(memory, obs.location)

        # Definite hits: enemies / enemy bases currently in blast.
        definite = 0.0
        for p in memory.enemy_agents:
            if p in targets:
                definite += 1.0
        for p in memory.enemy_bases:
            if p in targets:
                definite += 2.0  # bases score more (50 pts vs ~20 damage on agent)
        if definite >= 1.0:
            return int(Action.PLACE_BOMB)

        # Predictive: would the bomb plausibly hit a moving enemy by detonation?
        if self.predictive_bomb:
            expected = expected_blast_hits(memory, targets, BOMB_TIMER)
            if expected >= self.predictive_bomb_threshold:
                return int(Action.PLACE_BOMB)

        return None

    def _try_defend(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        if memory.ally_base is None or not memory.enemy_agents:
            return None
        bx, by = memory.ally_base
        threats = {
            p for p in memory.enemy_agents
            if abs(p[0] - bx) + abs(p[1] - by) <= DEFEND_RADIUS
        }
        if not threats:
            return None

        edge = self._edge_cost(memory, danger_avoid=danger_now)
        action = first_action_to(obs.location, obs.direction, threats, edge)
        if action is None:
            return None
        return self._maybe_wall_break(obs, memory, action)

    def _try_collect(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        candidates = memory.collectible_cells()
        if not candidates:
            return None

        edge = self._edge_cost(memory, danger_avoid=danger_now)
        distances = reachable_cells(
            obs.location, obs.direction, edge, max_cost=EXPLORE_BUDGET
        )

        best_score = 0.0
        best_cell: Optional[tuple[int, int]] = None
        for cell in candidates:
            if cell not in distances:
                continue
            value = memory.tile_value(cell)
            if value <= 0:
                continue
            score = value / (distances[cell] + 1.0)
            if score > best_score:
                best_score = score
                best_cell = cell
        if best_cell is None:
            return None

        action = first_action_to(obs.location, obs.direction, {best_cell}, edge)
        if action is None:
            return None
        return self._maybe_wall_break(obs, memory, action)

    def _try_explore(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        edge = self._edge_cost(memory, danger_avoid=danger_now)

        frontier: set[tuple[int, int]] = set()
        for cell in memory.last_seen_step:
            x, y = cell
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nbr = (x + dx, y + dy)
                if memory.in_bounds(nbr) and nbr not in memory.last_seen_step:
                    frontier.add(cell)
                    break
        if not frontier:
            return None

        action = first_action_to(obs.location, obs.direction, frontier, edge)
        if action is None:
            return None
        return self._maybe_wall_break(obs, memory, action)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _maybe_wall_break(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        action: int,
    ) -> Optional[int]:
        """If the planned move crosses a destructible wall, substitute PLACE_BOMB.

        If we already placed an ally bomb at our cell that will break the wall,
        STAY and let it detonate instead of wasting another bomb.
        """
        if action not in (Action.FORWARD, Action.BACKWARD):
            return int(action)
        next_pos = next_pos_after(obs.location, obs.direction, action)
        if not memory.edge_is_destructible_wall(obs.location, next_pos):
            return int(action)

        if not self.wall_breaking:
            return None  # planning shouldn't have produced this — reject

        sitting_bomb = memory.bombs.get(obs.location)
        if sitting_bomb is not None and sitting_bomb.ally:
            return int(Action.STAY)

        if obs.action_mask[Action.PLACE_BOMB] == 1 and obs.team_bombs > 0:
            return int(Action.PLACE_BOMB)
        return None

    def _immediate_neighbors(
        self,
        obs: ParsedObs,
        memory: MapMemory,
    ) -> list[tuple[Action, tuple[int, int]]]:
        out: list[tuple[Action, tuple[int, int]]] = []
        fdx, fdy = DIR_VECTOR[Direction(obs.direction)]
        fwd = (obs.location[0] + fdx, obs.location[1] + fdy)
        if memory.in_bounds(fwd) and memory.passable(obs.location, fwd):
            out.append((Action.FORWARD, fwd))
        bdx, bdy = DIR_VECTOR[Direction((obs.direction + 2) % 4)]
        back = (obs.location[0] + bdx, obs.location[1] + bdy)
        if memory.in_bounds(back) and memory.passable(obs.location, back):
            out.append((Action.BACKWARD, back))
        out.append((Action.LEFT, obs.location))
        out.append((Action.RIGHT, obs.location))
        out.append((Action.STAY, obs.location))
        return out

    def _mask_check(self, action: int, obs: ParsedObs) -> int:
        if 0 <= action < len(obs.action_mask) and obs.action_mask[action] == 1:
            return action
        for fallback in (Action.STAY, Action.LEFT, Action.RIGHT):
            if obs.action_mask[fallback] == 1:
                return int(fallback)
        return int(Action.STAY)
