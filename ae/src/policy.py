"""Policy interface and the heuristic implementation.

The Policy abstract class lets us swap a learned policy in later without
touching the manager or server. The heuristic policy implements a priority
decision tree:
    1. Frozen → STAY
    2. Imminent blast danger → dodge to nearest safe cell
    3. Attack opportunity (enemy in our bomb's blast) with safe escape → PLACE_BOMB
    4. Defend (enemy near our base) → intercept
    5. Collect highest value-per-distance tile
    6. Explore frontier
    7. STAY (final fallback)
"""

from abc import ABC, abstractmethod
from typing import Optional

from constants import (
    Action,
    BOMB_BLAST_RADIUS,
    BOMB_TIMER,
    GRID_SIZE,
    REWARD_MISSION,
    REWARD_RECON,
    REWARD_RESOURCE,
)
from map_memory import MapMemory
from observation import ParsedObs
from pathfinding import first_action_to, reachable_cells
from threat import (
    cells_in_blast,
    cells_safe_for_at_least,
    imminent_danger,
    project_danger,
)


# Tunables.
DEFEND_RADIUS = 4                  # enemy within this many cells of base = threat
EXPLORE_RADIUS = 60                 # max BFS depth when looking for frontier


class Policy(ABC):
    @abstractmethod
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        ...


class HeuristicPolicy(Policy):
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.frozen_ticks > 0:
            return int(Action.STAY)

        # Project bomb danger once per tick.
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

        # Attack: place bomb if it would hit something AND we can escape.
        attack = self._try_attack(obs, memory)
        if attack is not None:
            return self._mask_check(attack, obs)

        # Defend: enemy near base → intercept (which may lead to bomb next tick).
        defend = self._try_defend(obs, memory, danger_now)
        if defend is not None:
            return self._mask_check(defend, obs)

        # Collect: best value-per-distance tile.
        collect = self._try_collect(obs, memory, danger_now)
        if collect is not None:
            return self._mask_check(collect, obs)

        # Explore.
        explore = self._try_explore(obs, memory, danger_now)
        if explore is not None:
            return self._mask_check(explore, obs)

        return int(Action.STAY)

    # ── sub-strategies ──────────────────────────────────────────────────────

    def _dodge(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        timeline: dict[int, set[tuple[int, int]]],
    ) -> Optional[int]:
        safe = cells_safe_for_at_least(memory, BOMB_TIMER + 1)
        # Avoid stepping into immediate (next-tick) danger.
        immediate = timeline.get(0, set()) | timeline.get(1, set())

        def can_go(a: tuple[int, int], b: tuple[int, int]) -> bool:
            if not memory.in_bounds(b) or not memory.passable(a, b):
                return False
            return b not in immediate or b in safe

        if not safe:
            # No safe cell at all — pick the move that delays getting hit longest.
            return self._panic_move(obs, memory, timeline)

        action = first_action_to(obs.location, obs.direction, safe, can_go)
        if action is not None:
            return int(action)
        return self._panic_move(obs, memory, timeline)

    def _panic_move(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        timeline: dict[int, set[tuple[int, int]]],
    ) -> int:
        """No safe path — pick whichever immediate move maximizes our blast-tick."""
        best_action = int(Action.STAY)
        best_tick = imminent_danger(memory, obs.location) or 99

        candidates = self._immediate_neighbors(obs, memory)
        for action, dest in candidates:
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

        # Friendly fire is off, so our own bomb never harms us or our base —
        # no escape check needed. Just check that the blast hits something
        # worth hitting.
        targets = cells_in_blast(memory, obs.location)
        worth_it = any(p in targets for p in memory.enemy_agents) or any(
            p in targets for p in memory.enemy_bases
        )
        if not worth_it:
            return None

        return int(Action.PLACE_BOMB)

    def _try_defend(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        if memory.ally_base is None or not memory.enemy_agents:
            return None
        bx, by = memory.ally_base
        threats = [
            p for p in memory.enemy_agents
            if abs(p[0] - bx) + abs(p[1] - by) <= DEFEND_RADIUS
        ]
        if not threats:
            return None

        # Move toward the closest threat.
        def can_go(a: tuple[int, int], b: tuple[int, int]) -> bool:
            return (
                memory.in_bounds(b)
                and memory.passable(a, b)
                and b not in danger_now
            )

        action = first_action_to(obs.location, obs.direction, set(threats), can_go)
        return int(action) if action is not None else None

    def _try_collect(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        candidates = memory.collectible_cells()
        if not candidates:
            return None

        def can_go(a: tuple[int, int], b: tuple[int, int]) -> bool:
            return (
                memory.in_bounds(b)
                and memory.passable(a, b)
                and b not in danger_now
            )

        distances = reachable_cells(obs.location, obs.direction, can_go, max_steps=EXPLORE_RADIUS)

        best_score = 0.0
        best_cell: Optional[tuple[int, int]] = None
        for cell in candidates:
            if cell not in distances:
                continue
            dist = distances[cell]
            value = memory.tile_value(cell)
            if value <= 0:
                continue
            score = value / (dist + 1)
            if score > best_score:
                best_score = score
                best_cell = cell

        if best_cell is None:
            return None
        action = first_action_to(obs.location, obs.direction, {best_cell}, can_go)
        return int(action) if action is not None else None

    def _try_explore(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        def can_go(a: tuple[int, int], b: tuple[int, int]) -> bool:
            return (
                memory.in_bounds(b)
                and memory.passable(a, b)
                and b not in danger_now
            )

        # Frontier = known cell with at least one never-seen neighbor.
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
        action = first_action_to(obs.location, obs.direction, frontier, can_go)
        return int(action) if action is not None else None

    # ── helpers ─────────────────────────────────────────────────────────────

    def _immediate_neighbors(
        self,
        obs: ParsedObs,
        memory: MapMemory,
    ) -> list[tuple[Action, tuple[int, int]]]:
        """All single-step movement options + STAY, with destination cells."""
        from constants import DIR_VECTOR, Direction
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
        # Mask says illegal — try a safe alternative.
        for fallback in (Action.STAY, Action.LEFT, Action.RIGHT):
            if obs.action_mask[fallback] == 1:
                return int(fallback)
        return int(Action.STAY)
