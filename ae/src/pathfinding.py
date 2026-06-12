"""Dijkstra over (position, facing) state space, parameterised by edge cost.

The agent has a facing direction; turning is its own action. State expansion:
    FORWARD    — move 1 step in facing direction (cost = edge_cost(here, ahead))
    BACKWARD   — move 1 step opposite to facing (cost = edge_cost(here, behind))
    LEFT       — turn 90° CCW (cost = turn_cost)
    RIGHT      — turn 90° CW  (cost = turn_cost)

`edge_cost(a, b)` returns the traversal cost from cell `a` to adjacent cell
`b`, or `None` if impassable. This lets callers express "free passage = 1,
destructible wall = wall_break_cost, structural wall = impassable" in a
single function — pathfinding stays generic, policy decides the cost model.
"""

import heapq
import math
from typing import Callable, Optional

from constants import Action, DIR_VECTOR, Direction


EdgeCost = Callable[[tuple[int, int], tuple[int, int]], Optional[float]]
_DIRS = tuple(DIR_VECTOR[Direction(i)] for i in range(4))
_LEFT = (3, 0, 1, 2)
_RIGHT = (1, 2, 3, 0)
_OPPOSITE = (2, 3, 0, 1)


def _turn_left(d: int) -> int:
    return _LEFT[d]


def _turn_right(d: int) -> int:
    return _RIGHT[d]


def _opposite(d: int) -> int:
    return _OPPOSITE[d]


def next_pos_after(
    pos: tuple[int, int],
    facing: int,
    action: int,
) -> tuple[int, int]:
    """Cell the agent ends up in after taking `action` (assuming the move is legal)."""
    if action == Action.FORWARD:
        dx, dy = _DIRS[facing]
        return (pos[0] + dx, pos[1] + dy)
    if action == Action.BACKWARD:
        dx, dy = _DIRS[_OPPOSITE[facing]]
        return (pos[0] + dx, pos[1] + dy)
    return pos


def from_can_traverse(
    can_traverse: Callable[[tuple[int, int], tuple[int, int]], bool],
    cost: float = 1.0,
) -> EdgeCost:
    """Adapter: turn a boolean traversability check into an EdgeCost."""

    def f(a: tuple[int, int], b: tuple[int, int]) -> Optional[float]:
        return cost if can_traverse(a, b) else None

    return f


def first_action_to(
    start: tuple[int, int],
    facing: int,
    goals: set[tuple[int, int]],
    edge_cost: EdgeCost,
    *,
    turn_cost: float = 1.0,
    max_cost: float = 200.0,
) -> Optional[Action]:
    """Cheapest path from (start, facing) to any goal cell. Return first Action."""
    if start in goals:
        return Action.STAY

    counter = 0
    heap: list[tuple[float, int, tuple[int, int], int, Optional[Action]]] = [
        (0.0, counter, start, facing, None)
    ]
    seen: dict[tuple[tuple[int, int], int], float] = {(start, facing): 0.0}

    while heap:
        cost, _, pos, dirn, first = heapq.heappop(heap)
        if cost > max_cost:
            continue
        if pos in goals:
            return first
        if cost > seen.get((pos, dirn), float("inf")):
            continue
        for action, next_pos, next_dir, step_cost in _expand(
            pos, dirn, edge_cost, turn_cost
        ):
            new_cost = cost + step_cost
            if new_cost > max_cost:
                continue
            state = (next_pos, next_dir)
            if new_cost < seen.get(state, float("inf")):
                seen[state] = new_cost
                chosen_first = first if first is not None else action
                counter += 1
                heapq.heappush(
                    heap, (new_cost, counter, next_pos, next_dir, chosen_first)
                )
    return None


def temporal_first_action_to(
    start: tuple[int, int],
    facing: int,
    goals: set[tuple[int, int]],
    edge_cost: EdgeCost,
    danger_timeline: dict[int, set[tuple[int, int]]],
    *,
    turn_cost: float = 1.0,
    max_cost: float = 200.0,
) -> Optional[Action]:
    """Backward-compatible wrapper for true time-space pathfinding."""
    return time_space_first_action_to(
        start,
        facing,
        goals,
        edge_cost,
        danger_timeline,
        turn_cost=turn_cost,
        max_cost=max_cost,
    )


def time_space_first_action_to(
    start: tuple[int, int],
    facing: int,
    goals: set[tuple[int, int]],
    edge_cost: EdgeCost,
    danger_timeline: dict[int, set[tuple[int, int]]],
    *,
    turn_cost: float = 1.0,
    wait_cost: float = 1.0,
    max_cost: float = 200.0,
) -> Optional[Action]:
    """Cheapest path through `(position, facing, tick)` state space.

    A cell is blocked only at ticks where `danger_timeline[tick]` contains it.
    This lets callers move through a blast footprint before or after detonation,
    and also lets the search revisit the same `(position, facing)` at a later
    tick after temporary danger has cleared.
    """
    if start in goals and start not in danger_timeline.get(0, ()):
        return Action.STAY

    counter = 0
    start_tick = 0
    max_tick = int(math.ceil(max_cost))
    heap: list[tuple[float, int, tuple[int, int], int, int, Optional[Action]]] = [
        (0.0, counter, start, facing, start_tick, None)
    ]
    seen: dict[tuple[tuple[int, int], int, int], float] = {
        (start, facing, start_tick): 0.0
    }

    while heap:
        cost, _, pos, dirn, tick, first = heapq.heappop(heap)
        if cost > max_cost:
            continue
        state = (pos, dirn, tick)
        if cost > seen.get(state, float("inf")):
            continue
        if pos in goals:
            return first if first is not None else Action.STAY
        for action, next_pos, next_dir, step_cost in _expand_time_space(
            pos, dirn, edge_cost, turn_cost, wait_cost
        ):
            arrival = cost + step_cost
            arrival_tick = int(round(arrival))
            if arrival_tick > max_tick:
                continue
            if next_pos in danger_timeline.get(arrival_tick, ()):
                continue
            new_cost = arrival
            if new_cost > max_cost:
                continue
            next_state = (next_pos, next_dir, arrival_tick)
            if new_cost < seen.get(next_state, float("inf")):
                seen[next_state] = new_cost
                chosen_first = first if first is not None else action
                counter += 1
                heapq.heappush(
                    heap,
                    (
                        new_cost,
                        counter,
                        next_pos,
                        next_dir,
                        arrival_tick,
                        chosen_first,
                    ),
                )
    return None


def time_space_reachable_cells(
    start: tuple[int, int],
    facing: int,
    edge_cost: EdgeCost,
    danger_timeline: dict[int, set[tuple[int, int]]],
    *,
    max_cost: float = 50.0,
    turn_cost: float = 1.0,
    wait_cost: float = 1.0,
) -> dict[tuple[int, int], float]:
    """Cell -> earliest safe arrival cost through `(position, facing, tick)`."""
    counter = 0
    start_tick = 0
    max_tick = int(math.ceil(max_cost))
    out: dict[tuple[int, int], float] = {}
    if start not in danger_timeline.get(0, ()):
        out[start] = 0.0
    seen: dict[tuple[tuple[int, int], int, int], float] = {
        (start, facing, start_tick): 0.0
    }
    heap: list[tuple[float, int, tuple[int, int], int, int]] = [
        (0.0, counter, start, facing, start_tick)
    ]

    while heap:
        cost, _, pos, dirn, tick = heapq.heappop(heap)
        if cost > max_cost:
            continue
        state = (pos, dirn, tick)
        if cost > seen.get(state, float("inf")):
            continue
        for _action, next_pos, next_dir, step_cost in _expand_time_space(
            pos, dirn, edge_cost, turn_cost, wait_cost
        ):
            new_cost = cost + step_cost
            arrival_tick = int(round(new_cost))
            if new_cost > max_cost or arrival_tick > max_tick:
                continue
            if next_pos in danger_timeline.get(arrival_tick, ()):
                continue
            next_state = (next_pos, next_dir, arrival_tick)
            if new_cost < seen.get(next_state, float("inf")):
                seen[next_state] = new_cost
                if next_pos not in out or new_cost < out[next_pos]:
                    out[next_pos] = new_cost
                counter += 1
                heapq.heappush(
                    heap, (new_cost, counter, next_pos, next_dir, arrival_tick)
                )
    return out


def reachable_cells(
    start: tuple[int, int],
    facing: int,
    edge_cost: EdgeCost,
    *,
    max_cost: float = 50.0,
    turn_cost: float = 1.0,
) -> dict[tuple[int, int], float]:
    """Cell -> cheapest cost to reach it from (start, facing)."""
    counter = 0
    out: dict[tuple[int, int], float] = {start: 0.0}
    seen: dict[tuple[tuple[int, int], int], float] = {(start, facing): 0.0}
    heap: list[tuple[float, int, tuple[int, int], int]] = [
        (0.0, counter, start, facing)
    ]

    while heap:
        cost, _, pos, dirn = heapq.heappop(heap)
        if cost > max_cost:
            continue
        if cost > seen.get((pos, dirn), float("inf")):
            continue
        for _action, next_pos, next_dir, step_cost in _expand(
            pos, dirn, edge_cost, turn_cost
        ):
            new_cost = cost + step_cost
            if new_cost > max_cost:
                continue
            state = (next_pos, next_dir)
            if new_cost < seen.get(state, float("inf")):
                seen[state] = new_cost
                if next_pos not in out or new_cost < out[next_pos]:
                    out[next_pos] = new_cost
                counter += 1
                heapq.heappush(heap, (new_cost, counter, next_pos, next_dir))
    return out


def _expand(
    pos: tuple[int, int],
    facing: int,
    edge_cost: EdgeCost,
    turn_cost: float,
):
    fdx, fdy = _DIRS[facing]
    fwd = (pos[0] + fdx, pos[1] + fdy)
    fwd_cost = edge_cost(pos, fwd)
    if fwd_cost is not None:
        yield Action.FORWARD, fwd, facing, fwd_cost

    bdx, bdy = _DIRS[_OPPOSITE[facing]]
    back = (pos[0] + bdx, pos[1] + bdy)
    back_cost = edge_cost(pos, back)
    if back_cost is not None:
        # ε penalty: backward travel loses visibility (viewcone is 4 ahead vs 2 behind)
        # and breaks the LEFT+BACKWARD == RIGHT+FORWARD Dijkstra tie that suppressed RIGHT.
        yield Action.BACKWARD, back, facing, back_cost + 0.01

    yield Action.RIGHT, pos, _RIGHT[facing], turn_cost
    yield Action.LEFT, pos, _LEFT[facing], turn_cost


def _expand_time_space(
    pos: tuple[int, int],
    facing: int,
    edge_cost: EdgeCost,
    turn_cost: float,
    wait_cost: float,
):
    yield from _expand(pos, facing, edge_cost, turn_cost)
    yield Action.STAY, pos, facing, wait_cost
