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
from typing import Callable, Optional, Sequence

from constants import Action, DIR_VECTOR, Direction


EdgeCost = Callable[[tuple[int, int], tuple[int, int]], Optional[float]]


def _turn_left(d: int) -> int:
    return (d - 1) % 4


def _turn_right(d: int) -> int:
    return (d + 1) % 4


def _opposite(d: int) -> int:
    return (d + 2) % 4


def next_pos_after(
    pos: tuple[int, int],
    facing: int,
    action: int,
) -> tuple[int, int]:
    """Cell the agent ends up in after taking `action` (assuming the move is legal)."""
    if action == Action.FORWARD:
        dx, dy = DIR_VECTOR[Direction(facing)]
        return (pos[0] + dx, pos[1] + dy)
    if action == Action.BACKWARD:
        dx, dy = DIR_VECTOR[Direction(_opposite(facing))]
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
    action_mask: Optional[Sequence[int]] = None,
) -> Optional[Action]:
    """Cheapest path from (start, facing) to any goal cell. Return first Action.

    action_mask, when provided, restricts the first step to only actions the
    environment currently permits (canonical truth vs. our memory belief).
    Subsequent steps are unrestricted — the mask is only valid for step 0.
    """
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
        for action, next_pos, next_dir, step_cost in _expand(pos, dirn, edge_cost, turn_cost):
            if first is None and action_mask is not None and not action_mask[int(action)]:
                continue
            new_cost = cost + step_cost
            if new_cost > max_cost:
                continue
            state = (next_pos, next_dir)
            if new_cost < seen.get(state, float("inf")):
                seen[state] = new_cost
                chosen_first = first if first is not None else action
                counter += 1
                heapq.heappush(heap, (new_cost, counter, next_pos, next_dir, chosen_first))
    return None


def reachable_cells(
    start: tuple[int, int],
    facing: int,
    edge_cost: EdgeCost,
    *,
    max_cost: float = 50.0,
    turn_cost: float = 1.0,
    action_mask: Optional[Sequence[int]] = None,
) -> dict[tuple[int, int], float]:
    """Cell -> cheapest cost to reach it from (start, facing).

    action_mask gates first-step moves so distance estimates reflect what
    is actually reachable this tick (e.g. a masked FORWARD means the agent
    must turn first, adding a turn_cost to all cells in that direction).
    """
    counter = 0
    out: dict[tuple[int, int], float] = {start: 0.0}
    seen: dict[tuple[tuple[int, int], int], float] = {(start, facing): 0.0}
    # is_first: True only for the initial expansion from (start, facing).
    heap: list[tuple[float, int, tuple[int, int], int, bool]] = [
        (0.0, counter, start, facing, True)
    ]

    while heap:
        cost, _, pos, dirn, is_first = heapq.heappop(heap)
        if cost > max_cost:
            continue
        if cost > seen.get((pos, dirn), float("inf")):
            continue
        for action, next_pos, next_dir, step_cost in _expand(pos, dirn, edge_cost, turn_cost):
            if is_first and action_mask is not None and not action_mask[int(action)]:
                continue
            new_cost = cost + step_cost
            if new_cost > max_cost:
                continue
            state = (next_pos, next_dir)
            if new_cost < seen.get(state, float("inf")):
                seen[state] = new_cost
                if next_pos not in out or new_cost < out[next_pos]:
                    out[next_pos] = new_cost
                counter += 1
                heapq.heappush(heap, (new_cost, counter, next_pos, next_dir, False))
    return out


def _expand(
    pos: tuple[int, int],
    facing: int,
    edge_cost: EdgeCost,
    turn_cost: float,
):
    fdx, fdy = DIR_VECTOR[Direction(facing)]
    fwd = (pos[0] + fdx, pos[1] + fdy)
    fwd_cost = edge_cost(pos, fwd)
    if fwd_cost is not None:
        yield Action.FORWARD, fwd, facing, fwd_cost

    bdx, bdy = DIR_VECTOR[Direction(_opposite(facing))]
    back = (pos[0] + bdx, pos[1] + bdy)
    back_cost = edge_cost(pos, back)
    if back_cost is not None:
        yield Action.BACKWARD, back, facing, back_cost

    yield Action.LEFT, pos, _turn_left(facing), turn_cost
    yield Action.RIGHT, pos, _turn_right(facing), turn_cost
