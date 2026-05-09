"""BFS over the grid that respects walls and produces an Action sequence.

The agent has a facing direction; turning is a separate action. We model
a state as (position, facing) and expand neighbors via legal Actions:

    FORWARD    — move 1 step in facing direction (if no wall)
    BACKWARD   — move 1 step opposite to facing (if no wall); does not turn
    LEFT       — turn 90° CCW (no movement)
    RIGHT      — turn 90° CW (no movement)
"""

from collections import deque
from typing import Callable, Optional

from constants import Action, DIR_VECTOR, Direction


def _turn_left(d: int) -> int:
    return (d - 1) % 4


def _turn_right(d: int) -> int:
    return (d + 1) % 4


def _opposite(d: int) -> int:
    return (d + 2) % 4


def first_action_to(
    start: tuple[int, int],
    facing: int,
    goals: set[tuple[int, int]],
    can_traverse: Callable[[tuple[int, int], tuple[int, int]], bool],
) -> Optional[Action]:
    """BFS from (start, facing) to any goal cell. Return the first Action.

    `can_traverse(a, b)` answers whether the agent can move from cell a to
    adjacent cell b (i.e. no blocking wall between them). Unknown edges
    should return True (optimistic).
    """
    if start in goals:
        return Action.STAY

    seen = {(start, facing)}
    # frontier: (pos, facing, first_action)
    queue: deque[tuple[tuple[int, int], int, Optional[Action]]] = deque()
    queue.append((start, facing, None))

    while queue:
        pos, dirn, first = queue.popleft()

        for action, next_pos, next_dir in _expand(pos, dirn, can_traverse):
            state = (next_pos, next_dir)
            if state in seen:
                continue
            seen.add(state)
            chosen_first = first if first is not None else action
            if next_pos in goals:
                return chosen_first
            queue.append((next_pos, next_dir, chosen_first))

    return None


def _expand(
    pos: tuple[int, int],
    facing: int,
    can_traverse: Callable[[tuple[int, int], tuple[int, int]], bool],
):
    # FORWARD
    fdx, fdy = DIR_VECTOR[Direction(facing)]
    fwd = (pos[0] + fdx, pos[1] + fdy)
    if can_traverse(pos, fwd):
        yield Action.FORWARD, fwd, facing

    # BACKWARD (does not change facing)
    bdx, bdy = DIR_VECTOR[Direction(_opposite(facing))]
    back = (pos[0] + bdx, pos[1] + bdy)
    if can_traverse(pos, back):
        yield Action.BACKWARD, back, facing

    # Turns (no movement)
    yield Action.LEFT, pos, _turn_left(facing)
    yield Action.RIGHT, pos, _turn_right(facing)


def reachable_cells(
    start: tuple[int, int],
    facing: int,
    can_traverse: Callable[[tuple[int, int], tuple[int, int]], bool],
    max_steps: int = 50,
) -> dict[tuple[int, int], int]:
    """Return a {cell: action_count_to_reach} map for BFS within max_steps."""
    out: dict[tuple[int, int], int] = {start: 0}
    seen = {(start, facing)}
    queue: deque[tuple[tuple[int, int], int, int]] = deque()
    queue.append((start, facing, 0))

    while queue:
        pos, dirn, dist = queue.popleft()
        if dist >= max_steps:
            continue
        for _, next_pos, next_dir in _expand(pos, dirn, can_traverse):
            state = (next_pos, next_dir)
            if state in seen:
                continue
            seen.add(state)
            new_dist = dist + 1
            if next_pos not in out or new_dist < out[next_pos]:
                out[next_pos] = new_dist
            queue.append((next_pos, next_dir, new_dist))
    return out
