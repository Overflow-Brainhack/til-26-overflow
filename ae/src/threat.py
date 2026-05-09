"""Bomb-blast danger projection.

Given the map memory's known bombs and walls, produces:
  * `cells_in_danger_at(t)` — set of cells the blast covers `t` ticks from now.
  * `imminent_danger(pos)`  — soonest tick at which `pos` is hit (or None).

Blast model approximates the simulator's Chebyshev-LOS blast:
  the bomb damages every cell `p` with `max(|dx|, |dy|) <= blast_radius`
  whose supercover line from the bomb is not blocked by a wall edge.
"""

from typing import Optional

from constants import BOMB_BLAST_RADIUS, BOMB_TIMER, GRID_SIZE
from map_memory import MapMemory


# How many ticks ahead we project bomb risk. Slightly more than max timer.
LOOKAHEAD_TICKS = BOMB_TIMER + 2


def cells_in_blast(memory: MapMemory, bomb_pos: tuple[int, int]) -> set[tuple[int, int]]:
    """Return all cells the bomb at bomb_pos would damage if it detonated now."""
    ox, oy = bomb_pos
    out: set[tuple[int, int]] = set()
    for dx in range(-BOMB_BLAST_RADIUS, BOMB_BLAST_RADIUS + 1):
        for dy in range(-BOMB_BLAST_RADIUS, BOMB_BLAST_RADIUS + 1):
            tx, ty = ox + dx, oy + dy
            if not (0 <= tx < GRID_SIZE and 0 <= ty < GRID_SIZE):
                continue
            if _los(memory, (ox, oy), (tx, ty)):
                out.add((tx, ty))
    return out


def project_danger(memory: MapMemory) -> dict[int, set[tuple[int, int]]]:
    """Map tick offset -> cells that will be hit by an *enemy* bomb at that tick.

    Friendly fire is off in the simulator (dynamics.py: same-team defenders
    are skipped), so we never count our own bombs as threats to ourselves.
    """
    timeline: dict[int, set[tuple[int, int]]] = {}
    for bomb in memory.bombs.values():
        if bomb.ally:
            continue
        remaining = bomb.remaining(memory.current_step)
        if remaining > LOOKAHEAD_TICKS:
            continue
        tick = max(remaining, 0)
        cells = cells_in_blast(memory, bomb.pos)
        timeline.setdefault(tick, set()).update(cells)
    return timeline


def imminent_danger(memory: MapMemory, pos: tuple[int, int]) -> Optional[int]:
    """Soonest tick offset at which `pos` is in a blast, or None."""
    timeline = project_danger(memory)
    return min((t for t, cells in timeline.items() if pos in cells), default=None)


def cells_safe_for_at_least(memory: MapMemory, ticks: int) -> set[tuple[int, int]]:
    """Cells not in any projected blast within the next `ticks` ticks."""
    timeline = project_danger(memory)
    danger: set[tuple[int, int]] = set()
    for t, cells in timeline.items():
        if t <= ticks:
            danger.update(cells)
    return {(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if (x, y) not in danger}


def _los(memory: MapMemory, src: tuple[int, int], dst: tuple[int, int]) -> bool:
    """Supercover-line LOS check; edges blocked by either wall type stop it."""
    if src == dst:
        return True
    cur = src
    while cur != dst:
        dx = dst[0] - cur[0]
        dy = dst[1] - cur[1]
        sx = 1 if dx > 0 else -1 if dx < 0 else 0
        sy = 1 if dy > 0 else -1 if dy < 0 else 0
        nxt = (cur[0] + sx, cur[1] + sy)

        if sx != 0 and sy != 0:
            # Diagonal — passable iff at least one of the two L-paths is open.
            via_a = (cur[0] + sx, cur[1])
            via_b = (cur[0], cur[1] + sy)
            path_a = memory.passable(cur, via_a) and memory.passable(via_a, nxt)
            path_b = memory.passable(cur, via_b) and memory.passable(via_b, nxt)
            if not (path_a or path_b):
                return False
        else:
            if not memory.passable(cur, nxt):
                return False
        cur = nxt
    return True
