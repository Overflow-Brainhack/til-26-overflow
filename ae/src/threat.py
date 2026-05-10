"""Bomb-blast danger projection + predictive enemy targeting.

Danger projection
-----------------
* `project_danger(memory)`        — {tick: set_of_blast_cells} from enemy bombs
* `cells_in_blast(memory, pos)`   — what would a bomb at `pos` damage now?
* `imminent_danger(memory, pos)`  — soonest tick `pos` is hit (None if safe)
* `cells_safe_for_at_least(...)`  — cells with no projected blast in window

Predictive targeting
--------------------
* `predict_enemy_positions(...)`  — for each known enemy, the cells they
  could reach in the bomb's fuse window (uniform random walk through
  passable edges)
* `expected_blast_hits(...)`      — Σ |blast ∩ reachable| / |reachable|
  across all known enemies, treating each as uniform over its cloud

Blast model approximates the simulator's Chebyshev-LOS blast.
"""

import math
from typing import Optional

from constants import BOMB_BLAST_RADIUS, BOMB_TIMER, DIR_VECTOR, GRID_SIZE
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


def predict_enemy_positions(
    memory: MapMemory,
    horizon: int = BOMB_TIMER,
    max_age: int = 5,
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """For each recently-seen enemy, the set of cells it could reach.

    Horizon is extended by sighting age — an enemy seen 2 ticks ago could
    have moved `horizon + 2` more times before our hypothetical bomb detonates.
    """
    out: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for last_pos, last_step in memory.enemy_agents.items():
        age = memory.current_step - last_step
        if age > max_age:
            continue
        steps = horizon + age
        reachable = {last_pos}
        frontier = [last_pos]
        for _ in range(steps):
            next_frontier: list[tuple[int, int]] = []
            for p in frontier:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (p[0] + dx, p[1] + dy)
                    if not memory.in_bounds(n):
                        continue
                    if n in reachable:
                        continue
                    if memory.passable(p, n):
                        reachable.add(n)
                        next_frontier.append(n)
            frontier = next_frontier
            if not frontier:
                break
        out[last_pos] = reachable
    return out


def expected_blast_hits(
    memory: MapMemory,
    blast_cells: set[tuple[int, int]],
    horizon: int = BOMB_TIMER,
) -> float:
    """Expected number of enemy agents inside `blast_cells` at detonation.

    Treats each enemy as uniform over its reachability cloud. Returns the sum
    across enemies; with multiple enemies in range this can exceed 1.0.
    """
    total = 0.0
    for cloud in predict_enemy_positions(memory, horizon).values():
        if not cloud:
            continue
        in_blast = sum(1 for c in cloud if c in blast_cells)
        total += in_blast / len(cloud)
    return total


def expected_blast_hits_drift(
    memory: MapMemory,
    blast_cells: set[tuple[int, int]],
    horizon: int = BOMB_TIMER,
    drift_weight: float = 2.0,
) -> float:
    """Expected hits using a velocity-biased enemy position distribution.

    The uniform model (`expected_blast_hits`) treats each cell in the
    reachability cloud as equally likely — this overcounts because in practice
    enemies continue in their observed direction of travel. Here we weight each
    reachable cell by exp(drift_weight * dot(displacement, vel_unit)), so cells
    in the enemy's direction of travel receive exponentially more probability
    mass. When velocity is unknown the distribution collapses to uniform.

    drift_weight=2.0 makes a cell directly ahead ~7× more likely than one
    directly behind the enemy. Set to 0 to recover the uniform model.
    """
    total = 0.0
    enemy_clouds = predict_enemy_positions(memory, horizon)
    for last_pos, cloud in enemy_clouds.items():
        if not cloud:
            continue
        vel = memory.enemy_velocities.get(last_pos, (0, 0))
        vx, vy = vel
        vmag = math.hypot(vx, vy) or 1.0  # avoid divide-by-zero; 1.0 → uniform

        w_total = 0.0
        w_in_blast = 0.0
        for cell in cloud:
            dx = cell[0] - last_pos[0]
            dy = cell[1] - last_pos[1]
            dot = (dx * vx + dy * vy) / vmag
            w = math.exp(drift_weight * dot)
            w_total += w
            if cell in blast_cells:
                w_in_blast += w
        if w_total > 0:
            total += w_in_blast / w_total
    return total


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
