"""Feature extraction shared by AE RL training and lightweight inference.

The Colab notebook trains on this exact representation, then exports a tiny
NumPy MLP checkpoint consumed by ``rl_policy.py``. Keeping the feature contract
here avoids a common RL footgun: training one observation layout and serving a
slightly different one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from constants import (
    AGENT_MAX_HEALTH,
    BASE_MAX_HEALTH,
    BOMB_TIMER,
    BOMB_ATTACK,
    DIR_VECTOR,
    Action,
    Direction,
    FREEZE_TURNS,
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_CHANNELS,
    NUM_ITERS,
    VIEWCONE_LENGTH,
    VIEWCONE_WIDTH,
    BASE_VIEW_SIDE,
    BOMB_BLAST_RADIUS,
    BASE_DESTROY_BONUS,
)
from observation import ParsedObs
from pathfinding import first_action_to, next_pos_after, reachable_cells
from threat import cells_in_blast, expected_blast_hits_drift, imminent_danger


FEATURE_VERSION = 3

AGENT_VIEW_SIZE = VIEWCONE_LENGTH * VIEWCONE_WIDTH * NUM_CHANNELS
BASE_VIEW_SIZE = BASE_VIEW_SIDE * BASE_VIEW_SIDE * NUM_CHANNELS

SCALAR_FEATURE_NAMES = (
    "dir_right",
    "dir_down",
    "dir_left",
    "dir_up",
    "loc_x",
    "loc_y",
    "base_x",
    "base_y",
    "health",
    "base_health",
    "frozen",
    "resources",
    "bombs",
    "step",
    "mask_forward",
    "mask_backward",
    "mask_left",
    "mask_right",
    "mask_stay",
    "mask_bomb",
    "rel_base_dx",
    "rel_base_dy",
    "rel_base_dist",
    "enemy_base_dx",
    "enemy_base_dy",
    "enemy_base_dist",
    "enemy_base_hp",
    "enemy_base_known",
    "enemy_agent_count",
    "enemy_agent_dx",
    "enemy_agent_dy",
    "enemy_agent_dist",
    "enemy_agent_known",
    "bomb_count",
    "bomb_dx",
    "bomb_dy",
    "bomb_timer",
    "bomb_ally",
    "bomb_known",
    "danger_here",
    "danger_forward",
    "danger_backward",
    "danger_left",
    "danger_right",
    "danger_stay",
    "danger_bomb",
    "ally_bomb_here",
    "ally_bomb_timer",
    "bomb_enemy_base_now",
    "bomb_enemy_agent_now",
    "bomb_expected_hits",
    "seen_cell_frac",
    "known_tile_frac",
    "known_base_frac",
    "known_edge_frac",
    "frontier_neighbor_frac",
    "unseen_forward",
    "unseen_backward",
    "enemy_agent_age",
    "enemy_agent_vx",
    "enemy_agent_vy",
    "enemy_agent_speed",
    "enemy_agent_pred_dx",
    "enemy_agent_pred_dy",
    "enemy_agent_pred_dist",
    "mission_dx",
    "mission_dy",
    "mission_dist",
    "mission_known",
    "resource_dx",
    "resource_dy",
    "resource_dist",
    "resource_known",
    "recon_dx",
    "recon_dy",
    "recon_dist",
    "recon_known",
    "route_base_forward",
    "route_base_backward",
    "route_base_left",
    "route_base_right",
    "route_base_stay",
    "route_base_bomb",
    "route_base_known",
    "route_base_cost",
    "route_base_target_dx",
    "route_base_target_dy",
    "route_base_target_dist",
    "route_base_hit_value",
    "route_collect_forward",
    "route_collect_backward",
    "route_collect_left",
    "route_collect_right",
    "route_collect_stay",
    "route_collect_bomb",
    "route_collect_known",
    "route_collect_cost",
    "route_collect_value",
)

SCALAR_FEATURE_SIZE = len(SCALAR_FEATURE_NAMES)
FEATURE_SIZE = AGENT_VIEW_SIZE + BASE_VIEW_SIZE + SCALAR_FEATURE_SIZE

_GRID_DENOM = max(1.0, float(GRID_SIZE - 1))
_MAX_DIST = max(1.0, 2.0 * float(GRID_SIZE - 1))
_MAX_UNDIRECTED_EDGES = max(1.0, float(2 * GRID_SIZE * (GRID_SIZE - 1)))
_BASE_ROUTE_BUDGET = 90.0
_COLLECT_ROUTE_BUDGET = 35.0


def extract_features(obs: ParsedObs, memory: Any | None = None) -> np.ndarray:
    """Return the model input vector for one parsed observation.

    The first two blocks are raw sparse view tensors. The scalar tail carries
    normalized public state plus a few optional memory-derived signals. Missing
    memory is valid and results in zeros for memory-only fields; this is useful
    for quick notebook experiments.
    """
    agent_view = _fixed_flatten(obs.agent_view, AGENT_VIEW_SIZE)
    base_view = _fixed_flatten(obs.base_view, BASE_VIEW_SIZE)
    scalars = np.asarray(_scalar_features(obs, memory), dtype=np.float32)
    features = np.concatenate((agent_view, base_view, scalars)).astype(np.float32, copy=False)
    if features.shape != (FEATURE_SIZE,):
        out = np.zeros(FEATURE_SIZE, dtype=np.float32)
        n = min(out.size, features.size)
        out[:n] = features[:n]
        return out
    return features


def action_mask_from_obs(obs: ParsedObs) -> np.ndarray:
    """Return a clean float32 action mask with shape ``(6,)``."""
    mask = np.asarray(obs.action_mask, dtype=np.float32).reshape(-1)
    out = np.zeros(NUM_ACTIONS, dtype=np.float32)
    n = min(NUM_ACTIONS, mask.size)
    out[:n] = mask[:n]
    return out


def safe_action_mask_from_obs(obs: ParsedObs, memory: Any | None = None) -> np.ndarray:
    """Return a conservative tactical mask for learned policies.

    The environment mask only tells us what is legal. This one removes actions
    that are almost always bad for a model learning from sparse rewards:
    immediate bomb danger, redundant bombs, and bombs with no visible tactical
    target. It deliberately falls back to the raw mask if every action would be
    filtered so the server never gets stuck without a legal action.
    """
    raw = action_mask_from_obs(obs)
    if raw.sum() <= 0:
        return raw
    if memory is None:
        return raw
    if obs.frozen_ticks > 0:
        frozen = np.zeros(NUM_ACTIONS, dtype=np.float32)
        frozen[int(Action.STAY)] = raw[int(Action.STAY)]
        return frozen if frozen.sum() > 0 else raw

    mask = raw.copy()
    current_danger = _danger_tick(memory, obs.location)

    for action in (Action.FORWARD, Action.BACKWARD):
        idx = int(action)
        if mask[idx] <= 0.5:
            continue
        dest = _action_destination(obs.location, obs.direction, idx)
        if _danger_tick(memory, dest) is not None and _danger_tick(memory, dest) <= 1:
            mask[idx] = 0.0

    # Turning or staying leaves us in place, so suppress it when a blast is
    # imminent and at least one movement action remains.
    if current_danger is not None and current_danger <= 1:
        for action in (Action.LEFT, Action.RIGHT, Action.STAY):
            mask[int(action)] = 0.0

    if mask[int(Action.PLACE_BOMB)] > 0.5 and not _bomb_has_tactical_value(obs, memory):
        mask[int(Action.PLACE_BOMB)] = 0.0

    if mask.sum() <= 0:
        return raw
    return mask


def _fixed_flatten(array: np.ndarray, size: int) -> np.ndarray:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if flat.size == size:
        return flat
    out = np.zeros(size, dtype=np.float32)
    n = min(size, flat.size)
    out[:n] = flat[:n]
    return out


def _scalar_features(obs: ParsedObs, memory: Any | None) -> list[float]:
    loc = obs.location
    base = obs.base_location
    mask = action_mask_from_obs(obs)

    out: list[float] = []
    out.extend(1.0 if obs.direction == d else 0.0 for d in range(4))
    out.extend((_coord(loc[0]), _coord(loc[1]), _coord(base[0]), _coord(base[1])))
    out.extend(
        (
            _clip01(obs.health / AGENT_MAX_HEALTH),
            _clip01(obs.base_health / BASE_MAX_HEALTH),
            _clip01(obs.frozen_ticks / max(1.0, float(FREEZE_TURNS))),
            _clip01(obs.team_resources / 10.0),
            _clip01(obs.team_bombs / 10.0),
            _clip01(obs.step / max(1.0, float(NUM_ITERS))),
        )
    )
    out.extend(float(x) for x in mask)

    out.extend(_relative_features(loc, base))

    enemy_base = _nearest(loc, getattr(memory, "enemy_bases", ()))
    if enemy_base is None:
        out.extend((0.0, 0.0, 0.0, 1.0, 0.0))
    else:
        out.extend(_relative_features(loc, enemy_base))
        hp = getattr(memory, "enemy_base_health", {}).get(enemy_base, BASE_MAX_HEALTH)
        out.extend((_clip01(float(hp) / BASE_MAX_HEALTH), 1.0))

    enemy_agents = list(getattr(memory, "enemy_agents", {}).keys())
    enemy_agent = _nearest(loc, enemy_agents)
    out.append(_clip01(len(enemy_agents) / 6.0))
    if enemy_agent is None:
        out.extend((0.0, 0.0, 0.0, 0.0))
    else:
        out.extend((*_relative_features(loc, enemy_agent), 1.0))

    bombs = getattr(memory, "bombs", {})
    bomb_pos = _nearest(loc, bombs.keys())
    out.append(_clip01(len(bombs) / 12.0))
    if bomb_pos is None:
        out.extend((0.0, 0.0, 0.0, 0.0, 0.0))
    else:
        bomb = bombs[bomb_pos]
        dx, dy, _dist = _relative_features(loc, bomb_pos)
        timer = 0.0
        if hasattr(bomb, "remaining"):
            timer = _clip01(float(bomb.remaining(obs.step)) / max(1.0, float(BOMB_TIMER)))
        ally = 1.0 if bool(getattr(bomb, "ally", False)) else 0.0
        out.extend((dx, dy, timer, ally, 1.0))

    out.extend(_danger_features(obs, memory))
    out.extend(_bomb_opportunity_features(obs, memory))

    seen = getattr(memory, "last_seen_step", {})
    tiles = getattr(memory, "tile_contents", {})
    bases = getattr(memory, "base_positions", {})
    edges = getattr(memory, "known_edges", {})
    out.extend(
        (
            _clip01(len(seen) / float(GRID_SIZE * GRID_SIZE)),
            _clip01(len(tiles) / float(GRID_SIZE * GRID_SIZE)),
            _clip01(len(bases) / 6.0),
            _clip01(len(edges) / _MAX_UNDIRECTED_EDGES),
        )
    )
    out.extend(_exploration_features(obs, memory))
    out.extend(_enemy_motion_features(obs, memory, enemy_agent))
    out.extend(_tile_target_features(loc, tiles, "mission"))
    out.extend(_tile_target_features(loc, tiles, "resource"))
    out.extend(_tile_target_features(loc, tiles, "recon"))
    out.extend(_base_route_features(obs, memory))
    out.extend(_collect_route_features(obs, memory))

    if len(out) != SCALAR_FEATURE_SIZE:
        raise RuntimeError(
            f"RL scalar feature contract drifted: {len(out)} != {SCALAR_FEATURE_SIZE}"
        )
    return out


def _relative_features(
    origin: tuple[int, int],
    target: tuple[int, int],
) -> tuple[float, float, float]:
    dx = float(target[0] - origin[0])
    dy = float(target[1] - origin[1])
    return (
        float(np.clip(dx / _GRID_DENOM, -1.0, 1.0)),
        float(np.clip(dy / _GRID_DENOM, -1.0, 1.0)),
        _clip01((abs(dx) + abs(dy)) / _MAX_DIST),
    )


def _nearest(
    origin: tuple[int, int],
    candidates,
) -> tuple[int, int] | None:
    best = None
    best_dist = 10**9
    ox, oy = origin
    for candidate in candidates:
        cx, cy = int(candidate[0]), int(candidate[1])
        dist = abs(cx - ox) + abs(cy - oy)
        if dist < best_dist:
            best_dist = dist
            best = (cx, cy)
    return best


def _danger_features(obs: ParsedObs, memory: Any | None) -> list[float]:
    loc = obs.location
    direction = obs.direction
    return [
        _danger_score(memory, loc),
        _danger_score(memory, _action_destination(loc, direction, 0)),
        _danger_score(memory, _action_destination(loc, direction, 1)),
        _danger_score(memory, loc),
        _danger_score(memory, loc),
        _danger_score(memory, loc),
        _danger_score(memory, loc),
        _ally_bomb_here(memory, loc),
        _ally_bomb_timer(memory, loc),
    ]


def _danger_score(memory: Any | None, pos: tuple[int, int]) -> float:
    if memory is None:
        return 0.0
    try:
        tick = imminent_danger(memory, pos)
    except Exception:
        tick = None
    if tick is None:
        return 0.0
    horizon = max(1.0, float(BOMB_TIMER + 1))
    return _clip01((horizon - float(tick)) / horizon)


def _ally_bomb_here(memory: Any | None, pos: tuple[int, int]) -> float:
    bomb = getattr(memory, "bombs", {}).get(pos) if memory is not None else None
    return 1.0 if bomb is not None and bool(getattr(bomb, "ally", False)) else 0.0


def _ally_bomb_timer(memory: Any | None, pos: tuple[int, int]) -> float:
    bomb = getattr(memory, "bombs", {}).get(pos) if memory is not None else None
    if bomb is None or not bool(getattr(bomb, "ally", False)):
        return 0.0
    remaining = bomb.remaining(getattr(memory, "current_step", 0)) if hasattr(bomb, "remaining") else 0
    return _clip01(float(remaining) / max(1.0, float(BOMB_TIMER)))


def _bomb_opportunity_features(obs: ParsedObs, memory: Any | None) -> list[float]:
    if memory is None or obs.team_bombs <= 0:
        return [0.0, 0.0, 0.0]
    try:
        blast = cells_in_blast(memory, obs.location)
    except Exception:
        blast = _simple_blast(obs.location)
    enemy_bases = getattr(memory, "enemy_bases", set())
    enemy_agents = getattr(memory, "enemy_agents", {})
    base_hits = sum(1 for p in enemy_bases if p in blast)
    agent_hits = sum(1 for p in enemy_agents if p in blast)
    try:
        expected_hits = expected_blast_hits_drift(memory, blast, BOMB_TIMER)
    except Exception:
        expected_hits = 0.0
    return [
        _clip01(base_hits / 2.0),
        _clip01(agent_hits / 3.0),
        _clip01(expected_hits / 3.0),
    ]


def _danger_tick(memory: Any | None, pos: tuple[int, int]) -> int | None:
    if memory is None:
        return None
    try:
        return imminent_danger(memory, pos)
    except Exception:
        return None


def _bomb_has_tactical_value(obs: ParsedObs, memory: Any | None) -> bool:
    if memory is None or obs.team_bombs <= 0:
        return False
    sitting_bomb = getattr(memory, "bombs", {}).get(obs.location)
    if sitting_bomb is not None and bool(getattr(sitting_bomb, "ally", False)):
        return False

    try:
        blast = cells_in_blast(memory, obs.location)
    except Exception:
        blast = _simple_blast(obs.location)

    enemy_bases = {
        p for p in getattr(memory, "enemy_bases", set())
        if p in blast and getattr(memory, "enemy_base_health", {}).get(p, BASE_MAX_HEALTH) > 0
    }
    if enemy_bases:
        return True

    enemy_agents = set(getattr(memory, "enemy_agents", {}).keys())
    if enemy_agents & blast:
        return True

    try:
        if expected_blast_hits_drift(memory, blast, BOMB_TIMER) >= 0.35:
            return True
    except Exception:
        pass

    # Wall-breaking bombs are valuable only when we are adjacent to the wall.
    # This matches how the heuristic substitutes PLACE_BOMB for a planned move
    # through a destructible edge.
    for dx, dy in DIR_VECTOR.values():
        nbr = (obs.location[0] + dx, obs.location[1] + dy)
        try:
            if memory.edge_is_destructible_wall(obs.location, nbr):
                return True
        except Exception:
            continue
    for edge in getattr(memory, "destructible_edges", ()):
        try:
            a, b = tuple(edge)
        except Exception:
            continue
        if a in blast or b in blast:
            return True
    return False


def _exploration_features(obs: ParsedObs, memory: Any | None) -> list[float]:
    if memory is None:
        return [0.0, 0.0, 0.0]
    loc = obs.location
    seen = getattr(memory, "last_seen_step", {})
    neighbors = [
        (loc[0] + 1, loc[1]),
        (loc[0] - 1, loc[1]),
        (loc[0], loc[1] + 1),
        (loc[0], loc[1] - 1),
    ]
    frontier = 0
    valid = 0
    for nbr in neighbors:
        if _in_bounds(memory, nbr):
            valid += 1
            if nbr not in seen:
                frontier += 1
    fwd = _action_destination(loc, obs.direction, 0)
    back = _action_destination(loc, obs.direction, 1)
    return [
        _clip01(frontier / max(1.0, float(valid))),
        1.0 if _in_bounds(memory, fwd) and fwd not in seen else 0.0,
        1.0 if _in_bounds(memory, back) and back not in seen else 0.0,
    ]


def _enemy_motion_features(
    obs: ParsedObs,
    memory: Any | None,
    enemy_agent: tuple[int, int] | None,
) -> list[float]:
    if memory is None or enemy_agent is None:
        return [0.0] * 7
    last_step = getattr(memory, "enemy_agents", {}).get(enemy_agent, getattr(memory, "current_step", obs.step))
    age = max(0, int(getattr(memory, "current_step", obs.step)) - int(last_step))
    vx, vy = getattr(memory, "enemy_velocities", {}).get(enemy_agent, (0, 0))
    speed = abs(vx) + abs(vy)
    pred = (
        int(np.clip(enemy_agent[0] + vx * BOMB_TIMER, 0, GRID_SIZE - 1)),
        int(np.clip(enemy_agent[1] + vy * BOMB_TIMER, 0, GRID_SIZE - 1)),
    )
    pred_dx, pred_dy, pred_dist = _relative_features(obs.location, pred)
    return [
        _clip01(age / 12.0),
        float(np.clip(vx, -1, 1)),
        float(np.clip(vy, -1, 1)),
        _clip01(speed / 2.0),
        pred_dx,
        pred_dy,
        pred_dist,
    ]


def _tile_target_features(
    loc: tuple[int, int],
    tiles: dict[tuple[int, int], str],
    kind: str,
) -> list[float]:
    targets = [pos for pos, value in tiles.items() if value == kind]
    target = _nearest(loc, targets)
    if target is None:
        return [0.0, 0.0, 0.0, 0.0]
    dx, dy, dist = _relative_features(loc, target)
    return [dx, dy, dist, 1.0]


def _base_route_features(obs: ParsedObs, memory: Any | None) -> list[float]:
    """Oracle-like route hint to the best known enemy-base firing cell.

    Novice uses a fixed map, so the exported novice cache can tell the policy
    early which direction puts pressure on bases. This is a hint, not a rule:
    the network still sees the mask, danger, local view, and reward signal.
    """
    zeros = [0.0] * 12
    if memory is None:
        return zeros
    live_bases = [
        base for base in getattr(memory, "enemy_bases", set())
        if getattr(memory, "enemy_base_health", {}).get(base, BASE_MAX_HEALTH) > 0
    ]
    if not live_bases:
        return zeros

    edge = _route_edge_cost(memory, wall_cost=1.5)
    try:
        distances = reachable_cells(
            obs.location,
            obs.direction,
            edge,
            max_cost=_BASE_ROUTE_BUDGET,
            turn_cost=1.0,
        )
    except Exception:
        return zeros

    best_score = 0.0
    best_base: tuple[int, int] | None = None
    best_targets: set[tuple[int, int]] = set()
    best_cost = 0.0
    best_hit_value = 0.0

    for base in live_bases:
        try:
            firing_positions = cells_in_blast(memory, base)
        except Exception:
            firing_positions = _simple_blast(base)
        reachable_targets = {pos for pos in firing_positions if pos in distances}
        if not reachable_targets:
            continue
        target_cost = min(distances[pos] for pos in reachable_targets)
        base_hp = float(getattr(memory, "enemy_base_health", {}).get(base, BASE_MAX_HEALTH))
        hits_left = max(1.0, np.ceil(base_hp / max(1.0, float(BOMB_ATTACK))))
        hit_value = float(BOMB_ATTACK) + float(BASE_DESTROY_BONUS) / float(hits_left)
        score = hit_value / (target_cost + 1.0)
        if score > best_score:
            best_score = score
            best_base = base
            best_targets = {pos for pos in reachable_targets if abs(distances[pos] - target_cost) < 1e-6}
            best_cost = float(target_cost)
            best_hit_value = hit_value

    if best_base is None or not best_targets:
        return zeros

    if _bomb_has_base_value_now(obs, memory):
        action: int | None = int(Action.PLACE_BOMB)
    else:
        try:
            planned = first_action_to(
                obs.location,
                obs.direction,
                best_targets,
                edge,
                max_cost=_BASE_ROUTE_BUDGET,
            )
        except Exception:
            planned = None
        action = _action_with_wall_break(obs, memory, planned)

    out = _one_hot_action(action)
    out.extend(
        (
            1.0,
            _clip01(best_cost / _BASE_ROUTE_BUDGET),
            *_relative_features(obs.location, best_base),
            _clip01(best_hit_value / (float(BOMB_ATTACK) + float(BASE_DESTROY_BONUS))),
        )
    )
    return out


def _collect_route_features(obs: ParsedObs, memory: Any | None) -> list[float]:
    zeros = [0.0] * 9
    if memory is None:
        return zeros
    targets = [
        pos for pos in getattr(memory, "collectible_cells", lambda: [])()
        if pos != obs.location
    ]
    if not targets:
        return zeros

    edge = _route_edge_cost(memory, wall_cost=4.0)
    try:
        distances = reachable_cells(
            obs.location,
            obs.direction,
            edge,
            max_cost=_COLLECT_ROUTE_BUDGET,
            turn_cost=1.0,
        )
    except Exception:
        return zeros

    best_score = 0.0
    best_target: tuple[int, int] | None = None
    best_cost = 0.0
    best_value = 0.0
    for cell in targets:
        cost = distances.get(cell)
        if cost is None:
            continue
        try:
            value = float(memory.tile_value(cell))
        except Exception:
            value = 0.0
        if value <= 0.0:
            continue
        score = value / (float(cost) + 1.0)
        if score > best_score:
            best_score = score
            best_target = cell
            best_cost = float(cost)
            best_value = value

    if best_target is None:
        return zeros

    try:
        planned = first_action_to(
            obs.location,
            obs.direction,
            {best_target},
            edge,
            max_cost=_COLLECT_ROUTE_BUDGET,
        )
    except Exception:
        planned = None
    action = _action_with_wall_break(obs, memory, planned)

    out = _one_hot_action(action)
    out.extend(
        (
            1.0,
            _clip01(best_cost / _COLLECT_ROUTE_BUDGET),
            _clip01(best_value / 5.0),
        )
    )
    return out


def _route_edge_cost(memory: Any, wall_cost: float):
    def edge(a: tuple[int, int], b: tuple[int, int]) -> float | None:
        if not _in_bounds(memory, b):
            return None
        try:
            if memory.passable(a, b):
                return 1.0
        except Exception:
            pass
        try:
            if memory.edge_is_destructible_wall(a, b):
                return float(wall_cost)
        except Exception:
            pass
        return None

    return edge


def _action_with_wall_break(
    obs: ParsedObs,
    memory: Any | None,
    action: Action | int | None,
) -> int | None:
    if action is None:
        return None
    action_int = int(action)
    if action_int not in (int(Action.FORWARD), int(Action.BACKWARD)):
        return action_int
    if memory is None:
        return action_int
    next_pos = next_pos_after(obs.location, obs.direction, action_int)
    try:
        breaks_wall = memory.edge_is_destructible_wall(obs.location, next_pos)
    except Exception:
        breaks_wall = False
    if not breaks_wall:
        return action_int
    sitting_bomb = getattr(memory, "bombs", {}).get(obs.location)
    if sitting_bomb is not None and bool(getattr(sitting_bomb, "ally", False)):
        return int(Action.STAY)
    raw_mask = action_mask_from_obs(obs)
    if raw_mask[int(Action.PLACE_BOMB)] > 0.5 and obs.team_bombs > 0:
        return int(Action.PLACE_BOMB)
    return action_int


def _bomb_has_base_value_now(obs: ParsedObs, memory: Any | None) -> bool:
    if memory is None or obs.team_bombs <= 0:
        return False
    raw_mask = action_mask_from_obs(obs)
    if raw_mask[int(Action.PLACE_BOMB)] <= 0.5:
        return False
    sitting_bomb = getattr(memory, "bombs", {}).get(obs.location)
    if sitting_bomb is not None and bool(getattr(sitting_bomb, "ally", False)):
        return False
    try:
        blast = cells_in_blast(memory, obs.location)
    except Exception:
        blast = _simple_blast(obs.location)
    for base in getattr(memory, "enemy_bases", set()):
        hp = getattr(memory, "enemy_base_health", {}).get(base, BASE_MAX_HEALTH)
        if hp > 0 and base in blast:
            return True
    return False


def _one_hot_action(action: int | None) -> list[float]:
    out = [0.0] * NUM_ACTIONS
    if action is None:
        return out
    if 0 <= int(action) < NUM_ACTIONS:
        out[int(action)] = 1.0
    return out


def _action_destination(
    loc: tuple[int, int],
    direction: int,
    action: int,
) -> tuple[int, int]:
    direction = int(direction) % 4
    if action == 0:
        dx, dy = DIR_VECTOR[Direction(direction)]
        return (loc[0] + dx, loc[1] + dy)
    if action == 1:
        dx, dy = DIR_VECTOR[Direction((direction + 2) % 4)]
        return (loc[0] + dx, loc[1] + dy)
    return loc


def _simple_blast(pos: tuple[int, int]) -> set[tuple[int, int]]:
    ox, oy = pos
    out = set()
    for dx in range(-BOMB_BLAST_RADIUS, BOMB_BLAST_RADIUS + 1):
        for dy in range(-BOMB_BLAST_RADIUS, BOMB_BLAST_RADIUS + 1):
            tx, ty = ox + dx, oy + dy
            if 0 <= tx < GRID_SIZE and 0 <= ty < GRID_SIZE:
                out.add((tx, ty))
    return out


def _in_bounds(memory: Any | None, pos: tuple[int, int]) -> bool:
    if memory is not None and hasattr(memory, "in_bounds"):
        try:
            return bool(memory.in_bounds(pos))
        except Exception:
            pass
    return 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE


def _coord(value: int) -> float:
    return _clip01(float(value) / _GRID_DENOM)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))
