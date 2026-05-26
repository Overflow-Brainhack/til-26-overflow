"""Parses the raw observation dict from the AE server into a structured form.

The server hands us JSON-decoded dicts where numpy arrays were converted to
nested lists by `tolist()` (see test/test_ae.py). We re-numpyify and expose
typed accessors.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from constants import (
    AGENT_COL_OFFSET,
    AGENT_ROW_OFFSET,
    BASE_VIEW_SIDE,
    BASE_VISION_RADIUS,
    Direction,
    NUM_CHANNELS,
    VIEWCONE_LENGTH,
    VIEWCONE_WIDTH,
)


@dataclass
class ParsedObs:
    agent_view: np.ndarray   # (VIEWCONE_LENGTH, VIEWCONE_WIDTH, NUM_CHANNELS) float32
    base_view: np.ndarray    # (BASE_VIEW_SIDE, BASE_VIEW_SIDE, NUM_CHANNELS) float32
    direction: int           # Direction value (0-3)
    location: tuple[int, int]
    base_location: tuple[int, int]
    health: float
    base_health: float
    frozen_ticks: int
    team_resources: float
    team_bombs: int
    step: int
    action_mask: np.ndarray  # (6,) int8


def parse_observation(raw: dict[str, Any]) -> ParsedObs:
    agent_view = _as_array(raw.get("agent_viewcone"), (VIEWCONE_LENGTH, VIEWCONE_WIDTH, NUM_CHANNELS))
    base_view = _as_array(raw.get("base_viewcone"), (BASE_VIEW_SIDE, BASE_VIEW_SIDE, NUM_CHANNELS))

    location = _as_xy(raw.get("location"))
    base_location = _as_xy(raw.get("base_location"))

    action_mask = np.asarray(raw.get("action_mask", [1] * 6), dtype=np.int8)
    if action_mask.shape != (6,):
        action_mask = np.ones(6, dtype=np.int8)

    return ParsedObs(
        agent_view=agent_view,
        base_view=base_view,
        direction=_as_int(raw.get("direction", 0)),
        location=location,
        base_location=base_location,
        health=_as_float(raw.get("health", 0.0)),
        base_health=_as_float(raw.get("base_health", 0.0)),
        frozen_ticks=_as_int(raw.get("frozen_ticks", 0)),
        team_resources=_as_float(raw.get("team_resources", 0.0)),
        team_bombs=_as_int(raw.get("team_bombs", 0)),
        step=_as_int(raw.get("step", 0)),
        action_mask=action_mask,
    )


def _as_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    arr = np.asarray(value).flatten()
    return float(arr[0]) if arr.size else 0.0


def _as_int(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    arr = np.asarray(value).flatten()
    return int(arr[0]) if arr.size else 0


def _as_array(value: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.zeros(expected_shape, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != expected_shape:
        out = np.zeros(expected_shape, dtype=np.float32)
        slices = tuple(slice(0, min(a, b)) for a, b in zip(arr.shape, expected_shape))
        out[slices] = arr[slices]
        return out
    return arr


def _as_xy(value: Any) -> tuple[int, int]:
    if value is None:
        return (0, 0)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (int(value[0]), int(value[1]))
    arr = np.asarray(value).flatten()
    return (int(arr[0]), int(arr[1])) if arr.size >= 2 else (0, 0)


def view_to_world(
    agent_pos: tuple[int, int],
    agent_dir: int,
    view_row: int,
    view_col: int,
) -> tuple[int, int]:
    """Map an agent_viewcone (row, col) cell to world (x, y).

    The agent occupies (AGENT_ROW_OFFSET, AGENT_COL_OFFSET) in the cone.
    Row increases ahead-of-agent (down in array, forward in world).
    Col increases to agent's right.
    """
    fwd = view_row - AGENT_ROW_OFFSET
    side = view_col - AGENT_COL_OFFSET

    fdx, fdy = _facing_vec(agent_dir)
    rdx, rdy = _right_vec(agent_dir)
    return (agent_pos[0] + fwd * fdx + side * rdx,
            agent_pos[1] + fwd * fdy + side * rdy)


def base_view_to_world(
    base_pos: tuple[int, int],
    view_row: int,
    view_col: int,
) -> tuple[int, int]:
    """Map a base_viewcone cell to world coords.

    The base view is axis-aligned (RIGHT direction by simulator convention),
    so row offset goes in +x and col offset goes in +y... no wait:
    `build_radius_view` calls the same kernel as direction=RIGHT with offsets
    (-R..R, -R..R), where world_rel[n] = view_to_world(origin, RIGHT, view).
    For RIGHT: forward axis is +x, right axis is +y. The (i, j) iteration
    maps (i, j) → view_coord (i - R, j - R), then to world via:
    fwd=(i-R), side=(j-R) → (+fwd_x, +side_x). For RIGHT, fwd=(1,0)
    side=(0,1), giving world delta (i-R, j-R).
    """
    return (base_pos[0] + view_row - BASE_VISION_RADIUS,
            base_pos[1] + view_col - BASE_VISION_RADIUS)


def _facing_vec(direction: int) -> tuple[int, int]:
    if direction == Direction.RIGHT:
        return (1, 0)
    if direction == Direction.DOWN:
        return (0, 1)
    if direction == Direction.LEFT:
        return (-1, 0)
    return (0, -1)  # UP


def _right_vec(direction: int) -> tuple[int, int]:
    """Vector pointing to agent's right (90° clockwise from facing)."""
    if direction == Direction.RIGHT:
        return (0, 1)
    if direction == Direction.DOWN:
        return (-1, 0)
    if direction == Direction.LEFT:
        return (0, -1)
    return (1, 0)  # UP
