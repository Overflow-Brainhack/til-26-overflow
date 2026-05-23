"""Shared bootstrap, paths, device, and observation-encoding helpers for the
AE recurrent-maskable-PPO training stack.

Importing this module makes ``ae/src`` importable as flat top-level modules
(matching the Docker layout that ``ae_manager``/``policy``/etc. expect) and the
``til_environment`` simulator package (installed editable via ``-e ./til-26-ae``).
"""

from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

# Silence pygame's pkg_resources deprecation warning at the earliest import
# point — every module (and every spawned rollout worker) imports common before
# the env pulls in pygame, so this keeps it from spamming the logs N times.
warnings.filterwarnings("ignore", message=r"pkg_resources is deprecated.*")

import numpy as np

# ── path bootstrap ───────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # …/ae_rl
REPO = HERE.parent                               # repo root
AE_SRC = REPO / "ae" / "src"                     # flat-module source dir
if str(AE_SRC) not in sys.path:
    sys.path.insert(0, str(AE_SRC))

CKPT_DIR = HERE / "checkpoints"
LEAGUE_DIR = CKPT_DIR / "league"
STAGE2_SNAPSHOT_DIR = CKPT_DIR / "stage2_snapshots"
STAGE3_SNAPSHOT_DIR = CKPT_DIR / "stage3_snapshots"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LEAGUE_DIR.mkdir(parents=True, exist_ok=True)
STAGE2_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
STAGE3_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Stage checkpoint filenames (latest snapshot per stage).
STAGE1_CKPT = CKPT_DIR / "stage1_bc.pt"
STAGE2_CKPT = CKPT_DIR / "stage2_ppo.pt"
STAGE3_CKPT = CKPT_DIR / "stage3_league.pt"
STAGE2_BEST_CKPT = CKPT_DIR / "stage2_ppo_best.pt"
STAGE3_BEST_CKPT = CKPT_DIR / "stage3_league_best.pt"

# Normalisation constants — imported from the (now importable) ae/src/constants.
from constants import (  # noqa: E402
    AGENT_MAX_HEALTH,
    BASE_MAX_HEALTH,
    BASE_VIEW_SIDE,
    FREEZE_TURNS,
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_CHANNELS,
    NUM_ITERS,
    VIEWCONE_LENGTH,
    VIEWCONE_WIDTH,
)

MAX_TEAM_RESOURCES = 100.0   # cfg.resources.max_team_resources
TEAM_BOMBS_NORM = 10.0       # soft scale (starting bombs 3, rarely large)

# Scalar feature vector layout (see build_scalars): one-hot dir(4) + 10 scalars.
SCALAR_DIM = 4 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1   # = 14
NUM_AGENTS = 6


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(prefer_cuda: bool = True):
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── robust scalar extraction (env hands np scalars / 1-elem arrays) ───────────
def _f(value, default: float = 0.0) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    arr = np.asarray(value).flatten()
    return float(arr[0]) if arr.size else default


def _i(value, default: int = 0) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    arr = np.asarray(value).flatten()
    return int(arr[0]) if arr.size else default


def build_scalars(obs: dict) -> np.ndarray:
    """Pack the non-spatial observation fields into a normalised float32 vector.

    Layout (SCALAR_DIM = 14):
        [0:4]   direction one-hot
        [4:6]   location (x, y) / GRID_SIZE
        [6:8]   base_location (x, y) / GRID_SIZE
        [8]     health / AGENT_MAX_HEALTH
        [9]     frozen_ticks / FREEZE_TURNS
        [10]    base_health / BASE_MAX_HEALTH
        [11]    team_resources / MAX_TEAM_RESOURCES
        [12]    team_bombs / TEAM_BOMBS_NORM
        [13]    step / NUM_ITERS
    """
    out = np.zeros(SCALAR_DIM, dtype=np.float32)
    d = _i(obs.get("direction", 0))
    if 0 <= d < 4:
        out[d] = 1.0
    loc = np.asarray(obs.get("location", (0, 0))).flatten()
    base = np.asarray(obs.get("base_location", (0, 0))).flatten()
    if loc.size >= 2:
        out[4] = loc[0] / GRID_SIZE
        out[5] = loc[1] / GRID_SIZE
    if base.size >= 2:
        out[6] = base[0] / GRID_SIZE
        out[7] = base[1] / GRID_SIZE
    out[8] = _f(obs.get("health", 0.0)) / AGENT_MAX_HEALTH
    out[9] = _i(obs.get("frozen_ticks", 0)) / max(1.0, FREEZE_TURNS)
    out[10] = _f(obs.get("base_health", 0.0)) / BASE_MAX_HEALTH
    out[11] = _f(obs.get("team_resources", 0.0)) / MAX_TEAM_RESOURCES
    out[12] = _i(obs.get("team_bombs", 0)) / TEAM_BOMBS_NORM
    out[13] = _i(obs.get("step", 0)) / NUM_ITERS
    return out


def _fix_view(arr, shape) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.shape == shape:
        return a
    out = np.zeros(shape, dtype=np.float32)
    sl = tuple(slice(0, min(x, y)) for x, y in zip(a.shape, shape))
    out[sl] = a[sl]
    return out


STATIC_MAP_CHANNELS = 6


def obs_to_arrays(obs: dict, memory=None):
    """Convert a raw env observation dict into model-ready numpy arrays.

    Returns
    -------
    viewcone   : (NUM_CHANNELS, VIEWCONE_LENGTH, VIEWCONE_WIDTH) float32  (C, H, W)
    baseview   : (NUM_CHANNELS, BASE_VIEW_SIDE, BASE_VIEW_SIDE)   float32  (C, H, W)
    scalars    : (SCALAR_DIM,) float32
    mask       : (NUM_ACTIONS,) float32  (1 = legal)
    static_map : (STATIC_MAP_CHANNELS, GRID_SIZE, GRID_SIZE) float32. Zeros
                 when ``memory`` is None.
    """
    vc = _fix_view(obs.get("agent_viewcone"), (VIEWCONE_LENGTH, VIEWCONE_WIDTH, NUM_CHANNELS))
    bv = _fix_view(obs.get("base_viewcone"), (BASE_VIEW_SIDE, BASE_VIEW_SIDE, NUM_CHANNELS))
    vc = np.ascontiguousarray(np.transpose(vc, (2, 0, 1)))   # (C, H, W)
    bv = np.ascontiguousarray(np.transpose(bv, (2, 0, 1)))
    scal = build_scalars(obs)
    mask = np.asarray(obs.get("action_mask", [1] * NUM_ACTIONS), dtype=np.float32).flatten()
    if mask.shape != (NUM_ACTIONS,):
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    if memory is not None:
        smap = memory.static_map_layer()
        if smap.shape != (STATIC_MAP_CHANNELS, GRID_SIZE, GRID_SIZE):
            smap = np.zeros((STATIC_MAP_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    else:
        smap = np.zeros((STATIC_MAP_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    return vc, bv, scal, mask, smap


# Shapes exported for the model constructor.
VIEW_SHAPE = (NUM_CHANNELS, VIEWCONE_LENGTH, VIEWCONE_WIDTH)
BASE_SHAPE = (NUM_CHANNELS, BASE_VIEW_SIDE, BASE_VIEW_SIDE)
STATIC_MAP_SHAPE = (STATIC_MAP_CHANNELS, GRID_SIZE, GRID_SIZE)
