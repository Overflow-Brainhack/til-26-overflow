"""Privileged global-state encoder for the asymmetric (CTDE) critic.

The actor sees only its local, partial observation (viewcone + base view +
scalars + its own map memory). The **critic**, used only at training time,
gets a god's-eye view of the whole arena: every agent's position/health/frozen
state, every base's position/health, every bomb's imminence, and the
collectible layout — all ground truth, straight from ``env.dynamics``.

This is the standard Centralized-Training / Decentralized-Execution trick. In
this FFA game there are no teammates, so the privileged state is built relative
to the *evaluated* agent ("self" vs "others"): the same critic can value any
agent's situation by swapping which agent is marked self. The critic never runs
at deploy time and is dropped from the shipped checkpoint, so none of this
privileged information leaks into inference.

Outputs per (env, self_agent):
- ``grid``    : (N_GLOBAL_CHANNELS, GRID_SIZE, GRID_SIZE) float32
- ``scalars`` : (GLOBAL_SCALAR_DIM,) float32

Indexing is ``grid[c, x, y]`` to match the env's ``_state[x, y]`` convention.
The critic CNN is orientation-agnostic as long as this is internally consistent.
"""

from __future__ import annotations

import numpy as np

import common  # noqa: F401  (path bootstrap so the constants below import)
from common import (
    AGENT_MAX_HEALTH,
    BASE_MAX_HEALTH,
    GRID_SIZE,
    MAX_TEAM_RESOURCES,
    NUM_AGENTS,
    NUM_ITERS,
    TEAM_BOMBS_NORM,
)
from constants import BOMB_TIMER, REWARD_MISSION

# ── grid channel layout ───────────────────────────────────────────────────────
# 0  self agent position
# 1  self agent health (/AGENT_MAX_HEALTH, at the self cell)
# 2  self base position
# 3  self base health (/BASE_MAX_HEALTH, at the base cell)
# 4  other agents presence
# 5  other agents health (summed /AGENT_MAX_HEALTH)
# 6  other agents frozen
# 7  other bases presence
# 8  other bases health (/BASE_MAX_HEALTH)
# 9  bomb imminence (higher = detonates sooner; 0 where no bomb)
# 10 collectible value (/REWARD_MISSION)
N_GLOBAL_CHANNELS = 11

# Scalar summary (things that are global or awkward to localise on the grid).
# 0 self health                 5 num other agents alive /(NUM_AGENTS-1)
# 1 self frozen frac            6 num other bases alive  /(NUM_AGENTS-1)
# 2 self team_resources         7 sum other agents health (normalised)
# 3 self team_bombs             8 sum other bases health  (normalised)
# 4 step /NUM_ITERS             9 self base alive (1/0)
GLOBAL_SCALAR_DIM = 10

GLOBAL_GRID_SHAPE = (N_GLOBAL_CHANNELS, GRID_SIZE, GRID_SIZE)

_FREEZE_NORM = 10.0  # agent.freeze_duration default; only used to normalise


def zero_global_state() -> tuple[np.ndarray, np.ndarray]:
    """A zeroed (grid, scalars) pair — used for padding and when the privileged
    state is unavailable (e.g. an eval env we don't introspect)."""
    return (
        np.zeros(GLOBAL_GRID_SHAPE, dtype=np.float32),
        np.zeros(GLOBAL_SCALAR_DIM, dtype=np.float32),
    )


def _cell(pos) -> tuple[int, int] | None:
    """Clamp an (x, y) entity position to a valid in-bounds integer cell."""
    try:
        x = int(pos[0])
        y = int(pos[1])
    except (TypeError, IndexError, ValueError):
        return None
    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
        return x, y
    return None


def build_global_state(env, self_agent_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Build the privileged (grid, scalars) for ``self_agent_id`` from ``env``.

    Reads ``env.dynamics`` ground truth. Returns zeros if the env doesn't expose
    a dynamics/registry (defensive — keeps eval paths that reuse rollout code
    from crashing). Never raises on a malformed entity; it just skips it.
    """
    grid = np.zeros(GLOBAL_GRID_SHAPE, dtype=np.float32)
    scal = np.zeros(GLOBAL_SCALAR_DIM, dtype=np.float32)

    dyn = getattr(env, "dynamics", None)
    reg = getattr(dyn, "registry", None)
    if dyn is None or reg is None:
        return grid, scal

    try:
        self_agent = reg.get(self_agent_id)
    except Exception:
        self_agent = None
    self_team = getattr(self_agent, "team", None)

    # ── agents ────────────────────────────────────────────────────────────
    n_other_agents = 0
    sum_other_agent_hp = 0.0
    for ag in reg.agents():
        cell = _cell(getattr(ag, "position", None))
        if cell is None:
            continue
        x, y = cell
        hp = float(getattr(ag, "health", 0.0))
        frozen = float(getattr(ag, "frozen_ticks", 0)) > 0
        if getattr(ag, "entity_id", None) == self_agent_id:
            grid[0, x, y] = 1.0
            grid[1, x, y] = hp / AGENT_MAX_HEALTH
        else:
            grid[4, x, y] = 1.0
            grid[5, x, y] += hp / AGENT_MAX_HEALTH
            if frozen:
                grid[6, x, y] = 1.0
            n_other_agents += 1
            sum_other_agent_hp += hp

    # ── bases ─────────────────────────────────────────────────────────────
    n_other_bases = 0
    sum_other_base_hp = 0.0
    self_base_alive = 0.0
    for bs in reg.bases():
        cell = _cell(getattr(bs, "position", None))
        if cell is None:
            continue
        x, y = cell
        hp = float(getattr(bs, "health", 0.0))
        if getattr(bs, "team", None) == self_team and self_team is not None:
            grid[2, x, y] = 1.0
            grid[3, x, y] = hp / BASE_MAX_HEALTH
            self_base_alive = 1.0
        else:
            grid[7, x, y] = 1.0
            grid[8, x, y] = hp / BASE_MAX_HEALTH
            n_other_bases += 1
            sum_other_base_hp += hp

    # ── bombs (imminence: detonates sooner → larger) ──────────────────────
    try:
        bombs = reg.bombs()
    except Exception:
        bombs = []
    for bomb in bombs:
        cell = _cell(getattr(bomb, "position", None))
        if cell is None:
            continue
        x, y = cell
        timer = float(getattr(bomb, "timer", BOMB_TIMER))
        imminence = max(0.0, min(1.0, (BOMB_TIMER - timer + 1.0) / (BOMB_TIMER + 1.0)))
        grid[9, x, y] = max(grid[9, x, y], imminence)

    # ── collectibles (value-weighted) ─────────────────────────────────────
    for getter in ("missions", "resources", "recons"):
        fn = getattr(reg, getter, None)
        if fn is None:
            continue
        try:
            items = fn()
        except Exception:
            continue
        for it in items:
            cell = _cell(getattr(it, "position", None))
            if cell is None:
                continue
            x, y = cell
            val = float(getattr(it, "reward_value", 0.0))
            grid[10, x, y] = max(grid[10, x, y], val / REWARD_MISSION)

    # ── scalars ───────────────────────────────────────────────────────────
    team_res = 0.0
    team_bombs = 0.0
    if self_team is not None:
        try:
            team_res = float(dyn.team_resources.get(self_team, 0.0))
            team_bombs = float(dyn.team_bombs.get(self_team, 0))
        except Exception:
            pass
    step = float(getattr(env, "num_moves", getattr(dyn, "step_count", 0)) or 0)
    n_others_norm = max(1, NUM_AGENTS - 1)

    scal[0] = float(getattr(self_agent, "health", 0.0)) / AGENT_MAX_HEALTH
    scal[1] = float(getattr(self_agent, "frozen_ticks", 0)) / _FREEZE_NORM
    scal[2] = team_res / MAX_TEAM_RESOURCES
    scal[3] = team_bombs / TEAM_BOMBS_NORM
    scal[4] = step / NUM_ITERS
    scal[5] = n_other_agents / n_others_norm
    scal[6] = n_other_bases / n_others_norm
    scal[7] = sum_other_agent_hp / (n_others_norm * AGENT_MAX_HEALTH)
    scal[8] = sum_other_base_hp / (n_others_norm * BASE_MAX_HEALTH)
    scal[9] = self_base_alive

    return grid, scal
