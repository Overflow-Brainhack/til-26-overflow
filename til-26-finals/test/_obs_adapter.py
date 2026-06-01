"""Convert frame-stacked inference obs to the raw dict format participant bots expect.

The inference runner slices per-agent obs as ``{k: v[i:i+1]}`` where each
value retains its batch dim of 1.  After ``frame_stack_v3`` (stack_dim=-1):

  Box (H, W, C)      → shape (1, H, W, C*S)   — current frame = [..., -C:]
  Box (2,)           → shape (1, 2*S)           — current = [..., -2:]
  Box (6,)           → shape (1, 6*S)           — current = [..., -6:]
  Box (1,)           → shape (1, S)             — current = [..., -1]
  Discrete           → Box shape (1, S) int64   — current = [..., -1]

Output is a dict matching what the participant AEManagers / WorldModel.update()
expect (single-frame numpy arrays, scalars, and lists — no batch dim).
"""

from typing import Any

import numpy as np

_NUM_VC = 25  # ViewChannel count — must match til_environment NUM_CHANNELS
_NUM_ACTIONS = 6


def _last(arr: np.ndarray, n: int = 1) -> np.ndarray:
    """Return the last n elements along the final axis (current frame)."""
    return arr[..., -n:]


def raw_obs_from_agent_obs(agent_obs: dict[str, Any]) -> dict[str, Any]:
    """Convert a single-agent, frame-stacked inference obs to raw env format.

    Parameters
    ----------
    agent_obs : dict
        Sliced per-agent obs from the inference runner; each value has shape
        (1, ...) with the batch dim retained.

    Returns
    -------
    dict
        Raw obs dict matching what participant bots pass to their
        AEManager.ae() / WorldModel.update() / RuleAgent.act().
    """
    # ---- viewcones --------------------------------------------------------
    vc_stacked = agent_obs["agent_viewcone"][0]  # (H, W, C*S)
    viewcone = vc_stacked[..., -_NUM_VC:].astype(np.float32)  # (H, W, 25)

    if "base_viewcone" in agent_obs:
        bvc_stacked = agent_obs["base_viewcone"][0]  # (Bh, Bw, C*S)
        base_viewcone = bvc_stacked[..., -_NUM_VC:].astype(np.float32)
    else:
        base_viewcone = np.zeros((7, 7, _NUM_VC), dtype=np.float32)

    # ---- location / base_location -----------------------------------------
    loc_stacked = agent_obs["location"][0]  # (2*S,)
    location = loc_stacked[-2:].tolist()

    if "base_location" in agent_obs:
        bloc_stacked = agent_obs["base_location"][0]  # (2*S,)
        base_location = bloc_stacked[-2:].tolist()
    else:
        base_location = [0, 0]

    # ---- scalar / Discrete fields -----------------------------------------
    def _scalar(key: str, default: int | float, dtype=int) -> Any:
        if key not in agent_obs:
            return dtype(default)
        v = np.asarray(agent_obs[key][0]).flat[-1]
        return dtype(v)

    direction = _scalar("direction", 0, int)
    step = _scalar("step", 0, int)
    frozen_ticks = _scalar("frozen_ticks", 0, int)
    team_bombs = _scalar("team_bombs", 3, int)

    # ---- Box (1,)-shaped scalars — participant bots expect list[float] ----
    def _vec1(key: str, default: float) -> list[float]:
        if key not in agent_obs:
            return [default]
        v = np.asarray(agent_obs[key][0]).flat[-1]
        return [float(v)]

    health = _vec1("health", 60.0)
    base_health = _vec1("base_health", 100.0)
    team_resources = _vec1("team_resources", 0.0)

    # ---- action_mask (Box (6,) → Box (6*S,)) ------------------------------
    if "action_mask" in agent_obs:
        mask_arr = np.asarray(agent_obs["action_mask"][0])  # (6*S,) or (6,)
        action_mask = [int(x) for x in mask_arr[-_NUM_ACTIONS:]]
    else:
        action_mask = [1] * _NUM_ACTIONS

    return {
        "agent_viewcone": viewcone,
        "base_viewcone": base_viewcone,
        "location": location,
        "base_location": base_location,
        "direction": direction,
        "step": step,
        "frozen_ticks": frozen_ticks,
        "team_bombs": team_bombs,
        "health": health,
        "base_health": base_health,
        "team_resources": team_resources,
        "action_mask": action_mask,
    }
