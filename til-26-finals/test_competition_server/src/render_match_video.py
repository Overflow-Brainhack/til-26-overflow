"""Render match replay video from events.jsonl + match seed.

The env is deterministic given (seed, action sequence), so we don't need to
buffer rendered RGB frames in RAM during the live match. We replay them at
the end in a thread off the event loop.

All RNG draws inside the env happen at reset() (arena generation, agent
placement). env.step() is a pure function of (current state, actions).
See til_environment/dynamics.py and arena.py for the RNG audit.
"""

import json
import logging
from typing import Any, Iterable

import imageio
import numpy as np
from til_environment.bomberman_env import parallel_basic_env

logger = logging.getLogger("uvicorn.error")


def _read_step_actions(events_jsonl_path: str) -> list[dict[str, int]]:
    """Pull every 'step' event's `moves` dict from events.jsonl in order."""
    actions: list[dict[str, int]] = []
    with open(events_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "step":
                continue
            moves = rec.get("moves") or {}
            actions.append({k: int(v) for k, v in moves.items()})
    return actions


def render_match_video(
    events_jsonl_path: str,
    env_cfg: Any,
    match_seed: int,
    mp4_path: str,
    agent_ids: Iterable[str],
    default_action: int = 0,
    fps: int = 20,
) -> None:
    """Replay every step event and encode as mp4.

    A missing agent_id in a recorded `moves` dict is treated as STAY
    (matching the live loop's default-actions behavior).
    """
    actions_by_step = _read_step_actions(events_jsonl_path)
    if not actions_by_step:
        logger.info(f"no step events in {events_jsonl_path}; skipping mp4 render")
        return

    env = parallel_basic_env(cfg=env_cfg, env_wrappers=[])
    env.reset(seed=match_seed)
    frames: list[np.ndarray] = [env.render()]
    agent_ids = list(agent_ids)
    for step_actions in actions_by_step:
        full_actions = {a: step_actions.get(a, default_action) for a in agent_ids}
        env.step(full_actions)
        frames.append(env.render())

    imageio.mimsave(mp4_path, frames, fps=fps)  # ty: ignore[no-matching-overload]
    logger.info(f"wrote replay mp4: {mp4_path} ({len(frames)} frames)")
