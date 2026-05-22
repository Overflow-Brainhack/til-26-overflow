"""Opponent / teacher controllers that consume a raw env observation dict and
return an integer action — the same interface the live AE server uses.

  * HeuristicController — wraps the production EditedHeuristicPolicyV2 through an
    AEManager with an isolated MapMemory (mirrors auto_play's _build_managers).
    Serves as the BC teacher, the Stage-2 opponent, and the benchmark baseline.
  * NetController       — wraps a frozen RecurrentMaskableActorCritic with its own
    per-agent GRU hidden state; used for the Stage-3 self-play league.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import common  # noqa: F401  (path bootstrap)
from ae_manager import DEFAULT_CACHE_PATH, DEFAULT_POLICY_KWARGS, AEManager
from constants import Action
from edited_policy_v2 import EditedHeuristicPolicyV2
from map_memory import MapMemory

from common import obs_to_arrays


# ── heuristic ────────────────────────────────────────────────────────────────
def _load_cache_template():
    try:
        if DEFAULT_CACHE_PATH.exists():
            return MapMemory.load(DEFAULT_CACHE_PATH)
    except Exception:
        pass
    return None


_CACHE_TEMPLATE = _load_cache_template()


class HeuristicController:
    """The strong rule-based policy, isolated per agent so map memory doesn't leak."""

    name = "heuristic"

    def __init__(self, policy_kwargs: dict | None = None):
        self._kwargs = policy_kwargs or DEFAULT_POLICY_KWARGS
        self._mgr = self._build()

    def _build(self) -> AEManager:
        mem = MapMemory()
        if _CACHE_TEMPLATE is not None:
            mem.merge_static_from(_CACHE_TEMPLATE)
        return AEManager(policy=EditedHeuristicPolicyV2(**self._kwargs), memory=mem)

    def reset(self) -> None:
        self._mgr._memory.reset_round()
        if _CACHE_TEMPLATE is not None:
            self._mgr._memory.merge_static_from(_CACHE_TEMPLATE)

    def act(self, obs: dict) -> int:
        return int(self._mgr.ae(obs))


# ── frozen network ─────────────────────────────────────────────────────────────
class NetController:
    """A frozen learned policy used as a league opponent."""

    def __init__(self, model, device, name: str = "net", deterministic: bool = False):
        self.model = model
        self.device = device
        self.name = name
        self.deterministic = deterministic
        self._hidden = self.model.initial_hidden(1, device)

    def reset(self) -> None:
        self._hidden = self.model.initial_hidden(1, self.device)

    @torch.no_grad()
    def act(self, obs: dict) -> int:
        mask = np.asarray(obs.get("action_mask"), dtype=np.float32).flatten()
        if mask.size == len(Action) and mask.sum() == 0:
            return int(Action.STAY)
        vc, bv, sc, mk = obs_to_arrays(obs)
        t = lambda a: torch.as_tensor(a, device=self.device).unsqueeze(0)  # noqa: E731
        action, _, _, _, self._hidden = self.model.act(
            t(vc), t(bv), t(sc), t(mk), self._hidden, deterministic=self.deterministic
        )
        return int(action.item())


def league_checkpoints(league_dir: Path) -> list[Path]:
    return sorted(league_dir.glob("*.pt"))


# ── picklable opponent specs (for cross-process worker construction) ──────────
# A spec is a plain dict so it survives pickling to worker processes (lambdas /
# loaded models do not). build_controller turns a spec into a live controller,
# caching frozen nets per-process so each league checkpoint loads only once.
_NET_CACHE: dict[str, object] = {}


def heuristic_spec() -> dict:
    return {"kind": "heuristic"}


def net_spec(path, deterministic: bool = False) -> dict:
    return {"kind": "net", "path": str(path), "deterministic": deterministic}


def build_controller(spec: dict, device):
    kind = spec["kind"]
    if kind == "heuristic":
        return HeuristicController()
    if kind == "net":
        from model import load_checkpoint  # local import avoids a cycle at import time

        path = spec["path"]
        model = _NET_CACHE.get(path)
        if model is None:
            model = load_checkpoint(path, device, eval_mode=True)
            for p in model.parameters():
                p.requires_grad_(False)
            _NET_CACHE[path] = model
        return NetController(model, device, name=Path(path).stem,
                             deterministic=spec.get("deterministic", False))
    raise ValueError(f"unknown opponent spec kind: {kind!r}")
