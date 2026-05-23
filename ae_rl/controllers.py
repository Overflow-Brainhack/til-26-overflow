"""Opponent / teacher controllers that consume a raw env observation dict and
return an integer action — the same interface the live AE server uses.

  * HeuristicController — wraps the production EditedHeuristicPolicyV2 through an
    AEManager with an isolated MapMemory (mirrors auto_play's _build_managers).
    Serves as the BC teacher, the Stage-2 opponent, and the benchmark baseline.
  * NetController       — wraps a frozen RecurrentMaskableActorCritic with its own
    per-agent GRU hidden state; used for the Stage-3 self-play league.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

import common  # noqa: F401  (path bootstrap)
from ae_manager import DEFAULT_CACHE_PATH, DEFAULT_POLICY_KWARGS, AEManager
from constants import Action
from edited_policy_v2 import EditedHeuristicPolicyV2
from map_memory import MapMemory
from observation import parse_observation

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

    def __init__(self, policy_kwargs: dict | None = None, use_cache: bool = True):
        self._kwargs = policy_kwargs or DEFAULT_POLICY_KWARGS
        self.use_cache = use_cache
        self._mgr = self._build()

    def _build(self) -> AEManager:
        mem = MapMemory()
        if self.use_cache and _CACHE_TEMPLATE is not None:
            mem.merge_static_from(_CACHE_TEMPLATE)
        return AEManager(policy=EditedHeuristicPolicyV2(**self._kwargs), memory=mem)

    def reset(self) -> None:
        self._mgr._memory.reset_round()
        if self.use_cache and _CACHE_TEMPLATE is not None:
            self._mgr._memory.merge_static_from(_CACHE_TEMPLATE)

    def act(self, obs: dict) -> int:
        return int(self._mgr.ae(obs))


class StochasticHeuristicController(HeuristicController):
    """Heuristic opponent with per-round parameter jitter and light action noise."""

    name = "stochastic_heuristic"

    def __init__(
        self,
        policy_kwargs: dict | None = None,
        jitter: float = 0.35,
        action_noise: float = 0.03,
        use_cache: bool = True,
    ):
        self._base_kwargs = dict(DEFAULT_POLICY_KWARGS)
        if policy_kwargs:
            self._base_kwargs.update(policy_kwargs)
        self.use_cache = use_cache
        self.jitter = max(0.0, float(jitter))
        self.action_noise = max(0.0, min(1.0, float(action_noise)))
        self._kwargs = self._sample_kwargs()
        self._mgr = self._build()

    def _jitter_float(self, value: float, lo: float, hi: float) -> float:
        if self.jitter <= 0:
            return float(value)
        span = self.jitter
        return max(lo, min(hi, float(value) * random.uniform(1.0 - span, 1.0 + span)))

    def _jitter_int(self, value: int, lo: int, hi: int) -> int:
        return int(round(self._jitter_float(float(value), float(lo), float(hi))))

    def _sample_kwargs(self) -> dict:
        kw = dict(self._base_kwargs)

        for key, lo, hi in (
            ("predictive_bomb_threshold", 0.25, 0.95),
            ("wall_break_cost", 1.0, 12.0),
            ("bomb_tune_target", 0.15, 0.80),
            ("base_bomb_value", 1.0, 10.0),
            ("agent_bomb_value", 0.2, 4.0),
            ("bomb_reserve_threshold", 0.0, 5.0),
            ("wall_break_tile_threshold", 0.0, 6.0),
            ("base_route_weight", 10.0, 180.0),
            ("base_weight_min", 0.05, 1.5),
            ("base_weight_ramp_rate", 0.005, 0.12),
        ):
            if key in kw:
                kw[key] = self._jitter_float(kw[key], lo, hi)

        for key, lo, hi in (
            ("loop_window", 3, 12),
            ("base_weight_attack_cooldown", 5, 50),
        ):
            if key in kw:
                kw[key] = self._jitter_int(kw[key], lo, hi)

        # Occasionally remove or alter behavioural toggles, but keep dodge logic intact.
        for key, p_keep in (
            ("predictive_bomb", 0.85),
            ("wall_breaking", 0.90),
            ("smart_defend", 0.90),
            ("predictive_defend", 0.85),
            ("drift_aware_bomb", 0.85),
            ("auto_tune_bomb", 0.75),
            ("bomb_economy", 0.85),
            ("loop_detection", 0.90),
            ("proactive_base_routing", 0.90),
            ("adaptive_base_weight", 0.85),
        ):
            if key in kw:
                kw[key] = bool(kw[key]) and random.random() < p_keep

        return kw

    def reset(self) -> None:
        self._kwargs = self._sample_kwargs()
        self._mgr = self._build()

    def act(self, obs: dict) -> int:
        action = int(self._mgr.ae(obs))
        if self.action_noise <= 0 or random.random() >= self.action_noise:
            return action

        mask = np.asarray(obs.get("action_mask", []), dtype=np.float32).flatten()
        legal = [i for i, ok in enumerate(mask) if ok > 0]
        if not legal:
            return int(Action.STAY)
        return int(random.choice(legal))


# ── frozen network ─────────────────────────────────────────────────────────────
class NetController:
    """A frozen learned policy used as a league opponent."""

    def __init__(self, model, device, name: str = "net", deterministic: bool = False,
                 novice: bool = True):
        self.model = model
        self.device = device
        self.name = name
        self.deterministic = deterministic
        self.novice = novice
        self._hidden = self.model.initial_hidden(1, device)
        self._memory = self._fresh_memory()

    def _fresh_memory(self) -> MapMemory:
        mem = MapMemory()
        if self.novice and _CACHE_TEMPLATE is not None:
            mem.merge_static_from(_CACHE_TEMPLATE)
        return mem

    def reset(self) -> None:
        self._hidden = self.model.initial_hidden(1, self.device)
        self._memory = self._fresh_memory()

    @torch.no_grad()
    def act(self, obs: dict) -> int:
        mask = np.asarray(obs.get("action_mask"), dtype=np.float32).flatten()
        if mask.size == len(Action) and mask.sum() == 0:
            return int(Action.STAY)
        try:
            self._memory.update(parse_observation(obs))
        except Exception:
            pass
        vc, bv, sc, mk, smap = obs_to_arrays(obs, memory=self._memory)
        t = lambda a: torch.as_tensor(a, device=self.device).unsqueeze(0)  # noqa: E731
        action, _, _, _, self._hidden = self.model.act(
            t(vc), t(bv), t(sc), t(mk), t(smap), self._hidden, deterministic=self.deterministic
        )
        return int(action.item())


def league_checkpoints(league_dir: Path) -> list[Path]:
    return sorted(league_dir.glob("*.pt"))


# ── picklable opponent specs (for cross-process worker construction) ──────────
# A spec is a plain dict so it survives pickling to worker processes (lambdas /
# loaded models do not). build_controller turns a spec into a live controller,
# caching frozen nets per-process so each league checkpoint loads only once.
_NET_CACHE: dict[str, object] = {}


def heuristic_spec(use_cache: bool = True) -> dict:
    return {"kind": "heuristic", "use_cache": use_cache}


def stochastic_heuristic_spec(
    jitter: float = 0.35,
    action_noise: float = 0.03,
    use_cache: bool = True,
) -> dict:
    return {
        "kind": "stochastic_heuristic",
        "jitter": jitter,
        "action_noise": action_noise,
        "use_cache": use_cache,
    }


def net_spec(path, deterministic: bool = False, novice: bool = True) -> dict:
    return {"kind": "net", "path": str(path), "deterministic": deterministic,
            "novice": novice}


def build_controller(spec: dict, device):
    kind = spec["kind"]
    if kind == "heuristic":
        return HeuristicController(use_cache=spec.get("use_cache", True))
    if kind == "stochastic_heuristic":
        return StochasticHeuristicController(
            jitter=spec.get("jitter", 0.35),
            action_noise=spec.get("action_noise", 0.03),
            use_cache=spec.get("use_cache", True),
        )
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
                             deterministic=spec.get("deterministic", False),
                             novice=spec.get("novice", True))
    raise ValueError(f"unknown opponent spec kind: {kind!r}")
