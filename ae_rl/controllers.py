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
from berserker_policy import BerserkerPolicy
from constants import Action
from diverse_opponents import (
    IdlePolicy,
    KamikazePolicy,
    PatrollerPolicy,
    PureCollectorPolicy,
    RandomLegalPolicy,
    TacticalPolicy,
    TrapSetterPolicy,
)
from edited_policy_v2 import EditedHeuristicPolicyV2
from map_memory import MapMemory
from observation import parse_observation
from policy import HeuristicPolicy

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

    def __init__(
        self,
        model,
        device,
        name: str = "net",
        deterministic: bool = False,
        novice: bool = True,
    ):
        self.model = model
        self.device = device
        self.name = name
        self.deterministic = deterministic
        self.novice = novice
        self._hidden = self.model.initial_hidden(1, device)
        self._memory = self._fresh_memory()
        # Last forward diagnostics — populated by act(); used by
        # LayeredNetController to decide whether to fall back to the heuristic.
        self._last_value: float = 0.0
        self._last_entropy: float = 0.0

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
        action, _logp, value, entropy, self._hidden = self.model.act(
            t(vc),
            t(bv),
            t(sc),
            t(mk),
            t(smap),
            self._hidden,
            deterministic=self.deterministic,
        )
        self._last_value = float(value.item())
        self._last_entropy = float(entropy.item())
        return int(action.item())


class VanillaHeuristicController(HeuristicController):
    """The base ``HeuristicPolicy`` from ``ae/src/policy.py`` — same family
    as the strong heuristic but with simpler defaults and a different
    decision tree. Gives the RL exposure to a structurally similar opponent
    that nonetheless makes different micro-decisions."""

    name = "vanilla_heuristic"

    def _build(self) -> AEManager:
        mem = MapMemory()
        if self.use_cache and _CACHE_TEMPLATE is not None:
            mem.merge_static_from(_CACHE_TEMPLATE)
        return AEManager(policy=HeuristicPolicy(), memory=mem)


class BerserkerController(HeuristicController):
    """``BerserkerPolicy`` — rushes enemy bases, spams bombs, ignores incoming
    fire. Completely different objective from the strong heuristic; this is
    the most diverse training opponent we have."""

    name = "berserker"

    def _build(self) -> AEManager:
        mem = MapMemory()
        if self.use_cache and _CACHE_TEMPLATE is not None:
            mem.merge_static_from(_CACHE_TEMPLATE)
        return AEManager(policy=BerserkerPolicy(), memory=mem)


class PureCollectorController(HeuristicController):
    """``PureCollectorPolicy`` — only moves toward visible collectibles,
    never bombs, never attacks. Trains the RL to handle non-aggressive
    opponents (which is plausibly what the eval reference policy is)."""

    name = "pure_collector"

    def _build(self) -> AEManager:
        mem = MapMemory()
        if self.use_cache and _CACHE_TEMPLATE is not None:
            mem.merge_static_from(_CACHE_TEMPLATE)
        return AEManager(policy=PureCollectorPolicy(), memory=mem)


class RandomController(HeuristicController):
    """``RandomLegalPolicy`` — uniform over the legal action mask. Trains
    the RL not to assume opponent rationality."""

    name = "random"

    def _build(self) -> AEManager:
        mem = MapMemory()
        if self.use_cache and _CACHE_TEMPLATE is not None:
            mem.merge_static_from(_CACHE_TEMPLATE)
        return AEManager(policy=RandomLegalPolicy(), memory=mem)


class IdleController(HeuristicController):
    """``IdlePolicy`` — mostly STAY, occasionally turn. Approximates an
    effectively-empty opponent slot."""

    name = "idle"

    def _build(self) -> AEManager:
        mem = MapMemory()
        if self.use_cache and _CACHE_TEMPLATE is not None:
            mem.merge_static_from(_CACHE_TEMPLATE)
        return AEManager(policy=IdlePolicy(), memory=mem)


class TrapSetterController(HeuristicController):
    """``TrapSetterPolicy`` — wanders and drops bombs everywhere. Trains the RL
    against an environment where any cell may become hazardous."""

    name = "trap_setter"

    def _build(self) -> AEManager:
        mem = MapMemory()
        if self.use_cache and _CACHE_TEMPLATE is not None:
            mem.merge_static_from(_CACHE_TEMPLATE)
        return AEManager(policy=TrapSetterPolicy(), memory=mem)


class PatrollerController(HeuristicController):
    """``PatrollerPolicy`` — predictable FORWARD-until-blocked walk. Models a
    non-adversarial opponent that does not react to the learner's presence."""

    name = "patroller"

    def _build(self) -> AEManager:
        mem = MapMemory()
        if self.use_cache and _CACHE_TEMPLATE is not None:
            mem.merge_static_from(_CACHE_TEMPLATE)
        return AEManager(policy=PatrollerPolicy(), memory=mem)


class KamikazeController(HeuristicController):
    """``KamikazePolicy`` — bombs at own feet when low HP or adjacent to enemies.
    Models desperate end-game opponents that trade life for damage."""

    name = "kamikaze"

    def _build(self) -> AEManager:
        mem = MapMemory()
        if self.use_cache and _CACHE_TEMPLATE is not None:
            mem.merge_static_from(_CACHE_TEMPLATE)
        return AEManager(policy=KamikazePolicy(), memory=mem)


class TacticalController(HeuristicController):
    """``TacticalPolicy`` — 1-step lookahead with hand-tuned scoring. Provides
    a 'good but non-heuristic' opponent so the RL has a strong adversary
    that doesn't share EditedHeuristicPolicyV2's exploitable quirks."""

    name = "tactical"

    def _build(self) -> AEManager:
        mem = MapMemory()
        if self.use_cache and _CACHE_TEMPLATE is not None:
            mem.merge_static_from(_CACHE_TEMPLATE)
        return AEManager(policy=TacticalPolicy(), memory=mem)


class LayeredNetController(NetController):
    """NetController + the same dodge/loop-break guards as deploy-side
    ``LayeredRLPolicy``. Used by diagnose/benchmark when ``--layered`` is set.

    The network is still consulted (and its hidden state advanced) on every
    turn so the GRU stays in sync; the guards only swap the *emitted* action.
    """

    name = "rl_layered"

    def __init__(
        self,
        model,
        device,
        name: str = "rl_layered",
        deterministic: bool = False,
        novice: bool = True,
        *,
        dodge_override: bool = True,
        oscillation_break: bool = True,
        loop_window: int = 6,
        heuristic_fallback: bool = False,
        value_threshold: float | None = None,
        entropy_threshold_frac: float | None = None,
    ):
        super().__init__(model, device, name=name,
                         deterministic=deterministic, novice=novice)
        self.dodge_override = dodge_override
        self.oscillation_break = oscillation_break
        self.heuristic_fallback = heuristic_fallback
        self.value_threshold = value_threshold
        self.entropy_threshold_frac = entropy_threshold_frac
        from collections import deque as _deque
        self._action_history = _deque(maxlen=loop_window)
        # Heuristic instance used when fallback fires. Same MapMemory as the
        # network sees, so it gets the same world view.
        self._heuristic = None
        if self.heuristic_fallback:
            self._heuristic = EditedHeuristicPolicyV2()

    def reset(self) -> None:
        super().reset()
        self._action_history.clear()

    @torch.no_grad()
    def act(self, obs: dict) -> int:
        rl_action = super().act(obs)

        # Parse once for both guards; bail out cleanly if parsing fails.
        try:
            parsed = parse_observation(obs)
        except Exception:
            return rl_action
        loc = tuple(int(x) for x in parsed.location)
        mask = np.asarray(obs.get("action_mask"), dtype=np.float32).flatten()

        if parsed.frozen_ticks > 0:
            self._action_history.append((int(Action.STAY), loc))
            return int(Action.STAY)

        # Dodge override.
        if self.dodge_override:
            from constants import BOMB_TIMER
            from threat import (
                cells_safe_for_at_least,
                imminent_danger,
                project_danger,
            )
            try:
                timeline = project_danger(self._memory)
                blast_tick = imminent_danger(self._memory, loc, timeline)
            except Exception:
                timeline, blast_tick = None, None
            if blast_tick is not None and blast_tick <= BOMB_TIMER:
                dodge_action = self._dodge(parsed, timeline, mask)
                if dodge_action is not None:
                    self._action_history.append((int(dodge_action), loc))
                    return int(dodge_action)

        # Heuristic fallback: low value or high entropy means the RL is
        # confidently-wrong or uncertain. Replace with the heuristic, which is
        # bounded-bad on every state. Decision uses the diagnostics that
        # NetController.act() just populated.
        if self.heuristic_fallback and self._should_fall_back(parsed):
            try:
                heur_action = int(self._heuristic.choose(parsed, self._memory))
            except Exception:
                heur_action = rl_action
            if self._is_legal(heur_action, mask):
                self._action_history.append((heur_action, loc))
                return heur_action

        # Loop break.
        if self.oscillation_break and self._is_loop(rl_action, loc):
            alt = self._break_loop(rl_action, mask, loc)
            if alt is not None:
                self._action_history.append((int(alt), loc))
                return int(alt)

        self._action_history.append((int(rl_action), loc))
        return rl_action

    def _should_fall_back(self, parsed) -> bool:
        if self.value_threshold is not None and self._last_value < self.value_threshold:
            return True
        if self.entropy_threshold_frac is not None:
            n_legal = int(sum(1 for a in parsed.action_mask if a))
            if n_legal >= 2:
                import math
                max_ent = math.log(n_legal)
                if max_ent > 0 and (self._last_entropy / max_ent) > self.entropy_threshold_frac:
                    return True
        return False

    def _dodge(self, parsed, timeline, mask):
        from constants import BOMB_TIMER, DIR_VECTOR
        from pathfinding import temporal_first_action_to
        from threat import cells_safe_for_at_least, imminent_danger

        if timeline is None:
            return None

        def edge_cost(a, b):
            if not self._memory.in_bounds(b):
                return None
            if self._memory.passable(a, b):
                return 1.0
            return None  # no wall-breaking during evacuation

        loc = tuple(int(x) for x in parsed.location)
        try:
            safe = cells_safe_for_at_least(self._memory, BOMB_TIMER + 1, timeline)
        except Exception:
            safe = None

        if safe:
            try:
                action = temporal_first_action_to(
                    loc, parsed.direction, safe, edge_cost, timeline
                )
            except Exception:
                action = None
            if action is not None and self._is_legal(int(action), mask):
                return int(action)

        # panic fallback: legal neighbour that survives longest.
        best_action = int(Action.STAY)
        try:
            best_tick = imminent_danger(self._memory, loc, timeline) or 99
        except Exception:
            best_tick = 0
        for cand in (int(Action.FORWARD), int(Action.BACKWARD),
                     int(Action.LEFT), int(Action.RIGHT), int(Action.STAY)):
            if not self._is_legal(cand, mask):
                continue
            dest = loc
            if cand == int(Action.FORWARD) and 0 <= parsed.direction < 4:
                dx, dy = DIR_VECTOR[parsed.direction]
                dest = (loc[0] + dx, loc[1] + dy)
            elif cand == int(Action.BACKWARD) and 0 <= parsed.direction < 4:
                dx, dy = DIR_VECTOR[parsed.direction]
                dest = (loc[0] - dx, loc[1] - dy)
            try:
                tick = (imminent_danger(self._memory, dest, timeline)
                        if dest != loc else best_tick)
            except Exception:
                tick = None
            tick = tick if tick is not None else 99
            if tick > best_tick:
                best_tick = tick
                best_action = cand
        return best_action

    @staticmethod
    def _is_legal(action: int, mask) -> bool:
        return 0 <= action < len(mask) and mask[action] == 1

    def _is_loop(self, action: int, pos: tuple[int, int]) -> bool:
        entry = (int(action), pos)
        buf = list(self._action_history)
        n = len(buf)
        for period in (2, 3):
            needed = 2 * period - 1
            if n < needed:
                continue
            suffix = tuple(buf[n - (period - 1):]) + (entry,)
            prev = tuple(buf[n - (2 * period - 1): n - (period - 1)])
            if suffix == prev:
                return True
        return False

    def _break_loop(self, looping_action: int, mask, loc):
        for cand in (int(Action.LEFT), int(Action.RIGHT), int(Action.FORWARD),
                     int(Action.BACKWARD), int(Action.STAY)):
            if cand == looping_action:
                continue
            if not self._is_legal(cand, mask):
                continue
            if not self._is_loop(cand, loc):
                return cand
        for cand in (int(Action.LEFT), int(Action.RIGHT), int(Action.FORWARD),
                     int(Action.BACKWARD), int(Action.STAY)):
            if cand != looping_action and self._is_legal(cand, mask):
                return cand
        return None


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


def vanilla_heuristic_spec(use_cache: bool = True) -> dict:
    return {"kind": "vanilla_heuristic", "use_cache": use_cache}


def berserker_spec(use_cache: bool = True) -> dict:
    return {"kind": "berserker", "use_cache": use_cache}


def pure_collector_spec(use_cache: bool = True) -> dict:
    return {"kind": "pure_collector", "use_cache": use_cache}


def random_spec(use_cache: bool = True) -> dict:
    return {"kind": "random", "use_cache": use_cache}


def idle_spec(use_cache: bool = True) -> dict:
    return {"kind": "idle", "use_cache": use_cache}


def trap_setter_spec(use_cache: bool = True) -> dict:
    return {"kind": "trap_setter", "use_cache": use_cache}


def patroller_spec(use_cache: bool = True) -> dict:
    return {"kind": "patroller", "use_cache": use_cache}


def kamikaze_spec(use_cache: bool = True) -> dict:
    return {"kind": "kamikaze", "use_cache": use_cache}


def tactical_spec(use_cache: bool = True) -> dict:
    return {"kind": "tactical", "use_cache": use_cache}


def net_spec(path, deterministic: bool = False, novice: bool = True) -> dict:
    return {
        "kind": "net",
        "path": str(path),
        "deterministic": deterministic,
        "novice": novice,
    }


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
    if kind == "vanilla_heuristic":
        return VanillaHeuristicController(use_cache=spec.get("use_cache", True))
    if kind == "berserker":
        return BerserkerController(use_cache=spec.get("use_cache", True))
    if kind == "pure_collector":
        return PureCollectorController(use_cache=spec.get("use_cache", True))
    if kind == "random":
        return RandomController(use_cache=spec.get("use_cache", True))
    if kind == "idle":
        return IdleController(use_cache=spec.get("use_cache", True))
    if kind == "trap_setter":
        return TrapSetterController(use_cache=spec.get("use_cache", True))
    if kind == "patroller":
        return PatrollerController(use_cache=spec.get("use_cache", True))
    if kind == "kamikaze":
        return KamikazeController(use_cache=spec.get("use_cache", True))
    if kind == "tactical":
        return TacticalController(use_cache=spec.get("use_cache", True))
    if kind == "net":
        from model import load_checkpoint  # local import avoids a cycle at import time

        path = spec["path"]
        model = _NET_CACHE.get(path)
        if model is None:
            model = load_checkpoint(path, device, eval_mode=True)
            for p in model.parameters():
                p.requires_grad_(False)
            _NET_CACHE[path] = model
        return NetController(
            model,
            device,
            name=Path(path).stem,
            deterministic=spec.get("deterministic", False),
            novice=spec.get("novice", True),
        )
    raise ValueError(f"unknown opponent spec kind: {kind!r}")
