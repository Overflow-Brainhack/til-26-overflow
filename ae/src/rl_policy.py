"""Lightweight NumPy inference policy for exported AE RL models."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from constants import Action, NUM_ACTIONS
from map_memory import MapMemory
from observation import ParsedObs
from policy import Policy
from rl_features import FEATURE_SIZE, FEATURE_VERSION, extract_features, safe_action_mask_from_obs
from threat import cells_in_blast, expected_blast_hits_drift


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "ae_policy.npz"


class LearnedPolicy(Policy):
    """Run an exported PPO/BC policy, falling back when no checkpoint exists."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        fallback: Optional[Policy] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.fallback = fallback
        self._model: _NumpyActorCritic | None = None
        self._debug_mode = "learned"
        self._debug_target = None
        self._debug_pos = (0, 0)
        self.override_margin = float(os.getenv("AE_LEARNED_OVERRIDE_MARGIN", "0.25"))
        self.min_override_prob = float(os.getenv("AE_LEARNED_MIN_OVERRIDE_PROB", "0.45"))
        self.bomb_override_threshold = float(os.getenv("AE_LEARNED_BOMB_OVERRIDE_THRESHOLD", "0.35"))
        self._history: deque[tuple[tuple[int, int], int]] = deque(maxlen=8)
        self._load()

    @property
    def available(self) -> bool:
        return self._model is not None

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.step == 0:
            self._history.clear()
        self._debug_pos = obs.location
        self._debug_target = None

        if self._model is None:
            self._debug_mode = "learned:fallback"
            return self._choose_fallback(obs, memory)

        fallback_action: int | None = None
        if self.fallback is not None:
            try:
                fallback_action = int(self.fallback.choose(obs, memory))
            except Exception:
                fallback_action = None

        mask = safe_action_mask_from_obs(obs, memory)
        if mask.sum() <= 0:
            self._debug_mode = "learned:frozen"
            return fallback_action if fallback_action is not None else int(Action.STAY)

        features = extract_features(obs, memory)
        logits = self._model.policy_logits(features)
        action = _masked_argmax(logits, mask)
        if action is None:
            self._debug_mode = "learned:fallback"
            return fallback_action if fallback_action is not None else self._choose_fallback(obs, memory)

        bomb_gate = self._maybe_gate_bomb(action, fallback_action, obs, memory, mask)
        if bomb_gate is not None:
            self._debug_mode = "learned:bomb-gated"
            self._record(obs, int(bomb_gate))
            return int(bomb_gate)

        loop_gate = self._maybe_gate_loop(action, fallback_action, obs, mask)
        if loop_gate is not None:
            self._debug_mode = "learned:loop-gated"
            self._record(obs, int(loop_gate))
            return int(loop_gate)

        gated = self._maybe_gate_to_fallback(logits, mask, action, fallback_action)
        if gated is not None:
            self._debug_mode = "learned:gated"
            self._record(obs, int(gated))
            return int(gated)

        self._debug_mode = "learned"
        self._record(obs, int(action))
        return int(action)

    def _record(self, obs: ParsedObs, action: int) -> None:
        self._history.append((obs.location, int(action)))

    def _maybe_gate_bomb(
        self,
        action: int,
        fallback_action: int | None,
        obs: ParsedObs,
        memory: MapMemory,
        mask: np.ndarray,
    ) -> int | None:
        if int(action) != int(Action.PLACE_BOMB):
            return None
        if fallback_action == int(Action.PLACE_BOMB):
            return None
        if self._learned_bomb_has_direct_value(obs, memory):
            return None
        if fallback_action is not None and 0 <= fallback_action < NUM_ACTIONS and mask[fallback_action] > 0.5:
            return int(fallback_action)
        return self._choose_fallback(obs, memory)

    def _learned_bomb_has_direct_value(self, obs: ParsedObs, memory: MapMemory) -> bool:
        if obs.team_bombs <= 0 or obs.action_mask[int(Action.PLACE_BOMB)] != 1:
            return False
        try:
            blast = cells_in_blast(memory, obs.location)
        except Exception:
            return False
        if memory.enemy_bases & blast:
            return True
        if set(memory.enemy_agents) & blast:
            return True
        try:
            return expected_blast_hits_drift(memory, blast) >= self.bomb_override_threshold
        except Exception:
            return False

    def _maybe_gate_loop(
        self,
        action: int,
        fallback_action: int | None,
        obs: ParsedObs,
        mask: np.ndarray,
    ) -> int | None:
        if fallback_action is None or int(action) == int(fallback_action):
            return None
        if not (0 <= fallback_action < NUM_ACTIONS) or mask[fallback_action] <= 0.5:
            return None
        if len(self._history) < 5:
            return None
        recent_locs = [pos for pos, _ in self._history]
        recent_actions = [act for _, act in self._history]
        stuck_nearby = len(set(recent_locs)) <= 2
        turn_or_reverse_churn = sum(
            1 for act in recent_actions if act in (int(Action.LEFT), int(Action.BACKWARD))
        ) >= 4
        if stuck_nearby or turn_or_reverse_churn:
            return int(fallback_action)
        return None

    def _maybe_gate_to_fallback(
        self,
        logits: np.ndarray,
        mask: np.ndarray,
        action: int,
        fallback_action: int | None,
    ) -> int | None:
        if fallback_action is None or action == fallback_action:
            return None
        if not (0 <= fallback_action < NUM_ACTIONS) or mask[fallback_action] <= 0.5:
            return None
        probs = _masked_softmax(logits, mask)
        gap = float(logits[action] - logits[fallback_action])
        if probs[action] < self.min_override_prob or gap < self.override_margin:
            return int(fallback_action)
        return None

    def _load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            candidate = _NumpyActorCritic.load(self.model_path)
        except Exception:
            return
        if candidate.feature_version > FEATURE_VERSION:
            return
        if candidate.feature_size > FEATURE_SIZE:
            return
        self._model = candidate

    def _choose_fallback(self, obs: ParsedObs, memory: MapMemory) -> int:
        if self.fallback is not None:
            return int(self.fallback.choose(obs, memory))
        for action in (Action.STAY, Action.FORWARD, Action.LEFT, Action.RIGHT, Action.BACKWARD):
            if obs.action_mask[int(action)]:
                return int(action)
        return int(Action.STAY)


class _NumpyActorCritic:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.feature_version = int(np.asarray(arrays["feature_version"]).item())
        self.feature_size = int(np.asarray(arrays["feature_size"]).item())
        self.w0 = _array(arrays["w0"])
        self.b0 = _array(arrays["b0"])
        self.w1 = _array(arrays["w1"])
        self.b1 = _array(arrays["b1"])
        self.policy_w = _array(arrays["policy_w"])
        self.policy_b = _array(arrays["policy_b"])

    @classmethod
    def load(cls, path: Path) -> "_NumpyActorCritic":
        if path.suffix.lower() == ".pt":
            arrays = _load_torch_actor_critic(path)
        else:
            with np.load(path, allow_pickle=False) as data:
                arrays = {name: data[name] for name in data.files}
        required = {
            "feature_version",
            "feature_size",
            "w0",
            "b0",
            "w1",
            "b1",
            "policy_w",
            "policy_b",
        }
        missing = required - set(arrays)
        if missing:
            raise ValueError(f"missing arrays: {sorted(missing)}")
        return cls(arrays)

    def policy_logits(self, features: np.ndarray) -> np.ndarray:
        flat = np.asarray(features, dtype=np.float32).reshape(-1)
        if flat.size < self.feature_size:
            x = np.zeros(self.feature_size, dtype=np.float32)
            x[:flat.size] = flat
        elif flat.size > self.feature_size:
            x = flat[:self.feature_size]
        else:
            x = flat
        x = x.reshape(1, self.feature_size)
        h = np.tanh(x @ self.w0.T + self.b0)
        h = np.tanh(h @ self.w1.T + self.b1)
        logits = h @ self.policy_w.T + self.policy_b
        return logits.reshape(NUM_ACTIONS)


def _masked_argmax(logits: np.ndarray, mask: np.ndarray) -> int | None:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    if logits.size != NUM_ACTIONS or mask.size != NUM_ACTIONS:
        return None
    legal = mask > 0.5
    if not np.any(legal):
        return int(Action.STAY)
    masked = np.where(legal, logits, -1.0e9)
    return int(np.argmax(masked))


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    out = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if logits.size != NUM_ACTIONS or mask.size != NUM_ACTIONS:
        return out
    legal = mask > 0.5
    if not np.any(legal):
        return out
    masked = np.where(legal, logits, -1.0e9)
    masked = masked - np.max(masked[legal])
    exp = np.where(legal, np.exp(masked), 0.0)
    denom = float(exp.sum())
    if denom <= 0.0:
        return out
    return (exp / denom).astype(np.float32, copy=False)


def _array(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _load_torch_actor_critic(path: Path) -> dict[str, np.ndarray]:
    """Load the full-policy PPO/BC OrderedDict export used by local training."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required to load .pt learned AE policies") from exc

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload
    if isinstance(payload, dict):
        for key in ("model_state", "state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    required = {
        "fc0.weight",
        "fc0.bias",
        "fc1.weight",
        "fc1.bias",
        "policy_head.weight",
        "policy_head.bias",
    }
    missing = required - set(state)
    if missing:
        raise ValueError(f"missing torch policy tensors: {sorted(missing)}")

    w0 = _tensor_to_array(state["fc0.weight"])
    return {
        "feature_version": np.asarray(FEATURE_VERSION, dtype=np.int64),
        "feature_size": np.asarray(w0.shape[1], dtype=np.int64),
        "w0": w0,
        "b0": _tensor_to_array(state["fc0.bias"]),
        "w1": _tensor_to_array(state["fc1.weight"]),
        "b1": _tensor_to_array(state["fc1.bias"]),
        "policy_w": _tensor_to_array(state["policy_head.weight"]),
        "policy_b": _tensor_to_array(state["policy_head.bias"]),
    }


def _tensor_to_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)
