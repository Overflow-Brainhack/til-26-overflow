"""RL policy with handcrafted safety overrides layered on top.

Wraps ``RLPolicy`` with three guards that the learned policy struggles with:

  1. **Dodge override.** Before consulting the network, project the current bomb
     timeline. If the agent is standing in (or about to be in) a real blast,
     compute an evacuation move with the same temporal Dijkstra the heuristic
     uses (``edited_policy._dodge``) and use it instead. The RL network never
     sees the danger, so this is purely additive safety — when there is no
     imminent blast the RL action is taken unchanged.

  2. **Oscillation break.** Track recent (action, position) pairs per round and
     short-circuit the 2-step shake (LEFT/RIGHT/LEFT/RIGHT in place) that the
     RL policy falls into when its value head is poorly calibrated locally.
     Mirrors ``EditedHeuristicPolicy._is_loop`` / ``_break_loop`` but applied
     after the RL action is sampled.

  3. **Heuristic fallback.** If the RL's predicted value drops below
     ``value_threshold`` *or* its action distribution's entropy exceeds
     ``entropy_threshold_frac`` (as a fraction of the mask-aware maximum
     entropy), fall through to ``EditedHeuristicPolicyV2``. Idea: the RL has
     a strong average policy but occasionally outputs a confidently-wrong
     action in states it has never seen; the heuristic is bounded-bad on every
     state. Replacing the worst RL decisions with heuristic decisions caps
     downside risk without changing peak behaviour.

All three guards are toggleable so you can A/B exactly how much lift each one
gives. GRU hidden state is *not* rewound when a guard fires — the network
still sees the next observation and updates its recurrent state normally.

Wire into AEManager by replacing ``RLPolicy`` with ``LayeredRLPolicy``; the
public interface is identical.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Optional

from constants import Action, BOMB_TIMER
from map_memory import MapMemory
from observation import ParsedObs
from pathfinding import temporal_first_action_to
from rl_policy import RLPolicy
from threat import (
    cells_safe_for_at_least,
    imminent_danger,
    project_danger,
)


_LOOP_PERIODS = (2, 3)
_LOOP_WINDOW_DEFAULT = 6


def _edge_cost_no_walls(memory: MapMemory):
    """Edge cost that refuses to traverse destructible walls (no time to bomb
    out during evacuation) and treats every legal step as 1 tick — matches the
    cost function ``EditedHeuristicPolicy._dodge`` builds with allow_walls=False.
    """

    def cost(a: tuple[int, int], b: tuple[int, int]) -> Optional[float]:
        if not memory.in_bounds(b):
            return None
        if memory.passable(a, b):
            return 1.0
        # Destructible walls block dodging — no time for a bomb fuse.
        return None

    return cost


class LayeredRLPolicy(RLPolicy):
    """RLPolicy + dodge override + oscillation break + heuristic fallback."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: str = "cpu",
        deterministic: bool = False,
        *,
        dodge_override: bool = False,
        oscillation_break: bool = False,
        loop_window: int = _LOOP_WINDOW_DEFAULT,
        heuristic_fallback: bool = False,
        value_threshold: float | None = -0.5,
        entropy_threshold_frac: float | None = None,
    ):
        """Construct the layered policy.

        Args:
            heuristic_fallback: master switch for the fallback path. When
                True the heuristic policy is instantiated and consulted
                whenever a threshold below fires. When False the heuristic
                is never built (no import overhead).
            value_threshold: trigger fallback when the RL's predicted value
                for the chosen action is *below* this value. ``None``
                disables the value trigger. Values are in normalised units
                (the network was trained with ``RunningReturnNorm``), so
                useful range is roughly [-1.0, 0.5]. Start with ``-0.5``
                and tune against your benchmark.
            entropy_threshold_frac: trigger fallback when the RL's action
                distribution entropy exceeds this fraction of the maximum
                possible entropy given the action mask. ``None`` disables
                the entropy trigger. Useful range [0.5, 0.95]. Start with
                ``0.85`` (policy is using >85% of its entropy budget — i.e.
                near-uniform → not confident in any action).
        """
        super().__init__(
            checkpoint_path=checkpoint_path,
            device=device,
            deterministic=deterministic,
        )
        self.dodge_override = dodge_override
        self.oscillation_break = oscillation_break
        self.heuristic_fallback = heuristic_fallback
        self.value_threshold = value_threshold
        self.entropy_threshold_frac = entropy_threshold_frac
        self._action_history: deque[tuple[int, tuple[int, int]]] = deque(
            maxlen=loop_window
        )

        # Build the heuristic lazily — it pulls in pathfinding, threat, etc.
        # which are non-trivial imports. Skipped when fallback is disabled.
        self._heuristic = None
        if self.heuristic_fallback:
            # Import here so disabling the fallback completely avoids the
            # dependency chain on edited_policy_v2.
            from edited_policy_v2 import EditedHeuristicPolicyV2

            self._heuristic = EditedHeuristicPolicyV2()

    def reset(self) -> None:
        super().reset()
        self._action_history.clear()

    # ── public API ──────────────────────────────────────────────────────────
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.step == 0:
            self.reset()

        if obs.frozen_ticks > 0:
            self._debug_mode = "frozen"
            self._debug_pos = obs.location
            return int(Action.STAY)

        # ── guard 1: dodge override ────────────────────────────────────────
        if self.dodge_override and memory is not None:
            try:
                timeline = project_danger(memory)
                blast_tick = imminent_danger(memory, obs.location, timeline)
            except Exception:
                blast_tick = None
                timeline = None
            if blast_tick is not None and blast_tick <= BOMB_TIMER:
                dodge_action = self._dodge(obs, memory, timeline)
                if dodge_action is not None and self._is_legal(dodge_action, obs):
                    self._debug_mode = "dodge"
                    self._debug_pos = obs.location
                    # Run the network so its hidden state stays in sync with the
                    # trajectory, but discard the action.
                    super().choose(obs, memory)
                    return self._record(dodge_action, obs.location)

        # ── normal RL choice (also populates _last_value / _last_entropy) ──
        action = super().choose(obs, memory)

        # ── guard 2: heuristic fallback ────────────────────────────────────
        # Fires when the value head says "this state is bad" OR when the
        # action distribution is near-uniform (policy not confident). In
        # either case the heuristic's bounded-bad behaviour is preferable to
        # whatever the RL was about to do.
        if self.heuristic_fallback and self._should_fall_back(obs):
            heur_action = self._heuristic.choose(obs, memory)
            if self._is_legal(heur_action, obs):
                self._debug_mode = "heur_fallback"
                return self._record(heur_action, obs.location)

        # ── guard 3: oscillation break ─────────────────────────────────────
        if self.oscillation_break and self._is_loop(action, obs.location):
            alt = self._break_loop(action, obs)
            if alt is not None:
                self._debug_mode = "loop_break"
                return self._record(alt, obs.location)

        return self._record(action, obs.location)

    def _should_fall_back(self, obs: ParsedObs) -> bool:
        """Evaluate the two fallback triggers against the RL's diagnostics."""
        if self.value_threshold is not None and self._last_value < self.value_threshold:
            return True
        if self.entropy_threshold_frac is not None:
            # Max entropy given the mask = log(# legal actions). Avoid log(1)=0
            # by clamping the denominator at 2 (single-legal-action states
            # never trigger fallback regardless).
            n_legal = int(sum(1 for a in obs.action_mask if a))
            if n_legal >= 2:
                max_ent = math.log(n_legal)
                frac = self._last_entropy / max_ent
                if frac > self.entropy_threshold_frac:
                    return True
        return False

    # ── internals ───────────────────────────────────────────────────────────
    def _dodge(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        timeline,
    ) -> Optional[int]:
        if timeline is None:
            return None
        try:
            safe = cells_safe_for_at_least(memory, BOMB_TIMER + 1, timeline)
        except Exception:
            return None
        if not safe:
            return self._panic_move(obs, memory, timeline)
        edge = _edge_cost_no_walls(memory)
        try:
            action = temporal_first_action_to(
                obs.location, obs.direction, safe, edge, timeline
            )
        except Exception:
            action = None
        if action is not None:
            return int(action)
        return self._panic_move(obs, memory, timeline)

    def _panic_move(self, obs: ParsedObs, memory: MapMemory, timeline) -> int:
        """Pick the legal neighbour that survives longest. Mirrors the
        heuristic's last-ditch escape when no fully-safe cell is reachable."""
        best_action = int(Action.STAY)
        try:
            best_tick = imminent_danger(memory, obs.location, timeline) or 99
        except Exception:
            best_tick = 0

        from constants import DIR_VECTOR  # local import to keep top tidy

        # Forward / backward / turns are all legal-action candidates; evaluate
        # only those that physically reach a different cell (turns leave the
        # agent in place — neutral for danger purposes).
        for action_id in (
            int(Action.FORWARD),
            int(Action.BACKWARD),
            int(Action.LEFT),
            int(Action.RIGHT),
            int(Action.STAY),
        ):
            if not self._is_legal(action_id, obs):
                continue
            dest = obs.location
            if action_id == int(Action.FORWARD) and 0 <= obs.direction < 4:
                dx, dy = DIR_VECTOR[obs.direction]
                dest = (obs.location[0] + dx, obs.location[1] + dy)
            elif action_id == int(Action.BACKWARD) and 0 <= obs.direction < 4:
                dx, dy = DIR_VECTOR[obs.direction]
                dest = (obs.location[0] - dx, obs.location[1] - dy)
            try:
                tick = (
                    imminent_danger(memory, dest, timeline)
                    if dest != obs.location
                    else best_tick
                )
            except Exception:
                tick = None
            tick = tick if tick is not None else 99
            if tick > best_tick:
                best_tick = tick
                best_action = action_id
        return best_action

    def _is_legal(self, action: int, obs: ParsedObs) -> bool:
        mask = obs.action_mask
        return 0 <= action < len(mask) and mask[action] == 1

    # ── loop detection (mirrors EditedHeuristicPolicy._is_loop) ────────────
    def _record(self, action: int, pos: tuple[int, int]) -> int:
        self._action_history.append((action, pos))
        return int(action)

    def _is_loop(self, action: int, pos: tuple[int, int]) -> bool:
        entry = (action, pos)
        buf = list(self._action_history)
        n = len(buf)
        for period in _LOOP_PERIODS:
            needed = 2 * period - 1
            if n < needed:
                continue
            suffix = tuple(buf[n - (period - 1) :]) + (entry,)
            prev = tuple(buf[n - (2 * period - 1) : n - (period - 1)])
            if suffix == prev:
                return True
        return False

    def _break_loop(self, looping_action: int, obs: ParsedObs) -> Optional[int]:
        """First legal action that doesn't itself close a cycle. Prefer turns
        (cheap heading change) over committing to a cell."""
        for action in (
            Action.LEFT,
            Action.RIGHT,
            Action.FORWARD,
            Action.BACKWARD,
            Action.STAY,
        ):
            cand = int(action)
            if cand == looping_action:
                continue
            if not self._is_legal(cand, obs):
                continue
            if not self._is_loop(cand, obs.location):
                return cand
        for action in (
            Action.LEFT,
            Action.RIGHT,
            Action.FORWARD,
            Action.BACKWARD,
            Action.STAY,
        ):
            cand = int(action)
            if cand != looping_action and self._is_legal(cand, obs):
                return cand
        return None
