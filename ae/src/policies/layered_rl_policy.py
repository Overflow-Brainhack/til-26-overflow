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

  4. **Stagnation takeover.** The oscillation break only catches short
     period-2/3 cycles; a deterministic argmax policy can also wander a
     handful of cells for dozens of ticks (the "stuck in a corner" failure).
     Track position history per round and, when the agent has covered too few
     unique cells over a window — or is loitering in a *closed-off pocket*
     (every recent cell has a small passable neighbourhood, measured by a
     depth-limited flood fill over the current wall state) — hand control to
     ``EditedHeuristicPolicyV2`` for ``takeover_ticks`` ticks. The heuristic's
     goal-directed routing (collect, proactive base routing, exploration)
     pulls the agent back into productive play; a cooldown then keeps the RL
     in charge so the takeover can never dominate a round. Camping near the
     own base is exempt — that is deliberate defence, not stagnation. The
     openness test is what stands in for "hardcoding bad map locations": with
     the novice map cache loaded the closed-off pockets are known from tick 0,
     and on unknown maps the same test generalises from observed walls.

All guards are toggleable so you can A/B exactly how much lift each one
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
from .rl_policy import RLPolicy
from threat import (
    cells_safe_for_at_least,
    imminent_danger,
    project_danger,
)


_LOOP_PERIODS = (2, 3)
_LOOP_WINDOW_DEFAULT = 6

_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# What ae_manager ships. Kept here (next to the toggles they enable) so the
# auto_play harness can A/B the exact production configuration via --rl-guards.
PRODUCTION_GUARD_KWARGS: dict = dict(
    dodge_override=True,
    oscillation_break=True,
    stagnation_takeover=True,
)


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
        stagnation_takeover: bool = False,
        stagnation_window: int = 20,
        stagnation_unique_cells: int = 4,
        confined_window: int = 12,
        confined_unique_cells: int = 4,
        openness_depth: int = 3,
        openness_threshold: int = 12,
        takeover_ticks: int = 20,
        takeover_cooldown: int = 30,
        base_exempt_radius: int = 3,
        heuristic_kwargs: Optional[dict] = None,
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
            stagnation_takeover: master switch for guard 4. Triggers when
                the last ``stagnation_window`` positions span at most
                ``stagnation_unique_cells`` unique cells, or — the faster,
                confined variant — when the last ``confined_window``
                positions span at most ``confined_unique_cells`` cells *and*
                every one of those cells has openness (cells reachable within
                ``openness_depth`` moves; max ``2d²+2d+1`` = 25 at depth 3)
                of at most ``openness_threshold``. On trigger the heuristic
                drives for ``takeover_ticks`` ticks, then the trigger is
                suppressed for ``takeover_cooldown`` further ticks. Positions
                within ``base_exempt_radius`` (Manhattan) of the own base
                never trigger — sitting there is defence.
            heuristic_kwargs: constructor kwargs for the
                ``EditedHeuristicPolicyV2`` used by the fallback/takeover
                guards. Pass the production heuristic config
                (``ae_manager.DEFAULT_POLICY_KWARGS``) so guard behaviour
                matches the battle-tested heuristic, not its bare defaults.
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
        self.stagnation_takeover = stagnation_takeover
        self.stagnation_window = stagnation_window
        self.stagnation_unique_cells = stagnation_unique_cells
        self.confined_window = confined_window
        self.confined_unique_cells = confined_unique_cells
        self.openness_depth = openness_depth
        self.openness_threshold = openness_threshold
        self.takeover_ticks = takeover_ticks
        self.takeover_cooldown = takeover_cooldown
        self.base_exempt_radius = base_exempt_radius
        self._action_history: deque[tuple[int, tuple[int, int]]] = deque(
            maxlen=loop_window
        )
        self._pos_history: deque[tuple[int, int]] = deque(
            maxlen=max(stagnation_window, confined_window)
        )
        self._takeover_until: int = -1
        self._no_trigger_until: int = 0

        # Build the heuristic lazily — it pulls in pathfinding, threat, etc.
        # which are non-trivial imports. Skipped when no guard needs it.
        self._heuristic = None
        if self.heuristic_fallback or self.stagnation_takeover:
            # Import here so disabling both guards completely avoids the
            # dependency chain on edited_policy_v2.
            from .edited_policy_v2 import EditedHeuristicPolicyV2

            self._heuristic = EditedHeuristicPolicyV2(**(heuristic_kwargs or {}))

    def reset(self) -> None:
        super().reset()
        self._action_history.clear()
        self._pos_history.clear()
        self._takeover_until = -1
        self._no_trigger_until = 0

    # ── public API ──────────────────────────────────────────────────────────
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.step == 0:
            self.reset()

        if obs.frozen_ticks > 0:
            self._debug_mode = "frozen"
            self._debug_pos = obs.location
            return int(Action.STAY)

        # ── guard 4 bookkeeping: stagnation takeover ───────────────────────
        # Recorded before the dodge guard so evacuation ticks still extend the
        # history; the takeover itself is applied after the RL forward pass so
        # the GRU hidden state stays in sync with the trajectory.
        takeover_active = False
        if self.stagnation_takeover:
            self._pos_history.append(obs.location)
            if obs.step < self._takeover_until:
                takeover_active = True
            elif obs.step >= self._no_trigger_until and self._is_stagnant(
                obs, memory
            ):
                self._takeover_until = obs.step + self.takeover_ticks
                self._no_trigger_until = self._takeover_until + self.takeover_cooldown
                self._pos_history.clear()
                takeover_active = True

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

        # ── guard 4: stagnation takeover ───────────────────────────────────
        # The RL has been circling a few cells (or loitering in a closed-off
        # pocket) — let the heuristic's goal-directed routing drive for a
        # stretch. The dodge guard's early return above still pre-empts this,
        # and the heuristic's own choose() dodges first anyway.
        if takeover_active and self._heuristic is not None:
            heur_action = self._heuristic.choose(obs, memory)
            if self._is_legal(heur_action, obs):
                self._debug_mode = "stagnation"
                return self._record(int(heur_action), obs.location)

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

    # ── stagnation detection (guard 4) ──────────────────────────────────────
    def _is_stagnant(self, obs: ParsedObs, memory: MapMemory) -> bool:
        bx, by = obs.base_location
        x, y = obs.location
        if abs(x - bx) + abs(y - by) <= self.base_exempt_radius:
            return False

        hist = list(self._pos_history)
        n = len(hist)

        # Plain stagnation: too few unique cells over the long window.
        if n >= self.stagnation_window:
            recent = hist[-self.stagnation_window :]
            if len(set(recent)) <= self.stagnation_unique_cells:
                return True

        # Confined stagnation: a shorter dwell is already bad when the cells
        # involved are closed-off (dead-end, walled corner, narrow pocket).
        # Mean rather than max: the pocket's mouth cell legitimately sees out
        # into the open field and would defeat a per-cell test.
        if memory is not None and n >= self.confined_window:
            pocket = set(hist[-self.confined_window :])
            if len(pocket) <= self.confined_unique_cells:
                mean_openness = sum(
                    self._openness(memory, cell) for cell in pocket
                ) / len(pocket)
                if mean_openness <= self.openness_threshold:
                    return True
        return False

    def _openness(self, memory: MapMemory, cell: tuple[int, int]) -> int:
        """Cells reachable from `cell` within `openness_depth` moves, using the
        current wall knowledge (destructible walls count as blocking — escape
        has no time to bomb through). 25 on an open field at depth 3; corners,
        corridors and dead-ends score far lower."""
        seen = {cell}
        frontier = [cell]
        for _ in range(self.openness_depth):
            nxt: list[tuple[int, int]] = []
            for p in frontier:
                for dx, dy in _NEIGHBOURS:
                    q = (p[0] + dx, p[1] + dy)
                    if q in seen or not memory.in_bounds(q):
                        continue
                    if not memory.passable(p, q):
                        continue
                    seen.add(q)
                    nxt.append(q)
            frontier = nxt
        return len(seen)

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
