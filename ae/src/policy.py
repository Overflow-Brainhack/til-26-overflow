"""Policy interface and the heuristic implementation.

The Policy abstract class lets us swap in a learned policy later without
touching the manager or server. The heuristic policy implements a priority
decision tree:
    1. Frozen → STAY
    2. Imminent enemy-blast danger → dodge to nearest safe cell
    3. Attack opportunity (enemy in our bomb's blast OR predicted to be) → PLACE_BOMB
    4. Defend (enemy near our base) → intercept
    5. Collect highest value-per-distance tile (optionally through walls);
       when proactive_base_routing is on, enemy base cells compete in the same
       scoring pass using base_route_weight as their synthetic tile value —
       tiles and bases are ranked together so neither is blindly skipped
    6. Explore frontier
    7. STAY (final fallback)

Toggles (`HeuristicPolicy(**kwargs)` / CLI flags in auto_play.py):
    predictive_bomb              — bomb based on expected hits, not just current overlap
    predictive_bomb_threshold    — minimum expected hits to bomb predictively (starting value for auto-tune)
    wall_breaking                — pathfinding may route through destructible walls
    wall_break_cost              — extra cost (≈ ticks lost) to traverse a destructible wall
    smart_defend                 — pre-position between enemy and base; expand defend radius when base health is low
    drift_aware_bomb             — use velocity-biased enemy position distribution (reduces overcounting)
    auto_tune_bomb               — online EMA that raises/lowers threshold based on observed hit rate
    bomb_tune_target             — target hit rate for auto-tuning (default 0.40)
    bomb_economy                 — unified value scoring: only bomb if score >= bomb_reserve_threshold
    base_bomb_value              — value of hitting an enemy base (in agent-hit units; default 5.0)
    agent_bomb_value             — value of a single definite agent hit (default 1.0)
    bomb_reserve_threshold       — minimum score to place a bomb under economy mode (default 1.0)
    wall_break_tile_threshold    — min tile value to justify a wall-break bomb (0.0 = always break)
    loop_detection               — detect and break 2- or 3-step (action, position) cycles (default ON)
    loop_window                  — number of past (action, pos) entries to retain for cycle detection
    proactive_base_routing       — include known enemy bases in collect scoring so the agent routes
                                   toward them when no better tile target exists (default OFF)
    base_route_weight            — synthetic tile value assigned to an enemy base cell for scoring
                                   purposes; comparable to REWARD_MISSION (5.0) / RESOURCE (2.0) /
                                   RECON (1.0) — higher values pull the agent toward bases even when
                                   collectibles are still available nearby (default 3.0)
    adaptive_base_weight         — when ON (requires proactive_base_routing), automatically adjusts
                                   the effective base-route weight based on observed enemy aggression.
                                   Starts at base_weight_min each round and ramps up toward
                                   base_route_weight at base_weight_ramp_rate per step.  When an
                                   enemy enters DEFEND_RADIUS of our base, or our base health drops,
                                   the weight resets to base_weight_min and a defensive cooldown
                                   begins; the agent stays near home objectives until the cooldown
                                   expires, then ramps again (default OFF)
    base_weight_min              — effective weight floor after a detected attack (default 0.5)
    base_weight_ramp_rate        — weight increase per step during the ramp phase (default 0.05;
                                   reaches base_route_weight from base_weight_min in ~50 steps)
    base_weight_attack_cooldown  — steps to hold the defensive posture (weight = base_weight_min)
                                   after the last detected attack before ramping resumes (default 20)
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import NamedTuple, Optional

from constants import (
    Action,
    BASE_MAX_HEALTH,
    BOMB_TIMER,
    DIR_VECTOR,
    Direction,
)
from map_memory import MapMemory
from observation import ParsedObs
from pathfinding import EdgeCost, first_action_to, next_pos_after, reachable_cells
from threat import (
    cells_in_blast,
    cells_safe_for_at_least,
    expected_blast_hits,
    expected_blast_hits_drift,
    imminent_danger,
    project_danger,
)


# Search-space tunables (independent of feature toggles).
DEFEND_RADIUS = 4                  # enemy within this many cells of base = threat
EXPLORE_BUDGET = 60.0              # max Dijkstra cost when looking for frontier

# How many steps from our base toward the enemy the intercept position sits.
INTERCEPT_STEPS = 2

# Predictive-bomb auto-tuning constants.
_TUNE_EMA_ALPHA = 0.75   # smoothing factor for the hit-rate EMA
_TUNE_MIN = 0.05         # floor: never go below this threshold
_TUNE_MAX = 0.95         # ceiling: never go above this threshold
_TUNE_WARMUP = 3         # minimum resolved bombs before threshold updates

# Loop-detection constants.
_LOOP_PERIODS = (2, 3)   # cycle lengths to detect
# Minimum history depth needed: 2*max_period - 1 = 5. Default window of 6
# gives one entry of slack while keeping memory negligible.
_LOOP_WINDOW_DEFAULT = 6


class _PendingBomb(NamedTuple):
    """Record of a predictive bomb we placed, pending hit/miss resolution."""
    placed_step: int
    blast_cells: frozenset[tuple[int, int]]
    expected_hits: float


def _intercept_cells(
    threats: set[tuple[int, int]],
    ally_base: tuple[int, int],
    memory: MapMemory,
) -> set[tuple[int, int]]:
    """Cells INTERCEPT_STEPS steps from base toward each threat.

    Navigating here places us on the enemy's path to the base rather than
    chasing them from behind, making it far easier to block and bomb them.
    Falls back to the threat cell itself when the enemy is already adjacent.
    """
    bx, by = ally_base
    out: set[tuple[int, int]] = set()
    for ex, ey in threats:
        dist = abs(ex - bx) + abs(ey - by)
        n = min(INTERCEPT_STEPS, dist)
        cx, cy = bx, by
        for _ in range(n):
            rdx, rdy = ex - cx, ey - cy
            if abs(rdx) >= abs(rdy):
                cx += 1 if rdx > 0 else -1
            else:
                cy += 1 if rdy > 0 else -1
        cell = (cx, cy)
        if memory.in_bounds(cell):
            out.add(cell)
    return out


class Policy(ABC):
    @abstractmethod
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int: ...


class HeuristicPolicy(Policy):
    """Rule-based agent. All feature toggles are constructor args so callers
    can run baseline-vs-feature comparisons without forking the policy."""

    def __init__(
        self,
        *,
        predictive_bomb: bool = True,
        predictive_bomb_threshold: float = 0.25,
        wall_breaking: bool = True,
        wall_break_cost: float = 5.0,
        smart_defend: bool = True,
        drift_aware_bomb: bool = True,
        auto_tune_bomb: bool = False,
        bomb_tune_target: float = 0.40,
        bomb_economy: bool = False,
        base_bomb_value: float = 5.0,
        agent_bomb_value: float = 1.0,
        bomb_reserve_threshold: float = 1.0,
        wall_break_tile_threshold: float = 0.0,
        loop_detection: bool = True,
        loop_window: int = _LOOP_WINDOW_DEFAULT,
        proactive_base_routing: bool = False,
        base_route_weight: float = 3.0,
        adaptive_base_weight: bool = False,
        base_weight_min: float = 0.5,
        base_weight_ramp_rate: float = 0.05,
        base_weight_attack_cooldown: int = 20,
    ) -> None:
        self.predictive_bomb = predictive_bomb
        self.predictive_bomb_threshold = predictive_bomb_threshold
        self.wall_breaking = wall_breaking
        self.wall_break_cost = wall_break_cost
        self.smart_defend = smart_defend
        self.drift_aware_bomb = drift_aware_bomb
        self.auto_tune_bomb = auto_tune_bomb
        self.bomb_tune_target = bomb_tune_target
        self.bomb_economy = bomb_economy
        self.base_bomb_value = base_bomb_value
        self.agent_bomb_value = agent_bomb_value
        self.bomb_reserve_threshold = bomb_reserve_threshold
        self.wall_break_tile_threshold = wall_break_tile_threshold
        self.loop_detection = loop_detection
        self.loop_window = loop_window
        self.proactive_base_routing = proactive_base_routing
        self.base_route_weight = base_route_weight
        self.adaptive_base_weight = adaptive_base_weight
        self.base_weight_min = base_weight_min
        self.base_weight_ramp_rate = base_weight_ramp_rate
        self.base_weight_attack_cooldown = base_weight_attack_cooldown

        # Adaptive base-weight state — reset each round (step == 0).
        self._adaptive_weight: float = base_weight_min
        self._attack_cooldown: int = 0
        self._prev_base_health: Optional[float] = None

        # Mutable auto-tune state — persists across rounds (policy is not re-created on /reset).
        self._tuned_threshold: float = predictive_bomb_threshold
        self._hit_ema: float = bomb_tune_target   # warm-start at the target hit rate
        self._ema_n: int = 0
        self._pending_bombs: list[_PendingBomb] = []

        # Rolling history of (action, position) pairs for loop detection.
        # Both components must match for a cycle to be confirmed — same action at
        # a different position (e.g. two FORWARD moves in a corridor) is not a loop.
        self._action_history: deque[tuple[int, tuple[int, int]]] = deque(maxlen=loop_window)

    # ── main entrypoint ─────────────────────────────────────────────────────
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        self._resolve_pending_bombs(memory)
        self._update_adaptive_weight(obs, memory)

        # Frozen ticks are forced by the engine — nothing to decide.
        if obs.frozen_ticks > 0:
            return self._record_action(int(Action.STAY), obs.location)

        timeline = project_danger(memory)
        danger_now: set[tuple[int, int]] = set()
        for tick, cells in timeline.items():
            if tick <= 1:
                danger_now.update(cells)

        # Dodge is safety-critical: skip loop detection so we never get stuck
        # in a blast zone while trying to break a navigation cycle.
        my_blast_tick = imminent_danger(memory, obs.location)
        if my_blast_tick is not None and my_blast_tick <= BOMB_TIMER:
            chosen = self._dodge(obs, memory, timeline)
            if chosen is not None:
                return self._record_action(self._mask_check(chosen, obs), obs.location)

        attack = self._try_attack(obs, memory)
        if attack is not None:
            return self._finalize(self._mask_check(attack, obs), obs, memory)

        defend = self._try_defend(obs, memory, danger_now)
        if defend is not None:
            return self._finalize(self._mask_check(defend, obs), obs, memory)

        collect = self._try_collect(obs, memory, danger_now)
        if collect is not None:
            return self._finalize(self._mask_check(collect, obs), obs, memory)

        explore = self._try_explore(obs, memory, danger_now)
        if explore is not None:
            return self._finalize(self._mask_check(explore, obs), obs, memory)

        return self._finalize(int(Action.STAY), obs, memory)

    # ── loop detection ──────────────────────────────────────────────────────

    def _record_action(self, action: int, pos: tuple[int, int]) -> int:
        """Append (action, pos) to the rolling history and return action."""
        if self.loop_detection:
            self._action_history.append((action, pos))
        return action

    def _is_loop(self, action: int, pos: tuple[int, int]) -> bool:
        """Return True if taking (action, pos) now would complete a repeating cycle.

        Checks whether the proposed entry, appended to the current history,
        would form a tail that exactly repeats the preceding block of equal
        length.  Both the action *and* the position must match — identical
        actions at different coordinates (e.g. two consecutive FORWARD moves
        along a corridor) are not considered a loop.

        For a period-P check (P ∈ {2, 3}):
            proposed suffix  = history[-(P-1):] + [(action, pos)]   # length P
            preceding block  = history[-(2P-1):-(P-1)]              # length P
        A match means the agent has already executed this exact sequence once
        before and is about to start it again.
        """
        entry = (action, pos)
        buf = list(self._action_history)  # deque doesn't support slicing
        n = len(buf)
        for period in _LOOP_PERIODS:
            needed = 2 * period - 1   # minimum history length for this period
            if n < needed:
                continue
            suffix = tuple(buf[n - (period - 1):]) + (entry,)
            prev   = tuple(buf[n - (2 * period - 1): n - (period - 1)])
            if suffix == prev:
                return True
        return False

    def _break_loop(self, obs: ParsedObs, memory: MapMemory, looping_action: int) -> int:
        """Return a legal, non-looping action to escape the detected cycle.

        Priority: turns first (cheap, changes heading without committing to a
        cell), then linear motion, then STAY.  Skips the detected looping
        action and any action that would itself complete a cycle.
        """
        for action in (Action.LEFT, Action.RIGHT, Action.FORWARD, Action.BACKWARD, Action.STAY):
            candidate = int(action)
            if candidate == looping_action:
                continue
            if obs.action_mask[candidate] != 1:
                continue
            if not self._is_loop(candidate, obs.location):
                return candidate
        # All alternatives are either masked or themselves looping — return any
        # masked-legal action as a last resort.
        for action in (Action.LEFT, Action.RIGHT, Action.FORWARD, Action.BACKWARD, Action.STAY):
            if obs.action_mask[int(action)] == 1:
                return int(action)
        return int(Action.STAY)

    def _finalize(self, action: int, obs: ParsedObs, memory: MapMemory) -> int:
        """Apply loop detection and record the action before returning it."""
        if self.loop_detection and self._is_loop(action, obs.location):
            action = self._break_loop(obs, memory, action)
        return self._record_action(action, obs.location)

    # ── edge cost builder (incorporates wall-breaking flag) ────────────────
    def _edge_cost(
        self,
        memory: MapMemory,
        *,
        danger_avoid: Optional[set[tuple[int, int]]] = None,
        allow_walls: Optional[bool] = None,
    ) -> EdgeCost:
        """Build an EdgeCost reflecting current toggles + per-call overrides.

        allow_walls defaults to self.wall_breaking. Pass False explicitly when
        wall-breaking would be unsafe (e.g. dodging — no time to wait for a
        bomb to clear a wall).
        """
        if allow_walls is None:
            allow_walls = self.wall_breaking
        wall_cost = self.wall_break_cost

        def cost(a: tuple[int, int], b: tuple[int, int]) -> Optional[float]:
            if not memory.in_bounds(b):
                return None
            if memory.passable(a, b):
                if danger_avoid is not None and b in danger_avoid:
                    return None
                return 1.0
            if allow_walls and memory.edge_is_destructible_wall(a, b):
                return wall_cost
            return None

        return cost

    # ── sub-strategies ──────────────────────────────────────────────────────

    def _dodge(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        timeline: dict[int, set[tuple[int, int]]],
    ) -> Optional[int]:
        immediate = timeline.get(0, set()) | timeline.get(1, set())

        # Dodging never breaks walls — no time for a bomb fuse during evacuation.
        edge = self._edge_cost(memory, allow_walls=False)

        def dodge_cost(a: tuple[int, int], b: tuple[int, int]) -> Optional[float]:
            base = edge(a, b)
            if base is None:
                return None
            if b in immediate:
                return None
            return base

        safe = cells_safe_for_at_least(memory, BOMB_TIMER + 1)
        if not safe:
            return self._panic_move(obs, memory, timeline)

        action = first_action_to(obs.location, obs.direction, safe, dodge_cost)
        if action is not None:
            return int(action)
        return self._panic_move(obs, memory, timeline)

    def _panic_move(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        timeline: dict[int, set[tuple[int, int]]],
    ) -> int:
        best_action = int(Action.STAY)
        best_tick = imminent_danger(memory, obs.location) or 99

        for action, dest in self._immediate_neighbors(obs, memory):
            tick = imminent_danger(memory, dest) if dest != obs.location else best_tick
            tick = tick if tick is not None else 99
            if tick > best_tick:
                best_tick = tick
                best_action = int(action)
        return best_action

    def _bomb_opportunity_score(self, memory: MapMemory, blast: set[tuple[int, int]]) -> float:
        """Compute unified value score for placing a bomb at the current position.

        Score = base_hits * base_bomb_value + agent_hits * agent_bomb_value
                + (expected_hits * agent_bomb_value if predictive_bomb is on).
        """
        base_hits = sum(1 for p in memory.enemy_bases if p in blast)
        agent_hits = sum(1 for p in memory.enemy_agents if p in blast)
        score = base_hits * self.base_bomb_value + agent_hits * self.agent_bomb_value
        if self.predictive_bomb:
            score += self._expected_hits(memory, blast) * self.agent_bomb_value
        return score

    def _try_attack(self, obs: ParsedObs, memory: MapMemory) -> Optional[int]:
        if obs.action_mask[Action.PLACE_BOMB] != 1:
            return None
        if obs.team_bombs <= 0:
            return None
        # Don't double-bomb the same cell — friendly-fire is off so this isn't
        # unsafe, just wasteful.
        sitting_bomb = memory.bombs.get(obs.location)
        if sitting_bomb is not None and sitting_bomb.ally:
            return None

        blast = cells_in_blast(memory, obs.location)

        # Economy mode: score the opportunity and only bomb if score is sufficient.
        if self.bomb_economy:
            score = self._bomb_opportunity_score(memory, blast)
            if score < self.bomb_reserve_threshold:
                return None
            if self.auto_tune_bomb:
                expected = self._expected_hits(memory, blast)
                self._pending_bombs.append(_PendingBomb(
                    placed_step=obs.step,
                    blast_cells=frozenset(blast),
                    expected_hits=expected,
                ))
            return int(Action.PLACE_BOMB)

        # Definite hits: enemies / enemy bases currently in blast.
        definite = 0.0
        for p in memory.enemy_agents:
            if p in blast:
                definite += 1.0
        for p in memory.enemy_bases:
            if p in blast:
                definite += 2.0  # bases score more (50 pts vs ~20 damage on agent)
        if definite >= 1.0:
            return int(Action.PLACE_BOMB)

        # Predictive: would the bomb plausibly hit a moving enemy by detonation?
        if self.predictive_bomb:
            expected = self._expected_hits(memory, blast)
            if expected >= self._effective_threshold():
                if self.auto_tune_bomb:
                    self._pending_bombs.append(_PendingBomb(
                        placed_step=obs.step,
                        blast_cells=frozenset(blast),
                        expected_hits=expected,
                    ))
                return int(Action.PLACE_BOMB)

        return None

    def _try_defend(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        if memory.ally_base is None or not memory.enemy_agents:
            return None
        # Base is permanently destroyed for this episode — defending it gains nothing.
        if obs.base_health <= 0:
            return None
        bx, by = memory.ally_base

        # Expand the defend radius when base health is low so we intercept
        # enemies earlier — at full health DEFEND_RADIUS=4; at zero health +4.
        if self.smart_defend:
            health_frac = min(1.0, obs.base_health / BASE_MAX_HEALTH)
            effective_radius = DEFEND_RADIUS + int((1.0 - health_frac) * 4)
        else:
            effective_radius = DEFEND_RADIUS

        threats = {
            p for p in memory.enemy_agents
            if abs(p[0] - bx) + abs(p[1] - by) <= effective_radius
        }
        if not threats:
            return None

        edge = self._edge_cost(memory, danger_avoid=danger_now)

        # Pre-position between the enemy and our base rather than chasing the
        # enemy directly. This puts us on their inbound path so _try_attack can
        # bomb them on the next tick once they enter our blast radius.
        if self.smart_defend:
            targets = _intercept_cells(threats, memory.ally_base, memory)
            if not targets:
                targets = threats
        else:
            targets = threats

        action = first_action_to(obs.location, obs.direction, targets, edge)
        if action is None:
            return None
        return self._maybe_wall_break(obs, memory, action)

    def _try_collect(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        candidates = memory.collectible_cells()

        # Proactive base routing: include known enemy base cells as synthetic
        # targets. They compete with real tiles using base_route_weight as their
        # effective value, so a nearby high-value tile still beats a distant base
        # while an uncontested base close by (or after tiles are exhausted) wins.
        # Attack/defend fire before this method, so an enemy base we can already
        # bomb is handled by _try_attack, not here.
        base_candidates: list[tuple[int, int]] = []
        if self.proactive_base_routing:
            base_candidates = [
                p for p in memory.enemy_bases
                if p != obs.location
            ]

        if not candidates and not base_candidates:
            return None

        edge = self._edge_cost(memory, danger_avoid=danger_now)
        distances = reachable_cells(
            obs.location, obs.direction, edge, max_cost=EXPLORE_BUDGET
        )

        best_score = 0.0
        best_cell: Optional[tuple[int, int]] = None
        for cell in candidates:
            if cell not in distances:
                continue
            value = memory.tile_value(cell)
            if value <= 0:
                continue
            score = value / (distances[cell] + 1.0)
            if score > best_score:
                best_score = score
                best_cell = cell

        for cell in base_candidates:
            if cell not in distances:
                continue
            score = self._effective_base_weight() / (distances[cell] + 1.0)
            if score > best_score:
                best_score = score
                best_cell = cell

        if best_cell is None:
            return None

        action = first_action_to(obs.location, obs.direction, {best_cell}, edge)
        if action is None:
            return None
        return self._maybe_wall_break(obs, memory, action)

    def _try_explore(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        edge = self._edge_cost(memory, danger_avoid=danger_now)

        frontier: set[tuple[int, int]] = set()
        for cell in memory.last_seen_step:
            x, y = cell
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nbr = (x + dx, y + dy)
                if memory.in_bounds(nbr) and nbr not in memory.last_seen_step:
                    frontier.add(cell)
                    break
        if not frontier:
            return None

        action = first_action_to(obs.location, obs.direction, frontier, edge)
        if action is None:
            return None
        return self._maybe_wall_break(obs, memory, action)

    # ── auto-tune helpers ────────────────────────────────────────────────────

    def _effective_threshold(self) -> float:
        return self._tuned_threshold if self.auto_tune_bomb else self.predictive_bomb_threshold

    def _expected_hits(self, memory: MapMemory, blast: set[tuple[int, int]]) -> float:
        if self.drift_aware_bomb:
            return expected_blast_hits_drift(memory, blast, BOMB_TIMER)
        return expected_blast_hits(memory, blast, BOMB_TIMER)

    def _resolve_pending_bombs(self, memory: MapMemory) -> None:
        """Resolve predictive bombs whose fuse has elapsed and update the EMA.

        A bomb is considered a hit if any enemy_agents entry falls within its
        blast cells at or after the step it was placed. This is an approximate
        proxy — enemies that were hit and then moved away before we observed
        them count as misses, and enemies that wandered in after detonation
        count as hits. Good enough for EMA calibration.
        """
        if not self.auto_tune_bomb or not self._pending_bombs:
            return
        current = memory.current_step
        resolved = [b for b in self._pending_bombs if current >= b.placed_step + BOMB_TIMER + 1]
        for bomb in resolved:
            self._pending_bombs.remove(bomb)
            hit = any(
                p in bomb.blast_cells and memory.enemy_agents.get(p, -1) >= bomb.placed_step
                for p in memory.enemy_agents
            )
            self._update_bomb_ema(hit)

    def _update_bomb_ema(self, hit: bool) -> None:
        self._hit_ema = _TUNE_EMA_ALPHA * self._hit_ema + (1 - _TUNE_EMA_ALPHA) * (1.0 if hit else 0.0)
        self._ema_n += 1
        if self._ema_n >= _TUNE_WARMUP:
            # Positive error → hit rate below target → raise threshold (bomb less).
            error = self.bomb_tune_target - self._hit_ema
            self._tuned_threshold = max(_TUNE_MIN, min(_TUNE_MAX,
                self._tuned_threshold + 0.05 * error))

    @property
    def tuned_threshold(self) -> float:
        """Current effective threshold (auto-tuned or static)."""
        return self._effective_threshold()

    # ── adaptive base-weight helpers ─────────────────────────────────────────

    def _effective_base_weight(self) -> float:
        return self._adaptive_weight if self.adaptive_base_weight else self.base_route_weight

    def _update_adaptive_weight(self, obs: ParsedObs, memory: MapMemory) -> None:
        """Adjust _adaptive_weight based on observed enemy aggression.

        Called once per step before any action selection.  Two threat signals:
          1. An enemy agent is within DEFEND_RADIUS of our base (proactive).
          2. Base health dropped since the previous step (reactive — enemy made
             contact before we could see them).

        On threat: weight resets to base_weight_min and a defensive cooldown
        starts.  Cooldown ticks down while no new threat is detected; once it
        expires the weight ramps back up toward base_route_weight at
        base_weight_ramp_rate per step.

        State is reset at the start of each round (obs.step == 0) so every
        round begins with low base priority and ramps up from scratch.
        """
        if not self.adaptive_base_weight:
            return

        # Fresh round — reset state so every round starts with low base priority.
        if obs.step == 0:
            self._adaptive_weight = self.base_weight_min
            self._attack_cooldown = 0
            self._prev_base_health = None

        # Base destroyed — nothing left to defend; snap to full aggression and
        # stay there. _try_defend already returns None when base_health <= 0, so
        # the cooldown would only waste time on a dead base.
        if obs.base_health <= 0:
            self._adaptive_weight = self.base_route_weight
            self._attack_cooldown = 0
            self._prev_base_health = obs.base_health
            return

        # Signal 1: base health dropped (enemy hit our base).
        health_drop = (
            self._prev_base_health is not None
            and obs.base_health < self._prev_base_health
        )
        self._prev_base_health = obs.base_health

        # Signal 2: enemy visible within defend radius of our base.
        enemy_near_base = False
        if memory.ally_base is not None and memory.enemy_agents and obs.base_health > 0:
            bx, by = memory.ally_base
            enemy_near_base = any(
                abs(p[0] - bx) + abs(p[1] - by) <= DEFEND_RADIUS
                for p in memory.enemy_agents
            )

        if health_drop or enemy_near_base:
            # Attack detected — reset weight and restart defensive cooldown.
            self._adaptive_weight = self.base_weight_min
            self._attack_cooldown = self.base_weight_attack_cooldown
        elif self._attack_cooldown > 0:
            # Still in defensive posture; tick down but keep weight low.
            self._attack_cooldown -= 1
        else:
            # Peaceful — ramp weight up toward the base_route_weight ceiling.
            self._adaptive_weight = min(
                self.base_route_weight,
                self._adaptive_weight + self.base_weight_ramp_rate,
            )

    # ── helpers ─────────────────────────────────────────────────────────────

    def _maybe_wall_break(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        action: int,
    ) -> Optional[int]:
        """If the planned move crosses a destructible wall, substitute PLACE_BOMB.

        If we already placed an ally bomb at our cell that will break the wall,
        STAY and let it detonate instead of wasting another bomb.
        """
        if action not in (Action.FORWARD, Action.BACKWARD):
            return int(action)
        next_pos = next_pos_after(obs.location, obs.direction, action)
        if not memory.edge_is_destructible_wall(obs.location, next_pos):
            return int(action)

        if not self.wall_breaking:
            return None  # planning shouldn't have produced this — reject

        sitting_bomb = memory.bombs.get(obs.location)
        if sitting_bomb is not None and sitting_bomb.ally:
            return int(Action.STAY)

        if self.bomb_economy and self.wall_break_tile_threshold > 0.0:
            if memory.tile_value(next_pos) < self.wall_break_tile_threshold:
                return None

        if obs.action_mask[Action.PLACE_BOMB] == 1 and obs.team_bombs > 0:
            return int(Action.PLACE_BOMB)
        return None

    def _immediate_neighbors(
        self,
        obs: ParsedObs,
        memory: MapMemory,
    ) -> list[tuple[Action, tuple[int, int]]]:
        out: list[tuple[Action, tuple[int, int]]] = []
        fdx, fdy = DIR_VECTOR[Direction(obs.direction)]
        fwd = (obs.location[0] + fdx, obs.location[1] + fdy)
        if memory.in_bounds(fwd) and memory.passable(obs.location, fwd):
            out.append((Action.FORWARD, fwd))
        bdx, bdy = DIR_VECTOR[Direction((obs.direction + 2) % 4)]
        back = (obs.location[0] + bdx, obs.location[1] + bdy)
        if memory.in_bounds(back) and memory.passable(obs.location, back):
            out.append((Action.BACKWARD, back))
        out.append((Action.LEFT, obs.location))
        out.append((Action.RIGHT, obs.location))
        out.append((Action.STAY, obs.location))
        return out

    def _mask_check(self, action: int, obs: ParsedObs) -> int:
        if 0 <= action < len(obs.action_mask) and obs.action_mask[action] == 1:
            return action
        for fallback in (Action.STAY, Action.LEFT, Action.RIGHT):
            if obs.action_mask[fallback] == 1:
                return int(fallback)
        return int(Action.STAY)
