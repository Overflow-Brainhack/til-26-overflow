"""Policy interface and the heuristic implementation.

The Policy abstract class lets us swap in a learned policy later without
touching the manager or server. The heuristic policy implements a priority
decision tree:
    1. Frozen → STAY
    2. Imminent enemy-blast danger → dodge to nearest safe cell
    3. Attack opportunity (enemy in our bomb's blast OR predicted to be) → PLACE_BOMB
    4. Defend (enemy near our base) → intercept
    5. Collect highest value-per-distance tile (optionally through walls)
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
"""

from abc import ABC, abstractmethod
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

        # Mutable auto-tune state — persists across rounds (policy is not re-created on /reset).
        self._tuned_threshold: float = predictive_bomb_threshold
        self._hit_ema: float = bomb_tune_target   # warm-start at the target hit rate
        self._ema_n: int = 0
        self._pending_bombs: list[_PendingBomb] = []

    # ── main entrypoint ─────────────────────────────────────────────────────
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        self._resolve_pending_bombs(memory)

        if obs.frozen_ticks > 0:
            return int(Action.STAY)

        timeline = project_danger(memory)
        danger_now: set[tuple[int, int]] = set()
        for tick, cells in timeline.items():
            if tick <= 1:
                danger_now.update(cells)

        my_blast_tick = imminent_danger(memory, obs.location)
        if my_blast_tick is not None and my_blast_tick <= BOMB_TIMER:
            chosen = self._dodge(obs, memory, timeline)
            if chosen is not None:
                return self._mask_check(chosen, obs)

        attack = self._try_attack(obs, memory)
        if attack is not None:
            return self._mask_check(attack, obs)

        defend = self._try_defend(obs, memory, danger_now)
        if defend is not None:
            return self._mask_check(defend, obs)

        collect = self._try_collect(obs, memory, danger_now)
        if collect is not None:
            return self._mask_check(collect, obs)

        explore = self._try_explore(obs, memory, danger_now)
        if explore is not None:
            return self._mask_check(explore, obs)

        return int(Action.STAY)

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

        action = first_action_to(obs.location, obs.direction, safe, dodge_cost,
                                  action_mask=obs.action_mask)
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

        action = first_action_to(obs.location, obs.direction, targets, edge,
                                  action_mask=obs.action_mask)
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
        if not candidates:
            return None

        edge = self._edge_cost(memory, danger_avoid=danger_now)
        distances = reachable_cells(
            obs.location, obs.direction, edge,
            max_cost=EXPLORE_BUDGET, action_mask=obs.action_mask,
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
        if best_cell is None:
            return None

        action = first_action_to(obs.location, obs.direction, {best_cell}, edge,
                                  action_mask=obs.action_mask)
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

        action = first_action_to(obs.location, obs.direction, frontier, edge,
                                  action_mask=obs.action_mask)
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
        mask = obs.action_mask
        fdx, fdy = DIR_VECTOR[Direction(obs.direction)]
        fwd = (obs.location[0] + fdx, obs.location[1] + fdy)
        if mask[Action.FORWARD] and memory.in_bounds(fwd) and memory.passable(obs.location, fwd):
            out.append((Action.FORWARD, fwd))
        bdx, bdy = DIR_VECTOR[Direction((obs.direction + 2) % 4)]
        back = (obs.location[0] + bdx, obs.location[1] + bdy)
        if mask[Action.BACKWARD] and memory.in_bounds(back) and memory.passable(obs.location, back):
            out.append((Action.BACKWARD, back))
        if mask[Action.LEFT]:
            out.append((Action.LEFT, obs.location))
        if mask[Action.RIGHT]:
            out.append((Action.RIGHT, obs.location))
        out.append((Action.STAY, obs.location))  # always available as last resort
        return out

    def _mask_check(self, action: int, obs: ParsedObs) -> int:
        if 0 <= action < len(obs.action_mask) and obs.action_mask[action] == 1:
            return action
        for fallback in (Action.STAY, Action.LEFT, Action.RIGHT):
            if obs.action_mask[fallback] == 1:
                return int(fallback)
        return int(Action.STAY)
