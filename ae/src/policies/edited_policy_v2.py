"""Experimental clone of EditedHeuristicPolicy for testing new ideas behind toggles.

This subclasses the production `EditedHeuristicPolicy` and layers on independent
behaviours behind constructor toggles. Dodge stays the untouchable first priority
and `_try_defend` remains disabled, so these changes do not alter how the agent
escapes real danger.

New toggles
-----------
siege_base — once the agent is near a known live enemy base, lock onto it.
    Base positions are known up front (cached map); only their alive/health
    state is learned at run time, and `memory.enemy_bases` drops a base as soon
    as it is observed destroyed — so this targets only live bases. Long-range
    approach is left to the existing `proactive_base_routing` (which already
    pulls the agent toward bases while collecting en route). When the agent can
    reach a cell from which a bomb hits the nearest base within `siege_radius`
    steps, it instead: (1) sweeps resources within `siege_radius` of that base
    to fund bombs, and (2) when none remain, holds on / returns to a bombing
    stance so the existing `_try_attack` keeps detonating the base every time a
    bomb is available. Dodge and attack still fire first, untouched.

    A livelock guard is built directly into the siege routine: if siege keeps
    failing to advance for `siege_stuck_patience` consecutive steps (e.g. two
    agents turning in place, staring at each other), it backs off to normal
    collect routing for `siege_override_steps` steps to break the tie. A STAY
    hold on a valid bombing stance counts as productive, not stuck.
"""

from typing import Optional

from constants import (
    Action,
    BASE_DESTROY_BONUS,
    BASE_MAX_HEALTH,
    BOMB_ATTACK,
    BOMB_BLAST_RADIUS,
)
from .edited_policy import EXPLORE_BUDGET, EditedHeuristicPolicy
from map_memory import MapMemory
from observation import ParsedObs
from pathfinding import first_action_to, reachable_cells
from threat import cells_in_blast


class EditedHeuristicPolicyV2(EditedHeuristicPolicy):
    def __init__(
        self,
        *,
        siege_base: bool = True,
        siege_radius: int = 5,
        siege_stuck_patience: int = 3,
        siege_override_steps: int = 8,
        # ── attacking opponents ──────────────────────────────────────────────
        # kill_targeting: value a bomb hit on an enemy at <= BOMB_ATTACK HP as a
        # finishing blow (+kill_finish_bonus), mirroring the base-destroy bonus
        # for enemy bases. A kill is worth AGENT_KILL_BONUS=30 (split across
        # contributing bombs) plus a 3-turn freeze, so finishing a low-HP enemy
        # is far more valuable than the flat agent_bomb_value the base scorer
        # assigns. Requires map_memory.enemy_agent_health (HP is observable).
        kill_targeting: bool = False,
        kill_finish_bonus: float = 30.0,
        # ── threat awareness ─────────────────────────────────────────────────
        # anticipate_enemy_bombs: treat the blast footprint of every freshly
        # sighted enemy agent as soft danger for collect/explore routing, so we
        # don't wander into a cell an adjacent enemy can bomb next tick. Dodge is
        # untouched (it uses only the live-bomb timeline), so this never blocks a
        # real escape.
        anticipate_enemy_bombs: bool = False,
        # ── routing ──────────────────────────────────────────────────────────
        # cluster_collect: bonus a collectible's score by the value of other
        # collectibles within cluster_radius (Manhattan), so routes that bunch
        # several tiles together beat a lone distant tile of equal value. Tiles
        # respawn within ~40 steps, so sitting in a dense cluster compounds.
        cluster_collect: bool = False,
        cluster_radius: int = 2,
        cluster_weight: float = 0.35,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.siege_base = siege_base
        self.siege_radius = siege_radius
        self.siege_stuck_patience = siege_stuck_patience
        self.siege_override_steps = siege_override_steps
        self.kill_targeting = kill_targeting
        self.kill_finish_bonus = kill_finish_bonus
        self.anticipate_enemy_bombs = anticipate_enemy_bombs
        self.cluster_collect = cluster_collect
        self.cluster_radius = cluster_radius
        self.cluster_weight = cluster_weight

        # Livelock-guard state for siege (used inside _siege_collect): track
        # positional stagnation so siege can back off to normal collect routing
        # when it keeps turning in place without advancing. Reset on each round.
        self._siege_prev_pos: Optional[tuple[int, int]] = None
        self._siege_stagnant: int = 0
        self._siege_suppress_until: int = -1
        self._siege_holding: bool = False
        self._siege_last_step: int = -1

    # ── collect routing (siege → base) ───────────────────────────────────────
    def _try_collect(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        if self.siege_base and memory.enemy_bases:
            action = self._siege_collect(obs, memory, danger_now)
            if action is not None:
                return action
        return super()._try_collect(obs, memory, danger_now)

    # ── siege: lock onto a nearby live enemy base ────────────────────────────
    def _siege_collect(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        # Livelock guard: track positional progress across siege steps. If siege
        # keeps failing to advance (e.g. two agents turning in place against each
        # other), back off to normal collect routing for siege_override_steps
        # steps to break the tie. A STAY hold on a valid bombing stance is
        # productive (waiting for the next bomb) and does not count as stuck.
        if obs.step == 0 or obs.step < self._siege_last_step:
            self._siege_prev_pos = obs.location
            self._siege_stagnant = 0
            self._siege_suppress_until = -1
            self._siege_holding = False
        self._siege_last_step = obs.step

        moved = self._siege_prev_pos is None or obs.location != self._siege_prev_pos
        if moved or self._siege_holding:
            self._siege_stagnant = 0
        else:
            self._siege_stagnant += 1
        self._siege_prev_pos = obs.location
        self._siege_holding = False

        if self._siege_stagnant >= self.siege_stuck_patience:
            self._siege_suppress_until = obs.step + self.siege_override_steps
            self._siege_stagnant = 0

        if obs.step <= self._siege_suppress_until:
            return None  # defer to normal collect routing to break the livelock

        anchor = min(memory.enemy_bases, key=lambda b: self._manhattan(b, obs.location))

        edge = self._edge_cost(memory, danger_avoid=danger_now)
        distances = reachable_cells(
            obs.location, obs.direction, edge, max_cost=EXPLORE_BUDGET
        )

        stance = self._siege_stance_cells(memory, anchor, distances)
        if not stance:
            # Can't reach a bombing position yet — let proactive base routing
            # (and wall-breaking) handle the long-range approach.
            return None

        on_stance = obs.location in stance
        nearest_stance = min(distances[c] for c in stance)
        if not on_stance and nearest_stance > self.siege_radius:
            # Still too far to count as "near" the base — defer to normal
            # routing, which already drifts toward bases.
            return None

        # (1) Sweep resources within siege_radius of the base to fund bombs.
        best_cell: Optional[tuple[int, int]] = None
        best_score = 0.0
        for cell in memory.collectible_cells():
            if cell not in distances:
                continue
            if self._manhattan(cell, anchor) > self.siege_radius:
                continue
            value = memory.tile_value(cell)
            if value <= 0:
                continue
            score = value / (distances[cell] + 1.0)
            if score > best_score:
                best_score = score
                best_cell = cell

        if best_cell is not None:
            self._debug_target = best_cell
            action = first_action_to(obs.location, obs.direction, {best_cell}, edge)
            if action is not None:
                return self._maybe_wall_break(obs, memory, action)

        # (2) No local resource — hold on / return to a bombing stance so
        # _try_attack keeps detonating the base whenever a bomb is ready.
        self._debug_target = anchor
        if on_stance:
            # Productive wait — sitting on a bombing stance for the next bomb,
            # not a livelock; flag so the stuck-guard does not count it.
            self._siege_holding = True
            return int(Action.STAY)
        action = first_action_to(obs.location, obs.direction, set(stance), edge)
        if action is None:
            return None
        return self._maybe_wall_break(obs, memory, action)

    def _siege_stance_cells(
        self,
        memory: MapMemory,
        anchor: tuple[int, int],
        distances: dict[tuple[int, int], float],
    ) -> set[tuple[int, int]]:
        """Reachable cells from which a bomb placed there would hit `anchor`."""
        stance: set[tuple[int, int]] = set()
        ax, ay = anchor
        for dx in range(-BOMB_BLAST_RADIUS, BOMB_BLAST_RADIUS + 1):
            for dy in range(-BOMB_BLAST_RADIUS, BOMB_BLAST_RADIUS + 1):
                cell = (ax + dx, ay + dy)
                if cell not in distances:
                    continue
                if anchor in cells_in_blast(memory, cell):
                    stance.add(cell)
        return stance

    # ── attacking opponents: HP-aware kill targeting ─────────────────────────
    def _bomb_opportunity_score(
        self, memory: MapMemory, blast: set[tuple[int, int]]
    ) -> float:
        """Base scorer plus a finishing-blow bonus for low-HP enemy agents.

        Identical to the base implementation except that each definite,
        non-escapable agent hit on an enemy known to be at <= BOMB_ATTACK HP
        also adds kill_finish_bonus — the same finishing-blow treatment the base
        scorer already gives enemy bases. Predictive (expected-hit) contribution
        is unchanged: those clouds carry no per-agent HP, so we keep them flat.
        """
        if not self.kill_targeting:
            return super()._bomb_opportunity_score(memory, blast)

        base_score = 0.0
        base_hits = 0
        for p in memory.enemy_bases:
            if p in blast:
                base_hits += 1
                base_hp = memory.enemy_base_health.get(p, BASE_MAX_HEALTH)
                hit_value = self.base_bomb_value
                if base_hp <= BOMB_ATTACK:
                    hit_value += BASE_DESTROY_BONUS
                base_score += hit_value

        agent_score = 0.0
        agent_hits = 0
        for p in memory.enemy_agents:
            if p not in blast:
                continue
            if self._enemy_can_escape_blast(p, blast, memory):
                continue
            agent_hits += 1
            agent_score += self.agent_bomb_value
            hp = memory.enemy_agent_health.get(p)
            if hp is not None and hp <= BOMB_ATTACK:
                agent_score += self.kill_finish_bonus

        score = base_score + agent_score
        if self.predictive_bomb:
            expected = self._expected_hits(memory, blast)
            if (
                agent_hits > 0
                or base_hits > 0
                or expected >= self._effective_threshold()
            ):
                score += expected * self.agent_bomb_value
        return score

    # ── threat awareness: anticipate enemy bomb placement ────────────────────
    def _extra_danger_cells(
        self,
        obs: ParsedObs,
        memory: MapMemory,
    ) -> set[tuple[int, int]]:
        if not self.anticipate_enemy_bombs:
            return set()
        cells: set[tuple[int, int]] = set()
        for pos, last_step in memory.enemy_agents.items():
            # Only act on enemies we can actually see this tick — stale sightings
            # would over-restrict routing across half the map.
            if last_step != memory.current_step:
                continue
            cells |= cells_in_blast(memory, pos)
        # Never block the cell we're standing on; routing only checks destinations,
        # but this keeps the set clean and avoids pathological no-move states.
        cells.discard(obs.location)
        return cells

    # ── routing: cluster-aware collection ────────────────────────────────────
    def _collect_value(self, cell: tuple[int, int], memory: MapMemory) -> float:
        base_value = memory.tile_value(cell)
        if not self.cluster_collect or base_value <= 0:
            return base_value
        cx, cy = cell
        bonus = 0.0
        for other in memory.collectible_cells():
            if other == cell:
                continue
            if abs(other[0] - cx) + abs(other[1] - cy) <= self.cluster_radius:
                bonus += memory.tile_value(other)
        return base_value + self.cluster_weight * bonus

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
