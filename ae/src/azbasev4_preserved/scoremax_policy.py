"""Score-maximising AE policy.

This is deliberately less "nice" than the balanced heuristic. It treats the
round as a 200-step scoring problem: base damage is still the backbone, but
direct enemy-agent hits and kills are worth enough that we should route toward
multi-value bomb cells instead of only chasing the nearest base firing square.
"""

from __future__ import annotations

from typing import Optional

from berserker_base_policy import BerserkerBasePolicy
from constants import (
    AGENT_KILL_BONUS,
    BASE_DESTROY_BONUS,
    BASE_MAX_HEALTH,
    BOMB_ATTACK,
)
from edited_policy import EXPLORE_BUDGET
from map_memory import MapMemory
from observation import ParsedObs
from pathfinding import first_action_to, reachable_cells
from threat import cells_in_blast


class ScoreMaxPolicy(BerserkerBasePolicy):
    """Aggressive scorer that values bases, agent hits, kills, and refill routes."""

    def __init__(
        self,
        *,
        route_damage_weight: float = 1.0,
        route_agent_weight: float = 0.8,
        route_expected_agent_weight: float = 0.45,
        route_cost_penalty: float = 1.25,
        min_blast_route_score: float = 12.0,
        low_bomb_collect_cutoff: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(target_weakest_base=True, **kwargs)
        self.route_damage_weight = route_damage_weight
        self.route_agent_weight = route_agent_weight
        self.route_expected_agent_weight = route_expected_agent_weight
        self.route_cost_penalty = route_cost_penalty
        self.min_blast_route_score = min_blast_route_score
        self.low_bomb_collect_cutoff = low_bomb_collect_cutoff

    def _try_collect(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        edge = self._edge_cost(memory, danger_avoid=danger_now)

        if obs.team_bombs > 0 and memory.enemy_bases:
            action = super()._route_to_enemy_base(obs, memory, edge)
            if action is not None:
                return action

        collect = self._collect_tiles(obs, memory, edge)
        if collect is not None:
            return collect

        if obs.team_bombs > 0 and not self._live_enemy_bases(memory):
            return self._route_to_best_blast_cell(obs, memory, edge)
        return None

    def _try_attack(self, obs: ParsedObs, memory: MapMemory) -> Optional[int]:
        # Early and mid-game, do not spend route-critical bombs on isolated
        # agent shots. Base bombs still fire because the parent always accepts
        # direct base hits. Agent farming unlocks once the base race is mostly
        # over or no live bases remain in memory.
        original_agent_value = self.agent_bomb_value
        original_threshold = self.bomb_reserve_threshold
        if self._live_enemy_bases(memory):
            self.agent_bomb_value = min(self.agent_bomb_value, 1.0)
        else:
            self.agent_bomb_value = max(self.agent_bomb_value, 20.0)
            self.bomb_reserve_threshold = max(self.bomb_reserve_threshold, 12.0)
        try:
            return super()._try_attack(obs, memory)
        finally:
            self.agent_bomb_value = original_agent_value
            self.bomb_reserve_threshold = original_threshold

    def _route_to_best_blast_cell(self, obs, memory, edge) -> Optional[int]:
        distances = reachable_cells(
            obs.location,
            obs.direction,
            edge,
            max_cost=EXPLORE_BUDGET,
        )

        best_cell: Optional[tuple[int, int]] = None
        best_score = float("-inf")

        for cell, dist in distances.items():
            if cell == obs.location:
                continue
            raw = self._blast_reward_estimate(memory, cell)
            if raw <= 0.0:
                continue
            score = raw - self.route_cost_penalty * float(dist)
            if score > best_score:
                best_score = score
                best_cell = cell

        if best_cell is None or best_score < self.min_blast_route_score:
            return None

        self._debug_target = best_cell
        action = first_action_to(obs.location, obs.direction, {best_cell}, edge)
        if action is None:
            return None
        return self._maybe_wall_break(obs, memory, action)

    def _blast_reward_estimate(self, memory: MapMemory, bomb_cell: tuple[int, int]) -> float:
        blast = cells_in_blast(memory, bomb_cell)
        value = 0.0

        for base in memory.enemy_bases:
            hp = memory.enemy_base_health.get(base, BASE_MAX_HEALTH)
            if hp <= 0 or base not in blast:
                continue
            value += min(BOMB_ATTACK, hp) * self.route_damage_weight
            if hp <= BOMB_ATTACK:
                value += BASE_DESTROY_BONUS

        recent_agents = [
            pos
            for pos, seen_step in memory.enemy_agents.items()
            if memory.current_step - seen_step <= 5
        ]
        direct_hits = sum(1 for pos in recent_agents if pos in blast)
        if direct_hits:
            value += direct_hits * BOMB_ATTACK * self.route_agent_weight
            # We usually do not know enemy HP, but repeated contact near bases
            # makes kills plausible; add a partial kill premium, not the full 15.
            value += direct_hits * AGENT_KILL_BONUS * 0.35
        else:
            expected = self._expected_hits(memory, blast)
            value += expected * BOMB_ATTACK * self.route_expected_agent_weight

        return value

    def _live_enemy_bases(self, memory: MapMemory) -> list[tuple[int, int]]:
        return [
            base
            for base in memory.enemy_bases
            if memory.enemy_base_health.get(base, BASE_MAX_HEALTH) > 0
        ]

    def _route_to_enemy_base(self, obs, memory, edge) -> Optional[int]:
        # Use the more expensive score route first; fall back to the parent
        # shortest-base route if all blast cells are below threshold.
        scored = self._route_to_best_blast_cell(obs, memory, edge)
        if scored is not None:
            return scored
        return super()._route_to_enemy_base(obs, memory, edge)
