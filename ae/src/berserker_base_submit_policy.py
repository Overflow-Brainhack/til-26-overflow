"""Submission-preserved patched Berserker Base policy.

This keeps the high-scoring patched berserker-base behavior in its own file so
later experiments can change `berserker_base_policy.py` or `scoremax_policy.py`
without moving the submission target.
"""

from __future__ import annotations

from typing import Optional

from edited_policy import EXPLORE_BUDGET
from edited_policy_v2 import EditedHeuristicPolicyV2
from map_memory import MapMemory
from observation import ParsedObs
from pathfinding import first_action_to, reachable_cells
from threat import cells_in_blast


class BerserkerBaseSubmitPolicy(EditedHeuristicPolicyV2):
    """Route to enemy-base firing cells first, then collect for refill/value."""

    def __init__(
        self,
        *,
        target_weakest_base: bool = True,
        resource_refill_bias: float = 2.0,
        bombs_low_threshold: int = 1,
        **kwargs,
    ) -> None:
        kwargs["proactive_base_routing"] = True
        kwargs["adaptive_base_weight"] = False
        super().__init__(**kwargs)
        self.target_weakest_base = target_weakest_base
        self.resource_refill_bias = resource_refill_bias
        self.bombs_low_threshold = bombs_low_threshold

    def _try_collect(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        edge = self._edge_cost(memory, danger_avoid=danger_now)

        if obs.team_bombs > 0 and memory.enemy_bases:
            action = self._route_to_enemy_base(obs, memory, edge)
            if action is not None:
                return action

        return self._collect_tiles(obs, memory, edge)

    def _route_to_enemy_base(self, obs, memory, edge) -> Optional[int]:
        distances = reachable_cells(
            obs.location,
            obs.direction,
            edge,
            max_cost=EXPLORE_BUDGET,
        )

        for base in self._ordered_bases(memory):
            firing_cells = {
                cell
                for cell in cells_in_blast(memory, base)
                if memory.in_bounds(cell) and cell != base
            }
            reachable = [cell for cell in firing_cells if cell in distances]
            if not reachable:
                continue

            best_cell = min(reachable, key=lambda cell: distances[cell])
            if best_cell == obs.location:
                continue
            self._debug_target = best_cell
            action = first_action_to(obs.location, obs.direction, {best_cell}, edge)
            if action is None:
                continue
            return self._maybe_wall_break(obs, memory, action)

        return None

    def _collect_tiles(self, obs, memory, edge) -> Optional[int]:
        candidates = [
            cell
            for cell in memory.collectible_cells()
            if cell != obs.location and not self._tile_recently_collected(cell, obs.step)
        ]
        if not candidates:
            return None

        distances = reachable_cells(
            obs.location,
            obs.direction,
            edge,
            max_cost=EXPLORE_BUDGET,
        )
        best_score = 0.0
        best_cell: Optional[tuple[int, int]] = None

        for cell in candidates:
            dist = distances.get(cell)
            if dist is None:
                continue
            value = memory.tile_value(cell)
            if value <= 0:
                continue
            if (
                obs.team_bombs <= self.bombs_low_threshold
                and memory.tile_contents.get(cell) == "resource"
            ):
                value *= self.resource_refill_bias
            score = value / (dist + 1.0)
            if score > best_score:
                best_score = score
                best_cell = cell

        if best_cell is None:
            return None

        self._debug_target = best_cell
        action = first_action_to(obs.location, obs.direction, {best_cell}, edge)
        if action is None:
            return None
        return self._maybe_wall_break(obs, memory, action)

    def _pick_base(self, memory: MapMemory) -> Optional[tuple[int, int]]:
        if not memory.enemy_bases:
            return None
        return self._ordered_bases(memory)[0]

    def _ordered_bases(self, memory: MapMemory) -> list[tuple[int, int]]:
        if self.target_weakest_base:
            return sorted(
                memory.enemy_bases,
                key=lambda cell: memory.enemy_base_health.get(cell, 100.0),
            )
        return list(memory.enemy_bases)
