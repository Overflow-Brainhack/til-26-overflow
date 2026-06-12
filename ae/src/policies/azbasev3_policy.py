"""Berserker Base policy as shipped in the ``overflow-ae:azbasev3`` image.

Unfolded from ``azbasev3_preserved(0.748)/berserker_base_policy.py``
(byte-identical to the v3 image workspace; best evaluator score 0.754).
Only the two azbase imports were re-pointed at sibling modules in this
package. See ``policies/README.md`` for the full snapshot provenance.

This is also the **live submission policy**: the former top-level
``berserker_base_policy.py`` was byte-identical and was deduplicated into
this module; ``ae_manager`` imports ``HeuristicPolicy`` from here. If the
live policy needs to diverge from the v3 snapshot, fork it first.
"""

from __future__ import annotations

from typing import Optional

from .azbase_edited_policy import EXPLORE_BUDGET
from .azbase_edited_policy_v2 import EditedHeuristicPolicyV2
from map_memory import MapMemory
from observation import ParsedObs
from pathfinding import first_action_to, reachable_cells
from threat import cells_in_blast


class BerserkerBasePolicy(EditedHeuristicPolicyV2):
    """Keep the v2 safety stack, but route objectives toward enemy bases first."""

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
        self._collected: set[tuple[int, int]] = set()

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.step == 0:
            self._collected.clear()
        if memory.tile_contents.get(obs.location) in ("mission", "resource", "recon"):
            self._collected.add(obs.location)
        return super().choose(obs, memory)

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
            if cell != obs.location and cell not in self._collected
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
