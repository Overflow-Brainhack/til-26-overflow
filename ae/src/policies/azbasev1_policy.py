"""Azbase v1 — baseline aggressive-base-routing policy.

Preserved from the `overflow-ae:azbase` Docker image. Eval scores: roughly
0.66-0.72 across submissions (0.663, 0.694, 0.665, 0.702, 0.724, 0.720,
0.661, 0.720, 0.713).

NOTE: The v3 submission (eval 0.68-0.75) was *the same Python class as this
one* — its ae_manager imported the inner ``azbase_preserved.berserker_base_
azbase_policy.BerserkerBasePolicy``, which is byte-identical to the file this
class was extracted from. The score difference between v1 and v3 was
submission/env variance, not code, so we don't ship a separate
``AzbaseV3Policy``. v4 is genuinely distinct (collected-tile cooldown +
base-HP filtering) and lives in ``azbasev4_policy.py``.

Extends ``AzbaseEditedHeuristicPolicyV2`` (the preserved-azbase EditedHeuristic
stack — distinct from the live ``policies.edited_policy_v2`` which has
diverged since). Forces ``proactive_base_routing=True`` and
``adaptive_base_weight=False``; overrides ``_try_collect`` to prefer routing
to enemy-base firing cells when the team has at least one bomb.
"""

from __future__ import annotations

from typing import Optional

from .azbase_edited_policy import EXPLORE_BUDGET
from .azbase_edited_policy_v2 import EditedHeuristicPolicyV2
from map_memory import MapMemory
from observation import ParsedObs
from pathfinding import first_action_to, reachable_cells
from threat import cells_in_blast


class AzbaseV1Policy(EditedHeuristicPolicyV2):
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
