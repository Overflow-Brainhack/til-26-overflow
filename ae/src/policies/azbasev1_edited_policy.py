"""AzbaseV1 + experimental features, each behind an OFF-by-default toggle.

This subclasses the pristine `AzbaseV1Policy` (the A/B reference, eval ~0.61 on
the current deterministic eval) and layers on three independent behaviours so
each can be eval-tested in isolation and folded back into `AzbaseV1Policy` only
if it shows a positive real-eval delta. With every toggle off this is exactly
`AzbaseV1Policy`.

Game facts these features exploit (constants.py / threat.py):
  - **Friendly fire is OFF** — your own bombs never damage you, so bombing has
    no survival cost, only supply cost (BOMB_COST=1.5/bomb). Aggression is cheap.
  - destroy_base = +50 (base 100 HP / 20 dmg = 5 bombs), kill = +30 (agent 60 HP),
    and the +50/+30 land ONLY on the finishing blow. Concentrating fire is what
    converts bombs into points.
  - 6-player free-for-all: no allies; up to 5 enemy bases / 5 enemy agents.

Features
--------
hp_aware_kills
    The base scores an agent bomb-hit at `agent_bomb_value`≈1 and never credits
    the +30 kill (`_bomb_opportunity_score` only adds the base-destroy bonus).
    This (a) adds AGENT_KILL_BONUS to the opportunity score for a lethal,
    no-escape hit on a visible low-HP enemy, and (b) proactively routes to a
    firing cell on a one-bomb-kill enemy to take the free +30. Enemy HP is read
    from the viewcone (LOS-gated; `map_memory` does not persist agent HP).

base_siege
    `AzbaseV1Policy._route_to_enemy_base` *abandons* a base the tick after it
    bombs it: once we stand on the base's nearest firing cell, `best_cell ==
    obs.location` makes it `continue` to a different base — so it delivers one
    bomb per base then wanders, spreading chip damage that never earns the +50.
    This commits to one base and, after each bomb, hops to that base's NEXT
    firing cell to keep hammering it until it dies or we run out of bombs.

endgame_dump
    In the last `endgame_steps` steps, hoarding/exploring earns nothing: drop the
    predictive-bomb threshold to ~0 (bomb on the faintest opportunity) and stop
    exploring so leftover bombs get spent on nearby targets.
"""

from __future__ import annotations

from typing import Optional

from constants import (
    Action,
    AGENT_KILL_BONUS,
    AGENT_MAX_HEALTH,
    BASE_MAX_HEALTH,
    BOMB_ATTACK,
    GRID_SIZE,
    NUM_ITERS,
    ViewChannel,
)
from map_memory import MapMemory
from observation import ParsedObs, base_view_to_world, view_to_world
from pathfinding import first_action_to, reachable_cells
from threat import cells_in_blast

from .azbase_edited_policy import EXPLORE_BUDGET
from .azbasev1_policy import AzbaseV1Policy

# Enemy HP at or below this dies to a single bomb (one-shot kill candidate).
_ONE_BOMB_KILL_HP = BOMB_ATTACK


class AzbaseV1EditedPolicy(AzbaseV1Policy):
    """AzbaseV1 with hp_aware_kills / base_siege / endgame_dump (all default OFF)."""

    def __init__(
        self,
        *,
        hp_aware_kills: bool = False,
        finisher_max_cost: float = 4.0,
        base_siege: bool = False,
        endgame_dump: bool = False,
        endgame_steps: int = 30,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hp_aware_kills = hp_aware_kills
        self.finisher_max_cost = finisher_max_cost
        self.base_siege = base_siege
        self.endgame_dump = endgame_dump
        self.endgame_steps = endgame_steps

        # Per-tick scratch (set in choose, read by the override hooks).
        self._cur_step: int = 0
        self._enemy_hp: dict[tuple[int, int], float] = {}
        # Sticky siege commitment — reset each round (step == 0).
        self._siege_target: Optional[tuple[int, int]] = None

    # ── entrypoint: refresh per-tick scratch, then run the normal ladder ──────
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.step == 0:
            self._siege_target = None
        self._cur_step = obs.step
        self._enemy_hp = self._visible_enemy_hp(obs) if self.hp_aware_kills else {}
        return super().choose(obs, memory)

    # ── endgame gate ──────────────────────────────────────────────────────────
    def _in_endgame(self) -> bool:
        return self.endgame_dump and self._cur_step >= NUM_ITERS - self.endgame_steps

    def _effective_threshold(self) -> float:
        base = super()._effective_threshold()
        # Endgame: bomb on the faintest predictive opportunity.
        return min(base, 0.05) if self._in_endgame() else base

    def _try_explore(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        # Endgame: don't waste the closing steps revealing map; let collect/attack
        # keep us on offence near bases/enemies.
        if self._in_endgame():
            return None
        return super()._try_explore(obs, memory, danger_now)

    # ── kill-value credit (used by economy-mode bomb scoring) ─────────────────
    def _bomb_opportunity_score(
        self, memory: MapMemory, blast: set[tuple[int, int]]
    ) -> float:
        score = super()._bomb_opportunity_score(memory, blast)
        if not self.hp_aware_kills:
            return score
        for epos in memory.enemy_agents:
            if (
                epos in blast
                and self._enemy_hp.get(epos, AGENT_MAX_HEALTH) <= _ONE_BOMB_KILL_HP
                and not self._enemy_can_escape_blast(epos, blast, memory)
            ):
                score += AGENT_KILL_BONUS
        return score

    # ── collect ladder: finisher → siege → base AzbaseV1 collect ──────────────
    def _try_collect(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        if self.hp_aware_kills:
            action = self._try_finisher(obs, memory, danger_now)
            if action is not None:
                return action
        if self.base_siege:
            action = self._try_siege(obs, memory, danger_now)
            if action is not None:
                return action
        return super()._try_collect(obs, memory, danger_now)

    # ── feature: route to a free kill on a wounded, visible enemy ─────────────
    def _try_finisher(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        if obs.team_bombs <= 0:
            return None
        targets = [p for p, hp in self._enemy_hp.items() if hp <= _ONE_BOMB_KILL_HP]
        if not targets:
            return None

        edge = self._edge_cost(memory, danger_avoid=danger_now)
        distances = reachable_cells(
            obs.location, obs.direction, edge, max_cost=self.finisher_max_cost
        )

        best_cell: Optional[tuple[int, int]] = None
        best_cost = float("inf")
        for enemy in targets:
            # A bomb at C kills the enemy iff enemy ∈ blast(C); by blast/LOS
            # symmetry the firing cells for the enemy are blast(enemy).
            for cell in cells_in_blast(memory, enemy):
                cost = distances.get(cell)
                if cost is None or cost >= best_cost:
                    continue
                best_cost = cost
                best_cell = cell

        # On a firing cell already → _try_attack handles the bomb; nothing to route.
        if best_cell is None or best_cell == obs.location:
            return None

        self._debug_target = best_cell
        action = first_action_to(obs.location, obs.direction, {best_cell}, edge)
        if action is None:
            return None
        return self._maybe_wall_break(obs, memory, action)

    # ── feature: commit to one base and sustain fire on it ────────────────────
    def _try_siege(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        if obs.team_bombs <= 0 or not memory.enemy_bases:
            self._siege_target = None
            return None

        edge = self._edge_cost(memory, danger_avoid=danger_now)
        distances = reachable_cells(
            obs.location, obs.direction, edge, max_cost=EXPLORE_BUDGET
        )

        # Keep the commitment if the target is still alive and still has a
        # reachable firing cell; otherwise re-commit to the weakest reachable base.
        target = self._siege_target
        if (
            target is None
            or target not in memory.enemy_bases
            or not self._firing_cells(memory, target, distances)
        ):
            target = self._pick_siege_target(memory, distances)
            self._siege_target = target
        if target is None:
            return None

        firing = self._firing_cells(memory, target, distances)
        best_cell = min(firing, key=lambda c: distances[c])
        if best_cell == obs.location:
            # We just bombed from here (sitting ally bomb); hop to the next firing
            # cell on the SAME base to keep hammering instead of abandoning it.
            others = [c for c in firing if c != obs.location]
            if not others:
                return None
            best_cell = min(others, key=lambda c: distances[c])

        self._debug_target = best_cell
        action = first_action_to(obs.location, obs.direction, {best_cell}, edge)
        if action is None:
            return None
        return self._maybe_wall_break(obs, memory, action)

    def _firing_cells(
        self,
        memory: MapMemory,
        base: tuple[int, int],
        distances: dict[tuple[int, int], float],
    ) -> list[tuple[int, int]]:
        return [
            cell
            for cell in cells_in_blast(memory, base)
            if cell != base and memory.in_bounds(cell) and cell in distances
        ]

    def _pick_siege_target(
        self,
        memory: MapMemory,
        distances: dict[tuple[int, int], float],
    ) -> Optional[tuple[int, int]]:
        ordered = sorted(
            memory.enemy_bases,
            key=lambda b: memory.enemy_base_health.get(b, BASE_MAX_HEALTH),
        )
        for base in ordered:
            if self._firing_cells(memory, base, distances):
                return base
        return None

    # ── read enemy HP straight from the viewcone (LOS-gated) ──────────────────
    def _visible_enemy_hp(self, obs: ParsedObs) -> dict[tuple[int, int], float]:
        out: dict[tuple[int, int], float] = {}
        self._scan_enemy_hp(
            obs.agent_view,
            lambda r, c: view_to_world(obs.location, obs.direction, r, c),
            out,
        )
        self._scan_enemy_hp(
            obs.base_view,
            lambda r, c: base_view_to_world(obs.base_location, r, c),
            out,
        )
        return out

    @staticmethod
    def _scan_enemy_hp(view, world_for_cell, out: dict[tuple[int, int], float]) -> None:
        rows, cols = view.shape[:2]
        for r in range(rows):
            for c in range(cols):
                cell = view[r, c]
                if cell[ViewChannel.VISIBLE] < 0.5 or cell[ViewChannel.ENEMY_AGENT] < 0.5:
                    continue
                pos = world_for_cell(r, c)
                if not (0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE):
                    continue
                out[pos] = float(cell[ViewChannel.ENEMY_AGENT_HEALTH]) * AGENT_MAX_HEALTH
