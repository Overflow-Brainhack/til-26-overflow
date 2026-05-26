"""Experimental clone of EditedHeuristicPolicy with the handover ideas.

This subclasses the production `EditedHeuristicPolicy` and layers on
independent behaviours behind constructor toggles. Dodge stays the untouchable
first priority and `_try_defend` remains disabled, so these changes do not alter
how the agent escapes real danger.

New toggles
-----------
contested_route_penalty   — discount collectible-tile scores when the tile sits
    near a recent enemy sighting, so the agent doesn't chase a low-value tile
    into a hot zone.  Enemy bases are deliberately exempt: attacking them is
    high-value and low-risk, so we never discourage base routes.  (idea #4)

enemy_pressure_model      — aggregate stale enemy sightings into a lightweight
    probability map. It propagates mass through passable neighbors, biases
    transitions by observed velocity and movement toward our base, and removes
    mass from cells currently visible with no enemy. Used for predictive bomb
    scoring, and for contested-route scoring when that toggle is enabled.
    Default OFF until a sweep shows it beats the drift-aware baseline.

Dodge variants
--------------
Two dodge implementations live here, dispatched by `hardened_dodge`:

* `_dodge_v1` — the inherited behaviour: flee to the nearest cell that is safe
  for the whole fuse window.
* `_dodge_v2` — additionally avoids fleeing into a cell a nearby enemy could
  immediately re-bomb.  Strictly safer, marginally slower escapes.

`hardened_dodge` (default OFF) selects v2; otherwise v1 runs.
"""

from typing import Optional

from constants import BOMB_TIMER, GRID_SIZE, ViewChannel
from edited_policy import EXPLORE_BUDGET, EditedHeuristicPolicy
from map_memory import MapMemory
from observation import ParsedObs, base_view_to_world, view_to_world
from pathfinding import (
    first_action_to,
    reachable_cells,
    temporal_first_action_to,
)
from threat import cells_in_blast, cells_safe_for_at_least

_PRESSURE_STEPS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


class EditedHeuristicPolicyV2(EditedHeuristicPolicy):
    def __init__(
        self,
        *,
        contested_route_penalty: bool = False,  # dont know if this helps
        contested_radius: int = 3,
        contested_min_factor: float = 0.3,
        enemy_pressure_model: bool = False,
        enemy_pressure_max_age: int = 8,
        enemy_pressure_velocity_bias: float = 2.0,
        enemy_pressure_base_bias: float = 0.5,
        hardened_dodge: bool = False,  # cant tell if this does anything either
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.contested_route_penalty = contested_route_penalty
        self.contested_radius = contested_radius
        self.contested_min_factor = contested_min_factor
        self.enemy_pressure_model = enemy_pressure_model
        self.enemy_pressure_max_age = enemy_pressure_max_age
        self.enemy_pressure_velocity_bias = enemy_pressure_velocity_bias
        self.enemy_pressure_base_bias = enemy_pressure_base_bias
        self.hardened_dodge = hardened_dodge
        self._pressure_obs: Optional[ParsedObs] = None
        self._pressure_cache: dict[int, dict[tuple[int, int], float]] = {}

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        self._pressure_obs = obs
        self._pressure_cache = {}
        try:
            return super().choose(obs, memory)
        finally:
            self._pressure_obs = None
            self._pressure_cache = {}

    # ── small geometry helper ────────────────────────────────────────────────
    @staticmethod
    def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    # ── dodge: two variants, dispatched by the toggle ────────────────────────
    def _dodge(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        timeline: dict[int, set[tuple[int, int]]],
    ) -> Optional[int]:
        if self.hardened_dodge:
            return self._dodge_v2(obs, memory, timeline)
        return self._dodge_v1(obs, memory, timeline)

    def _dodge_v1(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        timeline: dict[int, set[tuple[int, int]]],
    ) -> Optional[int]:
        """Inherited behaviour: flee to the nearest fully-safe cell."""
        edge = self._edge_cost(memory, allow_walls=False)
        safe = cells_safe_for_at_least(memory, BOMB_TIMER + 1, timeline)
        if not safe:
            return self._panic_move(obs, memory, timeline)
        action = temporal_first_action_to(
            obs.location, obs.direction, safe, edge, timeline
        )
        if action is not None:
            return int(action)
        return self._panic_move(obs, memory, timeline)

    def _dodge_v2(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        timeline: dict[int, set[tuple[int, int]]],
    ) -> Optional[int]:
        """Hardened: also avoid fleeing into a nearby enemy's potential blast.

        We subtract every known enemy's current blast footprint from the safe
        set so the escape doesn't land us where an adjacent enemy can re-bomb
        us next tick.  If that empties the set (tight/contested map), we fall
        back to the plain safe set rather than strand the agent.
        """
        edge = self._edge_cost(memory, allow_walls=False)
        safe = cells_safe_for_at_least(memory, BOMB_TIMER + 1, timeline)

        potential: set[tuple[int, int]] = set()
        for epos in memory.enemy_agents:
            potential |= cells_in_blast(memory, epos)
        target_set = (safe - potential) or safe

        if not target_set:
            return self._panic_move(obs, memory, timeline)
        action = temporal_first_action_to(
            obs.location, obs.direction, target_set, edge, timeline
        )
        if action is not None:
            return int(action)
        return self._panic_move(obs, memory, timeline)

    # ── idea #4: contested-route penalty (tiles only) ────────────────────────
    def _contested_factor(self, cell: tuple[int, int], memory: MapMemory) -> float:
        if not self.contested_route_penalty or not memory.enemy_agents:
            return 1.0
        pressure = self._pressure_near(cell, memory)
        if pressure > 0.0:
            influence = min(1.0, pressure)
            return 1.0 - (1.0 - self.contested_min_factor) * influence
        d = min(self._chebyshev(cell, e) for e in memory.enemy_agents)
        if d >= self.contested_radius:
            return 1.0
        # Linearly fade from contested_min_factor (on top of the enemy) up to
        # 1.0 at contested_radius.
        return self.contested_min_factor + (1.0 - self.contested_min_factor) * (
            d / self.contested_radius
        )

    # ── collect: firing-cell base routing + contested penalty + filter ───────
    def _try_collect(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
    ) -> Optional[int]:
        candidates = [
            cell
            for cell in memory.collectible_cells()
            if cell != obs.location and not self._tile_recently_collected(cell, obs.step)
        ]
        base_tiles: list[tuple[int, int]] = []
        if self.proactive_base_routing:
            base_tiles = [p for p in memory.enemy_bases if p != obs.location]

        if not candidates and not base_tiles:
            return None

        edge = self._edge_cost(memory, danger_avoid=danger_now)
        distances = reachable_cells(
            obs.location, obs.direction, edge, max_cost=EXPLORE_BUDGET
        )

        best_score = 0.0
        best_cell: Optional[tuple[int, int]] = None

        # Collectible tiles — contested penalty applies here.
        for cell in candidates:
            dist = distances.get(cell)
            if dist is None:
                continue
            value = memory.tile_value(cell)
            if value <= 0:
                continue
            score = value / (dist + 1.0) * self._contested_factor(cell, memory)
            if score > best_score:
                best_score = score
                best_cell = cell

        # Enemy bases — full weight, never contested-penalised.
        if self.proactive_base_routing:
            base_w = self._effective_base_weight()
            for cell in base_tiles:
                dist = distances.get(cell)
                if dist is None:
                    continue
                score = base_w / (dist + 1.0)
                if score > best_score:
                    best_score = score
                    best_cell = cell

        if best_cell is None:
            return None

        self._debug_target = best_cell
        action = first_action_to(obs.location, obs.direction, {best_cell}, edge)
        if action is None:
            return None
        action = self._maybe_wall_break(obs, memory, action)
        return action

    # ── lightweight enemy-pressure model ────────────────────────────────────
    def _expected_hits(self, memory: MapMemory, blast: set[tuple[int, int]]) -> float:
        if not self.enemy_pressure_model or self._pressure_obs is None:
            return super()._expected_hits(memory, blast)

        pressure = self._enemy_pressure(memory, BOMB_TIMER)
        total = 0.0
        for cell, mass in pressure.items():
            if cell in blast:
                total += mass * self._pressure_escape_factor(memory, cell, blast)
        return total

    def _pressure_near(self, cell: tuple[int, int], memory: MapMemory) -> float:
        if not self.enemy_pressure_model or self._pressure_obs is None:
            return 0.0

        pressure = self._enemy_pressure(memory, BOMB_TIMER)
        return sum(
            mass
            for p, mass in pressure.items()
            if self._chebyshev(cell, p) <= self.contested_radius
        )

    def _enemy_pressure(
        self,
        memory: MapMemory,
        horizon: int,
    ) -> dict[tuple[int, int], float]:
        cached = self._pressure_cache.get(horizon)
        if cached is not None:
            return cached

        obs = self._pressure_obs
        if obs is None:
            return {}

        visible, visible_enemies = self._visible_cells(obs)
        pressure: dict[tuple[int, int], float] = {}
        for source, seen_step in self._pressure_sources(memory):
            age = max(0, memory.current_step - seen_step)
            vel = self._pressure_velocity(memory, source, seen_step)
            dist = {source: 1.0}

            for _ in range(age):
                dist = self._propagate_pressure(dist, memory, vel)
                if not dist:
                    break
            if not dist:
                continue

            dist = self._apply_visibility_correction(
                dist,
                visible,
                visible_enemies,
            )
            if not dist:
                continue

            for _ in range(horizon):
                dist = self._propagate_pressure(dist, memory, vel)
                if not dist:
                    break

            for cell, mass in dist.items():
                pressure[cell] = pressure.get(cell, 0.0) + mass

        self._pressure_cache[horizon] = pressure
        return pressure

    def _pressure_sources(
        self,
        memory: MapMemory,
    ) -> list[tuple[tuple[int, int], int]]:
        out: list[tuple[tuple[int, int], int]] = []
        sightings = sorted(
            memory.enemy_agents.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for pos, seen_step in sightings:
            age = memory.current_step - seen_step
            if age < 0 or age > self.enemy_pressure_max_age:
                continue

            # MapMemory keeps old positions until TTL. Suppress an older sighting
            # when a newer one could already explain it, so one moving enemy does
            # not leave a multi-agent trail.
            explained = False
            for newer_pos, newer_step in out:
                dt = newer_step - seen_step
                if dt < 0:
                    continue
                if self._manhattan(pos, newer_pos) <= dt:
                    explained = True
                    break
            if not explained:
                out.append((pos, seen_step))
        return out

    def _pressure_velocity(
        self,
        memory: MapMemory,
        source: tuple[int, int],
        seen_step: int,
    ) -> tuple[int, int]:
        vel = memory.enemy_velocities.get(source)
        if vel is not None:
            return vel

        prev_step = seen_step - 1
        best_prev: Optional[tuple[int, int]] = None
        best_dist = 2
        for pos, step in memory.enemy_agents.items():
            if step != prev_step:
                continue
            dist = self._manhattan(pos, source)
            if dist < best_dist:
                best_dist = dist
                best_prev = pos
        if best_prev is None:
            return (0, 0)
        return (source[0] - best_prev[0], source[1] - best_prev[1])

    def _propagate_pressure(
        self,
        dist: dict[tuple[int, int], float],
        memory: MapMemory,
        vel: tuple[int, int],
    ) -> dict[tuple[int, int], float]:
        out: dict[tuple[int, int], float] = {}
        for pos, mass in dist.items():
            weighted: list[tuple[tuple[int, int], float]] = []
            for dx, dy in _PRESSURE_STEPS:
                nxt = (pos[0] + dx, pos[1] + dy)
                if not memory.in_bounds(nxt):
                    continue
                if nxt != pos and not memory.passable(pos, nxt):
                    continue
                weighted.append(
                    (nxt, self._pressure_step_weight(pos, nxt, vel, memory))
                )

            total_weight = sum(weight for _, weight in weighted)
            if total_weight <= 0.0:
                continue
            for nxt, weight in weighted:
                out[nxt] = out.get(nxt, 0.0) + mass * weight / total_weight
        return out

    def _pressure_step_weight(
        self,
        pos: tuple[int, int],
        nxt: tuple[int, int],
        vel: tuple[int, int],
        memory: MapMemory,
    ) -> float:
        if nxt == pos:
            return 0.75

        step = (nxt[0] - pos[0], nxt[1] - pos[1])
        weight = 1.0

        vx, vy = vel
        if vel != (0, 0):
            dot = step[0] * vx + step[1] * vy
            if dot > 0:
                weight += self.enemy_pressure_velocity_bias * dot
            elif dot < 0:
                weight /= 1.0 + self.enemy_pressure_velocity_bias * abs(dot)

        if memory.ally_base is not None:
            old_d = self._manhattan(pos, memory.ally_base)
            new_d = self._manhattan(nxt, memory.ally_base)
            if new_d < old_d:
                weight += self.enemy_pressure_base_bias
            elif new_d > old_d:
                weight /= 1.0 + self.enemy_pressure_base_bias

        return max(0.05, weight)

    def _apply_visibility_correction(
        self,
        dist: dict[tuple[int, int], float],
        visible: set[tuple[int, int]],
        visible_enemies: set[tuple[int, int]],
    ) -> dict[tuple[int, int], float]:
        if not visible:
            return dist

        filtered = {
            cell: mass
            for cell, mass in dist.items()
            if cell not in visible or cell in visible_enemies
        }
        total = sum(filtered.values())
        if total <= 0.0:
            return {}
        return {cell: mass / total for cell, mass in filtered.items()}

    def _visible_cells(
        self,
        obs: ParsedObs,
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        visible: set[tuple[int, int]] = set()
        enemies: set[tuple[int, int]] = set()

        self._collect_visible_from_view(
            obs.agent_view,
            lambda r, c: view_to_world(obs.location, obs.direction, r, c),
            visible,
            enemies,
        )
        self._collect_visible_from_view(
            obs.base_view,
            lambda r, c: base_view_to_world(obs.base_location, r, c),
            visible,
            enemies,
        )
        return visible, enemies

    def _collect_visible_from_view(
        self,
        view,
        world_for_cell,
        visible: set[tuple[int, int]],
        enemies: set[tuple[int, int]],
    ) -> None:
        rows, cols = view.shape[:2]
        for r in range(rows):
            for c in range(cols):
                cell_view = view[r, c]
                if cell_view[ViewChannel.VISIBLE] < 0.5:
                    continue
                pos = world_for_cell(r, c)
                if not (0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE):
                    continue
                visible.add(pos)
                if cell_view[ViewChannel.ENEMY_AGENT] > 0.5:
                    enemies.add(pos)

    def _pressure_escape_factor(
        self,
        memory: MapMemory,
        cell: tuple[int, int],
        blast: set[tuple[int, int]],
    ) -> float:
        exits = 0
        for dx, dy in _PRESSURE_STEPS[1:]:
            nbr = (cell[0] + dx, cell[1] + dy)
            if (
                memory.in_bounds(nbr)
                and memory.passable(cell, nbr)
                and nbr not in blast
            ):
                exits += 1
        return 1.0 / (1.0 + exits)

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
