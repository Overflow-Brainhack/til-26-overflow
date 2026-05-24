"""Reckless base-destroyer policy.

Priority: bomb enemy bases into rubble. Minimal emergency dodging, no defending.
Maximises score ceiling at the cost of consistency — ideal when evaluation
takes the best score across many runs.

Scoring math (from the eval config):
  destroy_enemy_base:  50 pts (flat, per base)      ← main prize
  attack_damage:       20 pts per bomb hit on base
  attack_kill:         15 pts flat per agent killed
  own_base_destroyed: -50 pts penalty               ← we ignore this

5 enemy bases × (5 hits × 20 pts + 50 pts destroy) = 750 pts ceiling.

Key fact: our own bombs cannot harm us (friendly fire is disabled in the
simulator), so we can plant a bomb right next to an enemy base and stand
there without taking damage.
"""

from collections import deque
from typing import Optional

from constants import Action
from map_memory import MapMemory
from observation import ParsedObs
from pathfinding import first_action_to, from_can_traverse, next_pos_after
from policy import Policy
from threat import cells_in_blast, imminent_danger, project_danger


# Extra ticks to budget for bombing through a destructible wall.
# Dijkstra will naturally prefer going around when the detour is shorter.
_WALL_BREAK_COST = 1.0
_RESOURCE_COOLDOWN = 35
_CHURN_WINDOW = 6


class BerserkerPolicy(Policy):
    """Rush enemy bases, spam bombs, break walls when faster. Ignore incoming fire.

    Decision tree (evaluated every tick):
    1. Frozen            → STAY (can't act)
    2. Enemy base in our current blast radius + have bombs → PLACE_BOMB
    3. Enemy agent in our current blast radius + have bombs → PLACE_BOMB
    4. Navigate to firing position for weakest (lowest-HP) enemy base,
       bombing through destructible walls when that shortens the path
    5. Collect nearby resource tile to restock bombs faster
    6. Explore (advance / turn)
    """

    def __init__(self) -> None:
        # Resource tiles visited recently. Tile respawn is roughly 40 steps, so
        # use a shorter suppress window instead of blacklisting a tile forever.
        self._collected: dict[tuple[int, int], int] = {}
        self._history: deque[tuple[tuple[int, int], int]] = deque(
            maxlen=_CHURN_WINDOW
        )

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.step == 0:
            self._collected.clear()
            self._history.clear()
        self._expire_collected(obs.step)
        if obs.frozen_ticks > 0:
            return self._record(obs, int(Action.STAY))

        pos = obs.location
        facing = obs.direction
        mask = obs.action_mask

        emergency = self._emergency_dodge(obs, memory)
        if emergency is not None:
            return self._record(obs, emergency)

        # Bomb any enemy base or agent currently in blast range.
        if obs.team_bombs > 0 and mask[Action.PLACE_BOMB]:
            blast = cells_in_blast(memory, pos)
            if memory.enemy_bases & blast:
                return self._record(obs, int(Action.PLACE_BOMB))
            if set(memory.enemy_agents) & blast:
                return self._record(obs, int(Action.PLACE_BOMB))

        # Navigate toward a firing position for the weakest known enemy base.
        target = self._pick_target(memory)
        if target is not None:
            # Firing positions are cells from which a bomb would hit the target.
            # By LOS symmetry, these equal cells_in_blast centred on the base.
            firing_positions = {
                p for p in cells_in_blast(memory, target)
                if memory.in_bounds(p) and p != target
            }
            if firing_positions:
                edge_cost = self._edge_cost(memory)
                action = first_action_to(pos, facing, firing_positions, edge_cost)
                # Skip STAY: if we're already at a firing position but fell through
                # to here, we have no bombs — fall through to resource collection
                # rather than sitting idle waiting for regeneration.
                if action is not None and action != Action.STAY:
                    result = self._maybe_wall_break(obs, memory, int(action))
                    if mask[result]:
                        return self._finalize_move(obs, memory, result)

        # Collect a nearby resource tile to restock bombs faster.
        # No wall-breaking for resources — we don't spend bombs on non-critical paths.
        # Mark the current cell as collected if it's a resource tile, so we don't
        # navigate back to it after leaving (tile_contents doesn't clear on pickup).
        if memory.tile_contents.get(pos) == "resource":
            self._collected[pos] = int(obs.step)
        resources = {
            p for p, k in memory.tile_contents.items()
            if k == "resource"
            and p != pos
            and not self._resource_recently_collected(p, obs.step)
        }
        if resources:
            edge_cost = from_can_traverse(memory.passable)
            action = first_action_to(pos, facing, resources, edge_cost, max_cost=15.0)
            if action is not None and action != Action.STAY and mask[int(action)]:
                return self._finalize_move(obs, memory, int(action))

        # Keep exploring: forward, then turn, then back.
        for fallback in (Action.FORWARD, Action.LEFT, Action.RIGHT, Action.BACKWARD):
            if mask[int(fallback)]:
                return self._finalize_move(obs, memory, int(fallback))

        return self._record(obs, int(Action.STAY))

    def _record(self, obs: ParsedObs, action: int) -> int:
        self._history.append((obs.location, int(action)))
        return int(action)

    def _expire_collected(self, step: int) -> None:
        now = int(step)
        expired = [
            cell
            for cell, seen_step in self._collected.items()
            if now - seen_step >= _RESOURCE_COOLDOWN
        ]
        for cell in expired:
            del self._collected[cell]

    def _resource_recently_collected(self, cell: tuple[int, int], step: int) -> bool:
        seen_step = self._collected.get(cell)
        if seen_step is None:
            return False
        return int(step) - seen_step < _RESOURCE_COOLDOWN

    def _finalize_move(self, obs: ParsedObs, memory: MapMemory, action: int) -> int:
        if action == int(Action.PLACE_BOMB):
            return self._record(obs, action)

        replacement = None
        if self._enters_danger(obs, memory, action) or self._would_churn(obs, action):
            replacement = self._safe_alternative(obs, memory, avoid_action=action)

        if replacement is not None:
            action = replacement
        return self._record(obs, action)

    def _destination(self, obs: ParsedObs, memory: MapMemory, action: int) -> Optional[tuple[int, int]]:
        if action in (int(Action.FORWARD), int(Action.BACKWARD)):
            dest = next_pos_after(obs.location, obs.direction, action)
            if not memory.in_bounds(dest) or not memory.passable(obs.location, dest):
                return None
            return dest
        return obs.location

    def _enters_danger(self, obs: ParsedObs, memory: MapMemory, action: int) -> bool:
        dest = self._destination(obs, memory, action)
        if dest is None:
            return True
        tick = imminent_danger(memory, dest)
        return tick is not None and tick <= 2

    def _would_churn(self, obs: ParsedObs, action: int) -> bool:
        if action not in (int(Action.FORWARD), int(Action.BACKWARD), int(Action.STAY)):
            return False
        if len(self._history) < _CHURN_WINDOW - 1:
            return False
        dest = next_pos_after(obs.location, obs.direction, action)
        recent_positions = [pos for pos, _ in self._history] + [dest]
        return len(set(recent_positions[-_CHURN_WINDOW:])) <= 2

    def _safe_alternative(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        *,
        avoid_action: int,
    ) -> Optional[int]:
        timeline = project_danger(memory)
        recent_positions = {pos for pos, _ in self._history}
        best: tuple[float, int] | None = None

        for action in (
            Action.LEFT,
            Action.RIGHT,
            Action.FORWARD,
            Action.BACKWARD,
            Action.STAY,
        ):
            idx = int(action)
            if idx == avoid_action or obs.action_mask[idx] != 1:
                continue
            dest = self._destination(obs, memory, idx)
            if dest is None:
                continue
            tick = imminent_danger(memory, dest, timeline)
            if tick is not None and tick <= 2:
                continue

            score = 6.0 if tick is None else float(tick)
            if dest in recent_positions:
                score -= 2.0
            if action in (Action.FORWARD, Action.BACKWARD):
                score += 0.6
            elif action in (Action.LEFT, Action.RIGHT):
                score += 0.3
            else:
                score -= 1.5
            if best is None or score > best[0]:
                best = (score, idx)

        return best[1] if best is not None else None

    def _emergency_dodge(self, obs: ParsedObs, memory: MapMemory) -> Optional[int]:
        timeline = project_danger(memory)
        danger_tick = imminent_danger(memory, obs.location, timeline)
        if danger_tick is None or danger_tick > 2:
            return None

        best: tuple[float, int] | None = None
        for action in (
            Action.FORWARD,
            Action.BACKWARD,
            Action.LEFT,
            Action.RIGHT,
            Action.STAY,
        ):
            idx = int(action)
            if obs.action_mask[idx] != 1:
                continue
            dest = obs.location
            if action in (Action.FORWARD, Action.BACKWARD):
                dest = next_pos_after(obs.location, obs.direction, idx)
                if not memory.in_bounds(dest) or not memory.passable(obs.location, dest):
                    continue
            tick = imminent_danger(memory, dest, timeline)
            if tick is not None and tick <= 2:
                continue
            move_bonus = 1.0 if action in (Action.FORWARD, Action.BACKWARD) else 0.0
            score = (10.0 if tick is None else float(tick)) + move_bonus
            if best is None or score > best[0]:
                best = (score, idx)
        return best[1] if best is not None else None

    def _edge_cost(self, memory: MapMemory):
        """EdgeCost that allows traversal through destructible walls at extra cost."""
        def cost(a: tuple[int, int], b: tuple[int, int]) -> Optional[float]:
            if not memory.in_bounds(b):
                return None
            if memory.passable(a, b):
                return 1.0
            if memory.edge_is_destructible_wall(a, b):
                return _WALL_BREAK_COST
            return None
        return cost

    def _maybe_wall_break(self, obs: ParsedObs, memory: MapMemory, action: int) -> int:
        """If the planned move crosses a destructible wall, place a bomb to clear it.

        Returns PLACE_BOMB when we need to blast through, STAY when our own bomb
        is already in progress, or the original action when the path is clear.
        """
        if action not in (int(Action.FORWARD), int(Action.BACKWARD)):
            return action
        next_pos = next_pos_after(obs.location, obs.direction, action)
        if not memory.edge_is_destructible_wall(obs.location, next_pos):
            return action
        # Already have an ally bomb here — wait for it to detonate the wall.
        sitting_bomb = memory.bombs.get(obs.location)
        if sitting_bomb is not None and sitting_bomb.ally:
            return int(Action.STAY)
        if obs.action_mask[Action.PLACE_BOMB] == 1 and obs.team_bombs > 0:
            return int(Action.PLACE_BOMB)
        return action

    def _pick_target(self, memory: MapMemory) -> Optional[tuple[int, int]]:
        """Weakest (lowest HP) known enemy base, or None if none remain.

        Bases observed at exactly 0 HP are treated as destroyed and excluded —
        the simulator may keep them visible with health_ratio=0 for one tick
        before the entity is fully removed from the observation.
        """
        bases = {
            b for b in memory.enemy_bases
            if memory.enemy_base_health.get(b, 100.0) > 0
        }
        if not bases:
            return None
        return min(bases, key=lambda b: memory.enemy_base_health.get(b, 100.0))
