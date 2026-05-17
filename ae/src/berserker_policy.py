"""Reckless base-destroyer policy.

Priority: bomb enemy bases into rubble. No dodging, no defending.
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

from typing import Optional

from constants import Action
from map_memory import MapMemory
from observation import ParsedObs
from pathfinding import first_action_to, from_can_traverse, next_pos_after
from policy import Policy
from threat import cells_in_blast


# Extra ticks to budget for bombing through a destructible wall.
# Dijkstra will naturally prefer going around when the detour is shorter.
_WALL_BREAK_COST = 1.0


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
        # Resource tiles visited this round. tile_contents stores static tile
        # types (not current availability), so collected resources stay tagged
        # as "resource" forever unless we track visits ourselves.
        self._collected: set[tuple[int, int]] = set()

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.step == 0:
            self._collected.clear()
        if obs.frozen_ticks > 0:
            return int(Action.STAY)

        pos = obs.location
        facing = obs.direction
        mask = obs.action_mask

        # Bomb any enemy base or agent currently in blast range.
        if obs.team_bombs > 0 and mask[Action.PLACE_BOMB]:
            blast = cells_in_blast(memory, pos)
            if memory.enemy_bases & blast:
                return int(Action.PLACE_BOMB)
            if set(memory.enemy_agents) & blast:
                return int(Action.PLACE_BOMB)

        # Navigate toward a firing position for the weakest known enemy base.
        target = self._pick_target(memory)
        if target is not None:
            # Firing positions are cells from which a bomb would hit the target.
            # By LOS symmetry, these equal cells_in_blast centred on the base.
            firing_positions = {
                p for p in cells_in_blast(memory, target)
                if memory.in_bounds(p)
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
                        return result

        # Collect a nearby resource tile to restock bombs faster.
        # No wall-breaking for resources — we don't spend bombs on non-critical paths.
        # Mark the current cell as collected if it's a resource tile, so we don't
        # navigate back to it after leaving (tile_contents doesn't clear on pickup).
        if memory.tile_contents.get(pos) == "resource":
            self._collected.add(pos)
        resources = {
            p for p, k in memory.tile_contents.items()
            if k == "resource" and p not in self._collected
        }
        if resources:
            edge_cost = from_can_traverse(memory.passable)
            action = first_action_to(pos, facing, resources, edge_cost, max_cost=15.0)
            if action is not None and action != Action.STAY and mask[int(action)]:
                return int(action)

        # Keep exploring: forward, then turn, then back.
        for fallback in (Action.FORWARD, Action.LEFT, Action.RIGHT, Action.BACKWARD):
            if mask[int(fallback)]:
                return int(fallback)

        return int(Action.STAY)

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
