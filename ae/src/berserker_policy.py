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
from pathfinding import first_action_to, from_can_traverse
from policy import Policy
from threat import cells_in_blast


class BerserkerPolicy(Policy):
    """Rush enemy bases, spam bombs, ignore incoming fire.

    Decision tree (evaluated every tick):
    1. Frozen            → STAY (can't act)
    2. Enemy base in our current blast radius + have bombs → PLACE_BOMB
    3. Enemy agent in our current blast radius + have bombs → PLACE_BOMB
    4. Navigate to firing position for weakest (lowest-HP) enemy base
    5. Collect nearby resource tile to restock bombs faster
    6. Explore (advance / turn)
    """

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
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
                edge_cost = from_can_traverse(memory.passable)
                action = first_action_to(pos, facing, firing_positions, edge_cost)
                if action is not None and mask[int(action)]:
                    return int(action)

        # Collect a nearby resource tile to restock bombs faster.
        resources = {p for p, k in memory.tile_contents.items() if k == "resource"}
        if resources:
            edge_cost = from_can_traverse(memory.passable)
            action = first_action_to(pos, facing, resources, edge_cost, max_cost=15.0)
            if action is not None and mask[int(action)]:
                return int(action)

        # Keep exploring: forward, then turn, then back.
        for fallback in (Action.FORWARD, Action.LEFT, Action.RIGHT, Action.BACKWARD):
            if mask[int(fallback)]:
                return int(fallback)

        return int(Action.STAY)

    def _pick_target(self, memory: MapMemory) -> Optional[tuple[int, int]]:
        """Weakest (lowest HP) known enemy base, or None if none discovered."""
        bases = memory.enemy_bases
        if not bases:
            return None
        return min(bases, key=lambda b: memory.enemy_base_health.get(b, 100.0))
