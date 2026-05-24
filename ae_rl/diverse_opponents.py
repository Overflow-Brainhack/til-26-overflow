"""Training-only opponent policies.

These exist to widen the league opponent pool beyond the
``EditedHeuristicPolicyV2`` family. The RL learner only ever sees opponents
derived from one heuristic, so it overfits to that heuristic's specific
quirks — exploits which do not transfer to the eval reference policy or to
finals opponents.

Each policy implements the same ``Policy`` interface as
``ae/src/policy.py``, so they plug into ``AEManager`` and the existing
``HeuristicController`` pattern without changes elsewhere.

Living in ``ae_rl/`` rather than ``ae/src/`` keeps them out of the deploy
Docker image — they have no role in production inference.
"""

from __future__ import annotations

import random
from typing import Optional

import common  # noqa: F401  (path bootstrap)
from constants import AGENT_MAX_HEALTH, Action, BOMB_TIMER, DIR_VECTOR
from map_memory import MapMemory
from observation import ParsedObs
from pathfinding import first_action_to, from_can_traverse, next_pos_after
from policy import Policy
from threat import cells_in_blast, imminent_danger, project_danger


def _legal_actions(obs: ParsedObs) -> list[int]:
    return [i for i, ok in enumerate(obs.action_mask) if ok]


class RandomLegalPolicy(Policy):
    """Uniform random over the legal action mask.

    Trains the RL not to assume opponent rationality — useful both as a
    smoke-test opponent and as a generalisation signal (real eval opponents
    sometimes do things no heuristic would).
    """

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.frozen_ticks > 0:
            return int(Action.STAY)
        legal = _legal_actions(obs)
        if not legal:
            return int(Action.STAY)
        return int(random.choice(legal))


class PureCollectorPolicy(Policy):
    """Navigate to visible collectibles. Never bombs. Never engages.

    The RL has only seen aggressive heuristic opponents; if the eval reference
    is a passive collector, the RL may misread it as a threat and waste bombs
    chasing it. Training against this teaches "ignore non-aggressive agents."
    """

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.frozen_ticks > 0:
            return int(Action.STAY)
        targets = memory.collectible_cells()
        if targets:
            edge_cost = from_can_traverse(memory.passable)
            action = first_action_to(
                obs.location, obs.direction, set(targets), edge_cost, max_cost=20.0
            )
            if action is not None and obs.action_mask[int(action)]:
                return int(action)
        # No reachable tile — wander forward, else turn, else STAY.
        for fallback in (Action.FORWARD, Action.LEFT, Action.RIGHT, Action.BACKWARD):
            if obs.action_mask[int(fallback)]:
                return int(fallback)
        return int(Action.STAY)


class IdlePolicy(Policy):
    """Stay still most of the time; occasionally turn.

    Approximates an opponent slot that's effectively empty — tests whether
    the RL can keep racking up score when a competitor is no threat at all.
    Also exercises edge cases in the RL's relative-position logic.
    """

    TURN_PROB = 0.10

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.frozen_ticks > 0:
            return int(Action.STAY)
        if random.random() < self.TURN_PROB:
            for action in (Action.LEFT, Action.RIGHT):
                if obs.action_mask[int(action)]:
                    return int(action)
        return int(Action.STAY) if obs.action_mask[int(Action.STAY)] else int(Action.STAY)


class TrapSetterPolicy(Policy):
    """Wander a bit, then drop bombs on whatever cell we currently stand on.

    Bombs go down with no regard for tactical value — purely to litter the map
    with hazards. Forces the RL to handle an environment in which any cell may
    contain a freshly placed bomb at any time, not just cells near enemies.
    """

    BOMB_PROB = 0.25  # per turn when we have a bomb available

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.frozen_ticks > 0:
            return int(Action.STAY)
        if (
            obs.team_bombs > 0
            and obs.action_mask[int(Action.PLACE_BOMB)]
            and random.random() < self.BOMB_PROB
        ):
            return int(Action.PLACE_BOMB)
        # Otherwise wander forward, falling back to turn / back / STAY.
        for fallback in (Action.FORWARD, Action.LEFT, Action.RIGHT, Action.BACKWARD):
            if obs.action_mask[int(fallback)]:
                return int(fallback)
        return int(Action.STAY)


class PatrollerPolicy(Policy):
    """Walk a fixed loop of headings — FORWARD until blocked, then turn LEFT.

    Predictable movement pattern with zero strategy. Tests the RL against an
    opponent that doesn't react to its presence at all — the kind of stubborn,
    non-adversarial movement a poorly-trained competitor bot might produce.
    """

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.frozen_ticks > 0:
            return int(Action.STAY)
        if obs.action_mask[int(Action.FORWARD)]:
            return int(Action.FORWARD)
        if obs.action_mask[int(Action.LEFT)]:
            return int(Action.LEFT)
        if obs.action_mask[int(Action.RIGHT)]:
            return int(Action.RIGHT)
        if obs.action_mask[int(Action.BACKWARD)]:
            return int(Action.BACKWARD)
        return int(Action.STAY)


class KamikazePolicy(Policy):
    """When low HP or near an enemy, drop a bomb at our feet and STAY.

    Otherwise wander toward any visible enemy agent. Models the "desperate
    end-game" opponent that trades its life for damage — a behaviour the RL
    has never seen because EditedHeuristicPolicyV2 always values its own
    survival.
    """

    LOW_HP_THRESHOLD = 0.4  # fraction of max HP

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.frozen_ticks > 0:
            return int(Action.STAY)

        # Detonate if low HP or any enemy is in our immediate neighbourhood.
        low_hp = obs.health < self.LOW_HP_THRESHOLD * AGENT_MAX_HEALTH
        enemy_near = False
        for ex, ey in memory.enemy_agents:
            if abs(ex - obs.location[0]) + abs(ey - obs.location[1]) <= 2:
                enemy_near = True
                break
        if (
            obs.team_bombs > 0
            and obs.action_mask[int(Action.PLACE_BOMB)]
            and (low_hp or enemy_near)
        ):
            return int(Action.PLACE_BOMB)

        # Otherwise chase nearest known enemy (or wander if none seen).
        if memory.enemy_agents:
            target = min(
                memory.enemy_agents,
                key=lambda p: abs(p[0] - obs.location[0]) + abs(p[1] - obs.location[1]),
            )
            edge_cost = from_can_traverse(memory.passable)
            action = first_action_to(
                obs.location, obs.direction, {target}, edge_cost, max_cost=20.0
            )
            if action is not None and obs.action_mask[int(action)]:
                return int(action)
        for fallback in (Action.FORWARD, Action.LEFT, Action.RIGHT, Action.BACKWARD):
            if obs.action_mask[int(fallback)]:
                return int(fallback)
        return int(Action.STAY)


class TacticalPolicy(Policy):
    """1-step lookahead opponent. Scores each legal action by projecting one
    step forward and evaluating the resulting state with a hand-tuned feature
    combo (tile reward, distance to nearest collectible, blast safety, enemy
    proximity, bomb opportunity), then picks the highest-scoring action.

    Structurally different from ``EditedHeuristicPolicyV2``:
      - No Dijkstra pathfinding; uses 1-step lookahead instead.
      - No temporal danger projection; uses current-step blast zones only.
      - No bomb economy; bombs whenever it would hit an enemy base.

    The RL's heuristic-specific exploits (predicting EditedHeuristicPolicyV2's
    exact dodge / bomb-economy decisions) don't transfer to this opponent, so
    training against it forces broader generalisation.
    """

    # Score weights — hand-tuned to produce coherent behaviour.
    TILE_REWARD = 50.0           # value of stepping onto a known collectible
    NEAREST_TILE_PENALTY = 0.5   # cost per Manhattan step to the nearest known collectible
    BLAST_PENALTY = -200.0       # standing in any current-tick blast
    ENEMY_ADJACENT_PENALTY = -10.0  # within Manhattan-2 of a known enemy
    BASE_BOMB_REWARD = 30.0      # PLACE_BOMB when an enemy base sits inside our blast
    STAY_PENALTY = -3.0          # discourage idling
    INVALID = float("-inf")

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        if obs.frozen_ticks > 0:
            return int(Action.STAY)

        legal = [i for i, ok in enumerate(obs.action_mask) if ok]
        if not legal:
            return int(Action.STAY)

        # Snapshot current-tick blast cells (any bomb that fires next tick).
        try:
            danger_now: set[tuple[int, int]] = set()
            for bomb_pos in memory.bombs:
                danger_now |= cells_in_blast(memory, bomb_pos)
        except Exception:
            danger_now = set()

        collectibles = memory.collectible_cells()

        best_score = self.INVALID
        best_action = legal[0]
        for action in legal:
            score = self._score_action(
                action, obs, memory, danger_now, collectibles
            )
            if score > best_score:
                best_score = score
                best_action = action

        return int(best_action)

    def _score_action(
        self,
        action: int,
        obs: ParsedObs,
        memory: MapMemory,
        danger_now: set[tuple[int, int]],
        collectibles: list[tuple[int, int]],
    ) -> float:
        # Predict the position the agent will occupy AFTER this action.
        pos = obs.location
        if action == int(Action.STAY):
            new_pos = pos
            score = self.STAY_PENALTY
        elif action == int(Action.PLACE_BOMB):
            new_pos = pos  # bombing keeps us in place
            score = 0.0
            # Big bonus if this bomb hits an enemy base.
            try:
                blast = cells_in_blast(memory, pos)
                if any(b in blast for b in memory.enemy_bases):
                    score += self.BASE_BOMB_REWARD
            except Exception:
                pass
        elif action in (int(Action.FORWARD), int(Action.BACKWARD),
                        int(Action.LEFT), int(Action.RIGHT)):
            try:
                new_pos = next_pos_after(pos, obs.direction, action)
            except Exception:
                return self.INVALID
            if not memory.in_bounds(new_pos):
                return self.INVALID
            score = 0.0
        else:
            return self.INVALID

        # Blast safety — standing in a soon-to-explode cell is almost always wrong.
        if new_pos in danger_now:
            score += self.BLAST_PENALTY

        # Tile pickup.
        if memory.tile_contents.get(new_pos) in ("mission", "resource", "recon"):
            score += self.TILE_REWARD

        # Distance to nearest known collectible — fewer steps = better.
        if collectibles:
            nearest = min(
                abs(c[0] - new_pos[0]) + abs(c[1] - new_pos[1])
                for c in collectibles
            )
            score -= self.NEAREST_TILE_PENALTY * nearest

        # Avoid sitting right next to enemies (they may bomb us).
        for ex, ey in memory.enemy_agents:
            if abs(ex - new_pos[0]) + abs(ey - new_pos[1]) <= 2:
                score += self.ENEMY_ADJACENT_PENALTY
                break

        return score
