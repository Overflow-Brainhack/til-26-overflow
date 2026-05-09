"""Manages the AE model.

Per-round entrypoint. The server recreates this on `/reset` (and implicitly
on observations with `step == 0`). Static map knowledge is shared across
recreations via the module-level singleton in `map_memory`, so novice mode
(fixed map) doesn't re-explore each round.
"""

from typing import Any

from constants import Action
from map_memory import MapMemory, get_shared_memory
from observation import parse_observation
from policy import HeuristicPolicy, Policy


class AEManager:
    def __init__(
        self,
        policy: Policy | None = None,
        memory: MapMemory | None = None,
    ) -> None:
        # Production (Docker, single agent): use the module-level singleton so
        # static map knowledge survives /reset across rounds.
        # Multi-agent visualization: pass an isolated MapMemory per bot.
        self._memory = memory if memory is not None else get_shared_memory()
        self._memory.reset_round()
        self._policy: Policy = policy or HeuristicPolicy()

    def ae(self, observation: dict[str, Any]) -> int:
        """Choose the next action given the current observation.

        Args:
            observation: see `ae/README.md` for the schema. Note: the README
                shows the legacy TIL-25 format; the live env produces the
                richer TIL-26 dict (agent_viewcone, base_viewcone, etc.) which
                this manager parses.

        Returns:
            An integer in [0, 5]: see Action enum in constants.py.
        """
        try:
            obs = parse_observation(observation)
            self._memory.update(obs)
            return int(self._policy.choose(obs, self._memory))
        except Exception:
            # Never crash the server — losing one tick to STAY is better than
            # 500ing out of the round.
            return int(Action.STAY)
