"""Manages the AE model.

Per-round entrypoint. The server recreates this on `/reset` (and implicitly
on observations with `step == 0`). Static map knowledge is shared across
recreations via the module-level singleton in `map_memory`, so novice mode
(fixed map) doesn't re-explore each round.

Optional novice-map cache: if `ae/src/novice_map.json` exists, its static
state (walls, tile types, base positions) is merged into the singleton on
first construction. Bundle a captured cache via the Dockerfile's
`COPY src .` to start round 1 with full map knowledge.
"""

from pathlib import Path
from typing import Any, Optional

from constants import Action
from map_memory import MapMemory, get_shared_memory
from observation import parse_observation
from policy import HeuristicPolicy, Policy


# Default cache path: bundled into the Docker image alongside source.
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "novice_map.json"


class AEManager:
    def __init__(
        self,
        policy: Policy | None = None,
        memory: MapMemory | None = None,
        cache_path: Optional[Path] = None,
    ) -> None:
        # Production (single-bot Docker): use the singleton so static state
        # survives /reset across rounds.
        # Multi-agent visualization: pass an isolated MapMemory per bot.
        self._memory = memory if memory is not None else get_shared_memory()

        # Re-merge the cache on every construction (i.e. every /reset). The
        # merge is idempotent for the data structures we care about, and
        # crucially restores walls that were destroyed in the previous round
        # (novice mode resets the env's walls but our `blocked_edges` only
        # learns about that when we re-observe). Skipped when the caller
        # supplied their own MapMemory — they control preloading.
        if memory is None:
            self._maybe_load_cache(cache_path or DEFAULT_CACHE_PATH)

        self._memory.reset_round()
        self._policy: Policy = policy or HeuristicPolicy(
            predictive_bomb=True,
            predictive_bomb_threshold=0.7,

            wall_breaking=True,
            wall_break_cost=5.0,

            smart_defend=True,

            drift_aware_bomb=True,
            auto_tune_bomb=True,

            bomb_economy=True,
            base_bomb_value=5.0,
            agent_bomb_value=1.0,
            bomb_reserve_threshold=0.5,
            wall_break_tile_threshold=0.0,

            loop_detection=True,
        )

    def _maybe_load_cache(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            cached = MapMemory.load(path)
        except Exception:
            # Cache is corrupt or stale — silently ignore.
            return
        self._memory.merge_static_from(cached)

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
