"""PettingZoo environment wrapper.

EnvWrapper owns the env instance and exposes the subset of the PettingZoo
API the match loop needs.
"""

import logging
from typing import Any

import aiohttp
from aiohttp import ClientConnectorError
from til_environment.bomberman_env import parallel_basic_env
from til_environment.config import default_config

logger = logging.getLogger("uvicorn.error")


class EnvWrapper:
    """Thin wrapper around the PettingZoo parallel env.

    Owns the env, the wall-edge key set (for wallsDestroyed diff), and the
    robot-command HTTP client.  Business logic (scoring, missions, actions)
    stays outside.
    """

    AUTONOMY_URL = "http://host.docker.internal:3008/robot_goto"

    def __init__(self, track: str, team_names: list[str]) -> None:
        env_cfg = default_config()
        env_cfg.env.novice = track == "novice"
        env_cfg.env.num_teams = len(team_names)
        env_cfg.env.render_mode = "rgb_array"
        self.env = parallel_basic_env(cfg=env_cfg, env_wrappers=[])
        self._team_names = team_names
        self._wall_keys: set = set()
        self._session: aiohttp.ClientSession | None = None
        self._autonomy_unreachable_logged = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    def ensure_session(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def reset(self, seed: int) -> tuple[dict, dict]:
        obs, info = self.env.reset(seed=seed)
        self._wall_keys = set(self._benv.dynamics.arena_state.wall_edges.keys())
        return obs, info

    def step(self, actions: dict[str, int]):
        return self.env.step(actions)

    def render(self) -> Any:
        return self.env.render()

    # ── env internals ─────────────────────────────────────────────────────

    @property
    def _benv(self):
        return self.env.aec_env.unwrapped

    def agent_entity(self, agent_id: str):
        env = self.env.aec_env
        while hasattr(env, "env"):
            env = env.env
        return env.dynamics.registry.get(agent_id)

    @property
    def cfg(self):
        return self._benv.cfg

    # ── robot command ─────────────────────────────────────────────────────

    async def send_robot_command(self, team_agent_mapping: dict[str, str]) -> None:
        self.ensure_session()
        assert self._session is not None
        try:
            data = {}
            for agent_id in team_agent_mapping.values():
                ent = self.agent_entity(agent_id)
                if ent is None:
                    continue
                pos = ent.position
                dir_ = (4 - int(ent.direction)) % 4
                data[agent_id.split("_", 1)[1]] = {
                    "x": int(pos[0]),
                    "y": int(pos[1]),
                    "dir": dir_,
                }
            async with self._session.post(self.AUTONOMY_URL, json=data) as response:
                response.raise_for_status()
        except ClientConnectorError:
            if not self._autonomy_unreachable_logged:
                logger.warning(
                    f"Unable to connect to {self.AUTONOMY_URL}; skipping "
                    f"physical-robot updates (harmless for software-only tests)"
                )
                self._autonomy_unreachable_logged = True
        except Exception as e:
            logger.exception(e)
