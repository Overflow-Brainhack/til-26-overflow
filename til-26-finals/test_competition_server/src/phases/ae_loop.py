"""Bomberman AE loop.

AELoop owns the step-by-step PettingZoo control: collect actions, advance
env, trigger missions, log, broadcast, finalize.  It holds no scoring or
batch-processing logic — those live in MissionQueue.

The loop runs until auto_step is False or the env signals done.
"""

import asyncio
import logging
from time import time
from typing import Any, Callable

import constants
from artifacts import EventLog
from env_state import EnvWrapper
from missions import MissionQueue
from render_match_video import render_match_video
from til_environment.actions import Action
from transport import WebSocketManager

logger = logging.getLogger("uvicorn.error")

DEFAULT_ACTION = Action.STAY.value

# A single step's failure must not end the match — teams would be penalised for
# a transient server-side glitch. The critical path (collect actions + advance
# the env) is retried on the next iteration; the match only aborts after this
# many CONSECUTIVE critical-path failures (a successful step resets the count).
MAX_CONSECUTIVE_STEP_ERRORS = 10


class AELoop:
    """Drives the Bomberman environment step-by-step.

    Shared state accessed by the WS handler:
      record_action(team, step, action)  — called on incoming "ae" messages
      step_num                           — current step (read-only from outside)
      ae_leaderboard                     — accumulated rewards per team
      auto_step                          — set False to stop the loop
    """

    def __init__(
        self,
        team_names: list[str],
        team_agent_mapping: dict[str, str],
        agent_team_mapping: dict[str, str],
        env_wrapper: EnvWrapper,
        ws: WebSocketManager,
        mission_queue: MissionQueue,
        events: EventLog,
        match_seed: int,
        track: str,
        match_out_dir: str | None,
        get_scores_fn: Callable[[], list[dict]],  # injected by MatchCoordinator
    ) -> None:
        self._team_names = team_names
        self._team_agent_mapping = team_agent_mapping
        self._agent_team_mapping = agent_team_mapping
        self._env = env_wrapper
        self._ws = ws
        self._mission_queue = mission_queue
        self._events = events
        self._match_seed = match_seed
        self._track = track
        self._match_out_dir = match_out_dir
        self._get_scores = get_scores_fn

        self.auto_step = True
        self.step_num = 0
        self._consecutive_step_errors = 0
        self.ae_leaderboard: dict[str, float] = {t: 0.0 for t in team_names}
        self._observations: dict[str, Any] | None = None
        self._actions: dict[str, int] = {
            a: DEFAULT_ACTION for a in team_agent_mapping.values()
        }
        self._ae_responded: dict[str, bool] = {t: False for t in team_names}
        self._match_completed = False

    def record_action(self, team: str, step: int | None, action: int | None) -> None:
        """Called by WS handler for incoming AE action messages.

        action is None when the team sent an unparseable action: the team is
        still marked responded (so the step gate doesn't stall on it) but its
        move stays at the per-step default (STAY).
        """
        if step != self.step_num:
            logger.info(
                f"Rejecting AE data for {team}: arrived for step {step}, "
                f"current step is {self.step_num}"
            )
            return
        agent_id = self._team_agent_mapping.get(team)
        if agent_id and action is not None:
            self._actions[agent_id] = action
        self._ae_responded[team] = True

    def match_completed(self) -> bool:
        return self._match_completed

    # ── main loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Run the Bomberman loop until auto_step is False."""
        self._observations, _ = self._env.reset(self._match_seed)
        self.step_num = 0
        self._match_completed = False

        try:
            while self.auto_step:
                await self._step()
        except Exception as e:
            logger.exception(e)
            self.auto_step = False

    async def finalize(self) -> None:
        """Called after the batch drain completes. Emits match_end event."""
        if self._match_completed:
            self._events.emit(
                "match_end",
                ae_scores={t: round(v, 4) for t, v in self.ae_leaderboard.items()},
                mission_multipliers={
                    t: round(self._mission_queue.mission_multiplier(t), 4)
                    for t in self._team_names
                },
                batches_completed={
                    t: self._mission_queue.batches_completed(t)
                    for t in self._team_names
                },
                scores=self._get_scores(),
            )

    # ── step phases ───────────────────────────────────────────────────────

    async def _step(self) -> None:
        # ── critical path: collect actions + advance the env ───────────────
        # If this fails the step cannot proceed; retry next iteration and only
        # abort the whole match after many CONSECUTIVE failures, so a transient
        # glitch never strands every participant.
        try:
            actions = await self._collect_actions()
            rewards, terminations, truncations, infos = self._advance_env(actions)
        except Exception:
            self._consecutive_step_errors += 1
            logger.exception(
                f"AE step critical path failed "
                f"({self._consecutive_step_errors}/{MAX_CONSECUTIVE_STEP_ERRORS} "
                f"consecutive)"
            )
            if self._consecutive_step_errors >= MAX_CONSECUTIVE_STEP_ERRORS:
                logger.error("too many consecutive step failures; ending match")
                self.auto_step = False
            return
        self._consecutive_step_errors = 0

        # ── non-critical sub-phases ────────────────────────────────────────
        # Each is isolated: a failure in one (a mission trigger, logging)
        # must not skip the others or end the match. finalize always runs
        # last so end-of-match detection is never skipped.
        try:
            self._handle_mission_triggers(infos)
        except Exception:
            logger.exception("mission-trigger handling failed for this step")
        try:
            self._log_step(actions, rewards)
        except Exception:
            logger.exception("step logging failed for this step")
        try:
            await self._finalize_if_done(terminations, truncations)
        except Exception:
            logger.exception("finalize-if-done check failed for this step")
        # force a sleep here to pretend the robots are moving
        await asyncio.sleep(1)

    async def _collect_actions(self) -> dict[str, int]:
        # Reset per-step state
        for agent in self._team_agent_mapping.values():
            self._actions[agent] = DEFAULT_ACTION
        for team in self._team_names:
            self._ae_responded[team] = False

        # Send observations to each connected team (serialized via team locks)
        send_tasks = []
        for team, conn in self._ws.team_connections.items():
            if conn and self._observations:
                obs = self._observations.get(self._team_agent_mapping[team])
                if obs is not None:
                    obs_serializable = {
                        k: v if type(v) is int else v.tolist() for k, v in obs.items()
                    }
                    send_tasks.append(
                        self._ws.send_to_team(
                            team,
                            {
                                "type": "task",
                                "task": "ae",
                                "observation": obs_serializable,
                            },
                        )
                    )
        await asyncio.gather(*send_tasks, return_exceptions=True)

        # Wait for AE responses up to cutoff
        deadline = time() + constants.AE_TIME_CUTOFF
        while True:
            pending = [
                t
                for t, conn in self._ws.team_connections.items()
                if conn is not None and not self._ae_responded.get(t, False)
            ]
            if not pending:
                break
            if time() >= deadline:
                logger.info(
                    f"AE cutoff after {constants.AE_TIME_CUTOFF}s; "
                    f"defaulting actions for {pending}"
                )
                break
            await asyncio.sleep(0.02)

        return self._actions.copy()

    def _advance_env(self, actions: dict[str, int]):
        observations, rewards, terminations, truncations, infos = self._env.step(
            actions
        )
        self.step_num += 1
        self._observations = observations
        self._mission_queue.step_num = self.step_num
        for agent_id, reward in rewards.items():
            team = self._agent_team_mapping[agent_id]
            self.ae_leaderboard[team] += float(reward)
        return rewards, terminations, truncations, infos

    def _handle_mission_triggers(self, infos: dict) -> None:
        for agent_id, info in infos.items():
            if info.get("add_mission", False):
                self._mission_queue.enqueue_for_agent(agent_id)

    def _log_step(self, actions: dict[str, int], rewards: dict[str, float]) -> None:
        self._events.emit(
            "step",
            step=self.step_num,
            moves={
                aid: int(actions.get(aid, DEFAULT_ACTION))
                for aid in self._team_agent_mapping.values()
            },
            env_rewards={
                aid: float(rewards.get(aid, 0.0))
                for aid in self._team_agent_mapping.values()
            },
        )

    async def _finalize_if_done(self, terminations: dict, truncations: dict) -> None:
        if not (any(terminations.values()) or any(truncations.values())):
            return

        logger.info("match complete")
        scores_sorted = self._get_scores()
        self._events.emit(
            "round_end",
            step=self.step_num,
            reason="truncation" if any(truncations.values()) else "termination",
            ae_scores={t: round(v, 4) for t, v in self.ae_leaderboard.items()},
            mission_multipliers={
                t: round(self._mission_queue.mission_multiplier(t), 4)
                for t in self._team_names
            },
            batches_completed={
                t: self._mission_queue.batches_completed(t) for t in self._team_names
            },
            scores=scores_sorted,
        )
        if self._match_out_dir:
            agent_ids = list(self._team_agent_mapping.values())
            asyncio.create_task(
                asyncio.to_thread(
                    render_match_video,
                    f"{self._match_out_dir}/events.jsonl",
                    self._env.cfg,
                    self._match_seed,
                    f"{self._match_out_dir}/match.mp4",
                    agent_ids,
                    DEFAULT_ACTION,
                    20,
                )
            )
        self._match_completed = True
        self.auto_step = False
