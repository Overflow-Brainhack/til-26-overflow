"""MatchCoordinator — the thin coordinator replacing the MatchEngine god object.

Owns the match lifecycle by delegating to phase objects.  Exposes only the
public interface needed by server.py (the FastAPI layer):
  - ensure_artifacts()
  - start()             corpus + noise + NLP prewarm
  - run_until_stop()    AE loop + batch processors
  - resolve_batch_reply()  from WS AE/mission/noise handler
  - record_ae_action()     from WS AE handler
  - mark_corpus_ack()      from WS corpus handler
  - in_progress, auto_step
  - get_scores()

All match state is owned by the phase objects; MatchCoordinator only holds
references and wires them together.
"""

import asyncio
import logging
from time import time
from typing import Any, Awaitable, Callable

import constants
import nlp_eval
from artifacts import MatchDir
from config import MatchConfig
from env_state import EnvWrapper
from missions import (
    MissionQueue,
    TaskHandler,
    load_corpus_documents,
    load_nlp_questions,
)
from phases.ae_loop import AELoop
from phases.corpus import CorpusPhase
from phases.noise import NoisePhase
from setup import MatchSetup
from transport import WebSocketManager

logger = logging.getLogger("uvicorn.error")


class MatchCoordinator:
    """Coordinates the full match lifecycle.

    Construction is cheap and synchronous.  Call ensure_artifacts() then
    start() then run_until_stop() to drive a match.
    """

    def __init__(self, cfg: MatchConfig) -> None:
        self._cfg = cfg
        team_names = cfg.teams
        track = cfg.track
        stage = cfg.stage
        match_seed = cfg.match
        stage_dir = cfg.stage_dir

        # ── agent ↔ team mapping ───────────────────────────────────────────
        self.team_agent_mapping: dict[str, str] = {
            t: f"agent_{i}" for i, t in enumerate(team_names)
        }
        self.agent_team_mapping: dict[str, str] = {
            v: k for k, v in self.team_agent_mapping.items()
        }

        # ── transport ─────────────────────────────────────────────────────
        self.ws = WebSocketManager(team_names)

        # ── env ───────────────────────────────────────────────────────────
        self._env = EnvWrapper(track, team_names)

        # ── pre-match data ─────────────────────────────────────────────────
        self._corpus_documents = load_corpus_documents(stage_dir)
        self._nlp_questions = load_nlp_questions(stage_dir)
        self._setup = MatchSetup(
            stage_dir, team_names, self._nlp_questions, seed=match_seed
        )
        self._ref_handler = TaskHandler(stage_dir, task_pools=self._setup.task_pools)

        # ── shared correlation table (noise chunks + mission batches) ──────
        # batch_id -> (owning_team, future). The owner is recorded so a reply
        # is only ever accepted from the team the batch was sent to.
        self._pending_batches: dict[str, tuple[str, asyncio.Future]] = {}

        # ── artifact directory (created lazily by ensure_artifacts) ────────
        self._match_dir = MatchDir(track, stage)

        # ── state visible to server.py ─────────────────────────────────────
        self.auto_step: bool = True
        # `starting` latches True synchronously the instant /start is accepted
        # and stays True through the (multi-minute, blocking) pre-match gates
        # until the match loop ends. `in_progress` only flips True once the AE
        # loop is actually running. /start must reject when EITHER is set, so a
        # second /start during the pre-match window cannot spawn a second match.
        self.starting: bool = False
        self.in_progress: bool = False
        # Flips True once the match has wound down (completed, /stop, or crash);
        # surfaced via GET /match_status so runners can poll for completion.
        self.match_ended: bool = False
        # Optional async hook fired with the match_end payload after wind-down
        # (server.py wires this to broadcast_match_status for /ws/match_status).
        self.on_match_end: Callable[[dict], Awaitable[None]] | None = None

        # ── phase objects (set during start()) ────────────────────────────
        self._corpus_phase: CorpusPhase | None = None
        self._noise_phase: NoisePhase | None = None
        self._mission_queue: MissionQueue | None = None
        self._ae_loop: AELoop | None = None

        # ── internal ──────────────────────────────────────────────────────
        self._results_written = False
        logger.info(
            "pre-built batches: "
            + ", ".join(
                f"{t.value}={n} batches" for t, n in self._setup.pool_sizes.items()
            )
        )

    # ── public interface for server.py ────────────────────────────────────

    def ensure_artifacts(self) -> None:
        """Idempotent. Creates the match output directory + log files."""
        self._match_dir.ensure_created()
        if self._match_dir.events:
            self._match_dir.events.emit(
                "match_start",
                teams=self._cfg.teams,
                track=self._cfg.track,
                stage=self._cfg.stage,
                match=self._cfg.match,
                mission_batch_size=constants.MISSION_BATCH_SIZE,
                mission_batch_timeout_sec=constants.MISSION_BATCH_TIMEOUT_SEC,
                speed_score_tmax_sec=constants.MAX_TIME_PER_TEST_CASE,
                performance_weight=constants.PERFORMANCE_WEIGHT,
                speed_weight=constants.SPEED_WEIGHT,
                pool_sizes={t.value: n for t, n in self._setup.pool_sizes.items()},
            )

    def mark_corpus_ack(self, team: str) -> None:
        if self._corpus_phase:
            self._corpus_phase.mark_ack(team)

    def record_ae_action(self, team: str, step: int | None, action: int | None) -> None:
        if self._ae_loop:
            self._ae_loop.record_action(team, step, action)

    def resolve_batch_reply(self, team: str, batch_id: str, results: list) -> None:
        """Resolve a pending batch future from a team's WS reply.

        The reply is accepted ONLY from the team that owns the batch, so a team
        can never resolve (and inject results into) another team's pending batch
        even if it echoes or guesses the batch_id.
        """
        entry = self._pending_batches.get(batch_id)
        if entry is None:
            return
        owner, future = entry
        if owner != team:
            logger.warning(
                f"dropping cross-team reply for batch {batch_id!r}: "
                f"sender={team!r} owner={owner!r}"
            )
            return
        if not future.done():
            future.set_result(results if isinstance(results, list) else [])

    # ── leaderboard ───────────────────────────────────────────────────────

    def get_scores(self) -> list[dict[str, Any]]:
        scored = []
        for team in self._cfg.teams:
            ae = self._ae_loop.ae_leaderboard[team] if self._ae_loop else 0.0
            mult = (
                self._mission_queue.mission_multiplier(team)
                if self._mission_queue
                else 0.0
            )
            final = ae * mult
            scored.append(
                {
                    "team": team,
                    "ae": round(ae, 4),
                    "mission_multiplier": round(mult, 4),
                    "batches_completed": (
                        self._mission_queue.batches_completed(team)
                        if self._mission_queue
                        else 0
                    ),
                    "final": round(final, 4),
                    "score": round(final * 10),
                    "idx": int(self.team_agent_mapping[team].split("_", 1)[1]),
                }
            )
        scored.sort(key=lambda x: x["final"], reverse=True)
        return scored

    # ── match status (served by server.py: GET /match_status) ──────────────

    def status_snapshot(self) -> dict[str, Any]:
        """Pollable match status.

        `ended` flips True once the match has wound down (the bash test runner
        polls this until true); `started`/`starting`/`in_progress` distinguish
        the earlier phases. `match_dir` is the artifact dir (None until /start).
        """
        return {
            "started": self._match_dir.path is not None,
            "starting": self.starting,
            "in_progress": self.in_progress,
            "ended": self.match_ended,
            "step": self._ae_loop.step_num if self._ae_loop else 0,
            "match_dir": self._match_dir.path,
            "scores": self.get_scores(),
        }

    def _match_end_payload(self) -> dict[str, Any]:
        """Same shape the match_end event carries; reused for the on_match_end push."""
        ae = self._ae_loop.ae_leaderboard if self._ae_loop else {}
        mq = self._mission_queue
        return {
            "ae_scores": {t: round(v, 4) for t, v in ae.items()},
            "mission_multipliers": (
                {t: round(mq.mission_multiplier(t), 4) for t in self._cfg.teams}
                if mq
                else {}
            ),
            "batches_completed": (
                {t: mq.batches_completed(t) for t in self._cfg.teams} if mq else {}
            ),
            "scores": self.get_scores(),
        }

    # ── match lifecycle ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Corpus phase + noise phase + NLP prewarm. Call once before run()."""
        events = self._match_dir.events
        assert events is not None, "ensure_artifacts() must be called before start()"

        logger.info("━━━ PHASE 1/3 — corpus distribution ━━━")
        self._corpus_phase = CorpusPhase(
            ws=self.ws,
            corpus_documents=self._corpus_documents,
            team_names=self._cfg.teams,
            events=events,
        )
        await self._corpus_phase.run()
        acked = [t for t, ok in self._corpus_phase._acked.items() if ok]
        logger.info(
            f"corpus done — acked by {len(acked)}/{len(self._cfg.teams)} teams: {acked}"
        )

        logger.info("━━━ PHASE 2/3 — noise distribution ━━━")
        self._noise_phase = NoisePhase(
            team_names=self._cfg.teams,
            noise_partition=self._setup.noise_partition,
            noise_phase_skipped=self._setup.noise_phase_skipped,
            ref_handler=self._ref_handler,
            ws=self.ws,
            pending_batches=self._pending_batches,
            events=events,
            match_dir=self._match_dir,
        )
        noised_lookup = await self._noise_phase.run()
        logger.info(f"noise done — {len(noised_lookup)} images in lookup")

        logger.info("━━━ PHASE 3/3 — NLP model pre-warm ━━━")
        # Build per-team TaskHandlers now that noised_lookup is ready
        task_handlers = {
            team: TaskHandler(
                self._cfg.stage_dir,
                nlp_questions=self._nlp_questions,
                nlp_eval_model_path=self._cfg.nlp_eval_model_path,
                task_pools=self._setup.task_pools,
                per_team_cv_pool=self._setup.per_team_cv_pools.get(team),
                noised_lookup=noised_lookup,
            )
            for team in self._cfg.teams
        }

        # Pre-warm NLP eval model
        _t0 = time()
        events.emit("nlp_prewarm_start")
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    nlp_eval.ensure_loaded,
                    str(self._cfg.nlp_eval_model_path),
                    constants.NLP_EVAL_THRESHOLD,
                    constants.NLP_EVAL_MAX_LENGTH,
                ),
                timeout=120.0,
            )
            events.emit("nlp_prewarm_done", elapsed_sec=round(time() - _t0, 3))
            logger.info(f"NLP model ready ({round(time() - _t0, 1)}s)")
        except asyncio.TimeoutError:
            logger.warning(
                "NLP pre-warm timed out after 120s; falling back to lazy load"
            )
            events.emit("nlp_prewarm_failed", elapsed_sec=round(time() - _t0, 3))
        except Exception:
            logger.exception("NLP pre-warm failed; falling back to lazy load")
            events.emit("nlp_prewarm_failed", elapsed_sec=round(time() - _t0, 3))

        # Build MissionQueue + AELoop wired together
        self._mission_queue = MissionQueue(
            team_names=self._cfg.teams,
            agent_team_mapping=self.agent_team_mapping,
            task_handlers=task_handlers,
            ws=self.ws,
            pending_batches=self._pending_batches,
            events=events,
            results=self._match_dir.results,
            get_scores_fn=self.get_scores,
        )
        self._ae_loop = AELoop(
            team_names=self._cfg.teams,
            team_agent_mapping=self.team_agent_mapping,
            agent_team_mapping=self.agent_team_mapping,
            env_wrapper=self._env,
            ws=self.ws,
            mission_queue=self._mission_queue,
            events=events,
            match_seed=self._cfg.match,
            track=self._cfg.track,
            match_out_dir=self._match_dir.path,
            get_scores_fn=self.get_scores,
        )
        self._ae_loop.auto_step = self.auto_step

        # Write match_results.jsonl header line
        if self._match_dir.results:
            self._match_dir.results.append(
                {
                    "type": "match_start",
                    "teams": self._cfg.teams,
                    "track": self._cfg.track,
                }
            )

    async def run_until_stop(self) -> None:
        """AE loop + batch processors. Call as a BackgroundTask after start().

        The match-end path (drain → finalize → summary → `done`) ALWAYS runs,
        regardless of how the AE loop exits — normal termination, operator
        /stop, or an unhandled crash. Teams are never left hanging waiting for
        a `done` that never comes, and `in_progress`/`starting` are always
        reset so the server can never wedge into a permanently-busy state.
        """
        if self._ae_loop is None or self._mission_queue is None:
            logger.error("run_until_stop called before start()")
            self.starting = False
            return

        logger.info("━━━ AE LOOP starting ━━━")
        self.in_progress = True
        self.starting = False
        self._env.ensure_session()

        # Start mission processors as a background task (runs concurrently with AE loop)
        mq_task = asyncio.create_task(self._mission_queue.run())

        try:
            await self._ae_loop.run()
        except Exception:
            logger.exception("AE loop crashed; running match-end finalization anyway")
        finally:
            # Each finalization step is independently guarded: one failing must
            # not skip the rest, and the match must always be wound down cleanly.
            self._mission_queue.shutdown()
            try:
                await mq_task
            except Exception:
                logger.exception("mission-queue drain failed during shutdown")
            try:
                await self._ae_loop.finalize()
            except Exception:
                logger.exception("AE loop finalize() failed")
            try:
                self._write_summary()
            except Exception:
                logger.exception("writing match summary failed")
            try:
                await self.ws.broadcast_teams({"type": "done"})
            except Exception:
                logger.exception("broadcasting 'done' to teams failed")
            self.in_progress = False
            self.starting = False
            self.match_ended = True
            if self.on_match_end is not None:
                try:
                    await self.on_match_end(self._match_end_payload())
                except Exception:
                    logger.exception("on_match_end hook raised")
            logger.info("━━━ match wound down; server idle ━━━")

    def _write_summary(self) -> None:
        if not self._match_dir.results:
            return
        if self._results_written:
            return
        self._results_written = True
        ae_scores = self._ae_loop.ae_leaderboard if self._ae_loop else {}
        mq = self._mission_queue
        self._match_dir.results.append(
            {
                "type": "summary",
                "scores": self.get_scores(),
                "ae_scores": {t: round(v, 4) for t, v in ae_scores.items()},
                "mission_multipliers": (
                    {t: round(mq.mission_multiplier(t), 4) for t in self._cfg.teams}
                    if mq
                    else {}
                ),
                "batches_completed": (
                    {t: mq.batches_completed(t) for t in self._cfg.teams} if mq else {}
                ),
                "total_steps": self._ae_loop.step_num if self._ae_loop else 0,
                "total_batches": mq.total_batches if mq else 0,
                "timed_out_batches": mq.timed_out_batches if mq else 0,
            }
        )
