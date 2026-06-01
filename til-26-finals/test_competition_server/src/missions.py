"""Mission task handling: TaskHandler (data access) + MissionQueue (per-team processor).

TaskHandler owns disk access and batch deck management.  It has no scoring
logic — that lives in scoring.py.

MissionQueue owns the per-team batch queues and processes them FIFO, one
batch at a time per team, scoring results via scoring.score_batch and
accumulating via scoring.accumulate_batch_result.
"""

import asyncio
import base64
import json
import logging
from collections import deque
from pathlib import Path
from time import time
from typing import Any, Callable

import constants
import scoring
from artifacts import EventLog, ResultsLog
from domain import Batch, BatchItem, ScoredBatch, TaskType
from transport import WebSocketManager

logger = logging.getLogger("uvicorn.error")

DEFAULT_MISSION_MULTIPLIER = 0


# ── data loaders ──────────────────────────────────────────────────────────────


def load_nlp_questions(data_dir: Path) -> list[dict]:
    """Load `<stage>/nlp/nlp.jsonl`. Each row: question/answer/source_docs."""
    nlp_path = data_dir / "nlp" / "nlp.jsonl"
    with open(nlp_path, encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def load_corpus_documents(data_dir: Path) -> list[dict]:
    """Load `<stage>/nlp/documents/*.txt` as [{id, document}, ...].

    The corpus is shared across tracks (same docs, different question sets).
    """
    documents_dir = data_dir / "nlp" / "documents"
    docs: list[dict] = []
    for doc_file in sorted(documents_dir.glob("*.txt")):
        with open(doc_file, "r", encoding="utf-8") as f:
            docs.append({"id": doc_file.stem, "document": f.read()})
    return docs


# ── TaskHandler: data access, batch deck, wire encoding ──────────────────────


class TaskHandler:
    """Per-team data-access object.

    Owns disk paths, CV annotations, ASR transcripts, NLP questions, and
    the per-team batch deck.  No scoring logic — all scoring is in scoring.py.
    """

    def __init__(
        self,
        data_dir: Path,
        nlp_questions: list[dict] | None = None,
        nlp_eval_model_path: str | Path | None = None,
        task_pools: dict[TaskType, list[list[int]]] | None = None,
        per_team_cv_pool: list[list[int]] | None = None,
        noised_lookup: dict[str, bytes] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.nlp_eval_model_path = nlp_eval_model_path
        self.asr_dir = data_dir / "asr"
        self.cv_dir = data_dir / "cv"
        self.task_pools: dict[TaskType, list[list[int]]] = task_pools or {}
        self.per_team_cv_pool: list[list[int]] | None = per_team_cv_pool
        self.noised_lookup: dict[str, bytes] = (
            noised_lookup if noised_lookup is not None else {}
        )

        # ASR
        with open(self.asr_dir / "asr.jsonl", encoding="utf-8") as f:
            self.asr_instances: list[dict] = [
                json.loads(line.strip()) for line in f if line.strip()
            ]

        # CV
        with open(self.cv_dir / "annotations.json", encoding="utf-8") as fh:
            cv_anns_raw = json.load(fh)
        self.cv_img_info: dict[int, dict] = {
            img["id"]: img for img in cv_anns_raw["images"]
        }
        self.cv_image_ids: list[int] = [img["id"] for img in cv_anns_raw["images"]]
        self.cv_ann_info: dict[int, list[dict]] = {}
        for ann in cv_anns_raw["annotations"]:
            self.cv_ann_info.setdefault(ann["image_id"], []).append(ann)
        self.cv_categories = cv_anns_raw["categories"]

        # NLP
        self.nlp_questions: list[dict] = nlp_questions or []

        # Per-team batch deck
        self._available_batches: dict[TaskType, deque[list[int]]] = {
            t: deque() for t in TaskType
        }
        self._reset_deck()

    def _reset_deck(self) -> None:
        for task_type in TaskType:
            if task_type == TaskType.CV and self.per_team_cv_pool is not None:
                self._available_batches[task_type] = deque(
                    [list(b) for b in self.per_team_cv_pool]
                )
            else:
                batches = self.task_pools.get(task_type, [])
                self._available_batches[task_type] = deque([list(b) for b in batches])

    def batches_remaining(self, task_type: TaskType) -> int:
        return len(self._available_batches.get(task_type, ()))

    def _task_id_for(self, task_type: TaskType, index: int) -> Any:
        match task_type:
            case TaskType.ASR:
                return self.asr_instances[index].get("audio", index)
            case TaskType.CV:
                img_id = self.cv_image_ids[index]
                return self.cv_img_info[img_id]["file_name"]
            case _:  # NLP
                return index

    def draw_batch(self, task_type: TaskType) -> list[BatchItem]:
        """Pop the next pre-built batch. Returns [] when pool is exhausted."""
        deck = self._available_batches.get(task_type)
        if not deck:
            return []
        indices = deck.popleft()
        return [
            BatchItem(index=idx, task_id=self._task_id_for(task_type, idx))
            for idx in indices
        ]

    def wire_payload_for_batch(self, batch: Batch) -> dict:
        """Build the JSON message to send to the team for this batch."""
        task_type = batch.task
        items_out: list[dict] = []
        for item in batch.items:
            index = item.index
            task_id = item.task_id
            match task_type:
                case TaskType.ASR:
                    inst = self.asr_instances[index]
                    with open(self.asr_dir / inst["audio"], "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    items_out.append({"task_id": task_id, "b64": b64})
                case TaskType.CV:
                    img_id = self.cv_image_ids[index]
                    file_name = self.cv_img_info[img_id]["file_name"]
                    if file_name in self.noised_lookup:
                        b64 = base64.b64encode(self.noised_lookup[file_name]).decode(
                            "ascii"
                        )
                    else:
                        with open(self.cv_dir / "images" / file_name, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("ascii")
                    items_out.append({"task_id": task_id, "b64": b64})
                case TaskType.NLP:
                    question = self.nlp_questions[index]["question"]
                    items_out.append({"task_id": task_id, "question": question})
        return {
            "type": "mission_batch",
            "batch_id": batch.batch_id,
            "mission_id": batch.mission_id,
            "task": task_type.value,
            "items": items_out,
        }


# ── MissionQueue: per-team FIFO batch processor ───────────────────────────────


class MissionQueue:
    """Owns per-team batch queues and processes them concurrently.

    One asyncio task per team drains its queue FIFO.  Only one batch is in
    flight per team at a time.  Scoring and accumulation happen inline.

    Public interface for MatchCoordinator:
      enqueue_for_agent(agent_id)  — called from AELoop on mission trigger
      run()                        — start all processor coroutines (TaskGroup)
      shutdown()                   — signal all processors to drain and exit
      mission_multiplier(team)     — current mean batch score for a team
      ae_leaderboard               — read by AELoop for leaderboard builds
    """

    def __init__(
        self,
        team_names: list[str],
        agent_team_mapping: dict[str, str],
        task_handlers: dict[str, TaskHandler],
        ws: WebSocketManager,
        pending_batches: dict[str, tuple[str, asyncio.Future]],
        events: EventLog,
        results: ResultsLog | None,
        get_scores_fn: Callable[[], list[dict]] | None = None,
    ) -> None:
        self._team_names = team_names
        self._agent_team_mapping = agent_team_mapping
        self._task_handlers = task_handlers
        self._ws = ws
        self._pending_batches = pending_batches
        self._events = events
        self._results = results
        self._get_scores = get_scores_fn

        self._batch_queues: dict[str, deque[Batch]] = {t: deque() for t in team_names}
        self._batch_scores: dict[str, list[float]] = {t: [] for t in team_names}
        self._mission_counter = 0
        self._team_mission_counters: dict[str, int] = {t: 0 for t in team_names}
        self._shutdown = False
        # step_num is updated externally by AELoop each step
        self.step_num: int = 0
        # Summary counters
        self.total_batches: int = 0
        self.timed_out_batches: int = 0

    def mission_multiplier(self, team: str) -> float:
        scores = self._batch_scores[team]
        if not scores:
            return DEFAULT_MISSION_MULTIPLIER
        return sum(scores) / len(scores)

    def batches_completed(self, team: str) -> int:
        return len(self._batch_scores[team])

    def get_scores_for_leaderboard(self) -> dict[str, float]:
        return {t: self.mission_multiplier(t) for t in self._team_names}

    def enqueue_for_agent(self, agent_id: str) -> None:
        """Called from AELoop when an agent steps on a mission tile."""
        team = self._agent_team_mapping[agent_id]
        handler = self._task_handlers[team]
        task_types = (TaskType.ASR, TaskType.CV, TaskType.NLP)

        if any(handler.batches_remaining(t) == 0 for t in task_types):
            logger.info(
                f"team {team} has exhausted mission batches; "
                f"skipping mission for {agent_id} at step {self.step_num}"
            )
            self._events.emit(
                "mission_skipped",
                step=self.step_num,
                team=team,
                agent_id=agent_id,
                reason="pool_exhausted",
            )
            return

        self._mission_counter += 1
        self._team_mission_counters[team] += 1
        mission_id = f"{team}__s{self.step_num}__m{self._mission_counter}"
        team_mission_num = self._team_mission_counters[team]

        batches_queued: list[Batch] = []
        for task_type in task_types:
            batch = Batch(
                batch_id=f"{mission_id}__{task_type.value}",
                mission_id=mission_id,
                mission_num=team_mission_num,
                task=task_type,
                items=tuple(handler.draw_batch(task_type)),
            )
            self._batch_queues[team].append(batch)
            batches_queued.append(batch)

        batches_summary = [
            {
                "batch_id": b.batch_id,
                "task": b.task.value,
                "task_ids": [it.task_id for it in b.items],
            }
            for b in batches_queued
        ]
        tasks_str = " | ".join(b.task.value.upper() for b in batches_queued)
        logger.info(
            f"mission #{self._team_mission_counters[team]} triggered"
            f"  team={team}  step={self.step_num}  [{tasks_str}]"
        )
        self._events.emit(
            "mission_triggered",
            step=self.step_num,
            team=team,
            agent_id=agent_id,
            mission_id=mission_id,
            batches_queued=batches_summary,
        )

    def shutdown(self) -> None:
        self._shutdown = True

    async def run(self) -> None:
        """Drain all team queues concurrently until shutdown + all empty.

        Each team is supervised INDEPENDENTLY via gather(return_exceptions=True)
        — deliberately NOT asyncio.TaskGroup, whose all-or-nothing semantics
        would cancel every team's processor the moment one raised. A failure in
        one team's processing must never affect any other team.
        """
        await asyncio.gather(
            *(self._supervised_processor(team) for team in self._team_names),
            return_exceptions=True,
        )

    async def _supervised_processor(self, team: str) -> None:
        """Run one team's processor, restarting it if it ever crashes.

        _team_processor already contains per-batch errors (see its inner
        try/except), so this is defence-in-depth: even an unexpected crash in a
        team's processing stays contained to that team — it is restarted and the
        other five teams are untouched.
        """
        while not self._shutdown:
            try:
                await self._team_processor(team)
                return  # clean exit: shutdown requested and queue drained
            except Exception:
                logger.exception(
                    f"[{team}] processor crashed unexpectedly; restarting "
                    f"(other teams unaffected)"
                )

    async def _team_processor(self, team: str) -> None:
        while True:
            if self._batch_queues[team]:
                batch = self._batch_queues[team].popleft()
                # Per-batch containment: a crash while processing ONE batch is
                # logged and that batch is dropped; the queue keeps draining and
                # no other batch or team is affected.
                try:
                    await self._process_batch(team, batch)
                except Exception as e:
                    logger.exception(f"batch processor error for {team}: {e}")
            elif self._shutdown:
                return
            else:
                await asyncio.sleep(0.05)

    async def _process_batch(self, team: str, batch: Batch) -> None:
        batch_id = batch.batch_id
        connection = self._ws.team_connections.get(team)
        if connection is None:
            await self._record(
                team,
                batch,
                predictions=None,
                elapsed=0.0,
                timed_out=True,
                reason="team_disconnected",
            )
            return

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        sent_at = time()
        self._pending_batches[batch_id] = (team, future)

        try:
            payload = await asyncio.to_thread(
                self._task_handlers[team].wire_payload_for_batch, batch
            )
            sent = await self._ws.send_to_team(team, payload)
            if not sent:
                self._pending_batches.pop(batch_id, None)
                await self._record(
                    team,
                    batch,
                    predictions=None,
                    elapsed=0.0,
                    timed_out=True,
                    reason="send_failed",
                )
                return
        except Exception:
            self._pending_batches.pop(batch_id, None)
            await self._record(
                team,
                batch,
                predictions=None,
                elapsed=0.0,
                timed_out=True,
                reason="send_failed",
            )
            return

        try:
            results = await asyncio.wait_for(
                future, timeout=constants.MISSION_BATCH_TIMEOUT_SEC
            )
            elapsed = time() - sent_at
            self._pending_batches.pop(batch_id, None)
            await self._record(
                team, batch, predictions=results, elapsed=elapsed, timed_out=False
            )
        except asyncio.TimeoutError:
            self._pending_batches.pop(batch_id, None)
            await self._record(
                team,
                batch,
                predictions=None,
                elapsed=constants.MISSION_BATCH_TIMEOUT_SEC,
                timed_out=True,
                reason="batch_timeout",
            )

    async def _record(
        self,
        team: str,
        batch: Batch,
        predictions: list[dict] | None,
        elapsed: float,
        timed_out: bool,
        reason: str | None = None,
    ) -> None:
        # Score off the event loop: COCOeval (CV), jiwer (ASR) and the NLP model
        # forward pass are all blocking/CPU-bound and would otherwise stall every
        # other team's processor and the AE step gate while one batch is scored.
        scored: ScoredBatch = await asyncio.to_thread(
            scoring.score_batch,
            self._task_handlers[team],
            batch,
            predictions,
            elapsed,
        )
        scored_dict = scored.to_dict()
        acc = scoring.accumulate_batch_result(
            self._batch_scores[team], scored_dict, batch, elapsed, timed_out, team
        )
        self._batch_scores[team] = list(acc.new_team_scores)

        self.total_batches += 1
        if timed_out:
            self.timed_out_batches += 1
        if self._results:
            self._results.append({"type": "batch", **acc.record})

        tag = "TIMEOUT" if timed_out else "ok"
        mission_num = batch.mission_num or "?"
        logger.info(
            f"batch [{tag}]"
            f"  team={team}  mission=#{mission_num}  task={scored.task.value.upper()}"
            f"  acc={scored.batch_accuracy:.3f}"
            f"  time={scored.time_score:.3f}"
            f"  score={scored.batch_score:.3f}"
            f"  avg={acc.mission_multiplier:.3f}"
            f"  {elapsed:.2f}s"
        )
