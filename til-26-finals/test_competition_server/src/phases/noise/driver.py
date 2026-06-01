"""TeamNoiseDriver: drive one team's noise exchange.

Reads images from disk (in a thread), sends chunks over WS, collects noised
images back, runs fairness checks (also in threads, overlapping with
subsequent chunks), and returns a frozen TeamNoiseResult instead of mutating
caller-owned lists.
"""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import Any

import constants
import numpy as np
from artifacts import EventLog
from domain import TeamNoiseResult
from missions import TaskHandler
from transport import WebSocketManager

from .fairness import apply_fairness_fallback

logger = logging.getLogger("uvicorn.error")


@dataclass
class NoiseImage:
    """One image in a team's noise assignment."""

    filename: str
    index: int
    orig_bytes: bytes
    b64: str


class TeamNoiseDriver:
    """Drives the noise assignment for one team, returning a TeamNoiseResult."""

    def __init__(
        self,
        team: str,
        assignment: list[list[int]],
        ref_handler: TaskHandler,
        ws: WebSocketManager,
        pending_batches: dict[str, tuple[str, asyncio.Future]],
        events: EventLog,
    ) -> None:
        self._team = team
        self._assignment = assignment
        self._ref_handler = ref_handler
        self._ws = ws
        self._pending_batches = pending_batches
        self._events = events

    async def run(self) -> TeamNoiseResult:
        """Drive the full noise exchange for this team. Returns TeamNoiseResult."""
        team = self._team
        assignment = self._assignment

        if not assignment:
            return TeamNoiseResult(team=team, items=(), fairness={}, noised={})

        connection = self._ws.team_connections.get(team)
        if connection is None:
            logger.info(f"[{team}] not connected; originals used for noise assignment")
            return TeamNoiseResult(team=team, items=(), fairness={}, noised={})

        # Read all images from disk in one thread (avoids many small to_thread calls)
        items_data = await asyncio.to_thread(self._read_images)
        logger.info(f"[{team}] noise: read {len(items_data)} images from disk")

        # Re-check connection after blocking I/O
        current = self._ws.team_connections.get(team)
        if current is not connection:
            if current is None:
                return TeamNoiseResult(team=team, items=(), fairness={}, noised={})
            connection = current

        # Build all_items list (filename + orig_bytes) for fairness accumulation
        all_items = tuple((img.filename, img.orig_bytes) for img in items_data)

        noised: dict[str, bytes] = {}
        results_by_id: dict[str, str] = {}
        pending_fairness: list[
            tuple[list[str], list[bytes], list[str | None], asyncio.Task]
        ] = []

        chunks = [
            items_data[i : i + constants.MISSION_BATCH_SIZE]
            for i in range(0, len(items_data), constants.MISSION_BATCH_SIZE)
        ]
        n_chunks = len(chunks)
        self._events.emit(
            "noise_assignment_sent",
            team=team,
            n_images=len(items_data),
            n_chunks=n_chunks,
        )

        for i, chunk in enumerate(chunks):
            batch_id = f"noise__{team}__chunk{i}"
            timed_out = await self._send_chunk(
                chunk, batch_id, connection, i, n_chunks, noised, results_by_id
            )
            if not timed_out:
                fnames = [img.filename for img in chunk]
                orig_list = [img.orig_bytes for img in chunk]
                b64_list = [results_by_id.get(fn) or None for fn in fnames]
                task = asyncio.create_task(
                    asyncio.to_thread(
                        apply_fairness_fallback,
                        orig_list,
                        b64_list,
                        [self._boxes_for(img.index) for img in chunk],
                    )
                )
                pending_fairness.append((fnames, orig_list, b64_list, task))

        # Collect fairness results
        fairness: dict[str, Any] = {}
        for fnames, orig_list, _b64_list, task in pending_fairness:
            try:
                result_bytes_list, records = await asyncio.wait_for(task, timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[{team}] fairness check timed out for {fnames}; using originals"
                )
                task.cancel()
                for fn, ob in zip(fnames, orig_list):
                    noised.setdefault(fn, ob)
                continue
            except Exception:
                # apply_fairness_fallback contains per-image errors itself, so a
                # crash here is unexpected — contain it to this chunk (originals
                # used) rather than failing the whole team's driver.
                logger.exception(
                    f"[{team}] fairness check crashed for {fnames}; using originals"
                )
                for fn, ob in zip(fnames, orig_list):
                    noised.setdefault(fn, ob)
                continue
            for fn, rb, rec in zip(fnames, result_bytes_list, records):
                noised[fn] = rb
                fairness[fn] = rec

        return TeamNoiseResult(
            team=team, items=all_items, fairness=fairness, noised=noised
        )

    def _read_images(self) -> list[NoiseImage]:
        """Synchronous disk read for all images in the assignment."""
        ref = self._ref_handler
        data: list[NoiseImage] = []
        for batch in self._assignment:
            for idx in batch:
                img_id = ref.cv_image_ids[idx]
                file_name = ref.cv_img_info[img_id]["file_name"]
                with open(ref.cv_dir / "images" / file_name, "rb") as f:
                    orig_bytes = f.read()
                b64_str = base64.b64encode(orig_bytes).decode("ascii")
                data.append(
                    NoiseImage(
                        filename=file_name,
                        index=idx,
                        orig_bytes=orig_bytes,
                        b64=b64_str,
                    )
                )
        return data

    def _boxes_for(self, idx: int) -> np.ndarray:
        ref = self._ref_handler
        img_id = ref.cv_image_ids[idx]
        anns = ref.cv_ann_info.get(img_id)
        if anns:
            return np.array([ann["bbox"] for ann in anns])
        return np.zeros((0, 4))

    async def _send_chunk(
        self,
        chunk: list[NoiseImage],
        batch_id: str,
        connection,
        i: int,
        n_chunks: int,
        noised: dict[str, bytes],
        results_by_id: dict[str, str],
    ) -> bool:
        """Send one noise chunk, wait for reply. Returns True if timed out."""
        team = self._team
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_batches[batch_id] = (team, future)

        noise_items = [{"task_id": img.filename, "b64": img.b64} for img in chunk]
        try:
            text = await asyncio.to_thread(
                json.dumps,
                {
                    "type": "noise",
                    "batch_id": batch_id,
                    "deadline_sec": constants.NOISE_BATCH_TIMEOUT_SEC,
                    "items": noise_items,
                },
            )
            sent = await self._ws.send_to_team(team, text)
            if not sent:
                raise RuntimeError("send_to_team returned False")
            logger.info(
                f"[{team}] sent noise batch {i + 1}/{n_chunks} ({len(chunk)} images)"
            )
        except Exception as e:
            logger.warning(f"[{team}] failed to send noise batch {i}: {e!r}")
            self._pending_batches.pop(batch_id, None)
            for img in chunk:
                noised[img.filename] = img.orig_bytes
            return True

        try:
            resp = await asyncio.wait_for(
                future, timeout=constants.NOISE_BATCH_TIMEOUT_SEC
            )
            for r in resp:
                if isinstance(r, dict) and r.get("task_id"):
                    results_by_id[r["task_id"]] = r.get("b64", "")
            logger.info(f"[{team}] noise batch {i + 1}/{n_chunks} received")
            return False
        except asyncio.TimeoutError:
            logger.warning(f"[{team}] noise batch {i + 1}/{n_chunks} timed out")
            self._events.emit(
                "noise_batch_timeout", team=team, batch_index=i, reason="timeout"
            )
            for img in chunk:
                noised[img.filename] = img.orig_bytes
            return True
        finally:
            self._pending_batches.pop(batch_id, None)
