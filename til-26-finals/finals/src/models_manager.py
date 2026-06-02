import asyncio
import json
from logging import getLogger
from typing import Any

import httpx
import websockets

logger = getLogger("uvicorn.error")


class ModelsManager:
    """HTTP client for the 5 local model containers.

    For mission batches (asr / cv / nlp), the underlying servers already
    accept arrays of instances. So a batch of N items goes out as one
    POST with `{"instances": [N items]}` and comes back as
    `{"predictions": [N predictions]}`. We pair predictions back to
    `task_id`s so the server can correlate.
    """

    def __init__(self, local_ip: str):
        self.local_ip = local_ip
        logger.info("initializing participant finals server manager")
        self.client = httpx.AsyncClient()

    async def exit(self):
        await self.client.aclose()

    async def async_post(
        self, endpoint: str, json: dict | None = None, timeout: float | None = None
    ):
        return await self.client.post(endpoint, json=json, timeout=timeout)

    async def send_result(
        self, websocket: websockets.ClientConnection, data: dict[str, Any]
    ):
        return await websocket.send(json.dumps(data))

    # ── per-mission-batch endpoints ────────────────────────────────────────

    async def run_asr_batch(
        self, items: list[dict], timeout: float | None = None
    ) -> list[dict]:
        """`items` = [{"task_id": ..., "b64": ...}, ...]
        Returns [{"task_id": ..., "answer": str}, ...].
        """
        logger.info(f"Running ASR batch (n={len(items)})")
        response = await self.async_post(
            f"http://{self.local_ip}:5001/asr",
            json={"instances": [{"b64": it["b64"]} for it in items]},
            timeout=timeout,
        )
        predictions = response.json().get("predictions", [])
        results = [
            {"task_id": it["task_id"], "answer": pred}
            for it, pred in zip(items, predictions)
        ]
        logger.info(f"ASR batch complete: {len(results)} answers")
        return results

    async def run_cv_batch(
        self, items: list[dict], timeout: float | None = None
    ) -> list[dict]:
        """`items` = [{"task_id": ..., "b64": ...}, ...]
        Returns [{"task_id": ..., "detections": [{bbox, category_id}, ...]}, ...].
        """
        logger.info(f"Running CV batch (n={len(items)})")
        response = await self.async_post(
            f"http://{self.local_ip}:5002/cv",
            json={"instances": [{"b64": it["b64"]} for it in items]},
            timeout=timeout,
        )
        predictions = response.json().get("predictions", [])
        out: list[dict] = []
        for it, dets in zip(items, predictions):
            out.append({"task_id": it["task_id"], "detections": list(dets or [])})
        logger.info(f"CV batch complete: {len(out)} predictions")
        return out

    async def run_noise_batch(
        self, items: list[dict], timeout: float | None = None
    ) -> list[dict]:
        """`items` = [{"task_id": ..., "b64": ...}, ...]
        Returns [{"task_id": ..., "b64": <noised_b64>}, ...].
        """
        logger.info(f"Running noise batch (n={len(items)})")
        response = await self.async_post(
            f"http://{self.local_ip}:5003/noise",
            json={
                "instances": [{"key": it["task_id"], "b64": it["b64"]} for it in items]
            },
            timeout=timeout,
        )
        predictions = response.json().get("predictions", [])
        results = [
            {"task_id": it["task_id"], "b64": pred}
            for it, pred in zip(items, predictions)
            if pred is not None
        ]
        logger.info(
            f"Noise batch complete: {len(predictions)} predictions, {len(results)} non-null"
        )
        return results

    async def run_nlp_batch(
        self, items: list[dict], timeout: float | None = None
    ) -> list[dict]:
        """`items` = [{"task_id": ..., "question": str}, ...]
        Returns [{"task_id": ..., "answer": str, "documents": [str, ...]}, ...].
        """
        logger.info(f"Running NLP batch (n={len(items)})")
        response = await self.async_post(
            f"http://{self.local_ip}:5004/nlp",
            json={"instances": [{"question": it["question"]} for it in items]},
            timeout=timeout,
        )
        predictions = response.json().get("predictions", [])
        out: list[dict] = []
        for it, pred in zip(items, predictions):
            pred = pred or {}
            out.append(
                {
                    "task_id": it["task_id"],
                    "answer": pred.get("answer", ""),
                    "documents": list(pred.get("documents", []) or []),
                }
            )
        logger.info(f"NLP batch complete: {len(out)} predictions")
        return out

    # ── corpus ingest (one-shot, called from the `corpus` ws message) ─────

    async def ingest_corpus(
        self, documents: list[dict], poll_interval_sec: float = 2.0
    ) -> bool:
        """`documents` = [{"doc_id": ..., "content": str}, ...]
        Polls :5004/nlp until loaded. Returns True on success, False on error.
        """
        logger.info(f"Ingesting RAG corpus ({len(documents)} documents)")
        response = await self.async_post(
            f"http://{self.local_ip}:5004/nlp",
            json={"instances": [{"documents": documents}]},
        )
        body = response.json()
        status = body["predictions"][0].get("status", "loaded")
        if status == "error":
            logger.error(f"corpus ingest reported error: {body}")
            return False
        if status == "loaded":
            return True

        while True:
            poll_resp = await self.async_post(
                f"http://{self.local_ip}:5004/nlp",
                json={"instances": [{"poll": "true"}]},
            )
            poll_status = poll_resp.json()["predictions"][0].get("status")
            if poll_status == "loaded":
                return True
            if poll_status == "error":
                logger.error(f"corpus ingest poll reported error: {poll_resp.json()}")
                return False
            await asyncio.sleep(poll_interval_sec)

    # ── AE (per-step) ─────────────────────────────────────────────────────

    async def run_ae(self, observation: dict[str, int | list[int]]) -> int:
        """`observation` = {"field": int | [int], ...}
        Returns the chosen action integer.
        """
        logger.info("Running AE")
        results = await self.async_post(
            f"http://{self.local_ip}:5005/ae",
            json={"instances": [{"observation": observation}]},
        )
        action = results.json()["predictions"][0]["action"]
        logger.info(f"AE action: {action}")
        return action
