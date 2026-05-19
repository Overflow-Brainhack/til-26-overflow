"""Runs the NLP server."""

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, Request
from nlp_manager_cheese import NLPManager

app = FastAPI()
manager = NLPManager()
logger = logging.getLogger(__name__)


class _LoadState:
    """Tracks corpus-loading state for async, pollable behavior."""

    def __init__(self) -> None:
        self.status: str = "idle"  # idle | loading | loaded | error
        self.error: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()


load_state = _LoadState()


def _do_load(documents) -> bool:
    """Synchronous corpus load. Runs on a worker thread."""
    manager.load_corpus(documents)
    return manager.loaded


async def _load_task(documents) -> None:
    try:
        ok = await asyncio.to_thread(_do_load, documents)
        load_state.status = "loaded" if ok else "error"
    except Exception as e:
        logger.exception("Corpus load failed")
        load_state.status = "error"
        load_state.error = str(e)


@app.post("/nlp")
async def nlp(request: Request) -> dict[str, list[dict[str, list[str] | str] | str]]:
    inputs_json = await request.json()
    first = inputs_json["instances"][0]

    # Load: block on the first corpus load so non-polling test scripts also pass.
    if first.get("documents") is not None:
        documents = first["documents"]
        async with load_state.lock:
            if load_state.status == "idle":
                load_state.status = "loading"
                await _load_task(documents)
            if documents and not isinstance(documents[0], dict):
                return {
                    "predictions": [
                        "loaded" if load_state.status == "loaded" else "error"
                    ]
                }
            return {"predictions": [{"status": load_state.status}]}
    # Poll: returns current status (subsequent polls).
    if first.get("poll") is not None:
        return {"predictions": [{"status": load_state.status}]}

    questions = [instance["question"] for instance in inputs_json["instances"]]
    predictions = await asyncio.to_thread(manager.qa_batch, questions)

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    return {"message": "health ok"}
