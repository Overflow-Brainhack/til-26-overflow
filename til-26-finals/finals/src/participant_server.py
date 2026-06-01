"""Participant orchestrator.

Connects to the competition server's ws://<SERVER_IP>:<SERVER_PORT>/ws/<TEAM_NAME>, listens for
two kinds of messages:

  * type=task, task=ae  — per-step bomberman observation. Reply with an
                          action via ws (synchronous-per-step on HQ).
  * type=mission_batch  — a batch of 4 items for one task type (asr/cv/
                          nlp/noise). Run the model batch concurrently
                          with any other in-flight work, reply with results.
  * type=corpus         — RAG corpus broadcast. Ingest, then ack.
  * type=done           — match over.

Also exposes an HTTP server on port 5000:

  * GET /health         — returns a boolean indicating whether the WebSocket
                          connection to the competition server is live.
"""

import asyncio
import json
import logging
import os
import traceback
from urllib.parse import quote

import uvicorn
import websockets
from fastapi import FastAPI
from models_manager import ModelsManager
from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG

TEAM_NAME = os.environ["TEAM_NAME"]
LOCAL_IP = os.environ["LOCAL_IP"]
SERVER_IP = os.environ["COMPETITION_SERVER_IP"]
SERVER_PORT = os.environ["COMPETITION_SERVER_PORT"]
WS_PATH = quote(f"ws://{SERVER_IP}:{SERVER_PORT}/ws/{TEAM_NAME}", safe="/:")

# configure uvicorn logging to include timestamps
LOGGING_CONFIG = UVICORN_LOGGING_CONFIG.copy()
LOGGING_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
LOGGING_CONFIG["formatters"]["default"][
    "fmt"
] = "%(asctime)s - %(levelprefix)s %(message)s"
LOGGING_CONFIG["formatters"]["access"][
    "fmt"
] = '%(asctime)s - %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"


logger = logging.getLogger("uvicorn.error")

manager = ModelsManager(LOCAL_IP)

_ws_connected = False

app = FastAPI()


@app.get("/health")
async def health():
    return _ws_connected


async def handle_ae(websocket, data: dict) -> None:
    try:
        action = await manager.run_ae(data["observation"])
        await manager.send_result(
            websocket,
            {
                "task": "ae",
                "result": {"step": data["observation"]["step"], "action": action},
            },
        )
    except Exception as e:
        logger.error(f"AE error: {e}")
        traceback.print_exception(e)


async def handle_mission_batch(websocket, data: dict) -> None:
    """One batch = N items of a single task type. Dispatch to the
    appropriate model container, then reply with all N results in one
    message. Batches for different teams (or different task types within
    a mission) are handled in parallel as separate asyncio.tasks."""
    batch_id = data.get("batch_id")
    task = data.get("task") or data.get("type")
    items = data.get("items", [])
    try:
        if task == "asr":
            results = await manager.run_asr_batch(items)
        elif task == "cv":
            results = await manager.run_cv_batch(items)
        elif task == "nlp":
            results = await manager.run_nlp_batch(items)
        elif task == "noise":
            results = await manager.run_noise_batch(items)
        else:
            logger.warning(f"unknown mission_batch task {task!r}; replying empty")
            results = []
        await manager.send_result(
            websocket,
            {
                "task": "mission_batch",
                "batch_id": batch_id,
                "results": results,
            },
        )
    except Exception as e:
        logger.error(f"mission_batch error (batch_id={batch_id}): {e}")
        traceback.print_exception(e)
        # Best-effort: still reply with empty results so the server's
        # 10s wait_for can resolve without timing out unnecessarily.
        try:
            await manager.send_result(
                websocket,
                {
                    "task": "mission_batch",
                    "batch_id": batch_id,
                    "results": [],
                },
            )
        except Exception:
            pass


async def handle_corpus(websocket, data: dict) -> None:
    ok = await manager.ingest_corpus(data["documents"])
    await manager.send_result(websocket, {"task": "corpus_ack", "ok": ok})


async def ws_server():
    global _ws_connected
    logger.info(f"attempting to connect to competition server at `{WS_PATH}` ...")
    async for websocket in websockets.connect(
        WS_PATH,
        # 256 MiB, well above the largest mission_batch size.
        max_size=2**28,
    ):
        _ws_connected = True
        logger.info("connection with competition server established!")
        running_tasks: set[asyncio.Task] = set()

        def spawn(coro):
            t = asyncio.create_task(coro)
            running_tasks.add(t)
            t.add_done_callback(running_tasks.discard)

        try:
            while True:
                socket_input = await websocket.recv()
                if not isinstance(socket_input, str):
                    logger.warning(
                        f"received invalid data of type {type(socket_input)}"
                    )
                    continue
                data = json.loads(socket_input)
                msg_type = data.get("type")
                if msg_type == "task" and data.get("task") == "ae":
                    spawn(handle_ae(websocket, data))
                elif msg_type in ("mission_batch", "noise"):
                    spawn(handle_mission_batch(websocket, data))
                elif msg_type == "corpus":
                    spawn(handle_corpus(websocket, data))
                elif msg_type == "done":
                    logger.info("done!")
                    if running_tasks:
                        logger.info(
                            f"Waiting for {len(running_tasks)} tasks to complete..."
                        )
                        await asyncio.gather(*running_tasks, return_exceptions=True)
                    await manager.exit()
                    return
                elif msg_type == "health":
                    await manager.send_result(websocket, {"task": "health_ack"})
                else:
                    logger.warning(f"ignoring message of type {msg_type!r}: {data}")
        except websockets.ConnectionClosed:
            _ws_connected = False
            await shutdown(running_tasks)
            continue
        except KeyboardInterrupt:
            _ws_connected = False
            await shutdown(running_tasks)
            break
        except Exception as e:
            traceback.print_exception(e)
            _ws_connected = False
            await shutdown(running_tasks)


async def shutdown(running_tasks: set[asyncio.Task]) -> None:
    for task in running_tasks:
        task.cancel()


async def main():
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info",
        log_config=LOGGING_CONFIG,
    )
    http_server = uvicorn.Server(config)
    http_task = asyncio.create_task(http_server.serve())
    try:
        await ws_server()
    finally:
        http_server.should_exit = True
        await http_task


if __name__ == "__main__":
    asyncio.run(main())
