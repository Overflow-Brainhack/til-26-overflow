"""FastAPI app for the TIL-26 competition server (v2).

Thin transport layer: HTTP endpoints, WebSocket handlers, static-file
serving, and config loading.  All match logic lives in MatchCoordinator.
"""

import logging

from config import load_config
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from match import MatchCoordinator
from til_environment.actions import Action
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

cfg = load_config()
app = FastAPI()
coordinator = MatchCoordinator(cfg)


# ── /ws/match_status — push notification for harnesses watching the match ──
#
# Mirrors the real HQ: clients (typically just the finals test runner) subscribe
# before /start. On connect we send the current status; when the match winds
# down, MatchCoordinator calls broadcast_match_status via its on_match_end hook
# and every subscriber receives `{"type": "match_end", ...}`. events.jsonl stays
# the source of truth for history — this is purely low-latency notification.
# GET /match_status is the pollable HTTP equivalent.
match_status_subscribers: set[WebSocket] = set()


async def broadcast_match_status(payload: dict) -> None:
    msg = {"type": "match_end", **payload}
    for ws in list(match_status_subscribers):
        try:
            await ws.send_json(msg)
        except Exception:
            match_status_subscribers.discard(ws)


coordinator.on_match_end = broadcast_match_status


# ── health ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return "OK"


@app.get("/status")
async def status():
    return coordinator.in_progress


@app.get("/match_status")
async def match_status():
    """Pollable snapshot of the match — the HTTP companion to /ws/match_status.

    Phases: started=False (not started) -> starting/in_progress=True (running)
    -> ended=True (wound down; stop polling).
    """
    return coordinator.status_snapshot()


# ── match control ─────────────────────────────────────────────────────────────


@app.post("/start")
async def start(background_tasks: BackgroundTasks):
    # Reject re-entry BEFORE awaiting anything. The pre-match gates inside
    # start() block for up to a few minutes (corpus + noise + NLP prewarm);
    # `in_progress` isn't set until the AE loop runs, so a guard on it alone
    # leaves the whole pre-match window re-entrant. The `starting` latch closes
    # that window. Both checks happen synchronously, before the first await.
    if coordinator.in_progress or coordinator.starting:
        logger.info("/start ignored — a match is already starting or in progress")
        return
    coordinator.starting = True
    coordinator.auto_step = True
    try:
        coordinator.ensure_artifacts()
        await coordinator.start()
    except Exception:
        logger.exception("match start failed during pre-match phase; aborting")
        coordinator.starting = False
        raise
    # run_until_stop clears `starting` once it flips `in_progress` True, and
    # clears both when the match ends.
    background_tasks.add_task(coordinator.run_until_stop)


@app.post("/stop")
async def stop():
    coordinator.auto_step = False
    if coordinator._ae_loop:
        coordinator._ae_loop.auto_step = False


# ── match-status + team WebSockets ──────────────────────────────────────────


@app.websocket("/ws/match_status")
async def match_status_endpoint(websocket: WebSocket):
    # Registered before /ws/{team_name} so "match_status" isn't captured as a team.
    await websocket.accept()
    match_status_subscribers.add(websocket)
    try:
        await websocket.send_json(
            {"type": "status", "in_progress": coordinator.in_progress}
        )
        # Block until the client disconnects; we don't expect inbound messages.
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, ConnectionClosed):
        pass
    finally:
        match_status_subscribers.discard(websocket)


@app.websocket("/ws/{team_name}")
async def team_endpoint(websocket: WebSocket, team_name: str):
    logger.info(f"incoming connection from team {team_name}")
    try:
        await coordinator.ws.team_connect(websocket, team_name)
        while True:
            data = await websocket.receive_json()
            kind = data.get("task")

            if kind == "corpus_ack":
                coordinator.mark_corpus_ack(team_name)
                continue

            if kind == "ae":
                # An unparseable action must STILL mark the team as responded
                # for this step (move defaults to STAY); otherwise the per-step
                # AE gate blocks the full AE_TIME_CUTOFF waiting on it.
                result = data.get("result")
                if not isinstance(result, dict):
                    result = {}
                step = result.get("step")
                try:
                    action = Action(result["action"]).value
                except Exception:
                    action = None
                    logger.info(
                        f"Invalid AE action from {team_name}, move wasted: {data!r}"
                    )
                coordinator.record_ae_action(team_name, step, action)
                continue

            if kind in ("mission_batch", "noise"):
                batch_id = data.get("batch_id")
                if batch_id not in coordinator._pending_batches:
                    logger.info(
                        f"dropping stale {kind} from {team_name} (batch_id={batch_id})"
                    )
                    continue
                coordinator.resolve_batch_reply(
                    team_name, batch_id, data.get("results", [])
                )
                continue

            logger.debug(f"unhandled task {kind!r} from {team_name}")
    except (WebSocketDisconnect, ConnectionClosed, RuntimeError) as exc:
        logger.info(f"[{team_name}] disconnected: {exc!r}")
    finally:
        await coordinator.ws.team_disconnect(team_name, websocket)
