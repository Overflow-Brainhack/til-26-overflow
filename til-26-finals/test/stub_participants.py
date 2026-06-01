"""Stub participants for initial-test runs.

One process, N websocket clients. For every team in the configured match
*except* REAL_TEAM_NAME, opens a websocket to the competition server and:

  * acks the corpus broadcast immediately
  * returns a random *valid* action (respects the action_mask) for AE steps
  * returns an empty/dummy result for ASR / CV / NLP missions
  * exits cleanly on "done"

This lets you run the full bomberman match locally with a single real
participant container while the other 5 seats are filled by no-op stubs.

Env vars:
  COMPETITION_SERVER_HOST   default "til-competition"
  COMPETITION_SERVER_PORT   default "8000"
  CONFIG                    name of the json under competition_server/configs
                            (without .json). Read to discover the team list.
  CONFIGS_DIR               path to the configs directory. Default
                            "/workspace/configs" (matches the docker-compose
                            volume mount).
  REAL_TEAM_NAME            team name that is being played by a real
                            participant container; this script will NOT
                            stub that team.
"""

import asyncio
import base64
import io
import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path
from urllib.parse import quote

import websockets
from ae_manager import AEManager
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("stub_participants")


SERVER_HOST = os.environ.get("COMPETITION_SERVER_HOST", "til-competition")
SERVER_PORT = os.environ.get("COMPETITION_SERVER_PORT", "8000")
CONFIG = os.environ["CONFIG"]
CONFIGS_DIR = Path(os.environ.get("CONFIGS_DIR", "/workspace/configs"))
REAL_TEAM_NAME = os.environ.get("REAL_TEAM_NAME", "")
# Comma-separated team names whose noise responses are all-black images.
BLACK_IMAGE_TEAMS = {
    t.strip() for t in os.environ.get("BLACK_IMAGE_TEAMS", "").split(",") if t.strip()
}

# 16 MiB matches finals/src/participant_server.py — RAG corpus can be sizable
# 256 MiB. Matches the server's --ws-max-size; well above any plausible
# mission_batch / corpus payload.
MAX_WS_MESSAGE = 2**28


def load_team_names() -> list[str]:
    with open(CONFIGS_DIR / f"{CONFIG}.json") as f:
        cfg = json.load(f)
    teams = list(cfg["teams"])
    # Mirror the HQ (config.py): if the real team isn't a configured seat, the HQ seats it
    # at slot 0, displacing teams[0]. Apply the same rule here so we stub the surviving
    # seats and never spawn the displaced one (which the HQ would now reject → hot loop).
    if REAL_TEAM_NAME and teams and REAL_TEAM_NAME not in teams:
        teams = [REAL_TEAM_NAME, *teams[1:]]
    return teams


def random_valid_action(observation: dict) -> int:
    """Sample uniformly from actions allowed by the bomberman action_mask.

    Falls back to STAY (4) if the mask is missing or all-zero.
    """
    mask = observation.get("action_mask")
    if mask:
        valid = [i for i, v in enumerate(mask) if v]
        if valid:
            return random.choice(valid)
    return 4  # Action.STAY


def black_b64(src_b64: str) -> str:
    """Return a same-size all-black JPEG encoded as base64."""
    with Image.open(io.BytesIO(base64.b64decode(src_b64))) as img:
        black = Image.new(img.mode, img.size, 0)
    buf = io.BytesIO()
    black.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def dummy_batch_result(task: str, item: dict, *, team_name: str = "") -> dict:
    """Return a no-op per-item prediction for a mission_batch item."""
    task_id = item.get("task_id")
    if task == "asr":
        return {"task_id": task_id, "answer": ""}
    if task == "cv":
        return {"task_id": task_id, "detections": []}
    if task == "nlp":
        return {"task_id": task_id, "answer": "", "documents": []}
    if task == "noise":
        src = item.get("b64", "")
        b64 = black_b64(src) if (team_name in BLACK_IMAGE_TEAMS and src) else src
        return {"task_id": task_id, "b64": b64}
    return {"task_id": task_id}


async def run_stub(team_name: str) -> None:
    """One stub participant. Reconnects if the connection drops mid-match."""
    ae_manager = AEManager()
    uri = quote(f"ws://{SERVER_HOST}:{SERVER_PORT}/ws/{team_name}", safe="/:")
    async for websocket in websockets.connect(uri, max_size=MAX_WS_MESSAGE):
        logger.info(f"[{team_name}] connected to {SERVER_HOST}:{SERVER_PORT}")
        try:
            async for raw in websocket:
                if not isinstance(raw, str):
                    continue

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")
                if msg_type == "task" and data.get("task") == "ae":
                    obs = data.get("observation", {})
                    action = ae_manager.ae(obs)
                    await websocket.send(
                        json.dumps(
                            {
                                "task": "ae",
                                "result": {
                                    "step": obs["step"],
                                    "action": action,
                                },
                            }
                        )
                    )
                elif msg_type == "noise":
                    items = data.get("items", [])
                    batch_id = data.get("batch_id")

                    # black_b64 is PIL-heavy; json.dumps serialises large
                    # base64 payloads. Both block the event loop. With 6
                    # stubs in one process all receiving noise simultaneously,
                    # the serial PIL calls stack up and stochastically push
                    # responses past the server's noise timeout.
                    def _build_noise_results(_items, _team):
                        return [
                            dummy_batch_result("noise", it, team_name=_team)
                            for it in _items
                        ]

                    results = await asyncio.to_thread(
                        _build_noise_results, items, team_name
                    )
                    text = await asyncio.to_thread(
                        json.dumps,
                        {
                            "task": "mission_batch",
                            "batch_id": batch_id,
                            "results": results,
                        },
                    )
                    await websocket.send(text)
                elif msg_type == "mission_batch":
                    task = data.get("task")
                    items = data.get("items", [])
                    await websocket.send(
                        json.dumps(
                            {
                                "task": "mission_batch",
                                "batch_id": data.get("batch_id"),
                                "results": [
                                    dummy_batch_result(task, it, team_name=team_name)
                                    for it in items
                                ],
                            }
                        )
                    )
                elif msg_type == "corpus":
                    await websocket.send(json.dumps({"task": "corpus_ack", "ok": True}))
                elif msg_type == "done":
                    logger.info(f"[{team_name}] received done; exiting")
                    return
                elif msg_type == "health":
                    await websocket.send(json.dumps({"task": "health_ack"}))
                elif msg_type == "pong":
                    pass
                else:
                    logger.debug(f"[{team_name}] ignoring {msg_type}: {data}")
        except websockets.ConnectionClosed:
            logger.info(f"[{team_name}] disconnected, retrying")
            continue
        except Exception:
            logger.error(f"[{team_name}] stub crashed:\n{traceback.format_exc()}")
            continue


async def main() -> None:
    teams = load_team_names()
    stub_teams = [t for t in teams if t != REAL_TEAM_NAME]
    if not stub_teams:
        logger.error(
            f"no teams to stub (REAL_TEAM_NAME={REAL_TEAM_NAME!r}, teams={teams})"
        )
        return
    logger.info(
        f"launching stubs for {stub_teams} (real participant: {REAL_TEAM_NAME!r})"
    )
    await asyncio.gather(*(run_stub(t) for t in stub_teams), return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
