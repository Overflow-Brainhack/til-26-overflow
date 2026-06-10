"""Your player server. The competition calls POST /observe each turn.

You normally don't edit this — point it at your agent with the AGENT env var
(`algo` or `llm`, or wire in your own class below) and run it:

    python server.py                 # from this src/ directory
    AGENT=llm python server.py       # needs OPENROUTER_API_KEY

It exposes:
    GET  /health   → {"status":"ok", ...}   (competition health-check)
    POST /observe  → your ActionPayload as JSON

Any exception inside your agent is caught and turned into an empty (no-op) turn
so a single bad turn never knocks your server out of the game.
"""

from __future__ import annotations

import json
import logging
import os
import sys

from fastapi import FastAPI

from agent_base import PlayerAgent
from engine.actions import payload_to_dict


class _CloudLoggingFormatter(logging.Formatter):
    """One-line JSON that Cloud Logging (Vertex AI / Cloud Run) auto-parses.

    GCP routes a container's stdout to Cloud Logging. Plain text lands with no
    severity (you can't filter ERROR vs INFO in the Logs Explorer) and a
    multi-line traceback is split into one log entry PER LINE — unreadable. A
    single-line JSON object with `severity` + `message` fixes both: levels are
    filterable and a traceback stays a single entry.
    """

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        return json.dumps(
            {
                # Python level names (DEBUG/INFO/WARNING/ERROR/CRITICAL) map 1:1
                # to Cloud Logging severities.
                "severity": record.levelname,
                "message": msg,
                "logger": record.name,
            },
            default=str,
        )


def _configure_logging() -> None:
    """Send ALL logs (ours, the agent's, and uvicorn's) to stdout — structured
    JSON when running on GCP, readable plain text locally. Override the auto
    choice with LOG_FORMAT=json|text and the level with LOG_LEVEL=DEBUG|INFO|…
    (DEBUG makes /observe log the full action payload each turn).
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    on_gcp = bool(
        os.environ.get("AIP_PREDICT_ROUTE")  # Vertex AI custom prediction container
        or os.environ.get("K_SERVICE")  # Cloud Run
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    use_json = os.environ.get("LOG_FORMAT", "json" if on_gcp else "text").lower() == "json"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _CloudLoggingFormatter()
        if use_json
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Route uvicorn's own loggers through the SAME root handler (drop their
    # default handlers, let them propagate) so startup/access lines are in the
    # same format and never doubled.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


_configure_logging()
log = logging.getLogger("player")


def make_agent() -> PlayerAgent:
    kind = os.environ.get("AGENT", "algo").lower()
    if kind == "llm":
        from llm_agent import LLMAgent  # needs OPENROUTER_API_KEY

        return LLMAgent()
    # default: deterministic, no API key required
    from algo_agent import AlgoAgent

    return AlgoAgent()


_agent = make_agent()
log.info("serving agent: %s", type(_agent).__name__)

app = FastAPI(title="participant player server")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "agent": type(_agent).__name__}


@app.post("/observe")
async def observe(observation: dict) -> dict:
    turn = observation.get("turn_number", 0)
    try:
        out = payload_to_dict(await _agent.decide(observation))
        log.info("turn %s -> %d actions", turn, len(out.get("actions", [])))
        log.debug("turn %s payload: %s", turn, out)  # full payload only at LOG_LEVEL=DEBUG
    except Exception:  # noqa: BLE001 — never crash the turn loop
        # log.exception attaches the traceback (one entry on GCP thanks to the JSON formatter)
        log.exception("turn %s: agent error -> no-op turn", turn)
        out = {
            "player_id": observation.get("player_id", "unknown"),
            "turn_number": turn,
            "actions": [],
        }
    return out


if __name__ == "__main__":
    import uvicorn

    # log_config=None: don't let uvicorn install its own logging config — keep the
    # structured stdout setup from _configure_logging() so uvicorn's startup +
    # access lines flow through the same handler (and show up in Cloud Logging).
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "6700")),
        log_config=None,
    )
