"""Corpus distribution phase.

Broadcasts the RAG corpus to all connected teams and waits up to
CORPUS_INGEST_DEADLINE_SEC for corpus_ack from each team.  Teams not
connected at the time of the call are pre-acked so the gate doesn't stall.
"""

import asyncio
import logging

import constants
from artifacts import EventLog
from transport import WebSocketManager

logger = logging.getLogger("uvicorn.error")


class CorpusPhase:
    """Pre-match corpus distribution gate.

    Call run() once.  After it returns every team that could be notified
    has either acked or the deadline has elapsed.

    The corpus_ack dict and the all-acked Event are owned here so the
    WS handler (server.py) calls mark_ack() on this instance.
    """

    def __init__(
        self,
        ws: WebSocketManager,
        corpus_documents: list[dict],
        team_names: list[str],
        events: EventLog,
    ) -> None:
        self._ws = ws
        self._corpus_documents = corpus_documents
        self._team_names = team_names
        self._events = events

        self._acked: dict[str, bool] = {t: False for t in team_names}
        self._all_acked = asyncio.Event()
        self._distributed = False

    def mark_ack(self, team: str) -> None:
        """Called by the WS handler when a corpus_ack message arrives."""
        if self._acked.get(team):
            return
        self._acked[team] = True
        logger.info(f"corpus ack from {team}; status={self._acked}")
        self._events.emit("corpus_ack", team=team)
        if all(self._acked.values()):
            self._all_acked.set()

    async def run(self) -> None:
        """Broadcast corpus, wait for acks. Idempotent."""
        if self._distributed:
            return
        self._distributed = True

        connected = [t for t, c in self._ws.team_connections.items() if c is not None]
        # Pre-ack teams that aren't connected yet so they don't block the gate.
        self._acked = {t: (t not in connected) for t in self._team_names}
        if all(self._acked.values()):
            self._all_acked.set()

        payload = {
            "type": "corpus",
            "documents": self._corpus_documents,
            "deadline_sec": constants.CORPUS_INGEST_DEADLINE_SEC,
        }
        logger.info(
            f"broadcasting corpus ({len(self._corpus_documents)} docs) to "
            f"{len(connected)} connected teams"
        )
        await self._ws.broadcast_teams(payload)

        try:
            await asyncio.wait_for(
                self._all_acked.wait(),
                timeout=constants.CORPUS_INGEST_DEADLINE_SEC,
            )
            logger.info("all connected teams acked corpus")
        except asyncio.TimeoutError:
            logger.info(
                f"corpus ingestion deadline ({constants.CORPUS_INGEST_DEADLINE_SEC}s) "
                f"elapsed; acks={self._acked}"
            )
