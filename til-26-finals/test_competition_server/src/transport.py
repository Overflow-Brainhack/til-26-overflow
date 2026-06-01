"""WebSocket connection state and broadcast helpers.

WebSocketManager owns every team WebSocket slot. It is the
single place that knows whether a connection is alive, handles the
dead-connection probe-and-replace logic, and provides broadcast helpers
used throughout the match lifecycle.

Invariants:
  - team_connections maps every team name to its WebSocket or None.
    The dict is initialised at construction and never gains/loses keys.
  - All mutations go through the public methods; callers may read the
    dicts directly — they are public by design.
"""

import asyncio
import logging

from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger("uvicorn.error")


class WebSocketManager:
    def __init__(self, team_names: list[str]) -> None:
        self.team_names = team_names
        self.team_connections: dict[str, WebSocket | None] = {
            name: None for name in team_names
        }
        # Per-team locks serialize EVERY write to a given team's socket. They
        # protect against two races:
        #  (1) two simultaneous connections both seeing slot=None before either
        #      sets it (accept() yields the loop), and
        #  (2) concurrent writers (mission_batch + AE observation + the
        #      websockets library's auto-PONG response to a client PING) all
        #      hitting the same connection, which trips _drain_helper's
        #      `assert waiter is None or waiter.cancelled()` in the legacy
        #      websockets implementation.
        self._team_locks: dict[str, asyncio.Lock] = {
            name: asyncio.Lock() for name in team_names
        }

    # ── serialized sends (use these; never call .send_json/.send_text on a
    #    stored connection directly — concurrent writers will trip the
    #    websockets library's _drain_helper assertion) ─────────────────────

    async def send_to_team(self, team_name: str, payload: dict | str) -> bool:
        """Send to one team with the per-team lock held. Returns True on
        success, False if the slot is empty or the send failed."""
        if self.team_connections.get(team_name) is None:
            return False
        async with self._team_locks[team_name]:
            conn = self.team_connections.get(team_name)
            if conn is None:
                return False
            try:
                if isinstance(payload, str):
                    await conn.send_text(payload)
                else:
                    await conn.send_json(payload)
                return True
            except (WebSocketDisconnect, ConnectionClosed, RuntimeError):
                return False

    # ── team connections ──────────────────────────────────────────────────

    async def team_connect(self, websocket: WebSocket, team_name: str) -> None:
        if team_name not in self.team_names:
            logger.info(f"[{team_name}] not a valid team name, closing")
            await websocket.accept()
            await websocket.close(code=1008, reason=f"Invalid team {team_name}")
            return
        async with self._team_locks[team_name]:
            existing_team = self.team_connections[team_name]
            if existing_team is None:
                await websocket.accept()
                self.team_connections[team_name] = websocket
                logger.info(f"[{team_name}] connected (fresh slot)")
            else:
                logger.info(
                    f"[{team_name}] duplicate connect — probing existing connection"
                )
                try:
                    await existing_team.send_json({"type": "health"})
                except (WebSocketDisconnect, ConnectionClosed, RuntimeError) as exc:
                    logger.info(
                        f"[{team_name}] existing connection dead ({exc!r}); replacing"
                    )
                    await self.team_disconnect(team_name)
                    await websocket.accept()
                    self.team_connections[team_name] = websocket
                else:
                    logger.info(
                        f"[{team_name}] existing connection alive; rejecting new"
                    )
                    await websocket.accept()
                    await websocket.close(
                        code=1008,
                        reason=f"There is already a team connected with name {team_name}!",
                    )
        logger.info(
            f"team_connections: { {k: (v is not None) for k, v in self.team_connections.items()} }"
        )

    async def team_disconnect(
        self,
        team_name: str,
        websocket: WebSocket | None = None,
        message: str = "Disconnected",
    ) -> bool:
        """Close and clear a team's slot.

        If *websocket* is supplied the slot is only cleared when it still holds
        that exact object — guards against the race where a reconnecting
        client's new WebSocket replaces the slot before the old handler fires.

        Returns True if the slot was actually cleared, False if it was a no-op
        (slot held a different socket).
        """
        current = self.team_connections[team_name]
        if websocket is not None and current is not websocket:
            logger.info(
                f"[{team_name}] team_disconnect called for a stale websocket; "
                f"current slot already replaced — skipping close"
            )
            return False
        try:
            if current is not None:
                await current.close(reason=message)
        except Exception:
            pass
        self.team_connections[team_name] = None
        logger.info(f"[{team_name}] slot cleared")
        return True

    # ── broadcast helpers ─────────────────────────────────────────────────

    async def broadcast_teams(self, message: dict) -> list:
        return await asyncio.gather(
            *[
                self.send_to_team(name, message)
                for name, conn in self.team_connections.items()
                if conn is not None
            ],
            return_exceptions=True,
        )
