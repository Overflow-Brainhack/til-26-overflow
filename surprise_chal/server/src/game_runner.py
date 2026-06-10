"""game runner: orchestrates turn loop, dispatches observations, collects actions"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

import httpx

from engine.actions import ActionPayload, payload_from_dict
from engine.chat import ChatLog, ChatMessage
from engine.constants import (
    DEFAULT_MAP_HEIGHT,
    DEFAULT_MAP_WIDTH,
    MAX_TURNS,
    RESPONSE_TIMEOUT_SECONDS,
)
from engine.diplomacy import DiplomacyManager
from engine.entities.buildings.base_building import Base
from engine.hex_grid import HexGrid
from engine.map_gen import MapGenerator
from engine.player import Player
from engine.resources import ResourceBag
from engine.state import GameState
from engine.turn_processor import TurnProcessor
from replay.recorder import ReplayRecorder
from schemas.observation import build_observation

log = logging.getLogger(__name__)


@dataclass
class PlayerRegistration:
    player_id: str
    name: str
    callback_url: str  # e.g. http://player-0:6700


@dataclass
class GameConfig:
    seed: int = 42
    map_width: int = DEFAULT_MAP_WIDTH
    map_height: int = DEFAULT_MAP_HEIGHT
    max_turns: int = MAX_TURNS
    response_timeout: float = RESPONSE_TIMEOUT_SECONDS
    replay_path: str | None = None


def _dm_deliverable(sender_id: str, recipient_id: str | None, state: GameState) -> bool:
    """Whether a player chat message may be posted to the log.

    Global messages (``recipient_id is None``) always post. A DM only posts if its
    recipient is a CURRENTLY-ALIVE player the sender has actually MET — i.e. the
    recipient is in the sender's ``known_player_ids`` (met by sight, or having
    previously DM'd the sender, which is how a reply is allowed). This:
      - blocks DMs to dead/unknown players (a dead player never reads its mail), and
      - blocks "cold-call" DMs to a guessed id of a player you've never encountered,
        keeping diplomacy fog-consistent. Broadcasting globally grants no DM right.
    A dropped DM is a silent no-op and establishes NO meeting (see turn_processor).
    """
    if recipient_id is None:
        return True
    target = state.players.get(recipient_id)
    if target is None or not target.alive:
        return False
    sender = state.players.get(sender_id)
    if sender is None:
        return False
    return recipient_id in sender.known_player_ids


class GameRunner:
    def __init__(
        self, registrations: list[PlayerRegistration], config: GameConfig
    ) -> None:
        self.registrations = registrations
        self.config = config
        self.game_id = str(uuid.uuid4())[:8]
        self.state: GameState | None = None
        self.diplomacy = DiplomacyManager()
        self.chat_log = ChatLog()
        self.recorder: ReplayRecorder | None = None
        self._running = False

    def initialise(self) -> None:
        """generate the map, place starting bases, build initial game state"""
        n = len(self.registrations)
        gen = MapGenerator(
            self.config.seed, self.config.map_width, self.config.map_height, n
        )
        tiles, starts = gen.generate()
        grid = HexGrid(self.config.map_width, self.config.map_height)

        players: dict[str, Player] = {}
        for reg in self.registrations:
            players[reg.player_id] = Player(
                id=reg.player_id,
                name=reg.name,
                resources=ResourceBag(gold=500),
            )

        self.state = GameState(grid=grid, tiles=tiles, players=players, entities={})

        for i, reg in enumerate(self.registrations):
            base = Base(reg.player_id, starts[i])
            self.state.add_entity(base)

        # Default replay name is a dd-mm-yyyy-HHMMSS (24-hour) timestamp, with the
        # short game id appended so two games started in the same second can't clash.
        timestamp = datetime.now().strftime("%d-%m-%Y-%H%M%S")
        replay_path = (
            self.config.replay_path or f"replays/{timestamp}_{self.game_id}.jsonl"
        )
        self.recorder = ReplayRecorder(replay_path)

        self.chat_log.post_system(
            0, f"game started with {n} players, seed={self.config.seed}"
        )
        self.recorder.record_initial(
            self.state,
            self.config.seed,
            chat=[m.to_dict() for m in self.chat_log.messages],
            treaties=self.diplomacy.active_treaty_pairs(),
        )
        log.info(
            "game %s initialised: %d players, map %dx%d",
            self.game_id,
            n,
            self.config.map_width,
            self.config.map_height,
        )

    async def run(self) -> None:
        """main async turn loop"""
        if self.state is None:
            self.initialise()

        assert self.state is not None
        self._running = True
        processor = TurnProcessor(self.state, self.diplomacy)
        player_urls = {reg.player_id: reg.callback_url for reg in self.registrations}

        ended_by_elimination = False
        while self._running and self.state.turn_number < self.config.max_turns:
            # game is decided early only when ≤1 player remains (last one standing,
            # or nobody on a mutual elimination). Reaching the turn limit with
            # several players still alive is NOT a loss — see the post-loop banner.
            alive = self.state.alive_players()
            if len(alive) <= 1:
                winner = alive[0].name if alive else "nobody"
                self.chat_log.post_system(
                    self.state.turn_number, f"game over — winner: {winner}"
                )
                ended_by_elimination = True
                break

            # gather observations and dispatch
            action_map = await self._collect_actions(player_urls)

            chat_before = len(self.chat_log.messages)

            # process chat messages from actions (before turn processor).
            # DMs to a dead/unknown/unmet player are dropped (see _dm_deliverable)
            # so the log isn't flooded with messages no living player will read.
            for pid, payload in action_map.items():
                from engine.actions import SendChatAction

                for act in payload.actions:
                    if not isinstance(act, SendChatAction):
                        continue
                    if not _dm_deliverable(pid, act.recipient_id, self.state):
                        continue
                    self.chat_log.post(
                        ChatMessage(
                            turn=self.state.turn_number,
                            sender_id=pid,
                            text=act.text,
                            recipient_id=act.recipient_id,
                        )
                    )
                    # A DELIVERED DM lets the recipient reply later: they now "know"
                    # the sender (one-directional — sight meetings are handled by the
                    # turn processor). Only delivered DMs establish this, so a dropped
                    # cold guess can never bootstrap a meeting.
                    if act.recipient_id is not None:
                        self.state.players[act.recipient_id].known_player_ids.add(pid)

            # process the turn
            events = processor.process_turn(action_map)

            # post system events to chat
            for event in events:
                if "eliminated" in event:
                    pid = event.split("_")[-1]
                    name = self.state.players[pid].name
                    self.chat_log.post_system(
                        self.state.turn_number, f"{name} has been eliminated"
                    )
                elif (
                    event.startswith("treaty_formed_")
                    or event.startswith("treaty_breaking_")
                    or event.startswith("treaty_expired_")
                ):
                    parts = event.split("_")
                    # formats: treaty_formed_peace_A_B / treaty_breaking_peace_A_B / treaty_expired_peace_A_B
                    pid_a, pid_b = parts[3], parts[4]
                    name_a = (
                        self.state.players[pid_a].name
                        if pid_a in self.state.players
                        else pid_a
                    )
                    name_b = (
                        self.state.players[pid_b].name
                        if pid_b in self.state.players
                        else pid_b
                    )
                    # Every treaty message names BOTH parties so the omniscient
                    # "All" chat view shows who is involved — yet each message is a
                    # private DM to one party only (recipient_id below), never global
                    # and never sent to a third team. Treaties stay secret to outsiders.
                    if event.startswith("treaty_formed_"):
                        msgs = {
                            pid_a: f"peace treaty formed: {name_a} <-> {name_b}",
                            pid_b: f"peace treaty formed: {name_a} <-> {name_b}",
                        }
                    elif event.startswith("treaty_breaking_"):
                        # pid_a is the initiator (see turn_processor event format)
                        msgs = {
                            pid_a: f"{name_a} initiated a treaty break with {name_b} — 5 turns until war",
                            pid_b: f"{name_a} is breaking the peace treaty with {name_b} — 5 turns until war",
                        }
                    else:  # treaty_expired
                        msgs = {
                            pid_a: f"peace treaty expired: {name_a} <-> {name_b} — now at war",
                            pid_b: f"peace treaty expired: {name_a} <-> {name_b} — now at war",
                        }
                    for pid, text in msgs.items():
                        self.chat_log.post(
                            ChatMessage(
                                turn=self.state.turn_number,
                                sender_id="__system__",
                                text=text,
                                recipient_id=pid,  # private DM — only this party sees it
                            )
                        )

            # record the turn with new chat messages
            new_chat = [m.to_dict() for m in self.chat_log.messages[chat_before:]]
            if self.recorder:
                self.recorder.record_turn(
                    self.state.turn_number,
                    action_map,
                    self.state,
                    chat=new_chat,
                    treaties=self.diplomacy.active_treaty_pairs(),
                )

            log.info(
                "turn %d complete, %d players alive",
                self.state.turn_number,
                len(self.state.alive_players()),
            )

        # If the loop exited by hitting the turn limit (not by elimination), every
        # player still alive at the deadline is a co-winner. Survival IS a victory
        # condition: outlasting the clock with your Base intact wins, shared among
        # all survivors. (A Base whose construction finishes after you were already
        # eliminated does not revive you — elimination is permanent.)
        if not ended_by_elimination:
            survivors = self.state.alive_players()
            if survivors:
                names = ", ".join(p.name for p in survivors)
                label = "winner" if len(survivors) == 1 else "co-winners"
                self.chat_log.post_system(
                    self.state.turn_number,
                    f"game over (turn limit) — {label}: {names}",
                )
            else:
                self.chat_log.post_system(
                    self.state.turn_number, "game over (turn limit) — no survivors"
                )

        if self.recorder:
            self.recorder.close()

    async def _collect_actions(
        self, player_urls: dict[str, str]
    ) -> dict[str, ActionPayload]:
        """send observations to alive players in parallel; return collected actions.

        Dead players are skipped entirely — no observation is sent and no actions
        are processed for them. Unit decay for eliminated players is still applied
        by the TurnProcessor unconditionally.
        """
        assert self.state is not None
        state = self.state

        alive_urls = {
            pid: url for pid, url in player_urls.items() if state.players[pid].alive
        }

        async def fetch(
            client: httpx.AsyncClient, pid: str, url: str
        ) -> tuple[str, ActionPayload | None]:
            obs = build_observation(
                state, pid, self.diplomacy, self.chat_log, self.config.max_turns
            )
            try:
                resp = await client.post(
                    f"{url}/observe",
                    json=obs,
                    timeout=self.config.response_timeout,
                )
                resp.raise_for_status()
                return pid, payload_from_dict(resp.json())
            except Exception as exc:
                log.warning("player %s failed to respond: %s", pid, exc)
                return pid, None

        async with httpx.AsyncClient() as client:
            tasks = [fetch(client, pid, url) for pid, url in alive_urls.items()]
            results = await asyncio.gather(*tasks)

        action_map: dict[str, ActionPayload] = {}
        for pid, payload in results:
            if payload is not None:
                action_map[pid] = payload
            else:
                # no-op payload for alive players who didn't respond in time
                action_map[pid] = ActionPayload(
                    player_id=pid,
                    turn_number=self.state.turn_number,
                    actions=[],
                )
        return action_map

    def stop(self) -> None:
        self._running = False
