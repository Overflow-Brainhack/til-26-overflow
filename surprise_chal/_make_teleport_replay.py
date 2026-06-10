"""Generate a replay that demonstrates the teleport quirk.

A 2-player game where player-0 owns a single Infantry (movement_range = 1) that
"moves" across the whole map every turn via a single non-adjacent path hop. The
opponent just holds. Output is a standard engine replay you can open in the viewer:

    python server/src/watch_replay.py replays/teleport_demo.jsonl

Throwaway dev artifact — delete when done.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server", "src"))

from engine.actions import ActionPayload, MoveAction, SendChatAction
from engine.entities.unit import Unit
from engine.entities.units.infantry import Infantry
from engine.hex_grid import HexCoord
from game_runner import GameConfig, GameRunner, PlayerRegistration
from schemas.observation import build_observation

SEED = 7
MAP_W, MAP_H = 30, 20
MAX_TURNS = 12
REPLAY = "replays/teleport_demo.jsonl"


def free_cost1_near(state, around: HexCoord) -> HexCoord:
    """Nearest unoccupied, cost-1 (non-Difficult) tile to `around`, spiralling out."""
    grid = state.grid
    around = grid.wrap(around)
    for radius in range(0, 9):
        ring = [around] if radius == 0 else grid.ring(around, radius)
        for c in ring:
            if state.tile(c).movement_cost() == 1 and not state.is_ground_blocked(c):
                return c
    return around


class TeleportRunner(GameRunner):
    """In-process runner: player-0 teleports its Infantry, player-1 holds."""

    def __init__(self, regs, config):
        super().__init__(regs, config)
        self.home: HexCoord | None = None  # landmark A (near own base)
        self.away: HexCoord | None = None  # landmark B (across the map)

    def setup_demo(self):
        state = self.state
        base0 = next(e for e in state.entities.values() if e.owner_id == "player-0")
        base1 = next(e for e in state.entities.values() if e.owner_id == "player-1")
        self.home = free_cost1_near(state, base0.coord)
        self.away = free_cost1_near(state, base1.coord)
        state.add_entity(Infantry("player-0", self.home))  # the teleporter
        d = state.grid.distance(self.home, self.away)
        print(f"player-0 base ~({base0.coord.q},{base0.coord.r})  HOME=({self.home.q},{self.home.r})")
        print(f"player-1 base ~({base1.coord.q},{base1.coord.r})  AWAY=({self.away.q},{self.away.r})")
        print(f"each turn the move-1 Infantry will jump ~{d} tiles between HOME and AWAY\n")

    async def _collect_actions(self, player_urls):
        state = self.state
        out: dict[str, ActionPayload] = {}
        for pid in player_urls:
            if not state.players[pid].alive:
                continue
            build_observation(state, pid, self.diplomacy, self.chat_log, self.config.max_turns)
            if pid == "player-0":
                out[pid] = self._teleport_action()
            else:
                out[pid] = ActionPayload(pid, state.turn_number, [])  # hold
        return out

    def _teleport_action(self) -> ActionPayload:
        state = self.state
        inf = next(
            (e for e in state.entities.values()
             if e.owner_id == "player-0" and isinstance(e, Unit)),
            None,
        )
        if inf is None:
            return ActionPayload("player-0", state.turn_number, [])
        here = inf.coord
        # alternate destination each turn → a big jump every time
        target = self.away if state.turn_number % 2 == 0 else self.home
        if (target.q, target.r) == (here.q, here.r):
            target = self.home if (target is self.away) else self.away
        dist = state.grid.distance(here, target)
        note = (f"TELEPORT t{state.turn_number}: ({here.q},{here.r})->"
                f"({target.q},{target.r})  dist={dist}  (Infantry movement_range=1)")
        return ActionPayload(
            "player-0",
            state.turn_number,
            [
                # single NON-ADJACENT hop — the engine accepts it (no adjacency check)
                MoveAction(unit_id=inf.id, path=[here, target]),
                SendChatAction(text=note, recipient_id=None),
            ],
        )


def main():
    regs = [
        PlayerRegistration("player-0", "Teleporter", "local://p0"),
        PlayerRegistration("player-1", "Holder", "local://p1"),
    ]
    config = GameConfig(seed=SEED, map_width=MAP_W, map_height=MAP_H,
                        max_turns=MAX_TURNS, replay_path=REPLAY)
    runner = TeleportRunner(regs, config)
    runner.initialise()
    runner.setup_demo()
    asyncio.run(runner.run())
    print(f"\nwrote {REPLAY}")


if __name__ == "__main__":
    main()
