#!/usr/bin/env python3
"""Single-game debug runner: per-turn trace of player-0's survival picture.

Usage: python surprise_chal/debug_run.py --agent bastion --seed 67 --turns 150
Prints one line per turn: gold, bases (complete/total, hp), unit mix, and any
enemy units within 6 of a base — enough to see exactly how a death unfolds.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter

from evaluate_agent import (
    PLAYER_ID,
    EvaluatorRunner,
    EvalStats,
    GameConfig,
    PlayerRegistration,
    ROOT,
    load_agent,
    make_opponent,
)
from engine.resources import ResourceType


class TraceRunner(EvaluatorRunner):
    async def _collect_actions(self, player_urls):
        self._trace()
        return await super()._collect_actions(player_urls)

    def _trace(self) -> None:
        state = self.state
        assert state is not None
        player = state.players[PLAYER_ID]
        units = state.units_for(PLAYER_ID)
        buildings = state.buildings_for(PLAYER_ID)
        bases = [b for b in buildings if b.__class__.__name__ == "Base"]
        complete = [b for b in bases if b.is_complete]
        umix = Counter(u.__class__.__name__ for u in units)
        bmix = Counter(b.__class__.__name__ for b in buildings)
        base_coords = [b.coord for b in bases]

        threats = []
        for e in state.entities.values():
            if e.owner_id == PLAYER_ID or not hasattr(e, "movement_range"):
                continue
            for bc in base_coords:
                d = state.grid.distance(e.coord, bc)
                if d <= 6:
                    threats.append(f"{e.__class__.__name__[:4]}@{d}")
                    break

        base_desc = " ".join(
            f"({b.coord.q},{b.coord.r})hp{b.hp}{'' if b.is_complete else '*'}"
            for b in bases
        )
        print(
            f"T{state.turn_number:>3} alive={player.alive} "
            f"gold={player.resources.get(ResourceType.GOLD):>5} "
            f"bases={len(complete)}/{len(bases)} [{base_desc}] "
            f"units={dict(umix)} bldgs={dict(bmix)} "
            f"threats={threats[:12]}"
        )
        if not player.alive:
            raise SystemExit(f"player-0 ELIMINATED at turn {state.turn_number}")


async def amain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="bastion")
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--turns", type=int, default=150)
    ap.add_argument("--players", type=int, default=20)
    ap.add_argument("--opponents", default="hard")
    ap.add_argument(
        "--with-shadow",
        action="store_true",
        help="put shadow_agent at player-1 to mirror evaluate_agent.py's layout",
    )
    args = ap.parse_args()

    registrations = [PlayerRegistration(PLAYER_ID, PLAYER_ID, "local://agent")]
    registrations += [
        PlayerRegistration(f"player-{i}", f"player-{i}", "local://opponent")
        for i in range(1, args.players)
    ]
    actors = {PLAYER_ID: load_agent(args.agent)}
    start = 1
    if args.with_shadow:
        actors["player-1"] = load_agent("shadow")
        start = 2
    for i in range(start, args.players):
        actors[f"player-{i}"] = make_opponent(i, args.opponents)

    runner = TraceRunner(
        registrations,
        GameConfig(
            seed=args.seed,
            max_turns=args.turns,
            replay_path=str(ROOT / "replays" / f"debug_seed_{args.seed}.jsonl"),
        ),
        actors,
        {PLAYER_ID: EvalStats(PLAYER_ID)},
    )
    runner.initialise()
    if runner.recorder:
        runner.recorder.close()
        runner.recorder = None
    await runner.run()
    print("game over: survived" if runner.state.players[PLAYER_ID].alive else "game over: dead")


if __name__ == "__main__":
    asyncio.run(amain())
