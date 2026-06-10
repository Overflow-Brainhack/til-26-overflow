"""Smoke test: run ONE in-process game with the real agent in seat 0, surfacing
any exception (not swallowed) and dumping the agent's end-state so we can confirm
it actually builds Barracks/Mines/2nd-Base/rings rather than no-opping.

    uv run --no-project --with httpx python _agent_smoke.py [--turns 150] [--field aggressor]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server", "src"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "participant", "src"))

import game_runner  # noqa: E402
from engine.actions import ActionPayload  # noqa: E402
from engine.entities.buildings.base_building import Base  # noqa: E402
from engine.entities.building import Building  # noqa: E402
from engine.entities.unit import Unit  # noqa: E402
from game_runner import GameConfig, GameRunner, PlayerRegistration  # noqa: E402
from schemas.observation import build_observation  # noqa: E402
import seed_eval  # noqa: E402


class Runner(GameRunner):
    def __init__(self, regs, config, actors):
        super().__init__(regs, config)
        self.actors = actors
        self.errors = 0

    async def _collect_actions(self, player_urls):
        state = self.state
        alive = [pid for pid in player_urls if state.players[pid].alive]

        async def one(pid):
            obs = build_observation(state, pid, self.diplomacy, self.chat_log, self.config.max_turns)
            try:
                payload = await self.actors[pid].decide(obs)
            except Exception as exc:  # surface agent crashes
                if pid == "player-0":
                    import traceback
                    traceback.print_exc()
                    self.errors += 1
                payload = ActionPayload(player_id=pid, turn_number=state.turn_number, actions=[])
            return pid, payload

        return dict(await asyncio.gather(*[one(pid) for pid in alive]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=150)
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--field", default="turtle,aggressor,ambusher,splash,economist,random")
    args = ap.parse_args()

    import logging
    logging.disable(logging.CRITICAL)
    game_runner.ReplayRecorder = seed_eval._NullRecorder

    roster = seed_eval.build_roster("agent", args.field, 20)
    ids = [f"player-{i}" for i in range(20)]
    regs = [PlayerRegistration(pid, f"{roster[i]}:{pid}", "local://x") for i, pid in enumerate(ids)]
    actors = {pid: seed_eval.ARCHETYPES[roster[i]]() for i, pid in enumerate(ids)}
    cfg = GameConfig(seed=args.seed, map_width=35, map_height=30, max_turns=args.turns)
    runner = Runner(regs, cfg, actors)
    runner.initialise()
    asyncio.run(runner.run())

    st = runner.state
    me = "player-0"
    bld = [e for e in st.entities.values() if isinstance(e, Building) and e.owner_id == me]
    units = [e for e in st.entities.values() if isinstance(e, Unit) and e.owner_id == me]
    bases = [b for b in bld if isinstance(b, Base)]
    complete_bases = [b for b in bases if b.is_complete]
    from collections import Counter
    bc = Counter(b.entity_type() for b in bld)
    uc = Counter(u.entity_type() for u in units)
    print(f"\n=== agent (player-0) end state @ turn {st.turn_number} (seed {args.seed}, field={args.field}) ===")
    print(f"  alive            : {st.players[me].alive}")
    print(f"  agent exceptions : {runner.errors}")
    print(f"  gold             : {st.players[me].resources.to_dict().get('gold')}")
    print(f"  bases            : {len(bases)} ({len(complete_bases)} complete)")
    print(f"  buildings        : {dict(bc)}")
    print(f"  units            : {dict(uc)}  (total {len(units)})")
    # ring coverage of complete bases
    for b in complete_bases:
        ring = st.grid.neighbors(b.coord)
        occ = sum(1 for c in ring if st.is_ground_blocked(c))
        print(f"  base {b.coord}: ring {occ}/6 occupied")
    print(f"  survivors total  : {len(st.alive_players())}/20")


if __name__ == "__main__":
    main()
