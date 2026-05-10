"""Capture the novice-mode map and save it for the production agent.

In novice mode the simulator hardcodes maze seed 19 (arena.py:454) and
episode seed 88 (dynamics.py:304), so the map is identical every game.
We can therefore play through it once offline, dump the discovered walls /
tiles / base positions to JSON, and bundle that JSON into the Docker
image — the production agent then starts round 1 with full map knowledge
instead of wasting steps on re-exploration.

Usage:
    python ae/test_env/capture_novice_map.py
    python ae/test_env/capture_novice_map.py --rounds 5 --out ae/src/novice_map.json

Run again whenever the simulator's novice map changes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Headless pygame — no window opens.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from til_environment.bomberman_env import Bomberman  # noqa: E402
from til_environment.config import default_config  # noqa: E402

from ae_manager import AEManager  # noqa: E402
from map_memory import MapMemory  # noqa: E402
from policy import HeuristicPolicy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3,
                        help="Rounds to play (more = more thorough exploration)")
    parser.add_argument("--out", type=Path, default=SRC / "novice_map.json",
                        help="Where to save the cache JSON (bundled into Docker)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = default_config()
    cfg.env.render_mode = None
    cfg.env.novice = True

    env = Bomberman(cfg)
    env.reset(seed=42)  # seed is ignored by novice but required by the API

    # All 6 agents stamp into the same MapMemory so we accumulate from every
    # viewpoint at once. Static fields (walls, tile_contents, base_positions)
    # are union-style — overlapping observations don't conflict.
    shared = MapMemory()
    managers = {
        a: AEManager(policy=HeuristicPolicy(), memory=shared)
        for a in env.possible_agents
    }

    if not args.quiet:
        print(f"Capturing novice map across {args.rounds} round(s) → {args.out}")

    safety_cap = 1500  # one round is at most ~num_iters * num_teams steps
    for r in range(args.rounds):
        for _ in range(safety_cap):
            if all(env.terminations.values()) or all(env.truncations.values()):
                break
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            obs = env.observe(agent)
            action = managers[agent].ae(obs)
            env.step(int(action))
        if r < args.rounds - 1:
            env.reset(seed=42 + r + 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shared.save(args.out)

    if not args.quiet:
        from constants import GRID_SIZE
        n_walls = len(shared.blocked_edges)
        n_destr = len(shared.destructible_edges)
        n_tiles = len(shared.tile_contents)
        n_bases = len(shared.base_positions)
        cell_coverage = 100 * n_tiles / (GRID_SIZE * GRID_SIZE)
        print(
            f"Saved {args.out}\n"
            f"  walls:    {n_walls} blocked  ({n_destr} destructible)\n"
            f"  cells:    {n_tiles}/{GRID_SIZE * GRID_SIZE} observed ({cell_coverage:.1f}%)\n"
            f"  bases:    {n_bases}"
        )


if __name__ == "__main__":
    main()
