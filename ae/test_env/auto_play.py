"""Auto-play visualization for the heuristic AE agent.

Adapted from til-26-ae/play.py. Every agent in the round is controlled by
its own HeuristicPolicy + isolated MapMemory, so you can watch the bots
play each other.

Usage (from repo root, with the dev environment active):
    python ae/test_env/auto_play.py
    python ae/test_env/auto_play.py --rounds 3 --seed 42 --fps 4

Keys during play:
    Q / ESC   quit
    R         reset to a new round
    T         toggle the tile-respawn-timer overlay
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pygame

# Make ae/src importable as flat top-level modules (matching Docker layout).
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from til_environment.bomberman_env import Bomberman  # noqa: E402
from til_environment.config import default_config, load_config  # noqa: E402

from ae_manager import AEManager  # noqa: E402
from map_memory import MapMemory  # noqa: E402


def _build_managers(env: Bomberman) -> dict[str, AEManager]:
    """One AEManager per agent, each with an isolated MapMemory."""
    return {agent: AEManager(memory=MapMemory()) for agent in env.possible_agents}


def _reset_managers(managers: dict[str, AEManager]) -> None:
    for mgr in managers.values():
        mgr._memory.reset_round()


def _print_round_summary(env: Bomberman, round_idx: int) -> None:
    """Cumulative per-agent reward for the round.

    `env.rewards` is the *step* reward dict, which resets after termination.
    `env.dynamics.rewards._episode` is the cumulative reward we actually want.
    """
    episode = getattr(env.dynamics.rewards, "_episode", {})
    print(f"\n── round {round_idx} over ──")
    rewards = sorted(
        ((a, float(episode.get(a, 0.0))) for a in env.possible_agents),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for a, r in rewards:
        print(f"  {a}  reward={r:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config; defaults to bomberman_config.yaml")
    parser.add_argument("--seed", type=int, default=None,
                        help="Initial seed (random if omitted)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of rounds to play before quitting")
    parser.add_argument("--fps", type=int, default=None,
                        help="Override renderer fps")
    parser.add_argument("--novice", action="store_true", default=True,
                        help="Use the fixed novice map (default)")
    parser.add_argument("--advanced", dest="novice", action="store_false",
                        help="Use a randomized advanced map")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else default_config()
    cfg.env.render_mode = "human"
    cfg.env.novice = args.novice
    if args.fps is not None:
        cfg.renderer.render_fps = int(args.fps)

    env = Bomberman(cfg)
    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    env.reset(seed=seed)

    managers = _build_managers(env)
    selected_view = env.possible_agents[0]  # camera/highlight follows this agent

    print(f"Auto-play: {len(env.possible_agents)} HeuristicPolicy bots, seed={seed}, "
          f"novice={args.novice}, rounds={args.rounds}")
    print("Keys: Q/ESC quit · R reset · T toggle respawn overlay")

    clock = pygame.time.Clock()
    show_respawn = False
    running = True
    rounds_done = 0

    while running and rounds_done < args.rounds:
        # Render once per full round cycle.
        if env.agent_selector.is_first():
            overlay = env.dynamics.respawn_map if show_respawn else None
            env.render(selected_agent_id=selected_view, respawn_overlay=overlay)
            clock.tick(env.cfg.renderer.render_fps)

        # Drain pygame events so the window stays responsive.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    seed = random.randint(0, 99999)
                    env.reset(seed=seed)
                    _reset_managers(managers)
                    print(f"[reset] new seed={seed}")
                elif event.key == pygame.K_t:
                    show_respawn = not show_respawn
                    print(f"[respawn overlay] {'ON' if show_respawn else 'OFF'}")
        if not running:
            break

        agent = env.agent_selection

        # Episode-end housekeeping for this agent.
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            if all(env.terminations.values()) or all(env.truncations.values()):
                rounds_done += 1
                _print_round_summary(env, rounds_done)
                if rounds_done < args.rounds:
                    seed = random.randint(0, 99999)
                    env.reset(seed=seed)
                    _reset_managers(managers)
            continue

        # Policy chooses; AEManager catches its own exceptions and falls back
        # to STAY, so a buggy bot can't take down the whole visualization.
        obs = env.observe(agent)
        action = managers[agent].ae(obs)
        env.step(int(action))

    env.close()
    print(f"\nDone. Played {rounds_done} round(s).")


if __name__ == "__main__":
    main()
