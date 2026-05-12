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

from ae_manager import DEFAULT_CACHE_PATH, DEFAULT_POLICY_KWARGS, AEManager  # noqa: E402
from map_memory import MapMemory  # noqa: E402
from policy import HeuristicPolicy  # noqa: E402


def _build_managers(
    env: Bomberman,
    policy_factory,
    cache_path,
) -> dict[str, AEManager]:
    """One AEManager per agent. Each bot has an isolated MapMemory; if a
    cache_path is provided and exists, every bot pre-loads it (so round 1
    starts with full map knowledge)."""
    cached_template: MapMemory | None = None
    if cache_path is not None and cache_path.exists():
        cached_template = MapMemory.load(cache_path)

    out: dict[str, AEManager] = {}
    for agent in env.possible_agents:
        mem = MapMemory()
        if cached_template is not None:
            mem.merge_static_from(cached_template)
        out[agent] = AEManager(policy=policy_factory(), memory=mem)
    return out


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


_P = DEFAULT_POLICY_KWARGS  # short alias for argparse default= expressions below


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

    # Feature toggles — defaults come from DEFAULT_POLICY_KWARGS in ae_manager.py
    # so this script always mirrors the production container configuration.
    parser.add_argument("--predictive-bomb", dest="predictive_bomb",
                        action="store_true", default=_P["predictive_bomb"],
                        help="Bomb when an enemy is *likely* to be in blast at detonation")
    parser.add_argument("--no-predictive-bomb", dest="predictive_bomb",
                        action="store_false")
    parser.add_argument("--bomb-threshold", type=float,
                        default=_P["predictive_bomb_threshold"],
                        help="Min expected enemy hits required for a predictive bomb")

    parser.add_argument("--wall-breaking", dest="wall_breaking",
                        action="store_true", default=_P["wall_breaking"],
                        help="Allow pathfinding to route through destructible walls")
    parser.add_argument("--no-wall-breaking", dest="wall_breaking",
                        action="store_false")
    parser.add_argument("--wall-break-cost", type=float, default=_P["wall_break_cost"],
                        help="Extra path cost (≈ ticks lost) to break a wall")

    parser.add_argument("--smart-defend", dest="smart_defend",
                        action="store_true", default=_P["smart_defend"],
                        help="Pre-position between enemy and base when defending; "
                             "expand defend radius when base health is low")
    parser.add_argument("--no-smart-defend", dest="smart_defend",
                        action="store_false")

    parser.add_argument("--drift-aware-bomb", dest="drift_aware_bomb",
                        action="store_true", default=_P["drift_aware_bomb"],
                        help="Use velocity-biased enemy distribution for predictive bombing")
    parser.add_argument("--no-drift-aware-bomb", dest="drift_aware_bomb",
                        action="store_false")

    parser.add_argument("--auto-tune-bomb", dest="auto_tune_bomb",
                        action="store_true", default=_P["auto_tune_bomb"],
                        help="Adaptively tune the bomb threshold via EMA of observed hit rate")
    parser.add_argument("--no-auto-tune-bomb", dest="auto_tune_bomb",
                        action="store_false")
    parser.add_argument("--bomb-tune-target", type=float, default=_P["bomb_tune_target"],
                        help="Target predictive-bomb hit rate for auto-tuning")

    parser.add_argument("--bomb-economy", dest="bomb_economy",
                        action="store_true", default=_P["bomb_economy"],
                        help="Unified value scoring: only bomb when score >= bomb_reserve_threshold")
    parser.add_argument("--no-bomb-economy", dest="bomb_economy",
                        action="store_false")
    parser.add_argument("--base-bomb-value", type=float, default=_P["base_bomb_value"],
                        help="Value of hitting an enemy base in agent-hit units")
    parser.add_argument("--agent-bomb-value", type=float, default=_P["agent_bomb_value"],
                        help="Value of a single definite agent hit")
    parser.add_argument("--bomb-reserve-threshold", type=float,
                        default=_P["bomb_reserve_threshold"],
                        help="Minimum score required to place a bomb under economy mode")
    parser.add_argument("--wall-break-tile-threshold", type=float,
                        default=_P["wall_break_tile_threshold"],
                        help="Min tile value behind wall to justify a wall-break bomb; "
                             "0.0 = always break")

    parser.add_argument("--loop-detection", dest="loop_detection",
                        action="store_true", default=_P["loop_detection"],
                        help="Detect and break 2- or 3-step (action, position) cycles")
    parser.add_argument("--no-loop-detection", dest="loop_detection",
                        action="store_false")
    parser.add_argument("--loop-window", type=int, default=_P["loop_window"],
                        help="Past (action, pos) entries retained for cycle detection; "
                             "must be >= 5 to catch period-3 loops")

    parser.add_argument("--proactive-base-routing", dest="proactive_base_routing",
                        action="store_true", default=_P["proactive_base_routing"],
                        help="Include known enemy base cells in collect scoring")
    parser.add_argument("--no-proactive-base-routing", dest="proactive_base_routing",
                        action="store_false")
    parser.add_argument("--base-route-weight", type=float, default=_P["base_route_weight"],
                        help="Synthetic tile value for enemy base routing "
                             "(comparable to MISSION=5, RESOURCE=2, RECON=1)")

    parser.add_argument("--adaptive-base-weight", dest="adaptive_base_weight",
                        action="store_true", default=_P["adaptive_base_weight"],
                        help="Auto-adjust base-route weight based on enemy aggression "
                             "(requires --proactive-base-routing)")
    parser.add_argument("--no-adaptive-base-weight", dest="adaptive_base_weight",
                        action="store_false")
    parser.add_argument("--base-weight-min", type=float, default=_P["base_weight_min"],
                        help="Floor weight after a detected attack")
    parser.add_argument("--base-weight-ramp-rate", type=float,
                        default=_P["base_weight_ramp_rate"],
                        help="Weight increase per step during the ramp phase")
    parser.add_argument("--base-weight-attack-cooldown", type=int,
                        default=_P["base_weight_attack_cooldown"],
                        help="Steps to hold defensive posture after last attack "
                             "before ramping resumes")

    parser.add_argument("--cache", dest="cache_path", type=Path,
                        default=DEFAULT_CACHE_PATH,
                        help="Pre-load this novice-map cache (default: ae/src/novice_map.json)")
    parser.add_argument("--no-cache", dest="cache_path", action="store_const", const=None,
                        help="Start with empty map memory (for benchmarking)")

    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else default_config()
    cfg.env.render_mode = "human"
    cfg.env.novice = args.novice
    if args.fps is not None:
        cfg.renderer.render_fps = int(args.fps)

    env = Bomberman(cfg)
    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    env.reset(seed=seed)

    def make_policy() -> HeuristicPolicy:
        return HeuristicPolicy(
            predictive_bomb=args.predictive_bomb,
            predictive_bomb_threshold=args.bomb_threshold,
            wall_breaking=args.wall_breaking,
            wall_break_cost=args.wall_break_cost,
            smart_defend=args.smart_defend,
            drift_aware_bomb=args.drift_aware_bomb,
            auto_tune_bomb=args.auto_tune_bomb,
            bomb_tune_target=args.bomb_tune_target,
            bomb_economy=args.bomb_economy,
            base_bomb_value=args.base_bomb_value,
            agent_bomb_value=args.agent_bomb_value,
            bomb_reserve_threshold=args.bomb_reserve_threshold,
            wall_break_tile_threshold=args.wall_break_tile_threshold,
            loop_detection=args.loop_detection,
            loop_window=args.loop_window,
            proactive_base_routing=args.proactive_base_routing,
            base_route_weight=args.base_route_weight,
            adaptive_base_weight=args.adaptive_base_weight,
            base_weight_min=args.base_weight_min,
            base_weight_ramp_rate=args.base_weight_ramp_rate,
            base_weight_attack_cooldown=args.base_weight_attack_cooldown,
        )

    managers = _build_managers(env, make_policy, args.cache_path)
    selected_view = env.possible_agents[0]  # camera/highlight follows this agent

    cache_used = args.cache_path is not None and args.cache_path.exists()
    features = []
    features.append(f"predictive_bomb={'on' if args.predictive_bomb else 'off'}"
                    + (f" (≥{args.bomb_threshold})" if args.predictive_bomb else ""))
    features.append(f"wall_breaking={'on' if args.wall_breaking else 'off'}"
                    + (f" (cost={args.wall_break_cost})" if args.wall_breaking else ""))
    features.append(f"smart_defend={'on' if args.smart_defend else 'off'}")
    features.append(f"drift_aware={'on' if args.drift_aware_bomb else 'off'}")
    features.append(f"auto_tune={'on (target={args.bomb_tune_target})' if args.auto_tune_bomb else 'off'}")
    features.append(f"loop_detection={'on (window=' + str(args.loop_window) + ')' if args.loop_detection else 'off'}")
    features.append(f"proactive_base_routing={'on (weight=' + str(args.base_route_weight) + ')' if args.proactive_base_routing else 'off'}")
    if args.adaptive_base_weight:
        features.append(f"adaptive_base_weight=on (min={args.base_weight_min}, "
                        f"ramp={args.base_weight_ramp_rate}, cooldown={args.base_weight_attack_cooldown})")
    features.append(f"map_cache={'on (' + args.cache_path.name + ')' if cache_used else 'off'}")
    print(f"Auto-play: {len(env.possible_agents)} HeuristicPolicy bots, seed={seed}, "
          f"novice={args.novice}, rounds={args.rounds}")
    print(f"Features:  {' · '.join(features)}")
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
                if args.auto_tune_bomb:
                    sample_policy = next(iter(managers.values()))._policy
                    print(f"  [auto-tune] threshold={sample_policy.tuned_threshold:.3f}  "
                          f"hit_ema={sample_policy._hit_ema:.3f}")
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
