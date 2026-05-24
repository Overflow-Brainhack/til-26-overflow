"""Headless self-play benchmark: compare bomb economy parameter combinations.

Sweeps over reserve thresholds, base values, and wall-break tile thresholds,
reporting mean ± std reward per config in a ranked table.

Usage:
    uv run python ae/test_env/benchmark_bomb_economy.py
    uv run python ae/test_env/benchmark_bomb_economy.py --rounds 24 --seeds 0 1 2 3
    uv run python ae/test_env/benchmark_bomb_economy.py --reserve-thresholds 0.5 1.0 2.0 --base-values 5.0 10.0
    uv run python ae/test_env/benchmark_bomb_economy.py --no-baseline
    uv run python ae/test_env/benchmark_bomb_economy.py --wall-thresholds 0.0 1.0 3.0 5.0
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from til_environment.bomberman_env import Bomberman  # noqa: E402
from til_environment.config import default_config  # noqa: E402

from ae_manager import DEFAULT_CACHE_PATH, AEManager  # noqa: E402
from map_memory import MapMemory  # noqa: E402
from policies.policy import HeuristicPolicy  # noqa: E402


def _run_rounds(
    env: Bomberman,
    policy_factory,
    cache_path: Path | None,
    seeds: list[int],
) -> tuple[list[float], HeuristicPolicy | None]:
    """Run one round per seed; return per-round mean rewards + last policy."""
    cached_template: MapMemory | None = None
    if cache_path is not None and cache_path.exists():
        cached_template = MapMemory.load(cache_path)

    round_means: list[float] = []
    last_policy: HeuristicPolicy | None = None

    for seed in seeds:
        env.reset(seed=seed)

        managers: dict[str, AEManager] = {}
        for agent in env.possible_agents:
            mem = MapMemory()
            if cached_template is not None:
                mem.merge_static_from(cached_template)
            policy = policy_factory()
            managers[agent] = AEManager(policy=policy, memory=mem)
            last_policy = policy

        safety_cap = 2000
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

        episode = getattr(env.dynamics.rewards, "_episode", {})
        if episode:
            mean_r = sum(episode.values()) / len(episode)
        else:
            mean_r = 0.0
        round_means.append(mean_r)

    return round_means, last_policy


def _stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, std)."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(variance)


def _col(s: str, width: int) -> str:
    return s.ljust(width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--rounds", type=int, default=12,
        help="Rounds per config (more = lower variance; each ~1-2 s)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit seeds to use; overrides --rounds",
    )
    parser.add_argument(
        "--reserve-thresholds", type=float, nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        help="bomb_reserve_threshold values to sweep (space-separated floats)",
    )
    parser.add_argument(
        "--base-values", type=float, nargs="+",
        default=[2.0, 5.0, 10.0],
        help="base_bomb_value values to sweep (space-separated floats)",
    )
    parser.add_argument(
        "--wall-thresholds", type=float, nargs="+",
        default=[0.0, 1.0, 3.0, 5.0],
        help="wall_break_tile_threshold values to sweep (space-separated floats)",
    )
    parser.add_argument(
        "--agent-bomb-value", type=float, default=1.0,
        help="Fixed agent_bomb_value used across all economy configs (default 1.0)",
    )
    parser.add_argument(
        "--no-baseline", action="store_true", default=False,
        help="Skip the no-economy baseline config",
    )
    parser.add_argument(
        "--cache", dest="cache_path", type=Path, default=DEFAULT_CACHE_PATH,
        help="Novice-map cache to pre-load (default: ae/src/novice_map.json)",
    )
    parser.add_argument(
        "--no-cache", dest="cache_path", action="store_const", const=None,
        help="Start with empty map memory",
    )
    parser.add_argument(
        "--novice", action="store_true", default=True,
        help="Use fixed novice map (default)",
    )
    parser.add_argument(
        "--advanced", dest="novice", action="store_false",
        help="Use randomized advanced map",
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else list(range(args.rounds))

    cfg = default_config()
    cfg.env.render_mode = None
    cfg.env.novice = args.novice
    env = Bomberman(cfg)

    # ── build config list ────────────────────────────────────────────────────
    all_configs: list[tuple[str, dict]] = []

    # Baseline: no economy (existing default behavior)
    if not args.no_baseline:
        all_configs.append(("baseline (no economy)", {
            "bomb_economy": False,
            "wall_breaking": True,
            "smart_defend": True,
            "predictive_bomb": True,
            "drift_aware_bomb": True,
        }))

    # Attack economy sweep: reserve_threshold × base_value, wall_break_tile_threshold=0.0
    for t in args.reserve_thresholds:
        for b in args.base_values:
            label = f"rsv={t:.1f} base={b:.0f}"
            all_configs.append((label, {
                "bomb_economy": True,
                "bomb_reserve_threshold": t,
                "base_bomb_value": b,
                "agent_bomb_value": args.agent_bomb_value,
                "wall_break_tile_threshold": 0.0,
                "wall_breaking": True,
                "smart_defend": True,
                "predictive_bomb": True,
                "drift_aware_bomb": True,
            }))

    # Wall-break sweep: fixed reserve/base, vary wall_break_tile_threshold (skip 0.0)
    for w in args.wall_thresholds:
        if w == 0.0:
            continue  # covered by attack sweep above
        label = f"wall_thr={w:.1f}"
        all_configs.append((label, {
            "bomb_economy": True,
            "bomb_reserve_threshold": 1.5,
            "base_bomb_value": 5.0,
            "agent_bomb_value": args.agent_bomb_value,
            "wall_break_tile_threshold": w,
            "wall_breaking": True,
            "smart_defend": True,
            "predictive_bomb": True,
            "drift_aware_bomb": True,
        }))

    n_configs = len(all_configs)
    map_tag = "novice" if args.novice else "advanced"
    print(f"\nBomb Economy Benchmark: {len(seeds)} seeds × {n_configs} configs  [{map_tag} map]")
    print(f"Seeds: {seeds}")
    print()

    results: list[tuple[str, list[float]]] = []

    for label, kwargs in all_configs:
        def policy_factory(kw=kwargs) -> HeuristicPolicy:
            return HeuristicPolicy(**kw)

        rounds, _ = _run_rounds(env, policy_factory, args.cache_path, seeds)
        results.append((label, rounds))

        mean, std = _stats(rounds)
        print(f"  {label:<40}  mean={mean:7.2f}  std={std:6.2f}")

    # ── ranked summary ────────────────────────────────────────────────────────
    print()
    print("━" * 65)
    print(_col("Rank", 6) + _col("Config", 42) + _col("Mean reward", 12) + "Std")
    print("─" * 65)
    ranked = sorted(results, key=lambda r: _stats(r[1])[0], reverse=True)
    for rank, (label, rounds) in enumerate(ranked, 1):
        mean, std = _stats(rounds)
        marker = " ◀ best" if rank == 1 else ""
        print(f"{str(rank)+'.':<6}{label:<42}{mean:>8.2f}     {std:>6.2f}{marker}")
    print("━" * 65)

    best_label, best_rounds = ranked[0]
    best_mean, _ = _stats(best_rounds)
    print(f"\nRecommended: {best_label}  (mean reward {best_mean:.2f})")


if __name__ == "__main__":
    main()
