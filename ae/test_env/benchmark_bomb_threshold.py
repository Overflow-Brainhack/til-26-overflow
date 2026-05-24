"""Headless self-play benchmark: compare predictive-bomb thresholds.

Runs N rounds of novice-mode self-play per threshold (all agents use the same
policy), reports mean ± std reward, and prints a ranked table. Also supports
benchmarking the auto-tune policy against fixed thresholds.

Usage:
    uv run python ae/test_env/benchmark_bomb_threshold.py
    uv run python ae/test_env/benchmark_bomb_threshold.py --thresholds 0.1 0.25 0.5 0.75
    uv run python ae/test_env/benchmark_bomb_threshold.py --rounds 24 --seeds 0 1 2 3 4 5
    uv run python ae/test_env/benchmark_bomb_threshold.py --include-auto-tune
    uv run python ae/test_env/benchmark_bomb_threshold.py --no-drift-aware   # uniform model
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
    """Run one round per seed; return per-round mean rewards + last policy (for auto-tune state)."""
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
        "--thresholds", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80],
        help="Fixed thresholds to benchmark (space-separated floats)",
    )
    parser.add_argument(
        "--rounds", type=int, default=12,
        help="Rounds per threshold (more = lower variance; each ~1–2 s)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit seeds to use; overrides --rounds",
    )
    parser.add_argument(
        "--include-auto-tune", action="store_true", default=False,
        help="Also benchmark the EMA auto-tuning policy",
    )
    parser.add_argument(
        "--auto-tune-target", type=float, default=0.40,
        help="Target hit rate for auto-tune (default 0.40)",
    )
    parser.add_argument(
        "--drift-aware", dest="drift_aware", action="store_true", default=True,
        help="Use velocity-biased enemy model (default ON)",
    )
    parser.add_argument(
        "--no-drift-aware", dest="drift_aware", action="store_false",
        help="Use uniform random-walk enemy model",
    )
    parser.add_argument(
        "--no-predictive", action="store_true", default=False,
        help="Disable predictive bombing entirely (baseline)",
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

    results: list[tuple[str, list[float], HeuristicPolicy | None]] = []

    # ── fixed thresholds ─────────────────────────────────────────────────────
    all_configs: list[tuple[str, dict]] = []

    if args.no_predictive:
        all_configs.append(("no-pred", {"predictive_bomb": False}))

    for t in args.thresholds:
        all_configs.append((f"thresh={t:.2f}", {
            "predictive_bomb": True,
            "predictive_bomb_threshold": t,
            "drift_aware_bomb": args.drift_aware,
            "auto_tune_bomb": False,
        }))

    if args.include_auto_tune:
        all_configs.append((f"auto-tune(tgt={args.auto_tune_target:.2f})", {
            "predictive_bomb": True,
            "predictive_bomb_threshold": 0.25,   # starting value
            "drift_aware_bomb": args.drift_aware,
            "auto_tune_bomb": True,
            "bomb_tune_target": args.auto_tune_target,
        }))

    n_configs = len(all_configs)
    drift_tag = "drift" if args.drift_aware else "uniform"
    print(f"\nBenchmark: {len(seeds)} seeds × {n_configs} configs  [{drift_tag} model]")
    print(f"Seeds: {seeds}")
    print()

    for label, kwargs in all_configs:
        def policy_factory(kw=kwargs) -> HeuristicPolicy:
            return HeuristicPolicy(
                wall_breaking=True,
                wall_break_cost=5.0,
                smart_defend=True,
                **kw,
            )

        rounds, last_policy = _run_rounds(env, policy_factory, args.cache_path, seeds)
        results.append((label, rounds, last_policy))

        mean, std = _stats(rounds)
        suffix = ""
        if last_policy is not None and kwargs.get("auto_tune_bomb"):
            suffix = f"  → final threshold={last_policy._tuned_threshold:.3f}"
        print(f"  {label:<35}  mean={mean:7.2f}  std={std:6.2f}{suffix}")

    # ── ranked summary ────────────────────────────────────────────────────────
    print()
    print("━" * 62)
    print(_col("Rank", 6) + _col("Config", 37) + _col("Mean reward", 12) + "Std")
    print("─" * 62)
    ranked = sorted(results, key=lambda r: _stats(r[1])[0], reverse=True)
    for rank, (label, rounds, _) in enumerate(ranked, 1):
        mean, std = _stats(rounds)
        marker = " ◀ best" if rank == 1 else ""
        print(f"{str(rank)+'.':<6}{label:<37}{mean:>8.2f}     {std:>6.2f}{marker}")
    print("━" * 62)

    best_label, best_rounds, _ = ranked[0]
    best_mean, _ = _stats(best_rounds)
    print(f"\nRecommended: {best_label}  (mean reward {best_mean:.2f})")
    if not args.include_auto_tune:
        print("Re-run with --include-auto-tune to compare against the adaptive policy.")


if __name__ == "__main__":
    main()
