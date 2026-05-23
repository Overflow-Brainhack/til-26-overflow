"""Benchmark a learned checkpoint against the heuristic baseline (and random).

Runs head-to-head games and reports mean per-agent episode reward, broken down
by controller type. The reference point is the heuristic's own mean reward in a
6×heuristic game — beat that and the learned policy is competitive.

Usage:
    python ae_rl/benchmark.py                       # newest stage checkpoint
    python ae_rl/benchmark.py --ckpt ae_rl/checkpoints/stage2_ppo.pt --rounds 50
    python ae_rl/benchmark.py --learners 1 --novice
"""

from __future__ import annotations

import argparse
from statistics import mean, stdev

import numpy as np
from tqdm.auto import tqdm, trange

import common  # noqa: F401  (path bootstrap)
from common import CKPT_DIR, STAGE1_CKPT, STAGE2_CKPT, STAGE3_CKPT, get_device
from controllers import HeuristicController, NetController
from model import load_checkpoint
from rollout import make_env


def _play_game(env, controllers, seed: int) -> dict[str, float]:
    env.reset(seed=seed)
    for c in controllers.values():
        c.reset()
    while True:
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            if all(env.terminations.values()) or all(env.truncations.values()):
                break
            continue
        obs = env.observe(agent)
        env.step(int(controllers[agent].act(obs)))
    episode = getattr(env.dynamics.rewards, "_episode", {})
    return {a: float(episode.get(a, 0.0)) for a in env.possible_agents}


def benchmark(
    ckpt_path,
    rounds: int,
    n_learners: int,
    novice: bool,
    seed: int = 0,
    *,
    model=None,
    quiet: bool = False,
    deterministic: bool = True,
    rotate_slots: bool = True,
):
    import random

    random.seed(seed)
    device = get_device()
    env = make_env(novice)
    agents = list(env.possible_agents)

    if model is None:
        model = load_checkpoint(ckpt_path, device, eval_mode=True) if ckpt_path else None
        restore_train = False
    else:
        restore_train = model.training
        model.eval()

    def make_controllers(learner_ids):
        ctrl = {}
        for a in agents:
            if a in learner_ids and model is not None:
                ctrl[a] = NetController(model, device, name="rl",
                                        deterministic=deterministic, novice=novice)
            else:
                ctrl[a] = HeuristicController(use_cache=novice)
        return ctrl

    rl_scores: list[float] = []
    heur_scores: list[float] = []
    base_scores: list[float] = []   # 6×heuristic reference

    if not quiet:
        print(
            f"\nBenchmark: {'RL='+str(n_learners)+' vs heuristic='+str(len(agents)-n_learners) if model else 'heuristic only'}"
            f"  | {rounds} rounds | novice={novice}\n"
        )
    iterator = range(rounds) if quiet else trange(rounds, desc="benchmark", unit="round")
    for r in iterator:
        s = random.randint(0, 2_000_000_000)
        if model is not None:
            if rotate_slots:
                start = r % len(agents)
                learner_ids = {agents[(start + i) % len(agents)] for i in range(n_learners)}
            else:
                learner_ids = set(agents[:n_learners])
        else:
            learner_ids = set()

        # Matchup game.
        ctrl = make_controllers(learner_ids)
        res = _play_game(env, ctrl, s)
        if model is not None:
            rl_scores.extend(res[a] for a in agents if a in learner_ids)
            heur_scores.extend(res[a] for a in agents if a not in learner_ids)

        # Reference game: same seed, all heuristic.
        ref = _play_game(env, {a: HeuristicController(use_cache=novice) for a in agents}, s)
        base_scores.extend(ref.values())

        if model is not None and not quiet:
            tqdm.write(
                f"  round {r+1:3d}  rl={','.join(sorted(learner_ids)):>15s}"
                f"  rl_mean={mean(res[a] for a in learner_ids):7.1f}"
                f"  heur_mean={mean(res[a] for a in agents if a not in learner_ids):7.1f}"
                f"  ref_heur={mean(ref.values()):7.1f}"
            )
        elif not quiet:
            tqdm.write(f"  round {r+1:3d}  heur_mean={mean(ref.values()):7.1f}")

    def _fmt(xs):
        if not xs:
            return "   n/a"
        s = stdev(xs) if len(xs) > 1 else 0.0
        return f"{mean(xs):7.2f} ± {s:5.2f}"

    if not quiet:
        print("\n" + "═" * 56)
        if model is not None:
            print(f"  RL agents          {_fmt(rl_scores)}")
            print(f"  Heuristic (in-game){_fmt(heur_scores)}")
        print(f"  Heuristic baseline {_fmt(base_scores)}  (6×heuristic reference)")
        print("═" * 56)
    delta = None
    if model is not None and rl_scores and base_scores:
        delta = mean(rl_scores) - mean(base_scores)
        verdict = "BEATS" if delta > 0 else "below"
        if not quiet:
            print(f"  RL {verdict} heuristic baseline by {delta:+.2f} mean reward.\n")
    if restore_train:
        model.train()
    return {
        "rl_mean": mean(rl_scores) if rl_scores else None,
        "heur_baseline": mean(base_scores) if base_scores else None,
        "delta": delta,
    }


def _newest_ckpt():
    for p in (STAGE3_CKPT, STAGE2_CKPT, STAGE1_CKPT):
        if p.exists():
            return p
    found = sorted(CKPT_DIR.glob("*.pt"))
    return found[-1] if found else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint path (default: newest stage)")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--learners", type=int, default=1, help="number of RL-controlled agents (1-6)")
    ap.add_argument("--novice", dest="novice", action="store_true", default=True,
                    help="fixed novice map (default)")
    ap.add_argument("--advanced", dest="novice", action="store_false",
                    help="randomised advanced maps")
    ap.add_argument("--deterministic", dest="deterministic", action="store_true", default=True,
                    help="use argmax actions for RL policy evaluation (default)")
    ap.add_argument("--sample-actions", dest="deterministic", action="store_false",
                    help="sample from the RL policy distribution")
    ap.add_argument("--rotate-slots", dest="rotate_slots", action="store_true", default=True,
                    help="rotate which agent slots are controlled by RL each round (default)")
    ap.add_argument("--fixed-slots", dest="rotate_slots", action="store_false",
                    help="always use the first N agent slots for RL")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = _newest_ckpt()
        if ckpt is None:
            print("No checkpoint found — benchmarking the heuristic baseline only.")
    else:
        from pathlib import Path

        ckpt = Path(ckpt)
    if ckpt:
        print(f"Loading checkpoint: {ckpt}")
    benchmark(
        ckpt,
        args.rounds,
        max(0, min(args.learners, common.NUM_AGENTS)),
        args.novice,
        args.seed,
        deterministic=args.deterministic,
        rotate_slots=args.rotate_slots,
    )


if __name__ == "__main__":
    main()
