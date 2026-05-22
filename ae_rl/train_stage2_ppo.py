"""Stage 2 — PPO self-play against the heuristic.

Warm-starts from the Stage-1 BC checkpoint (auto-discovered) and fine-tunes with
PPO. The learner controls a subset of the 6 agents; the rest are the production
heuristic. This lets the policy improve on the heuristic it was cloned from while
playing in the same free-for-all it will be evaluated in.

Auto-discovery: loads ae_rl/checkpoints/stage2_ppo.pt if it exists (to resume),
else ae_rl/checkpoints/stage1_bc.pt (to warm-start). Falls back to a fresh net.

Output: ae_rl/checkpoints/stage2_ppo.pt  (overwritten each save)

Usage:
    python ae_rl/train_stage2_ppo.py
    python ae_rl/train_stage2_ppo.py --updates 200 --episodes-per-update 8 --learners 3
"""

from __future__ import annotations

import argparse
import time

import torch
from tqdm.auto import tqdm, trange

import common  # noqa: F401  (path bootstrap)
from common import STAGE1_CKPT, STAGE2_CKPT, get_device, seed_everything
from controllers import heuristic_spec
from model import RecurrentMaskableActorCritic, load_checkpoint, save_checkpoint
from ppo import ppo_update
from rollout import SelfPlayCollector, default_workers


def _load_start_model(device):
    if STAGE2_CKPT.exists():
        print(f"Resuming from {STAGE2_CKPT}")
        return load_checkpoint(STAGE2_CKPT, device)
    if STAGE1_CKPT.exists():
        print(f"Warm-starting from BC checkpoint {STAGE1_CKPT}")
        return load_checkpoint(STAGE1_CKPT, device)
    print("No prior checkpoint found — starting from a fresh network "
          "(run train_stage1_bc.py first for a proper jump-start).")
    return RecurrentMaskableActorCritic().to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--episodes-per-update", type=int, default=8)
    ap.add_argument("--learners", type=int, default=3, help="RL-controlled agents per game (1-6)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seq-minibatch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--novice", dest="novice", action="store_true", default=True,
                    help="train on the fixed novice map (default)")
    ap.add_argument("--advanced", dest="novice", action="store_false",
                    help="train on randomised advanced maps")
    ap.add_argument("-j", "--num-workers", type=int, default=default_workers(),
                    help="parallel rollout processes (default: cpus-1; 1 = serial)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    model = _load_start_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    collector = SelfPlayCollector(
        model, device,
        opponent_specs=[heuristic_spec()],
        n_learners=args.learners,
        novice=args.novice,
        gamma=args.gamma, lam=args.lam,
        num_workers=args.num_workers,
    )
    print(f"Rollout workers: {args.num_workers}")

    best_return = float("-inf")
    bar = trange(1, args.updates + 1, desc="PPO", unit="upd")
    for update in bar:
        t0 = time.time()
        batch, stats = collector.collect(args.episodes_per_update, progress=True)
        losses = ppo_update(
            model, opt, batch, device,
            epochs=args.epochs, seq_minibatch=args.seq_minibatch,
            clip=args.clip, entropy_coef=args.entropy_coef,
        )
        dt = time.time() - t0
        bar.set_postfix(ret=f"{stats['learner_return_mean']:.0f}",
                        opp=f"{stats['opp_return_mean']:.0f}",
                        kl=f"{losses['approx_kl']:.3f}")
        tqdm.write(
            f"upd {update:3d}/{args.updates}  "
            f"ret={stats['learner_return_mean']:7.1f} (opp {stats['opp_return_mean']:6.1f})  "
            f"max={stats['learner_return_max']:6.1f}  "
            f"pi={losses['policy_loss']:+.3f} v={losses['value_loss']:.2f} "
            f"H={losses['entropy']:.3f} kl={losses['approx_kl']:.4f}  "
            f"{stats['n_seqs']}seq {dt:.1f}s"
        )

        if update % args.save_every == 0 or update == args.updates:
            save_checkpoint(STAGE2_CKPT, model, meta={
                "stage": "ppo_vs_heuristic", "update": update,
                "learner_return_mean": stats["learner_return_mean"],
            })
        if stats["learner_return_mean"] > best_return:
            best_return = stats["learner_return_mean"]

    collector.close()
    save_checkpoint(STAGE2_CKPT, model, meta={"stage": "ppo_vs_heuristic", "update": args.updates})
    print(f"\nSaved Stage-2 checkpoint → {STAGE2_CKPT}  (best mean return {best_return:.1f})")
    print("Benchmark:  python ae_rl/benchmark.py")
    print("Next:       python ae_rl/train_stage3_league.py")


if __name__ == "__main__":
    main()
