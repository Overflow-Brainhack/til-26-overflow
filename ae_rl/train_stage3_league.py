"""Stage 3 — PPO self-play league.

Warm-starts from Stage 2 (or resumes Stage 3) and trains against a *pool* of
opponents: the heuristic plus frozen snapshots of earlier versions of the policy
itself. Periodically the current policy is snapshotted into the league pool, so
the learner must keep beating an ever-growing set of past selves rather than
overfitting to one opponent style — important because the real tournament
opponents are unknown.

Auto-discovery: resumes ae_rl/checkpoints/stage3_league.pt if present, else
warm-starts from stage2_ppo.pt, else stage1_bc.pt, else a fresh net. League
snapshots live in ae_rl/checkpoints/league/.

Output: ae_rl/checkpoints/stage3_league.pt + league snapshots.

Usage:
    python ae_rl/train_stage3_league.py
    python ae_rl/train_stage3_league.py --updates 300 --snapshot-every 25 --heuristic-prob 0.4
"""

from __future__ import annotations

import argparse
import time

import torch
from tqdm.auto import tqdm, trange

import common  # noqa: F401  (path bootstrap)
from common import (
    LEAGUE_DIR,
    STAGE1_CKPT,
    STAGE2_CKPT,
    STAGE3_CKPT,
    get_device,
    seed_everything,
)
from controllers import heuristic_spec, league_checkpoints, net_spec
from model import RecurrentMaskableActorCritic, load_checkpoint, save_checkpoint
from ppo import ppo_update
from rollout import SelfPlayCollector, default_workers


def _load_start_model(device):
    for path, label in ((STAGE3_CKPT, "resume Stage 3"),
                        (STAGE2_CKPT, "warm-start from Stage 2"),
                        (STAGE1_CKPT, "warm-start from BC")):
        if path.exists():
            print(f"{label}: {path}")
            return load_checkpoint(path, device)
    print("No prior checkpoint — starting fresh (run earlier stages first for a jump-start).")
    return RecurrentMaskableActorCritic().to(device)


def _build_opponent_specs(heuristic_prob: float):
    """Return a weighted list of picklable opponent specs.

    The heuristic is replicated so ``random.choice`` selects it ~heuristic_prob of
    the time; each frozen league snapshot contributes one net spec (workers load
    and cache the actual weights per-process on first use).
    """
    pool = league_checkpoints(LEAGUE_DIR)
    n_nets = max(1, len(pool))
    heuristic_copies = (
        max(1, round(heuristic_prob / (1 - heuristic_prob) * n_nets))
        if heuristic_prob < 1 else 999
    )
    specs = [heuristic_spec()] * heuristic_copies
    specs += [net_spec(ckpt) for ckpt in pool]
    print(f"League opponents: {heuristic_copies}×heuristic + {len(pool)} frozen snapshots")
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--episodes-per-update", type=int, default=8)
    ap.add_argument("--learners", type=int, default=2, help="RL-controlled agents per game (rest = league opponents)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seq-minibatch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--heuristic-prob", type=float, default=0.5, help="approx fraction of opponents that are heuristic")
    ap.add_argument("--snapshot-every", type=int, default=20, help="updates between adding self to the league pool")
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

    # Seed the league with the starting policy so there's at least one net opponent.
    if not league_checkpoints(LEAGUE_DIR):
        seed_path = LEAGUE_DIR / "gen_000.pt"
        save_checkpoint(seed_path, model, meta={"stage": "league_seed"})
        print(f"Seeded league with {seed_path}")

    specs = _build_opponent_specs(args.heuristic_prob)
    collector = SelfPlayCollector(
        model, device,
        opponent_specs=specs,
        n_learners=args.learners,
        novice=args.novice,
        gamma=args.gamma, lam=args.lam,
        num_workers=args.num_workers,
    )
    print(f"Rollout workers: {args.num_workers}")

    gen = len(league_checkpoints(LEAGUE_DIR))
    bar = trange(1, args.updates + 1, desc="League", unit="upd")
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
                        gen=gen, kl=f"{losses['approx_kl']:.3f}")
        tqdm.write(
            f"upd {update:3d}/{args.updates}  "
            f"ret={stats['learner_return_mean']:7.1f} (opp {stats['opp_return_mean']:6.1f})  "
            f"max={stats['learner_return_max']:6.1f}  "
            f"pi={losses['policy_loss']:+.3f} v={losses['value_loss']:.2f} "
            f"H={losses['entropy']:.3f} kl={losses['approx_kl']:.4f}  "
            f"{stats['n_seqs']}seq {dt:.1f}s"
        )

        if update % args.snapshot_every == 0:
            snap = LEAGUE_DIR / f"gen_{gen:03d}.pt"
            save_checkpoint(snap, model, meta={"stage": "league", "update": update})
            gen += 1
            tqdm.write(f"  + league snapshot {snap.name}  (rebuilding opponent pool)")
            collector.set_opponent_specs(_build_opponent_specs(args.heuristic_prob))

        if update % args.save_every == 0 or update == args.updates:
            save_checkpoint(STAGE3_CKPT, model, meta={"stage": "league", "update": update,
                                                      "learner_return_mean": stats["learner_return_mean"]})

    collector.close()
    save_checkpoint(STAGE3_CKPT, model, meta={"stage": "league", "update": args.updates})
    print(f"\nSaved Stage-3 checkpoint → {STAGE3_CKPT}")
    print("Benchmark:  python ae_rl/benchmark.py --ckpt ae_rl/checkpoints/stage3_league.pt")


if __name__ == "__main__":
    main()
