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
    STAGE3_BEST_CKPT,
    STAGE3_CKPT,
    get_device,
    seed_everything,
)
from controllers import heuristic_spec, league_checkpoints, net_spec, stochastic_heuristic_spec
from model import RecurrentMaskableActorCritic, load_checkpoint, save_checkpoint
from ppo import ppo_update
from rollout import SelfPlayCollector, default_workers
from validation import validate_model


def _load_start_model(device):
    for path, label in ((STAGE3_CKPT, "resume Stage 3"),
                        (STAGE2_CKPT, "warm-start from Stage 2"),
                        (STAGE1_CKPT, "warm-start from BC")):
        if path.exists():
            print(f"{label}: {path}")
            return load_checkpoint(path, device)
    print("No prior checkpoint — starting fresh (run earlier stages first for a jump-start).")
    return RecurrentMaskableActorCritic().to(device)


def _checkpoint_score(path) -> float:
    if not path.exists():
        return float("-inf")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return float(ckpt.get("meta", {}).get("validation_score", float("-inf")))
    except Exception:
        return float("-inf")


def _build_opponent_specs(
    heuristic_prob: float,
    stochastic_heuristic_prob: float = 0.0,
    stochastic_jitter: float = 0.35,
    stochastic_action_noise: float = 0.03,
):
    """Return a weighted list of picklable opponent specs.

    The heuristic is replicated so ``random.choice`` selects it ~heuristic_prob of
    the time; each frozen league snapshot contributes one net spec (workers load
    and cache the actual weights per-process on first use).
    """
    pool = league_checkpoints(LEAGUE_DIR)
    n_nets = max(1, len(pool))
    if heuristic_prob <= 0:
        heuristic_copies = 0
    elif heuristic_prob < 1:
        heuristic_copies = max(1, round(heuristic_prob / (1 - heuristic_prob) * n_nets))
    else:
        heuristic_copies = 999

    stoch_p = max(0.0, min(1.0, stochastic_heuristic_prob))
    stochastic_copies = round(heuristic_copies * stoch_p)
    fixed_copies = heuristic_copies - stochastic_copies
    specs = [heuristic_spec()] * fixed_copies
    specs += [
        stochastic_heuristic_spec(stochastic_jitter, stochastic_action_noise)
        for _ in range(stochastic_copies)
    ]
    specs += [net_spec(ckpt) for ckpt in pool]
    print(
        f"League opponents: {fixed_copies}xheuristic + "
        f"{stochastic_copies}xstochastic heuristic + {len(pool)} frozen snapshots"
    )
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
    ap.add_argument("--stochastic-heuristic-prob", type=float, default=0.0,
                    help="fraction of heuristic opponents built from randomized heuristic parameters")
    ap.add_argument("--stochastic-jitter", type=float, default=0.35,
                    help="relative jitter for stochastic heuristic numeric knobs")
    ap.add_argument("--stochastic-action-noise", type=float, default=0.03,
                    help="chance stochastic heuristic takes a random legal action")
    ap.add_argument("--snapshot-every", type=int, default=20, help="updates between adding self to the league pool")
    ap.add_argument("--gated-snapshots", action="store_true",
                    help="only add league snapshots that pass the validation gate")
    ap.add_argument("--snapshot-margin", type=float, default=50.0,
                    help="allowed validation-score drop from best for gated snapshots")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--advanced-prob", type=float, default=0.0,
                    help="when training on --novice, probability a rollout episode uses an advanced random map")
    ap.add_argument("--validate-every", type=int, default=0,
                    help="run a quiet benchmark validation every N updates; 0 disables")
    ap.add_argument("--validation-rounds", type=int, default=20,
                    help="novice benchmark rounds per validation")
    ap.add_argument("--validation-advanced-rounds", type=int, default=0,
                    help="advanced-map benchmark rounds per validation")
    ap.add_argument("--validation-learners", type=int, default=3,
                    help="RL agents used in validation benchmark")
    ap.add_argument("--validation-seed", type=int, default=22345)
    ap.add_argument("--rollback-on-regress", action="store_true",
                    help="reload the best validated checkpoint if validation falls below best by rollback-margin")
    ap.add_argument("--rollback-margin", type=float, default=75.0,
                    help="allowed validation-score drop before rollback")
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

    specs = _build_opponent_specs(
        args.heuristic_prob,
        args.stochastic_heuristic_prob,
        args.stochastic_jitter,
        args.stochastic_action_noise,
    )
    collector = SelfPlayCollector(
        model, device,
        opponent_specs=specs,
        n_learners=args.learners,
        novice=args.novice,
        advanced_prob=args.advanced_prob,
        gamma=args.gamma, lam=args.lam,
        num_workers=args.num_workers,
    )
    print(f"Rollout workers: {args.num_workers}")
    if args.novice and args.advanced_prob > 0:
        print(f"Arena mix: novice with advanced_prob={args.advanced_prob:.2f}")

    gen = len(league_checkpoints(LEAGUE_DIR))
    best_validation = _checkpoint_score(STAGE3_BEST_CKPT)
    if best_validation > float("-inf"):
        print(f"Best validation checkpoint: {STAGE3_BEST_CKPT} score={best_validation:+.1f}")
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
            f"min={stats['learner_return_min']:6.1f} max={stats['learner_return_max']:6.1f} "
            f"sd={stats['learner_return_std']:5.1f}  "
            f"pi={losses['policy_loss']:+.3f} v={losses['value_loss']:.2f} "
            f"H={losses['entropy']:.3f} kl={losses['approx_kl']:.4f}  "
            f"{stats['n_seqs']}seq {dt:.1f}s"
        )

        val = None
        rolled_back = False
        validation_due = args.validate_every > 0 and update % args.validate_every == 0
        snapshot_due = update % args.snapshot_every == 0
        if validation_due or (snapshot_due and args.gated_snapshots):
            val = validate_model(
                model,
                rounds=args.validation_rounds,
                learners=args.validation_learners,
                novice=args.novice,
                seed=args.validation_seed,
                advanced_rounds=args.validation_advanced_rounds,
            )
            tqdm.write(
                f"  [val] score={val['score']:+.1f} rl={val['rl_mean']:.1f} "
                f"heur={val['heur_baseline']:.1f} suites={val['num_suites']}"
            )
            if val["score"] > best_validation:
                best_validation = val["score"]
                save_checkpoint(STAGE3_BEST_CKPT, model, meta={
                    "stage": "league_best",
                    "update": update,
                    "validation_score": val["score"],
                    "validation_rl_mean": val["rl_mean"],
                    "validation_heur_baseline": val["heur_baseline"],
                })
                tqdm.write(f"  [val] promoted best checkpoint -> {STAGE3_BEST_CKPT}")
            elif (
                args.rollback_on_regress
                and STAGE3_BEST_CKPT.exists()
                and val["score"] < best_validation - args.rollback_margin
            ):
                best_model = load_checkpoint(STAGE3_BEST_CKPT, device)
                model.load_state_dict(best_model.state_dict())
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                rolled_back = True
                tqdm.write(
                    f"  [val] rollback to best checkpoint "
                    f"(score {val['score']:+.1f} < best {best_validation:+.1f})"
                )

        if snapshot_due:
            skip_snapshot = (
                args.gated_snapshots
                and val is not None
                and best_validation > float("-inf")
                and (rolled_back or val["score"] < best_validation - args.snapshot_margin)
            )
            if skip_snapshot:
                tqdm.write(
                    f"  - skipped league snapshot (validation {val['score']:+.1f}, "
                    f"best {best_validation:+.1f})"
                )
            else:
                snap = LEAGUE_DIR / f"gen_{gen:03d}.pt"
                save_checkpoint(snap, model, meta={"stage": "league", "update": update})
                gen += 1
                tqdm.write(f"  + league snapshot {snap.name}  (rebuilding opponent pool)")
                collector.set_opponent_specs(_build_opponent_specs(
                    args.heuristic_prob,
                    args.stochastic_heuristic_prob,
                    args.stochastic_jitter,
                    args.stochastic_action_noise,
                ))

        if update % args.save_every == 0 or update == args.updates:
            save_checkpoint(STAGE3_CKPT, model, meta={"stage": "league", "update": update,
                                                      "learner_return_mean": stats["learner_return_mean"]})

    collector.close()
    save_checkpoint(STAGE3_CKPT, model, meta={"stage": "league", "update": args.updates})
    print(f"\nSaved Stage-3 checkpoint → {STAGE3_CKPT}")
    print("Benchmark:  python ae_rl/benchmark.py --ckpt ae_rl/checkpoints/stage3_league.pt")


if __name__ == "__main__":
    main()
