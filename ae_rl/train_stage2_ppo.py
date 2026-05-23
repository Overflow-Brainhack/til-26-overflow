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
from common import (
    STAGE1_CKPT,
    STAGE2_BEST_CKPT,
    STAGE2_CKPT,
    STAGE2_SNAPSHOT_DIR,
    get_device,
    seed_everything,
)
from controllers import heuristic_spec, stochastic_heuristic_spec
from model import RecurrentMaskableActorCritic, load_checkpoint, save_checkpoint
from ppo import ppo_update
from rollout import SelfPlayCollector, default_workers
from validation import validate_model


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


def _checkpoint_score(path) -> float:
    if not path.exists():
        return float("-inf")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return float(ckpt.get("meta", {}).get("validation_score", float("-inf")))
    except Exception:
        return float("-inf")


def _build_opponent_specs(args) -> list[dict]:
    p = max(0.0, min(1.0, args.stochastic_heuristic_prob))
    if p <= 0:
        return [heuristic_spec()]
    if p >= 1:
        return [stochastic_heuristic_spec(args.stochastic_jitter, args.stochastic_action_noise)]

    slots = 20
    n_stochastic = max(1, round(slots * p))
    n_fixed = max(1, slots - n_stochastic)
    return (
        [heuristic_spec()] * n_fixed
        + [stochastic_heuristic_spec(args.stochastic_jitter, args.stochastic_action_noise)] * n_stochastic
    )


def _parse_learner_slots(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--episodes-per-update", type=int, default=8)
    ap.add_argument("--learners", type=int, default=3, help="RL-controlled agents per game (1-6)")
    ap.add_argument("--learner-slots", type=str, default="",
                    help="comma-separated agent ids to sample learners from, e.g. agent_0,agent_2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seq-minibatch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--critic-warmup", type=int, default=10,
                    help="value-only updates before PPO (trains the critic the BC stage "
                         "left random, so early advantages aren't garbage; 0 to skip)")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--snapshot-every", type=int, default=50,
                    help="save a unique per-run checkpoint every N updates; 0 disables")
    ap.add_argument("--stochastic-heuristic-prob", type=float, default=0.0,
                    help="fraction of heuristic opponents built from randomized heuristic parameters")
    ap.add_argument("--stochastic-jitter", type=float, default=0.35,
                    help="relative jitter for stochastic heuristic numeric knobs")
    ap.add_argument("--stochastic-action-noise", type=float, default=0.03,
                    help="chance stochastic heuristic takes a random legal action")
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
    ap.add_argument("--validation-seed", type=int, default=12345)
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
    snapshot_dir = None
    if args.snapshot_every > 0:
        snapshot_dir = STAGE2_SNAPSHOT_DIR / time.strftime("%Y%m%d_%H%M%S")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Stage-2 snapshots: {snapshot_dir}")

    opponent_specs = _build_opponent_specs(args)
    collector = SelfPlayCollector(
        model, device,
        opponent_specs=opponent_specs,
        n_learners=args.learners,
        learner_slots=_parse_learner_slots(args.learner_slots),
        novice=args.novice,
        advanced_prob=args.advanced_prob,
        gamma=args.gamma, lam=args.lam,
        num_workers=args.num_workers,
    )
    print(f"Rollout workers: {args.num_workers}")
    if args.learner_slots:
        print(f"Learner slots: {_parse_learner_slots(args.learner_slots)}")
    print(
        f"Opponent mix: {sum(1 for s in opponent_specs if s['kind'] == 'heuristic')} fixed heuristic, "
        f"{sum(1 for s in opponent_specs if s['kind'] == 'stochastic_heuristic')} stochastic heuristic"
    )
    if args.novice and args.advanced_prob > 0:
        print(f"Arena mix: novice with advanced_prob={args.advanced_prob:.2f}")
    print(
        f"Checkpoint cadence: latest every {args.save_every} updates, "
        f"snapshots every {args.snapshot_every if args.snapshot_every > 0 else 'never'} updates, "
        f"validation every {args.validate_every if args.validate_every > 0 else 'never'} updates "
        f"({args.validation_rounds} novice rounds, {args.validation_advanced_rounds} advanced rounds)"
    )

    # ── critic warm-up: fit the value head (BC left it random) without touching
    # the cloned actor/trunk, so the first real PPO advantages are meaningful. ──
    if args.critic_warmup > 0:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.critic.parameters():
            p.requires_grad_(True)
        for _ in trange(args.critic_warmup, desc="critic warmup", unit="upd"):
            batch, _ = collector.collect(args.episodes_per_update)
            wl = ppo_update(model, opt, batch, device, epochs=args.epochs,
                            seq_minibatch=args.seq_minibatch, value_only=True)
            tqdm.write(f"  [warmup] v_loss={wl['value_loss']:.1f}")
        for p in model.parameters():
            p.requires_grad_(True)

    best_return = float("-inf")
    best_validation = _checkpoint_score(STAGE2_BEST_CKPT)
    if best_validation > float("-inf"):
        print(f"Best validation checkpoint: {STAGE2_BEST_CKPT} score={best_validation:+.1f}")
    bar = trange(1, args.updates + 1, desc="PPO", unit="upd")
    for update in bar:
        val = None
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
            f"min={stats['learner_return_min']:6.1f} max={stats['learner_return_max']:6.1f} "
            f"sd={stats['learner_return_std']:5.1f}  "
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

        if args.validate_every > 0 and update % args.validate_every == 0:
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
                save_checkpoint(STAGE2_BEST_CKPT, model, meta={
                    "stage": "ppo_vs_diverse_heuristic_best",
                    "update": update,
                    "validation_score": val["score"],
                    "validation_rl_mean": val["rl_mean"],
                    "validation_heur_baseline": val["heur_baseline"],
                })
                tqdm.write(f"  [val] promoted best checkpoint -> {STAGE2_BEST_CKPT}")
            elif (
                args.rollback_on_regress
                and STAGE2_BEST_CKPT.exists()
                and val["score"] < best_validation - args.rollback_margin
            ):
                best_model = load_checkpoint(STAGE2_BEST_CKPT, device)
                model.load_state_dict(best_model.state_dict())
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                tqdm.write(
                    f"  [val] rollback to best checkpoint "
                    f"(score {val['score']:+.1f} < best {best_validation:+.1f})"
                )

        if snapshot_dir is not None and update % args.snapshot_every == 0:
            snap = snapshot_dir / f"stage2_update_{update:04d}.pt"
            meta = {
                "stage": "ppo_vs_diverse_heuristic_snapshot",
                "update": update,
                "learner_return_mean": stats["learner_return_mean"],
                "learner_return_min": stats["learner_return_min"],
                "learner_return_max": stats["learner_return_max"],
                "learner_return_std": stats["learner_return_std"],
            }
            if val is not None:
                meta.update({
                    "validation_score": val["score"],
                    "validation_rl_mean": val["rl_mean"],
                    "validation_heur_baseline": val["heur_baseline"],
                })
            save_checkpoint(snap, model, meta=meta)
            tqdm.write(f"  [snap] saved {snap}")

    collector.close()
    save_checkpoint(STAGE2_CKPT, model, meta={"stage": "ppo_vs_heuristic", "update": args.updates})
    print(f"\nSaved Stage-2 checkpoint → {STAGE2_CKPT}  (best mean return {best_return:.1f})")
    print("Benchmark:  python ae_rl/benchmark.py")
    print("Next:       python ae_rl/train_stage3_league.py")


if __name__ == "__main__":
    main()
