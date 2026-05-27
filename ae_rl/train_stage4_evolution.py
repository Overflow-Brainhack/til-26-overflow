"""Stage 4 — Population-Based Self-Play (Evolution).

K active learners train concurrently, each playing against a mix of the OTHER
live learners' current weights + a growing archive of frozen tournament
snapshots + a small scripted slice. Every ``--tournament-every`` updates a
round-robin ranks them; the bottom learner is cloned from the top and its
hyperparameters are perturbed (exploit-and-explore).

Compared with Stage 3 (single-learner league with rollback-on-regress):
- More-diverse training opponents → less overfitting to a single style
- Natural selection across K trajectories replaces "rollback to best.pt", so a
  bad gradient step in one slot doesn't reset the rest
- Long continuous runs (default --updates 5000) without validation gating

Auto-discovery for warm-start (in order):
- explicit ``--ckpt PATH``
- ``ae_rl/checkpoints/stage4_evolution.pt`` (resume full population state)
- ``ae_rl/checkpoints/stage3_league_best.pt`` (warm-start every slot from best Stage 3)
- ``ae_rl/checkpoints/stage3_league.pt``
- ``ae_rl/checkpoints/stage2_ppo_best.pt`` / ``stage2_ppo.pt``

Usage:
    python ae_rl/train_stage4_evolution.py
    python ae_rl/train_stage4_evolution.py --updates 10000 --pop-size 4 --tournament-every 100
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm, trange

import common  # noqa: F401  (path bootstrap)
from common import (
    EVOLUTION_ARCHIVE_DIR,
    STAGE2_BEST_CKPT,
    STAGE2_CKPT,
    STAGE3_BEST_CKPT,
    STAGE3_CKPT,
    STAGE4_BEST_CKPT,
    STAGE4_CKPT,
    get_device,
    seed_everything,
)
from evolution import EvolutionTrainer
from rollout import default_workers
from run_summary import RunSummary, default_summary_path
from validation import validate_model


def _auto_discover_seed(explicit: str) -> Path | None:
    if explicit:
        p = Path(explicit)
        print(f"seed checkpoint (--ckpt): {p}")
        return p
    for path, label in (
        (STAGE3_BEST_CKPT, "warm-start every slot from Stage 3 best"),
        (STAGE3_CKPT, "warm-start every slot from Stage 3 latest"),
        (STAGE2_BEST_CKPT, "warm-start every slot from Stage 2 best"),
        (STAGE2_CKPT, "warm-start every slot from Stage 2 latest"),
    ):
        if path.exists():
            print(f"{label}: {path}")
            return path
    print("no seed checkpoint found — initialising every slot from scratch")
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── population & schedule ─────────────────────────────────────────────
    ap.add_argument("--pop-size", type=int, default=4,
                    help="K — number of concurrent learners. Each update cycle "
                         "runs K (collect+PPO) passes, so wall time scales linearly.")
    ap.add_argument("--updates", type=int, default=5000,
                    help="number of evolutionary updates (K mini-updates each). "
                         "Default 5000 → ~5000*K*episodes_per_update episodes. "
                         "Designed for multi-day continuous runs.")
    ap.add_argument("--episodes-per-update", type=int, default=8)
    ap.add_argument("--learners", type=int, default=2,
                    help="RL-controlled agents per episode (rest = opponents from pool)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seq-minibatch", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    # ── opponent mix (global shares; per-learner scripted vector mutates) ─
    ap.add_argument("--live-share", type=float, default=0.55,
                    help="share of every learner's opponent pool drawn from "
                         "the OTHER live learners' current weights. Default 0.55 "
                         "= the dominant signal is co-evolving peers.")
    ap.add_argument("--archive-share", type=float, default=0.25,
                    help="share drawn from frozen tournament-archive snapshots. "
                         "Grows the historical opponent pool over time without "
                         "freezing the live signal.")
    ap.add_argument("--scripted-share", type=float, default=0.20,
                    help="share drawn from scripted policies (tactical, berserker, "
                         "etc.). Per-learner mix vector controls the breakdown "
                         "and gets mutated at exploit-and-explore time.")
    # ── tournament & reselection ──────────────────────────────────────────
    ap.add_argument("--tournament-every", type=int, default=100,
                    help="updates between K-way round-robin tournaments. Lower "
                         "= more frequent natural selection but more wall time "
                         "spent on evaluation instead of training.")
    ap.add_argument("--tournament-rounds", type=int, default=16,
                    help="ordered (a vs b) episodes per pair in the tournament. "
                         "K=4 → ~12*16 = 192 episodes per tournament (≈one collect).")
    ap.add_argument("--mutation-jitter", type=float, default=0.5,
                    help="log-normal stddev for hyperparameter perturbation at "
                         "exploit-and-explore time. 0.5 ≈ ±50% per knob per mutation.")
    # ── archive & validation ──────────────────────────────────────────────
    ap.add_argument("--archive-every", type=int, default=200,
                    help="updates between adding the current best learner to the "
                         "frozen archive. Drives the archive_share opponent pool.")
    ap.add_argument("--archive-max-size", type=int, default=20,
                    help="cap on archive snapshots; oldest deleted past this. "
                         "0 = unlimited. Keeps the opponent distribution recent.")
    ap.add_argument("--validate-every", type=int, default=500,
                    help="updates between validation runs (logging only — no "
                         "rollback gating in evolutionary mode). 0 disables.")
    ap.add_argument("--validation-rounds", type=int, default=20)
    ap.add_argument("--validation-advanced-rounds", type=int, default=0)
    ap.add_argument("--validation-learners", type=int, default=1)
    ap.add_argument("--validation-baseline", type=str, default="vanilla",
                    choices=("strong", "vanilla", "berserker"),
                    help="opponent for the validation benchmark. Default 'vanilla' "
                         "(held-out) measures generalisation; switch to 'strong' "
                         "if you specifically want within-distribution scores.")
    ap.add_argument("--validation-seed", type=int, default=22345)
    # ── anti-stagnation ───────────────────────────────────────────────────
    ap.add_argument("--stagnation-window", type=int, default=3,
                    help="consecutive bottom-half tournaments before a learner "
                         "gets an entropy bump next cycle")
    ap.add_argument("--stagnation-entropy-mult", type=float, default=2.0,
                    help="multiplier applied to a stagnant learner's entropy coef")
    # ── env / arena ───────────────────────────────────────────────────────
    ap.add_argument("--novice", dest="novice", action="store_true", default=True)
    ap.add_argument("--advanced", dest="novice", action="store_false")
    ap.add_argument("--advanced-prob", type=float, default=0.0)
    ap.add_argument("-j", "--num-workers", type=int, default=default_workers())
    ap.add_argument("--no-shaping", dest="shape_rewards", action="store_false", default=True)
    # ── io & misc ─────────────────────────────────────────────────────────
    ap.add_argument("--save-every", type=int, default=50,
                    help="updates between full-population checkpoint saves")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default="",
                    help="explicit seed checkpoint path; otherwise auto-discovered")
    ap.add_argument("--summary-json", type=str, default="")
    args = ap.parse_args()

    summary_path = Path(args.summary_json) if args.summary_json else default_summary_path("stage4_evolution")
    with RunSummary(stage="stage4_evolution", args=vars(args), path=summary_path) as summary:
        _run(args, summary)


def _run(args, summary: RunSummary):
    seed_everything(args.seed)
    device = get_device()
    summary.set("device", str(device))
    summary.set("summary_path", str(summary.path))
    print(f"Device: {device}")
    print(f"Run summary: {summary.path}")

    seed_path = _auto_discover_seed(args.ckpt)

    rng = random.Random(args.seed)
    trainer = EvolutionTrainer(
        n_learners_pop=args.pop_size,
        seed_model_path=seed_path,
        device=device,
        episodes_per_update=args.episodes_per_update,
        n_learners_per_episode=args.learners,
        novice=args.novice,
        advanced_prob=args.advanced_prob,
        gamma=args.gamma,
        lam=args.lam,
        epochs=args.epochs,
        seq_minibatch=args.seq_minibatch,
        clip=args.clip,
        num_workers=args.num_workers,
        shape_rewards=args.shape_rewards,
        live_share=args.live_share,
        archive_share=args.archive_share,
        scripted_share=args.scripted_share,
        stagnation_window=args.stagnation_window,
        stagnation_entropy_mult=args.stagnation_entropy_mult,
        mutation_jitter=args.mutation_jitter,
        rng=rng,
    )

    # Resume full-population checkpoint if one exists. Otherwise we've already
    # warm-started every slot from the same seed_path (so the K models start
    # identical, diverge during training).
    start_update = 0
    if STAGE4_CKPT.exists():
        try:
            start_update = trainer.load_state(STAGE4_CKPT)
            print(f"Resumed Stage 4 from {STAGE4_CKPT} at update {start_update}")
        except Exception as e:
            print(f"Could not resume {STAGE4_CKPT} ({e}); starting fresh")

    summary.set("pop_size", args.pop_size)
    summary.set("archive_dir", str(EVOLUTION_ARCHIVE_DIR))
    summary.set("starting_update", start_update)

    best_validation = float("-inf")
    best_slot = 0

    total_updates = args.updates
    bar = trange(start_update + 1, start_update + total_updates + 1,
                 desc="Evo", unit="upd")
    last_ranked: list[tuple[int, float]] | None = None
    try:
        for update in bar:
            t0 = time.time()
            step_stats = trainer.step(update)
            dt = time.time() - t0

            ret_mean = step_stats["ret_mean"]
            ret_std = step_stats["ret_std"]
            bar.set_postfix(
                ret=f"{ret_mean:.0f}",
                sd=f"{ret_std:.0f}",
                arch=len(list(EVOLUTION_ARCHIVE_DIR.glob('*.pt'))),
            )
            tqdm.write(
                f"upd {update:5d}  ret={ret_mean:+7.1f} sd={ret_std:5.1f}  "
                + "  ".join(
                    f"L{p['slot']}={p['ret_mean']:+.0f}(ec={p['entropy_coef']:.3f},lr={p['lr']:.4f})"
                    for p in step_stats["per_learner"]
                )
                + f"  {dt:.1f}s"
            )

            summary.increment("updates_completed")
            summary.record("updates", {
                "update": update,
                "ret_mean": ret_mean,
                "ret_std": ret_std,
                "seconds": round(dt, 2),
                "per_learner": step_stats["per_learner"],
            })

            # ── tournament + reselect ────────────────────────────────────
            ran_tournament_this_update = False
            if args.tournament_every > 0 and update % args.tournament_every == 0:
                t_t0 = time.time()
                ranked = trainer.tournament(update, rounds_per_pair=args.tournament_rounds)
                last_ranked = ranked
                ran_tournament_this_update = True
                t_dt = time.time() - t_t0
                tqdm.write(
                    f"  [tournament] " + "  ".join(
                        f"#{r}: slot{slot}={score:+.1f}"
                        for r, (slot, score) in enumerate(ranked)
                    ) + f"  ({t_dt:.1f}s)"
                )
                summary.record("tournaments", {
                    "update": update,
                    "ranked": [{"slot": s, "score": sc} for s, sc in ranked],
                    "seconds": round(t_dt, 2),
                })

                reselect_info = trainer.reselect(ranked, update)
                if reselect_info is not None:
                    tqdm.write(
                        f"  [reselect] slot{reselect_info['into']} cloned from "
                        f"slot{reselect_info['cloned_from']}  "
                        f"(top={reselect_info['top_score']:+.1f}, bot={reselect_info['bot_score']:+.1f})  "
                        f"new ec={reselect_info['new_entropy']:.4f} lr={reselect_info['new_lr']:.5f}"
                    )
                    summary.record("reselections", reselect_info)
                else:
                    tqdm.write("  [reselect] no action (gap below noise floor or K<2)")

            # ── archive snapshot ─────────────────────────────────────────
            if args.archive_every > 0 and update % args.archive_every == 0:
                # Prefer the ranking from the tournament that just fired; if
                # no tournament happened this update (archive_every and
                # tournament_every can be set independently), fall back to
                # the most recent ranking we have, then to recent_returns.
                ranked_for_snapshot = last_ranked if last_ranked else None
                snap = trainer.snapshot_to_archive(update, ranked_for_snapshot)
                if snap is not None:
                    tqdm.write(f"  [archive] saved {snap.name}")
                    summary.record("archive_snapshots", {
                        "update": update, "path": str(snap),
                    })
                pruned = trainer.prune_archive(args.archive_max_size)
                if pruned:
                    tqdm.write(f"  [archive] pruned {len(pruned)} old snapshot(s): {', '.join(pruned)}")

            # ── validation (logging only) ────────────────────────────────
            if args.validate_every > 0 and update % args.validate_every == 0:
                # Validate every live learner — pick the best as the deploy candidate.
                val_scores: list[tuple[int, dict]] = []
                for learner in trainer.learners:
                    val = validate_model(
                        learner.model,
                        rounds=args.validation_rounds,
                        learners=args.validation_learners,
                        novice=args.novice,
                        seed=args.validation_seed,
                        advanced_rounds=args.validation_advanced_rounds,
                        baseline=args.validation_baseline,
                    )
                    val_scores.append((learner.slot, val))
                val_scores.sort(key=lambda x: x[1]["score"], reverse=True)
                best_slot_this, best_val = val_scores[0]
                tqdm.write(
                    f"  [val] best slot{best_slot_this}: score={best_val['score']:+.1f} "
                    f"(rl={best_val['rl_mean']:.1f} base={best_val['heur_baseline']:.1f}, "
                    f"baseline={args.validation_baseline})  "
                    + ", ".join(f"L{s}={v['score']:+.1f}" for s, v in val_scores)
                )
                summary.record("validations", {
                    "update": update,
                    "best_slot": best_slot_this,
                    "best_score": best_val["score"],
                    "per_slot": [{"slot": s, "score": v["score"]} for s, v in val_scores],
                    "baseline": args.validation_baseline,
                })
                if best_val["score"] > best_validation:
                    best_validation = best_val["score"]
                    best_slot = best_slot_this
                    trainer.save_best_single(
                        STAGE4_BEST_CKPT, slot=best_slot, update_idx=update,
                        score=best_validation,
                    )
                    tqdm.write(f"  [val] promoted best -> {STAGE4_BEST_CKPT} (slot {best_slot})")
                    summary.set("best_validation_score", float(best_validation))
                    summary.set("best_slot", int(best_slot))
                    summary.set("best_checkpoint", str(STAGE4_BEST_CKPT))

            # ── periodic save ────────────────────────────────────────────
            if update % args.save_every == 0 or update == start_update + total_updates:
                trainer.save_state(STAGE4_CKPT, update)
                summary.set("latest_checkpoint", str(STAGE4_CKPT))

            summary.write()
    finally:
        # Always persist + tear down even on exception/Ctrl-C.
        try:
            trainer.save_state(STAGE4_CKPT, locals().get("update", start_update))
        except Exception:
            pass
        trainer.close()

    print(f"\nSaved Stage-4 population → {STAGE4_CKPT}")
    if STAGE4_BEST_CKPT.exists():
        print(f"Best single learner → {STAGE4_BEST_CKPT}")
        print(f"Benchmark:  uv run ae_rl/benchmark.py --ckpt {STAGE4_BEST_CKPT}")


if __name__ == "__main__":
    main()
