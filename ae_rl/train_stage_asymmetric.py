"""Asymmetric (CTDE) PPO — privileged critic + PFSP + KL-anchor-to-BC.

This is the "real fix" run, built to close the gap to the strong scripted /
league tier. Three levers stack on top of the Stage-2/3 recurrent-PPO core:

1. **Privileged critic (CTDE).** The actor still sees only its local, partial
   observation and ships to deploy unchanged. The *critic*, used only in
   training, sees the whole arena (every agent/base/bomb/collectible — see
   ``global_state.py``). In this 6-way FFA the per-agent return depends on five
   opponents the actor can't observe, so a local critic's value estimate is
   swamped by that hidden variance; the privileged critic removes it and the
   policy gradient stops chasing noise. (Asymmetric actor-critic — Pinto 2017;
   the MAPPO centralized-critic idea, adapted to FFA where "global" replaces
   "joint teammate".)

2. **PFSP opponent curriculum.** Opponents are sampled weighted toward the ones
   the policy currently loses to (azbase, league snapshots), not uniformly. See
   ``pfsp.py``.

3. **KL anchor to the BC policy.** A ``coef·KL(π ‖ π_BC)`` penalty keeps the
   policy from unlearning the behaviour-cloned competence under the new reward /
   opponent distribution. Annealed to zero over training.

Deploy story is unchanged: ``save_asymmetric_checkpoint`` writes the actor under
``model_state`` (which ``ae/src/policies/rl_policy.py`` loads by name) and the
privileged critic under a separate key the deploy loader ignores.

Usage:
    python ae_rl/train_stage_asymmetric.py --updates 4000 --validate-every 50 \
        --rollback-on-regress
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm, trange

import common  # noqa: F401  (path bootstrap)
from common import (
    ASYM_LEAGUE_DIR,
    STAGE1_CKPT,
    STAGE2_BEST_CKPT,
    STAGE2_CKPT,
    STAGE3_BEST_CKPT,
    STAGE3_CKPT,
    STAGE_ASYM_BEST_CKPT,
    STAGE_ASYM_CKPT,
    get_device,
    seed_everything,
)
from controllers import (
    azbasev1_spec,
    azbasev4_spec,
    berserker_spec,
    league_checkpoints,
    net_spec,
    tactical_spec,
)
from global_state import GLOBAL_GRID_SHAPE, GLOBAL_SCALAR_DIM
from model import (
    AsymmetricActorCritic,
    RecurrentMaskableActorCritic,
    load_asymmetric_checkpoint,
    load_checkpoint,
    save_asymmetric_checkpoint,
    save_checkpoint,
)
from pfsp import PFSPSampler
from ppo import RunningReturnNorm, ppo_update_asymmetric
from rollout import SelfPlayCollector, default_workers
from run_summary import RunSummary, default_summary_path
from validation import validate_model


def _auto_seed(explicit: str) -> Path | None:
    """Find a checkpoint to warm-start the actor from. Prefer an azbase-BC seed
    if one was created, else the standard stage progression."""
    if explicit:
        p = Path(explicit)
        print(f"actor seed (--seed-ckpt): {p}")
        return p
    candidates = [
        (STAGE_ASYM_CKPT, "resume asymmetric population"),
        (Path("ae_rl/checkpoints/stage1_bc_azbase.pt"), "azbase-BC seed"),
        (STAGE3_BEST_CKPT, "Stage 3 best"),
        (STAGE3_CKPT, "Stage 3 latest"),
        (STAGE2_BEST_CKPT, "Stage 2 best"),
        (STAGE2_CKPT, "Stage 2 latest"),
        (STAGE1_CKPT, "Stage 1 BC"),
    ]
    for path, label in candidates:
        if path.exists():
            print(f"actor seed ({label}): {path}")
            return path
    print("no seed found — actor starts from scratch (slow; prefer a BC seed)")
    return None


def _checkpoint_score(path: Path) -> float:
    if not path.exists():
        return float("-inf")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        return float(ckpt.get("meta", {}).get("validation_score", float("-inf")))
    except Exception:
        return float("-inf")


def _build_candidates(args) -> list[tuple[str, dict]]:
    """Scripted strong opponents + every league snapshot currently on disk."""
    cands: list[tuple[str, dict]] = [
        ("tactical", tactical_spec()),
        ("berserker", berserker_spec()),
        ("azbasev1", azbasev1_spec()),
        ("azbasev4", azbasev4_spec()),
    ]
    for ckpt in league_checkpoints(ASYM_LEAGUE_DIR):
        cands.append((f"league/{ckpt.stem}", net_spec(ckpt)))
    return cands


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # schedule
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--episodes-per-update", type=int, default=8)
    ap.add_argument("--learners", type=int, default=2,
                    help="RL-controlled agents per game (rest = PFSP opponents)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seq-minibatch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    # KL anchor (annealed linearly to 0 over --kl-anneal-until updates)
    ap.add_argument("--kl-anchor-coef", type=float, default=0.1,
                    help="initial weight of KL(policy‖BC). 0 disables the anchor.")
    ap.add_argument("--kl-anneal-until", type=int, default=2000,
                    help="updates over which the KL anchor decays to 0. 0 = constant.")
    ap.add_argument("--bc-anchor", type=str, default="",
                    help="checkpoint whose actor is the frozen KL-anchor target. "
                         "Default: the actor seed checkpoint. Empty + no seed = no anchor.")
    # critic warmup
    ap.add_argument("--critic-warmup", type=int, default=12,
                    help="value-only updates before policy training, so the fresh "
                         "privileged critic isn't producing garbage advantages.")
    # PFSP
    ap.add_argument("--pfsp-mode", type=str, default="hard", choices=("hard", "even"))
    ap.add_argument("--pfsp-q", type=float, default=2.0)
    ap.add_argument("--pfsp-every", type=int, default=25,
                    help="updates between PFSP win-rate refreshes + pool rebuilds")
    ap.add_argument("--pfsp-eval-episodes", type=int, default=6,
                    help="games per opponent per PFSP refresh")
    # league
    ap.add_argument("--snapshot-every", type=int, default=200,
                    help="updates between adding the current actor to the PFSP "
                         "league as a frozen opponent. 0 = never.")
    ap.add_argument("--league-max-size", type=int, default=15)
    # validation + rollback
    ap.add_argument("--validate-every", type=int, default=50)
    ap.add_argument("--validation-rounds", type=int, default=20)
    ap.add_argument("--validation-advanced-rounds", type=int, default=0)
    ap.add_argument("--validation-learners", type=int, default=1)
    ap.add_argument("--validation-baseline", type=str, default="vanilla",
                    choices=("strong", "vanilla", "berserker", "azbasev1", "azbasev4"),
                    help="held-out opponent gating best-promotion + rollback. "
                         "'vanilla' = broad generalisation; 'azbasev1'/'azbasev4' "
                         "gate on the STRONG tier we actually need to beat — use "
                         "these if vanilla-promotion isn't tracking real eval.")
    ap.add_argument("--validation-seed", type=int, default=22345)
    ap.add_argument("--rollback-on-regress", action="store_true",
                    help="reload best checkpoint if validation drops > rollback-margin")
    ap.add_argument("--rollback-margin", type=float, default=75.0)
    # env
    ap.add_argument("--novice", dest="novice", action="store_true", default=True)
    ap.add_argument("--advanced", dest="novice", action="store_false")
    ap.add_argument("--advanced-prob", type=float, default=0.0)
    ap.add_argument("--no-shaping", dest="shape_rewards", action="store_false", default=True)
    ap.add_argument("-j", "--num-workers", type=int, default=default_workers())
    # io
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-ckpt", type=str, default="")
    ap.add_argument("--summary-json", type=str, default="")
    args = ap.parse_args()

    summary_path = Path(args.summary_json) if args.summary_json else default_summary_path("stage_asym")
    with RunSummary(stage="stage_asym", args=vars(args), path=summary_path) as summary:
        _run(args, summary)


def _run(args, summary: RunSummary):
    seed_everything(args.seed)
    device = get_device()
    summary.set("device", str(device))
    summary.set("summary_path", str(summary.path))
    print(f"Device: {device}")
    print(f"Run summary: {summary.path}")

    seed_path = _auto_seed(args.seed_ckpt)

    # ── build asymmetric model ────────────────────────────────────────────
    model = AsymmetricActorCritic(GLOBAL_GRID_SHAPE, GLOBAL_SCALAR_DIM).to(device)
    resumed = False
    if seed_path is not None and seed_path.exists():
        if seed_path == STAGE_ASYM_CKPT:
            model = load_asymmetric_checkpoint(seed_path, device)
            resumed = True
        else:
            # Warm-start the actor only (plain BC/league checkpoint).
            actor_seed = load_checkpoint(seed_path, device)
            model.actor.load_state_dict(actor_seed.state_dict())
            print(f"warm-started actor from {seed_path.name}")
    model = model.to(device)

    # ── frozen BC anchor ──────────────────────────────────────────────────
    bc_model = None
    anchor_path = Path(args.bc_anchor) if args.bc_anchor else seed_path
    if args.kl_anchor_coef > 0 and anchor_path is not None and Path(anchor_path).exists():
        try:
            bc_model = load_checkpoint(Path(anchor_path), device, eval_mode=True)
            for p in bc_model.parameters():
                p.requires_grad_(False)
            print(f"KL anchor target: {Path(anchor_path).name}")
            summary.set("kl_anchor_target", str(anchor_path))
        except Exception as e:
            print(f"could not load KL anchor ({e}); training without it")
            bc_model = None
    if bc_model is None and args.kl_anchor_coef > 0:
        print("KL anchor requested but no anchor checkpoint available — disabled")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    return_norm = RunningReturnNorm()

    # ── PFSP sampler + collector ──────────────────────────────────────────
    candidates = _build_candidates(args)
    sampler = PFSPSampler(candidates, mode=args.pfsp_mode, q=args.pfsp_q)
    print(f"PFSP candidates ({len(candidates)}): {sampler.ids()}")
    summary.set("pfsp_candidates", sampler.ids())

    collector = SelfPlayCollector(
        model=model.actor, device=device,
        opponent_specs=[tactical_spec()],   # placeholder; PFSP overrides per collect
        n_learners=args.learners, novice=args.novice,
        advanced_prob=args.advanced_prob,
        gamma=args.gamma, lam=args.lam,
        num_workers=args.num_workers, shape_rewards=args.shape_rewards,
        collect_global_state=True,
    )

    best_validation = _checkpoint_score(STAGE_ASYM_BEST_CKPT)
    if best_validation > float("-inf"):
        print(f"seeded best_validation from {STAGE_ASYM_BEST_CKPT}: {best_validation:+.1f}")
        summary.set("best_validation_score", float(best_validation))

    # initial PFSP estimate → first training pool
    print("PFSP: initial opponent evaluation …")
    info = sampler.refresh(collector, args.pfsp_eval_episodes)
    pool = sampler.weighted_pool()
    summary.record("pfsp_refreshes", {"update": 0, **info})
    tqdm.write("  PFSP " + "  ".join(f"{o['id']}:p={o['p_win']:.2f}" for o in info["per_opponent"]))

    # ── critic warm-up (value-only) ───────────────────────────────────────
    if args.critic_warmup > 0 and not resumed:
        for _ in trange(args.critic_warmup, desc="critic warmup", unit="upd"):
            batch, _ = collector.collect(args.episodes_per_update, opp_specs_override=pool)
            wl = ppo_update_asymmetric(
                model, opt, batch, device,
                gamma=args.gamma, lam=args.lam, epochs=args.epochs,
                seq_minibatch=args.seq_minibatch, value_only=True,
                return_norm=return_norm,
            )
            tqdm.write(f"  [warmup] v_loss={wl['value_loss']:.3f}")

    # ── main loop ─────────────────────────────────────────────────────────
    def _kl_coef(update: int) -> float:
        if args.kl_anchor_coef <= 0 or bc_model is None:
            return 0.0
        if args.kl_anneal_until <= 0:
            return args.kl_anchor_coef
        frac = min(1.0, max(0.0, update / args.kl_anneal_until))
        return args.kl_anchor_coef * (1.0 - frac)

    bar = trange(1, args.updates + 1, desc="Asym", unit="upd")
    for update in bar:
        t0 = time.time()

        if args.pfsp_every > 0 and update % args.pfsp_every == 0:
            info = sampler.refresh(collector, args.pfsp_eval_episodes)
            pool = sampler.weighted_pool()
            summary.record("pfsp_refreshes", {"update": update, **info})
            tqdm.write("  [pfsp] " + "  ".join(
                f"{o['id']}:p={o['p_win']:.2f}(m={o['margin']:+.0f})" for o in info["per_opponent"]))

        batch, stats = collector.collect(args.episodes_per_update, opp_specs_override=pool)
        kl_coef = _kl_coef(update)
        losses = ppo_update_asymmetric(
            model, opt, batch, device,
            gamma=args.gamma, lam=args.lam, epochs=args.epochs,
            seq_minibatch=args.seq_minibatch, clip=args.clip,
            value_coef=args.value_coef, entropy_coef=args.entropy_coef,
            return_norm=return_norm, bc_model=bc_model, kl_anchor_coef=kl_coef,
        )
        dt = time.time() - t0

        bar.set_postfix(ret=f"{stats['learner_return_mean']:.0f}",
                        v=f"{losses['value_loss']:.2f}",
                        kl=f"{losses['approx_kl']:.3f}")
        tqdm.write(
            f"upd {update:4d}/{args.updates}  ret={stats['learner_return_mean']:7.1f} "
            f"(opp {stats['opp_return_mean']:6.1f})  pi={losses['policy_loss']:+.3f} "
            f"v={losses['value_loss']:.2f} H={losses['entropy']:.3f} "
            f"kl={losses['approx_kl']:.4f} klA={losses['kl_anchor']:.3f}(c={kl_coef:.3f})  {dt:.1f}s"
        )

        summary.increment("updates_completed")
        summary.record("updates", {
            "update": update,
            "ret_mean": float(stats["learner_return_mean"]),
            "opp_ret_mean": float(stats["opp_return_mean"]),
            "policy_loss": float(losses["policy_loss"]),
            "value_loss": float(losses["value_loss"]),
            "entropy": float(losses["entropy"]),
            "approx_kl": float(losses["approx_kl"]),
            "kl_anchor": float(losses["kl_anchor"]),
            "kl_coef": float(kl_coef),
            "seconds": round(dt, 2),
        })

        # ── validation + rollback ─────────────────────────────────────────
        if args.validate_every > 0 and update % args.validate_every == 0:
            val = validate_model(
                model.actor, rounds=args.validation_rounds,
                learners=args.validation_learners, novice=args.novice,
                seed=args.validation_seed,
                advanced_rounds=args.validation_advanced_rounds,
                baseline=args.validation_baseline,
            )
            tqdm.write(
                f"  [val] score={val['score']:+.1f} rl={val['rl_mean']:.1f} "
                f"base={val['heur_baseline']:.1f} (baseline={args.validation_baseline})"
            )
            summary.record("validations", {
                "update": update, "score": float(val["score"]),
                "rl_mean": float(val["rl_mean"]),
                "heur_baseline": float(val["heur_baseline"]),
                "baseline": args.validation_baseline,
            })
            if val["score"] > best_validation:
                best_validation = val["score"]
                save_asymmetric_checkpoint(STAGE_ASYM_BEST_CKPT, model, meta={
                    "stage": "stage_asym_best", "update": update,
                    "validation_score": val["score"],
                    "validation_rl_mean": val["rl_mean"],
                })
                tqdm.write(f"  [val] promoted best -> {STAGE_ASYM_BEST_CKPT} ({best_validation:+.1f})")
                summary.set("best_validation_score", float(best_validation))
                summary.set("best_checkpoint", str(STAGE_ASYM_BEST_CKPT))
            elif (args.rollback_on_regress and STAGE_ASYM_BEST_CKPT.exists()
                  and val["score"] < best_validation - args.rollback_margin):
                restored = load_asymmetric_checkpoint(STAGE_ASYM_BEST_CKPT, device)
                model.load_state_dict(restored.state_dict())
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                tqdm.write(f"  [val] rollback to best ({val['score']:+.1f} < {best_validation:+.1f})")

        # ── league snapshot ───────────────────────────────────────────────
        if args.snapshot_every > 0 and update % args.snapshot_every == 0:
            gen = len(league_checkpoints(ASYM_LEAGUE_DIR))
            snap = ASYM_LEAGUE_DIR / f"gen_{gen:03d}.pt"
            save_checkpoint(snap, model.actor, meta={"stage": "asym_league", "update": update})
            sampler.add_candidate(f"league/{snap.stem}", net_spec(snap))
            tqdm.write(f"  [league] snapshot {snap.name}; PFSP now tracks {len(sampler.ids())} opponents")
            summary.record("snapshots", {"update": update, "path": str(snap)})
            # prune oldest
            if args.league_max_size > 0:
                snaps = league_checkpoints(ASYM_LEAGUE_DIR)
                while len(snaps) > args.league_max_size:
                    old = snaps.pop(0)
                    try:
                        old.unlink()
                    except OSError:
                        pass

        # ── periodic save ─────────────────────────────────────────────────
        if update % args.save_every == 0 or update == args.updates:
            save_asymmetric_checkpoint(STAGE_ASYM_CKPT, model, meta={
                "stage": "stage_asym", "update": update,
                "learner_return_mean": float(stats["learner_return_mean"]),
            })
            summary.set("latest_checkpoint", str(STAGE_ASYM_CKPT))
            summary.set("pfsp_state", sampler.summary())

        summary.write()

    collector.close()
    print(f"\nSaved asymmetric checkpoint → {STAGE_ASYM_CKPT}")
    if STAGE_ASYM_BEST_CKPT.exists():
        print(f"Best (deploy-loadable actor in model_state) → {STAGE_ASYM_BEST_CKPT}")
        print(f"Benchmark:  uv run ae_rl/benchmark.py --ckpt {STAGE_ASYM_BEST_CKPT}")


if __name__ == "__main__":
    main()
