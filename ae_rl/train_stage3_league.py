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
import shutil
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm, trange

import common  # noqa: F401  (path bootstrap)
from common import (
    LEAGUE_DIR,
    STAGE2_CKPT,
    STAGE3_BEST_CKPT,
    STAGE3_CKPT,
    get_device,
    seed_everything,
)
from run_summary import RunSummary, default_summary_path
from controllers import (
    berserker_spec,
    heuristic_spec,
    idle_spec,
    kamikaze_spec,
    league_checkpoints,
    net_spec,
    patroller_spec,
    pure_collector_spec,
    random_spec,
    stochastic_heuristic_spec,
    tactical_spec,
    trap_setter_spec,
    vanilla_heuristic_spec,
)
from model import RecurrentMaskableActorCritic, load_checkpoint, load_extras, save_checkpoint
from ppo import RunningReturnNorm, ppo_update
from rollout import SelfPlayCollector, default_workers
from validation import validate_model


def _load_start_model(device, init_ckpt=None):
    """Return ``(model, source_path)``; caller uses source_path to load extras."""
    if init_ckpt is not None:
        path = Path(init_ckpt)
        print(f"init checkpoint (--ckpt): {path}")
        return load_checkpoint(path, device), path
    for path, label in ((STAGE3_CKPT, "resume Stage 3"),
                        (STAGE2_CKPT, "warm-start from Stage 2")):
        if path.exists():
            print(f"{label}: {path}")
            return load_checkpoint(path, device), path
    raise FileNotFoundError(
        "Stage 3 requires a prerequisite checkpoint. Looked for:\n"
        f"  {STAGE3_CKPT}  (resume an in-progress Stage 3 run)\n"
        f"  {STAGE2_CKPT}  (warm-start from Stage 2)\n"
        "Pass --ckpt PATH to specify an explicit init checkpoint, or run "
        "train_stage2_ppo.py first."
    )


def _checkpoint_score(path) -> float:
    if not path.exists():
        return float("-inf")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        return float(ckpt.get("meta", {}).get("validation_score", float("-inf")))
    except Exception:
        return float("-inf")


_POOL_GRANULARITY = 100


def _build_opponent_specs(args, league_dir=None):
    """Return a weighted list of picklable opponent specs.

    Each non-net opponent type gets ``round(N * prob)`` entries in the pool where
    N=_POOL_GRANULARITY; ``random.choice`` then selects each type with frequency
    ≈ its prob. The remaining share (1 − sum of non-net probs) is split equally
    across frozen league snapshots, so as the league grows, individual snapshots
    are sampled less but the net-as-a-class share stays roughly constant.

    Diverse opponents (vanilla heuristic, berserker, pure collector, random,
    idle) widen the pool beyond the EditedHeuristicPolicyV2 family to fight the
    overfit-to-own-heuristic problem.
    """
    if league_dir is None:
        league_dir = LEAGUE_DIR
    pool = league_checkpoints(league_dir)
    n_nets = max(1, len(pool))

    # Per-type non-net probabilities, clamped to [0, 1].
    probs = {
        "heuristic": max(0.0, min(1.0, args.heuristic_prob)),
        "stochastic_heuristic": max(0.0, min(1.0, args.stochastic_heuristic_prob)),
        "vanilla_heuristic": max(0.0, min(1.0, args.vanilla_heuristic_prob)),
        "berserker": max(0.0, min(1.0, args.berserker_prob)),
        "pure_collector": max(0.0, min(1.0, args.pure_collector_prob)),
        "random": max(0.0, min(1.0, args.random_prob)),
        "idle": max(0.0, min(1.0, args.idle_prob)),
        "trap_setter": max(0.0, min(1.0, args.trap_setter_prob)),
        "patroller": max(0.0, min(1.0, args.patroller_prob)),
        "kamikaze": max(0.0, min(1.0, args.kamikaze_prob)),
        "tactical": max(0.0, min(1.0, args.tactical_prob)),
    }
    non_net_share = sum(probs.values())
    if non_net_share > 1.0:
        # Renormalise so the user can specify probs that sum to >1 without
        # blowing up — nets get 0% in that case.
        probs = {k: v / non_net_share for k, v in probs.items()}
        non_net_share = 1.0
    # Reserve adversary_prob out of the remaining pool. Adversary copies are
    # high-temperature reruns of the same league snapshots; they compete with
    # regular (deterministic-temperature) net opponents for the remaining share.
    adv_share = max(0.0, min(1.0, getattr(args, "adversary_prob", 0.0)))
    if non_net_share + adv_share > 1.0:
        adv_share = max(0.0, 1.0 - non_net_share)
    net_share = max(0.0, 1.0 - non_net_share - adv_share)

    spec_builders = {
        "heuristic": lambda: heuristic_spec(),
        "stochastic_heuristic": lambda: stochastic_heuristic_spec(
            args.stochastic_jitter, args.stochastic_action_noise
        ),
        "vanilla_heuristic": lambda: vanilla_heuristic_spec(),
        "berserker": lambda: berserker_spec(),
        "pure_collector": lambda: pure_collector_spec(),
        "random": lambda: random_spec(),
        "idle": lambda: idle_spec(),
        "trap_setter": lambda: trap_setter_spec(),
        "patroller": lambda: patroller_spec(),
        "kamikaze": lambda: kamikaze_spec(),
        "tactical": lambda: tactical_spec(),
    }

    specs: list[dict] = []
    counts: dict[str, int] = {}
    for kind, p in probs.items():
        c = round(_POOL_GRANULARITY * p)
        counts[kind] = c
        specs.extend(spec_builders[kind]() for _ in range(c))

    # Distribute the net share across snapshots — each snapshot gets the same
    # number of entries. Always at least 1 entry per snapshot so a new snapshot
    # is sampleable immediately.
    net_copies_per_snap = max(1, round(_POOL_GRANULARITY * net_share / n_nets)) if net_share > 0 else 0
    if net_copies_per_snap == 0 and net_share > 0:
        net_copies_per_snap = 1
    for ckpt in pool:
        for _ in range(net_copies_per_snap):
            specs.append(net_spec(ckpt))

    # Adversary copies: same snapshots but with --adversary-temperature applied at
    # action sampling. Widens the opponent distribution beyond what frozen
    # deterministic-temperature snapshots cover, without needing extra training.
    adv_temperature = float(getattr(args, "adversary_temperature", 2.0))
    adv_copies_per_snap = (
        max(1, round(_POOL_GRANULARITY * adv_share / n_nets)) if adv_share > 0 else 0
    )
    if adv_copies_per_snap == 0 and adv_share > 0:
        adv_copies_per_snap = 1
    for ckpt in pool:
        for _ in range(adv_copies_per_snap):
            specs.append(net_spec(ckpt, temperature=adv_temperature))

    if not specs:
        # Pathological all-zero config — fall back to one heuristic so training
        # doesn't crash.
        specs = [heuristic_spec()]
        counts["heuristic"] = 1

    total = len(specs)
    parts = [f"{c}x{kind}" for kind, c in counts.items() if c > 0]
    parts.append(f"{net_copies_per_snap}x{len(pool)} frozen snapshots")
    if adv_copies_per_snap > 0:
        parts.append(f"{adv_copies_per_snap}x{len(pool)} adversary @T={adv_temperature:.1f}")
    print("League opponents (" + str(total) + " total): " + " + ".join(parts))
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
    ap.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy bonus coefficient at the START of training. "
                         "If --entropy-coef-end is set, this is linearly annealed to "
                         "--entropy-coef-end over --entropy-anneal-until updates.")
    ap.add_argument("--entropy-coef-end", type=float, default=-1.0,
                    help="entropy coef at the end of the anneal window. Negative "
                         "(default) disables annealing — entropy_coef is held at "
                         "--entropy-coef for the whole run. Typical schedule for a "
                         "20k-update run: --entropy-coef 0.02 --entropy-coef-end 0.005.")
    ap.add_argument("--entropy-anneal-until", type=int, default=0,
                    help="updates over which to linearly anneal from --entropy-coef to "
                         "--entropy-coef-end. 0 (default) = use --updates so the anneal "
                         "covers the whole run. Has no effect if --entropy-coef-end < 0.")
    ap.add_argument("--clip", type=float, default=0.2)
    # Default opponent mix is "heuristic-free": the EditedHeuristicPolicyV2
    # family (heuristic, stochastic_heuristic, vanilla_heuristic) is zeroed out
    # so the policy stops overfitting to it. The strong-opponent slot is
    # filled by TacticalPolicy (1-step lookahead) + frozen self-snapshots.
    # Set --heuristic-prob > 0 explicitly if you want some heuristic exposure
    # back (e.g. as a regulariser).
    ap.add_argument("--heuristic-prob", type=float, default=0.0,
                    help="fraction of opponents drawn from the strong EditedHeuristicPolicyV2 "
                         "(default 0 — RL has overfit to this; re-enable explicitly if needed)")
    ap.add_argument("--stochastic-heuristic-prob", type=float, default=0.0,
                    help="fraction of opponents drawn from a parameter-jittered EditedHeuristicPolicyV2 "
                         "(default 0 — same overfitting concern as --heuristic-prob)")
    ap.add_argument("--vanilla-heuristic-prob", type=float, default=0.0,
                    help="fraction of opponents drawn from the vanilla HeuristicPolicy "
                         "(default 0 — shares the same heuristic family parent class)")
    ap.add_argument("--tactical-prob", type=float, default=0.30,
                    help="fraction of opponents drawn from TacticalPolicy "
                         "(1-step lookahead; the new 'strong but non-heuristic' opponent)")
    ap.add_argument("--berserker-prob", type=float, default=0.12,
                    help="fraction of opponents drawn from BerserkerPolicy (rushes enemy bases)")
    ap.add_argument("--pure-collector-prob", type=float, default=0.08,
                    help="fraction of opponents that only collect tiles, never bomb")
    ap.add_argument("--random-prob", type=float, default=0.05,
                    help="fraction of opponents that pick uniform random legal actions")
    ap.add_argument("--idle-prob", type=float, default=0.05,
                    help="fraction of opponents that mostly STAY (≈empty slot)")
    ap.add_argument("--trap-setter-prob", type=float, default=0.05,
                    help="fraction of opponents that wander and drop bombs anywhere")
    ap.add_argument("--patroller-prob", type=float, default=0.05,
                    help="fraction of opponents that walk FORWARD-until-blocked (non-adversarial)")
    ap.add_argument("--kamikaze-prob", type=float, default=0.05,
                    help="fraction of opponents that bomb at own feet when low-HP / cornered")
    ap.add_argument("--stochastic-jitter", type=float, default=0.35,
                    help="relative jitter for stochastic heuristic numeric knobs")
    ap.add_argument("--stochastic-action-noise", type=float, default=0.03,
                    help="chance stochastic heuristic takes a random legal action")
    ap.add_argument("--adversary-prob", type=float, default=0.0,
                    help="fraction of opponent pool drawn from HIGH-TEMPERATURE copies of "
                         "league snapshots. Reuses existing frozen snapshots but samples "
                         "actions with --adversary-temperature, exposing the learner to "
                         "stochastic variants of past selves without needing best-response "
                         "training. Comes out of the net share (regular snapshots get less).")
    ap.add_argument("--adversary-temperature", type=float, default=2.0,
                    help="logit temperature for adversary copies. >1 flattens the action "
                         "distribution (more exploration); 1 is the deterministic-temperature "
                         "policy. Default 2.0 is moderately noisy without becoming uniform.")
    ap.add_argument("--critic-warmup", type=int, default=8,
                    help="value-only updates before PPO begins (re-fits the value head to "
                         "the new opponent/reward distribution after warm-starting from Stage 2; "
                         "0 to skip)")
    ap.add_argument("--snapshot-every", type=int, default=20, help="updates between adding self to the league pool")
    ap.add_argument("--gated-snapshots", action="store_true",
                    help="only add league snapshots that pass the validation gate")
    ap.add_argument("--snapshot-margin", type=float, default=50.0,
                    help="allowed validation-score drop from best for gated snapshots")
    ap.add_argument("--league-max-size", type=int, default=0,
                    help="cap on number of league snapshots kept on disk. When a new "
                         "snapshot pushes the count above this, the oldest gen_*.pt is "
                         "deleted. 0 (default) = unlimited. Keeps the pool fresh — old "
                         "snapshots represent obsolete play styles that consume sample "
                         "budget without teaching anything new.")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--advanced-prob", type=float, default=0.0,
                    help="when training on --novice, probability a rollout episode uses an advanced random map")
    ap.add_argument("--validate-every", type=int, default=0,
                    help="run a quiet benchmark validation every N updates; 0 disables")
    ap.add_argument("--validation-rounds", type=int, default=20,
                    help="novice benchmark rounds per validation")
    ap.add_argument("--validation-advanced-rounds", type=int, default=0,
                    help="advanced-map benchmark rounds per validation")
    ap.add_argument("--validation-learners", type=int, default=1,
                    help="RL agents used in validation benchmark (default matches "
                         "benchmark.py's --learners 1 so scores are comparable)")
    ap.add_argument("--validation-baseline", type=str, default="strong",
                    choices=("strong", "vanilla", "berserker"),
                    help="opponent used in validation. 'strong' = EditedHeuristicPolicyV2 "
                         "(what you trained against — measures within-distribution fit). "
                         "'vanilla' / 'berserker' = held-out, measures generalisation. "
                         "Switch to a held-out baseline so checkpoint promotion + rollback "
                         "optimise for transfer to unseen opponents.")
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
    ap.add_argument("--no-shaping", dest="shape_rewards", action="store_false", default=True,
                    help="disable ALL training-time reward shaping (env-cfg overrides, "
                         "offensive multipliers, PBRS, oscillation + turn-spam penalties) "
                         "and train against raw env reward. Use as a polish phase on an "
                         "already-converged checkpoint to remove shaping bias and align "
                         "the gradient with the real eval objective.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-json", type=str, default="",
                    help="path for the run-summary JSON (default: ae_rl/runs/stage3_league/latest.json). "
                         "Read this from an autonomous caller instead of parsing stdout.")
    ap.add_argument("--ckpt", type=str, default="",
                    help="explicit init checkpoint path; overrides auto-discovery of stage3_league.pt / stage2_ppo.pt")
    ap.add_argument("--output-ckpt", type=str, default="",
                    help="path for the running checkpoint (default: ae_rl/checkpoints/stage3_league.pt)")
    ap.add_argument("--output-best", type=str, default="",
                    help="path for the best validated checkpoint (default: ae_rl/checkpoints/stage3_league_best.pt)")
    ap.add_argument("--league-dir", type=str, default="",
                    help="directory for league snapshots (default: ae_rl/checkpoints/league/)")
    ap.add_argument("--milestone-every", type=int, default=0,
                    help="save a timestamped copy of the current checkpoint AND the current best "
                         "every N updates into <output-ckpt-dir>/milestones/. 0 = disabled. "
                         "Recommended: 1000. Milestones are never overwritten and give safe "
                         "resume points regardless of rollback state or baseline changes.")
    args = ap.parse_args()

    summary_path = Path(args.summary_json) if args.summary_json else default_summary_path("stage3_league")
    with RunSummary(stage="stage3_league", args=vars(args), path=summary_path) as summary:
        _run_stage3(args, summary)


def _run_stage3(args, summary: RunSummary):
    seed_everything(args.seed)
    device = get_device()
    summary.set("device", str(device))
    summary.set("summary_path", str(summary.path))
    print(f"Device: {device}")
    print(f"Run summary: {summary.path}")

    # Resolve configurable output paths (flags override defaults from common.py).
    ckpt_path = Path(args.output_ckpt) if args.output_ckpt else STAGE3_CKPT
    best_ckpt_path = Path(args.output_best) if args.output_best else STAGE3_BEST_CKPT
    league_dir = Path(args.league_dir) if args.league_dir else LEAGUE_DIR
    league_dir.mkdir(parents=True, exist_ok=True)
    milestone_dir = ckpt_path.parent / "milestones" if args.milestone_every > 0 else None
    if milestone_dir is not None:
        milestone_dir.mkdir(parents=True, exist_ok=True)

    model, init_ckpt_path = _load_start_model(device, init_ckpt=args.ckpt or None)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Shared running stats for return normalisation — critical at Stage 3 because
    # raw shaped returns are in the hundreds, which makes the value loss explode
    # (saw v_loss=3000+ in the un-normalised version) and the critic chases noise.
    return_norm = RunningReturnNorm()
    # Restore running stats from the init checkpoint if present. Without this a
    # restart cold-starts the normaliser, so the critic re-fits to a shifting
    # scale for the first few hundred updates — visible as a value-loss spike
    # and a brief return dip on every resume.
    init_extras = load_extras(init_ckpt_path) if init_ckpt_path is not None else {}
    if "return_norm" in init_extras:
        return_norm.load_state_dict(init_extras["return_norm"])
        print(
            f"Restored return_norm from {init_ckpt_path.name}: "
            f"mean={return_norm.mean:+.2f} std={return_norm.std():.2f} count={return_norm.count}"
        )

    def _extras() -> dict:
        return {"return_norm": return_norm.state_dict()}

    def _entropy_coef_for(update: int) -> float:
        if args.entropy_coef_end < 0:
            return args.entropy_coef
        anneal_until = args.entropy_anneal_until or args.updates
        if anneal_until <= 0:
            return args.entropy_coef_end
        frac = min(1.0, max(0.0, (update - 1) / anneal_until))
        return args.entropy_coef + (args.entropy_coef_end - args.entropy_coef) * frac

    # Seed the league with the starting policy so there's at least one net opponent.
    if not league_checkpoints(league_dir):
        seed_path = league_dir / "gen_000.pt"
        save_checkpoint(seed_path, model, meta={"stage": "league_seed"})
        print(f"Seeded league with {seed_path}")

    specs = _build_opponent_specs(args, league_dir)
    summary.set("opponent_kinds", sorted({s["kind"] for s in specs}))
    summary.set("opponent_pool_size", len(specs))
    collector = SelfPlayCollector(
        model, device,
        opponent_specs=specs,
        n_learners=args.learners,
        novice=args.novice,
        advanced_prob=args.advanced_prob,
        gamma=args.gamma, lam=args.lam,
        num_workers=args.num_workers,
        shape_rewards=args.shape_rewards,
    )
    print(f"Rollout workers: {args.num_workers}  shape_rewards={args.shape_rewards}")
    if args.novice and args.advanced_prob > 0:
        print(f"Arena mix: novice with advanced_prob={args.advanced_prob:.2f}")

    gen = len(league_checkpoints(league_dir))
    summary.set("starting_generation", gen)
    best_validation = _checkpoint_score(best_ckpt_path)
    if best_validation > float("-inf"):
        print(f"Best validation checkpoint: {best_ckpt_path} score={best_validation:+.1f}")
        summary.set("best_validation_score", best_validation)
        summary.set("best_checkpoint", str(best_ckpt_path))

    # ── critic warm-up ───────────────────────────────────────────────────────
    # Stage 2's value head was fitted against (heuristic-only) opponents under the
    # old reward shaping. Stage 3 introduces (a) frozen league snapshots in the
    # opponent mix and (b) potentially-changed shaping. Without warm-up the first
    # few PPO advantages are garbage and the policy walks the wrong way before
    # the critic can catch up — observed empirically as val score dropping ~50
    # points in the first 10 updates of an un-warmed Stage 3 run.
    if args.critic_warmup > 0:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.critic.parameters():
            p.requires_grad_(True)
        for _ in trange(args.critic_warmup, desc="critic warmup", unit="upd"):
            batch, _ = collector.collect(args.episodes_per_update)
            wl = ppo_update(model, opt, batch, device, epochs=args.epochs,
                            seq_minibatch=args.seq_minibatch, value_only=True,
                            return_norm=return_norm)
            tqdm.write(f"  [warmup] v_loss={wl['value_loss']:.2f}")
        for p in model.parameters():
            p.requires_grad_(True)

    bar = trange(1, args.updates + 1, desc="League", unit="upd")
    for update in bar:
        t0 = time.time()
        batch, stats = collector.collect(args.episodes_per_update, progress=True)
        ent_coef = _entropy_coef_for(update)
        losses = ppo_update(
            model, opt, batch, device,
            epochs=args.epochs, seq_minibatch=args.seq_minibatch,
            clip=args.clip, entropy_coef=ent_coef,
            return_norm=return_norm,
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
            f"H={losses['entropy']:.3f} kl={losses['approx_kl']:.4f} ec={ent_coef:.4f}  "
            f"{stats['n_seqs']}seq {dt:.1f}s"
        )

        summary.increment("updates_completed")
        summary.set("latest_checkpoint", str(ckpt_path))
        summary.set("current_generation", gen)
        summary.record("updates", {
            "update": update,
            "ret_mean": float(stats["learner_return_mean"]),
            "opp_ret_mean": float(stats["opp_return_mean"]),
            "policy_loss": float(losses["policy_loss"]),
            "value_loss": float(losses["value_loss"]),
            "entropy": float(losses["entropy"]),
            "approx_kl": float(losses["approx_kl"]),
            "entropy_coef": float(ent_coef),
            "seconds": round(dt, 2),
            "gen": gen,
        })

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
                baseline=args.validation_baseline,
            )
            tqdm.write(
                f"  [val] score={val['score']:+.1f} rl={val['rl_mean']:.1f} "
                f"heur={val['heur_baseline']:.1f} suites={val['num_suites']}"
            )
            summary.record("validations", {
                "update": update,
                "score": float(val["score"]),
                "rl_mean": float(val["rl_mean"]),
                "heur_baseline": float(val["heur_baseline"]),
                "num_suites": int(val.get("num_suites", 0)),
                "baseline": args.validation_baseline,
            })
            if val["score"] > best_validation:
                best_validation = val["score"]
                summary.set("best_validation_score", float(best_validation))
                summary.set("best_checkpoint", str(best_ckpt_path))
                save_checkpoint(best_ckpt_path, model, meta={
                    "stage": "league_best",
                    "update": update,
                    "validation_score": val["score"],
                    "validation_rl_mean": val["rl_mean"],
                    "validation_heur_baseline": val["heur_baseline"],
                }, extras=_extras())
                tqdm.write(f"  [val] promoted best checkpoint -> {best_ckpt_path}")
            elif (
                args.rollback_on_regress
                and best_ckpt_path.exists()
                and val["score"] < best_validation - args.rollback_margin
            ):
                best_model = load_checkpoint(best_ckpt_path, device)
                model.load_state_dict(best_model.state_dict())
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                best_extras = load_extras(best_ckpt_path)
                if "return_norm" in best_extras:
                    return_norm.load_state_dict(best_extras["return_norm"])
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
                snap = league_dir / f"gen_{gen:03d}.pt"
                save_checkpoint(snap, model, meta={"stage": "league", "update": update})
                gen += 1
                tqdm.write(f"  + league snapshot {snap.name}  (rebuilding opponent pool)")
                summary.record("snapshots", {"update": update, "path": str(snap), "gen": gen})
                summary.set("current_generation", gen)
                # Prune oldest snapshots if the league has outgrown its cap. Keeps
                # the opponent distribution centred on recent play styles instead
                # of bleeding sample budget on obsolete generations.
                if args.league_max_size > 0:
                    all_snaps = sorted(league_dir.glob("*.pt"))
                    pruned = []
                    while len(all_snaps) > args.league_max_size:
                        oldest = all_snaps.pop(0)
                        try:
                            oldest.unlink()
                            pruned.append(oldest.name)
                        except OSError as e:
                            tqdm.write(f"  ! prune failed for {oldest.name}: {e}")
                    if pruned:
                        tqdm.write(f"  - pruned {len(pruned)} old snapshot(s): {', '.join(pruned)}")
                        summary.record("snapshot_prunes", {"update": update, "pruned": pruned})
                collector.set_opponent_specs(_build_opponent_specs(args, league_dir))

        if update % args.save_every == 0 or update == args.updates:
            save_checkpoint(ckpt_path, model, meta={"stage": "league", "update": update,
                                                    "learner_return_mean": stats["learner_return_mean"]},
                            extras=_extras())

        if milestone_dir is not None and args.milestone_every > 0 and update % args.milestone_every == 0:
            ms = milestone_dir / f"update_{update:06d}.pt"
            save_checkpoint(ms, model, meta={"stage": "league_milestone", "update": update,
                                             "learner_return_mean": stats["learner_return_mean"]},
                            extras=_extras())
            tqdm.write(f"  [milestone] saved {ms.name}")
            if best_ckpt_path.exists():
                ms_best = milestone_dir / f"update_{update:06d}_best.pt"
                shutil.copy2(best_ckpt_path, ms_best)
                tqdm.write(f"  [milestone] saved {ms_best.name}")

        # Persist progress every update so a polling reader sees fresh state.
        summary.write()

    collector.close()
    save_checkpoint(ckpt_path, model, meta={"stage": "league", "update": args.updates},
                    extras=_extras())
    summary.set("latest_checkpoint", str(ckpt_path))
    summary.set("final_generation", gen)
    print(f"\nSaved Stage-3 checkpoint → {ckpt_path}")
    print(f"Benchmark:  uv run ae_rl/benchmark.py --ckpt {ckpt_path}")


if __name__ == "__main__":
    main()
