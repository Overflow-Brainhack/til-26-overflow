"""Per-event scoring breakdown for a trained RL checkpoint.

Runs N rounds with the RL model controlling one slot (rotating each round so
every slot gets exercised) and reports, per slot and overall:

  * episode score (mean, std, min, max)
  * which reward events drove the score (mission tiles, kills, base destruction,
    invalid actions, stationary penalty, …) — by spying on
    ``env.dynamics.rewards.award``
  * action distribution (FORWARD / BACKWARD / LEFT / RIGHT / STAY / PLACE_BOMB)
  * illegal-action attempts (network sampled a masked-out action)
  * heuristic-baseline reference (same map, all 6 agents heuristic) so you can
    tell whether the RL slot is over- or under-performing the heuristic.

Usage:
    uv run ae_rl/diagnose.py
    uv run ae_rl/diagnose.py --ckpt ae_rl/checkpoints/stage2_ppo_best.pt --rounds 30
    uv run ae_rl/diagnose.py --focus-slot agent_0 --rounds 20 --sample-actions
    uv run ae_rl/diagnose.py --advanced --rounds 30
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

import common  # noqa: F401  (path bootstrap)
from common import CKPT_DIR, STAGE1_CKPT, STAGE2_CKPT, STAGE3_CKPT, get_device
from constants import Action
from controllers import HeuristicController, LayeredNetController, NetController
from model import load_checkpoint
from rollout import make_env


def _newest_ckpt() -> Path | None:
    for p in (STAGE3_CKPT, STAGE2_CKPT, STAGE1_CKPT):
        if p.exists():
            return p
    found = sorted(CKPT_DIR.glob("*.pt"))
    return found[-1] if found else None


def _play_with_spies(env, controllers, focus_agent, seed):
    """Play one game; return (per-agent score, focus event counter,
    focus action counter, focus illegal-action count)."""
    env.reset(seed=seed)
    for c in controllers.values():
        c.reset()

    by_agent_event: dict[str, Counter[str]] = defaultdict(Counter)
    original_award = env.dynamics.rewards.award

    # Events the env fires bidirectionally (positive to attacker, negative to victim)
    # — split them by sign so the diagnostic shows dealt vs taken separately.
    _BIDIR_EVENTS = {"attack_damage"}

    def award_spy(recipient_id: str, event: str, multiplier: float = 1.0) -> float:
        value = original_award(recipient_id, event, multiplier)
        if value != 0.0:
            if event in _BIDIR_EVENTS:
                tag = f"{event}_{'dealt' if value > 0 else 'taken'}"
                by_agent_event[recipient_id][tag] += value
            else:
                by_agent_event[recipient_id][event] += value
        return value

    env.dynamics.rewards.award = award_spy

    focus_actions: Counter[str] = Counter()
    focus_illegal = 0

    try:
        while True:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                if all(env.terminations.values()) or all(env.truncations.values()):
                    break
                continue
            obs = env.observe(agent)
            action = int(controllers[agent].act(obs))
            if agent == focus_agent:
                focus_actions[Action(action).name] += 1
                mask = obs.get("action_mask")
                if mask is not None and 0 <= action < len(mask) and not bool(mask[action]):
                    focus_illegal += 1
            env.step(action)
    finally:
        env.dynamics.rewards.award = original_award

    episode = getattr(env.dynamics.rewards, "_episode", {})
    scores = {a: float(episode.get(a, 0.0)) for a in env.possible_agents}
    return scores, by_agent_event[focus_agent], focus_actions, focus_illegal


def _fmt_signed(x: float) -> str:
    return f"{x:+.1f}" if x else "  0.0"


def _print_event_breakdown(title: str, events: Counter[str], total_rounds: int) -> None:
    print(f"\n{title}")
    if not events:
        print("  (no events recorded)")
        return
    grand = sum(abs(v) for v in events.values()) or 1.0
    rows = sorted(events.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print(f"  {'event':<28s} {'total':>10s} {'avg/round':>11s} {'|share|':>9s}")
    for name, total in rows:
        avg = total / max(1, total_rounds)
        share = 100.0 * abs(total) / grand
        print(f"  {name:<28s} {_fmt_signed(total):>10s} {_fmt_signed(avg):>11s} {share:8.1f}%")


def _print_action_breakdown(actions: Counter[str], illegal: int) -> None:
    print("\nFocus-slot action distribution")
    total = sum(actions.values()) or 1
    for name in (a.name for a in Action):
        n = actions.get(name, 0)
        print(f"  {name:<12s} {n:6d}  {100.0 * n / total:5.1f}%")
    print(f"\n  illegal-action attempts: {illegal} / {total}  "
          f"({100.0 * illegal / max(1, total):.2f}%)")


def _print_per_slot_table(per_slot_scores: dict[str, list[float]],
                          per_slot_baseline: dict[str, list[float]]) -> None:
    print("\nPer-slot RL focus score vs heuristic-baseline reference")
    print(f"  {'slot':<10s} {'n':>3s} {'rl_mean':>9s} {'rl_std':>8s} "
          f"{'rl_min':>8s} {'rl_max':>8s} {'heur_ref':>9s} {'delta':>9s}")
    for slot in sorted(per_slot_scores):
        rl = per_slot_scores[slot]
        ref = per_slot_baseline.get(slot, [])
        if not rl:
            continue
        rl_mean = mean(rl)
        rl_std = stdev(rl) if len(rl) > 1 else 0.0
        ref_mean = mean(ref) if ref else 0.0
        delta = rl_mean - ref_mean
        print(f"  {slot:<10s} {len(rl):3d} {rl_mean:9.1f} {rl_std:8.1f} "
              f"{min(rl):8.1f} {max(rl):8.1f} {ref_mean:9.1f} {_fmt_signed(delta):>9s}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint path (default: newest stage)")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--novice", dest="novice", action="store_true", default=True)
    ap.add_argument("--advanced", dest="novice", action="store_false")
    ap.add_argument("--focus-slot", type=str, default=None,
                    help="if set, RL always controls this slot (e.g. agent_0); "
                         "otherwise rotates across all 6 slots")
    ap.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    ap.add_argument("--sample-actions", dest="deterministic", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-baseline", dest="run_baseline", action="store_false", default=True,
                    help="skip the all-heuristic reference game (saves ~half the wallclock)")
    ap.add_argument("--layered", dest="layered", action="store_true", default=False,
                    help="wrap the RL controller with dodge override + loop break "
                         "(deploy-side LayeredRLPolicy equivalent)")
    ap.add_argument("--no-dodge", dest="dodge_override", action="store_false", default=True,
                    help="when --layered is set, disable the dodge override guard")
    ap.add_argument("--no-loop-break", dest="oscillation_break", action="store_false", default=True,
                    help="when --layered is set, disable the loop-break guard")
    ap.add_argument("--heuristic-fallback", action="store_true", default=False,
                    help="when --layered is set, fall back to the heuristic on low-value "
                         "or high-entropy steps")
    ap.add_argument("--value-threshold", type=float, default=None,
                    help="fall back to heuristic when RL value < this (try -0.5)")
    ap.add_argument("--entropy-threshold-frac", type=float, default=None,
                    help="fall back to heuristic when entropy/max_entropy > this (try 0.85)")
    args = ap.parse_args()

    random.seed(args.seed)

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = _newest_ckpt()
        if ckpt is None:
            raise SystemExit("No checkpoint found.")
    ckpt = Path(ckpt)
    print(f"Checkpoint: {ckpt}")
    print(f"Map: {'novice' if args.novice else 'advanced'}   "
          f"rounds: {args.rounds}   "
          f"focus: {args.focus_slot or '(rotating all slots)'}   "
          f"deterministic: {args.deterministic}   "
          f"layered: {args.layered}"
          + (f" (dodge={args.dodge_override}, loop_break={args.oscillation_break})"
             if args.layered else ""))

    device = get_device()
    model = load_checkpoint(ckpt, device, eval_mode=True)
    env = make_env(args.novice)
    agents = list(env.possible_agents)

    if args.focus_slot is not None and args.focus_slot not in agents:
        raise SystemExit(f"--focus-slot {args.focus_slot!r} not in {agents}")

    per_slot_scores: dict[str, list[float]] = defaultdict(list)
    per_slot_baseline: dict[str, list[float]] = defaultdict(list)
    total_events: Counter[str] = Counter()
    total_actions: Counter[str] = Counter()
    total_illegal = 0
    opp_scores: list[float] = []

    for r in range(args.rounds):
        s = random.randint(0, 2_000_000_000)
        if args.focus_slot is not None:
            focus = args.focus_slot
        else:
            focus = agents[r % len(agents)]

        ctrl = {}
        for a in agents:
            if a == focus:
                if args.layered:
                    ctrl[a] = LayeredNetController(
                        model, device, name="rl_layered",
                        deterministic=args.deterministic, novice=args.novice,
                        dodge_override=args.dodge_override,
                        oscillation_break=args.oscillation_break,
                        heuristic_fallback=args.heuristic_fallback,
                        value_threshold=args.value_threshold,
                        entropy_threshold_frac=args.entropy_threshold_frac,
                    )
                else:
                    ctrl[a] = NetController(model, device, name="rl",
                                            deterministic=args.deterministic,
                                            novice=args.novice)
            else:
                ctrl[a] = HeuristicController(use_cache=args.novice)

        scores, events, actions, illegal = _play_with_spies(env, ctrl, focus, s)
        per_slot_scores[focus].append(scores[focus])
        for a in agents:
            if a != focus:
                opp_scores.append(scores[a])
        total_events.update(events)
        total_actions.update(actions)
        total_illegal += illegal

        # Reference game: same seed, all heuristic.
        if args.run_baseline:
            ref_ctrl = {a: HeuristicController(use_cache=args.novice) for a in agents}
            ref_scores, _, _, _ = _play_with_spies(env, ref_ctrl, focus, s)
            per_slot_baseline[focus].append(ref_scores[focus])

        print(f"  round {r+1:3d}  focus={focus}  rl={scores[focus]:7.1f}  "
              f"opp_mean={mean(scores[a] for a in agents if a != focus):7.1f}"
              + (f"  heur_ref={ref_scores[focus]:7.1f}"
                 if args.run_baseline else ""))

    # ── aggregate tables ──────────────────────────────────────────────────
    print("\n" + "═" * 80)
    all_rl_scores = [s for lst in per_slot_scores.values() for s in lst]
    print(f"Overall RL focus score:  mean={mean(all_rl_scores):.1f}  "
          f"std={(stdev(all_rl_scores) if len(all_rl_scores) > 1 else 0):.1f}  "
          f"min={min(all_rl_scores):.1f}  max={max(all_rl_scores):.1f}  "
          f"n={len(all_rl_scores)}")
    if opp_scores:
        print(f"Opponent (heuristic) in-game mean: {mean(opp_scores):.1f}")
    if args.run_baseline:
        all_ref = [s for lst in per_slot_baseline.values() for s in lst]
        if all_ref:
            print(f"All-heuristic reference mean:      {mean(all_ref):.1f}")
            print(f"Delta (RL − reference):            "
                  f"{_fmt_signed(mean(all_rl_scores) - mean(all_ref))}")

    _print_per_slot_table(per_slot_scores, per_slot_baseline)

    _print_event_breakdown(
        "Reward events attributed to RL focus slot (totals across all RL rounds)",
        total_events, total_rounds=len(all_rl_scores),
    )

    _print_action_breakdown(total_actions, total_illegal)


if __name__ == "__main__":
    main()
