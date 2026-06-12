"""Population trainer — diverse league runs, selected by the REAL eval.

This is the "evolution" idea done with the correct fitness function. Instead of
one self-play run selected by the heuristic-benchmark proxy (which plateaus), it:

  1. Launches K *diverse* league-training processes in parallel — different
     shaping ablations, PFSP modes, and seeds — each emitting milestone
     checkpoints into its own dir. Training runs continuously and never blocks.
  2. Launches ONE `eval_selector` process that drains every variant's milestones
     through the real organiser eval (serial — one submission in flight at a
     time, ~12–15 min each) and promotes the global best by real score.

So the evals run *concurrently with* training: the trainers keep producing
candidates while the selector grinds through them in the background. The real
eval, not a proxy, decides which run/shaping/curriculum actually wins.

Diversity is the point: each variant probes a different hypothesis about why RL
plateaued (over-aggressive shaping? misaligned reward? wrong curriculum?), and
the real eval adjudicates between them instead of you guessing.

Usage:
    # Launch the default 4-variant population + selector (needs a live Discord
    # watcher for the selector — see eval_selector --require-watcher):
    uv run ae_rl/train_population.py --init-ckpt ae_rl/checkpoints/stage1_bc_azbase.pt \
        --updates 4000 --milestone-every 150

    # Training only (you run the selector yourself, or later):
    uv run ae_rl/train_population.py --no-launch-selector ...

    # See the commands without launching anything:
    uv run ae_rl/train_population.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
POP_DIR = HERE / "checkpoints" / "pop"
RUNS_DIR = HERE / "runs"


# Each variant = a hypothesis. `args` are extra flags appended to the shared
# train_stage3_league.py command. All use PFSP (the anti-plateau curriculum) and
# an entropy floor; they differ in the reward they optimise and the curriculum.
VARIANTS: list[dict] = [
    {
        "name": "even-raw",
        "desc": "PFSP-even on RAW eval reward (no shaping) — pure align-to-eval.",
        "args": ["--pfsp", "--pfsp-mode", "even", "--no-shaping"],
    },
    {
        "name": "even-nomult",
        "desc": "PFSP-even, drop offensive multipliers, tiny base defense (-5).",
        "args": [
            "--pfsp",
            "--pfsp-mode",
            "even",
            "--no-offensive-multipliers",
            "--own-base-penalty",
            "-5",
        ],
    },
    {
        "name": "even-allon",
        "desc": "PFSP-even with all shaping on — the current shaping, prioritised.",
        "args": ["--pfsp", "--pfsp-mode", "even"],
    },
    {
        "name": "hard-nomult",
        "desc": "PFSP-hard (close the gap vs opponents we lose to), no multipliers.",
        "args": ["--pfsp", "--pfsp-mode", "hard", "--no-offensive-multipliers"],
    },
]


def _select_variants(args) -> list[dict]:
    if args.only:
        names = {n.strip() for n in args.only.split(",") if n.strip()}
        chosen = [v for v in VARIANTS if v["name"] in names]
        unknown = names - {v["name"] for v in VARIANTS}
        if unknown:
            sys.exit(
                f"unknown variant(s): {', '.join(sorted(unknown))}; "
                f"available: {', '.join(v['name'] for v in VARIANTS)}"
            )
        return chosen
    return VARIANTS[: args.variants]


def _variant_paths(name: str) -> dict:
    base = POP_DIR / name
    return {
        "league_dir": base / "league",
        "milestones": base / "milestones",  # = output-ckpt.parent / "milestones"
        "output_ckpt": base / "stage3.pt",
        "output_best": base / "best.pt",
        "summary": RUNS_DIR / f"pop_{name}" / "latest.json",
    }


def _build_train_cmd(variant: dict, args, jworkers: int, seed: int) -> list[str]:
    p = _variant_paths(variant["name"])
    cmd = [
        "uv",
        "run",
        "python",
        "ae_rl/train_stage3_league.py",
        "--ckpt",
        str(args.init_ckpt),
        "--updates",
        str(args.updates),
        "--episodes-per-update",
        str(args.episodes_per_update),
        "--milestone-every",
        str(args.milestone_every),
        "--snapshot-every",
        str(args.snapshot_every),
        "--league-max-size",
        "0",  # keep the whole archive for PFSP
        "--entropy-floor",
        str(args.entropy_floor),
        "--explore-burst-every",
        str(args.explore_burst_every),
        "--explore-burst-len",
        str(args.explore_burst_len),
        "--pfsp-every",
        str(args.pfsp_every),
        "-j",
        str(jworkers),
        "--seed",
        str(seed),
        "--league-dir",
        str(p["league_dir"]),
        "--output-ckpt",
        str(p["output_ckpt"]),
        "--output-best",
        str(p["output_best"]),
        "--summary-json",
        str(p["summary"]),
    ]
    if args.advanced:
        cmd.append("--advanced")
    cmd += variant["args"]
    return cmd


def _build_selector_cmd(variants: list[dict], args) -> list[str]:
    cmd = [
        "uv",
        "run",
        "ae_rl/eval_selector.py",
        "--promote-to",
        str(args.promote_to),
        "--tag-prefix",
        args.tag_prefix,
        "--timeout",
        str(int(args.eval_timeout)),
    ]
    for v in variants:
        cmd += ["--watch-dir", str(_variant_paths(v["name"])["milestones"])]
    if args.launch_watcher:
        cmd.append("--launch-watcher")
    if args.no_require_watcher:
        cmd.append("--no-require-watcher")
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--init-ckpt",
        type=str,
        default="ae_rl/checkpoints/stage3_league_best.pt",
        help="shared warm-start checkpoint for every variant. Default is the best "
        "LEAGUE checkpoint (~0.65), NOT the BC seed — raw BC evals ~0.2, so "
        "the self-play phase is the value-add and each variant should push "
        "PAST the league result, not re-climb to it. Swap for whichever of "
        "your league checkpoints scored highest on the real eval.",
    )
    ap.add_argument(
        "--variants",
        type=int,
        default=len(VARIANTS),
        help=f"how many of the default variants to run (max {len(VARIANTS)}).",
    )
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="comma-separated variant names to run instead "
        f"(available: {', '.join(v['name'] for v in VARIANTS)}).",
    )
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--episodes-per-update", type=int, default=8)
    ap.add_argument(
        "--milestone-every",
        type=int,
        default=150,
        help="emit a milestone checkpoint every N updates (the selector's "
        "candidate cadence). Match it roughly to eval throughput.",
    )
    ap.add_argument("--snapshot-every", type=int, default=25)
    ap.add_argument("--pfsp-every", type=int, default=25)
    ap.add_argument("--entropy-floor", type=float, default=0.008)
    ap.add_argument("--explore-burst-every", type=int, default=300)
    ap.add_argument("--explore-burst-len", type=int, default=5)
    ap.add_argument(
        "--seed-base",
        type=int,
        default=100,
        help="variant i gets seed = seed-base + i.",
    )
    ap.add_argument(
        "--advanced",
        action="store_true",
        help="train on randomised advanced maps instead of novice.",
    )
    ap.add_argument(
        "--workers-total",
        type=int,
        default=0,
        help="total rollout workers across all variants (default: cpus-2, "
        "split evenly, leaving headroom for the selector + docker build).",
    )
    ap.add_argument(
        "--launch-selector",
        dest="launch_selector",
        action="store_true",
        default=True,
        help="also launch the eval_selector over all milestone dirs (default).",
    )
    ap.add_argument(
        "--no-launch-selector",
        dest="launch_selector",
        action="store_false",
        help="training only; run eval_selector yourself.",
    )
    ap.add_argument(
        "--launch-watcher",
        dest="launch_watcher",
        action="store_true",
        default=True,
        help="let the selector spawn an ingest-only Discord watcher "
        "(`rl_autorun.py --watch-only`) so the whole pipeline is self-contained "
        "and nothing auto-submits behind your back (default on).",
    )
    ap.add_argument(
        "--no-launch-watcher",
        dest="launch_watcher",
        action="store_false",
        help="don't spawn a watcher — you're already running `rl_autorun.py --watch-only`.",
    )
    ap.add_argument(
        "--promote-to",
        type=str,
        default="ae_rl/checkpoints/eval_best.pt",
        help="selector copies the global real-eval best here (deploy candidate).",
    )
    ap.add_argument("--tag-prefix", type=str, default="pop")
    ap.add_argument("--eval-timeout", type=float, default=1800.0)
    ap.add_argument(
        "--no-require-watcher",
        action="store_true",
        help="pass through to the selector: submit even if the Discord watcher "
        "freshness check fails.",
    )
    ap.add_argument(
        "--poll-interval",
        type=float,
        default=120.0,
        help="seconds between status-table prints.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands that would launch and exit.",
    )
    args = ap.parse_args()

    variants = _select_variants(args)
    if not variants:
        sys.exit("no variants selected")

    init = Path(args.init_ckpt)
    if not init.is_absolute():
        init = REPO / init
    if not init.exists() and not args.dry_run:
        sys.exit(f"init checkpoint not found: {init}")

    cpus = os.cpu_count() or 4
    total_workers = args.workers_total or max(1, cpus - 2)
    jper = max(1, total_workers // len(variants))

    # Build commands.
    train_cmds = [
        (_build_train_cmd(v, args, jper, args.seed_base + i), v)
        for i, v in enumerate(variants)
    ]
    selector_cmd = _build_selector_cmd(variants, args) if args.launch_selector else None

    print(
        f"Population: {len(variants)} variant(s), {jper} workers each "
        f"({total_workers} total of {cpus} cpus). init={init.name}"
    )
    for cmd, v in train_cmds:
        print(f"\n[{v['name']}] {v['desc']}")
        print("   " + " ".join(cmd))
    if selector_cmd:
        print("\n[selector] real-eval fitness over all milestone dirs")
        print("   " + " ".join(selector_cmd))
    if args.dry_run:
        return

    # Launch. Per-variant stdout/stderr → log files so the console stays readable.
    # PYTHONUNBUFFERED so the logs flush line-by-line (otherwise block buffering
    # makes `tail -f` look frozen for minutes).
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    POP_DIR.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[str, subprocess.Popen, Path]] = []
    for cmd, v in train_cmds:
        paths = _variant_paths(v["name"])
        paths["league_dir"].mkdir(parents=True, exist_ok=True)
        paths["summary"].parent.mkdir(parents=True, exist_ok=True)
        log_path = POP_DIR / v["name"] / "train.log"
        log = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, env=child_env
        )
        procs.append((v["name"], proc, log_path))
        print(f"  launched {v['name']} pid={proc.pid} → {log_path}")
        time.sleep(2)  # stagger spawns so the env imports don't thundering-herd

    selector_proc = None
    if selector_cmd:
        sel_log = POP_DIR / "selector.log"
        selector_proc = subprocess.Popen(
            selector_cmd,
            cwd=str(REPO),
            env=child_env,
            stdout=open(sel_log, "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        print(f"  launched selector pid={selector_proc.pid} → {sel_log}")

    leaderboard = HERE / "tuning" / "eval_leaderboard.json"

    def _shutdown(*_):
        print("\nShutting down population…")
        for _, proc, _l in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        if selector_proc and selector_proc.poll() is None:
            selector_proc.send_signal(signal.SIGINT)
        time.sleep(5)
        for _, proc, _l in procs:
            if proc.poll() is None:
                proc.terminate()
        if selector_proc and selector_proc.poll() is None:
            selector_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Monitor: print a compact status table until all trainers exit.
    try:
        while True:
            time.sleep(args.poll_interval)
            rows = []
            for name, proc, _l in procs:
                status = "running" if proc.poll() is None else f"exit={proc.returncode}"
                upd, ret = "?", "?"
                sp = _variant_paths(name)["summary"]
                if sp.exists():
                    try:
                        d = json.loads(sp.read_text())
                        upd = d.get("updates_completed", "?")
                        ups = d.get("updates", [])
                        ret = round(ups[-1]["ret_mean"], 1) if ups else "?"
                    except (OSError, json.JSONDecodeError):
                        pass
                rows.append(f"  {name:14s} {status:12s} upd={upd} ret={ret}")
            best = "—"
            if leaderboard.exists():
                try:
                    b = json.loads(leaderboard.read_text()).get("best")
                    if b:
                        best = f"{b['score']:.4f} ({b['tag']})"
                except (OSError, json.JSONDecodeError):
                    pass
            print(
                f"\n[{time.strftime('%H:%M:%S')}] population status  "
                f"| real-eval best: {best}"
            )
            print("\n".join(rows))
            if all(proc.poll() is not None for _, proc, _l in procs):
                print(
                    "\nAll trainers finished. Selector left running "
                    "(Ctrl-C to stop it)."
                    if selector_proc
                    else "\nAll trainers finished."
                )
                if selector_proc:
                    selector_proc.wait()
                return
    except KeyboardInterrupt:
        _shutdown()


if __name__ == "__main__":
    main()
