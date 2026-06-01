"""Eval-in-the-loop checkpoint selector — make the REAL organiser eval the fitness function.

The training-time benchmark (`benchmark.py` / `validation.py`) scores a checkpoint
against the heuristic family. That proxy does *not* predict the organiser eval —
the whole plateau is the proxy saturating while the real objective never enters
the loop. This module closes that loop: it submits candidate checkpoints to the
real eval (via the existing `rl_autorun.py` pipeline) and keeps a leaderboard
ranked by the *real* score.

Decoupled from training by design. Training keeps emitting candidate checkpoints
(league milestones, snapshots, population members) into one or more watch
directories; this selector drains them one at a time — the eval is serial (one
submission in flight, ~12–15 min round-trip) so we never need more than a queue.

Mechanics (no new infra — reuses `rl_autorun.py` exactly as AUTOTUNE documents):

    RL_AUTORUN_CHECKPOINT=<abs path> RL_AUTORUN_STAGE=3 \
        uv run rl_autorun.py --submit ae <tag>          # stage + build + upload
    uv run rl_autorun.py --await-eval ae <tag> \
        --since-iso <ts> --timeout <T> > result.json     # block on the Discord result

`--submit` copies the resolved checkpoint to `ae/models/ppo.pt`, force-rebuilds
the AE image, and uploads it; `--await-eval` blocks until the watcher ingests the
matching result into `logs/eval_results.jsonl`. A Discord watcher must be running
for the result to land (see `--require-watcher`).

State (all under `ae_rl/tuning/`, append-only / resumable):
  * `eval_leaderboard.json` — every evaluated checkpoint + the current best.
  * a content hash per checkpoint dedupes identical files across rescans, so a
    restart never re-spends an eval on something already scored.

Usage:
    # Evaluate a fixed set of baselines once (ground-truth ranking):
    uv run ae_rl/eval_selector.py --once \
        --candidates ae_rl/checkpoints/stage1_bc_azbase.pt \
                     ae_rl/checkpoints/stage3_league_best.pt

    # Continuously drain a training run's milestones, promoting the real best:
    uv run ae_rl/eval_selector.py \
        --watch-dir ae_rl/checkpoints/milestones \
        --promote-to ae_rl/checkpoints/eval_best.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TUNING_DIR = HERE / "tuning"
LEADERBOARD_PATH = TUNING_DIR / "eval_leaderboard.json"
EVAL_LOG_PATH = REPO / "logs" / "eval_results.jsonl"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_hash(path: Path) -> str:
    """Content hash (first 12 hex chars of sha1) — dedupes identical checkpoints
    that appear under multiple names (e.g. a milestone and its `_best` copy)."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


@dataclass
class EvalEntry:
    tag: str
    checkpoint: str
    sha: str
    score: float
    speed: float
    errors: int
    submitted_at: str
    finished_at: str
    source: str = ""
    confirmed: bool = False
    note: str = ""


# ── leaderboard persistence ───────────────────────────────────────────────────
def load_leaderboard() -> dict:
    if LEADERBOARD_PATH.exists():
        try:
            return json.loads(LEADERBOARD_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"entries": [], "evaluated_shas": {}, "best": None, "updated_at": None}


def save_leaderboard(board: dict) -> None:
    board["updated_at"] = _utcnow_iso()
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LEADERBOARD_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(board, indent=2), encoding="utf-8")
    tmp.replace(LEADERBOARD_PATH)


# ── watcher preflight ──────────────────────────────────────────────────────────
def watcher_fresh(max_age_s: float) -> bool:
    """True if the Discord watcher looks alive (eval_results.jsonl touched
    recently). Without a live watcher, `--await-eval` hangs until timeout because
    the organiser result never reaches the log file."""
    if not EVAL_LOG_PATH.exists():
        return False
    return (time.time() - EVAL_LOG_PATH.stat().st_mtime) <= max_age_s


def launch_watch_only(logger=print) -> subprocess.Popen:
    """Spawn an INGEST-ONLY Discord watcher (`rl_autorun.py --watch-only`).

    Critical: plain `rl_autorun.py` (no flag) loads queue.toml and AUTO-SUBMITS
    the next queued tag on every eval result (WatcherClient.on_message → _fire).
    That races this selector for the single in-flight eval slot and clobbers
    ae/models/ppo.pt staging mid-build. `--watch-only` uses an empty queue, so it
    only writes eval_results.jsonl and never submits — exactly what we want, since
    the selector is the sole AE submitter."""
    logger("launching ingest-only watcher: rl_autorun.py --watch-only")
    return subprocess.Popen(
        ["uv", "run", "rl_autorun.py", "--watch-only"], cwd=str(REPO))


# ── the eval round-trip ────────────────────────────────────────────────────────
def evaluate_checkpoint(
    ckpt: Path,
    tag: str,
    *,
    stage: str = "3",
    timeout_s: float = 1800.0,
    dry_run: bool = False,
    logger=print,
) -> EvalEntry | None:
    """Submit one checkpoint to the real eval and block for the result.

    Returns an EvalEntry on success, or None on submit failure / timeout. The
    eval is serial; this function does not return until the organiser result
    lands or the timeout elapses.
    """
    ckpt = ckpt.resolve()
    submit_env_note = f"RL_AUTORUN_CHECKPOINT={ckpt} RL_AUTORUN_STAGE={stage}"
    submitted_at = _utcnow_iso()

    if dry_run:
        logger(f"[dry-run] would submit {ckpt.name} as tag={tag} ({submit_env_note})")
        return None

    env = os.environ.copy()
    env["RL_AUTORUN_CHECKPOINT"] = str(ckpt)
    env["RL_AUTORUN_STAGE"] = str(stage)

    logger(f"→ submit {ckpt.name}  tag={tag}")
    submit = subprocess.run(
        ["uv", "run", "rl_autorun.py", "--submit", "ae", tag],
        cwd=str(REPO), env=env,
    )
    if submit.returncode != 0:
        logger(f"  ! submit failed (rc={submit.returncode}) for {tag}")
        return None

    logger(f"  …awaiting eval result for {tag} (timeout {int(timeout_s)}s)")
    await_proc = subprocess.run(
        ["uv", "run", "rl_autorun.py", "--await-eval", "ae", tag,
         "--since-iso", submitted_at, "--timeout", str(int(timeout_s))],
        cwd=str(REPO), capture_output=True, text=True,
    )
    if await_proc.returncode != 0:
        logger(f"  ! await timed out / failed for {tag} (rc={await_proc.returncode})")
        return None

    try:
        result = json.loads(await_proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        logger(f"  ! could not parse await output for {tag}: {await_proc.stdout[-200:]!r}")
        return None

    # The Discord channel is shared with other teams — verify this is our result.
    if result.get("tag") != tag:
        logger(f"  ! tag mismatch: expected {tag}, got {result.get('tag')!r}; ignoring")
        return None

    entry = EvalEntry(
        tag=tag,
        checkpoint=str(ckpt),
        sha=_file_hash(ckpt),
        score=float(result.get("score", 0.0)),
        speed=float(result.get("speed", 0.0)),
        errors=int(result.get("errors", 0)),
        submitted_at=submitted_at,
        finished_at=result.get("timestamp", _utcnow_iso()),
    )
    logger(f"  ✓ {tag}: score={entry.score:.4f} speed={entry.speed:.3f} errors={entry.errors}")
    return entry


# ── candidate discovery ────────────────────────────────────────────────────────
def discover_candidates(watch_dirs: list[Path], explicit: list[Path]) -> list[Path]:
    """All .pt files under the watch dirs plus any explicit paths, newest first."""
    found: list[Path] = []
    for d in watch_dirs:
        if d.is_dir():
            found.extend(sorted(d.glob("**/*.pt")))
    for p in explicit:
        if p.is_file():
            found.append(p)
    # De-dupe by resolved path, keep newest-mtime first.
    uniq = {p.resolve(): p for p in found}
    return sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _tag_for(ckpt: Path, sha: str, prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    stem = ckpt.stem.replace("_", "-")[:24]
    return f"{prefix}-{stem}-{sha}-{ts}"


# ── main selection loop ────────────────────────────────────────────────────────
def run_selector(args) -> int:
    board = load_leaderboard()
    evaluated: dict[str, dict] = board.setdefault("evaluated_shas", {})
    watch_dirs = [Path(d) for d in args.watch_dir]
    explicit = [Path(p) for p in args.candidates]
    promote_to = Path(args.promote_to) if args.promote_to else None

    def log(msg: str) -> None:
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

    best = board.get("best")
    best_score = best["score"] if best else float("-inf")
    log(f"Leaderboard: {len(board['entries'])} entries, "
        f"best={best['score']:.4f} ({best['tag']})" if best else "Leaderboard: empty")

    if args.dry_run:
        candidates = discover_candidates(watch_dirs, explicit)
        pending = [c for c in candidates
                   if args.reeval or _file_hash(c) not in evaluated]
        log(f"[dry-run] {len(candidates)} candidate(s), {len(pending)} pending eval:")
        for c in pending:
            log(f"    {_file_hash(c)}  {c}")
        return 0

    # Optionally bring up our own ingest-only watcher. When we own it we trust
    # it's coming up and skip the (mtime-based) freshness gate, which would false-
    # negative until the first result lands.
    owns_watcher = None
    if args.launch_watcher and not watcher_fresh(args.watcher_max_age):
        owns_watcher = launch_watch_only(log)
        import atexit
        atexit.register(
            lambda: owns_watcher.poll() is None and owns_watcher.terminate())
        time.sleep(8)  # let it log in to Discord
    require_watcher = args.require_watcher and owns_watcher is None

    while True:
        if require_watcher and not watcher_fresh(args.watcher_max_age):
            log(f"! Discord watcher looks stale (no {EVAL_LOG_PATH.name} write in "
                f"{args.watcher_max_age}s). Submissions would hang. Start an INGEST-ONLY "
                f"watcher: `uv run rl_autorun.py --watch-only &` (NOT bare `rl_autorun.py`, "
                f"which auto-submits from queue.toml on every result and would race this "
                f"selector). Or pass --launch-watcher, or --no-require-watcher.")
            if args.once:
                return 2
            time.sleep(args.poll_interval)
            continue

        candidates = discover_candidates(watch_dirs, explicit)
        pending = [c for c in candidates
                   if args.reeval or _file_hash(c) not in evaluated]

        if not pending:
            if args.once:
                log("No pending candidates. Done.")
                return 0
            time.sleep(args.poll_interval)
            continue

        ckpt = pending[0]
        sha = _file_hash(ckpt)
        tag = _tag_for(ckpt, sha, args.tag_prefix)

        entry = evaluate_checkpoint(
            ckpt, tag, stage=args.stage, timeout_s=args.timeout,
            dry_run=args.dry_run, logger=log,
        )
        if entry is None:
            # Record the failure against the sha so we don't spin on a broken
            # candidate; --reeval or deleting the leaderboard clears it.
            evaluated[sha] = {"tag": tag, "score": None, "note": "submit/await failed",
                              "at": _utcnow_iso()}
            save_leaderboard(board)
            if args.once and len(pending) == 1:
                return 1
            continue

        entry.source = str(ckpt.parent)
        board["entries"].append(asdict(entry))
        evaluated[sha] = {"tag": entry.tag, "score": entry.score, "at": entry.finished_at}

        # New real-eval best → optionally confirm with a second eval (cheap
        # insurance against the 30-round average's residual noise), then promote.
        if entry.score > best_score + args.min_delta:
            confirmed = entry
            if args.confirm_best and not args.dry_run:
                log(f"  new leader {entry.score:.4f} > {best_score:.4f}; confirming…")
                tag2 = _tag_for(ckpt, sha, args.tag_prefix + "c")
                second = evaluate_checkpoint(
                    ckpt, tag2, stage=args.stage, timeout_s=args.timeout, logger=log)
                if second is not None:
                    board["entries"].append(asdict(second))
                    # Use the mean of the two as the confirmed score.
                    confirmed = entry
                    confirmed.score = (entry.score + second.score) / 2.0
                    confirmed.confirmed = True
                    confirmed.note = f"mean of {entry.score:.4f},{second.score:.4f}"
            if confirmed.score > best_score + args.min_delta:
                best_score = confirmed.score
                board["best"] = asdict(confirmed)
                log(f"  ★ promoted new best: {confirmed.score:.4f} ({confirmed.tag})")
                if promote_to is not None and not args.dry_run:
                    promote_to.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ckpt, promote_to)
                    log(f"  ★ copied {ckpt.name} → {promote_to}")

        save_leaderboard(board)
        if args.once and not [c for c in discover_candidates(watch_dirs, explicit)
                              if _file_hash(c) not in evaluated]:
            log("All candidates evaluated. Done.")
            return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch-dir", action="append", default=[],
                    help="directory scanned recursively for candidate .pt files "
                         "(repeatable). Point at a training run's milestones dir.")
    ap.add_argument("--candidates", nargs="*", default=[],
                    help="explicit checkpoint paths to evaluate (e.g. baselines).")
    ap.add_argument("--promote-to", type=str, default="",
                    help="copy the current real-eval best here (the deploy candidate).")
    ap.add_argument("--stage", type=str, default="3",
                    help="RL_AUTORUN_STAGE passed through (path overrides it anyway).")
    ap.add_argument("--tag-prefix", type=str, default="sel",
                    help="submission tag prefix.")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="per-eval await timeout in seconds (default 1800).")
    ap.add_argument("--min-delta", type=float, default=0.0,
                    help="real-eval improvement required to crown a new best "
                         "(30-round avg is low-noise, so 0 is fine; raise to ~0.01 "
                         "to be conservative).")
    ap.add_argument("--confirm-best", dest="confirm_best", action="store_true", default=True,
                    help="re-eval a new leader once and average before promoting (default on).")
    ap.add_argument("--no-confirm-best", dest="confirm_best", action="store_false",
                    help="promote on a single eval (spend half the evals; trust the 30-round avg).")
    ap.add_argument("--reeval", action="store_true",
                    help="evaluate even checkpoints already in the leaderboard.")
    ap.add_argument("--once", action="store_true",
                    help="drain the current candidate set and exit (no polling).")
    ap.add_argument("--poll-interval", type=float, default=60.0,
                    help="seconds between rescans when waiting for new candidates.")
    ap.add_argument("--launch-watcher", action="store_true",
                    help="spawn an ingest-only watcher (`rl_autorun.py --watch-only`) for the "
                         "selector's lifetime, so you don't have to run one separately. Does "
                         "NOT auto-submit (unlike bare `rl_autorun.py`).")
    ap.add_argument("--require-watcher", dest="require_watcher", action="store_true", default=True,
                    help="refuse to submit if the Discord watcher looks stale (default on).")
    ap.add_argument("--no-require-watcher", dest="require_watcher", action="store_false",
                    help="submit even if the watcher freshness check fails.")
    ap.add_argument("--watcher-max-age", type=float, default=21600.0,
                    help="max seconds since last eval_results.jsonl write to consider "
                         "the watcher alive (default 6h).")
    ap.add_argument("--dry-run", action="store_true",
                    help="discover + log what would be submitted, without submitting.")
    args = ap.parse_args()

    if not args.watch_dir and not args.candidates:
        ap.error("provide at least one --watch-dir or --candidates path")

    try:
        sys.exit(run_selector(args))
    except KeyboardInterrupt:
        print("\nInterrupted; leaderboard saved.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
