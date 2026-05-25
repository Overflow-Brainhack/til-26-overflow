#!/usr/bin/env python3
"""hourly_eval.py — Submit selfplay best checkpoint, await eval result, log to CSV."""

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SELFPLAY_BEST = REPO_ROOT / "ae_rl" / "checkpoints" / "selfplay" / "stage3_selfplay_best.pt"
LATEST_JSON = REPO_ROOT / "ae_rl" / "runs" / "stage3_selfplay" / "latest.json"
EVAL_LOG_CSV = REPO_ROOT / "ae_rl" / "tuning" / "eval_log.csv"
LOGS_DIR = REPO_ROOT / "logs"

CSV_COLUMNS = [
    "timestamp", "tag", "eval_score", "eval_speed", "eval_errors",
    "training_updates", "best_val_score", "latest_rl_mean", "latest_heur_baseline",
]


def read_training_stats() -> dict:
    if not LATEST_JSON.exists():
        return {}
    with open(LATEST_JSON) as f:
        data = json.load(f)
    validations = data.get("validations", [])
    result: dict = {"training_updates": data.get("updates_completed", "")}
    if validations:
        best_val = max(v["score"] for v in validations)
        latest = validations[-1]
        result["best_val_score"] = round(best_val, 2)
        result["latest_rl_mean"] = round(latest.get("rl_mean", 0), 2)
        result["latest_heur_baseline"] = round(latest.get("heur_baseline", 0), 2)
    return result


def append_csv(row: dict) -> None:
    EVAL_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not EVAL_LOG_CSV.exists()
    with open(EVAL_LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for col in CSV_COLUMNS:
            row.setdefault(col, "")
        writer.writerow({col: row[col] for col in CSV_COLUMNS})


def last_eval_age_seconds() -> float | None:
    """Return seconds since the most recent eval_log.csv entry, or None if no log."""
    if not EVAL_LOG_CSV.exists():
        return None
    with open(EVAL_LOG_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    ts_raw = rows[-1].get("timestamp", "")
    try:
        ts = datetime.fromisoformat(ts_raw)
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except ValueError:
        return None


def main() -> None:
    # Skip if an eval was logged within the last 45 minutes (dedup for overlapping wakeups)
    age = last_eval_age_seconds()
    if age is not None and age < 2700:
        print(f"[hourly_eval] skipping — last eval was {age:.0f}s ago (<45min)", flush=True)
        sys.exit(0)

    if not SELFPLAY_BEST.exists():
        print(f"[hourly_eval] ERROR: selfplay best not found: {SELFPLAY_BEST}", flush=True)
        sys.exit(1)

    stats = read_training_stats()
    now = datetime.now(timezone.utc)
    tag = now.strftime("selfplay-%m%dT%H%M")
    uv = shutil.which("uv") or "uv"

    # RL_AUTORUN_CHECKPOINT tells --submit which checkpoint to stage into ae/models/ppo.pt
    env = {**os.environ, "RL_AUTORUN_CHECKPOINT": str(SELFPLAY_BEST)}

    print(f"[hourly_eval] tag={tag}  updates={stats.get('training_updates', '?')}", flush=True)

    # Start watch-only Discord watcher in background (captures eval results, never submits)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    watcher_log_path = LOGS_DIR / f"watcher_{tag}.log"
    watcher_log = open(watcher_log_path, "w")
    watcher_proc = subprocess.Popen(
        [uv, "run", "python", str(REPO_ROOT / "rl_autorun.py"), "--watch-only"],
        env=env, cwd=str(REPO_ROOT),
        stdout=watcher_log, stderr=watcher_log,
    )
    print(f"[hourly_eval] watcher pid={watcher_proc.pid}  log={watcher_log_path}", flush=True)
    time.sleep(8)  # let Discord connection establish

    # One-shot submit: build Docker image, push, upload model
    submitted_at = datetime.now(timezone.utc).isoformat()
    print(f"[hourly_eval] submitting ae:{tag} at {submitted_at}", flush=True)
    submit_rc = subprocess.run(
        [uv, "run", "python", str(REPO_ROOT / "rl_autorun.py"), "--submit", "ae", tag],
        env=env, cwd=str(REPO_ROOT),
    ).returncode

    if submit_rc != 0:
        print(f"[hourly_eval] ERROR: --submit failed (rc={submit_rc})", flush=True)
        watcher_proc.kill()
        watcher_log.close()
        sys.exit(1)

    print("[hourly_eval] submit complete — awaiting eval (timeout=1800s)", flush=True)

    # Block until eval result appears in logs/eval_results.jsonl (written by watcher)
    await_proc = subprocess.run(
        [
            uv, "run", "python", str(REPO_ROOT / "rl_autorun.py"),
            "--await-eval", "ae", tag,
            "--since-iso", submitted_at,
            "--timeout", "1800",
        ],
        env=env, cwd=str(REPO_ROOT),
        capture_output=True, text=True,
    )

    # Kill watcher — done regardless of eval outcome
    print(f"[hourly_eval] killing watcher pid={watcher_proc.pid}", flush=True)
    try:
        watcher_proc.kill()
        watcher_proc.wait(timeout=5)
    except Exception as exc:
        print(f"[hourly_eval] warning: {exc}", flush=True)
    watcher_log.close()

    if await_proc.returncode != 0:
        print("[hourly_eval] ERROR: await-eval timed out or failed", flush=True)
        print(await_proc.stderr, flush=True)
        sys.exit(1)

    try:
        eval_data = json.loads(await_proc.stdout.strip())
    except json.JSONDecodeError:
        print(f"[hourly_eval] ERROR: unparseable eval output: {await_proc.stdout!r}", flush=True)
        sys.exit(1)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "eval_score": eval_data.get("score", ""),
        "eval_speed": eval_data.get("speed", ""),
        "eval_errors": eval_data.get("errors", ""),
        **stats,
    }
    append_csv(row)
    print(
        f"[hourly_eval] done — score={eval_data.get('score')}  speed={eval_data.get('speed')}",
        flush=True,
    )
    print(f"[hourly_eval] logged to {EVAL_LOG_CSV}", flush=True)


if __name__ == "__main__":
    main()
