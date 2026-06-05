#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sp0788 EVAL SELECTOR — real-eval fitness for the self-play-from-0788 run.
#
# Runs "on the side" of the sp0788 trainer (train_stage3_league.py with
# --output-ckpt ae_rl/checkpoints/sp0788/run.pt). Watches that run's milestone
# dir, submits each NEW milestone — both the "best" and the "latest" copies, de-
# duped by content sha — to the organiser eval ONE AT A TIME, records them in
# tuning/eval_leaderboard.json + tuning/eval_selector_log.jsonl, and promotes the
# real-eval best to ae_rl/checkpoints/sp0788/eval_best.pt. Same machinery the
# frozen-core probe used (phase1_frozen_core.sh).
#
# The leaderboard is ANCHORED on best-0788 (--candidates), so a milestone is only
# promoted if it BEATS the current champion on the real eval. The new eval is
# deterministic => --no-confirm-best (don't waste a slot reconfirming) and
# --min-delta 0.0.
#
# OUTWARD-FACING: submits our model to the organiser server and (unless a watcher
# is already up) launches the ingest-only Discord watcher. Needs Docker +
# competition creds. Run it yourself; nothing here auto-launches.
#
# Eval is serial (~13 min/slot) so this trickles through milestones over hours —
# that's expected. It doubles as ground truth to validate robustness_battery.py
# against the new eval beyond our n=2 calibration pair.
#
# Env overrides: ANCHOR= WATCH_DIR= PROMOTE_TO= TAG_PREFIX= TIMEOUT= FRESH=1
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.." # repo root

ANCHOR="${ANCHOR:-ae_rl/checkpoints/best-0788.pt}"            # the bar to beat
WATCH_DIR="${WATCH_DIR:-ae_rl/checkpoints/sp0788/milestones}" # must match the trainer's --output-ckpt parent
PROMOTE_TO="${PROMOTE_TO:-ae_rl/checkpoints/sp0788/eval_best.pt}"
TAG_PREFIX="${TAG_PREFIX:-sp0788}"
TIMEOUT="${TIMEOUT:-1800}"
mkdir -p logs "$WATCH_DIR" "$(dirname "$PROMOTE_TO")"

if [[ ! -f "$ANCHOR" ]]; then
  echo "!! anchor checkpoint not found: $ANCHOR" >&2
  exit 1
fi

# Optional clean slate for a fresh ranking. Never touch logs/eval_results.jsonl
# (append-only; the selector's --await-eval source).
if [[ "${FRESH:-0}" == "1" ]]; then
  rm -f tuning/eval_leaderboard.json
  echo "[sp0788-sel] cleared tuning/eval_leaderboard.json (FRESH=1)"
fi

# Two watchers on one selfbot token conflict (HANDOFF). Only --launch-watcher if
# none is already running; otherwise reuse the existing one.
WATCHER_ARG=(--launch-watcher)
if pgrep -af "rl_autorun.py --watch-only" >/dev/null 2>&1; then
  echo "[sp0788-sel] existing --watch-only watcher detected; reusing it (not launching a second)."
  WATCHER_ARG=()
fi

echo "[sp0788-sel] anchor=$ANCHOR  watch=$WATCH_DIR  promote=$PROMOTE_TO  prefix=$TAG_PREFIX"
uv run ae_rl/eval_selector.py \
  --candidates "$ANCHOR" \
  --watch-dir "$WATCH_DIR" \
  --promote-to "$PROMOTE_TO" \
  --stage 3 \
  --tag-prefix "$TAG_PREFIX" \
  --timeout "$TIMEOUT" \
  --min-delta 0.0 \
  --no-confirm-best \
  "${WATCHER_ARG[@]}" \
  2>&1 | tee "logs/sp0788_selector.log"
