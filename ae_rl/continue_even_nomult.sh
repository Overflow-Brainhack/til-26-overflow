#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Isolated even-nomult CONTINUATION from the hard-nomult update_000200 checkpoint
# that scored 0.755 (real, DETERMINISTIC eval — beats the heuristic family).
#
# Why this shape: BOTH prior runs corroded with more training (hard-nomult
# 0.755 -> <0.70, even-nomult 0.667 -> 0.531). The lever is NOT even-vs-hard;
# it's DRIFT CONTROL. So we start AT the 0.755 policy and:
#   - low LR (small local steps, don't sprint off the peak)
#   - short horizon (harvest the early window, not 6000 updates)
#   - dense milestones (a candidate every 25 updates for the real-eval selector)
#   - anchor the leaderboard on 0.755 and promote ONLY a milestone that BEATS it
#
# Recipe note: this uses even-mode + nomult + own_base=0 — i.e. the EXACT 0.755
# recipe with only the PFSP mode swapped hard->even. That isolates the mode and
# matches the intentional own_base=0. To reproduce the LITERAL even-nomult
# (which also had -5 base defense, a confound), add: --own-base-penalty -5
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.." # repo root

# --- the 0.755 checkpoint (VERIFY this path/filename on the box) ---
INIT_CKPT="${INIT_CKPT:-ae_rl/checkpoints/hard-nomult-0755.pt}"

RUN="cont_even_nomult"
OUT="ae_rl/checkpoints/$RUN"
mkdir -p "$OUT" "ae_rl/runs/$RUN" logs

# --- drift-control hyperparameters (override via env) ---
UPDATES="${UPDATES:-500}"                # short: harvest the peak window
LR="${LR:-1e-4}"                         # half the 2e-4 default (5e-5 = more conservative)
MILESTONE_EVERY="${MILESTONE_EVERY:-25}" # one eval candidate every 25 updates
SEED="${SEED:-1}"

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "!! init checkpoint not found: $INIT_CKPT" >&2
  echo "   set INIT_CKPT=<path to the 0.755 hard-nomult milestone> and re-run." >&2
  exit 1
fi
echo "[continue] init=$INIT_CKPT updates=$UPDATES lr=$LR milestone-every=$MILESTONE_EVERY seed=$SEED"

# --- 1) trainer (background; survives terminal hangup) ---
setsid uv run python ae_rl/train_stage3_league.py \
  --ckpt "$INIT_CKPT" \
  --updates "$UPDATES" \
  --lr "$LR" \
  --pfsp --pfsp-mode even \
  --no-offensive-multipliers \
  --milestone-every "$MILESTONE_EVERY" \
  --snapshot-every 25 \
  --league-max-size 0 \
  --seed "$SEED" \
  --output-ckpt "$OUT/stage3.pt" \
  --output-best "$OUT/best.pt" \
  --league-dir "$OUT/league" \
  --summary-json "ae_rl/runs/$RUN/latest.json" \
  >"logs/${RUN}_train.log" 2>&1 &
TRAIN_PID=$!
echo "[continue] trainer pid=$TRAIN_PID  log=logs/${RUN}_train.log"

# --- 2) eval selector (foreground; the one in-flight eval at a time) ---
# --candidates anchors the leaderboard on the 0.755 baseline first, then the
# selector drains milestones and promotes only what beats it. Deterministic eval
# => --no-confirm-best (no point re-evaling) + --min-delta 0.0 (any beat is real).
# --launch-watcher spawns the ingest-only Discord watcher itself.
uv run ae_rl/eval_selector.py \
  --candidates "$INIT_CKPT" \
  --watch-dir "$OUT/milestones" \
  --promote-to ae_rl/checkpoints/eval_best.pt \
  --tag-prefix cont-even \
  --timeout 1800 \
  --min-delta 0.0 \
  --no-confirm-best \
  --launch-watcher \
  2>&1 | tee "logs/${RUN}_selector.log"
