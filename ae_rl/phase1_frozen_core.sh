#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PHASE 1 — FROZEN-CORE PROBE (the one mechanistically-motivated shot past 0.755)
#
# Hypothesis (HANDOFF.md): the 0.755 plateau DECAYS under continued self-play
# because, once the scripted bots are beaten, hard-mode PFSP's "hardest opponent"
# becomes your own recent league snapshots -> you specialise to non-transitive
# mirror-match exploits that DON'T transfer to the organiser agents (Goodhart).
#
# The probe removes that ladder: pin the opponent pool to the STRONG SCRIPTED
# CORE (tactical / azbasev1 / azbasev4 / berserker) with NO self-snapshots
# (--frozen-core), then run hard-mode PFSP (the only recipe that ever climbed) at
# a TINY LR from the 0.755 checkpoint, harvesting dense milestones with the real-
# eval selector.
#   - hard-mode keeps gaining vs the fixed strong target  -> the gain is REAL
#     (transferable; promote the new max).
#   - it plateaus / decays here too                        -> 0.755 IS the ceiling.
#
# Cross-device: run the TRAINER on your GPU box (TRAIN_ONLY=1), rsync
# "$OUT/milestones" back to the Workbench, run the SELECTOR there (SELECT_ONLY=1).
# Default (no flag) runs both on one box, like continue_even_nomult.sh.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.." # repo root

# --- the 0.755 checkpoint (the finals model) ---
INIT_CKPT="${INIT_CKPT:-ae_rl/checkpoints/hard-nomult-0755.pt}"

RUN="phase1_frozen_core"
OUT="ae_rl/checkpoints/$RUN"
mkdir -p "$OUT" "ae_rl/runs/$RUN" logs

# --- drift-control hyperparameters (override via env) ---
UPDATES="${UPDATES:-200}"                  # short horizon: harvest the gaining window
LR="${LR:-2e-5}"                           # tiny: stay near the 0.755 peak (HANDOFF Phase 1)
MILESTONE_EVERY="${MILESTONE_EVERY:-10}"   # dense: one eval candidate every 10 updates
PFSP_EVERY="${PFSP_EVERY:-25}"             # win-rate refresh / pool rebuild cadence
SEED="${SEED:-1}"
NUM_WORKERS="${NUM_WORKERS:-}"             # blank = train default (cpus-1)

# Which halves to run (cross-device split). TRAIN_ONLY / SELECT_ONLY override.
RUN_TRAINER="${RUN_TRAINER:-1}"
RUN_SELECTOR="${RUN_SELECTOR:-1}"
[[ "${TRAIN_ONLY:-0}" == "1" ]] && RUN_SELECTOR=0
[[ "${SELECT_ONLY:-0}" == "1" ]] && RUN_TRAINER=0

if [[ "$RUN_TRAINER" == "1" && ! -f "$INIT_CKPT" ]]; then
  echo "!! init checkpoint not found: $INIT_CKPT" >&2
  echo "   set INIT_CKPT=<path to the 0.755 checkpoint> and re-run." >&2
  exit 1
fi
echo "[phase1] init=$INIT_CKPT updates=$UPDATES lr=$LR milestone-every=$MILESTONE_EVERY seed=$SEED"
echo "[phase1] trainer=$RUN_TRAINER selector=$RUN_SELECTOR  out=$OUT"

# --- 1) trainer (background; survives terminal hangup) ---
if [[ "$RUN_TRAINER" == "1" ]]; then
  TRAIN_ARGS=(
    --ckpt "$INIT_CKPT"
    --updates "$UPDATES"
    --lr "$LR"
    --pfsp --pfsp-mode hard
    --pfsp-every "$PFSP_EVERY"
    --frozen-core
    --no-offensive-multipliers
    # Pin the pool to the strong scripted core: zero the weak filler bots so the
    # PFSP candidate set is exactly {tactical, azbasev1, azbasev4, berserker}.
    --pure-collector-prob 0 --random-prob 0 --idle-prob 0
    --trap-setter-prob 0 --patroller-prob 0 --kamikaze-prob 0
    --milestone-every "$MILESTONE_EVERY"
    --seed "$SEED"
    --output-ckpt "$OUT/stage3.pt"
    --output-best "$OUT/best.pt"
    --league-dir "$OUT/league"
    --summary-json "ae_rl/runs/$RUN/latest.json"
  )
  [[ -n "$NUM_WORKERS" ]] && TRAIN_ARGS+=(-j "$NUM_WORKERS")

  setsid uv run python ae_rl/train_stage3_league.py "${TRAIN_ARGS[@]}" \
    >"logs/${RUN}_train.log" 2>&1 &
  TRAIN_PID=$!
  echo "[phase1] trainer pid=$TRAIN_PID  log=logs/${RUN}_train.log"
fi

# --- 2) eval selector (foreground; one in-flight eval at a time) ---
# --candidates anchors the leaderboard on the 0.755 baseline, then the selector
# drains milestones and promotes only what BEATS it. Deterministic eval =>
# --no-confirm-best + --min-delta 0.0. --launch-watcher spawns the ingest-only
# Discord watcher itself.
if [[ "$RUN_SELECTOR" == "1" ]]; then
  uv run ae_rl/eval_selector.py \
    --candidates "$INIT_CKPT" \
    --watch-dir "$OUT/milestones" \
    --promote-to ae_rl/checkpoints/eval_best.pt \
    --tag-prefix p1-frozen \
    --timeout 1800 \
    --min-delta 0.0 \
    --no-confirm-best \
    --launch-watcher \
    2>&1 | tee "logs/${RUN}_selector.log"
else
  echo "[phase1] selector skipped (TRAIN_ONLY). Sync '$OUT/milestones' to the"
  echo "         Workbench, then run there:  SELECT_ONLY=1 ae_rl/phase1_frozen_core.sh"
fi
