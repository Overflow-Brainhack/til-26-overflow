#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PHASE 1B — FROZEN-CORE EXTEND (continue the 0.788 breakthrough)
#
# phase1_frozen_core.sh broke 0.755 -> 0.788 with hard-PFSP vs the fixed scripted
# core {tactical, azbasev1, azbasev4, berserker}, LR 2e-5, NO self-snapshots.
# Rerunning it longer just replateaus: a frozen pool has finite signal and a flat
# tiny LR settles into one basin. This extends it with genuinely NEW signal instead
# of more steps, while keeping the SAME 4-bot core (no opponent changes):
#   1. warm-start from 0.788 (not 0.755);
#   2. light anti-plateau exploration (entropy floor + bursts) the original lacked;
#   3. mild PFSP floor bump so the longer run can't over-focus a single core bot.
# Still frozen-core: no self-snapshots, so the mirror-match corrosion stays gone.
# Selector anchored on best-0788 => only a REAL gain promotes.
#
# SEEDS: defaults to a single seed (1). The eval is DETERMINISTIC but JAGGED and
# only the MAX checkpoint counts (HANDOFF.md), so set SEEDS="1 2 3" to fan out K
# identical frozen-core runs that wander DIFFERENT paths across the jagged surface
# from the same 0.788 start — each is an independent draw on the upper envelope,
# all feeding ONE shared selector that keeps the global max. No self-play is added,
# so this buys more draws WITHOUT reintroducing mirror-match corrosion.
#   Eval budget = ONE in flight (~13 min). Total evals = (#SEEDS)x(UPDATES/MILESTONE_EVERY).
#   Multi-seed only trustworthy if the eval REUSES seeds (resolve via the 1-slot
#   resubmit test); if it resamples, more draws = more eval-seed overfit.
#
# Cross-device: TRAIN_ONLY=1 on the GPU box, rsync "$OUT"/seed_*/milestones to the
# Workbench, SELECT_ONLY=1 there (pass the SAME SEEDS so the watch-dirs match).
# Concurrency: K parallel trainers each spawn ~(cpus-1) workers -> oversubscribes
# the CPU. Set NUM_WORKERS≈cpus/K, or launch fewer seeds at a time.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.." # repo root

INIT_CKPT="${INIT_CKPT:-ae_rl/checkpoints/best-0788.pt}"   # was hard-nomult-0755.pt
ANCHOR="${ANCHOR:-ae_rl/checkpoints/best-0788.pt}"         # selector bar to beat

RUN="phase1b_frozen_core_extend"
OUT="ae_rl/checkpoints/$RUN"
mkdir -p "$OUT" "ae_rl/runs/$RUN" logs

# --- drift-control + anti-plateau hyperparameters (override via env) ---
SEEDS="${SEEDS:-1}"                         # single seed for now; "1 2 3" to fan out the lottery
UPDATES="${UPDATES:-300}"                  # was 200: give exploration room to find a new basin
LR="${LR:-2e-5}"                           # unchanged: stay near the 0.788 peak
MILESTONE_EVERY="${MILESTONE_EVERY:-10}"   # unchanged: dense harvest
PFSP_EVERY="${PFSP_EVERY:-25}"
PFSP_FLOOR="${PFSP_FLOOR:-0.05}"           # 0788 used 0.03; mild bump so the longer run can't
                                           # collapse onto one of the 4. Set 0.03 to match exactly.
ENTROPY_FLOOR="${ENTROPY_FLOOR:-0.008}"    # NEW: stop collapse into the deterministic plateau
BURST_EVERY="${BURST_EVERY:-40}"           # NEW: periodic exploration kicks
BURST_LEN="${BURST_LEN:-3}"
BURST_COEF="${BURST_COEF:-0.04}"           # modest — don't blow up a 0.788 policy
SUBMIT_COOLDOWN="${SUBMIT_COOLDOWN:-300}"  # extra seconds between eval submissions so the
                                           # organiser server isn't flooded (0 = back-to-back)
NUM_WORKERS="${NUM_WORKERS:-}"             # blank = train default (cpus-1); see concurrency note

RUN_TRAINER="${RUN_TRAINER:-1}"; RUN_SELECTOR="${RUN_SELECTOR:-1}"
[[ "${TRAIN_ONLY:-0}" == "1" ]] && RUN_SELECTOR=0
[[ "${SELECT_ONLY:-0}" == "1" ]] && RUN_TRAINER=0

if [[ "$RUN_TRAINER" == "1" && ! -f "$INIT_CKPT" ]]; then
  echo "!! init checkpoint not found: $INIT_CKPT (set INIT_CKPT=)" >&2; exit 1
fi

# One shared selector watches EVERY seed's milestones dir.
WATCH_ARGS=()
for S in $SEEDS; do
  WATCH_ARGS+=(--watch-dir "$OUT/seed_$S/milestones")
done

echo "[p1b] init=$INIT_CKPT seeds='$SEEDS' updates=$UPDATES lr=$LR floor=$PFSP_FLOOR ent_floor=$ENTROPY_FLOOR"
echo "[p1b] trainer=$RUN_TRAINER selector=$RUN_SELECTOR  out=$OUT"

# --- 1) trainers (one per seed; background, survive terminal hangup) ---
if [[ "$RUN_TRAINER" == "1" ]]; then
  for S in $SEEDS; do
    SOUT="$OUT/seed_$S"
    mkdir -p "$SOUT" "ae_rl/runs/$RUN/seed_$S"
    TRAIN_ARGS=(
      --ckpt "$INIT_CKPT"
      --updates "$UPDATES" --lr "$LR"
      --pfsp --pfsp-mode hard --pfsp-every "$PFSP_EVERY" --pfsp-floor "$PFSP_FLOOR"
      --frozen-core
      --no-offensive-multipliers
      # Pin the pool to the strong scripted core: zero the weak filler bots so the
      # PFSP candidate set stays exactly {tactical, azbasev1, azbasev4, berserker}.
      --pure-collector-prob 0 --random-prob 0 --idle-prob 0
      --trap-setter-prob 0 --patroller-prob 0 --kamikaze-prob 0
      # anti-plateau (NEW vs phase1)
      --entropy-floor "$ENTROPY_FLOOR"
      --explore-burst-every "$BURST_EVERY" --explore-burst-len "$BURST_LEN" --explore-burst-coef "$BURST_COEF"
      --milestone-every "$MILESTONE_EVERY" --seed "$S"
      --output-ckpt "$SOUT/stage3.pt" --output-best "$SOUT/best.pt"
      --league-dir "$SOUT/league" --summary-json "ae_rl/runs/$RUN/seed_$S/latest.json"
    )
    [[ -n "$NUM_WORKERS" ]] && TRAIN_ARGS+=(-j "$NUM_WORKERS")

    setsid uv run python ae_rl/train_stage3_league.py "${TRAIN_ARGS[@]}" \
      >"logs/${RUN}_seed${S}_train.log" 2>&1 &
    echo "[p1b] seed $S trainer pid=$!  log=logs/${RUN}_seed${S}_train.log"
  done
fi

# --- 2) ONE shared eval selector over all seeds' milestones (foreground) ---
# --candidates anchors the leaderboard on 0.788; the selector drains milestones and
# promotes only what BEATS it. Deterministic eval => --no-confirm-best + --min-delta 0.
# promote-to is RUN-SCOPED so it never clobbers a global best mid-experiment.
if [[ "$RUN_SELECTOR" == "1" ]]; then
  uv run ae_rl/eval_selector.py \
    --candidates "$ANCHOR" \
    "${WATCH_ARGS[@]}" \
    --promote-to "$OUT/eval_best.pt" \
    --stage 3 \
    --tag-prefix p1b-fcx \
    --timeout 1800 \
    --min-delta 0.0 \
    --no-confirm-best \
    --submit-cooldown "$SUBMIT_COOLDOWN" \
    --launch-watcher \
    2>&1 | tee "logs/${RUN}_selector.log"
else
  echo "[p1b] selector skipped (TRAIN_ONLY). rsync '$OUT'/seed_*/milestones to the Workbench,"
  echo "      then run there:  SELECT_ONLY=1 SEEDS='$SEEDS' ae_rl/phase1b_frozen_core_extend.sh"
fi
