# Diagnosis: hard-nomult 0.755 @ update 200, then apparent decline

_Date: 2026-06-02._

> **UPDATE 2026-06-02 — CONFIRMED.** Eval is **deterministic** (0.755 reproduced
> twice on the same ckpt → zero noise). Scenario **A is confirmed**: the regression
> is real, not interleaving. hard-nomult 0.755 → never crossed 0.70 after; even-nomult
> 0.667 (@200) → 0.531 (@1400). **Both PFSP modes corrode with more training**, so the
> lever is drift control, not even-vs-hard. The 0.755 hard-nomult update_000200 ckpt is
> the best real artifact and beats heuristic — **bank it as the finals fallback now.**
> Continuation built: `ae_rl/continue_even_nomult.sh` (warm-start from 0.755, even-mode,
> low LR, short horizon, dense milestones, `--no-confirm-best`, anchor-on-0.755).
> Confound found: even-nomult also carried `--own-base-penalty -5` (hard-nomult did not),
> so even's flat result is partly the base-defense penalty, not the mode.

## What was observed
- A `hard-nomult` variant (PFSP **hard** mode + `--no-offensive-multipliers`,
  warm-started from the league best ~0.65) scored **0.755 at update 200** — beating the
  heuristic family (0.6–0.72) — on a **single** eval.
- All later eval submissions scored **< 0.70**.
- Training stopped at ~**2600 / 6000** updates (server down).
- `easy/even-nomult` was **never scored** while tracking submissions.

## The confound — can't yet tell a real decline from a population artifact
The selector is **serial + newest-first across 4 variants**, one eval / ~15 min, so the
eval stream **interleaves variants**. "All the following updates scored < 0.700" may be
*other recipes* (even-raw, even-allon), not hard-nomult's own later checkpoints. That
`easy-nomult` was never scored is the **starvation** signature, not evidence any variant
is bad.

Two scenarios, not yet distinguished:
- **A — hard-nomult's own curve declined** (its 200-update ckpt = 0.755, its ~1000/2000
  ckpts < 0.70). A real regression.
- **B — hard-nomult was evaled ~once** (0.755 @ 200) and the sub-0.70 evals are the
  *other* variants. No decline at all — just a strong lead never followed up.

Given easy-nomult never scored, **B is at least as likely as A.**

## Two cheap tests gate everything (zero retraining) — DO FIRST when server is back
1. **Read the per-variant curve.** Filter `ae_rl/tuning/eval_leaderboard.json` /
   `ae_rl/tuning/eval_selector_log.jsonl` to the hard-nomult tag prefix and look at *its
   own* score-vs-update. Decides A vs B by itself.
2. **Re-eval the saved 200-update checkpoint 2–3×.** 30-round avg is low-noise if per-eval
   σ≈0.02, so 0.755 over a sub-0.70 cluster is *probably* real — but one confirm eval is
   cheap winner's-curse insurance. Also check whether `--confirm-best` already logged a 2nd
   eval for that tag before the server died (may already be on disk).

If it re-confirms (~0.73+): **bank that checkpoint as the finals candidate immediately.**
It beats heuristic — a win in hand regardless of what training does next.

## Diagnosis IF the decline is real (scenario A)
Same target-misalignment as the original plateau, now visible as regression because the run
warm-started high. Likely chain:
1. `--no-offensive-multipliers` removed a reward distortion → genuine fast gain to ~0.75 in
   the first ~200 updates (still near the warm-start).
2. `--pfsp hard` then increasingly weighted the opponents *you lose to* — our **own**
   scripted/league bots, which are **not** the organiser eval agents — so the policy
   specialized against a non-representative adversary and corroded general competence.
3. League-snapshot pollution (feeding snapshots of the now-declining policy back into the
   pool) compounds it.

Net: more updates = optimizing the wrong target harder = eval falls. The peak being *early*
is the tell.

## The three options, ranked
- **Retrain hard-nomult from scratch — NO.** Throws away the warm-start that *produced* the
  0.755 and keeps the suspect objective. Worst option.
- **Finetune — YES, but as "continue *that* variant from its 0.755 ckpt," not more
  hard-from-init.** Switch the continuation to **even-mode** (the "easy" one), keep
  **nomult**, drop LR (or add decay), keep exploration modest (refining a good policy, not
  escaping a plateau), short horizon, eval every ~25–50 updates.
- **Try easy-nomult — YES, regardless of A/B.** Safer objective that should *hold* the
  nomult gain rather than corrode it, and there's **zero** data on it (starved). Run it
  isolated.

## Process fix (the real lesson)
Stop running 4 variants against one 1-per-15-min eval — that breadth is why easy-nomult
never scored and why A vs B is currently indistinguishable. For the finals push, run
**1–2 variants at a time**, warm-started from the 0.755 ckpt, so each gets a dense, readable
per-variant eval curve and the real eval can pick between even and hard.

## Suggested next run (once 0.755 is confirmed real)
Isolated **even-nomult continuation**, warm-started from the 0.755 ckpt: even-mode PFSP +
`--no-offensive-multipliers`, lower LR (or decay), short horizon, dense eval (every ~25–50
updates). Run alongside (or before) the same recipe warm-started from the original 0.65
league best as the clean "easy-nomult you never saw" comparison. _(Exact CLI flags: confirm
against `train_stage3_league.py` / `eval_selector.py` signatures before launching.)_
