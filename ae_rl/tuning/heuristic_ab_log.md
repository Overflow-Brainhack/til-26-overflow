# Heuristic A/B campaign — azbasev1 levers vs the real eval

Started 2026-06-02 (branch `tuning/auto`, max-effort session). Goal per user:
**not** to maximise the deterministic eval, but to build a heuristic that
generalises across a wide range of opponent types. A heuristic has no
opponent-specific weights, so it cannot seed-overfit the eval the way the RL
does — eval-score deltas between heuristic variants therefore reflect genuine
behavioural differences, which makes the real eval a *legitimate* arbiter here
(unlike for the RL, where it Goodharts).

## Method

- Base policy: `AzbaseV1Policy()` (historical eval ~0.66–0.72). Best heuristic.
- Each variant flips **exactly one** lever vs baseline → eval delta is attributable.
- Only attack/dodge-side levers (azbasev1 overrides `_try_collect`, so collect-side
  toggles are inert).
- Deploy mechanism: `ae/src/_campaign_variant.py::VARIANT` selects the policy;
  `ae_manager` calls `make_policy()`. One-line change per submission.
- Eval loop: `RL_AUTORUN_CHECKPOINT=ae/models/ppo.pt uv run rl_autorun.py --submit ae <tag>`
  (checkpoint copy is a same-file no-op; heuristic ignores it), then
  `--await-eval ae <tag>` against the `--watch-only` Discord watcher. ~15–18 min/slot, serial.
- Raw results: `ae_rl/tuning/ab_results/<tag>.json`.

## ⚠ Deploy state / restore

`ae_manager` now deploys the campaign heuristic, **not** the finals RL model.
To restore: set `VARIANT = "rl"` in `_campaign_variant.py` (and re-stage the
0.755 checkpoint, which lives on another device), or revert the `make_policy()`
line in `ae_manager.py` back to `RLPolicy()`.

## Results

Tag scheme: `az-<variant>-<MMDDTHHMM>`. Delta = score − baseline.

**Pivot (per user):** skip the knob sweep; test NEW features instead. Built in
`ae/src/policies/azbasev1_edited_policy.py::AzbaseV1EditedPolicy` (subclass of the
pristine `AzbaseV1Policy`), each behind an OFF-by-default toggle. Fold a feature
into azbasev1 only if its eval delta is positive. Crash-smoke
(`ae_rl/_smoke_edited.py`, all features on, 6-FFA novice) passed: 2400 steps, 0
exceptions. Game facts driving the features: **friendly fire is OFF** (own bombs
never self-damage → aggression is free), destroy_base +50 (5 bombs), kill +30,
both only on the finishing blow → concentration wins.

| variant   | feature (AzbaseV1EditedPolicy toggle)                          | tag | score | Δ vs base | notes |
|-----------|---------------------------------------------------------------|-----|-------|-----------|-------|
| baseline  | — (pristine AzbaseV1Policy)                                    | az-baseline-0602T1448 | **0.612** | 0 (ref) | errors 0, speed 0.829 |
| kills     | hp_aware_kills: credit +30 lethal hits + route to wounded enemy | az-kills-0602T1512 | _in flight_ | | fixes the 30× kill blind spot |
| siege     | base_siege: commit to one base, sustain fire (stop abandoning) | —   | | | secures the +50 finishing bonus |
| endgame   | endgame_dump: last 30 steps → ~0 threshold, no explore         | —   | | | spend leftover bombs late |
| all3      | hp_aware_kills + base_siege + endgame_dump                     | —   | | | fold-in candidate if positive |

## Log
- 2026-06-02: harness built, all 12 variants construct, submit pipeline verified
  (docker + gcloud impersonation token OK). Starting baseline.
