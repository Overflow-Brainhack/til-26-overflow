# AE Next Session Handover

Last updated: 2026-05-20.

## Session update (2026-05-20, latest): auto_play cleanup, dodge → v2, berserker_base fix

This session was housekeeping + one real bug fix. Read this before trusting the
older sections below — the diagnostic-policy machinery they describe is **gone**.

1. **`diagnostic_policies.py` and `collect_dodge_policy.py` no longer exist** in
   `src/`. `test_env/auto_play.py` was pruned to only reference policies that
   actually ship: `normal` (EditedHeuristicPolicyV2), `berserker`,
   `berserker_base`, and the inline `random`. The `PROFILES`/`make_diagnostic_policy`
   imports, the `collect_dodge` branch, and all diagnostic entries in
   `_TYPE_COLORS` were removed. Anything below referencing `AE_POLICY_VARIANT` /
   diagnostic variants is **stale** — `ae_manager.py` has those env-var lines
   commented out and now hardcodes `EditedHeuristicPolicyV2 as HeuristicPolicy`.

2. **Dodge moved up into `EditedHeuristicPolicyV2`.** `_dodge` (dispatcher),
   `_dodge_v1` (flee to nearest fully-safe cell — identical to the base
   `EditedHeuristicPolicy._dodge`), `_dodge_v2` (also subtracts nearby enemies'
   blast footprints from the safe set), and the `hardened_dodge` toggle
   (default OFF → v1) now live on v2. berserker_base inherits them and its
   duplicate copies were deleted. `normal` behaviour is unchanged (v1 == old
   inherited dodge).

3. **berserker_base all-0.0s bug — FIXED.** It called `self._pessimistic_adjust(...)`
   in `_route_to_base` and `_collect_tiles`, but that method exists nowhere.
   `AEManager.ae` wraps `policy.choose` in a bare `except Exception: return STAY`
   (ae_manager.py:120), so the `AttributeError` was swallowed every tick the bot
   tried to route/collect → it STAYed forever → flat 0.0 across all 6 agents.
   Removed both calls (they now just return `_maybe_wall_break(...)` like the v2
   collector). Headless benchmark after the fix: berserker_base ≈160 mean vs
   normal ≈182 (was 0.0). **Caveat:** that catch-all in `ae` will silently mask
   any future policy crash as a STAY — when a policy mysteriously scores 0, check
   for an exception in `choose` first.

   To run the benchmark locally (top-level imports need pygame + til_environment,
   so use the repo venv, not bare `python`):
   ```bash
   cd ae/test_env && SDL_VIDEODRIVER=dummy \
     ../../.venv/bin/python auto_play.py --benchmark \
     --benchmark-types berserker_base normal --rounds 2 --advanced
   ```

## Session update (2026-05-20): policy cleanup + v2 clone + berserker_base

Reward math that drives all of this (from `til-26-ae` dynamics reward hooks —
do NOT run/drive the sim to re-verify, it is trusted):

- Collect mission/resource/recon = +5 / +2 / +1.
- Deal bomb damage = +1 per HP; kill = +30 (split); destroy base = +50 (split).
- **Take bomb damage = -1 per HP** (one 20-dmg blast = -20 ≈ 4 missions).
- **Your base taking damage = -1 per HP to you** (-100 max, +-50 on destroy).
- **Friendly fire is OFF** — your own bomb never harms you. Placing bombs is
  near-free (cost = 1.5 resource + 1 tick); the only real risk while attacking
  is *enemy* bombs hitting you. This is why avoiding damage ≈ dealing it, and
  why the hidden eval punishes contact.

Changes made this session:

1. **Folded `edited_policy_conservative.py` into `edited_policy.py`** as the new
   defaults (`minimum_aggression=1.0`, `aggression_ramp_rate=0.04`,
   `defensive_force=0.75`, `defense_cooldown_scale=0.6`,
   `defense_abandon_margin=2`, `max_defense_distance=9`) and **deleted** the
   conservative file. `_try_defend` stays commented out in `choose()` (base
   defense rarely succeeds; we maximise score instead).

2. **`src/edited_policy_v2.py`** — `EditedHeuristicPolicyV2(EditedHeuristicPolicy)`,
   a subclass clone. Dodge stays first priority. (UPDATED in latest session — see
   top. Firing-cell base routing is now in v2's `_try_collect` directly, not a
   toggle; the `base_firing_cell_routing`/`pessimistic_filter` toggles described
   in earlier drafts were never landed.) Actual constructor toggles today:
   - `contested_route_penalty=False` (+ `contested_radius`, `contested_min_factor`)
     — discount collectible tiles near enemies; bases exempt (idea #4).
   - Productive wall-wait is always on in `EditedHeuristicPolicy`; tune
     `_PRODUCTIVE_WALL_WAIT_RADIUS` in `src/edited_policy.py` directly.
   - `hardened_dodge=False` — selects `_dodge_v2` over `_dodge_v1` (added latest
     session).

3. **`src/berserker_base_policy.py`** — `BerserkerBasePolicy(EditedHeuristicPolicyV2)`.
   Inherits the v2 safety stack; overrides only the objective router:
   - bombs > 0 + reachable live base → route to nearest firing cell, weighted
     toward the **weakest** base (`target_weakest_base=True`); inherited
     `_try_attack` bombs on arrival.
   - out of bombs / no base → collect, **resource-biased** when bombs are low
     (`resource_refill_bias=2.0`, `bombs_low_threshold=1`) to refill faster.
   - Dodge (`_dodge_v1`/`_dodge_v2`, `hardened_dodge` toggle) is now inherited
     from v2 (moved there in the latest session); berserker_base no longer
     defines its own.

**Selecting a policy** (ae_manager.py import lines, `as HeuristicPolicy`):
- v2 clone is the active import; comment it to fall back to plain edited_policy.
- Uncomment the `berserker_base_policy` import to make the berserker active.

**Testing**: `auto_play.py` gained a `berserker_base` agent type. Run e.g.
`PYTHONPATH=ae/src:til-26-ae uv run python ae/test_env/auto_play.py --benchmark
--rounds 1 --novice --benchmark-types berserker_base normal --no-cache`.
Caveat: the fixed-novice-seed benchmark is **non-discriminative** for these
changes (v2 already rushes the same base). A synthetic check confirmed v2 vs
berserker diverge correctly (berserker ignores tiles when armed, collects
resources when out of bombs). Real attribution = hidden-eval submission.

## Current State

The AE policy has been refactored so production defaults to `HeuristicPolicy`,
with optional score-only diagnostic variants selected by Docker build arg:

```bash
docker build --platform linux/amd64 \
  --build-arg AE_POLICY_VARIANT=stealth_rotate \
  -t "$TEAM_ID-ae:stealth_rotate" \
  ae

./submit.sh ae stealth_rotate
```

`AE_POLICY_VARIANT=normal` uses the merged conservative heuristic. Any other
variant is constructed by `src/diagnostic_policies.py`.

Available diagnostic variants:

```text
base_race
collector_race
opportunist
stealth_base
trap_bot
resource_then_bases
base_rotate
finish_base
stealth_rotate
stealth_finish
stealth_resource
```

`rush`, `rush_collect`, and old beam policy hooks were removed from the local
harnesses because those source files are not present.

## Submitted Results

Results are as such:

```text
base_race             Score 0.385  Speed 0.822
collector_race        Score 0.395  Speed 0.810
opportunist           Score 0.311  Speed 0.817
stealth_base          Score 0.464  Speed 0.817
trap_bot              Score 0.215  Speed 0.826
resource_then_bases   Score 0.385  Speed 0.809
base_rotate           Score 0.458  Speed 0.823
finish_base           Score 0.454  Speed 0.813
stealth_rotate        Score 0.529  Speed 0.834
stealth_finish        Score 0.343  Speed 0.822
stealth_resource      Score 0.539  Speed 0.821
normal/latest         Score 0.550  Speed 0.827  # after agent escape-check change
```

Interpretation:

- Contact-heavy policies are bad: `trap_bot` and `opportunist` underperformed.
- Enemy avoidance matters: `stealth_base` was best.
- Base pressure still matters: `base_rotate` and `finish_base` were close.
- Resource-heavy did not clearly help: `resource_then_bases` tied `base_race`.
- `stealth_resource`/`stealth_rotate` beat earlier diagnostics, but normal
  policy remains competitive and had a reported lucky high around `0.624`.
- `stealth_finish` is a strong negative signal: do not overcommit to finishing
  damaged bases when that creates contested or inefficient routes.
- The `normal/latest` `0.550` run suggests stricter agent-bomb logic helped.

The three v2 variants were submitted:

```text
stealth_rotate     # stealth avoidance + rotate after 2 base bombs
stealth_finish     # stealth avoidance + persist on damaged bases
stealth_resource   # stealth avoidance + resource refill bias
```

Next submissions should focus on normal-policy improvements rather than more
fixed diagnostic profiles.

## Important Files

- `src/ae_manager.py`
  - Reads `AE_POLICY_VARIANT`.
  - `normal` -> `HeuristicPolicy`.
  - otherwise -> `make_diagnostic_policy(variant, **DEFAULT_POLICY_KWARGS)`.
  - Also supports optional RL attack module env vars:
    `AE_ATTACK_MODEL`, `AE_ATTACK_BOMB_MARGIN`, `AE_ATTACK_MODULE_MODE`.

- `src/diagnostic_policies.py`
  - Contains `VariantProfile`, `PROFILES`, and `FreeForAllDiagnosticPolicy`.
  - Variant behavior is mostly weight tuning over bases, tiles, enemy pressure,
    contest avoidance, and danger.

- `Dockerfile`
  - Has:

    ```dockerfile
    ARG AE_POLICY_VARIANT=normal
    ARG AE_ATTACK_MODEL=
    ARG AE_ATTACK_MODULE_MODE=hybrid
    ARG AE_INSTALL_TORCH=0
    ENV AE_POLICY_VARIANT=${AE_POLICY_VARIANT}
    ```

- `src/policy.py`
  - `HeuristicPolicy` now has an orientation-aware escape check for enemy-agent
    bomb hits: an agent in our blast is only treated as definite if it cannot
    reach a non-blast cell within `BOMB_TIMER` from any possible facing.
  - RL attack module modes:
    - `replace`: training semantics; RL owns attack decisions.
    - `hybrid`: submission semantics; scripted base/high-confidence bombs are
      preserved and RL only handles marginal attack choices.

- `src/rl_attack.py`, `src/rl_attack_model.py`, `src/rl_attack_ppo_model.py`
  - Optional DQN/PPO attack modules using scalar features plus a CNN over a
    full-map stale-memory tensor.

- `test_env/train_attack_dqn.py`, `test_env/train_attack_ppo.py`
  - Local RL training harnesses. DQN has `--train-every`; both support CUDA and
    tqdm logging.

- `test_env/auto_play.py`
  - Local smoke harness. Use from repo root with:

    ```bash
    PYTHONPATH=ae/src:til-26-ae uv run python ae/test_env/auto_play.py --help
    ```

## Verification Already Done

Commands that passed:

```bash
uv run python -m compileall src test_env/auto_play.py test_env/diagnose_policy.py

PYTHONPATH=ae/src:til-26-ae AE_POLICY_VARIANT=base_race \
  uv run python -c "from ae_manager import AEManager; mgr=AEManager(); print(type(mgr._policy).__name__, getattr(mgr._policy, 'variant', None))"
```

Short smoke benchmark also ran successfully:

```bash
PYTHONPATH=ae/src:til-26-ae uv run python ae/test_env/auto_play.py \
  --benchmark --rounds 1 --novice --benchmark-types base_race trap_bot --no-cache
```

## RL Attacker Note

Earlier user-reported DQN single-agent RL test:

```text
self-play-ish score: ~0.250
hidden eval Score: 0.517
Speed: 0.837
```

Later hidden submissions with local trained DQN were weaker:

```text
dqn_v_heuristic  Score 0.389  Speed 0.828
dqn_selfplay     Score 0.497  Speed 0.830
```

Important caveat: user later realized the supposed self-play checkpoint may
have been trained against heuristic/stationary twice by mistake. Real self-play
training may still be worth testing, but do not assume RL is ahead of the
heuristic baseline yet.

Current RL integration status:

- `requirements-rl.txt` contains optional local training deps (`torch`, `tqdm`).
- Production `requirements.txt` intentionally does not include torch.
- Docker can optionally install torch with `--build-arg AE_INSTALL_TORCH=1`.
- `.pt` checkpoints should be copied under `ae/models/` and referenced as
  `models/<name>.pt` inside the image.

Recommended RL integration path:

1. Keep scripted dodge/pathfinding/objective selection.
2. Prefer `AE_ATTACK_MODULE_MODE=hybrid` for submissions so scripted base bombs
   are preserved.
3. Use `replace` only for training or controlled comparison.
4. If submitting `.pt`, include `--build-arg AE_INSTALL_TORCH=1` and test image
   startup/score separately.

## Next Engineering Ideas

Highest priority normal-policy ideas:

1. Port base-firing-cell routing into normal policy.
   - Normal currently routes to enemy base cells.
   - Diagnostic policies route to cells whose bomb blast can hit the base.
   - This should reduce wasted walking: the agent only needs a firing cell, not
     the base tile itself.

2. Add a pessimistic action filter.
   - Inspired by Pommerman top agents (`dypm`, Skynet/action pruning).
   - Before returning an action, reject legal actions that leave no safe escape
     path under projected bomb danger and simple enemy-bomb assumptions.
   - This should be cheap: evaluate the 6 immediate actions, not full MCTS.

3. Make predictive agent bombs stricter.
   - The escape-check helped.
   - Continue treating base bombs as valuable, but reject open-space predictive
     agent bombs unless the enemy is trapped/chokepointed or the expected hit is
     very high.

4. Add contested-route penalty to normal objective routing.
   - `stealth_resource`/`stealth_rotate` suggest enemy avoidance helps.
   - Avoid full diagnostic fixed scoring; just penalize collect/base targets near
     recent enemy sightings.

5. Improve post-wall-break waiting behavior.
   - Observed behavior: agent places a bomb for a destructible wall, then stands
     around waiting for the 3-tick fuse even when it could move left/right to
     collect nearby mission/resource tiles.
   - Desired behavior: after bombing for space/wall breaking, use the fuse time
     productively while preserving the ability to return to the opened route.
   - Prefer moves that:
     - stay safe from enemy bombs,
     - collect nearby mission/resource tiles,
     - avoid leaving the wall-opening route,
     - return orientation toward the opened area once the wall clears.
   - If bombing for space, prioritize temporary movement along cells in/near the
     bomb's blast area that lead toward destructible walls/opened area.

Endgame note:

- Do **not** add a naive "last 30 steps attack bases" mode.
- User noted that in hidden eval endgame there may be no bases left because all
  6 agents attack bases and bases only have 100 HP (5 bombs).
- Endgame logic should instead be conditional:
  - if enemy bases remain and are reachable, target firing cells;
  - otherwise prioritize mission/resource tiles and safe movement, not ghost
    base pressure.

External strategy references:

- RBC Borealis Pommerman blog:
  https://rbcborealis.com/research-blogs/pommerman-team-competition-or-how-we-learned-stop-worrying-and-love-battle/
- DYPm/Hakozaki pessimistic tree search:
  https://proceedings.mlr.press/v101/osogami19a.html
- Skynet action pruning / Pommerman baseline:
  https://github.com/BorealisAI/pommerman-baseline

Useful caution:

- Local `auto_play` is only a smoke test. It does not predict hidden eval well.
- Hidden eval appears to punish contact; avoid combat unless DQN is handling it.
