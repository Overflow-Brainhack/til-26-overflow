# AE Next Session Handover

Last updated: **2026-05-22**.

This file was rewritten from scratch this session — earlier dated session-logs had
drifted badly out of sync with the code (they referenced `diagnostic_policies.py`,
`AE_POLICY_VARIANT`, `berserker_base`/`base_race`/`trap_bot` agent types, and
`hardened_dodge`/`contested_route_penalty`/`tour_lookahead` toggles — **none of which
exist anymore**). Treat this document as the single source of truth; if anything below
disagrees with the code, the code wins — re-verify and update this file.

> ⚠️ **EVERYTHING IN `edited_policy_v2.py` IS EXPERIMENTAL.**
> The `EditedHeuristicPolicyV2` (subclass) is in
> active flux. Production currently *ships* v2 for testing, but do not treat any class, method,
> toggle, or default in that files as stable/blessed. New ideas land as v2
> `__init__` toggles (default-tuned), get benchmarked, and only later get folded down.

## Production wiring (what actually runs)

- `ae/src/ae_manager.py` imports `EditedHeuristicPolicyV2 as HeuristicPolicy` and
  instantiates it with `DEFAULT_POLICY_KWARGS` (defined in the same file). There is no
  env-var policy switch anymore (the old `AE_POLICY_VARIANT` lines are gone).
- The server (`ae_server.py`) recreates `AEManager` on `/reset` and on `step == 0`.
  Static map knowledge survives via the module-level singleton in `map_memory` plus the
  bundled `novice_map.json` cache.
- `DEFAULT_POLICY_KWARGS` (production tuning, base-policy params only):
  `predictive_bomb=True, predictive_bomb_threshold=0.7, wall_breaking=True,
  wall_break_cost=5.0, adaptive_wall_break_cost=False, smart_defend=True,
  predictive_defend=True, drift_aware_bomb=True, auto_tune_bomb=True,
  bomb_tune_target=0.40, bomb_economy=True, base_bomb_value=5.0, agent_bomb_value=1.0,
  bomb_reserve_threshold=1.5, wall_break_tile_threshold=0.0, loop_detection=True,
  loop_window=6, proactive_base_routing=True, base_route_weight=100,
  adaptive_base_weight=True, base_weight_min=0.2, base_weight_ramp_rate=0.02,
  base_weight_attack_cooldown=20`.

## `EditedHeuristicPolicyV2` — current state (experimental)

V2-only toggles, set ONLY as `__init__` defaults (never in `DEFAULT_POLICY_KWARGS`,
never as auto_play CLI args — see Conventions):

- `siege_base: bool = True` — once the agent can reach a *bombing stance* (a cell whose
  blast hits the nearest live enemy base) within `siege_radius` steps, lock on: sweep
  resources within `siege_radius` of the base to fund bombs, else hold on / return to a
  stance so the base policy's `_try_attack` keeps detonating it. Long-range approach is
  left to `proactive_base_routing`. Base positions are known up front (cached map); only
  alive/health is learned at run time, and `memory.enemy_bases` drops a base once observed
  destroyed, so this targets only live bases.
- `siege_radius: int = 5`.

- `cluster_collect: bool = True` (`cluster_radius: int = 2`, `cluster_weight: float = 0.35`)
  — **routing**. Overrides `_collect_value`: a collectible's score numerator is bonused by
  `cluster_weight * Σ value(other collectibles within cluster_radius)`, so tile clusters
  beat a lone distant tile of equal value. Tiles respawn ~≤40 steps, so a dense cluster
  compounds.

One **inert base hook** was added to `EditedHeuristicPolicy` to support this without forking
it: `_collect_value(cell, memory)` (returns `tile_value`) is the `_try_collect` scoring
numerator. Base behaviour is byte-identical when `cluster_collect` is off.

There is no `time_aware_routing` / `temporal_collect` toggle anymore, and
`_enhanced_collect`/`tour_lookahead` were removed.

Dodge is **not** overridden in v2 — it is inherited from the base policy. (Earlier
handovers claiming "dodge moved to v2 with `hardened_dodge`/`_dodge_v2`" are stale.)

## `EditedHeuristicPolicy` — base pipeline (experimental)

`choose()` order: `dodge → attack → (defend, COMMENTED OUT) → collect → explore → STAY`.

- **Dodge** (`_dodge`): temporal/time-space flee to a cell safe for ≥ `BOMB_TIMER+1`;
  safety-critical, skips loop detection. Falls back to `_panic_move`.
- **Attack** (`_try_attack` / `_bomb_opportunity_score` / `_expected_hits`): bombs bases
  and *trapped* moving agents. `bomb_economy=True` is the live path; auto-tuned predictive
  threshold via EMA (`_resolve_pending_bombs`, `_update_bomb_ema`). `_enemy_can_escape_blast`
  excludes enemies that can flee the blast before detonation.
- **Defend** (`_try_defend`): intentionally disabled in `choose()` — base defense rarely
  succeeds; we maximise score instead. `_intercept_cells` still exists (offense reuse).
- **Collect/Explore**: value-over-distance (`value/(dist+1)`) Dijkstra routing with
  wall-break substitution (`_maybe_wall_break`, `_productive_wait_move`), loop detection.

Notable base tunables (defaults): `minimum_aggression=1.0, aggression_ramp_rate=0.04,
defensive_force=0.75, defense_abandon_margin=2, max_defense_distance=9,
defense_cooldown_scale=0.6`. Productive wall-wait radius is `_PRODUCTIVE_WALL_WAIT_RADIUS`
in `edited_policy.py` (always on; tune directly).

## Trusted reward math + mechanics (do NOT drive the sim to re-verify)

- Collect mission/resource/recon = **+5 / +2 / +1**. Tiles respawn ≤ 40 steps (Perlin-modulated).
- Bomb damage **+1/HP**; **kill +30** (split across contributing bombs); **destroy base +50** (split).
- **Take bomb damage −1/HP** (one 20-dmg blast ≈ −20 ≈ 4 missions). **Your base damage −1/HP to you.**
- **Friendly fire OFF** — your own bomb never harms you (so placing bombs is near-free: 1.5
  resource + 1 tick; the only attacking risk is *enemy* bombs). Avoiding damage ≈ dealing it.
- Anti-moving-target specifics: enemies **see your bomb + its timer** and move before it
  detonates (phase order place→move→detonate→damage), so hits come from *removing escape*,
  not proximity. HP≤0 → **frozen `FREEZE_TURNS=3`** (STAY only), then respawn full HP in place;
  agent HP=60 = 3 bomb hits. Blast = Chebyshev radius 2, LOS-gated by both wall types.
  Enemy agent HP **is observable** (`ViewChannel.ENEMY_AGENT_HEALTH=22`) but is **not tracked**
  in `map_memory` today (only `enemy_base_health` is).

## Repo conventions (learned the hard way)

- New v2 toggles go ONLY in `EditedHeuristicPolicyV2.__init__` defaults. Do **not** add them
  to `DEFAULT_POLICY_KWARGS` (`ae_manager.py`) and do **not** add them to `auto_play.py`'s
  argparse/`policy_kwargs`. Benchmark by flipping the `__init__` default.
- Don't hand-drive the `til-26-ae` sim; test policies via the `auto_play.py` harness.
- The user runs long scripts and all `git` themselves — hand over commands, don't execute.

## Files (current, accurate)

- `src/ae_manager.py` — entry; `DEFAULT_POLICY_KWARGS`; ships v2.
- `src/ae_server.py` — FastAPI wrapper (do not modify the contract).
- `src/edited_policy.py` — **experimental** base `EditedHeuristicPolicy`.
- `src/edited_policy_v2.py` — **experimental** `EditedHeuristicPolicyV2` (siege toggles).
- `src/berserker_policy.py` — `BerserkerPolicy` (benchmark opponent).
- `src/threat.py` — danger projection + enemy prediction (`predict_enemy_positions`,
  `expected_blast_hits_drift`, `cells_in_blast`).
- `src/pathfinding.py` — Dijkstra / time-space search over `(pos, facing[, tick])`.
- `src/map_memory.py` — persistent map + dynamic entity tracking.
- `src/rl_attack.py`, `src/rl_attack_model.py` — optional DQN attack module (torch not in
  prod `requirements.txt`; opt-in via `--build-arg AE_INSTALL_TORCH=1`). RL is **not** known
  to beat the heuristic baseline; treat as exploratory.
- `test_env/auto_play.py` — local visual + headless benchmark harness. Agent types:
  `normal` (= `EditedHeuristicPolicy` by default in the harness), `berserker`, `random`.

### Endgame note

Do NOT add a naive "last 30 steps attack bases" mode — in hidden eval, bases may already be
gone (6 agents × 100-HP bases). Keep it conditional: if live enemy bases remain & reachable,
target firing cells; else prioritise tiles + safe movement.

## Verification (hand these to the user; don't auto-run)

```bash
# import / instantiate sanity
.venv/bin/python -c "import sys; sys.path.insert(0,'ae/src'); \
  from ae_manager import AEManager; AEManager(); print('ok')"

# paired benchmark for any new toggle (flip its __init__ default off vs on)
.venv/bin/python ae/test_env/auto_play.py --benchmark --rounds 30 \
  --benchmark-types normal berserker

# visual spot-check (no --benchmark) on a cornered-enemy seed
.venv/bin/python ae/test_env/auto_play.py --rounds 1 --seed 42
```

Note: we run **novice mode** (the default — a single fixed map and start positions). Compare
a toggle off vs on as the same fixed-map A/B; per-round score still varies because map memory
accumulates across rounds (the singleton cache), so later rounds reflect learned knowledge.
