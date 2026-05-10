# AE Agent — Deferred Work

Items the heuristic agent v1 deliberately does NOT do. Listed roughly in
priority order; each note explains why it was skipped and the rough cost
of adding it.

## High value


## Medium value

- ~~Stale tile_contents in novice mode.~~ — fixed by the novice-map
  cache: `AEManager.__init__` re-merges the cache on every `/reset`, so
  walls/tiles destroyed or consumed in round N are restored to their
  initial state at the start of round N+1.

- **Action mask isn't fed into BFS.** `pathfinding` uses
  `memory.passable` (our belief). The action_mask is canonical truth. If
  the chosen action is masked off, we just STAY. Better: include action
  mask in the first step's options and replan if needed.

## Low value / nice-to-have

- **Learned policy slot.** `Policy` ABC exists; a `LearnedPolicy(Policy)`
  could load a small model (PPO, etc.) and replace `HeuristicPolicy`.
  Cost: training infra in `til-26-ae` sim, model serialization, and
  loading code. AE Docker is CPU-only so model must be small.

- **Coordinated bomb chains.** Two bombs placed near each other can
  cascade-clear walls. We never plan multi-bomb sequences.

- **Use `health` and `base_health`.** Currently ignored. Could:
  - Retreat when low HP (we'll respawn anyway, but freezing for 3 ticks
    at 0 HP is real time loss).
  - Prioritize defense when our base is at low HP.

- **Enemy sighting decay.** `ENEMY_AGENT_TTL` is a flat 12 steps. Better
  would be uncertainty propagation (enemy could be anywhere within
  reachable Manhattan distance from last sighting).

- **`base_view` is parsed but underused.** We stamp tile/wall info from
  it, but don't use it for early threat detection (e.g. enemy approaching
  our base from outside our agent's viewcone).

## Resolved

- ~~Bomb economy~~ — implemented as `bomb_economy=True` (opt-in, default OFF) on
  `HeuristicPolicy`. When enabled, `_try_attack` replaces the hard threshold
  with a unified value score: `base_hits * base_bomb_value + agent_hits *
  agent_bomb_value + expected_hits * agent_bomb_value` (predictive term included
  when `predictive_bomb=True`). A bomb is placed only when `score >=
  bomb_reserve_threshold`. Bases are worth far more than agents by default
  (`base_bomb_value=5.0`, `agent_bomb_value=1.0`). Additionally,
  `wall_break_tile_threshold > 0.0` suppresses wall-break bombs when the tile
  behind the wall has insufficient value, conserving bombs for high-value targets.
  Toggle: `HeuristicPolicy(bomb_economy=True, bomb_reserve_threshold=1.5,
  base_bomb_value=5.0, wall_break_tile_threshold=3.0)` /
  `auto_play.py --bomb-economy --bomb-reserve-threshold 1.5 --base-bomb-value 5.0`.
  Sweep tool: `ae/test_env/benchmark_bomb_economy.py` — headless self-play
  over (reserve_threshold × base_value) and wall_break_tile_threshold grids,
  prints ranked table.

- ~~Predictive bomb threshold auto-tuning~~ — two mechanisms added:
  (1) **Drift-aware model** (`drift_aware_bomb=True`, default): instead of
  the uniform random-walk cloud, each reachable cell is weighted by
  `exp(drift_weight * dot(displacement, vel_unit))`, concentrating probability
  mass in the enemy's observed direction of travel. Enemy velocities are inferred
  each step by matching adjacent consecutive sightings in `MapMemory`. When
  velocity is unknown the distribution collapses to uniform.
  (2) **Online EMA auto-tuning** (`auto_tune_bomb=True`, opt-in): after each
  predictive bomb, we check whether any enemy appeared in the blast cells at or
  after placement; the hit/miss result feeds a per-session EMA. When hit rate
  drifts below `bomb_tune_target` (default 0.40) the threshold rises; above
  target it falls. Clamped to [0.05, 0.95].
  (3) **`benchmark_bomb_threshold.py`**: headless self-play sweep over arbitrary
  threshold grids + auto-tune, prints ranked table and recommendation.
  Toggles: `HeuristicPolicy(drift_aware_bomb=..., auto_tune_bomb=...)` /
  `auto_play.py --no-drift-aware-bomb --auto-tune-bomb`.

- ~~Defend stance is naive~~ — implemented as `smart_defend=True` (default) on
  `HeuristicPolicy`. `_try_defend` now: (1) computes an *intercept* cell
  `INTERCEPT_STEPS=2` out from the base toward the enemy, navigating there
  instead of chasing the enemy directly — placing us on their inbound path so
  `_try_attack` can bomb them the next tick they enter our blast radius;
  (2) dynamically expands `effective_radius` from 4 → up to 8 as
  `base_health` drops to zero, so we engage threats earlier when the base is
  at risk. Toggle: `HeuristicPolicy(smart_defend=...)` /
  `auto_play.py --no-smart-defend`.

- ~~Friendly-fire safety check (`_can_escape_after_self_bomb`)~~ — removed
  after confirming `dynamics.py:691-692` skips same-team defenders. Our
  own bomb cannot damage our agent or our base.

- ~~Predictive bomb targeting~~ — implemented as
  `threat.expected_blast_hits`. Each known enemy is treated as uniform
  over its `BOMB_TIMER`-step random-walk reachability cloud; we bomb when
  Σ |cloud ∩ blast| / |cloud| ≥ `predictive_bomb_threshold` (default 0.25).
  Toggle: `HeuristicPolicy(predictive_bomb=...)` /
  `auto_play.py --no-predictive-bomb`.

- ~~Wall-breaking pathfinding~~ — `pathfinding` is now Dijkstra over
  (pos, facing) with a generic `EdgeCost` callback. `HeuristicPolicy`
  builds an EdgeCost where destructible walls cost `wall_break_cost`
  (default 5.0). When the chosen first action crosses a destructible
  wall, `_maybe_wall_break` substitutes `PLACE_BOMB` (and `STAY` if our
  own bomb is already placed at this cell, to avoid double-bombing).
  Toggle: `HeuristicPolicy(wall_breaking=...)` /
  `auto_play.py --no-wall-breaking`. Multi-seed self-play (n=48) showed
  +43% mean reward when enabled.

- ~~Novice map cache~~ — confirmed novice mode hardcodes maze seed 19
  and episode seed 88, so the map (walls, base positions, initial tile
  layout) is byte-identical every game regardless of user seed.
  `MapMemory.save()/load()/merge_static_from()` serialize the static
  subset to JSON. `ae/test_env/capture_novice_map.py` plays a few rounds
  to populate the cache and writes it to `ae/src/novice_map.json`, which
  the Dockerfile bundles via `COPY src .`. `AEManager` auto-loads the
  cache on every `/reset`, so destroyed walls / consumed tiles get
  restored at round boundaries (matching the env's own reset). Toggle:
  `AEManager(cache_path=None)` / `auto_play.py --no-cache`. First-round
  self-play (n=48) showed +7 mean reward when cache is loaded.
