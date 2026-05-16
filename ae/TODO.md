# AE Agent — Deferred Work

Items the heuristic agent v1 deliberately does NOT do. Listed roughly in
priority order; each note explains why it was skipped and the rough cost
of adding it.

## Medium value

- **Action mask isn't fed into BFS.** `pathfinding` uses
  `memory.passable` (our belief). The action_mask is canonical truth. If
  the chosen action is masked off, we just STAY. Better: include action
  mask in the first step's options and replan if needed.
  (NOTE: there was a serious bug that caused agents to be stuck in a loop where they spin
  on the spot despite having possible actions, therefore this item is delayed for future
  review)

- **Multi-step dodge lookahead.** `_panic_move` greedily picks whichever
  single neighbor has the latest first-blast tick. When three neighbors
  are all dangerous, the greedy step may pick a cell that is a dead end
  at step N+1. Replace with a short BFS/DFS (depth ≤ 3) that finds a full
  escape path (every cell in the path is safe for its arrival tick) rather
  than just the safest single step.

- **Frontier scoring by revealed area.** `_try_explore` picks the cheapest
  reachable frontier cell but all frontier cells are treated equally. A
  cell bordering four unknown cells reveals four times as much map as one
  bordering a single unknown. Add a term to the exploration score:
  `value = unknown_neighbor_count / (travel_cost + 1)`, breaking ties
  toward cells that expose the most new information per tick spent.

## Low value / nice-to-have

- **Learned policy slot.** `Policy` ABC exists; a `LearnedPolicy(Policy)`
  could load a small model (PPO, etc.) and replace `HeuristicPolicy`.
  Cost: training infra in `til-26-ae` sim, model serialization, and
  loading code. AE Docker is CPU-only so model must be small.

- **Facing-aware exploration bias.** The exploration step costs in Dijkstra
  include turn costs, but `_try_explore` does not bias toward the agent's
  current facing direction. An agent facing east that has equal-cost
  frontier cells north and east should prefer east (no turn cost), saving
  one tick per exploration move on average. Expose a `facing_bias` weight
  on the exploration score.

- **Bomb reserve awareness in attack/defend decisions.** `HeuristicPolicy`
  does not track how many bombs are currently in flight versus the per-agent
  bomb limit. If our bomb is already placed (or not yet detonated), placing
  another is a no-op. Surface `bomb_in_flight` state and gate `_try_attack`
  / `_try_defend` on it, allowing the policy to explicitly wait or reroute
  rather than wasting the PLACE_BOMB action on a silent no-op.

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

- ~~Temporal danger-aware pathfinding~~ — implemented as `temporal_first_action_to`
  in `pathfinding.py`. The function is a drop-in variant of `first_action_to` that
  takes the full `project_danger()` timeline and, at each Dijkstra expansion, checks
  whether `next_pos in danger_timeline[arrival_tick]` where `arrival_tick =
  round(accumulated_cost + step_cost)`. Since every action costs 1 game tick,
  accumulated cost equals the tick offset at which the agent reaches a cell.
  Blocked: moving into a cell at the exact tick a bomb fires there.
  Allowed: passing through a cell at tick 1 when the bomb fires at tick 3; entering
  a cell at tick 4 that was blasted at tick 2 (already cleared). Turn actions
  (LEFT/RIGHT) keep `next_pos == pos` so staying in a cell that becomes dangerous
  at `cost+1` is also correctly rejected. `_dodge` in `policy.py` now uses
  `temporal_first_action_to` with the raw edge cost (no wall breaking) and drops
  the previous `immediate` filter and `dodge_cost` closure, which only blocked
  tick-0/1 cells and missed all later-tick blast paths.

- ~~Proactive enemy base routing~~ — implemented as `proactive_base_routing=True`
  (opt-in, default OFF) on `HeuristicPolicy`. When enabled, known enemy base
  cells are included in the `_try_collect` scoring pass with a synthetic value of
  `base_route_weight` (default 3.0, comparable to MISSION=5/RESOURCE=2/RECON=1).
  The same `value / (distance + 1)` formula ranks tiles and bases together, so a
  nearby high-value tile always beats a distant base while an uncontested close base
  (or post-tile-exhaustion) wins naturally. Attack/defend still fire before collect,
  so approaching a base never blocks an immediate bomb opportunity.
  Toggles: `HeuristicPolicy(proactive_base_routing=True, base_route_weight=3.0)` /
  `auto_play.py --proactive-base-routing --base-route-weight 3.0`.

- ~~Anti-oscillation / loop detection~~ — implemented as `loop_detection=True`
  (default ON) on `HeuristicPolicy`. A `deque` of `(action, position)` pairs
  is maintained with configurable `loop_window` (default 6). Before committing
  to a non-dodge action, `_is_loop()` checks whether adding that `(action, pos)`
  entry would complete a period-2 or period-3 repeating suffix in the history
  (both action and coordinates must match — same action at different cells is
  not a loop). When a loop is detected, `_break_loop()` selects the first legal
  non-looping alternative, preferring turns (LEFT/RIGHT) over linear motion over
  STAY. Dodge actions bypass the check entirely to avoid blocking safety moves.
  Toggle: `HeuristicPolicy(loop_detection=False)` /
  `auto_play.py --no-loop-detection`. Window: `--loop-window N` (must be ≥ 5
  to catch period-3 cycles).

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

- ~~Defend stance is naive~~ — `smart_defend=True` (default) redesigned to
  coverage-based positioning with four scoring tiers and proactive hotspot
  pre-positioning. `_ensure_av_hotspots()` sweeps all 256 cells once per round
  (no-op after first call), caching each passable cell's attack-vector coverage
  count. `_defend_coverage_score()` tiers: (1) 2×agent_bomb_value per enemy in
  covered cells; (2) velocity-projected bonus (predictive_defend); (3) 0.5×
  per active enemy bomb in covered cells — confirmed attack corridor; (4)
  strategic baseline coverage/max_cov in (0,1] when any enemy is visible,
  enabling proactive movement to hotspots before an attack begins.
  `_try_defend` now enters defend mode when any enemy is visible (not just
  within DEFEND_RADIUS), uses bomb positions as virtual threats so the
  bomb-and-retreat exploit keeps defend active, applies a danger-zone filter to
  intercept targets so the agent never navigates into a live blast, and switches
  to aggressive chase (advance on enemy's current position) when the enemy
  planted a bomb then retreated.
  Toggles: `HeuristicPolicy(smart_defend=True, predictive_defend=True)` /
  `auto_play.py --no-smart-defend --no-predictive-defend`.

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

- ~~Stale tile_contents in novice mode.~~ — fixed by the novice-map
  cache: `AEManager.__init__` re-merges the cache on every `/reset`, so
  walls/tiles destroyed or consumed in round N are restored to their
  initial state at the start of round N+1.

- ~~Adaptive wall-break cost based on tile value behind the wall~~ — implemented as
  `adaptive_wall_break_cost=True` (opt-in, default OFF) on `HeuristicPolicy`. When
  enabled, `_edge_cost` replaces the flat `wall_break_cost` with
  `wall_break_cost / (1 + tile_value(target_cell))` for destructible wall edges.
  Effect at the defaults (`wall_break_cost=5.0`): mission tile (value 5) attracts
  wall-breaking at ~0.83 cost, resource (2) at ~1.67, recon (1) at 2.5, empty/unknown
  (0) keeps the full 5.0 penalty — so high-value targets draw the agent through walls
  naturally without a separate planning pass. Orthogonal to `wall_break_tile_threshold`
  (which gates the bomb action) — this feature only adjusts pathfinding cost.
  Toggle: `HeuristicPolicy(adaptive_wall_break_cost=True)` /
  `auto_play.py --adaptive-wall-break-cost`.
