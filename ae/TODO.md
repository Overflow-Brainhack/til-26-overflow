# AE Agent — Deferred Work

Items the heuristic agent v1 deliberately does NOT do. Listed roughly in
priority order; each note explains why it was skipped and the rough cost
of adding it.

## High value

- **Defend stance is naive.** `_try_defend` just walks toward the closest
  enemy near our base — doesn't bomb them, doesn't intercept on the line
  between enemy and base. Should: pre-position between enemy and base,
  bomb when enemy steps in range, weight base-health against tile-pursuit.

- **Predictive bomb threshold isn't auto-tuned.** Multi-seed self-play
  (n=48 across 8 seeds) shows predictive bombing slightly *hurts* in
  self-play (213 → 192 mean reward). Either the threshold is too low or
  the random-walk uniform-distribution model over-counts hits. Try a
  drift-aware enemy model (last-velocity continuation), or raise the
  threshold so we only bomb on ≥0.5 expected hits.

## Medium value

- **Bomb economy.** Resources accumulate at 0.1/step → ~1.5 resources
  every 15 steps = 1 bomb. We currently bomb whenever an enemy is in
  range. Should reserve bombs for high-value targets (enemy bases > enemy
  agents > destructible walls blocking missions). Especially in late game
  when no enemies are in range, holding a bomb to break a wall to a
  mission is +5 points vs 0 for a wasted attack.

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
