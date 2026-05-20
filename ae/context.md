# AE Agent — Working Context

Last refreshed: 2026-05-18. Read this first when picking up the AE task in a new session. Pair with `TODO.md` (open + resolved feature log) and `README.md` (server I/O contract).

## Snapshot

**Decision tree** (`policy.py:238` `HeuristicPolicy.choose`):
Frozen → Dodge → Attack → Defend → Collect → Explore → STAY.
Every non-dodge step passes through `_finalize` for loop detection.

**Live config** — `ae_manager.py:31-61` `DEFAULT_POLICY_KWARGS` (single source of truth; `auto_play.py` mirrors it for the visualiser):

```python
predictive_bomb=True, predictive_bomb_threshold=0.7
wall_breaking=True, wall_break_cost=5.0, adaptive_wall_break_cost=False
smart_defend=True, predictive_defend=True
drift_aware_bomb=True, auto_tune_bomb=True, bomb_tune_target=0.40
bomb_economy=True, base_bomb_value=5.0, agent_bomb_value=1.0,
  bomb_reserve_threshold=1.5, wall_break_tile_threshold=0.0
loop_detection=True, loop_window=6
proactive_base_routing=True, base_route_weight=100,
  adaptive_base_weight=True, base_weight_min=0.2,
  base_weight_ramp_rate=0.02, base_weight_attack_cooldown=20
```

`BerserkerPolicy` is a "reckless base-rush" alternative, currently commented out in `ae_manager.py:86`.

**Files** (all in `ae/src/`):
- `ae_server.py` — FastAPI wrapper. Do not edit unless the contract changes.
- `ae_manager.py` — per-round entrypoint, recreated on `/reset` and on `step==0`.
- `policy.py` — `HeuristicPolicy` (~1000 lines, single class, every feature toggleable).
- `berserker_policy.py` — alt policy.
- `map_memory.py` — `MapMemory` singleton; static state (walls, bases, tile types) persists across `/reset`, dynamic state (bombs, enemies) cleared per round.
- `threat.py` — `cells_in_blast`, `project_danger`, `expected_blast_hits_drift`.
- `pathfinding.py` — Dijkstra over (pos, facing) with a generic `EdgeCost`. `first_action_to`, `temporal_first_action_to`, `reachable_cells`.
- `observation.py` — `parse_observation` → `ParsedObs`.
- `constants.py` — grid/blast/reward constants.
- `novice_map.json` — pre-captured static map (novice mode has fixed seeds 19/88).

**Score model**: mission +5, resource +2, recon +1, attack_kill +30, destroy_base +50, own_base_destroyed −50. `BOMB_ATTACK=20`, `BASE_MAX_HEALTH=100`, `NUM_ITERS=200`, grid 16×16, CPU-only Docker, AE = 40 % of total TIL-AI score.

**Benchmarking** (`ae/test_env/`): `auto_play.py` (visualiser), `benchmark_bomb_economy.py`, `benchmark_bomb_threshold.py`. These are the templates to clone when sweeping a new toggle.

## Already shipped

See `TODO.md` "Resolved" for full notes. One-liners:
predictive bomb · drift-aware enemy cloud · online EMA auto-tune · `smart_defend` with attack-vector hotspot precompute · `predictive_defend` (velocity projection) · bomb economy with finishing-blow bonus · loop detection (period 2 & 3) · proactive base routing · adaptive base weight (ramp + attack cooldown) · novice-map cache · temporal danger-aware dodge · wall-breaking pathfinding · adaptive wall-break cost.

## Still open

See `TODO.md` "Medium value" + "Low value": action-mask-in-BFS (gated on spin-loop bug), multi-step dodge lookahead, frontier-by-revealed-area, learned policy slot, facing-aware exploration bias, bomb-in-flight awareness, coordinated bomb chains, use `health` / `base_health`, enemy sighting decay refinement, `base_view` for early threat detection.

## Improvement ideas

Grouped by category. Each idea names the file/function to touch and (where relevant) the toggle name to add. Default to OFF on new toggles; promote to ON in `DEFAULT_POLICY_KWARGS` only after a self-play sweep shows a non-trivial Δmean_reward.

### A. Heuristic refinements (cheap)

- **Endgame mode** (last ~30 steps). At `obs.step >= NUM_ITERS - 30` drop `_try_defend` entirely and lower `bomb_reserve_threshold` to ~0.2. The defensive cooldown and ramping base weight stop earning points once the round is nearly over. Touch: `HeuristicPolicy.choose` and `_bomb_opportunity_score`. Toggle: `endgame_mode`, `endgame_steps`.
- **Kamikaze finisher**. When any `memory.enemy_base_health[b] <= BOMB_ATTACK` and a firing cell is reachable in ≤ 3 ticks, override economy and accept own-base risk. Adds a pre-check above `_try_defend`. Toggle: `kamikaze_finisher`.
- **Resource-bomb timing**. Currently resources are flat value 2 in `tile_value`. Better: when `team_resources >= BOMB_COST + 0.4` and `team_bombs < cap`, deprioritise resource tiles (no point hoarding); when `team_bombs == 0`, boost the nearest resource score. Touch: `_try_collect` scoring loop.
- **Tile-respawn cycling**. `tile_contents` keeps tagging a tile as "resource" forever; `memory.collectible_cells()` already filters via `last_seen_step`, but only based on visibility, not consumption. Add `consumed_at: dict[(x,y), int]` updated when our agent stands on a known tile, and re-include after ~40 steps. Touch: `MapMemory`, `_try_collect`.
- **Facing-aware tiebreak** (`TODO.md:40`). In `_try_collect` and `_try_explore`, add a small bonus to candidates whose first action requires no turn. Cheap; saves ~1 tick per exploration move on average.
- **Frontier scoring by revealed area** (`TODO.md:25`). Replace flat-cost frontier in `_try_explore` with `value = unknown_neighbor_count / (cost+1)`. Cells bordering 4 unknowns reveal 4× the area.
- **Bomb-in-flight gate** (`TODO.md:46`). Track `bomb_in_flight` (we already have it via `memory.bombs[pos].ally`). Suppress `PLACE_BOMB` in `_try_attack` when our previous bomb is still ticking; reroute instead of issuing a silent no-op.
- **Low-HP risk aversion** (`TODO.md:55`). When `obs.health <= 20`, downweight attack and prefer dodge / collect routes that minimise blast exposure. Death freezes 3 ticks at spawn — that is real time lost.

### B. Search / planning

- **Multi-step dodge BFS** (`TODO.md:19`). Replace greedy 1-step `_panic_move` with a depth-3 DFS that finds a *path* where every cell is safe at its arrival tick. Reuse `project_danger` timeline.
- **Bomb chain planner**. Bounded 2-step lookahead: "place bomb here → dodge to X → place bomb there" for cascade-clearing corridors or hitting clustered targets. ≤ 6 expansions per call. Pairs naturally with proactive base routing on hardened approaches.
- **Local TSP over collectibles**. Replace greedy `value/(dist+1)` in `_try_collect` with: enumerate the top-K reachable collectibles (K ≤ 4), brute-force the 4! orderings, pick the one with highest total value within `EXPLORE_BUDGET`. Avoids the failure mode where the greedy pick leaves us next to a low-value tile when a slightly farther route would chain two high-value ones.
- **Action-mask-aware replan** (`TODO.md:7`). When the chosen first action is mask-blocked, try the second-best plan instead of falling through to STAY. Gated on resolving the spin-loop bug noted in TODO.
- **Trap detection in `_edge_cost`**. Penalise cells with ≤ 1 escape neighbour (computed via `memory.passable`) when enemies are visible and we lack a bomb to break out. Add `cul_de_sac_penalty` term. Touch: `_edge_cost`.
- **Time-discounted scoring**. Multiply tile/base value by `(NUM_ITERS - step) / NUM_ITERS` in `_try_collect`. Late-game distant tiles weigh less, freeing the agent to camp finishing positions.

### C. Opponent modeling / meta

- **Per-enemy behavior classifier**. Tag each tracked enemy as `{rusher, camper, patroller}` from velocity history + bomb cadence. Shape the `expected_blast_hits` cloud accordingly: rusher → drift-heavy (already what `drift_aware_bomb` does, but per-enemy); camper → collapse to point; patroller → expand along observed cycle. Touch: `threat.expected_blast_hits_drift`, `MapMemory.enemy_velocities`.
- **Enemy bomb cadence model**. Track inter-bomb intervals per enemy (`map_memory`). When an enemy is "due" for another bomb, increase Tier 3 weighting in `_defend_coverage_score` on their projected approach corridor.
- **Threat ranking by intent**. Rank enemies by `dot(velocity, base_direction) * 1/distance`. Defend against the top-1 unless multiple converge — avoids the ping-pong between enemies on opposite sides of the base that the current "any enemy in DEFEND_RADIUS" check can trigger.
- **Aggression-aware adaptive base weight**. Extend `_update_adaptive_weight` to track per-enemy aggression and ramp `_adaptive_weight` faster when no enemy has *ever* shown aggression in this round. Currently the cooldown resets on any threat signal, which over-defends against passive opponents.
- **Endgame attack-vector seeding**. At step ≥ 150, precompute the best firing cell for each surviving enemy base (intersect `cells_in_blast(base)` with our reachable cells). Park the "collect" subgoal at that cell so we are ready to finish-blow as soon as the base HP drops within `BOMB_ATTACK`.

## Verification recipe (per idea)

1. Add a kwarg to `HeuristicPolicy.__init__` and a default in `DEFAULT_POLICY_KWARGS` (default OFF).
2. Implement behind the toggle. Keep the baseline path untouched.
3. Sanity check with `python ae/test_env/auto_play.py --<toggle>`.
4. Self-play sweep (clone `benchmark_bomb_economy.py`) over ≥ 48 seeds. Require Δmean_reward > 0 with a margin larger than seed noise before flipping the default to ON.
5. Move the idea from this file's "Improvement ideas" section into `TODO.md` "Resolved" with a one-paragraph note on the toggle and the sweep result.
