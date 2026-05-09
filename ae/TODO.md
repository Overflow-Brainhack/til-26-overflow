# AE Agent — Deferred Work

Items the heuristic agent v1 deliberately does NOT do. Listed roughly in
priority order; each note explains why it was skipped and the rough cost
of adding it.

## High value

- **Predictive bomb targeting.** We drop bombs reacting to *current* enemy
  positions; the bomb takes `BOMB_TIMER` (3) ticks to detonate. By then the
  enemy has likely moved out of blast. Improvement: reason about likely
  enemy moves (e.g. enemies usually advance toward collectibles) and drop
  bombs at predicted-future positions or chokepoints. Cost: needs an enemy
  model, even a crude "most likely action" one.

- **Wall-breaking pathfinding.** Destructible walls are treated as
  impassable. If a high-value tile (mission, +5) sits behind a destructible
  wall, we ignore it. Improvement: `pathfinding` should optionally treat
  destructible edges as passable-with-cost (cost ~ BOMB_TIMER + a bomb
  charge), and `policy._try_collect` should consider wall-breaking when
  the value justifies the detour. Cost: extend `can_traverse` API to
  return cost, A* instead of BFS.

- **Defend stance is naive.** `_try_defend` just walks toward the closest
  enemy near our base — doesn't bomb them, doesn't intercept on the line
  between enemy and base. Should: pre-position between enemy and base,
  bomb when enemy steps in range, weight base-health against tile-pursuit.

## Medium value

- **Bomb economy.** Resources accumulate at 0.1/step → ~1.5 resources
  every 15 steps = 1 bomb. We currently bomb whenever an enemy is in
  range. Should reserve bombs for high-value targets (enemy bases > enemy
  agents > destructible walls blocking missions). Especially in late game
  when no enemies are in range, holding a bomb to break a wall to a
  mission is +5 points vs 0 for a wasted attack.

- **Stale tile_contents in novice mode.** We persist `tile_contents`
  across rounds (singleton survives `/reset`). If we saw a tile consumed
  in round N, we'll start round N+1 thinking it's empty until we
  re-observe. In novice mode tiles reset at round start, so we miss
  collectibles in cells we don't visit early. Fix: clear tile_contents
  on reset_round (still keep walls / base positions), or re-stamp on
  every observation regardless.

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
