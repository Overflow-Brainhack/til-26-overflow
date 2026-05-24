# AE RL Handoff — shadow-rl-experiment branch

Snapshot for the next agent picking up this branch. Last touched 2026-05-23.

## What's uncommitted

All work is on `shadow-rl-experiment`, unstaged. Nothing has been committed yet.

```
M ae/src/rl_policy.py
M ae_rl/model.py
M ae_rl/ppo.py
M ae_rl/rollout.py
M ae_rl/train_stage2_ppo.py
M ae_rl/train_stage3_league.py
M week_long_edits/ae_rl/ppo.py            # mirror of ae_rl/ppo.py
M week_long_edits/ae_rl/train_stage2_ppo.py
?? ae_rl/diagnose.py                       # new — per-event scoring breakdown
?? ae_rl/review-the-reinforcement-learning-bright-lemur.md   # ultrareview output
?? week_long_edits/ae_rl/diagnose.py
?? week_long_edits/ae_rl/train_stage3_league.py
?? ae_cpp/                                 # unrelated; ignore
```

`week_long_edits/` is a copy used for offline tinkering. Keep its `ae_rl/` files in sync with the canonical `ae_rl/` when shipping anything substantive.

## What changed and why

Three independent experiments stacked onto Stage 2 / Stage 3 PPO training. Each can be toggled or reverted on its own.

### 1. Return normalisation (`ae_rl/ppo.py`, `train_stage2_ppo.py`)

- New `RunningReturnNorm` (Welford running mean/var over scalar returns).
- `ppo_update(..., return_norm=...)` rescales `returns` and `val_old` by `std` (no mean subtraction — keep sign).
- Stage 2 wires it through both critic warmup and main PPO loop, sharing one instance.
- Stage 3 does **not** yet thread it through — intentional, want to confirm Stage 2 first.
- Critic outputs are in normalised units while training. Logging shows normalised value loss; that's expected.

### 2. PBRS (Potential-Based Reward Shaping) in `ae_rl/rollout.py`

- Per-step shaping `γ·Φ(s') − Φ(s)` added to the learner's reward inside `_collect_selfplay_episodes`.
- `Φ(s) = -0.2 · dist_to_nearest_known_collectible_tile - 0.2 · dist_to_nearest_known_enemy_base (if team_bombs ≥ 1)`.
- Manhattan distance over `MapMemory` contents. Falls back to 0 when nothing known.
- Policy-invariant in the limit (Ng et al. 1999), so it should accelerate learning without changing the optimal policy.

### 3. Reward shaping at env level (training only) in `ae_rl/rollout.py`

- `make_env(novice, shape_rewards=False)` — new flag.
- When `shape_rewards=True`:
  - Sets `cfg.rewards.step_penalty=-0.02`, `stationary_penalty=-0.05`, `invalid_action=-0.5`.
  - Wraps `env.dynamics.rewards.award` to multiply offensive events: `attack_damage` (positive only) ×1.5, `destroy_enemy_base` ×2.0, `attack_kill` ×1.5. Damage-taken untouched so survival pressure stays intact.
- `_make_env_pool(...)` defaults to `shape_rewards=True` (training-side default).
- `collect_teacher_dataset(...)` and the teacher-worker init both call `make_env(..., shape_rewards=False)` — BC demos must be on raw reward.
- **Eval/benchmark/diagnostic callers must explicitly pass `shape_rewards=False`.** The new `ae_rl/diagnose.py` does this; double-check anything you add.

### 4. Safety: `weights_only=True` on `torch.load`

- `ae_rl/model.py::load_checkpoint`, `ae/src/rl_policy.py::RLPolicy.__init__`, and the `_checkpoint_score` helpers in both training scripts now use `weights_only=True`.
- Our checkpoints only contain `{model_state, arch, meta}` — all primitives, all safe under the restriction.
- If you add anything exotic to a checkpoint meta dict (e.g. arbitrary objects), this load will fail.

### 5. New: `ae_rl/diagnose.py`

Per-event scoring breakdown for a checkpoint. Spies on `env.dynamics.rewards.award` to bucket score into mission tiles / kills / base destruction / invalid / stationary / etc, rotates the RL slot across rounds, and prints a heuristic baseline. Run via `uv run ae_rl/diagnose.py [--ckpt ...] [--rounds N] [--focus-slot agent_0] [--sample-actions] [--advanced]`.

### 6. Ultrareview output

`ae_rl/review-the-reinforcement-learning-bright-lemur.md` — output of `/ultrareview` against the RL changes. Worth skimming before committing.

## Suggested next steps

1. **Validate Stage 2 first.** Run `python ae_rl/train_stage2_ppo.py --validate-every 10 --rollback-on-regress` and compare to the pre-change baseline. The user runs scripts themselves — don't kick off long training runs from Claude.
2. **If Stage 2 looks good**, thread `RunningReturnNorm` into Stage 3 (`train_stage3_league.py`). Be aware league snapshots loaded as opponents come from the *current* normalised value-head era; opponents only need `model_state` for `act()`, so this should be safe but verify.
3. **Decide on the shaping defaults.** Right now PBRS + env shaping + offensive multipliers are *all* on by default in training. If the diagnostic shows the policy gaming one of them (e.g. spamming bombs because of `attack_damage ×1.5`), turn that one off first — they're independent.
4. **`shape_rewards` is footgun-shaped.** If you add a new entry point that builds an env, audit whether it's training (default True via pool) or eval (must be False). Consider flipping the default to False and making training explicitly opt in.
5. **Commit boundaries.** The three experiments are independent — split into three commits (safety/`weights_only` is a fourth trivial one) so a regression can be bisected.
6. **`week_long_edits/` drift.** Currently `ppo.py` + `train_stage2_ppo.py` are mirrored but `rollout.py` is not. Decide whether to mirror or to delete `week_long_edits/ae_rl/` outright.

## Things to be careful about

- **User runs scripts.** Don't burn time on long RL training runs from inside Claude; hand the command over.
- **No git commits without asking** — `Bash(git:*)` is denied for `ae/` and the user prefers to drive git themselves.
- **AE container reset semantics** (per CLAUDE.md): the RL policy state must clear on `obs.step == 0` or `/reset`. The deployment-side `RLPolicy.choose` already resets `self._hidden` on step 0; nothing here changes that, but if you add round-persistent state in the RL stack, mirror it.
- **The deploy bundle** (`ae/src/rl_policy.py`) loads `models/stage2_ppo.pt` by default. If you ship a Stage 3 league checkpoint, either update `DEFAULT_CHECKPOINT` or wire `checkpoint_path` in `ae_manager.py`. The arch fallback in `_ActorCritic.__init__` defaults must stay in sync with `ae_rl/model.py::RecurrentMaskableActorCritic` defaults.
