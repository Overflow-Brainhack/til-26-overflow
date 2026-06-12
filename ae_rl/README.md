# ae_rl — Recurrent Maskable PPO for the AE Bomberman game

A self-contained PyTorch implementation of a **recurrent, action-masked PPO**
agent for the TIL-26 AE challenge, jump-started by behaviour-cloning the
production heuristic (`EditedHeuristicPolicyV2`) and trained through self-play.

No `stable-baselines3` dependency — sb3-contrib has no combined *recurrent +
maskable* PPO class, and the PettingZoo AEC env needs custom self-play wrapping
regardless, so the PPO loop, masking, and GRU recurrence are implemented here
directly.

## Why this design

- **Partial observability** (a directional viewcone) → a **GRU** integrates the
  view over time instead of relying on a single frame.
- **Illegal actions** (move into wall, bomb with no bombs, frozen → STAY-only)
  → **action masking** on the logits, applied both when acting and during the
  PPO update, so probability mass is never wasted on illegal moves.
- **Sparse-ish rewards** (destroy base +50, kill +15) → **BC warm-start** from
  the strong heuristic gives PPO a competent starting policy instead of cold
  exploration. This is the high-leverage move: it strictly improves on the
  heuristic rather than betting on RL discovering good play from scratch.
- **Unknown opponents** → a **self-play league** (Stage 3) trains against a pool
  of frozen past selves plus the heuristic, to avoid overfitting one opponent.

## Architecture

```
agent_viewcone (25×7×5) ─ CNN ─┐
base_viewcone  (25×7×7) ─ CNN ─┼─ concat → Linear → GRU ─┬─ actor (6 logits, masked)
scalars        (14,)    ─ MLP ─┘                         └─ critic (value)
```

~1.1M parameters. The 14 scalars are direction (one-hot), location,
base-location, health, frozen-ticks, base-health, team-resources, team-bombs,
and step — all normalised (`common.build_scalars`).

Episodes are a fixed 200 steps (agents freeze/respawn, never terminate), so
every learner trajectory is exactly 200 long and trajectories stack into a clean
`(T=200, B)` batch with no padding. PPO minibatches over the **sequence (B)**
dimension and unrolls the GRU over full trajectories (zero initial hidden) to
keep the recurrence intact.

## The three training stages

Each stage auto-discovers the previous stage's checkpoint in `checkpoints/`.

```bash
# Stage 1 — behaviour cloning (jump-start) from the heuristic.
python ae_rl/train_stage1_bc.py --episodes 64 --epochs 8
#   → checkpoints/stage1_bc.pt

# Stage 2 — PPO self-play vs the heuristic (warm-starts from stage1).
python ae_rl/train_stage2_ppo.py --updates 200 --episodes-per-update 8 --learners 3
#   → checkpoints/stage2_ppo.pt   (resumes itself if present)

# Stage 3 — PPO league self-play vs frozen past selves + heuristic
#           (warm-starts from stage2, snapshots itself into checkpoints/league/).
python ae_rl/train_stage3_league.py --updates 300 --snapshot-every 25 --heuristic-prob 0.5
#   → checkpoints/stage3_league.pt
```

All stages **default to `--novice`** (the fixed competition map); pass
`--advanced` for randomised maps (better for generalisation). All take `--seed`.
Run `--help` on any script for the full knob list (lr, clip, entropy, gamma/lam,
minibatch, etc.). Progress bars (tqdm) show update/epoch/round progress and ETA;
the detailed per-step log lines are preserved via `tqdm.write`.

### Performance — rollout is CPU-bound, so it runs multi-process

~99% of wall-clock is the Python AEC loop + heuristic pathfinding (the PPO
update itself is ~1% on GPU). So collection is parallelised across processes
with `-j/--num-workers` (default = CPU cores − 1). Each worker owns its own env
and a CPU copy of the policy; the parent broadcasts fresh weights each update and
gathers the trajectories. The pool persists across updates (spawn cost paid
once). A faster GPU barely helps; **more/faster CPU cores is the lever** —
near-linear speedup until you saturate cores.

```bash
python ae_rl/train_stage2_ppo.py -j 16        # 16 parallel rollout workers
python ae_rl/train_stage1_bc.py  -j 16        # parallel teacher collection too
python ae_rl/train_stage2_ppo.py -j 1         # force serial (debugging)
```

## Benchmarking

`benchmark.py` pits the learned policy against the heuristic and prints mean
per-agent reward, with a same-seed `6×heuristic` reference so you can see
whether RL actually beats the baseline it was cloned from.

```bash
python ae_rl/benchmark.py                         # newest stage checkpoint, 1 RL vs 5 heuristic
python ae_rl/benchmark.py --ckpt ae_rl/checkpoints/stage3_league.pt --rounds 50 --learners 3
```

## Deploying a trained policy

Inference runs through `ae/src/rl_policy.py` (`RLPolicy`, implements the same
`Policy` interface as the heuristic). The network there mirrors `model.py`
exactly so checkpoints load directly.

1. Copy a checkpoint into the image source: `cp ae_rl/checkpoints/stage3_league.pt ae/models/ae_rl.pt`
2. Wire it in `ae/src/ae_manager.py` (left un-wired so the heuristic stays
   production):
   ```python
   from rl_policy import RLPolicy
   ...
   self._policy = policy or RLPolicy()          # device="cpu" by default
   ```
3. `torch` (CPU build) is already added to `ae/requirements.txt`; the slim
   Docker base is sufficient. See `ae/Dockerfile` for the GPU-base swap if ever
   needed.

## Files

| File | Role |
|------|------|
| `common.py`              | path bootstrap, device, seeding, obs → arrays / scalars |
| `model.py`               | `RecurrentMaskableActorCritic` (act / sequence eval), checkpoint I/O |
| `controllers.py`         | `HeuristicController` (teacher/opponent), `NetController` (frozen league opponent) |
| `rollout.py`             | AEC self-play collector + GAE; BC teacher-dataset collector |
| `ppo.py`                 | recurrent PPO update (clipped policy + value, entropy, masked) |
| `train_stage1_bc.py`     | Stage 1 — behaviour cloning |
| `train_stage2_ppo.py`    | Stage 2 — PPO vs heuristic |
| `train_stage3_league.py` | Stage 3 — PPO league self-play |
| `benchmark.py`           | evaluate a checkpoint vs the heuristic baseline |
| `../ae/src/rl_policy.py` | deployment-side `Policy` wrapper for the live AE server |
```
