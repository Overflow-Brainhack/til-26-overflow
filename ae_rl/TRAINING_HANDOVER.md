# AE RL Training Handover

This note records reported training/evaluation results and related checkpoint
state from the Stage 2 / Stage 2.5 RL experiments. It does not interpret the
results.

## Checkpoint And Deploy State

- Runtime RL checkpoint path in `ae/src/rl_policy.py`: `models/stage2_ppo.pt`.
- AE Dockerfile copies `ae/models/` into the image with `COPY models/ models/`.
- `ae/src/ae_manager.py` defaults to `RLPolicy()`.
- `ae/test_env/auto_play.py` supports `--agent-type rl` and `--rl-checkpoint`.
- `rl_autorun.py` stages an AE checkpoint into `ae/models/stage2_ppo.pt` before AE build/submit.
- Default `rl_autorun.py` checkpoint source: `ae_rl/checkpoints/stage2_ppo_best.pt`.
- Alternative `rl_autorun.py` source modes:
  - `RL_AUTORUN_CHECKPOINT=best`
  - `RL_AUTORUN_CHECKPOINT=current`
  - `RL_AUTORUN_CHECKPOINT=<explicit .pt path>`

## Training Script Changes In Use

- `ae_rl/train_stage2_ppo.py`
  - supports stochastic heuristic opponents.
  - supports `--advanced-prob`.
  - supports validation checkpointing to `ae_rl/checkpoints/stage2_ppo_best.pt`.
  - supports per-run snapshots under `ae_rl/checkpoints/stage2_snapshots/<timestamp>/`.
  - supports `--learner-slots agent_X`.
- `ae_rl/benchmark.py`
  - defaults to deterministic RL actions.
  - rotates learner slots by default.
  - prints learner slot IDs in per-round output.
- `ae_rl/rollout.py`
  - reports learner return `min`, `max`, and `sd`.
  - supports targeted learner slots.
- `ae_rl/controllers.py`
  - has deterministic and stochastic heuristic opponent specs.
  - uses novice map cache only for novice games.

## Reported Runs And Results

### Early Stage 2 PPO

User report:

- First approximately 130 updates had returns around `-20` to `-10`.
- Last approximately 20 updates were positive, up to about `20`.

Benchmark summary reported after that run:

```text
RL agents           -57.30 +/- 86.08
Heuristic in-game   258.53 +/- 98.93
Heuristic baseline  231.50 +/- 107.49
Delta vs baseline  -288.80
```

### Stage 2 PPO, Longer Run With `learners=4`

Representative final log lines:

```text
upd 192/200  ret=105.9  opp=363.8  max=359.0
upd 193/200  ret=118.7  opp=332.6  max=538.0
upd 194/200  ret= 96.2  opp=384.9  max=340.0
upd 195/200  ret=115.4  opp=339.7  max=437.0
upd 196/200  ret=108.4  opp=344.6  max=320.0
upd 197/200  ret= 96.2  opp=386.0  max=358.0
upd 198/200  ret=111.4  opp=352.6  max=313.0
upd 199/200  ret=103.7  opp=357.7  max=362.0
upd 200/200  ret=124.2  opp=331.7  max=408.0
```

20 benchmark rounds were pasted after this run. The pasted per-round RL means
included values from `-14.3` to `173.0`. No final summary line was pasted for
that benchmark in the chat.

### Stage 2 PPO, Conservative `learners=3` Run

Representative final log lines:

```text
upd 147/150  ret= 98.1  opp=306.6  max=298.0
upd 148/150  ret=118.5  opp=293.0  max=451.0
upd 149/150  ret=141.4  opp=265.7  max=382.0
upd 150/150  ret=115.2  opp=291.5  max=417.0
```

Benchmark summary reported after that run:

```text
RL agents            89.81 +/- 109.67
Heuristic in-game   298.98 +/- 111.13
Heuristic baseline  231.50 +/- 107.22
```

### Stage 2 PPO, Actual Scoring Result

User reported actual scoring:

```text
Score: 0.294
```

Related training log line reported:

```text
upd 187/200  ret=112.7  opp=257.8  max=358.0
```

### Stage 2.5 PPO With Stochastic Heuristic Opponents

Example early log lines from a 600-update run:

```text
upd  2/600  ret=124.2  opp=287.5  min= -84.0  max=559.0  sd=109.5
upd  3/600  ret=127.0  opp=267.0  min=-110.0  max=429.0  sd=115.6
upd  4/600  ret=106.5  opp=293.5  min= -77.0  max=338.0  sd= 97.4
upd  5/600  ret=128.1  opp=281.4  min=-110.0  max=368.0  sd=110.9
upd  6/600  ret=128.7  opp=275.3  min=-146.0  max=408.0  sd=102.7
upd  7/600  ret=132.2  opp=283.2  min=-109.0  max=431.0  sd=107.5
upd  8/600  ret=136.1  opp=262.8  min= -74.0  max=448.0  sd=113.9
upd  9/600  ret=138.6  opp=266.5  min=-102.0  max=568.0  sd=126.4
upd 10/600  ret=122.2  opp=278.6  min=-195.0  max=416.0  sd=120.4
```

Later log range from the same training phase:

```text
upd  51/600  ret=129.4  opp=276.2  min= -82.0  max=395.0  sd=102.4
upd  60/600  ret=126.7  opp=272.4  min= -79.0  max=427.0  sd=102.4
upd  70/600  ret=136.7  opp=276.3  min= -84.0  max=470.0  sd=118.7
upd  80/600  ret=153.5  opp=259.8  min= -59.0  max=474.0  sd=104.5
upd  90/600  ret=134.1  opp=276.9  min=-112.0  max=421.0  sd=115.3
upd 100/600  ret=141.5  opp=264.4  min=-127.0  max=341.0  sd=101.7
```

Snapshot recorded:

```text
ae_rl/checkpoints/stage2_snapshots/20260522_185458/stage2_update_0100.pt
```

### Stage 2.5 Validation Metadata

Checkpoint metadata inspected for `stage2_ppo_best.pt`:

```text
stage: ppo_vs_diverse_heuristic_best
update: 50
validation_score: -87.16666666666666
validation_rl_mean: 144.33333333333334
validation_heur_baseline: 231.5
```

Checkpoint metadata inspected for `stage2_ppo.pt` at one point:

```text
stage: ppo_vs_heuristic
update: 100
learner_return_mean: 154.23958333333334
```

Validation output reported at update 100 of another run:

```text
upd 100/400  ret=154.2  opp=239.4  min=-108.0  max=385.0  sd=100.1
[val] score=-98.8  rl=132.7  heur=231.5  suites=1
```

Snapshot recorded:

```text
ae_rl/checkpoints/stage2_snapshots/20260522_201751/stage2_update_0100.pt
```

### Deterministic Benchmarking After Benchmark Patch

`stage2_ppo_best.pt`, `learners=3`, 5 rounds:

```text
RL agents           110.33 +/- 119.42
Heuristic in-game   312.33 +/- 145.70
Heuristic baseline  231.50 +/- 108.87
Delta vs baseline  -121.17
```

`stage2_ppo_best.pt`, `learners=1`, 5 rounds:

```text
round scores: 43.0, 89.0, 272.0, 250.0, 385.0
```

No final summary line was pasted for that 5-round run in the chat.

`stage2_ppo.pt`, approximately update 74, `learners=1`, 5 rounds:

```text
RL agents           179.60 +/- 53.78
Heuristic in-game   215.08 +/- 130.59
Heuristic baseline  231.50 +/- 108.87
Delta vs baseline   -51.90
```

`stage2_ppo.pt`, approximately update 100, `learners=1`, 50 rounds:

```text
RL agents           197.26 +/- 115.07
Heuristic in-game   219.61 +/- 106.40
Heuristic baseline  231.50 +/- 107.22
```

The printed 50-round sequence repeated a 6-slot deterministic pattern:

```text
-29.0, 244.0, 249.0, 280.0, 163.0, 299.0
```

### Best Reported Evaluation And Benchmark

User reported best actual eval:

```text
Score: 0.433
Speed: 0.845
```

Benchmark associated with that checkpoint:

```text
RL agents           247.42 +/- 88.53
Heuristic in-game   208.35 +/- 123.98
Heuristic baseline  231.50 +/- 108.56
```

User also reported that this best eval checkpoint was worse than some
intermediate steps. No numeric intermediate eval score was pasted in the chat.

### Advanced-Probability Training

Reported advanced-probability experiments:

- `advanced-prob 0.05` was discussed as a post-plateau fine-tune setting.
- `advanced-prob 0.10` was started.
- The `advanced-prob 0.10` run was stopped before a numeric summary was pasted.

### Targeted Weak-Slot Training

Targeted weak-slot training was added via:

```text
--learner-slots agent_X
```

User reported:

```text
After 30 updates, the weak-slot training had not changed the score pattern.
```

No numeric summary for targeted weak-slot training was pasted in the chat.

### Stage 3 League Training

No completed Stage 3 league training result was pasted in the chat.

