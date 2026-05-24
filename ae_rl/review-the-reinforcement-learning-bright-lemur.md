# Review: RL Strategy in `ae/` and `ae_rl/`

## Context

This is a review (not an implementation task) of the RL strategy that the TIL-26 AE submission uses, and the training pipeline that produces it. The goal is to summarise the approach, flag concrete problems, and propose targeted improvements that map to the AE Bomberman challenge described in [til-26-ae/AE-with-til_environment.md](til-26-ae/AE-with-til_environment.md).

The review is written assuming the reader has seen the source but wants a fast audit of correctness, risk, and headroom.

---

## 1. Strategy summary

### Algorithm
- **PPO** with maskable, recurrent actor-critic. Pure PyTorch — no Stable-Baselines3 / RLlib.
- **Net**: two 2-layer 3×3 CNNs over `agent_viewcone (25×7×5)` and `base_viewcone (25×S×S)`, a 3rd CNN over a 6-channel static map built from a shared `MapMemory`, plus an MLP over a 14-dim scalar vector (one-hot direction + normalised location, base_location, health, frozen_ticks, base_health, team_resources, team_bombs, step). All fused → 256-dim feature → **GRU(256, 1 layer)** → actor (6 logits) + critic (1). ~1.1M params. See [ae/src/rl_policy.py:75-131](ae/src/rl_policy.py#L75-L131) and [ae_rl/model.py](ae_rl/model.py).
- **Action masking** applied to logits before `Categorical.sample()` at both rollout and PPO loss time; deterministic argmax at inference (`deterministic=True` default in [ae/src/rl_policy.py:161](ae/src/rl_policy.py#L161)).

### Training curriculum (3 stages)
1. **Stage 1 — Behaviour cloning** ([ae_rl/train_stage1_bc.py](ae_rl/train_stage1_bc.py)). Warm-start from the heuristic teacher (`EditedHeuristicPolicyV2`).
2. **Stage 2 — PPO vs heuristic** ([ae_rl/train_stage2_ppo.py](ae_rl/train_stage2_ppo.py)). `n_learners=3` of 6 FFA agents learn, the other 3 are the production heuristic (optionally jittered). 10-update critic warm-up, then 150 PPO updates × 8 episodes × 200 steps × 3 learners ≈ **720k steps total**. GAE γ=0.99, λ=0.95, clip=0.2, entropy=0.01, LR=2.5e-4, Adam, advantage normalisation only.
3. **Stage 3 — League self-play** ([ae_rl/train_stage3_league.py](ae_rl/train_stage3_league.py)). Same loop but opponents are sampled from a pool of frozen prior checkpoints + heuristic. Scaffolded but `TRAINING_HANDOVER.md` notes it was never finished at scale.

### Reward signal
- **No custom shaping.** Training consumes the default env rewards verbatim (mission/recon/resource tiles, attack damage ± defender penalty, kill bonus, base destruction ±50, no step / stationary / wall-collide penalties). See [rollout.py:185-216](ae_rl/rollout.py#L185-L216).
- Episodes always run to truncation (`num_iters=200`); GAE bootstraps with the model's own value estimate at cutoff rather than treating cutoff as terminal — this is correct ([rollout.py:128-142](ae_rl/rollout.py#L128-L142)).

### Inference path (production)
- [ae/src/ae_manager.py:83](ae/src/ae_manager.py#L83) constructs **`RLPolicy()`** — the heuristic alternatives at lines 81-82 are commented out.
- `RLPolicy` re-creates per `/reset` or `step==0`; carries GRU hidden state across the 200 ticks of a round.
- The shared `MapMemory` singleton feeds the static-map channel and survives `/reset`, which lets novice-mode wall knowledge persist round-to-round (good).

---

## 2. Critical issues

### A. **The server will crash at startup.** *(severity: blocking)*
- `RLPolicy.__init__` unconditionally calls `torch.load("models/stage2_ppo.pt")` at [ae/src/rl_policy.py:166](ae/src/rl_policy.py#L166).
- **`ae/models/` does not exist.** No `.pt` files exist anywhere in the repo — `ae_rl/checkpoints/` is also empty.
- The outer `try/except` in [ae_manager.py:107-114](ae/src/ae_manager.py#L107-L114) only catches per-step exceptions inside `ae()` — it does **not** protect `AEManager.__init__`, where `RLPolicy()` is instantiated. Result: `FileNotFoundError` on the very first `/reset` or `step==0` observation.
- Heuristic fallback is one-line-comment away ([ae_manager.py:82](ae/src/ae_manager.py#L82)) but presently dead.

### B. **No reward shaping → sparse-reward exploration problem.**
- Default rewards give the agent ~0.1/step passively, +1–5 from picking up tiles, but the big signals (`attack_kill=15`, `destroy_enemy_base=50`, `own_base_destroyed=-50`) are exceptionally rare events early in training. From a randomly-initialised policy this is a textbook hard-exploration setup.
- The BC warm-start mitigates this — but Stage 2 then has to *improve on the heuristic* using nothing but raw env reward, which `TRAINING_HANDOVER.md` already notes only got marginal results (~247±89 RL vs ~231±109 heuristic mean episode return — likely within noise).
- The config explicitly leaves `agent_collide_wall`, `step_penalty`, `stationary_penalty`, `invalid_action`, `destroy_wall` as 0 — these are *shaping slots intended for competitors to fill in*. None are.

### C. **Self-play roster is biased toward the heuristic.**
- Stage 2 default = 3 RL + 3 heuristic. The heuristic is well-tuned and aggressive (`base_route_weight=100`, `predictive_bomb=True`, `auto_tune_bomb=True`). The learner, BC-cloned *from* that same heuristic, then re-learns to mimic it. Hard to discover novel strategy under that pressure.
- Stage 3 was the intended fix (frozen snapshots add diversity) but is unfinished.

### D. **Training compute budget is small for this domain.**
- ~720k steps total at default settings. Typical PPO benchmarks on grid-world / Bomberman-scale envs use 10M+. The PPO update is GPU but rollouts are CPU-only (Python AEC loop + heuristic pathfinding), and `default_workers() = cpus-1`. Wall-clock is the bottleneck.

### E. **Likely novice-map overfit.**
- `train_stage2_ppo.py` defaults to `--novice` (fixed map every episode) and `--advanced-prob 0.0`. The challenge doc says qualifiers run the **default** config, which has `novice: false` ([AE-with-til_environment.md#L142](til-26-ae/AE-with-til_environment.md#L142)). A static-map CNN trained on one map shape will likely not generalise.
- The novice-map cache is also injected into the learner's `MapMemory` ([rollout.py:46-48](ae_rl/rollout.py#L46-L48)) — at advanced inference there is no such cache, so the static-map channel will be much sparser than what the network saw in training.

### F. **No reward / return normalisation in PPO.**
- Only advantage normalisation at batch level ([rollout.py:348](ae_rl/rollout.py#L348)). Episode returns vary by ~±150 depending on whether a base is destroyed; without return normalisation the value loss dominates early updates and the entropy schedule has to soak up that variance. Standard fix is a running mean/std on returns.

### G. **`torch.load(..., weights_only=False)` is unnecessary.** ([rl_policy.py:166](ae/src/rl_policy.py#L166))
- The checkpoint is only used to read `state_dict` + a small `arch` dict — both are loadable with `weights_only=True`. Current code allows arbitrary code execution during a checkpoint load, which is a soft footgun for a contest submission downloading artifacts from anywhere.

### H. **Validation defaults off.**
- `--validate-every 0` is the default ([train_stage2_ppo.py:112](ae_rl/train_stage2_ppo.py#L112)). Without it, `stage2_ppo_best.pt` is never produced, the rollback-on-regress safety net never fires, and the saved "latest" checkpoint can be a worse policy than one from 20 updates earlier.

---

## 3. Smaller issues / smells

- **`mb.any()` guard** at [rl_policy.py:126-127](ae/src/rl_policy.py#L126-L127): if every action is masked, the code does *not* apply the mask and samples from raw logits. The earlier `mask.sum() == 0 → STAY` short-circuit at line 200 catches this, but if the upstream guard is ever moved, the asymmetry will bite.
- **Stochastic vs deterministic at inference**: `RLPolicy(deterministic=True)` default is correct for evaluation. Worth confirming nothing constructs it with `deterministic=False`.
- **`base_view` shape mismatch tolerance** in `_fix()` ([rl_policy.py:231-238](ae/src/rl_policy.py#L231-L238)) silently zero-pads — if the env changes `vision_radius`, the network gets a partly-zero input rather than an error. Useful for robustness but worth a warning log.
- **Stage 1 BC**: ensure the heuristic teacher (`EditedHeuristicPolicyV2` with `DEFAULT_POLICY_KWARGS` from `ae_manager.py`) is the same one used at inference. A divergence between teacher config and runtime heuristic config invisibly biases the BC target distribution.
- **`COPY models/ models/` in the Dockerfile** will fail the build if `ae/models/` is missing. Check whether the Dockerfile actually has this line — if so it's another way the current state breaks evaluation.

---

## 4. Recommended improvements (ranked by ROI for the qualifier)

### Must do before the qualifier
1. **Wire a fallback at policy construction.** In `ae_manager.py`, wrap `RLPolicy()` in `try/except` and fall back to `HeuristicPolicy(**DEFAULT_POLICY_KWARGS)` if the checkpoint load fails. Heuristic is the safe default; RL only ever runs once a checkpoint is genuinely present.
2. **Decide: ship heuristic or ship RL.** The handover notes already say Stage 2 only marginally beats the heuristic and Stage 3 was never finished. Default to heuristic for the qualifier unless a validated checkpoint clearly wins; do not deploy an unvalidated checkpoint just because RL is wired.
3. **Train on the qualifier's actual config.** Default `default_config()` has `novice=False`. Either train with `--advanced` or set `--advanced-prob 1.0`. If you must use novice for compute reasons, validate explicitly against advanced before shipping.

### High-value (≤1 day each)
4. **Add reward shaping.** See dedicated section §6 below for a full menu.
5. **Reward / return normalisation in PPO.** Running mean/std on `returns` before computing the value loss. Standard PPO knob, ~10 lines of code in [ppo.py](ae_rl/ppo.py).
6. **Diversify Stage 2 opponents.** Default-on `--stochastic-heuristic-prob 0.3` and bump `--stochastic-jitter` to ~0.5 so the learner sees a wider behaviour distribution than the single fixed heuristic.
7. **Always validate.** Set `--validate-every 5` (and a reasonable rounds/seed) as the default. The best-checkpoint + rollback safety net is already in place at [train_stage2_ppo.py:224-258](ae_rl/train_stage2_ppo.py#L224-L258); just enable it.
8. **`weights_only=True`** on the load + drop the `arch` dict from the file by storing arch as a small `json` next to the `.pt`. Tiny change, real safety win.

### Medium-value (multi-day)
9. **Finish Stage 3.** League with ~10-20 snapshots gives meaningful opponent diversity and tends to lift PPO above heuristic-mimicry in this class of game.
10. **Scale rollout throughput.** Bigger `num_workers`, `episodes-per-update`, total updates. 720k steps is the ceiling on what the current setup can extract from PPO.
11. **Curriculum on `n_learners`.** Start `--learners 1` for the first ~30 updates (clean signal: 1 RL vs 5 heuristic) then ramp to 3+ so credit assignment improves before opponent density does.
12. **Replace static-map CNN with positional encoding + global pooling**, or attention over viewcone channels. The current 16×16×6 grid → 2-layer CNN may under-fit map structure; a small per-tile MLP + mean/max pool is robust to absent cache and still cheap.

---

## 5. Files to look at when acting on this review

- [ae/src/ae_manager.py](ae/src/ae_manager.py) — line 83 wiring; needs the fallback wrap.
- [ae/src/rl_policy.py](ae/src/rl_policy.py) — checkpoint load (line 166), arch defaults, action-mask handling.
- [ae/src/edited_policy_v2.py](ae/src/edited_policy_v2.py) — the production-quality heuristic that should be the safe default.
- [ae_rl/train_stage2_ppo.py](ae_rl/train_stage2_ppo.py) — defaults for novice flag, validation, opponent mix.
- [ae_rl/rollout.py](ae_rl/rollout.py) — reward attribution, GAE, advantage normalisation.
- [ae_rl/ppo.py](ae_rl/ppo.py) — where return normalisation would go.
- [ae_rl/common.py](ae_rl/common.py) — scalar normalisation constants; must stay in lockstep with `rl_policy.py` constants if either changes.
- [ae/requirements.txt](ae/requirements.txt) — confirms `torch` is shipped CPU-only; check the version matches what `ae_rl/` trains on.
- [ae/Dockerfile](ae/Dockerfile) — confirm whether `COPY models/ ...` will silently fail without a checkpoint present.

---

## 6. Reward shaping — brainstorm

### Why shape at all?
The default reward distribution is dominated by two rare events: `destroy_enemy_base = +50` and `own_base_destroyed = -50`. With 6 teams on a 16×16 grid and at most 3 bombs starting per team, most 200-step episodes will see zero base destructions, so the learner is left with sparse +5/+1/+2 collect signals and +1×damage / +30 kill events. PPO with a random init has very little to anchor its critic on, and even the BC warm-start drifts quickly when the only fine-tuning signal is the same noisy stream. The point of shaping is to densify the per-step signal *without* changing the optimal policy — or, when we accept changing it, to do so in a direction that mirrors what good play looks like.

Two implementation surfaces are available:

- **Surface A — cfg overrides.** The env already accumulates reward through `cfg.rewards.<event_type>` events ([bomberman_config.yaml:63-77](til-26-ae/til_environment/bomberman_config.yaml#L63-L77), [events/rewards.py](til-26-ae/til_environment/events/rewards.py)). Any of the seven 0-valued slots (`agent_collide_wall`, `agent_collide_agent`, `destroy_wall`, `step_penalty`, `stationary_penalty`, `invalid_action`, `truncation`) plus tweaks to the non-zero ones can be applied with one `OmegaConf.merge` in [ae_rl/rollout.py::make_env()](ae_rl/rollout.py). Zero code change to the env. **Lowest-risk, highest-leverage**.
- **Surface B — rollout shaping hook.** For anything the env doesn't already emit (potentials, novelty, distance deltas), inject a `RewardShaper` callable in [rollout.py:185](ae_rl/rollout.py#L185), right where `env._cumulative_rewards[agent]` is read. Needs prev_obs/curr_obs bookkeeping per learner and resets on `env.reset()`. Slightly more code; preserves env API.

### Menu of shaping signals

Use these as a buffet, not a checklist. Apply 2-4 at a time and ablate; piling them all on at once *will* over-constrain the policy.

#### Group 1 — fill in the env's existing zero slots (Surface A)
| Slot | Suggested value | Intent | Risks / notes |
| ---- | --------------- | ------ | ------------- |
| `step_penalty` | `-0.02` | Discourages stalling, idle bobbing, and 200-step "do nothing" trajectories that the GAE bootstrap may otherwise tolerate. | Too negative → suicide rushes into walls (because dying then respawning ends the per-step cost early). Keep magnitude ≪ per-tile reward. |
| `stationary_penalty` | `-0.05` | Targets `STAY` specifically; useful because `STAY` is legitimate when dodging but cheap padding otherwise. Stacks with `step_penalty`. | Set to 0 once the policy stops camping; high values destroy legitimate "stand still while bomb blocks the lane" plays. |
| `agent_collide_wall` | `-0.1` | The action mask should already prevent walking into walls — this is *redundancy* for the network's own logit distribution, not behaviour shaping. | Largely cosmetic if masking works; useful early in BC drift recovery. |
| `agent_collide_agent` | `-0.05` | Penalises wasted turns trying to push through teammates/opponents. | Could over-discourage close combat / chokepoint blocking. |
| `invalid_action` | `-0.5` | Hard "you wrote a logit that violated the mask" signal. | Should never fire if masking is correct — treat the rate of this firing as a *diagnostic* signal more than a behavioural one. |
| `destroy_wall` | `+0.3` | Rewards opening map structure — useful if the heuristic baseline rarely breaks walls and the RL agent is BC'd to copy that. | Encourages bomb-spam on walls just for the points. Pair with a higher `bomb_cost` if you turn this on. |
| `truncation` | `0.0` (leave) | Awarding here would bias every learner uniformly — it's the same flat signal every game. Don't bother. | — |

#### Group 2 — re-weight the *non-zero* defaults (Surface A)
| Event | Default | Why you might change it |
| ----- | ------- | ----------------------- |
| `collect_recon` | 1.0 | Bump to 2.0 if the policy underweights cheap-but-frequent tile pickup. The current heuristic in `EditedHeuristicPolicyV2` already routes for these — re-weighting helps RL find the same route. |
| `collect_resource` | 2.0 | Could lift to 3.0 to encourage resource economy. But note `team_resources` auto-converts to bombs at 1.5, so collected resources also indirectly improve `team_bombs` — there's already a multiplier on the offensive arc. |
| `attack_damage` | 1.0× | The defender penalty is symmetric, so this is already zero-sum per damage event in self-play — meaningless on net for the learner across episodes. Bumping it makes attack/defence both *more* salient (good for exploration), but raises gradient variance. |
| `attack_kill` | 30.0 (yaml) / 15 (doc) | YAML is authoritative. Already very high; further bumping makes the policy lottery-tickety. Leave. |
| `destroy_enemy_base` | 50.0 | Same caveat — bumping further makes the rare positive episode dominate batches and starves the critic of mid-game signal. **Consider lowering to 25 during early training** so per-episode return variance shrinks; restore to 50 in later stages. |

#### Group 3 — potential-based shaping (Surface B, PBRS-safe)
PBRS ([Ng, Harada, Russell 1999](https://www.cs.cornell.edu/courses/cs6700/2007fa/papers/ng-PolicyInvariance.pdf)) preserves the optimal policy when the shaped reward is `F(s, s') = γ·Φ(s') − Φ(s)` for some state potential `Φ`. Implementation in our case: at each learner step, compute Φ from `obs` and emit `(γ·Φ(s') − Φ(s)) · α` as an additive reward. The `γ` here must match the trainer's `args.gamma = 0.99`.

| Potential `Φ` | Definition | Comment |
| ------------- | ---------- | ------- |
| **Distance-to-nearest-tile potential** | `Φ(s) = −w · min_dist_to_any_known_tile(mission/recon/resource)` using Manhattan distance over the `MapMemory`. | Densest, easiest signal. Falls to 0 when the agent has nothing to chase. Tune `w ≈ 0.2` so a 1-step approach is worth ~0.04, well below `collect_recon=1.0`. |
| **Distance-to-enemy-base potential** | `Φ(s) = −w · min_dist_to_any_known_enemy_base(s)` weighted by remaining `team_bombs > 0`. | Aggressive shaping; only meaningful if the agent has bombs to use. Gate on `team_bombs >= 1`. |
| **Distance-from-bomb-blast potential** | `Φ(s) = +w · min_dist_to_any_armed_bomb_in_range(s)` for bombs with `timer ≤ 2`. | Survival shaping. Pairs well with the kill/freeze penalty below. |
| **Distance-from-own-base potential** | `Φ(s) = −w · dist_to_own_base(s) · I[base_health < threshold]` — only active when own base is in danger. | Defensive shaping; can backfire by camping the base when no threat exists, which is why the indicator gate matters. |

Cheap to implement: `MapMemory` already tracks visible tiles and base positions; you compute Manhattan distance in O(known_tiles). The shaping cost per step is O(grid_size²) worst case, negligible.

#### Group 4 — non-PBRS task shaping (Surface B; will bias the policy)
These are *not* PBRS-safe — they may shift the optimal policy. Use only if you accept that and anneal them out before the policy is finalised.

| Signal | Definition | Why try it |
| ------ | ---------- | ---------- |
| **Exploration / novelty bonus** | `+β` (e.g. 0.05) per *previously-unobserved* grid cell that newly enters the agent's viewcone. Track via `set` of visited (x,y) in the MapMemory or an episode-local bitmap. | Cheap "intrinsic curiosity" substitute. Strongest effect in novice (fixed map) where exploration is one-shot per episode. Anneal `β` to 0 by ~70% of training. |
| **State-visit count bonus** | `+β / sqrt(N(s))` where `s = (x, y, direction)`. Reset per episode. | Classic count-based exploration. Works on a 16×16 grid because state space is small. |
| **Bomb placement scoring** | At `PLACE_BOMB`, add a reward proportional to `predicted_damage = Σ enemies_in_blast · expected_dmg + walls_in_blast · 0.3 + base_in_blast · 5`. | Teaches "good bombs vs spam bombs." Risk: if your prediction is wrong (e.g. enemy moves away) you've taught the network the bomb was good when it wasn't. Use only if you can compute `predicted_damage` deterministically from the *current* env state. |
| **Suicide penalty** | `-3.0` once when the agent's HP drops to 0 *and* the killing bomb was placed by this agent. | Specifically deters self-bombing. The env's default already gives the agent the symmetric attack_damage penalty (so −20 implicit), but a one-off explicit signal is sharper. |
| **Resource-threshold milestone** | `+0.5` each time `team_resources` crosses an integer multiple of `bomb_cost = 1.5`. | Tells the value head "you're closer to a usable bomb." Mild; only helps in resource-starved openings. |
| **Survival per-step** | `+0.01` per step where the agent is non-frozen. | Counter-balances `step_penalty` for the "stay alive" arc. Net should still be slightly negative so the agent doesn't camp forever. |
| **Asymmetric base-damage multiplier** | Wrap the `attack_damage` event so damage *to bases* gets a 2× multiplier on the attacker side. | The base is the only thing the agent ultimately needs to destroy — damaging it should look much more valuable than chipping an opposing agent. Implement via the `multiplier` arg in `Rewards.award()` ([events/rewards.py:76-92](til-26-ae/til_environment/events/rewards.py#L76-L92)). |

#### Group 5 — scheduling / curriculum on the shaping itself
- **Linear anneal**: `α(t) = max(0, 1 − t / T_anneal)` with `T_anneal ≈ 0.7 · total_updates`. Apply as a scalar multiplier on the *whole* Surface-B shaping reward. Ensures the late-stage policy is optimising the real env reward.
- **Stage-gated shaping**: Stage 1 (BC) — no shaping. Stage 2 first 30 updates — full shaping (Groups 1+3). Stage 2 mid — Group 3 only. Stage 3 — Group 1 only (cfg-level penalties).
- **Per-event annealing**: novelty bonus anneals fast (it's a "warm-up" signal); distance-to-tile potential can stay on longer (it's PBRS-safe and policy-invariant, so it costs little to leave in).
- **Anti-Goodhart audit**: every ~20 updates, compute the per-event reward breakdown for an episode (env reward vs each shaping term). If shaping dominates, your `α` is too high. Target ≤ 30 % of total reward from shaping at any given update.

### Recommended starter mix
If you just want one configuration to try first:

**cfg overrides** (Surface A):
```python
cfg = OmegaConf.merge(cfg, {"rewards": {
    "step_penalty": -0.02,
    "stationary_penalty": -0.05,
    "invalid_action": -0.5,
    "destroy_wall": 0.3,
}})
```

**Shaping hook** (Surface B), applied with linearly-annealed `α(t)`:
- Distance-to-nearest-tile potential, `w=0.2` (PBRS).
- Distance-from-armed-bomb potential, `w=0.3` (PBRS).
- Novelty bonus, `β=0.05` per new viewcone cell, annealed to 0 by update 100.

This gives a dense, mostly-policy-invariant gradient signal for the first ~70 % of training, then cleanly hands off to the env's native reward by the time the league phase starts.

### What to *avoid* shaping
- **Don't reward "place_bomb at all"** — the heuristic baseline already does this and shaping it explicitly biases the policy toward spam rather than placement quality.
- **Don't reward "agent did the same thing the heuristic would have done"** — that's just hidden BC, and you already have a real BC stage.
- **Don't directly reward `team_bombs > 0`** — that's a side-effect, not a goal; rewarding it discourages spending bombs.
- **Don't reward `health > k`** — flat survival rewards encourage camping; use the potential-based dodge term instead.

---

## 7. Verification

This is a review; nothing to test. To validate any improvement that comes out of it:
- Run `python ae_rl/train_stage2_ppo.py --updates 30 --validate-every 5 --advanced` and confirm `stage2_ppo_best.pt` is produced with `meta.validation_score > 0`.
- Spin up the AE container with the new checkpoint, hit it with `test/test_ae.py`, and compare the resulting score against the heuristic-only baseline produced by reverting [ae_manager.py:83](ae/src/ae_manager.py#L83) back to the heuristic.
- If RL doesn't clearly beat heuristic on the **advanced** map, ship the heuristic.
