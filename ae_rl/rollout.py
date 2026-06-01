"""Self-play rollout collection over the PettingZoo AEC Bomberman env.

The env is turn-based (one agent acts per ``step``); a full game round only
executes when the last agent has submitted its action, at which point per-agent
rewards land in ``env._cumulative_rewards`` (zeroed at the start of each agent's
turn — exactly PettingZoo's ``last()`` semantics). So at agent *X*'s turn,
``env._cumulative_rewards[X]`` is the reward earned by *X*'s previous action.
We use that to attribute reward to the prior transition.

Agents never terminate in this game (they freeze + respawn); episodes always run
the full ``num_iters`` and end via truncation. Every learner trajectory is
therefore exactly ``num_iters`` steps long, so trajectories stack into a clean
(T, B) batch with no padding.

Collection is ~99% CPU-bound (the Python AEC loop plus the heuristic opponents'
pathfinding), so it is parallelised across processes: each worker owns its own
env and a CPU copy of the policy, runs a slice of the games, and ships back the
stacked numpy trajectories. The GPU is only used for the (cheap) PPO update in
the parent. Workers use the 'spawn' start method and never touch CUDA.
"""

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

# Stage 4 (evolution) ships 1 learner + K live-opponent state dicts per chunk,
# which is ~5x more CPU tensors per task than Stage 3. torch's default
# 'file_descriptor' sharing strategy opens an fd per shared CPU tensor and
# exhausts ulimit -n quickly under that traffic ("OSError: [Errno 24] Too
# many open files" during reduce_storage). 'file_system' uses filesystem
# entries instead — no fd ceiling. Set in the PARENT process; workers use
# spawn, and we re-set it in _sp_worker_init so child reductions match.
try:
    mp.set_sharing_strategy("file_system")
except (RuntimeError, AttributeError):
    # Some platforms (notably Windows) don't expose this. Harmless to skip.
    pass

import common  # noqa: F401  (path bootstrap)
from common import obs_to_arrays
from constants import Action
from controllers import _CACHE_TEMPLATE, build_controller
from global_state import (
    GLOBAL_GRID_SHAPE,
    GLOBAL_SCALAR_DIM,
    build_global_state,
)
from map_memory import MapMemory
from observation import parse_observation
from til_environment.bomberman_env import Bomberman
from til_environment.config import default_config


def _fresh_learner_memory(novice: bool) -> MapMemory:
    """Per-learner MapMemory. Novice gets the bundled static cache so it starts
    with the same map knowledge the heuristic has."""
    mem = MapMemory()
    if novice and _CACHE_TEMPLATE is not None:
        mem.merge_static_from(_CACHE_TEMPLATE)
    return mem


# ── PBRS potentials ──────────────────────────────────────────────────────────
# Φ(s) is decomposed into two policy-invariant terms; per-step shaping reward
# added to env reward is γ·Φ(s') − Φ(s). Off-policy invariance holds for any
# choice of Φ as long as it depends only on s (not on the action taken).
PBRS_TILE_WEIGHT = 0.2
PBRS_BASE_WEIGHT = 0.2


def _agent_xy(obs: dict) -> tuple[int, int]:
    loc = obs.get("location", (0, 0))
    arr = np.asarray(loc).flatten()
    if arr.size < 2:
        return (0, 0)
    return (int(arr[0]), int(arr[1]))


def _compute_phi(obs: dict, memory: MapMemory) -> float:
    """State potential Φ(s) = -w_tile·dist_to_nearest_known_tile
                              -w_base·dist_to_nearest_known_enemy_base  (if team_bombs >= 1).
    Manhattan distance over MapMemory contents. Returns 0 when nothing is known."""
    pos = _agent_xy(obs)
    phi = 0.0

    tiles = memory.collectible_cells()
    if tiles:
        d_tile = min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in tiles)
        phi += -PBRS_TILE_WEIGHT * d_tile

    team_bombs = obs.get("team_bombs", 0)
    try:
        team_bombs = int(np.asarray(team_bombs).flatten()[0])
    except Exception:
        team_bombs = 0
    if team_bombs >= 1:
        bases = memory.enemy_bases
        if bases:
            d_base = min(abs(pos[0] - b[0]) + abs(pos[1] - b[1]) for b in bases)
            phi += -PBRS_BASE_WEIGHT * d_base

    return phi


# ── reward shaping configuration (training-time only) ────────────────────────
# Env-level cfg overrides applied in make_env. Set to 0.0 / 1.0 to disable.
SHAPING_STEP_PENALTY = -0.02
SHAPING_STATIONARY_PENALTY = -0.05
SHAPING_INVALID_ACTION = -0.5
# Own base loss is treated as inevitable / out of the policy's control, so
# zero out the -50 default penalty. The bot still profits from breaking
# enemy bases (SHAPING_DESTROY_ENEMY_BASE_MULT), it just doesn't try to
# defend its own.
SHAPING_OWN_BASE_DESTROYED = 0.0

# Multipliers applied via a Rewards.award wrapper. Only boost positive
# attack_damage (= damage dealt to enemies/bases). Negative attack_damage
# (= damage taken / your base hit) is left at 1.0 so survival pressure is
# preserved unchanged.
SHAPING_ATTACK_DAMAGE_DEALT_MULT = 1.5
SHAPING_DESTROY_ENEMY_BASE_MULT = 2.0
SHAPING_ATTACK_KILL_MULT = 1.5
# Amplify damage *taken* (negative attack_damage) so the policy weighs survival
# more. >1.0 = stronger aversion. Pairs with the deploy-side dodge override.
SHAPING_ATTACK_DAMAGE_TAKEN_MULT = 2.0

# Per-step oscillation penalty: applied to the action that *completes* a
# 2-step (action, position) cycle — i.e. the LEFT/RIGHT/LEFT/RIGHT shake the
# RL policy falls into. Bookkept on the learner side in
# ``_collect_selfplay_episodes`` so it doesn't depend on the env at all.
SHAPING_OSCILLATION_PENALTY = -0.25
# Rolling history depth fed into the loop check; matches the deploy-side
# LayeredRLPolicy default so train/eval signals are consistent.
OSCILLATION_WINDOW = 6
_OSCILLATION_PERIODS = (2, 3)

# ── turn-spam penalty ────────────────────────────────────────────────────────
# Separate from oscillation: penalises *any* sustained turning without forward
# progress, including unidirectional spins (LEFT, LEFT, LEFT, ...) that the
# oscillation matcher misses. Applies to LEFT / RIGHT after they've been used
# this many consecutive turns without a FORWARD / BACKWARD breaking the streak.
TURN_SPAM_THRESHOLD = 2
SHAPING_TURN_SPAM_PENALTY = -0.75

# The env's own default own-base penalty (til_environment config.py). Used as the
# "raw eval reward" value when shaping is fully disabled.
ENV_DEFAULT_OWN_BASE = -50.0


@dataclass
class ShapingConfig:
    """Per-run reward-shaping configuration — each component independently
    toggleable so its effect on the REAL eval can be A/B'd in isolation.

    The field defaults reproduce the historical all-on shaping exactly, so
    ``ShapingConfig()`` == the old ``shape_rewards=True`` behaviour and
    ``ShapingConfig.from_master(False)`` == the old ``shape_rewards=False`` (raw
    env reward) behaviour. Anything that previously threaded a ``shape_rewards``
    bool still works; the bool is just mapped to one of those two presets.

    Components
    ----------
    offensive_multipliers : boost attack_damage / kill / base-destroy at award.
    env_penalties         : step / stationary / invalid-action cfg penalties.
    pbrs                  : potential-based shaping (γ·Φ(s')−Φ(s)).
    anti_oscillation      : the LEFT/RIGHT shake + turn-spam penalties.
    own_base_destroyed    : value for losing your own base. The eval charges
                            ``ENV_DEFAULT_OWN_BASE`` (−50); training has used 0.0
                            deliberately (defending isn't worth the lost offense),
                            but it's exposed so a small negative can be tried.
    """

    offensive_multipliers: bool = True
    env_penalties: bool = True
    pbrs: bool = True
    anti_oscillation: bool = True
    own_base_destroyed: float = 0.0
    # Tunable magnitudes (defaults = the historical module constants).
    attack_damage_dealt_mult: float = SHAPING_ATTACK_DAMAGE_DEALT_MULT
    attack_damage_taken_mult: float = SHAPING_ATTACK_DAMAGE_TAKEN_MULT
    destroy_enemy_base_mult: float = SHAPING_DESTROY_ENEMY_BASE_MULT
    attack_kill_mult: float = SHAPING_ATTACK_KILL_MULT
    step_penalty: float = SHAPING_STEP_PENALTY
    stationary_penalty: float = SHAPING_STATIONARY_PENALTY
    invalid_action: float = SHAPING_INVALID_ACTION

    @classmethod
    def from_master(cls, enabled: bool) -> "ShapingConfig":
        """Map the legacy ``shape_rewards`` bool to a config. ``True`` = all-on
        (own_base 0.0); ``False`` = raw eval reward (own_base −50, nothing shaped)."""
        if enabled:
            return cls()
        return cls(
            offensive_multipliers=False,
            env_penalties=False,
            pbrs=False,
            anti_oscillation=False,
            own_base_destroyed=ENV_DEFAULT_OWN_BASE,
        )

    def describe(self) -> str:
        on = [n for n, v in (("mult", self.offensive_multipliers),
                             ("envpen", self.env_penalties),
                             ("pbrs", self.pbrs),
                             ("antiosc", self.anti_oscillation)) if v]
        return f"shaping[{'+'.join(on) or 'none'} own_base={self.own_base_destroyed:g}]"


def _is_oscillating(history, action: int, pos: tuple[int, int]) -> bool:
    """True if appending (action, pos) would close a 2- or 3-step cycle.

    Same matcher as ``EditedHeuristicPolicy._is_loop`` / LayeredRLPolicy so
    train-time penalty and deploy-time guard agree on what counts as a loop.
    """
    entry = (action, pos)
    buf = list(history)
    n = len(buf)
    for period in _OSCILLATION_PERIODS:
        needed = 2 * period - 1
        if n < needed:
            continue
        suffix = tuple(buf[n - (period - 1):]) + (entry,)
        prev = tuple(buf[n - (2 * period - 1): n - (period - 1)])
        if suffix == prev:
            return True
    return False


def _wrap_offensive_rewards(env: Bomberman, shaping: "ShapingConfig") -> None:
    """Boost offensive reward events at award time (training only).

    Positive ``attack_damage`` (dealt) and ``destroy_enemy_base`` / ``attack_kill``
    are amplified to encourage aggression; negative ``attack_damage`` (taken) is
    amplified by ``shaping.attack_damage_taken_mult`` to teach damage aversion.
    """
    original_award = env.dynamics.rewards.award

    def award_boosted(recipient_id: str, event: str, multiplier: float = 1.0) -> float:
        if event == "attack_damage":
            if multiplier > 0:
                multiplier = multiplier * shaping.attack_damage_dealt_mult
            elif multiplier < 0:
                multiplier = multiplier * shaping.attack_damage_taken_mult
        elif event == "destroy_enemy_base":
            multiplier = multiplier * shaping.destroy_enemy_base_mult
        elif event == "attack_kill":
            multiplier = multiplier * shaping.attack_kill_mult
        return original_award(recipient_id, event, multiplier)

    env.dynamics.rewards.award = award_boosted


def _resolve_shaping(shape_rewards: bool, shaping: "ShapingConfig | None") -> "ShapingConfig":
    """A ``shaping`` config takes precedence; otherwise derive one from the
    legacy ``shape_rewards`` bool so every old call site keeps working."""
    return shaping if shaping is not None else ShapingConfig.from_master(shape_rewards)


def make_env(novice: bool = True, shape_rewards: bool = False,
             shaping: "ShapingConfig | None" = None) -> Bomberman:
    """Build a Bomberman env. Pass a ``ShapingConfig`` for per-component control,
    or the legacy ``shape_rewards`` bool (True = all-on, False = raw eval reward).
    Eval / benchmark / diagnostic callers leave both defaulted → raw env reward."""
    sh = _resolve_shaping(shape_rewards, shaping)
    cfg = default_config()
    cfg.env.novice = novice
    cfg.env.render_mode = None
    # Own-base penalty is always set explicitly: the eval default is −50, training
    # has used 0.0; either way we want the value the run asked for.
    cfg.rewards.own_base_destroyed = float(sh.own_base_destroyed)
    if sh.env_penalties:
        # Surface A — env-config-level shaping. Fills in otherwise-0 slots so eval
        # semantics are unchanged (the eval container builds its own env).
        cfg.rewards.step_penalty = sh.step_penalty
        cfg.rewards.stationary_penalty = sh.stationary_penalty
        cfg.rewards.invalid_action = sh.invalid_action
    env = Bomberman(cfg)
    if sh.offensive_multipliers:
        _wrap_offensive_rewards(env, sh)
    return env


def _make_env_pool(novice: bool = True, advanced_prob: float = 0.0,
                   shape_rewards: bool = True,
                   shaping: "ShapingConfig | None" = None) -> dict[bool, Bomberman]:
    """Build the training-side env pool. ``shape_rewards`` defaults True here
    because every caller is a training collector; eval/benchmark constructs its
    own envs via ``make_env`` directly with default raw reward."""
    sh = _resolve_shaping(shape_rewards, shaping)
    if novice and advanced_prob > 0.0:
        return {True: make_env(True, shaping=sh),
                False: make_env(False, shaping=sh)}
    return {novice: make_env(novice, shaping=sh)}


def _select_env(envs: dict[bool, Bomberman], advanced_prob: float, rng) -> Bomberman:
    if True in envs and False in envs:
        novice = not (rng.random() < advanced_prob)
        return novice, envs[novice]
    novice, env = next(iter(envs.items()))
    return novice, env


def _spec_for_map(spec: dict, novice: bool) -> dict:
    if spec.get("kind") not in {"heuristic", "stochastic_heuristic"}:
        return spec
    out = dict(spec)
    out["use_cache"] = novice
    return out


def default_workers() -> int:
    """Leave one core free for the parent / OS."""
    return max(1, (os.cpu_count() or 2) - 1)


@dataclass
class RolloutBatch:
    # All tensors are (T, B, …) on CPU; moved to device by the trainer.
    viewcone: torch.Tensor
    baseview: torch.Tensor
    scalars: torch.Tensor
    mask: torch.Tensor
    staticmap: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    # Privileged global state for the asymmetric critic — only populated when the
    # collector was built with ``collect_global_state=True``; otherwise None.
    global_grid: torch.Tensor | None = None
    global_scalars: torch.Tensor | None = None

    @property
    def num_seqs(self) -> int:
        return self.viewcone.shape[1]

    @property
    def has_global(self) -> bool:
        return self.global_grid is not None


def _new_trajectory() -> dict:
    # ``global_grid`` / ``global_scalars`` stay empty unless the collector was
    # asked for privileged state (asymmetric/CTDE training). Keeping the keys
    # present always lets _stack_trajectory branch on emptiness rather than on
    # a flag it would have to be threaded.
    return {k: [] for k in (
        "viewcone", "baseview", "scalars", "mask", "staticmap",
        "actions", "logp", "values", "rewards", "dones",
        "global_grid", "global_scalars",
    )}


def _stack_trajectory(traj: dict) -> dict:
    out = {
        "viewcone": np.stack(traj["viewcone"]).astype(np.float32),
        "baseview": np.stack(traj["baseview"]).astype(np.float32),
        "scalars": np.stack(traj["scalars"]).astype(np.float32),
        "mask": np.stack(traj["mask"]).astype(np.float32),
        "staticmap": np.stack(traj["staticmap"]).astype(np.float32),
        "actions": np.asarray(traj["actions"], dtype=np.int64),
        "logp": np.asarray(traj["logp"], dtype=np.float32),
        "values": np.asarray(traj["values"], dtype=np.float32),
        "rewards": np.asarray(traj["rewards"], dtype=np.float32),
        "dones": np.asarray(traj["dones"], dtype=np.float32),
    }
    if traj["global_grid"]:
        out["global_grid"] = np.stack(traj["global_grid"]).astype(np.float32)
        out["global_scalars"] = np.stack(traj["global_scalars"]).astype(np.float32)
    return out


def _compute_gae(rewards, values, dones, gamma: float, lam: float,
                 bootstrap_v: float = 0.0):
    """Per-trajectory GAE. The episode end in this env is truncation, not
    termination, so we bootstrap with ``bootstrap_v`` (the model's value at the
    last seen state) instead of treating the cutoff as terminal."""
    t = len(rewards)
    adv = np.zeros(t, dtype=np.float32)
    last = 0.0
    for i in reversed(range(t)):
        nonterminal = 1.0 - dones[i]
        next_v = values[i + 1] if i + 1 < t else bootstrap_v
        delta = rewards[i] + gamma * next_v * nonterminal - values[i]
        last = delta + gamma * lam * nonterminal * last
        adv[i] = last
    return adv, adv + values


# ── core episode loops (process-agnostic) ─────────────────────────────────────
@torch.no_grad()
def _collect_selfplay_episodes(envs, model, device, opponent_specs, n_learners,
                               gamma, lam, n_episodes, rng, advanced_prob=0.0,
                               learner_slots=None, shape_rewards=True,
                               live_nets=None, collect_global_state=False,
                               shaping=None):
    """Run *n_episodes* self-play games. Returns (trajs, learner_returns, opp_returns).

    ``shape_rewards`` controls the trajectory-level shaping (PBRS, oscillation
    penalty, turn-spam penalty). Env-level shaping (step penalty, multipliers,
    own_base_destroyed=0) is set when the env was built — see ``make_env``.
    Set both to ``False`` for a polish phase that optimises raw eval reward.

    ``collect_global_state`` (asymmetric/CTDE training): also record the
    privileged global-state arrays per learner step. The privileged critic is
    NOT run here — workers stay actor-only — so we just dump the raw arrays and
    let the PPO update compute V(global) + GAE in the main process.

    ``shaping`` (a ShapingConfig) controls trajectory-level shaping (PBRS,
    oscillation/turn-spam). When None it's derived from ``shape_rewards`` for
    backward compatibility. Env-level shaping (penalties, offensive multipliers,
    own_base) is baked into the env when it was built — see ``make_env``.
    """
    sh = _resolve_shaping(shape_rewards, shaping)
    model.eval()
    all_trajs: list[dict] = []
    learner_returns: list[float] = []
    opp_returns: list[float] = []

    for _ in range(n_episodes):
        episode_novice, env = _select_env(envs, advanced_prob, rng)
        env.reset(seed=rng.randint(0, 2_000_000_000))
        agents = list(env.possible_agents)
        eligible = [a for a in (learner_slots or agents) if a in agents]
        if len(eligible) < n_learners:
            eligible = agents
        learner_ids = set(rng.sample(eligible, n_learners))
        opp_ids = [a for a in agents if a not in learner_ids]
        controllers = {
            a: build_controller(
                _spec_for_map(rng.choice(opponent_specs), episode_novice),
                device,
                live_nets=live_nets,
            )
            for a in opp_ids
        }

        # hidden=None lets model.act build the initial state from the spawn
        # embedding (using the first observation's base_location).
        hidden: dict = {a: None for a in learner_ids}
        traj = {a: _new_trajectory() for a in learner_ids}
        opened = {a: False for a in learner_ids}
        memories = {a: _fresh_learner_memory(episode_novice) for a in learner_ids}
        # PBRS bookkeeping: prev_phi[a] holds Φ(s_t-1) — used to shape r_t-1.
        prev_phi: dict[str, float] = {a: 0.0 for a in learner_ids}
        # Anti-oscillation bookkeeping: per-agent rolling (action, pos) history
        # and a pending penalty that gets folded into the next reward we record
        # (which is the reward attributed to the action that completed the cycle).
        action_history: dict[str, deque] = {
            a: deque(maxlen=OSCILLATION_WINDOW) for a in learner_ids
        }
        pending_penalty: dict[str, float] = {a: 0.0 for a in learner_ids}
        # Consecutive turn-action count per agent. Reset when the agent does
        # any non-turn action (FORWARD / BACKWARD / STAY / PLACE_BOMB).
        consecutive_turns: dict[str, int] = {a: 0 for a in learner_ids}

        while True:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                if all(env.terminations.values()) or all(env.truncations.values()):
                    break
                continue

            reward = float(env._cumulative_rewards.get(agent, 0.0))
            obs = env.observe(agent)

            if agent in learner_ids:
                mem = memories[agent]
                try:
                    mem.update(parse_observation(obs))
                except Exception:
                    pass
                # PBRS: Φ(s_t). Shape the just-completed transition's reward
                # with γ·Φ(s_t) − Φ(s_{t-1}); store Φ(s_t) for next time.
                # Also fold in any pending oscillation / turn-spam penalty
                # attributed to the action that produced this transition.
                # All three shaping terms gated by shape_rewards so the polish
                # phase can train against the raw env reward.
                phi_curr = _compute_phi(obs, mem) if sh.pbrs else 0.0
                if opened[agent]:
                    shaped = reward
                    if sh.pbrs:
                        shaped += gamma * phi_curr - prev_phi[agent]
                    # pending_penalty is 0 unless anti_oscillation queued one.
                    shaped += pending_penalty[agent]
                    traj[agent]["rewards"].append(shaped)
                    traj[agent]["dones"].append(0.0)
                    pending_penalty[agent] = 0.0
                prev_phi[agent] = phi_curr
                vc, bv, sc, mk, smap = obs_to_arrays(obs, memory=mem)
                tv = lambda a: torch.as_tensor(a, device=device).unsqueeze(0)  # noqa: E731
                action, logp, value, _, hidden[agent] = model.act(
                    tv(vc), tv(bv), tv(sc), tv(mk), tv(smap), hidden[agent]
                )
                a_int = int(action.item())
                loc_tuple = _agent_xy(obs)
                if sh.anti_oscillation:
                    # Anti-oscillation: this action closes a 2- or 3-step
                    # (action, position) cycle → queue penalty against it.
                    if _is_oscillating(action_history[agent], a_int, loc_tuple):
                        pending_penalty[agent] += SHAPING_OSCILLATION_PENALTY
                    # Turn-spam: agent spinning in place without forward motion.
                    # Penalises every turn after TURN_SPAM_THRESHOLD consecutive
                    # turns; a FORWARD / BACKWARD / STAY / BOMB resets the count.
                    if a_int in (int(Action.LEFT), int(Action.RIGHT)):
                        consecutive_turns[agent] += 1
                        if consecutive_turns[agent] > TURN_SPAM_THRESHOLD:
                            pending_penalty[agent] += SHAPING_TURN_SPAM_PENALTY
                    else:
                        consecutive_turns[agent] = 0
                action_history[agent].append((a_int, loc_tuple))
                traj[agent]["viewcone"].append(vc)
                traj[agent]["baseview"].append(bv)
                traj[agent]["scalars"].append(sc)
                traj[agent]["mask"].append(mk)
                traj[agent]["staticmap"].append(smap)
                traj[agent]["actions"].append(a_int)
                traj[agent]["logp"].append(float(logp.item()))
                traj[agent]["values"].append(float(value.item()))
                if collect_global_state:
                    # Ground-truth privileged state, aligned with this obs/action.
                    g_grid, g_scal = build_global_state(env, agent)
                    traj[agent]["global_grid"].append(g_grid)
                    traj[agent]["global_scalars"].append(g_scal)
                opened[agent] = True
                env.step(a_int)
            else:
                env.step(controllers[agent].act(obs))

        episode = getattr(env.dynamics.rewards, "_episode", {})
        for a in learner_ids:
            ep_total = float(episode.get(a, 0.0))
            learner_returns.append(ep_total)
            # Episodes end via truncation, not termination. Don't dump the
            # leftover reward onto the final action (it overrepresents that
            # action's value) and don't mark done=1 (which would zero the GAE
            # bootstrap). Instead, attribute only the per-turn rewards we
            # actually observed and let GAE bootstrap with the model's last
            # value estimate.
            if opened[a]:
                # Pad rewards/dones up to action length with 0 — final action
                # is treated as "we saw an action but the game ended before we
                # observed the reward it caused"; the bootstrap value handles
                # the "would have earned more if game continued" signal.
                while len(traj[a]["rewards"]) < len(traj[a]["actions"]):
                    traj[a]["rewards"].append(0.0)
                    traj[a]["dones"].append(0.0)
            stacked = _stack_trajectory(traj[a])
            n = len(stacked["actions"])
            for key in ("rewards", "dones"):
                if len(stacked[key]) != n:
                    stacked[key] = stacked[key][:n]
            bootstrap_v = float(traj[a]["values"][-1]) if traj[a]["values"] else 0.0
            adv, ret = _compute_gae(stacked["rewards"], stacked["values"],
                                    stacked["dones"], gamma, lam,
                                    bootstrap_v=bootstrap_v)
            stacked["advantages"] = adv
            stacked["returns"] = ret
            all_trajs.append(stacked)
        for a in opp_ids:
            opp_returns.append(float(episode.get(a, 0.0)))

    return all_trajs, learner_returns, opp_returns


def _collect_teacher_episodes(env, n_episodes, rng, novice: bool = True,
                              teacher_spec: dict | None = None):
    """Every agent driven by the chosen teacher; records (obs, action) per agent.

    ``teacher_spec`` is a picklable spec from ``controllers`` (e.g.
    ``heuristic_spec()``, ``azbasev1_spec()``, ``azbasev4_spec()``). Defaults
    to the production heuristic if None. All 6 agents in every episode use
    the SAME teacher — to get a mixed-teacher dataset, run BC collection
    multiple times with different teachers and concatenate.
    """
    from controllers import HeuristicController, build_controller
    import torch as _torch

    seqs: list[dict] = []
    _device = _torch.device("cpu")
    for _ in range(n_episodes):
        env.reset(seed=rng.randint(0, 2_000_000_000))
        agents = list(env.possible_agents)
        if teacher_spec is None:
            controllers = {a: HeuristicController() for a in agents}
        else:
            controllers = {a: build_controller(teacher_spec, _device) for a in agents}
        memories = {a: _fresh_learner_memory(novice) for a in agents}
        rec = {a: {"viewcone": [], "baseview": [], "scalars": [], "mask": [],
                   "staticmap": [], "actions": []}
               for a in agents}
        while True:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                if all(env.terminations.values()) or all(env.truncations.values()):
                    break
                continue
            obs = env.observe(agent)
            action = int(controllers[agent].act(obs))
            mem = memories[agent]
            try:
                mem.update(parse_observation(obs))
            except Exception:
                pass
            vc, bv, sc, mk, smap = obs_to_arrays(obs, memory=mem)
            rec[agent]["viewcone"].append(vc)
            rec[agent]["baseview"].append(bv)
            rec[agent]["scalars"].append(sc)
            rec[agent]["mask"].append(mk)
            rec[agent]["staticmap"].append(smap)
            rec[agent]["actions"].append(action)
            env.step(action)
        for a in agents:
            if rec[a]["actions"]:
                seqs.append({k: (np.stack(v) if k != "actions" else np.asarray(v, dtype=np.int64))
                             for k, v in rec[a].items()})
    return seqs


# ── multiprocessing workers (spawn; CPU only) ─────────────────────────────────
_SP: dict = {}   # self-play worker globals
_TE: dict = {}   # teacher worker globals


def _sp_worker_init(opponent_specs, n_learners, novice, advanced_prob,
                    gamma, lam, learner_slots, shape_rewards,
                    n_live_slots=0, collect_global_state=False, shaping=None):
    torch.set_num_threads(1)
    # Match the parent's sharing strategy — see module-level comment.
    try:
        mp.set_sharing_strategy("file_system")
    except (RuntimeError, AttributeError):
        pass
    from model import RecurrentMaskableActorCritic
    sh = _resolve_shaping(shape_rewards, shaping)
    live_nets = {
        i: RecurrentMaskableActorCritic().to("cpu").eval()
        for i in range(int(n_live_slots))
    }
    _SP.update(
        device=torch.device("cpu"),
        model=RecurrentMaskableActorCritic().to("cpu").eval(),
        live_nets=live_nets,
        envs=_make_env_pool(novice, advanced_prob, shaping=sh),
        specs=opponent_specs, n_learners=n_learners, gamma=gamma, lam=lam,
        advanced_prob=advanced_prob, learner_slots=learner_slots,
        shape_rewards=shape_rewards, shaping=sh,
        collect_global_state=bool(collect_global_state),
    )


def _sp_worker_task(args):
    # Two task shapes are accepted:
    #   legacy: (state_dict, n_episodes, seed) — Stage 3 / earlier
    #   evolutionary: (state_dict, live_state_dicts, opp_specs_or_None, n_episodes, seed)
    # The evolutionary form lets the parent ship the latest weights of every
    # active population learner per chunk and override the opponent spec list
    # per learner (each learner can train against a different mix).
    if len(args) == 3:
        state_dict, n_episodes, seed = args
        live_state_dicts = None
        opp_specs = None
    else:
        state_dict, live_state_dicts, opp_specs, n_episodes, seed = args
    random.seed(seed)
    _SP["model"].load_state_dict(state_dict)
    if live_state_dicts:
        for slot, sd in live_state_dicts.items():
            slot = int(slot)
            if slot in _SP["live_nets"]:
                _SP["live_nets"][slot].load_state_dict(sd)
    specs = opp_specs if opp_specs is not None else _SP["specs"]
    rng = random.Random(seed)
    return _collect_selfplay_episodes(
        _SP["envs"], _SP["model"], _SP["device"], specs,
        _SP["n_learners"], _SP["gamma"], _SP["lam"], n_episodes, rng,
        _SP["advanced_prob"], _SP["learner_slots"],
        shape_rewards=_SP["shape_rewards"],
        live_nets=_SP["live_nets"],
        collect_global_state=_SP.get("collect_global_state", False),
        shaping=_SP.get("shaping"),
    )


def _te_worker_init(novice, teacher_spec=None):
    torch.set_num_threads(1)
    # Teacher dataset = BC demonstrations — keep RAW reward (shaping is for
    # the RL learner, not for cloning targets).
    _TE.update(
        env=make_env(novice, shape_rewards=False),
        novice=novice,
        teacher_spec=teacher_spec,
    )


def _te_worker_task(args):
    n_episodes, seed = args
    return _collect_teacher_episodes(
        _TE["env"], n_episodes, random.Random(seed), novice=_TE["novice"],
        teacher_spec=_TE.get("teacher_spec"),
    )


def _split(n: int, k: int) -> list[int]:
    """Split n items into k chunks as evenly as possible (drop empty chunks)."""
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k) if base + (1 if i < extra else 0) > 0]


def _normalise_and_assemble(trajs: list[dict]) -> RolloutBatch:
    t = min(len(tr["actions"]) for tr in trajs)

    def stack(key):
        return torch.as_tensor(np.stack([tr[key][:t] for tr in trajs], axis=1))

    adv = stack("advantages")
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    has_global = all("global_grid" in tr for tr in trajs) and bool(trajs) and \
        all(len(tr.get("global_grid", [])) for tr in trajs)
    g_grid = stack("global_grid") if has_global else None
    g_scal = stack("global_scalars") if has_global else None
    return RolloutBatch(
        viewcone=stack("viewcone"), baseview=stack("baseview"), scalars=stack("scalars"),
        mask=stack("mask"), staticmap=stack("staticmap"),
        actions=stack("actions"), logp=stack("logp"),
        values=stack("values"), rewards=stack("rewards"), dones=stack("dones"),
        advantages=adv, returns=stack("returns"),
        global_grid=g_grid, global_scalars=g_scal,
    )


class SelfPlayCollector:
    """Collects PPO rollouts: the learner controls a subset of agents, opponents
    (built from picklable specs) control the rest. With ``num_workers > 1`` games
    run across processes."""

    def __init__(
        self,
        model,
        device,
        opponent_specs,
        n_learners: int = 3,
        novice: bool = True,
        advanced_prob: float = 0.0,
        learner_slots: list[str] | None = None,
        gamma: float = 0.99,
        lam: float = 0.95,
        num_workers: int = 1,
        shape_rewards: bool = True,
        n_live_slots: int = 0,
        collect_global_state: bool = False,
        shaping: "ShapingConfig | None" = None,
    ):
        self.model = model
        self.device = device
        self.opponent_specs = list(opponent_specs)
        self.n_learners = max(1, min(n_learners, common.NUM_AGENTS))
        self.learner_slots = list(learner_slots or [])
        self.novice = novice
        self.advanced_prob = max(0.0, min(1.0, float(advanced_prob)))
        self.gamma = gamma
        self.lam = lam
        self.num_workers = max(1, int(num_workers))
        self.shape_rewards = bool(shape_rewards)
        # Per-component shaping config (None → derived from shape_rewards).
        self.shaping = _resolve_shaping(shape_rewards, shaping)
        # Asymmetric/CTDE: record privileged global state for the critic.
        self.collect_global_state = bool(collect_global_state)
        # Evolutionary mode: K live-opponent models pre-allocated per worker.
        # When > 0, .collect() expects live_state_dicts keyed by slot index.
        self.n_live_slots = max(0, int(n_live_slots))
        # Serial-path counterpart of the per-worker live_nets dict. Lazily
        # instantiated on first collect() so the cost is only paid in
        # evolutionary mode.
        self._serial_live_nets: dict = {}
        self._pool = None
        self.envs = (
            _make_env_pool(novice, self.advanced_prob, shaping=self.shaping)
            if self.num_workers == 1 else None
        )

    # Allow Stage 3 to swap in a larger opponent pool after a league snapshot.
    def set_opponent_specs(self, specs) -> None:
        self.opponent_specs = list(specs)
        self._close_pool()   # recreate with the new specs on next collect

    def _ensure_pool(self):
        if self._pool is None:
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                processes=self.num_workers,
                initializer=_sp_worker_init,
                initargs=(
                    self.opponent_specs, self.n_learners, self.novice,
                    self.advanced_prob, self.gamma, self.lam, self.learner_slots,
                    self.shape_rewards, self.n_live_slots, self.collect_global_state,
                    self.shaping,
                ),
            )

    def _close_pool(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def close(self):
        self._close_pool()

    def collect(
        self,
        n_episodes: int,
        progress: bool = False,
        live_state_dicts: dict | None = None,
        opp_specs_override: list | None = None,
    ):
        """Collect rollouts using ``self.model`` as the learner.

        In evolutionary mode the caller passes:
        - ``live_state_dicts``: dict[slot_idx -> state_dict] for the K active
          population learners; workers refresh their live-net models from this
          dict before sampling opponents.
        - ``opp_specs_override``: per-collect spec list (per-learner mix) that
          replaces the init-time pool for this collect only.

        Both default to None for backwards compatibility with Stage 3.
        """
        if self.num_workers == 1:
            trajs, lr, opr = self._collect_serial(
                n_episodes, progress, live_state_dicts, opp_specs_override
            )
            return self._finish(trajs, lr, opr)

        self._ensure_pool()
        cpu_sd = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        chunks = _split(n_episodes, self.num_workers)
        base = random.randint(0, 2_000_000_000)
        # CPU-detach the live state dicts once per collect (workers all see the
        # same snapshot for a given chunk). Workers only load the slot ints
        # that were allocated at init time; extra slots are silently ignored.
        live_cpu: dict | None = None
        if live_state_dicts is not None:
            live_cpu = {
                int(slot): {k: v.detach().cpu() for k, v in sd.items()}
                for slot, sd in live_state_dicts.items()
            }
        use_new_task_shape = (
            live_cpu is not None or opp_specs_override is not None
        )
        if use_new_task_shape:
            tasks = [
                (cpu_sd, live_cpu or {}, opp_specs_override, k, base + i)
                for i, k in enumerate(chunks)
            ]
        else:
            tasks = [(cpu_sd, k, base + i) for i, k in enumerate(chunks)]

        results = self._pool.imap_unordered(_sp_worker_task, tasks)
        if progress:
            results = tqdm(results, total=len(tasks), desc="  collect", leave=False, unit="chunk")

        trajs: list[dict] = []
        lr: list[float] = []
        opr: list[float] = []
        for tj, l, o in results:
            trajs.extend(tj)
            lr.extend(l)
            opr.extend(o)
        return self._finish(trajs, lr, opr)

    def _ensure_serial_live_nets(self):
        """Build the serial-path live-net cache lazily — mirrors per-worker setup."""
        if self.n_live_slots <= 0:
            return
        if len(self._serial_live_nets) == self.n_live_slots:
            return
        from model import RecurrentMaskableActorCritic
        self._serial_live_nets = {
            i: RecurrentMaskableActorCritic().to(self.device).eval()
            for i in range(self.n_live_slots)
        }

    def _collect_serial(self, n_episodes, progress,
                        live_state_dicts=None, opp_specs_override=None):
        rng = random
        live_nets = None
        if live_state_dicts is not None and self.n_live_slots > 0:
            self._ensure_serial_live_nets()
            for slot, sd in live_state_dicts.items():
                slot = int(slot)
                if slot in self._serial_live_nets:
                    self._serial_live_nets[slot].load_state_dict(sd)
            live_nets = self._serial_live_nets
        specs = opp_specs_override if opp_specs_override is not None else self.opponent_specs
        # Inline the loop so we can show a per-game bar in the serial path.
        trajs, lr, opr = [], [], []
        it = range(n_episodes)
        if progress:
            it = tqdm(it, desc="  collect", leave=False, unit="game")
        for _ in it:
            tj, l, o = _collect_selfplay_episodes(
                self.envs, self.model, self.device, specs,
                self.n_learners, self.gamma, self.lam, 1, rng,
                self.advanced_prob, self.learner_slots,
                shape_rewards=self.shape_rewards,
                live_nets=live_nets,
                collect_global_state=self.collect_global_state,
                shaping=self.shaping,
            )
            trajs.extend(tj); lr.extend(l); opr.extend(o)
        return trajs, lr, opr

    def _finish(self, trajs, learner_returns, opp_returns):
        batch = _normalise_and_assemble(trajs)
        stats = {
            "learner_return_mean": float(np.mean(learner_returns)) if learner_returns else 0.0,
            "learner_return_std": float(np.std(learner_returns)) if learner_returns else 0.0,
            "learner_return_min": float(np.min(learner_returns)) if learner_returns else 0.0,
            "learner_return_max": float(np.max(learner_returns)) if learner_returns else 0.0,
            "opp_return_mean": float(np.mean(opp_returns)) if opp_returns else 0.0,
            "n_seqs": batch.num_seqs,
        }
        return batch, stats


# ── BC teacher dataset (parallelisable) ───────────────────────────────────────
def collect_teacher_dataset(teacher_factory=None, n_episodes: int = 48, novice: bool = True,
                            progress: bool = False, num_workers: int = 1,
                            teacher_spec: dict | None = None):
    """Run *n_episodes* teacher-only games, recording (obs, action) per agent.

    ``teacher_spec`` is a picklable controller spec (e.g. ``heuristic_spec()``,
    ``azbasev1_spec()``, ``azbasev4_spec()``). ``None`` → the production
    heuristic. ``teacher_factory`` is unused (legacy call-site arg). Returns a
    dict of (T, B, …) numpy arrays.
    """
    num_workers = max(1, int(num_workers))

    if num_workers == 1:
        env = make_env(novice, shape_rewards=False)
        seqs = []
        it = range(n_episodes)
        if progress:
            it = tqdm(it, desc="  teacher games", unit="game")
        for _ in it:
            seqs.extend(_collect_teacher_episodes(
                env, 1, random, novice=novice, teacher_spec=teacher_spec))
    else:
        ctx = mp.get_context("spawn")
        chunks = _split(n_episodes, num_workers)
        base = random.randint(0, 2_000_000_000)
        tasks = [(k, base + i) for i, k in enumerate(chunks)]
        with ctx.Pool(num_workers, initializer=_te_worker_init,
                      initargs=(novice, teacher_spec)) as pool:
            results = pool.imap_unordered(_te_worker_task, tasks)
            if progress:
                results = tqdm(results, total=len(tasks), desc="  teacher chunks", unit="chunk")
            seqs = []
            for s in results:
                seqs.extend(s)

    t = min(len(s["actions"]) for s in seqs)
    return {
        "viewcone": np.stack([s["viewcone"][:t] for s in seqs], axis=1).astype(np.float32),
        "baseview": np.stack([s["baseview"][:t] for s in seqs], axis=1).astype(np.float32),
        "scalars": np.stack([s["scalars"][:t] for s in seqs], axis=1).astype(np.float32),
        "mask": np.stack([s["mask"][:t] for s in seqs], axis=1).astype(np.float32),
        "staticmap": np.stack([s["staticmap"][:t] for s in seqs], axis=1).astype(np.float32),
        "actions": np.stack([s["actions"][:t] for s in seqs], axis=1).astype(np.int64),
    }
