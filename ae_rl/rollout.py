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
from dataclasses import dataclass

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

import common  # noqa: F401  (path bootstrap)
from common import obs_to_arrays
from controllers import _CACHE_TEMPLATE, build_controller
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


def make_env(novice: bool = True) -> Bomberman:
    cfg = default_config()
    cfg.env.novice = novice
    cfg.env.render_mode = None
    return Bomberman(cfg)


def _make_env_pool(novice: bool = True, advanced_prob: float = 0.0) -> dict[bool, Bomberman]:
    if novice and advanced_prob > 0.0:
        return {True: make_env(True), False: make_env(False)}
    return {novice: make_env(novice)}


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

    @property
    def num_seqs(self) -> int:
        return self.viewcone.shape[1]


def _new_trajectory() -> dict:
    return {k: [] for k in (
        "viewcone", "baseview", "scalars", "mask", "staticmap",
        "actions", "logp", "values", "rewards", "dones",
    )}


def _stack_trajectory(traj: dict) -> dict:
    return {
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


def _compute_gae(rewards, values, dones, gamma: float, lam: float):
    """Per-trajectory GAE. Truncation at the final step is treated as terminal."""
    t = len(rewards)
    adv = np.zeros(t, dtype=np.float32)
    last = 0.0
    for i in reversed(range(t)):
        nonterminal = 1.0 - dones[i]
        next_v = values[i + 1] if i + 1 < t else 0.0
        delta = rewards[i] + gamma * next_v * nonterminal - values[i]
        last = delta + gamma * lam * nonterminal * last
        adv[i] = last
    return adv, adv + values


# ── core episode loops (process-agnostic) ─────────────────────────────────────
@torch.no_grad()
def _collect_selfplay_episodes(envs, model, device, opponent_specs, n_learners,
                               gamma, lam, n_episodes, rng, advanced_prob=0.0,
                               learner_slots=None):
    """Run *n_episodes* self-play games. Returns (trajs, learner_returns, opp_returns)."""
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
            a: build_controller(_spec_for_map(rng.choice(opponent_specs), episode_novice), device)
            for a in opp_ids
        }

        hidden = {a: model.initial_hidden(1, device) for a in learner_ids}
        traj = {a: _new_trajectory() for a in learner_ids}
        opened = {a: False for a in learner_ids}
        memories = {a: _fresh_learner_memory(episode_novice) for a in learner_ids}

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
                if opened[agent]:
                    traj[agent]["rewards"].append(reward)
                    traj[agent]["dones"].append(0.0)
                mem = memories[agent]
                try:
                    mem.update(parse_observation(obs))
                except Exception:
                    pass
                vc, bv, sc, mk, smap = obs_to_arrays(obs, memory=mem)
                tv = lambda a: torch.as_tensor(a, device=device).unsqueeze(0)  # noqa: E731
                action, logp, value, _, hidden[agent] = model.act(
                    tv(vc), tv(bv), tv(sc), tv(mk), tv(smap), hidden[agent]
                )
                a_int = int(action.item())
                traj[agent]["viewcone"].append(vc)
                traj[agent]["baseview"].append(bv)
                traj[agent]["scalars"].append(sc)
                traj[agent]["mask"].append(mk)
                traj[agent]["staticmap"].append(smap)
                traj[agent]["actions"].append(a_int)
                traj[agent]["logp"].append(float(logp.item()))
                traj[agent]["values"].append(float(value.item()))
                opened[agent] = True
                env.step(a_int)
            else:
                env.step(controllers[agent].act(obs))

        episode = getattr(env.dynamics.rewards, "_episode", {})
        for a in learner_ids:
            ep_total = float(episode.get(a, 0.0))
            learner_returns.append(ep_total)
            if opened[a]:
                traj[a]["rewards"].append(ep_total - float(sum(traj[a]["rewards"])))
                traj[a]["dones"].append(1.0)
            stacked = _stack_trajectory(traj[a])
            n = len(stacked["actions"])
            for key in ("rewards", "dones"):
                if len(stacked[key]) != n:
                    stacked[key] = stacked[key][:n]
            adv, ret = _compute_gae(stacked["rewards"], stacked["values"],
                                    stacked["dones"], gamma, lam)
            stacked["advantages"] = adv
            stacked["returns"] = ret
            all_trajs.append(stacked)
        for a in opp_ids:
            opp_returns.append(float(episode.get(a, 0.0)))

    return all_trajs, learner_returns, opp_returns


def _collect_teacher_episodes(env, n_episodes, rng, novice: bool = True):
    """Every agent driven by the heuristic teacher; records (obs, action) per agent."""
    from controllers import HeuristicController

    seqs: list[dict] = []
    for _ in range(n_episodes):
        env.reset(seed=rng.randint(0, 2_000_000_000))
        agents = list(env.possible_agents)
        controllers = {a: HeuristicController() for a in agents}
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


def _sp_worker_init(opponent_specs, n_learners, novice, advanced_prob, gamma, lam, learner_slots):
    torch.set_num_threads(1)
    from model import RecurrentMaskableActorCritic
    _SP.update(
        device=torch.device("cpu"),
        model=RecurrentMaskableActorCritic().to("cpu").eval(),
        envs=_make_env_pool(novice, advanced_prob),
        specs=opponent_specs, n_learners=n_learners, gamma=gamma, lam=lam,
        advanced_prob=advanced_prob, learner_slots=learner_slots,
    )


def _sp_worker_task(args):
    state_dict, n_episodes, seed = args
    random.seed(seed)
    _SP["model"].load_state_dict(state_dict)
    rng = random.Random(seed)
    return _collect_selfplay_episodes(
        _SP["envs"], _SP["model"], _SP["device"], _SP["specs"],
        _SP["n_learners"], _SP["gamma"], _SP["lam"], n_episodes, rng,
        _SP["advanced_prob"], _SP["learner_slots"],
    )


def _te_worker_init(novice):
    torch.set_num_threads(1)
    _TE.update(env=make_env(novice), novice=novice)


def _te_worker_task(args):
    n_episodes, seed = args
    return _collect_teacher_episodes(
        _TE["env"], n_episodes, random.Random(seed), novice=_TE["novice"]
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
    return RolloutBatch(
        viewcone=stack("viewcone"), baseview=stack("baseview"), scalars=stack("scalars"),
        mask=stack("mask"), staticmap=stack("staticmap"),
        actions=stack("actions"), logp=stack("logp"),
        values=stack("values"), rewards=stack("rewards"), dones=stack("dones"),
        advantages=adv, returns=stack("returns"),
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
        self._pool = None
        self.envs = _make_env_pool(novice, self.advanced_prob) if self.num_workers == 1 else None

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
                ),
            )

    def _close_pool(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def close(self):
        self._close_pool()

    def collect(self, n_episodes: int, progress: bool = False):
        if self.num_workers == 1:
            trajs, lr, opr = self._collect_serial(n_episodes, progress)
            return self._finish(trajs, lr, opr)

        self._ensure_pool()
        cpu_sd = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        chunks = _split(n_episodes, self.num_workers)
        base = random.randint(0, 2_000_000_000)
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

    def _collect_serial(self, n_episodes, progress):
        rng = random
        # Inline the loop so we can show a per-game bar in the serial path.
        trajs, lr, opr = [], [], []
        it = range(n_episodes)
        if progress:
            it = tqdm(it, desc="  collect", leave=False, unit="game")
        for _ in it:
            tj, l, o = _collect_selfplay_episodes(
                self.envs, self.model, self.device, self.opponent_specs,
                self.n_learners, self.gamma, self.lam, 1, rng,
                self.advanced_prob, self.learner_slots,
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
                            progress: bool = False, num_workers: int = 1):
    """Run *n_episodes* heuristic-only games, recording (obs, action) per agent.

    ``teacher_factory`` is ignored (the teacher is always the heuristic); kept for
    call-site compatibility. Returns a dict of (T, B, …) numpy arrays.
    """
    num_workers = max(1, int(num_workers))

    if num_workers == 1:
        env = make_env(novice)
        seqs = []
        it = range(n_episodes)
        if progress:
            it = tqdm(it, desc="  teacher games", unit="game")
        for _ in it:
            seqs.extend(_collect_teacher_episodes(env, 1, random, novice=novice))
    else:
        ctx = mp.get_context("spawn")
        chunks = _split(n_episodes, num_workers)
        base = random.randint(0, 2_000_000_000)
        tasks = [(k, base + i) for i, k in enumerate(chunks)]
        with ctx.Pool(num_workers, initializer=_te_worker_init, initargs=(novice,)) as pool:
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
