"""Stage 4 — Population-Based Self-Play (Evolution).

Trains K active learners in parallel, each with its own optimizer,
``RunningReturnNorm``, and a perturbed hyperparameter vector (entropy_coef, lr,
opponent-mix weights). Each "evolutionary update" runs K mini-collects: one
per learner, in which the learner plays as the RL agent and the other K-1
learners' current weights are sampled as live opponents (alongside frozen
archive snapshots and a small scripted slice).

Periodically (every ``--tournament-every`` updates) a round-robin tournament
ranks the live learners; the bottom learner is cloned from the top and its
hyperparams are perturbed (exploit-and-explore). This replaces Stage 3's
"rollback to best.pt on validation drop" pattern with natural selection across
multiple concurrent learning trajectories — diverse opponents emerge, and
nothing has to roll back if one learner falters because another is already
ahead.

Anti-stagnation: a learner whose tournament rank has been bottom-2 for
``--stagnation-window`` consecutive tournaments gets an entropy coef bump
(scaled by ``--stagnation-entropy-mult``) for the next cycle. The bottom-most
learner also gets cloned during reselection, so this is largely a backstop.

Live-opponent plumbing lives in:
- ``controllers.live_net_spec(slot)`` — spec referencing a slot index
- ``rollout._sp_worker_init(..., n_live_slots=K)`` — pre-allocates K CPU models
  in each worker
- ``rollout.SelfPlayCollector.collect(..., live_state_dicts=..., opp_specs_override=...)``
  — refreshes worker live-net weights per chunk; lets each learner train
  against its own opponent mix.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

import common  # noqa: F401  (path bootstrap)
from common import EVOLUTION_ARCHIVE_DIR, EVOLUTION_DIR
from controllers import (
    berserker_spec,
    heuristic_spec,
    idle_spec,
    kamikaze_spec,
    league_checkpoints,
    live_net_spec,
    net_spec,
    pure_collector_spec,
    random_spec,
    tactical_spec,
    trap_setter_spec,
    vanilla_heuristic_spec,
)
from model import RecurrentMaskableActorCritic, load_checkpoint, save_checkpoint
from ppo import RunningReturnNorm, ppo_update
from rollout import SelfPlayCollector


# ── hyperparameter spec ───────────────────────────────────────────────────────
# Per-learner mix weights for the non-net (scripted) slice. Each learner has its
# own values that get perturbed at exploit-and-explore time. We use *log-space*
# perturbation on positive quantities so a single mutation doesn't accidentally
# zero out a knob.
SCRIPTED_MIX_KEYS = (
    "tactical",
    "berserker",
    "vanilla_heuristic",
    "random",
    "idle",
    "trap_setter",
    "kamikaze",
    "pure_collector",
    "heuristic",
)

# Default mix vector seeded for every learner at init. Mutation jitters these.
# Note: heuristic + vanilla_heuristic are intentionally low because of the
# overfit-to-own-heuristic finding from Stage 3.
DEFAULT_SCRIPTED_MIX = {
    "tactical": 0.30,
    "berserker": 0.15,
    "vanilla_heuristic": 0.05,
    "random": 0.10,
    "idle": 0.05,
    "trap_setter": 0.05,
    "kamikaze": 0.10,
    "pure_collector": 0.10,
    "heuristic": 0.10,
}

# Bounds on the perturbed scalars. Hard clamps so a chain of mutations can't
# walk the hyperparameter into nonsense.
ENTROPY_RANGE = (5e-3, 5e-2)
LR_RANGE = (5e-5, 1e-3)


@dataclass
class Hyperparams:
    """One learner's mutable training knobs.

    All quantities here are perturbed at exploit-and-explore time. The mix
    vector is renormalised to sum to 1 every time it's perturbed; it controls
    the SCRIPTED slice of opponents only — live/archive shares are global.
    """

    entropy_coef: float = 0.015
    lr: float = 2e-4
    scripted_mix: dict = field(default_factory=lambda: dict(DEFAULT_SCRIPTED_MIX))

    def clone(self) -> "Hyperparams":
        return Hyperparams(
            entropy_coef=self.entropy_coef,
            lr=self.lr,
            scripted_mix=dict(self.scripted_mix),
        )

    def perturb(self, rng: random.Random, jitter: float = 0.5) -> None:
        """Multiplicative log-space jitter on all positive scalars.

        ``jitter`` is the standard deviation of the log-normal perturbation.
        0.5 means roughly ±50% per knob per mutation, clamped to the configured
        range. Mix entries get the same treatment and are then renormalised.
        """
        def _ln_jitter(value: float, lo: float, hi: float) -> float:
            log_v = math.log(value) + rng.gauss(0.0, jitter)
            return float(min(hi, max(lo, math.exp(log_v))))

        self.entropy_coef = _ln_jitter(self.entropy_coef, *ENTROPY_RANGE)
        self.lr = _ln_jitter(self.lr, *LR_RANGE)
        new_mix = {}
        for k in SCRIPTED_MIX_KEYS:
            v = max(1e-4, float(self.scripted_mix.get(k, 0.0)))
            log_v = math.log(v) + rng.gauss(0.0, jitter)
            new_mix[k] = math.exp(log_v)
        total = sum(new_mix.values())
        self.scripted_mix = {k: v / total for k, v in new_mix.items()}

    def to_dict(self) -> dict:
        return {
            "entropy_coef": self.entropy_coef,
            "lr": self.lr,
            "scripted_mix": dict(self.scripted_mix),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hyperparams":
        if d is None:
            return cls()
        out = cls(
            entropy_coef=float(d.get("entropy_coef", 0.015)),
            lr=float(d.get("lr", 2e-4)),
            scripted_mix=dict(d.get("scripted_mix") or DEFAULT_SCRIPTED_MIX),
        )
        return out


# ── learner container ─────────────────────────────────────────────────────────
@dataclass
class Learner:
    """One member of the population. Holds the model, optimizer, return-norm,
    hyperparams, and bookkeeping used for tournament ranking and anti-stagnation."""

    slot: int
    model: RecurrentMaskableActorCritic
    optimizer: torch.optim.Optimizer
    return_norm: RunningReturnNorm
    hp: Hyperparams
    # Recent learner_return_mean values from training collects (latest first).
    recent_returns: list[float] = field(default_factory=list)
    # Tournament history: list of {"update": int, "rank": int, "score": float}.
    tournament_history: list[dict] = field(default_factory=list)
    # Number of consecutive recent tournaments where this slot was in the bottom half.
    stagnation_streak: int = 0
    # When True, the next train step uses a bumped entropy coef.
    entropy_bump_active: bool = False

    def state_summary(self) -> dict:
        return {
            "slot": self.slot,
            "hp": self.hp.to_dict(),
            "recent_returns": list(self.recent_returns[-10:]),
            "stagnation_streak": self.stagnation_streak,
            "entropy_bump_active": self.entropy_bump_active,
            "tournament_history": list(self.tournament_history[-10:]),
        }


# ── opponent spec builder per learner ─────────────────────────────────────────
_POOL_GRANULARITY = 100

_SCRIPTED_SPEC_BUILDERS = {
    "heuristic": lambda: heuristic_spec(),
    "vanilla_heuristic": lambda: vanilla_heuristic_spec(),
    "berserker": lambda: berserker_spec(),
    "pure_collector": lambda: pure_collector_spec(),
    "random": lambda: random_spec(),
    "idle": lambda: idle_spec(),
    "trap_setter": lambda: trap_setter_spec(),
    "kamikaze": lambda: kamikaze_spec(),
    "tactical": lambda: tactical_spec(),
}


def build_opponent_specs(
    learner_slot: int,
    n_learners: int,
    archive_paths: list[Path],
    live_share: float,
    archive_share: float,
    scripted_share: float,
    scripted_mix: dict,
) -> list[dict]:
    """Weighted spec list for one learner.

    Three independent shares (renormalised internally to sum to 1):
    - ``live_share``: spread across the K-1 OTHER live learners.
    - ``archive_share``: spread across frozen tournament-snapshot files.
    - ``scripted_share``: split by ``scripted_mix`` across scripted policies.

    The pool granularity (100 entries) means a 0.5 live share with K=4 yields
    ~17 entries per other-learner slot — enough resolution for the per-episode
    ``rng.choice`` to be stable.
    """
    total = max(1e-6, live_share + archive_share + scripted_share)
    live_share /= total
    archive_share /= total
    scripted_share /= total

    specs: list[dict] = []
    counts: dict[str, int] = {}

    # Live opponents: every OTHER learner slot.
    other_slots = [j for j in range(n_learners) if j != learner_slot]
    if other_slots and live_share > 0:
        per_slot = max(1, round(_POOL_GRANULARITY * live_share / len(other_slots)))
        for j in other_slots:
            for _ in range(per_slot):
                specs.append(live_net_spec(slot=j))
            counts[f"live[{j}]"] = per_slot

    # Frozen archive snapshots.
    if archive_paths and archive_share > 0:
        per_arch = max(1, round(_POOL_GRANULARITY * archive_share / len(archive_paths)))
        for p in archive_paths:
            for _ in range(per_arch):
                specs.append(net_spec(p))
        counts["archive"] = per_arch * len(archive_paths)

    # Scripted slice: per-learner mix.
    if scripted_share > 0:
        mix_total = max(1e-6, sum(max(0.0, scripted_mix.get(k, 0.0)) for k in SCRIPTED_MIX_KEYS))
        for k in SCRIPTED_MIX_KEYS:
            w = max(0.0, scripted_mix.get(k, 0.0)) / mix_total
            n = round(_POOL_GRANULARITY * scripted_share * w)
            if n <= 0:
                continue
            builder = _SCRIPTED_SPEC_BUILDERS[k]
            for _ in range(n):
                specs.append(builder())
            counts[k] = n

    if not specs:
        # Pathological — fall back to something so collect() doesn't divide by 0.
        specs = [tactical_spec()]
        counts["tactical"] = 1
    return specs


# ── trainer ───────────────────────────────────────────────────────────────────
class EvolutionTrainer:
    """Owns K learners, drives per-update training, tournament, and reselection.

    Public surface (used by ``train_stage4_evolution.py``):
    - ``step(update_idx)`` — one evolutionary update: K mini-collects + K PPO
      updates. Returns a stats dict.
    - ``tournament(update_idx, rounds_per_pair)`` — round-robin between live
      learners, returns ranked list of (slot, score).
    - ``reselect(ranked, update_idx)`` — exploit-and-explore: bottom slot
      cloned from top + perturb.
    - ``snapshot_to_archive(update_idx)`` — save the current best live learner
      as a frozen archive entry that future opponent pools draw from.
    - ``save_state(path)`` / ``load_state(path)`` — full population resume.
    """

    def __init__(
        self,
        n_learners_pop: int,
        seed_model_path: Path | None,
        device: torch.device,
        # rollout/PPO config (shared across learners; per-learner LR comes from Hyperparams)
        episodes_per_update: int,
        n_learners_per_episode: int,
        novice: bool,
        advanced_prob: float,
        gamma: float,
        lam: float,
        epochs: int,
        seq_minibatch: int,
        clip: float,
        num_workers: int,
        shape_rewards: bool,
        # opponent shares (per-learner mix vector overrides only the SCRIPTED slice)
        live_share: float,
        archive_share: float,
        scripted_share: float,
        # anti-stagnation
        stagnation_window: int,
        stagnation_entropy_mult: float,
        # mutation
        mutation_jitter: float,
        rng: random.Random,
    ):
        self.K = int(n_learners_pop)
        self.device = device
        self.episodes_per_update = int(episodes_per_update)
        self.epochs = int(epochs)
        self.seq_minibatch = int(seq_minibatch)
        self.clip = float(clip)
        self.live_share = float(live_share)
        self.archive_share = float(archive_share)
        self.scripted_share = float(scripted_share)
        self.stagnation_window = int(stagnation_window)
        self.stagnation_entropy_mult = float(stagnation_entropy_mult)
        self.mutation_jitter = float(mutation_jitter)
        self.rng = rng

        # Build K learners. All share the seed model's WEIGHTS to start
        # (warm-starting from Stage 3); divergence comes from per-learner RNG
        # in collect (different seeds → different trajectories) plus mutated
        # hyperparams after the first tournament.
        self.learners: list[Learner] = []
        for slot in range(self.K):
            model = self._fresh_model(seed_model_path)
            hp = Hyperparams()
            # Seed light initial diversity: jitter entropy and LR a bit so the
            # first tournament has signal to act on. Mix stays at the default.
            hp.entropy_coef = self._initial_jitter(hp.entropy_coef, *ENTROPY_RANGE, jitter=0.3)
            hp.lr = self._initial_jitter(hp.lr, *LR_RANGE, jitter=0.3)
            opt = torch.optim.Adam(model.parameters(), lr=hp.lr)
            self.learners.append(Learner(
                slot=slot, model=model, optimizer=opt,
                return_norm=RunningReturnNorm(), hp=hp,
            ))

        # ONE collector for all K learners. ``model`` arg is overwritten before
        # each collect (we pass the slot's state dict); ``opponent_specs`` is
        # also overridden per-call via ``opp_specs_override``.
        # The collector owns the worker pool, so we pay pool startup once.
        self.collector = SelfPlayCollector(
            model=self.learners[0].model,   # placeholder; replaced per call
            device=device,
            opponent_specs=[tactical_spec()],   # placeholder pool — every call overrides
            n_learners=n_learners_per_episode,
            novice=novice,
            advanced_prob=advanced_prob,
            gamma=gamma, lam=lam,
            num_workers=num_workers,
            shape_rewards=shape_rewards,
            n_live_slots=self.K,
        )

    # ── construction helpers ──────────────────────────────────────────────
    def _fresh_model(self, seed_path: Path | None) -> RecurrentMaskableActorCritic:
        if seed_path is not None and seed_path.exists():
            return load_checkpoint(seed_path, self.device)
        return RecurrentMaskableActorCritic().to(self.device)

    @staticmethod
    def _initial_jitter(value, lo, hi, jitter):
        log_v = math.log(value) + random.gauss(0.0, jitter)
        return float(min(hi, max(lo, math.exp(log_v))))

    # ── opponent spec building ────────────────────────────────────────────
    def _archive_paths(self) -> list[Path]:
        return league_checkpoints(EVOLUTION_ARCHIVE_DIR)

    def _specs_for(self, slot: int) -> list[dict]:
        learner = self.learners[slot]
        return build_opponent_specs(
            learner_slot=slot,
            n_learners=self.K,
            archive_paths=self._archive_paths(),
            live_share=self.live_share,
            archive_share=self.archive_share,
            scripted_share=self.scripted_share,
            scripted_mix=learner.hp.scripted_mix,
        )

    def _live_state_dicts(self) -> dict[int, dict]:
        """Snapshot every live learner's current weights. The collector
        CPU-detaches these once per collect."""
        return {l.slot: l.model.state_dict() for l in self.learners}

    # ── per-update step ───────────────────────────────────────────────────
    def step(self, update_idx: int) -> dict:
        """Run one evolutionary update: K (collect + PPO) cycles.

        Returns a stats dict aggregated across the K learners (mean return,
        mean policy/value loss, etc.) plus per-learner detail.
        """
        live_sds = self._live_state_dicts()
        per_learner_stats = []
        learner_returns_acc = []

        for learner in self.learners:
            specs = self._specs_for(learner.slot)
            # Swap the collector's "learner" model to this slot's model. Its
            # state_dict goes out to workers under the ``learner`` key; the
            # OTHER slots' state dicts go out as live opponents.
            self.collector.model = learner.model
            # Workers' learner_state_dict is shipped from ``self.collector.model``;
            # live_state_dicts ships the entire population.
            t0 = time.time()
            batch, coll_stats = self.collector.collect(
                self.episodes_per_update,
                progress=False,
                live_state_dicts=live_sds,
                opp_specs_override=specs,
            )
            ent_coef = learner.hp.entropy_coef
            if learner.entropy_bump_active:
                ent_coef = min(ENTROPY_RANGE[1], ent_coef * self.stagnation_entropy_mult)
                learner.entropy_bump_active = False
            # Sync the optimiser LR in case ``hp.lr`` was just mutated.
            for pg in learner.optimizer.param_groups:
                pg["lr"] = learner.hp.lr
            losses = ppo_update(
                learner.model, learner.optimizer, batch, self.device,
                epochs=self.epochs, seq_minibatch=self.seq_minibatch,
                clip=self.clip, entropy_coef=ent_coef,
                return_norm=learner.return_norm,
            )
            dt = time.time() - t0

            ret = float(coll_stats["learner_return_mean"])
            learner.recent_returns.append(ret)
            if len(learner.recent_returns) > 32:
                learner.recent_returns = learner.recent_returns[-32:]
            learner_returns_acc.append(ret)

            per_learner_stats.append({
                "slot": learner.slot,
                "ret_mean": ret,
                "opp_ret_mean": float(coll_stats["opp_return_mean"]),
                "policy_loss": float(losses["policy_loss"]),
                "value_loss": float(losses["value_loss"]),
                "entropy": float(losses["entropy"]),
                "approx_kl": float(losses["approx_kl"]),
                "entropy_coef": float(ent_coef),
                "lr": float(learner.hp.lr),
                "seconds": round(dt, 2),
            })

        return {
            "update": update_idx,
            "per_learner": per_learner_stats,
            "ret_mean": float(np.mean(learner_returns_acc)) if learner_returns_acc else 0.0,
            "ret_std": float(np.std(learner_returns_acc)) if learner_returns_acc else 0.0,
        }

    # ── tournament + reselection ──────────────────────────────────────────
    def tournament(self, update_idx: int, rounds_per_pair: int = 16) -> list[tuple[int, float]]:
        """Round-robin between live learners. Each ordered pair (a vs b) plays
        ``rounds_per_pair`` episodes where ``a`` is the rollout learner and
        ``b`` is the single live opponent. Score for slot ``a`` is mean
        learner_return_mean across all such pairings; we rank slots by this.

        We could rank by win/loss instead, but in this env the per-episode
        cumulative reward is a smoother proxy and matches what PPO is optimising.
        Sampling-based with K=4 this is K*(K-1)*rounds_per_pair = ~12*16 = 192
        episodes — same order as one training collect.
        """
        live_sds = self._live_state_dicts()
        # Score = sum of learner_return_mean over all (slot, opponent_slot) pairs.
        totals = {l.slot: 0.0 for l in self.learners}
        n_pairings = {l.slot: 0 for l in self.learners}
        for a in self.learners:
            for b in self.learners:
                if a.slot == b.slot:
                    continue
                self.collector.model = a.model
                specs = [live_net_spec(slot=b.slot) for _ in range(_POOL_GRANULARITY)]
                _, stats = self.collector.collect(
                    rounds_per_pair,
                    progress=False,
                    live_state_dicts=live_sds,
                    opp_specs_override=specs,
                )
                totals[a.slot] += float(stats["learner_return_mean"])
                n_pairings[a.slot] += 1
        ranked = sorted(
            [(s, totals[s] / max(1, n_pairings[s])) for s in totals],
            key=lambda x: x[1], reverse=True,
        )
        # Record tournament outcome on each learner.
        for rank_idx, (slot, score) in enumerate(ranked):
            learner = self.learners[slot]
            learner.tournament_history.append({
                "update": update_idx, "rank": rank_idx, "score": score,
            })
            # Stagnation: bottom-half rank counts as a "stagnant" tournament.
            bottom_half = rank_idx >= self.K // 2
            if bottom_half:
                learner.stagnation_streak += 1
            else:
                learner.stagnation_streak = 0
            if learner.stagnation_streak >= self.stagnation_window:
                learner.entropy_bump_active = True
        return ranked

    def reselect(self, ranked: list[tuple[int, float]], update_idx: int) -> dict | None:
        """Exploit-and-explore: replace the worst learner with a clone of the
        best (weights + return_norm copied) and perturb its hyperparams.

        Returns a description of the action taken (or None if nothing changed).
        """
        if self.K < 2 or not ranked:
            return None
        top_slot, top_score = ranked[0]
        bot_slot, bot_score = ranked[-1]
        if top_slot == bot_slot:
            return None
        # Threshold: only clone if the gap is non-trivial. The check uses absolute
        # gap; with shaped returns in the hundreds, even 10–20 points is a real
        # difference, so 5.0 is a sensible noise floor.
        if abs(top_score - bot_score) < 5.0:
            return None

        top = self.learners[top_slot]
        bot = self.learners[bot_slot]
        bot.model.load_state_dict(top.model.state_dict())
        bot.return_norm.load_state_dict(top.return_norm.state_dict())
        bot.hp = top.hp.clone()
        bot.hp.perturb(self.rng, jitter=self.mutation_jitter)
        bot.optimizer = torch.optim.Adam(bot.model.parameters(), lr=bot.hp.lr)
        bot.recent_returns = []
        bot.stagnation_streak = 0
        bot.entropy_bump_active = False
        return {
            "update": update_idx,
            "cloned_from": top_slot,
            "into": bot_slot,
            "top_score": top_score,
            "bot_score": bot_score,
            "new_entropy": bot.hp.entropy_coef,
            "new_lr": bot.hp.lr,
        }

    # ── archive snapshots ─────────────────────────────────────────────────
    def snapshot_to_archive(self, update_idx: int, ranked: list[tuple[int, float]] | None = None) -> Path | None:
        """Persist the current best live learner as a frozen archive checkpoint.

        Archive entries are added to every learner's opponent pool on the next
        ``_specs_for`` call (because ``_archive_paths`` re-globs each time), so
        the population sees freshly-snapshotted ancestors immediately.
        """
        if ranked:
            best_slot = ranked[0][0]
        else:
            best_slot = max(self.learners,
                            key=lambda l: float(np.mean(l.recent_returns)) if l.recent_returns else float("-inf")).slot
        EVOLUTION_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        existing = league_checkpoints(EVOLUTION_ARCHIVE_DIR)
        gen = len(existing)
        path = EVOLUTION_ARCHIVE_DIR / f"gen_{gen:04d}_slot{best_slot}.pt"
        save_checkpoint(path, self.learners[best_slot].model, meta={
            "stage": "stage4_archive",
            "update": update_idx,
            "slot": best_slot,
        })
        return path

    def prune_archive(self, max_size: int) -> list[str]:
        """Keep the most recent ``max_size`` archive snapshots; delete the
        oldest. Same rationale as Stage 3's --league-max-size: ancient
        snapshots represent obsolete play styles."""
        if max_size <= 0:
            return []
        snaps = league_checkpoints(EVOLUTION_ARCHIVE_DIR)
        pruned = []
        while len(snaps) > max_size:
            oldest = snaps.pop(0)
            try:
                oldest.unlink()
                pruned.append(oldest.name)
            except OSError:
                pass
        return pruned

    # ── save / resume ─────────────────────────────────────────────────────
    def save_state(self, path: Path, update_idx: int) -> None:
        """Whole-population checkpoint: K models, optimizers, return_norms, hyperparams."""
        payload = {
            "stage": "stage4_evolution",
            "update": update_idx,
            "K": self.K,
            "learners": [
                {
                    "slot": l.slot,
                    "model_state": l.model.state_dict(),
                    "optimizer_state": l.optimizer.state_dict(),
                    "return_norm": l.return_norm.state_dict(),
                    "hp": l.hp.to_dict(),
                    "recent_returns": list(l.recent_returns),
                    "tournament_history": list(l.tournament_history),
                    "stagnation_streak": l.stagnation_streak,
                    "entropy_bump_active": l.entropy_bump_active,
                }
                for l in self.learners
            ],
            "arch": {
                "feature_dim": self.learners[0].model.fuse[0].out_features,
                "gru_hidden": self.learners[0].model.gru_hidden,
                "gru_layers": self.learners[0].model.gru_layers,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_state(self, path: Path) -> int:
        """Restore population from a save_state checkpoint. Returns last update_idx."""
        # weights_only=False because we save optimizer state, which includes
        # non-tensor objects torch.save can't serialise under weights_only.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if ckpt.get("K") != self.K:
            raise ValueError(
                f"checkpoint has K={ckpt.get('K')} but trainer was configured "
                f"with K={self.K}; rerun with matching --pop-size or delete the checkpoint"
            )
        for saved in ckpt["learners"]:
            slot = int(saved["slot"])
            learner = self.learners[slot]
            learner.model.load_state_dict(saved["model_state"])
            learner.hp = Hyperparams.from_dict(saved.get("hp"))
            # Rebuild optimiser fresh, then load saved state so the LR matches hp.
            learner.optimizer = torch.optim.Adam(learner.model.parameters(), lr=learner.hp.lr)
            try:
                learner.optimizer.load_state_dict(saved["optimizer_state"])
            except Exception:
                pass   # tolerate optimiser shape drift across restarts
            learner.return_norm.load_state_dict(saved.get("return_norm") or {})
            learner.recent_returns = list(saved.get("recent_returns") or [])
            learner.tournament_history = list(saved.get("tournament_history") or [])
            learner.stagnation_streak = int(saved.get("stagnation_streak", 0))
            learner.entropy_bump_active = bool(saved.get("entropy_bump_active", False))
        return int(ckpt.get("update", 0))

    def save_best_single(self, path: Path, slot: int, update_idx: int, score: float | None = None) -> None:
        """Persist a single learner's weights as a standalone checkpoint
        compatible with ``model.load_checkpoint`` — used to publish the best
        learner for downstream deploy (ae/src/rl_policy.py)."""
        meta = {"stage": "stage4_best", "update": update_idx, "slot": slot}
        if score is not None:
            meta["validation_score"] = float(score)
        save_checkpoint(path, self.learners[slot].model, meta=meta)

    def close(self) -> None:
        self.collector.close()

    # ── summaries for logging ─────────────────────────────────────────────
    def population_summary(self) -> list[dict]:
        return [l.state_summary() for l in self.learners]
