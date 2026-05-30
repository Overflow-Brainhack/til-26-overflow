"""Prioritized Fictitious Self-Play (PFSP) opponent sampling.

Uniform opponent sampling wastes most of the rollout budget on opponents the
policy already crushes or has no chance against. PFSP (AlphaStar, Vinyals et al.
2019) instead samples opponents weighted by how *informative* they are right
now — concentrating play on the matchups near the policy's current frontier.

Implementation note — why pool-rebuild rather than per-episode tracking:
this sampler sits *above* the rollout. It never touches the inner episode loop.
Periodically (every ``--pfsp-every`` updates) the trainer calls ``refresh`` with
the live collector; for each candidate opponent we run a few games (reusing the
collector's ``opp_specs_override`` so all opponent slots are that one policy)
and read back the mean return margin. That margin → a pseudo win-probability
``p``; the training pool then replicates each opponent in proportion to a
weight derived from ``p``. The trainer feeds ``weighted_pool()`` straight back
into ``collect(..., opp_specs_override=...)``. No changes to the rollout, the
worker plumbing, or the batch format.

Weighting modes (AlphaStar's two canonical curricula):
- ``"hard"`` : w(p) = (1 - p)^q   — focus on opponents we currently lose to.
               Right when we *know* a strong tier (azbase, league) is beating
               us and we want to close that gap.
- ``"even"`` : w(p) = p·(1 - p)    — focus on ~50/50 matchups, the classic
               "learn against opponents you can just barely handle" curriculum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class _Candidate:
    cid: str
    spec: dict
    # Exponential-moving pseudo win-probability of the LEARNER vs this opponent.
    # Starts at 0.5 (no information) so every candidate gets real play early.
    p_win: float = 0.5
    margin: float = 0.0
    games: int = 0


# Margin → pseudo win-probability. The return margin (learner_return_mean −
# opp_return_mean) is squashed through a logistic so PFSP weights are bounded
# and smooth. ``MARGIN_SCALE`` sets how many reward points correspond to a
# decisive win; shaped returns here are in the hundreds, so ~150 is a sensible
# "clearly winning" scale.
_MARGIN_SCALE = 150.0


def _margin_to_p(margin: float) -> float:
    return 1.0 / (1.0 + math.exp(-margin / _MARGIN_SCALE))


class PFSPSampler:
    """Maintains per-opponent win estimates and builds weighted opponent pools.

    Parameters
    ----------
    candidates : list[tuple[str, dict]]
        ``(id, opponent_spec)`` pairs. ids are for logging / stable identity
        across refreshes (e.g. ``"azbasev1"``, ``"league/gen_007"``).
    mode : "hard" | "even"
    q : float
        Exponent for the "hard" curriculum sharpness.
    ema : float
        Smoothing for the per-opponent win-prob update (0 = replace, 1 = frozen).
    floor : float
        Minimum share of the pool reserved for every candidate so a temporarily
        dominated opponent is never starved to zero (keeps the estimate fresh).
    """

    def __init__(self, candidates, mode: str = "hard", q: float = 2.0,
                 ema: float = 0.5, floor: float = 0.02):
        self.cands: list[_Candidate] = [
            _Candidate(cid=str(cid), spec=dict(spec)) for cid, spec in candidates
        ]
        self.mode = mode
        self.q = float(q)
        self.ema = float(ema)
        self.floor = float(floor)

    # ── identity / mutation ───────────────────────────────────────────────
    def ids(self) -> list[str]:
        return [c.cid for c in self.cands]

    def add_candidate(self, cid: str, spec: dict) -> None:
        """Register a new opponent (e.g. a freshly snapshotted league member).
        Seeded at p=0.5 so PFSP plays it enough to get a real estimate."""
        if any(c.cid == cid for c in self.cands):
            return
        self.cands.append(_Candidate(cid=str(cid), spec=dict(spec)))

    # ── refresh win estimates ─────────────────────────────────────────────
    def refresh(self, collector, eval_episodes: int, granularity: int = 100) -> dict:
        """Re-estimate each opponent's win-prob by playing the current policy
        (``collector.model``) against it for ``eval_episodes`` games.

        Reuses the collector's ``opp_specs_override`` so all opponent slots are
        the candidate. Returns a summary dict for logging.
        """
        out = []
        for c in self.cands:
            specs = [c.spec for _ in range(granularity)]
            _, stats = collector.collect(
                eval_episodes, progress=False, opp_specs_override=specs
            )
            margin = float(stats["learner_return_mean"]) - float(stats["opp_return_mean"])
            p = _margin_to_p(margin)
            c.p_win = (1 - self.ema) * p + self.ema * c.p_win if c.games else p
            c.margin = margin
            c.games += eval_episodes
            out.append({"id": c.cid, "p_win": round(c.p_win, 3),
                        "margin": round(margin, 1)})
        return {"mode": self.mode, "per_opponent": out}

    # ── build the training pool ───────────────────────────────────────────
    def _weight(self, p: float) -> float:
        if self.mode == "even":
            return max(1e-6, p * (1.0 - p))
        # "hard": opponents we lose to (low p) get more weight.
        return max(1e-6, (1.0 - p) ** self.q)

    def weighted_pool(self, granularity: int = 100) -> list[dict]:
        """Return a spec list whose composition ≈ the PFSP weights, with a
        per-candidate floor so no opponent is fully starved."""
        if not self.cands:
            return []
        weights = [self._weight(c.p_win) for c in self.cands]
        total = sum(weights) or 1.0
        shares = [w / total for w in weights]
        # Apply the floor and renormalise.
        shares = [max(self.floor, s) for s in shares]
        total = sum(shares) or 1.0
        shares = [s / total for s in shares]

        pool: list[dict] = []
        for c, share in zip(self.cands, shares):
            n = max(1, round(granularity * share))
            pool.extend(dict(c.spec) for _ in range(n))
        return pool

    def summary(self) -> list[dict]:
        return [
            {"id": c.cid, "p_win": round(c.p_win, 3), "margin": round(c.margin, 1),
             "games": c.games, "weight": round(self._weight(c.p_win), 4)}
            for c in self.cands
        ]
