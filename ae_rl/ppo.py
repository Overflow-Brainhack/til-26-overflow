"""PPO update over recurrent (T, B) trajectories.

Minibatching is done over the *sequence* (B) dimension — never the time (T)
dimension — so the GRU can be unrolled over each full trajectory with a zero
initial hidden state (every trajectory starts at env reset). This keeps the
recurrence intact while still giving several gradient steps per rollout.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rollout import RolloutBatch


class RunningReturnNorm:
    """Welford running mean/std over scalar returns.

    The critic sees ``returns / std`` (no mean subtraction — we want to preserve
    sign so positive returns stay positive), which makes the value loss scale-
    invariant and stable across reward magnitudes. ``targets_from(...)`` rescales
    the GAE returns before they hit the value head; ``unnormalised(...)`` undoes
    the rescaling for logging or downstream consumers.

    A single shared instance lives on the trainer and is updated once per PPO
    batch. ``min_count`` and ``eps`` keep early-training values finite.
    """

    def __init__(self, eps: float = 1e-8, min_count: int = 8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.eps = eps
        self.min_count = min_count

    def update(self, x: torch.Tensor | np.ndarray) -> None:
        flat = np.asarray(x).reshape(-1).astype(np.float64)
        if flat.size == 0:
            return
        batch_mean = flat.mean()
        batch_var = flat.var()
        batch_count = flat.size
        if self.count == 0:
            self.mean = float(batch_mean)
            self.var = float(batch_var)
            self.count = int(batch_count)
            return
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.mean = float(new_mean)
        self.var = float(m2 / tot)
        self.count = int(tot)

    def std(self) -> float:
        if self.count < self.min_count:
            return 1.0
        return float(np.sqrt(self.var) + self.eps)

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.std()

    def unnormalise(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std()

    def state_dict(self) -> dict:
        return {"mean": float(self.mean), "var": float(self.var), "count": int(self.count)}

    def load_state_dict(self, state: dict) -> None:
        self.mean = float(state.get("mean", 0.0))
        self.var = float(state.get("var", 1.0))
        self.count = int(state.get("count", 0))


def ppo_update(
    model,
    optimizer,
    batch: RolloutBatch,
    device,
    *,
    epochs: int = 4,
    seq_minibatch: int = 8,
    clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    value_only: bool = False,
    return_norm: RunningReturnNorm | None = None,
) -> dict:
    """One PPO update over the rollout batch.

    ``value_only=True`` runs a critic warm-up: only the value loss is optimised.
    Pair it with freezing every parameter except ``model.critic`` (done by the
    caller) so the BC-trained actor/trunk is left untouched while the value head
    learns to predict returns — giving subsequent real PPO updates sane
    advantages instead of advantages derived from a random critic.
    """
    model.train()
    b = batch.num_seqs

    vc = batch.viewcone.to(device)
    bv = batch.baseview.to(device)
    sc = batch.scalars.to(device)
    mk = batch.mask.to(device)
    sm = batch.staticmap.to(device)
    act = batch.actions.to(device)
    logp_old = batch.logp.to(device)
    adv = batch.advantages.to(device)
    ret = batch.returns.to(device)
    val_old = batch.values.to(device)

    # Return normalisation: update running stats with this batch's returns,
    # then rescale the regression targets + value-old (which the value head
    # was trained against in the previous iteration's scale). The critic
    # outputs are in *normalised* units throughout, so adv is also implicitly
    # rescaled — but adv has already been advantage-normalised at rollout
    # collection time, so we leave it untouched.
    if return_norm is not None:
        return_norm.update(ret.detach().cpu().numpy())
        scale = return_norm.std()
        ret = ret / scale
        val_old = val_old / scale

    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clipfrac": []}

    for _ in range(epochs):
        perm = np.random.permutation(b)
        for start in range(0, b, seq_minibatch):
            cols = perm[start : start + seq_minibatch]
            idx = torch.as_tensor(cols, device=device)

            logp, ent, values = model.evaluate_actions(
                vc[:, idx], bv[:, idx], sc[:, idx], mk[:, idx], sm[:, idx], act[:, idx]
            )
            mb_adv = adv[:, idx]
            mb_ret = ret[:, idx]

            ratio = torch.exp(logp - logp_old[:, idx])
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Clipped value loss.
            v_clip = val_old[:, idx] + torch.clamp(
                values - val_old[:, idx], -clip, clip
            )
            v_loss = torch.max((values - mb_ret) ** 2, (v_clip - mb_ret) ** 2).mean()

            entropy = ent.mean()
            if value_only:
                loss = v_loss
            else:
                loss = policy_loss + value_coef * v_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = logp - logp_old[:, idx]
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip).float().mean()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(v_loss.item())
            stats["entropy"].append(entropy.item())
            stats["approx_kl"].append(approx_kl.item())
            stats["clipfrac"].append(clipfrac.item())

    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}


# ── asymmetric (CTDE) PPO ─────────────────────────────────────────────────────
def _gae_batched(rewards, values, dones, gamma: float, lam: float):
    """Batched GAE over (T, B). Episodes truncate (never terminate) in this env,
    so the final step bootstraps with its own value — matching the per-sequence
    convention in ``rollout._compute_gae`` (next_v at the last step = last value)."""
    t = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for i in reversed(range(t)):
        nonterminal = 1.0 - dones[i]
        next_v = values[i + 1] if i + 1 < t else values[t - 1]
        delta = rewards[i] + gamma * next_v * nonterminal - values[i]
        last = delta + gamma * lam * nonterminal * last
        adv[i] = last
    return adv, adv + values


def ppo_update_asymmetric(
    model,
    optimizer,
    batch: RolloutBatch,
    device,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    epochs: int = 4,
    seq_minibatch: int = 8,
    clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    return_norm: RunningReturnNorm | None = None,
    value_only: bool = False,
    bc_model=None,
    kl_anchor_coef: float = 0.0,
) -> dict:
    """One PPO update for an ``AsymmetricActorCritic`` (CTDE).

    Advantages come from the **privileged** critic, run over the recorded global
    state, NOT from the actor's local value head. GAE is recomputed here (once)
    because the privileged critic isn't evaluated during rollout — workers stay
    actor-only. The actor's own local critic head is left untrained (deploy may
    read it but the user's pipeline doesn't rely on it).

    ``bc_model`` (a frozen recurrent actor) + ``kl_anchor_coef`` add a
    ``coef · KL(π_current ‖ π_BC)`` penalty that keeps the policy from drifting
    off the behaviour-cloned manifold — the catastrophic-forgetting guard.
    ``value_only=True`` trains just the privileged critic (warm-up).
    """
    assert batch.has_global, "ppo_update_asymmetric needs a batch with global state"
    model.train()
    b = batch.num_seqs

    vc = batch.viewcone.to(device)
    bv = batch.baseview.to(device)
    sc = batch.scalars.to(device)
    mk = batch.mask.to(device)
    sm = batch.staticmap.to(device)
    act = batch.actions.to(device)
    logp_old = batch.logp.to(device)
    rewards = batch.rewards.to(device)
    dones = batch.dones.to(device)
    g_grid = batch.global_grid.to(device)
    g_scal = batch.global_scalars.to(device)

    # ── privileged V_old + GAE (computed once, no grad) ───────────────────
    with torch.no_grad():
        v_old = model.critic(g_grid, g_scal)            # (T, B)
        adv, ret = _gae_batched(rewards, v_old, dones, gamma, lam)
        # Advantage normalisation (whole batch) — same as the symmetric path.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    if return_norm is not None:
        return_norm.update(ret.detach().cpu().numpy())
        scale = return_norm.std()
        ret = ret / scale
        v_old_scaled = v_old / scale
    else:
        v_old_scaled = v_old

    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [],
             "clipfrac": [], "kl_anchor": []}

    for _ in range(epochs):
        perm = np.random.permutation(b)
        for start in range(0, b, seq_minibatch):
            cols = perm[start : start + seq_minibatch]
            idx = torch.as_tensor(cols, device=device)

            # Actor forward (recurrent BPTT over the sequence minibatch).
            logits, _, _ = model.actor.forward_sequence(
                vc[:, idx], bv[:, idx], sc[:, idx], mk[:, idx], sm[:, idx]
            )
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act[:, idx])
            entropy = dist.entropy().mean()

            # Privileged critic forward on the same minibatch's global state.
            values = model.critic(g_grid[:, idx], g_scal[:, idx])

            mb_adv = adv[:, idx]
            mb_ret = ret[:, idx]
            mb_v_old = v_old_scaled[:, idx]

            ratio = torch.exp(logp - logp_old[:, idx])
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            v_clip = mb_v_old + torch.clamp(values - mb_v_old, -clip, clip)
            v_loss = torch.max((values - mb_ret) ** 2, (v_clip - mb_ret) ** 2).mean()

            # KL anchor to the frozen BC policy (optional).
            kl_anchor = torch.zeros((), device=device)
            if bc_model is not None and kl_anchor_coef > 0.0:
                with torch.no_grad():
                    bc_logits, _, _ = bc_model.forward_sequence(
                        vc[:, idx], bv[:, idx], sc[:, idx], mk[:, idx], sm[:, idx]
                    )
                    bc_dist = Categorical(logits=bc_logits)
                # KL(current ‖ bc): pushes current back toward the BC policy.
                kl_anchor = torch.distributions.kl.kl_divergence(dist, bc_dist).mean()

            if value_only:
                loss = v_loss
            else:
                loss = (
                    policy_loss
                    + value_coef * v_loss
                    - entropy_coef * entropy
                    + kl_anchor_coef * kl_anchor
                )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = logp - logp_old[:, idx]
                approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip).float().mean()
            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(v_loss.item())
            stats["entropy"].append(entropy.item())
            stats["approx_kl"].append(approx_kl.item())
            stats["clipfrac"].append(clipfrac.item())
            stats["kl_anchor"].append(float(kl_anchor.item()))

    out = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
    # Surface the unnormalised return scale for logging parity with Stage 3.
    out["return_mean"] = float(ret.mean().item())
    return out
