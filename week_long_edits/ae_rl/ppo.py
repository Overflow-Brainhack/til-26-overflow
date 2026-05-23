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

from rollout import RolloutBatch


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
