"""Recurrent, action-masked actor-critic for the AE Bomberman game.

Architecture
------------
    agent_viewcone (C×7×5) ─┐
    base_viewcone  (C×7×7) ─┼─ small CNNs → flatten ─┐
    scalars        (14,)   ─┴─ MLP ──────────────────┴─ concat → Linear → GRU
                                                                       │
                                                          ┌────────────┴───────────┐
                                                       actor (6 logits)        critic (1)

Action masking is applied to the logits before forming the Categorical so
illegal actions (move into wall, bomb with no bombs, every-action-but-STAY when
frozen) receive zero probability — both at acting and PPO-evaluation time.

Two entry points:
  * ``act``              — single timestep, batch of agents, for rollout collection.
  * ``forward_sequence`` — full (T, B) sequence with an initial hidden state, for
                           BPTT in the PPO/BC update.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from common import BASE_SHAPE, NUM_ACTIONS, SCALAR_DIM, STATIC_MAP_SHAPE, VIEW_SHAPE

_MASK_FILL = -1e9


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits of illegal actions (mask == 0) to a large negative value.

    Rows whose mask is entirely zero (a dead agent) are left untouched so the
    softmax stays finite; callers must avoid sampling from such rows.
    """
    mask = mask.to(dtype=torch.bool)
    any_valid = mask.any(dim=-1, keepdim=True)
    safe_mask = torch.where(any_valid, mask, torch.ones_like(mask))
    return logits.masked_fill(~safe_mask, _MASK_FILL)


class _SpatialEncoder(nn.Module):
    """Two padded 3×3 convs that preserve spatial size, then flatten."""

    def __init__(self, channels: int, h: int, w: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden * h * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, C, H, W)
        return self.net(x).flatten(1)


class RecurrentMaskableActorCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        gru_hidden: int = 256,
        gru_layers: int = 1,
        cnn_hidden: int = 32,
        scalar_hidden: int = 64,
        static_cnn_hidden: int = 16,
    ):
        super().__init__()
        c, vh, vw = VIEW_SHAPE
        _, bh, bw = BASE_SHAPE
        sc_c, sh, sw = STATIC_MAP_SHAPE

        self.agent_cnn = _SpatialEncoder(c, vh, vw, cnn_hidden)
        self.base_cnn = _SpatialEncoder(c, bh, bw, cnn_hidden)
        self.static_cnn = _SpatialEncoder(sc_c, sh, sw, static_cnn_hidden)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(SCALAR_DIM, scalar_hidden),
            nn.ReLU(inplace=True),
        )
        fused_in = (
            self.agent_cnn.out_dim
            + self.base_cnn.out_dim
            + self.static_cnn.out_dim
            + scalar_hidden
        )
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.gru = nn.GRU(feature_dim, gru_hidden, num_layers=gru_layers)

        self.actor = nn.Linear(gru_hidden, NUM_ACTIONS)
        self.critic = nn.Linear(gru_hidden, 1)

        self.apply(self._init_weights)
        # Small actor head → near-uniform initial policy.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # ── feature extraction (handles arbitrary leading batch dims) ─────────────
    def _features(self, viewcone, baseview, scalars, staticmap) -> torch.Tensor:
        lead = viewcone.shape[:-3]              # (...,) before (C, H, W)
        c, vh, vw = viewcone.shape[-3:]
        _, bh, bw = baseview.shape[-3:]
        sc_c, sh, sw = staticmap.shape[-3:]
        n = int(torch.tensor(lead).prod().item()) if lead else 1

        v = self.agent_cnn(viewcone.reshape(n, c, vh, vw))
        b = self.base_cnn(baseview.reshape(n, c, bh, bw))
        m = self.static_cnn(staticmap.reshape(n, sc_c, sh, sw))
        s = self.scalar_mlp(scalars.reshape(n, scalars.shape[-1]))
        f = self.fuse(torch.cat([v, b, m, s], dim=-1))
        return f.reshape(*lead, -1) if lead else f.reshape(-1)

    def initial_hidden(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch, self.gru_hidden, device=device)

    # ── single-step acting (rollout) ──────────────────────────────────────────
    @torch.no_grad()
    def act(self, viewcone, baseview, scalars, mask, staticmap, hidden, deterministic: bool = False):
        """One timestep for a batch of B agents.

        Shapes: viewcone (B, C, H, W); baseview (B, C, H, W); scalars (B, D);
        mask (B, A); staticmap (B, Cs, Gs, Gs); hidden (layers, B, gru_hidden).
        Returns action (B,), logp (B,), value (B,), entropy (B,), new_hidden.
        """
        feat = self._features(viewcone, baseview, scalars, staticmap).unsqueeze(0)   # (1, B, F)
        out, new_hidden = self.gru(feat, hidden)                          # (1, B, H)
        out = out.squeeze(0)
        logits = masked_logits(self.actor(out), mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        value = self.critic(out).squeeze(-1)
        return action, dist.log_prob(action), value, dist.entropy(), new_hidden

    # ── full-sequence evaluation (BPTT for PPO / BC) ──────────────────────────
    def forward_sequence(self, viewcone, baseview, scalars, mask, staticmap, hidden=None):
        """Run the GRU over a (T, B, …) sequence.

        Returns logits (T, B, A), values (T, B), final_hidden.
        """
        t, b = viewcone.shape[0], viewcone.shape[1]
        if hidden is None:
            hidden = self.initial_hidden(b, viewcone.device)
        feat = self._features(viewcone, baseview, scalars, staticmap)        # (T, B, F)
        out, hidden = self.gru(feat, hidden)                      # (T, B, H)
        logits = masked_logits(self.actor(out), mask)
        values = self.critic(out).squeeze(-1)
        return logits, values, hidden

    def evaluate_actions(self, viewcone, baseview, scalars, mask, staticmap, actions, hidden=None):
        """For PPO: return logp (T, B), entropy (T, B), values (T, B) for taken actions."""
        logits, values, _ = self.forward_sequence(
            viewcone, baseview, scalars, mask, staticmap, hidden
        )
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


def save_checkpoint(path, model: RecurrentMaskableActorCritic, meta: dict | None = None):
    torch.save(
        {
            "model_state": model.state_dict(),
            "arch": {
                "feature_dim": model.fuse[0].out_features,
                "gru_hidden": model.gru_hidden,
                "gru_layers": model.gru_layers,
            },
            "meta": meta or {},
        },
        path,
    )


def load_checkpoint(path, device, eval_mode: bool = False) -> RecurrentMaskableActorCritic:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    model = RecurrentMaskableActorCritic(
        feature_dim=arch.get("feature_dim", 256),
        gru_hidden=arch.get("gru_hidden", 256),
        gru_layers=arch.get("gru_layers", 1),
    ).to(device)
    state = ckpt["model_state"]
    # Tolerate partial loads (e.g. an old checkpoint missing static_cnn or whose
    # fuse layer has a smaller input than the new architecture). Missing/shape-
    # mismatched tensors are left at their fresh initialisation so we can still
    # warm-start most of the network from a prior run.
    own = model.state_dict()
    loadable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in state if k not in loadable]
    own.update(loadable)
    model.load_state_dict(own)
    if skipped:
        print(f"[load_checkpoint] partial load: kept {len(loadable)}/{len(state)} tensors; "
              f"skipped (shape mismatch or unknown): {skipped[:6]}{'…' if len(skipped) > 6 else ''}")
    if eval_mode:
        model.eval()
    return model
