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

from common import BASE_SHAPE, GRID_SIZE, NUM_ACTIONS, SCALAR_DIM, STATIC_MAP_SHAPE, VIEW_SHAPE

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

        # Spawn-position embedding: explicit slot identity into the initial GRU
        # hidden state. Each (x, y) base-location bucket gets its own learned
        # vector. On novice the slot↔location map is fixed, so this is a clean
        # slot-id signal; on advanced it still gives spawn-position conditioning.
        # Index GRID_SIZE * GRID_SIZE is reserved as "unknown" (zeros).
        self.spawn_embedding = nn.Embedding(GRID_SIZE * GRID_SIZE + 1, gru_hidden)
        nn.init.zeros_(self.spawn_embedding.weight[GRID_SIZE * GRID_SIZE])

        self.actor = nn.Linear(gru_hidden, NUM_ACTIONS)
        self.critic = nn.Linear(gru_hidden, 1)

        self.apply(self._init_weights)
        # Small actor head → near-uniform initial policy.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        # Small spawn-embedding init: starts as a gentle nudge to the all-zero
        # baseline so behaviour matches the pre-embedding model on day 1, then
        # the optimiser grows it.
        nn.init.normal_(self.spawn_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.spawn_embedding.weight[GRID_SIZE * GRID_SIZE])

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
        """Zero initial hidden state. Used by tests / fallback."""
        return torch.zeros(self.gru_layers, batch, self.gru_hidden, device=device)

    def initial_hidden_from_loc(self, base_locs, device) -> torch.Tensor:
        """Spawn-conditioned initial hidden state.

        ``base_locs`` is an integer tensor of shape (B, 2) giving each agent's
        own base (x, y). The returned tensor is shape (gru_layers, B, gru_hidden)
        — every layer gets the same per-agent embedding. Out-of-range coordinates
        map to the reserved "unknown" slot (zeros).
        """
        if base_locs.dtype != torch.long:
            base_locs = base_locs.long()
        x = base_locs[..., 0].clamp(0, GRID_SIZE - 1)
        y = base_locs[..., 1].clamp(0, GRID_SIZE - 1)
        valid = ((base_locs[..., 0] >= 0) & (base_locs[..., 0] < GRID_SIZE)
                 & (base_locs[..., 1] >= 0) & (base_locs[..., 1] < GRID_SIZE))
        idx = torch.where(valid, x * GRID_SIZE + y,
                          torch.full_like(x, GRID_SIZE * GRID_SIZE))
        emb = self.spawn_embedding(idx)                       # (B, gru_hidden)
        return emb.unsqueeze(0).expand(self.gru_layers, -1, -1).contiguous()

    @staticmethod
    def _scalars_to_base_loc(scalars_t0: torch.Tensor) -> torch.Tensor:
        """Recover integer base (x, y) from the first-timestep scalar slice.

        ``common.build_scalars`` writes ``base_location / GRID_SIZE`` into
        indices 6 and 7, so multiplying and rounding inverts that within ±1.
        """
        return (scalars_t0[..., 6:8] * GRID_SIZE).round().long()

    # ── single-step acting (rollout) ──────────────────────────────────────────
    @torch.no_grad()
    def act(self, viewcone, baseview, scalars, mask, staticmap, hidden=None,
            deterministic: bool = False):
        """One timestep for a batch of B agents.

        ``hidden=None`` → initial hidden is built from the spawn embedding
        using ``scalars`` (which carry the agent's own base_location).
        """
        if hidden is None:
            base_locs = self._scalars_to_base_loc(scalars)
            hidden = self.initial_hidden_from_loc(base_locs, viewcone.device)
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

        ``hidden=None`` → initial hidden is derived from each sequence's first
        timestep base_location via the spawn embedding.
        """
        t, b = viewcone.shape[0], viewcone.shape[1]
        if hidden is None:
            base_locs = self._scalars_to_base_loc(scalars[0])       # (B, 2)
            hidden = self.initial_hidden_from_loc(base_locs, viewcone.device)
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
