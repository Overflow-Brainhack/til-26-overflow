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
            nn.init.orthogonal_(m.weight, gain=2**0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # ── feature extraction (handles arbitrary leading batch dims) ─────────────
    def _features(self, viewcone, baseview, scalars, staticmap) -> torch.Tensor:
        lead = viewcone.shape[:-3]  # (...,) before (C, H, W)
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
    def act(
        self,
        viewcone,
        baseview,
        scalars,
        mask,
        staticmap,
        hidden,
        deterministic: bool = False,
        temperature: float = 1.0,
    ):
        """One timestep for a batch of B agents.

        Shapes: viewcone (B, C, H, W); baseview (B, C, H, W); scalars (B, D);
        mask (B, A); staticmap (B, Cs, Gs, Gs); hidden (layers, B, gru_hidden).
        Returns action (B,), logp (B,), value (B,), entropy (B,), new_hidden.

        ``temperature`` scales logits before forming the Categorical. >1 flattens
        the distribution (more exploration); <1 sharpens it. Has no effect when
        ``deterministic=True`` since argmax is invariant to positive scaling.
        Used by adversary opponents to inject stochastic versions of past selves
        into the league pool.
        """
        feat = self._features(viewcone, baseview, scalars, staticmap).unsqueeze(
            0
        )  # (1, B, F)
        out, new_hidden = self.gru(feat, hidden)  # (1, B, H)
        out = out.squeeze(0)
        logits = masked_logits(self.actor(out), mask)
        if temperature != 1.0:
            logits = logits / float(temperature)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        value = self.critic(out).squeeze(-1)
        return action, dist.log_prob(action), value, dist.entropy(), new_hidden

    # ── full-sequence evaluation (BPTT for PPO / BC) ──────────────────────────
    def forward_sequence(
        self, viewcone, baseview, scalars, mask, staticmap, hidden=None
    ):
        """Run the GRU over a (T, B, …) sequence.

        Returns logits (T, B, A), values (T, B), final_hidden.
        """
        t, b = viewcone.shape[0], viewcone.shape[1]
        if hidden is None:
            hidden = self.initial_hidden(b, viewcone.device)
        feat = self._features(viewcone, baseview, scalars, staticmap)  # (T, B, F)
        out, hidden = self.gru(feat, hidden)  # (T, B, H)
        logits = masked_logits(self.actor(out), mask)
        values = self.critic(out).squeeze(-1)
        return logits, values, hidden

    def evaluate_actions(
        self, viewcone, baseview, scalars, mask, staticmap, actions, hidden=None
    ):
        """For PPO: return logp (T, B), entropy (T, B), values (T, B) for taken actions."""
        logits, values, _ = self.forward_sequence(
            viewcone, baseview, scalars, mask, staticmap, hidden
        )
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


# ── asymmetric (CTDE) privileged critic ───────────────────────────────────────
class PrivilegedCritic(nn.Module):
    """Feed-forward value head over the privileged global state.

    Used only at TRAINING time. Sees the whole arena (every agent/base/bomb/
    collectible — see ``global_state.build_global_state``) and estimates the
    value of the *evaluated* agent's situation. Because it gets the full state
    every step, it does not need to be recurrent: the temporally-relevant facts
    a GRU would have to remember (bomb timers, who's where) are already in the
    state. Keeping it feed-forward keeps the PPO update simple — no second
    hidden state to thread through BPTT.

    Input shapes mirror ``global_state``:
        grid    : (..., C, G, G)
        scalars : (..., D)
    Output: (...,) scalar value.
    """

    def __init__(self, grid_shape, scalar_dim: int, cnn_hidden: int = 48,
                 scalar_hidden: int = 64, feature_dim: int = 256):
        super().__init__()
        c, h, w = grid_shape
        self.grid_cnn = _SpatialEncoder(c, h, w, cnn_hidden)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, scalar_hidden),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(self.grid_cnn.out_dim + scalar_hidden, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Linear(feature_dim, 1)
        self.apply(_init_orthogonal)
        # Small value head so initial V ≈ 0 (matches the actor-init convention).
        nn.init.orthogonal_(self.value.weight, gain=0.01)
        nn.init.constant_(self.value.bias, 0.0)

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        lead = grid.shape[:-3]
        c, h, w = grid.shape[-3:]
        n = int(torch.tensor(lead).prod().item()) if lead else 1
        g = self.grid_cnn(grid.reshape(n, c, h, w))
        s = self.scalar_mlp(scalars.reshape(n, scalars.shape[-1]))
        f = self.fuse(torch.cat([g, s], dim=-1))
        v = self.value(f).squeeze(-1)
        return v.reshape(*lead) if lead else v.reshape(())


def _init_orthogonal(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class AsymmetricActorCritic(nn.Module):
    """CTDE wrapper: a normal recurrent actor + a privileged feed-forward critic.

    - ``actor`` is a standard ``RecurrentMaskableActorCritic``; its CNN/GRU/actor
      head drive action selection and ship to deploy *unchanged*. (Its own local
      critic head is retained for checkpoint-format parity with deploy and is
      cheaply kept in sync as an auxiliary, but advantages come from the
      privileged critic, not from it.)
    - ``critic`` is the ``PrivilegedCritic`` over the global state; training-only.

    The actor and critic are deliberately separate modules so the deploy
    checkpoint can serialise just ``actor.state_dict()`` (which loads into the
    deploy ``_ActorCritic`` by name) and leave the privileged critic out of the
    inference path entirely.
    """

    def __init__(self, global_grid_shape, global_scalar_dim: int,
                 feature_dim: int = 256, gru_hidden: int = 256, gru_layers: int = 1,
                 cnn_hidden: int = 32, scalar_hidden: int = 64,
                 static_cnn_hidden: int = 16,
                 critic_cnn_hidden: int = 48, critic_feature_dim: int = 256):
        super().__init__()
        self.global_grid_shape = tuple(global_grid_shape)
        self.global_scalar_dim = int(global_scalar_dim)
        self.actor = RecurrentMaskableActorCritic(
            feature_dim=feature_dim, gru_hidden=gru_hidden, gru_layers=gru_layers,
            cnn_hidden=cnn_hidden, scalar_hidden=scalar_hidden,
            static_cnn_hidden=static_cnn_hidden,
        )
        self.critic = PrivilegedCritic(
            global_grid_shape, global_scalar_dim,
            cnn_hidden=critic_cnn_hidden, feature_dim=critic_feature_dim,
        )

    # convenience pass-throughs to the actor (so call sites can stay terse)
    @property
    def gru_hidden(self) -> int:
        return self.actor.gru_hidden

    @property
    def gru_layers(self) -> int:
        return self.actor.gru_layers

    def initial_hidden(self, batch: int, device):
        return self.actor.initial_hidden(batch, device)

    def value_global(self, grid, scalars):
        """Privileged value estimate V(global_state)."""
        return self.critic(grid, scalars)


def save_asymmetric_checkpoint(
    path,
    model: AsymmetricActorCritic,
    meta: dict | None = None,
    extras: dict | None = None,
):
    """Persist an asymmetric model so that:

    - ``model_state`` holds the ACTOR weights (deploy loads these by name);
    - ``critic_global_state`` holds the privileged critic (training-only,
      ignored by deploy's shape-matched partial load);
    - ``arch`` mirrors the actor arch + the global-state dims so a resume can
      rebuild the critic.

    Deploy (``ae/src/policies/rl_policy.py``) only reads ``model_state`` +
    ``arch`` and shape-matches, so the extra critic key is harmless there.
    """
    actor = model.actor
    crit = model.critic
    payload = {
        "model_state": actor.state_dict(),
        "critic_global_state": crit.state_dict(),
        "arch": {
            "feature_dim": actor.fuse[0].out_features,
            "gru_hidden": actor.gru_hidden,
            "gru_layers": actor.gru_layers,
            "global_grid_shape": list(model.global_grid_shape),
            "global_scalar_dim": model.global_scalar_dim,
        },
        "meta": meta or {},
    }
    if extras:
        payload["extras"] = extras
    torch.save(payload, path)


def load_asymmetric_checkpoint(path, device, eval_mode: bool = False) -> AsymmetricActorCritic:
    """Rebuild an ``AsymmetricActorCritic`` from a ``save_asymmetric_checkpoint``
    payload. Tolerates partial loads (warm-start a new arch from an old one).

    If ``critic_global_state`` is absent (e.g. seeding from a plain Stage-1/2/3
    actor-only checkpoint), only the actor is loaded and the critic stays at
    fresh init — exactly what we want when bootstrapping asymmetric training
    from a BC/league seed.
    """
    from global_state import GLOBAL_GRID_SHAPE, GLOBAL_SCALAR_DIM

    ckpt = torch.load(path, map_location=device, weights_only=True)
    arch = ckpt.get("arch", {})
    grid_shape = tuple(arch.get("global_grid_shape", GLOBAL_GRID_SHAPE))
    scalar_dim = int(arch.get("global_scalar_dim", GLOBAL_SCALAR_DIM))
    model = AsymmetricActorCritic(
        global_grid_shape=grid_shape,
        global_scalar_dim=scalar_dim,
        feature_dim=arch.get("feature_dim", 256),
        gru_hidden=arch.get("gru_hidden", 256),
        gru_layers=arch.get("gru_layers", 1),
    ).to(device)

    def _partial_load(module, state):
        own = module.state_dict()
        loadable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        own.update(loadable)
        module.load_state_dict(own)
        return len(loadable), len(state)

    if "model_state" in ckpt:
        kept, tot = _partial_load(model.actor, ckpt["model_state"])
        print(f"[asymmetric] actor load: kept {kept}/{tot} tensors")
    if "critic_global_state" in ckpt:
        kept, tot = _partial_load(model.critic, ckpt["critic_global_state"])
        print(f"[asymmetric] privileged-critic load: kept {kept}/{tot} tensors")
    else:
        print("[asymmetric] no privileged critic in checkpoint — fresh critic init")
    if eval_mode:
        model.eval()
    return model


def save_checkpoint(
    path,
    model: RecurrentMaskableActorCritic,
    meta: dict | None = None,
    extras: dict | None = None,
):
    """Persist model weights, arch, meta, and optional ``extras``.

    ``extras`` carries auxiliary training state (e.g. RunningReturnNorm running
    stats) that must survive a restart. Kept separate from ``meta`` so meta can
    remain a small human-readable dict of summary numbers.
    """
    payload = {
        "model_state": model.state_dict(),
        "arch": {
            "feature_dim": model.fuse[0].out_features,
            "gru_hidden": model.gru_hidden,
            "gru_layers": model.gru_layers,
        },
        "meta": meta or {},
    }
    if extras:
        payload["extras"] = extras
    torch.save(payload, path)


def load_extras(path) -> dict:
    """Return the ``extras`` dict from a checkpoint, or ``{}`` if absent / unreadable.

    Safe to call on checkpoints saved before extras existed.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return {}
    raw = ckpt.get("extras", {})
    return dict(raw) if isinstance(raw, dict) else {}


def load_checkpoint(
    path, device, eval_mode: bool = False
) -> RecurrentMaskableActorCritic:
    # weights_only=True restricts unpickling to torch tensors + primitives so
    # a hostile checkpoint can't execute code on load. Our checkpoints contain
    # {model_state, arch, meta} — all safe under the restriction.
    ckpt = torch.load(path, map_location=device, weights_only=True)
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
        print(
            f"[load_checkpoint] partial load: kept {len(loadable)}/{len(state)} tensors; "
            f"skipped (shape mismatch or unknown): {skipped[:6]}{'…' if len(skipped) > 6 else ''}"
        )
    if eval_mode:
        model.eval()
    return model
