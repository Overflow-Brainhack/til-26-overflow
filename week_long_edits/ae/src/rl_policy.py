"""Deployment-side learned policy (recurrent maskable actor-critic).

Implements the same ``Policy`` interface as the heuristic so it drops into
``AEManager`` without touching the server or the rule-based policy. It loads a
checkpoint trained by the ``ae_rl/`` stack and runs the network at inference,
carrying GRU hidden state across the steps of a round and resetting it when a
new round begins (``obs.step == 0``).

This file is intentionally self-contained (only depends on ``torch`` +
``constants``/``observation``/``policy`` from ``ae/src``) so it works inside the
Docker image, which only copies ``ae/src``. The network definition MUST stay in
sync with ``ae_rl/model.py`` — the layer names match so ``state_dict`` loads
 directly. Bundle the ``.pt`` checkpoint into ``ae/models`` (e.g. ``ae/models/ae_rl.pt``)
 and point ``RLPolicy(checkpoint_path=...)`` at it.

To deploy: in ``ae_manager.py`` swap the policy construction to
``from rl_policy import RLPolicy`` / ``policy = RLPolicy()``. Left un-wired by
default so the heuristic remains production.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from constants import (
    AGENT_MAX_HEALTH,
    BASE_MAX_HEALTH,
    BASE_VIEW_SIDE,
    FREEZE_TURNS,
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_CHANNELS,
    NUM_ITERS,
    VIEWCONE_LENGTH,
    VIEWCONE_WIDTH,
    Action,
)
from map_memory import MapMemory
from observation import ParsedObs
from policy import Policy

# Normalisation constants — must match ae_rl/common.py.
MAX_TEAM_RESOURCES = 100.0
TEAM_BOMBS_NORM = 10.0
SCALAR_DIM = 14
STATIC_MAP_CHANNELS = 6

DEFAULT_CHECKPOINT = "models/stage2_ppo.pt"

_MASK_FILL = -1e9


# ── network (mirrors ae_rl/model.py — keep layer names identical) ─────────────
class _SpatialEncoder(nn.Module):
    def __init__(self, channels: int, h: int, w: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden * h * w

    def forward(self, x):
        return self.net(x).flatten(1)


class _ActorCritic(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        gru_hidden=256,
        gru_layers=1,
        cnn_hidden=32,
        scalar_hidden=64,
        static_cnn_hidden=16,
    ):
        super().__init__()
        self.agent_cnn = _SpatialEncoder(
            NUM_CHANNELS, VIEWCONE_LENGTH, VIEWCONE_WIDTH, cnn_hidden
        )
        self.base_cnn = _SpatialEncoder(
            NUM_CHANNELS, BASE_VIEW_SIDE, BASE_VIEW_SIDE, cnn_hidden
        )
        self.static_cnn = _SpatialEncoder(
            STATIC_MAP_CHANNELS, GRID_SIZE, GRID_SIZE, static_cnn_hidden
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(SCALAR_DIM, scalar_hidden), nn.ReLU(inplace=True)
        )
        fused_in = (
            self.agent_cnn.out_dim
            + self.base_cnn.out_dim
            + self.static_cnn.out_dim
            + scalar_hidden
        )
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, feature_dim), nn.ReLU(inplace=True)
        )
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.gru = nn.GRU(feature_dim, gru_hidden, num_layers=gru_layers)
        # Spawn-position embedding — mirrors ae_rl/model.py. Index
        # GRID_SIZE*GRID_SIZE is the "unknown" slot (zero-initialised).
        self.spawn_embedding = nn.Embedding(GRID_SIZE * GRID_SIZE + 1, gru_hidden)
        # Small init so an old checkpoint without spawn_embedding (partial load
        # leaves these at fresh init) doesn't catastrophically perturb the GRU.
        nn.init.normal_(self.spawn_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.spawn_embedding.weight[GRID_SIZE * GRID_SIZE])
        self.actor = nn.Linear(gru_hidden, NUM_ACTIONS)
        self.critic = nn.Linear(gru_hidden, 1)

    def initial_hidden(self, device):
        return torch.zeros(self.gru_layers, 1, self.gru_hidden, device=device)

    def initial_hidden_from_base(self, base_xy, device):
        """Build initial hidden from the agent's own (x, y) base location."""
        bx, by = int(base_xy[0]), int(base_xy[1])
        if 0 <= bx < GRID_SIZE and 0 <= by < GRID_SIZE:
            idx = bx * GRID_SIZE + by
        else:
            idx = GRID_SIZE * GRID_SIZE  # unknown slot
        with torch.no_grad():
            emb = self.spawn_embedding(torch.tensor([idx], device=device))  # (1, H)
        return emb.unsqueeze(0).expand(self.gru_layers, -1, -1).contiguous()

    @torch.no_grad()
    def act(self, viewcone, baseview, scalars, mask, staticmap, hidden, deterministic=False):
        v = self.agent_cnn(viewcone)
        b = self.base_cnn(baseview)
        m = self.static_cnn(staticmap)
        s = self.scalar_mlp(scalars)
        feat = self.fuse(torch.cat([v, b, m, s], dim=-1)).unsqueeze(0)  # (1, 1, F)
        out, hidden = self.gru(feat, hidden)
        logits = self.actor(out.squeeze(0))
        mb = mask.to(dtype=torch.bool)
        if mb.any():
            logits = logits.masked_fill(~mb, _MASK_FILL)
        action = (
            logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()
        )
        return int(action.item()), hidden


# ── feature encoding from ParsedObs ───────────────────────────────────────────
def _scalars_from_obs(obs: ParsedObs) -> np.ndarray:
    out = np.zeros(SCALAR_DIM, dtype=np.float32)
    if 0 <= obs.direction < 4:
        out[obs.direction] = 1.0
    out[4] = obs.location[0] / GRID_SIZE
    out[5] = obs.location[1] / GRID_SIZE
    out[6] = obs.base_location[0] / GRID_SIZE
    out[7] = obs.base_location[1] / GRID_SIZE
    out[8] = obs.health / AGENT_MAX_HEALTH
    out[9] = obs.frozen_ticks / max(1.0, FREEZE_TURNS)
    out[10] = obs.base_health / BASE_MAX_HEALTH
    out[11] = obs.team_resources / MAX_TEAM_RESOURCES
    out[12] = obs.team_bombs / TEAM_BOMBS_NORM
    out[13] = obs.step / NUM_ITERS
    return out


class RLPolicy(Policy):
    """Learned recurrent maskable policy. Consumes the shared ``MapMemory`` to
    build a static-map observation channel (walls + bases), matching what the
    training loop fed the network."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: str = "cpu",
        deterministic: bool = True,
    ):
        self.device = torch.device(device)
        self.deterministic = deterministic
        path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        arch = ckpt.get("arch", {})
        self.model = _ActorCritic(
            feature_dim=arch.get("feature_dim", 256),
            gru_hidden=arch.get("gru_hidden", 256),
            gru_layers=arch.get("gru_layers", 1),
        ).to(self.device)
        state = ckpt["model_state"]
        own = self.model.state_dict()
        loadable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        own.update(loadable)
        self.model.load_state_dict(own)
        self.model.eval()
        # Built from spawn embedding on the first obs.
        self._hidden = None

        # Analysis/debug fields read by auto_play's overlay (kept for parity).
        self._debug_mode = "rl"
        self._debug_target = None
        self._debug_pos = (0, 0)

    def reset(self) -> None:
        # Hidden is rebuilt from the spawn embedding on the next step==0 obs.
        self._hidden = None

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        # New round → initial hidden state seeded by our own base_location via
        # the spawn embedding (encodes slot identity for novice, spawn region
        # for advanced).
        if obs.step == 0 or self._hidden is None:
            self._hidden = self.model.initial_hidden_from_base(obs.base_location, self.device)
        self._debug_pos = obs.location

        if obs.frozen_ticks > 0:
            self._debug_mode = "frozen"
            return int(Action.STAY)

        mask = np.asarray(obs.action_mask, dtype=np.float32).flatten()
        if mask.size == NUM_ACTIONS and mask.sum() == 0:
            return int(Action.STAY)

        vc = np.ascontiguousarray(
            np.transpose(
                _fix(obs.agent_view, (VIEWCONE_LENGTH, VIEWCONE_WIDTH, NUM_CHANNELS)),
                (2, 0, 1),
            )
        )
        bv = np.ascontiguousarray(
            np.transpose(
                _fix(obs.base_view, (BASE_VIEW_SIDE, BASE_VIEW_SIDE, NUM_CHANNELS)),
                (2, 0, 1),
            )
        )
        sc = _scalars_from_obs(obs)
        smap = memory.static_map_layer() if memory is not None else np.zeros(
            (STATIC_MAP_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32
        )

        t = lambda a: torch.as_tensor(a, device=self.device).unsqueeze(0)  # noqa: E731
        action, self._hidden = self.model.act(
            t(vc), t(bv), t(sc), t(mask), t(smap), self._hidden, deterministic=self.deterministic
        )
        self._debug_mode = "rl"
        # Final mask guard (network can only pick legal actions, but be defensive).
        if 0 <= action < mask.size and mask[action] == 1:
            return int(action)
        return int(Action.STAY)


def _fix(arr, shape) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.shape == shape:
        return a
    out = np.zeros(shape, dtype=np.float32)
    sl = tuple(slice(0, min(x, y)) for x, y in zip(a.shape, shape))
    out[sl] = a[sl]
    return out
