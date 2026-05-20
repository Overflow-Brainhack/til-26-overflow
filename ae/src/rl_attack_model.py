"""Torch network definition for the attack-only DQN.

Kept separate from rl_attack.py so production can import feature extraction and
JSON inference without importing torch.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - training dependency only
    raise RuntimeError("Install ae/requirements-rl.txt to train the attack DQN") from exc


class AttackDQN(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 96,
        spatial_channels: int = 16,
    ) -> None:
        super().__init__()
        self.spatial_net = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48 * 16 * 16, hidden_dim),
            nn.ReLU(),
        )
        self.scalar_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, scalar: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        spatial_features = self.spatial_net(spatial)
        scalar_features = self.scalar_net(scalar)
        return self.head(torch.cat((scalar_features, spatial_features), dim=1))
