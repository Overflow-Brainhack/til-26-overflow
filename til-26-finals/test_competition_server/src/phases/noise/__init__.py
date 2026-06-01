"""Noise distribution phase.

Split into three modules:
  fairness.py — FairnessRecord + apply_fairness_fallback (pure, sync)
  driver.py   — NoiseImage + TeamNoiseDriver (one team's exchange)
  phase.py    — NoisePhase (orchestrates all teams, builds noised_lookup)
"""

from .driver import NoiseImage, TeamNoiseDriver
from .fairness import FairnessRecord, apply_fairness_fallback
from .phase import NoisePhase

__all__ = [
    "NoisePhase",
    "TeamNoiseDriver",
    "NoiseImage",
    "FairnessRecord",
    "apply_fairness_fallback",
]
