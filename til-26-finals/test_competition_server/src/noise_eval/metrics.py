"""Distortion / similarity metrics for comparing original and noised images.

Every metric is a callable object that takes two numpy HWC uint8 images and
returns a scalar float (lower = less distortion for distance metrics, higher
= less distortion for similarity metrics).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from skimage.metrics import structural_similarity


@dataclass
class MetricResult:
    name: str
    value: float
    higher_is_better: bool

    def __repr__(self) -> str:
        return (
            f"{self.name}: {self.value:.4f} ({'↑' if self.higher_is_better else '↓'})"
        )


@dataclass
class ImageReport:
    results: list[MetricResult] = field(default_factory=list)

    def __repr__(self) -> str:
        return "\n".join(str(r) for r in self.results)

    def to_dict(self) -> dict[str, float]:
        return {r.name: r.value for r in self.results}


@dataclass
class BatchSummary:
    per_image: list[ImageReport]
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    min: dict[str, float] = field(default_factory=dict)
    max: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        header = f"BatchSummary over {len(self.per_image)} images\n{'=' * 50}"
        rows = [
            f"  {name:30s}  mean={self.mean[name]:8.4f}  std={self.std[name]:8.4f}  "
            f"min={self.min[name]:8.4f}  max={self.max[name]:8.4f}"
            for name in self.mean
        ]
        return header + "\n" + "\n".join(rows)


class DistortionMetric(ABC):
    """Base class for all distortion / similarity metrics."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def higher_is_better(self) -> bool:
        return False

    @abstractmethod
    def compute(self, original: np.ndarray, noised: np.ndarray) -> float:
        """Return the metric value for a single HWC uint8 image pair."""
        ...

    def __call__(self, original: np.ndarray, noised: np.ndarray) -> float:
        return self.compute(original, noised)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class L2Distance(DistortionMetric):
    """Root-mean-square L2 distance: ``sqrt( mean( (orig - noised)^2 ) )``"""

    @property
    def name(self) -> str:
        return "L2 (RMSE)"

    def compute(self, original: np.ndarray, noised: np.ndarray) -> float:
        diff = original.astype(np.float32) - noised.astype(np.float32)
        return float(np.sqrt(np.mean(diff**2)))


class SSIM(DistortionMetric):
    """Mean SSIM computed per channel and averaged (scikit-image)."""

    def __init__(self, win_size: int = 7, use_cuda: bool = False) -> None:
        self.win_size = win_size
        self.use_cuda = use_cuda

    @property
    def name(self) -> str:
        return "SSIM"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute(self, original: np.ndarray, noised: np.ndarray) -> float:
        return float(
            structural_similarity(
                original.astype(np.float32),
                noised.astype(np.float32),
                channel_axis=2,
                data_range=255.0,
                win_size=self.win_size,
            )
        )
