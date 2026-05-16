from __future__ import annotations

import base64
import io
from typing import Any, Callable

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
from ultralytics import YOLO
from ultralytics.models import RTDETR


RMSE_GLOBAL_MAX = 67.0
RMSE_INSIDE_MAX = 50.0
SSIM_INSIDE_MIN = 0.3

CV_SERVER_URL = "http://localhost:5002/cv"
DEFAULT_SURROGATES: tuple[tuple[Callable[..., Any], str], ...] = (
    (YOLO, "yolov8x.pt"),
    (YOLO, "yolo11m.pt"),
    (YOLO, "yolo26m.pt"),
)


def _load_surrogate(
    model_class: Callable[[str], Any], model_id: str, device: str
) -> torch.nn.Module:
    model = model_class(model_id)
    model.model.train()
    return model.model.to(device)


def _surrogate_loss(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Confidence-suppression loss, robust to YOLO output shapes."""
    if isinstance(model, YOLO) or isinstance(model, RTDETR):
        raw = model(x, imgsz=1280, rect=True, half=True)
    else:
        raw = model(x)
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    if raw.dim() == 3:
        # (B, 4+nc, num_anchors)
        return torch.sigmoid(raw[:, 4:, :]).max(dim=1).values.mean()
    return torch.sigmoid(raw).mean()


def _project_global_rmse(delta: torch.Tensor, max_rmse: float) -> torch.Tensor:
    rmse = (delta * 255.0).pow(2).mean().sqrt()
    if rmse > max_rmse:
        delta = delta * (max_rmse / rmse)
    return delta


def _project_inside_rmse(
    delta: torch.Tensor,
    mask: torch.Tensor,
    max_rmse: float,
) -> torch.Tensor:
    """Scale down inside-bbox portion of delta to satisfy the inside RMSE cap.

    Matches the evaluation formula in pipeline.py:
        sqrt(mean_over_inside_pixels(mean_over_channels(diff^2)))
    """
    inside = delta * mask  # (1, 3, H, W), zero outside bboxes
    sq_per_pixel = (inside * 255.0).pow(2).mean(dim=1)  # mean over channels: (1, H, W)
    n_pixels = mask.squeeze(1).sum().clamp(min=1)
    rmse = (sq_per_pixel.sum() / n_pixels).sqrt()
    if rmse > max_rmse:
        scale = max_rmse / rmse
        delta = delta * (1.0 - mask) + inside * scale
    return delta


def _make_bbox_mask(
    H: int, W: int, boxes_xywh: list[list[float]], device: str
) -> torch.Tensor:
    """Boolean-valued (1, 1, H, W) mask: 1 inside any bbox, 0 outside."""
    mask = torch.zeros(1, 1, H, W, device=device)
    for x, y, w, h in boxes_xywh:
        x1, y1 = int(x), int(y)
        x2, y2 = min(int(x + w), W), min(int(y + h), H)
        if x2 > x1 and y2 > y1:
            mask[0, 0, y1:y2, x1:x2] = 1.0
    return mask


def _fetch_bboxes(image_b64: str) -> list[list[float]]:
    """Call CV server to get detected bboxes in [x, y, w, h] format."""
    try:
        resp = requests.post(
            CV_SERVER_URL,
            json={"instances": [{"key": 0, "b64": image_b64}]},
            timeout=8.0,
        )
        resp.raise_for_status()
        preds = resp.json()["predictions"]
        if not preds or not preds[0]:
            return []
        return [det["bbox"] for det in preds[0]]
    except Exception as e:
        print(f"[attack] CV server unavailable: {e}, running without bbox mask")
        return []


def _ssim_inside(
    orig_np: np.ndarray,
    adv_np: np.ndarray,
    mask_np: np.ndarray,
) -> float:
    if not mask_np.any():
        return 1.0
    _, ssim_map = structural_similarity(
        orig_np.astype(np.float32),
        adv_np.astype(np.float32),
        channel_axis=2,
        data_range=255.0,
        win_size=7,
        full=True,
    )
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=-1)
    return float(ssim_map[mask_np].mean())


class EnsemblePGDAttacker:
    """Dual-budget PGD over an ensemble of surrogate models.

    Budget constraints (from eval_thresholds_v2.yaml):
        - Global RMSE  ≤ 67 (all pixels)
        - Inside RMSE  ≤ 50 (inside detected bboxes)
        - Inside SSIM  ≥ 0.3 (enforced by scale-back at the end)

    Bboxes are fetched from the CV server at localhost:5002. If the server
    is unavailable, the attack falls back to a global-only budget.
    """

    STEPS = 6
    STEP_SIZE = 8.0  # pixel space [0, 255]

    def __init__(
        self,
        surrogate_ids: tuple[tuple[Callable[..., Any], str], ...] = DEFAULT_SURROGATES,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.surrogates = [
            _load_surrogate(clas, sid, self.device) for (clas, sid) in surrogate_ids
        ]

    def _ensemble_loss(self, adv: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(adv, (1280, 1280), mode="bilinear", align_corners=False)
        total = sum(_surrogate_loss(m, x) for m in self.surrogates)
        return total / len(self.surrogates)

    def _pgd(
        self,
        orig: torch.Tensor,
        mask: torch.Tensor,
        has_boxes: bool,
    ) -> torch.Tensor:
        alpha = self.STEP_SIZE / 255.0
        init_eps = (RMSE_INSIDE_MAX if has_boxes else RMSE_GLOBAL_MAX) / 255.0

        delta = torch.empty_like(orig).uniform_(-init_eps * 0.1, init_eps * 0.1)
        delta.requires_grad_(True)

        for _ in range(self.STEPS):
            adv = (orig + delta).clamp(0.0, 1.0)
            loss = self._ensemble_loss(adv)
            loss.backward()

            with torch.no_grad():
                delta.data -= alpha * delta.grad.sign()
                delta.data = _project_global_rmse(delta.data, RMSE_GLOBAL_MAX)
                if has_boxes:
                    delta.data = _project_inside_rmse(delta.data, mask, RMSE_INSIDE_MAX)
                # keep adversarial image in valid range
                delta.data.clamp_(-orig, 1.0 - orig)

            delta.grad.zero_()

        return delta.detach()

    def _enforce_ssim(
        self,
        orig_np: np.ndarray,
        adv_np: np.ndarray,
        orig: torch.Tensor,
        delta: torch.Tensor,
        mask_np: np.ndarray,
        mask: torch.Tensor,
    ) -> np.ndarray:
        """Binary-search scale-back of inside delta until SSIM_inside ≥ 0.3."""
        scale = 1.0
        for _ in range(6):
            if _ssim_inside(orig_np, adv_np, mask_np) >= SSIM_INSIDE_MIN:
                break
            scale *= 0.75
            with torch.no_grad():
                d = delta * (1.0 - mask) + delta * mask * scale
                adv = (orig + d).clamp(0.0, 1.0)
                adv_np = (
                    (adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
                    .clip(0, 255)
                    .astype(np.uint8)
                )
        return adv_np

    def noise(self, image_bytes: bytes) -> str:
        """Run dual-budget PGD on a single JPEG image.

        Args:
            image_bytes: Raw JPEG bytes.

        Returns:
            Base64-encoded JPEG of the adversarially perturbed image.
        """
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_np = np.array(img_pil, dtype=np.uint8)
        H, W = orig_np.shape[:2]

        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        boxes = _fetch_bboxes(image_b64)
        has_boxes = len(boxes) > 0
        mask = _make_bbox_mask(H, W, boxes, self.device)

        arr = orig_np.astype(np.float32) / 255.0
        orig = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.enable_grad():
            delta = self._pgd(orig, mask, has_boxes)

        adv = (orig + delta).clamp(0.0, 1.0)
        adv_np = (
            (adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )

        if has_boxes:
            mask_np = mask.squeeze().cpu().numpy().astype(bool)
            adv_np = self._enforce_ssim(orig_np, adv_np, orig, delta, mask_np, mask)

        buf = io.BytesIO()
        Image.fromarray(adv_np).save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("ascii")
