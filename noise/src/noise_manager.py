"""CV-based adversarial image noising — no surrogate models or gradients required."""

from __future__ import annotations

import base64
import io
from io import BytesIO
import json
import os
import random
from pathlib import Path

import cupy as cp
import numpy as np
import requests
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity

import torch
from ultralytics import RTDETR


# ── Budget constants (match eval_thresholds_v2.yaml) ──────────────────────────
RMSE_GLOBAL_MAX = 67.0  # evaluator's global threshold
RMSE_INSIDE_MAX = (
    44.0  # 6-unit margin: JPEG can amplify pixelation block edges, pushing RMSE up
)
SSIM_INSIDE_MIN = (
    0.36  # post-JPEG target; extra margin above 0.30 for outside-noise boundary effects
)


# ── CV container access ───────────────────────────────────────────────────────
# In the finals compose stack the CV model runs in a sibling container; the noise
# container reaches it over the compose network by its service name. CV_HOST lets
# the same image work under the test compose, a custom compose, or the real finals
# stack without a rebuild. Running this module directly (__main__) talks to a
# CV server on localhost instead.
_CV_HOST = "localhost" if __name__ == "__main__" else os.environ.get("CV_HOST", "til-cv")
_CV_BASE_URL = f"http://{_CV_HOST}:5002"

# Short timeouts so a slow / unreachable / wedged CV container falls back to the
# in-process model quickly instead of stalling the per-image noise budget.
_CV_HEALTH_TIMEOUT = 1.0
_CV_NOISE_TIMEOUT = 2.0


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_mask(H: int, W: int, boxes: list[list[float]]) -> np.ndarray:
    """Boolean (H, W) mask: True inside any detected bbox."""
    mask = np.zeros((H, W), dtype=bool)
    for x, y, w, h in boxes:
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def _rmse_global(delta: np.ndarray) -> float:
    return float(np.sqrt(np.mean((delta**2).mean(axis=-1))))


def _rmse_inside(delta: np.ndarray, mask: np.ndarray) -> float:
    sq = (delta**2).mean(axis=-1)  # (H, W)
    return float(sq[mask].mean() ** 0.5)


def _dispatch_ssim(*args, **kwargs):
    if torch.cuda.is_available():
        return _ssim_torch(*args, **kwargs)

    _, ssim_map = structural_similarity(*args, **kwargs, channel_axis=2, full=True)
    return ssim_map


def _ssim_torch(
    orig: np.ndarray, adv: np.ndarray, win_size: int = 7, data_range: float = 255.0
) -> np.ndarray:
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    orig_t = torch.from_numpy(orig).permute(2, 0, 1).contiguous().to(device)
    adv_t = torch.from_numpy(adv).permute(2, 0, 1).contiguous().to(device)

    pad = win_size // 2
    n_channels = orig_t.size(0)
    NP = win_size * win_size
    cov_norm = NP / (NP - 1)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = (
        torch.ones(n_channels, 1, win_size, win_size, device=device, dtype=orig_t.dtype)
        / NP
    )

    mu1 = F.conv2d(orig_t, kernel, padding=pad, groups=n_channels)
    mu2 = F.conv2d(adv_t, kernel, padding=pad, groups=n_channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = cov_norm * (
        F.conv2d(orig_t * orig_t, kernel, padding=pad, groups=n_channels) - mu1_sq
    )
    sigma2_sq = cov_norm * (
        F.conv2d(adv_t * adv_t, kernel, padding=pad, groups=n_channels) - mu2_sq
    )
    sigma12 = cov_norm * (
        F.conv2d(orig_t * adv_t, kernel, padding=pad, groups=n_channels) - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.permute(1, 2, 0).detach().cpu().numpy()


def _ssim_inside(orig: np.ndarray, adv: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 1.0
    ssim_map = _dispatch_ssim(
        orig.astype(np.float32),
        adv.astype(np.float32),
        data_range=255.0,
        win_size=7,
    )

    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=-1)
    return float(ssim_map[mask].mean())


# ── Main attacker class ────────────────────────────────────────────────────────


class NoiseManager:
    """CV-technique adversarial noiser. No surrogate models or gradients.

    Inside detected bboxes (RMSE ≤ 44, SSIM ≥ 0.36 post-JPEG):
        Pixelation + gray pull + inversion targeting spatial, color, and contrast cues.

    Outside detected bboxes (global RMSE → 66):
        Object-bank PNGs composited onto a center-weighted grid, filling the budget.

    SSIM inside is guaranteed by a 12-step binary search on post-JPEG output.
    """

    _BANK_PATH = (
        Path(__file__).parent.parent / "object_bank"
        if __name__ == "__main__"
        else Path("./object_bank")
    )

    _t_cuda = torch.cuda.is_available()

    def __init__(self):
        self.model = None

        self._bank_cache = self._load_bank_cache()

    def _load_bank_cache(self) -> dict[str, list[np.ndarray]]:
        """Pre-load all object-bank PNGs as uint8 (60×60 RGBA) at startup."""
        TILE = 60
        cache: dict[str, list[np.ndarray]] = {}
        if not self._BANK_PATH.exists():
            print(
                f"[NoiseManager] Bank path missing: {self._BANK_PATH}, skipping cache"
            )
            return cache
        for cat_dir in self._BANK_PATH.iterdir():
            if not cat_dir.is_dir():
                continue
            imgs = []
            for png in cat_dir.glob("*.png"):
                try:
                    arr = np.array(
                        Image.open(png)
                        .convert("RGBA")
                        .resize((TILE, TILE), Image.LANCZOS),
                        dtype=np.uint8,
                    )
                    imgs.append(arr)
                except Exception:
                    continue
            if imgs:
                cache[cat_dir.name] = imgs
        return cache

    def _cv_healthy(self) -> bool:
        try:
            resp = requests.get(
                f"{_CV_BASE_URL}/health", timeout=_CV_HEALTH_TIMEOUT
            ).json()

            return resp.get("message", "") == "health ok"
        except Exception as _:
            return False

    def _fetch_bboxes(self, image_b64: str) -> list[list[float]]:
        if self._cv_healthy():
            try:
                resp = requests.post(
                    f"{_CV_BASE_URL}/noise",
                    data=json.dumps({"b64": image_b64}),
                    timeout=_CV_NOISE_TIMEOUT,
                )
                preds = resp.json()["detections"]
                if preds:
                    return [det["bbox"] for det in preds]
            except Exception as _:
                # CV reachable-but-slow or a malformed reply — fall through to the
                # in-process model rather than failing the whole noise op.
                pass

        if self.model is None:
            print("[NoiseManager] CV container unhealthy, loading model")
            self.model = RTDETR("models/rtdetr-l-70.pt")

        im = Image.open(BytesIO(base64.b64decode(image_b64)))
        results = self.model.predict(
            im, verbose=False, imgsz=1280, rect=True, half=True
        )
        preds = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            preds.append(
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            )
        return preds

    def _inside_attack(
        self, orig: np.ndarray, boxes: list[list[float]], mask: np.ndarray
    ) -> np.ndarray:
        """Per-bbox pixelation + inversion, scaled to fill RMSE budget."""
        H, W = orig.shape[:2]
        delta = np.zeros((H, W, 3), dtype=np.float32)
        if not mask.any():
            return delta

        orig_pil = Image.fromarray(orig)
        augmented = orig.astype(np.float32).copy()

        for x, y, bw, bh in boxes:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(W, int(x + bw)), min(H, int(y + bh))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = orig_pil.crop((x1, y1, x2, y2))
            cw, ch = crop.size

            # Pixelate: downsample to ~1/8 then restore with NEAREST — destroys
            # spatial structure and CNN texture features while keeping RMSE moderate
            factor = max(2, min(cw, ch) // 8)
            small = crop.resize(
                (max(1, cw // factor), max(1, ch // factor)), Image.NEAREST
            )
            pixelated = small.resize((cw, ch), Image.NEAREST)

            augmented[y1:y2, x1:x2] = np.array(pixelated, dtype=np.float32)

        xp = cp if self._t_cuda > 0 else np
        if xp is cp:
            aug_xp = cp.asarray(augmented)
            orig_xp = cp.asarray(orig, dtype=cp.float32)
            mask_xp = cp.asarray(mask)
        else:
            aug_xp, orig_xp, mask_xp = augmented, orig.astype(np.float32), mask

        # Structural delta (pixelate + grayscale) + colour inversion
        # Inversion (255 - orig) is maximally wrong for colour-based class features
        # while preserving luminance structure (low SSIM cost relative to mAP impact)
        struct_delta = aug_xp - orig_xp
        invert_delta = 255.0 - 2.0 * orig_xp  # = 255 - 2*orig
        combined = 0.6 * struct_delta + 0.15 * invert_delta

        delta_xp = xp.zeros((H, W, 3), dtype=xp.float32)
        delta_xp[mask_xp] = combined[mask_xp]

        rmse = _rmse_inside(delta_xp, mask_xp)
        if rmse > 1e-8:
            delta_xp[mask_xp] *= RMSE_INSIDE_MAX / rmse

        return delta_xp.get() if xp is cp else delta_xp

    def _enforce_ssim(
        self, orig: np.ndarray, delta: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """12-step binary search on inside scale to guarantee post-JPEG SSIM ≥ SSIM_INSIDE_MIN.

        Checks SSIM on the JPEG-encoded output rather than raw uint8, because pixelation
        creates hard block edges that JPEG ringing artifacts worsen post-encoding.
        Checking pre-JPEG would consistently underestimate the evaluator's actual SSIM.
        """

        def _ssim_post_jpeg(d: np.ndarray) -> float:
            import torchvision.io as tvio

            t = (
                torch.from_numpy(orig)
                .float()
                .add(torch.from_numpy(d))
                .clamp_(0, 255)
                .to(torch.uint8)
                .permute(2, 0, 1)
                .contiguous()
            )
            if self._t_cuda:
                device = "cuda"
            else:
                device = "cpu"
            t = tvio.decode_jpeg(tvio.encode_jpeg(t, quality=95), device=device)
            adv_jpeg = t.permute(1, 2, 0).cpu().numpy()
            return _ssim_inside(orig, adv_jpeg, mask)

        if _ssim_post_jpeg(delta) >= SSIM_INSIDE_MIN:
            return delta

        lo, hi = 0.0, 1.0
        best = np.zeros_like(delta)

        for _ in range(12):
            mid = (lo + hi) / 2.0
            d = delta * mid
            if _ssim_post_jpeg(d) >= SSIM_INSIDE_MIN:
                lo = mid
                best = d
            else:
                hi = mid

        return best

    def _outside_bank_attack(
        self,
        orig: np.ndarray,
        mask: np.ndarray,
        full_delta: np.ndarray,
        H: int,
        W: int,
    ) -> None:
        """Fill the outside-bbox RMSE budget by compositing object-bank PNGs onto a grid.

        Grid tiles are 60×60, sorted by a center-weighted opacity (1 at image center,
        0 at corners). Tiles overlapping any bbox are skipped. Objects are pasted
        from center outward until global RMSE reaches 66.
        """
        if not self._bank_cache:
            print(
                f"[NoiseManager] Cannot find {self._BANK_PATH} for object bank spamming."
            )
            return
        cat_names = list(self._bank_cache.keys())

        TILE = 60
        RMSE_TARGET = 66.0
        img_cy, img_cx = H / 2.0, W / 2.0
        max_dist = np.sqrt(img_cy**2 + img_cx**2)

        # Build grid: one 60×60 tile per non-overlapping position
        tiles: list[tuple[float, int, int]] = []
        for py in range(0, H - TILE + 1, TILE):
            for px in range(0, W - TILE + 1, TILE):
                if mask[py : py + TILE, px : px + TILE].any():
                    continue
                tile_cy = py + TILE / 2.0
                tile_cx = px + TILE / 2.0
                dist = np.sqrt((tile_cy - img_cy) ** 2 + (tile_cx - img_cx) ** 2)
                opacity = max(0.0, 1.0 - dist / max_dist)
                tiles.append((opacity, py, px))

        if not tiles:
            return

        # Center-first order
        tiles.sort(key=lambda t: t[0], reverse=True)

        # Track cumulative squared error incrementally to avoid full-image recompute
        n_total = H * W
        sum_sq = float((full_delta**2).mean(axis=-1).sum())

        i = 0
        while i < len(tiles):
            current_rmse = np.sqrt(sum_sq / n_total)
            if current_rmse >= RMSE_TARGET:
                break

            opacity, py, px = tiles[i]
            i += 1

            rgba = random.choice(self._bank_cache[random.choice(cat_names)]).astype(
                np.float32
            )

            alpha = rgba[:, :, 3] / 255.0  # (TILE, TILE) natural transparency
            rgb = rgba[:, :, :3]
            bg = orig[py : py + TILE, px : px + TILE].astype(np.float32)

            # Blend: opacity weights the grid point, alpha is the PNG's own mask
            blend = (alpha * opacity)[:, :, None]
            patch_delta = blend * (rgb - bg)

            # Incremental RMSE update
            old_sq = float(
                (full_delta[py : py + TILE, px : px + TILE] ** 2).mean(axis=-1).sum()
            )
            sum_sq = sum_sq - old_sq + float((patch_delta**2).mean(axis=-1).sum())
            full_delta[py : py + TILE, px : px + TILE] = patch_delta

    def noise(self, image_bytes: bytes) -> str:
        """Apply CV-based adversarial perturbation to a JPEG image.

        Args:
            image_bytes: Raw JPEG bytes.

        Returns:
            Base64-encoded JPEG of the adversarially perturbed image.
        """
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig = np.array(img_pil, dtype=np.uint8)
        H, W = orig.shape[:2]

        b64 = base64.b64encode(image_bytes).decode("ascii")
        boxes = self._fetch_bboxes(b64)
        has_boxes = len(boxes) > 0
        mask = _make_mask(H, W, boxes)

        inside_delta = self._inside_attack(orig, boxes, mask)
        if has_boxes:
            inside_delta = self._enforce_ssim(orig, inside_delta, mask)

        full_delta = inside_delta.copy()
        self._outside_bank_attack(orig, mask, full_delta, H, W)

        adv = np.clip(orig.astype(np.float32) + full_delta, 0, 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(adv).save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("ascii")


if __name__ == "__main__":
    import random

    attacker = NoiseManager()
    with open(
        f"/home/shadowmachete/dev/til-26-overflow/data/cv/images/{random.randint(1, 1000)}.jpg",
        "rb",
    ) as f:
        img_data = f.read()
    Image.open(io.BytesIO(img_data)).convert("RGB").show()
    adv_img_bytes = base64.b64decode(attacker.noise(img_data))
    Image.open(io.BytesIO(adv_img_bytes)).convert("RGB").show()
