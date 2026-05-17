"""CV-based adversarial image noising — no surrogate models or gradients required."""

from __future__ import annotations

import base64
import io
import json
import random

import numpy as np
import requests
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity


# ── Budget constants (match eval_thresholds_v2.yaml) ──────────────────────────
RMSE_GLOBAL_MAX = 67.0  # evaluator's global threshold
RMSE_INSIDE_MAX = 47.0  # 3-unit margin below the 50 inside threshold
SSIM_INSIDE_MIN = 0.31  # 0.01 margin above the 0.30 inside threshold


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fetch_bboxes(image_b64: str) -> list[list[float]]:
    url = (
        "http://localhost:5002/cv"
        if __name__ == "__main__"
        else "http://host.docker.internal:5002/cv"
    )
    try:
        resp = requests.post(
            url,
            data=json.dumps({"instances": [{"key": 0, "b64": image_b64}]}),
        )
        preds = resp.json()["predictions"]
        if not preds or not preds[0]:
            return []
        return [det["bbox"] for det in preds[0]]
    except Exception as e:
        print(f"[attack] CV server unavailable: {e}")
        return []


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
    return float(np.sqrt(sq[mask].mean()))


def _ssim_inside(orig: np.ndarray, adv: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 1.0
    _, ssim_map = structural_similarity(
        orig.astype(np.float32),
        adv.astype(np.float32),
        channel_axis=2,
        data_range=255.0,
        win_size=7,
        full=True,
    )
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=-1)
    return float(ssim_map[mask].mean())


def _gray_overlay(orig: np.ndarray) -> np.ndarray:
    """Delta that pulls every pixel toward mid-gray (128), destroying contrast."""
    return 128.0 - orig.astype(np.float32)


def _hue_rotate(orig: np.ndarray, degrees: float) -> np.ndarray:
    """Delta from rotating the HSV hue channel by `degrees`.

    Attacks color-based detection features while leaving luminance and texture
    largely intact. Grayscale pixels (s=0) are unaffected by construction.
    """
    f = orig.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c + 1e-8

    h = (
        np.where(
            max_c == r,
            (g - b) / diff % 6,
            np.where(max_c == g, (b - r) / diff + 2, (r - g) / diff + 4),
        )
        / 6.0
    )
    s = np.where(max_c < 1e-8, 0.0, diff / (max_c + 1e-8))
    v = max_c

    h = (h + degrees / 360.0) % 1.0

    i = (h * 6).astype(int)
    frac = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - frac * s)
    t = v * (1 - (1 - frac) * s)
    i = i % 6

    r2 = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g2 = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b2 = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])

    rotated = np.stack([r2, g2, b2], axis=-1) * 255.0
    return rotated - orig.astype(np.float32)


def _mid_freq_noise(H: int, W: int) -> np.ndarray:
    """Unit-std band-pass noise concentrated in mid spatial frequencies.

    Constructed as the difference of two Gaussian-blurred white noise images.
    Mid-frequency content survives JPEG recompression and median filtering
    (both of which act on the high-frequency end), making it persistent against
    common defender preprocessing stacks.
    """
    raw = np.random.randn(H, W, 3).astype(np.float32)
    # Encode into uint8 range so PIL can filter it; offset cancels in the difference
    pil = Image.fromarray(np.clip(raw * 40 + 128, 0, 255).astype(np.uint8))
    light = np.array(pil.filter(ImageFilter.GaussianBlur(radius=2))).astype(np.float32)
    heavy = np.array(pil.filter(ImageFilter.GaussianBlur(radius=20))).astype(np.float32)
    band = (light - heavy) / 40.0  # offset cancels; undo amplitude scaling
    std = band.std()
    return band / std if std > 1e-8 else band


# ── Main attacker class ────────────────────────────────────────────────────────


class NoiseManager:
    """CV-technique adversarial noiser. No surrogate models or gradients.

    Inside detected bboxes (RMSE ≤ 47, SSIM ≥ 0.31):
        Weighted combination of three attacks targeting different detection cues:
        - Gray overlay   → contrast/luminance features
        - Hue rotation   → color-based class features
        - Mid-freq noise → texture/edge features (survives JPEG and median filter)

    Outside detected bboxes (global RMSE ≤ 67):
        Gaussian noise spending the remaining global budget. If inside fraction
        f ≈ 0.2 at RMSE 47, the outside budget is:
            sqrt((67² − 0.2·47²) / 0.8) ≈ 71 RMSE
        — stronger per-pixel than the global limit, achieved by concentrating the
        residual budget on the non-object regions.

    SSIM inside is guaranteed ≥ 0.31 by a 12-step binary search (scale=0 always
    passes, so convergence is unconditional even for tiny bboxes or extreme noise).
    """

    def _inside_attack(
        self, orig: np.ndarray, boxes: list[list[float]], mask: np.ndarray
    ) -> np.ndarray:
        """Per-bbox pixelation + grayscale + gray pull, scaled to fill RMSE budget.

        Pixelation destroys spatial/edge features (drives SSIM toward the 0.31 floor).
        Grayscale removes colour-based class features.
        Gray pull reduces contrast so object boundaries wash out.
        All three attack independent detection cues; the SSIM binary search then
        finds the largest admissible scale (as close to 0.31 as possible).
        """
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
            small = crop.resize((max(1, cw // factor), max(1, ch // factor)), Image.NEAREST)
            pixelated = small.resize((cw, ch), Image.NEAREST)

            # Grayscale: strip colour channels that anchor class identity
            gray_crop = pixelated.convert("L").convert("RGB")

            augmented[y1:y2, x1:x2] = np.array(gray_crop, dtype=np.float32)

        # Structural delta (pixelate + grayscale) + additional gray pull for contrast
        struct_delta = augmented - orig.astype(np.float32)
        gray_pull = _gray_overlay(orig)
        combined = 0.7 * struct_delta + 0.3 * gray_pull

        delta[mask] = combined[mask]

        # Scale to fill RMSE budget; SSIM enforcement will scale back if needed
        rmse = _rmse_inside(delta, mask)
        if rmse > 1e-8:
            delta[mask] *= RMSE_INSIDE_MAX / rmse

        return delta

    def _enforce_ssim(
        self, orig: np.ndarray, delta: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """12-step binary search on inside scale to guarantee SSIM ≥ SSIM_INSIDE_MIN."""
        adv = np.clip(orig.astype(np.float32) + delta, 0, 255).astype(np.uint8)
        if _ssim_inside(orig, adv, mask) >= SSIM_INSIDE_MIN:
            return delta

        lo, hi = 0.0, 1.0
        best = np.zeros_like(delta)

        for _ in range(12):
            mid = (lo + hi) / 2.0
            d = delta * mid
            cand = np.clip(orig.astype(np.float32) + d, 0, 255).astype(np.uint8)
            if _ssim_inside(orig, cand, mask) >= SSIM_INSIDE_MIN:
                lo = mid
                best = d
            else:
                hi = mid

        return best

    def _outside_attack(
        self, H: int, W: int, mask: np.ndarray, inside_delta: np.ndarray
    ) -> np.ndarray:
        """Gaussian noise on outside pixels, scaled to consume the remaining global budget.

        Computes the exact outside RMSE target so that combined global RMSE = 67
        without any post-hoc projection — which would otherwise scale inside down.
        """
        n_total = H * W
        n_outside = int((~mask).sum())
        noise = np.random.randn(H, W, 3).astype(np.float32)
        noise[mask] = 0.0

        if n_outside == 0:
            return noise

        inside_sq_sum = float((inside_delta**2).mean(axis=-1)[mask].sum()) if mask.any() else 0.0
        remaining_sq = RMSE_GLOBAL_MAX**2 * n_total - inside_sq_sum
        if remaining_sq <= 0:
            return np.zeros_like(noise)

        target_rmse = float(np.sqrt(remaining_sq / n_outside))
        current_rmse = float(np.sqrt((noise**2).mean(axis=-1)[~mask].mean()))
        if current_rmse > 1e-8:
            noise *= target_rmse / current_rmse

        return noise

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
        boxes = _fetch_bboxes(b64)
        has_boxes = len(boxes) > 0
        mask = _make_mask(H, W, boxes)

        # Phase 1: inside-bbox attack (RMSE ≤ 47, SSIM ≥ 0.31)
        inside_delta = self._inside_attack(orig, boxes, mask)
        if has_boxes:
            inside_delta = self._enforce_ssim(orig, inside_delta, mask)

        # Phase 2: outside-bbox noise scaled to fill remaining global budget exactly.
        # No global projection — it would scale inside down and waste the budget.
        full_delta = self._outside_attack(H, W, mask, inside_delta)
        full_delta[mask] = inside_delta[mask]

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
    res = attacker.noise(img_data)
    img_bytes = base64.b64decode(res)
    Image.open(io.BytesIO(img_bytes)).convert("RGB").show()
