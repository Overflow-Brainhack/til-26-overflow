"""CV-based adversarial image noising — no surrogate models or gradients required."""

from __future__ import annotations

import base64
import io
import json
import random
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity


# ── Budget constants (match eval_thresholds_v2.yaml) ──────────────────────────
RMSE_GLOBAL_MAX = 67.0  # evaluator's global threshold
RMSE_INSIDE_MAX = (
    44.0  # 6-unit margin: JPEG can amplify pixelation block edges, pushing RMSE up
)
SSIM_INSIDE_MIN = (
    0.36  # post-JPEG target; extra margin above 0.30 for outside-noise boundary effects
)


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

    def _inside_attack(
        self, orig: np.ndarray, boxes: list[list[float]], mask: np.ndarray
    ) -> np.ndarray:
        """Per-bbox pixelation + gray pull + inversion, scaled to fill RMSE budget."""
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

        # Structural delta (pixelate + grayscale) + gray pull + colour inversion
        # Inversion (255 - orig) is maximally wrong for colour-based class features
        # while preserving luminance structure (low SSIM cost relative to mAP impact)
        struct_delta = augmented - orig.astype(np.float32)
        gray_pull = _gray_overlay(orig)
        invert_delta = (255.0 - orig.astype(np.float32)) - orig.astype(
            np.float32
        )  # = 255 - 2*orig
        combined = 0.6 * struct_delta + 0.25 * gray_pull + 0.15 * invert_delta

        delta[mask] = combined[mask]

        # Scale to fill RMSE budget; SSIM enforcement will scale back if needed
        rmse = _rmse_inside(delta, mask)
        if rmse > 1e-8:
            delta[mask] *= RMSE_INSIDE_MAX / rmse

        return delta

    def _enforce_ssim(
        self, orig: np.ndarray, delta: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """12-step binary search on inside scale to guarantee post-JPEG SSIM ≥ SSIM_INSIDE_MIN.

        Checks SSIM on the JPEG-encoded output rather than raw uint8, because pixelation
        creates hard block edges that JPEG ringing artifacts worsen post-encoding.
        Checking pre-JPEG would consistently underestimate the evaluator's actual SSIM.
        """

        def _ssim_post_jpeg(d: np.ndarray) -> float:
            adv = np.clip(orig.astype(np.float32) + d, 0, 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(adv).save(buf, format="JPEG", quality=95)
            buf.seek(0)
            adv_jpeg = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
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
        if not self._BANK_PATH.exists():
            print(f"Cannot find {self._BANK_PATH} for object bank spamming.")
            return
        categories = [d for d in self._BANK_PATH.iterdir() if d.is_dir()]
        if not categories:
            return

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

            cat_dir = random.choice(categories)
            pngs = list(cat_dir.glob("*.png"))
            if not pngs:
                continue

            try:
                rgba = np.array(
                    Image.open(random.choice(pngs))
                    .convert("RGBA")
                    .resize((TILE, TILE), Image.LANCZOS),
                    dtype=np.float32,
                )
            except Exception:
                continue

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
        boxes = _fetch_bboxes(b64)
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
