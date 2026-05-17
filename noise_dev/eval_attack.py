"""Batch evaluation of noise-server attack effectiveness against yolo11x-finetuned.

Usage (noise server must be running on :5003):
    python -m noise_dev.eval_attack
    python -m noise_dev.eval_attack --n 200 --seed 7

What it measures
----------------
  1. Detection drop  — how many objects the target model can no longer find.
  2. Confidence shift — mean confidence before vs after.
  3. Pixel metrics   — global RMSE, inside RMSE, inside SSIM (fairness budget).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import random
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm
from ultralytics import YOLO, RTDETR

try:
    from faster_coco_eval import COCO as CocoAPI
    from faster_coco_eval.core.faster_eval_api import COCOeval_faster as COCOeval
except ImportError:
    try:
        from pycocotools.coco import COCO as CocoAPI
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        CocoAPI = None
        COCOeval = None

# TARGET_MODEL = "noise_dev/yolo11x-finetuned.pt"
TARGET_MODEL = "cv/models/rtdetr-l-70.pt"
NOISE_SERVER = "http://localhost:5003/noise"
DATA_DIR = Path("data/cv")
CONF_THRESH = 0.25
RMSE_GLOBAL_BUDGET = 67.0
RMSE_INSIDE_BUDGET = 50.0
SSIM_INSIDE_BUDGET = 0.30


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def build_coco_gt(annotations: dict, image_ids: set) -> "CocoAPI":
    coco = CocoAPI()
    coco.dataset = {
        "images": [img for img in annotations["images"] if img["id"] in image_ids],
        "annotations": [
            a for a in annotations["annotations"] if a["image_id"] in image_ids
        ],
        "categories": annotations.get("categories", []),
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def compute_map(coco_gt: "CocoAPI", preds: list[dict]) -> float:
    if not preds:
        return 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])


def yolo_detect(model: YOLO | RTDETR, img: Image.Image) -> list[dict]:
    results = model(img, imgsz=1280, rect=True, conf=CONF_THRESH, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    return [
        {"conf": float(c), "cls": int(cl), "xyxy": box.tolist()}
        for c, cl, box in zip(boxes.conf.cpu(), boxes.cls.cpu(), boxes.xyxy.cpu())
    ]


def request_noise(img_bytes: bytes) -> bytes | None:
    b64 = base64.b64encode(img_bytes).decode("ascii")
    try:
        resp = requests.post(
            NOISE_SERVER,
            json={"instances": [{"key": 0, "b64": b64}]},
            timeout=120,
        )
        resp.raise_for_status()
        return base64.b64decode(resp.json()["predictions"][0])
    except Exception as e:
        print(f"  [noise server] {e}")
        return None


def pixel_metrics(
    orig_np: np.ndarray,
    noised_np: np.ndarray,
    gt_boxes: list[list[float]],
) -> dict:
    diff = orig_np.astype(np.float32) - noised_np.astype(np.float32)
    rmse_global = float(np.sqrt(np.mean(diff**2)))

    H, W = orig_np.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    for x, y, w, h in gt_boxes:
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True

    if mask.any():
        sq_per_pixel = (diff**2).mean(axis=-1)
        rmse_inside = float(np.sqrt(sq_per_pixel[mask].mean()))
        _, ssim_map = structural_similarity(
            orig_np.astype(np.float32),
            noised_np.astype(np.float32),
            channel_axis=2,
            data_range=255.0,
            win_size=7,
            full=True,
        )
        if ssim_map.ndim == 3:
            ssim_map = ssim_map.mean(axis=-1)
        ssim_inside = float(ssim_map[mask].mean())
    else:
        rmse_inside = rmse_global
        ssim_inside = float(
            structural_similarity(
                orig_np.astype(np.float32),
                noised_np.astype(np.float32),
                channel_axis=2,
                data_range=255.0,
                win_size=7,
            )
        )

    passes = (
        rmse_global <= RMSE_GLOBAL_BUDGET
        and rmse_inside <= RMSE_INSIDE_BUDGET
        and ssim_inside >= SSIM_INSIDE_BUDGET
    )
    return {
        "rmse_global": rmse_global,
        "rmse_inside": rmse_inside,
        "ssim_inside": ssim_inside,
        "passes": passes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=100, help="Number of images to test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading target model: {TARGET_MODEL}")
    # model = YOLO(TARGET_MODEL)
    model = RTDETR(TARGET_MODEL)

    with open(DATA_DIR / "annotations.json") as f:
        annotations = json.load(f)

    id_to_boxes: dict[int, list] = {}
    for ann in annotations["annotations"]:
        id_to_boxes.setdefault(ann["image_id"], []).append(ann["bbox"])

    # Map model class index (0-based) → COCO category_id
    cat_ids = sorted(set(a["category_id"] for a in annotations["annotations"]))
    idx_to_cat_id = dict(enumerate(cat_ids))

    sample = annotations["images"][:]
    random.shuffle(sample)
    sample = sample[: args.n]

    rows: list[dict] = []
    done = 0
    skipped = 0
    coco_preds_before: list[dict] = []
    coco_preds_after: list[dict] = []
    sample_ids: set[int] = set()

    for img_info in tqdm(sample, desc="Evaluating"):
        img_id = img_info["id"]
        img_path = DATA_DIR / "images" / img_info["file_name"]
        img_bytes = img_path.read_bytes()
        orig_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        orig_np = np.array(orig_pil)
        gt_boxes = id_to_boxes.get(img_id, [])

        dets_before = yolo_detect(model, orig_pil)

        noised_bytes = request_noise(img_bytes)
        if noised_bytes is None:
            skipped += 1
            continue

        done += 1

        noised_pil = Image.open(io.BytesIO(noised_bytes)).convert("RGB")
        noised_pil.save(
            f"/home/shadowmachete/dev/til-26-overflow/results/adv_images/noised_{done}.jpg",
            format="JPEG",
        )

        noised_np = np.array(noised_pil)
        dets_after = yolo_detect(model, noised_pil)

        metrics = pixel_metrics(orig_np, noised_np, gt_boxes)

        sample_ids.add(img_id)
        for det in dets_before:
            x1, y1, x2, y2 = det["xyxy"]
            coco_preds_before.append(
                {
                    "image_id": img_id,
                    "category_id": idx_to_cat_id.get(det["cls"], det["cls"]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": det["conf"],
                }
            )
        for det in dets_after:
            x1, y1, x2, y2 = det["xyxy"]
            coco_preds_after.append(
                {
                    "image_id": img_id,
                    "category_id": idx_to_cat_id.get(det["cls"], det["cls"]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": det["conf"],
                }
            )

        rows.append(
            {
                "file": img_info["file_name"],
                "dets_before": len(dets_before),
                "dets_after": len(dets_after),
                "conf_before": [d["conf"] for d in dets_before],
                "conf_after": [d["conf"] for d in dets_after],
                **metrics,
            }
        )

    n = len(rows)
    if n == 0:
        print("No results — is the noise server running?")
        return

    # ── aggregate ──────────────────────────────────────────────────────────────
    det_b = np.array([r["dets_before"] for r in rows], dtype=float)
    det_a = np.array([r["dets_after"] for r in rows], dtype=float)
    conf_b = [c for r in rows for c in r["conf_before"]]
    conf_a = [c for r in rows for c in r["conf_after"]]
    rmse_g = np.array([r["rmse_global"] for r in rows])
    rmse_i = np.array([r["rmse_inside"] for r in rows])
    ssim_i = np.array([r["ssim_inside"] for r in rows])
    passes = np.array([r["passes"] for r in rows])

    mean_b = det_b.mean()
    mean_a = det_a.mean()
    pct_drop = 100.0 * (1.0 - mean_a / max(mean_b, 1e-6))
    fully_blind = int((det_a == 0).sum())
    conf_drop = (
        (np.mean(conf_b) - np.mean(conf_a)) if conf_b and conf_a else float("nan")
    )

    W = 58
    print(f"\n{'═' * W}")
    print(f"  Attack Effectiveness  —  {n} images  |  target: {TARGET_MODEL}")
    print(f"{'─' * W}")
    print(f"  Detections before :  {mean_b:.2f} ± {det_b.std():.2f}")
    print(f"  Detections after  :  {mean_a:.2f} ± {det_a.std():.2f}")
    print(f"  Detection drop    :  {pct_drop:.1f}%")
    print(f"  Fully blind images:  {fully_blind}/{n} ({100 * fully_blind / n:.1f}%)")
    print(f"{'─' * W}")
    if conf_b:
        print(f"  Mean conf before  :  {np.mean(conf_b):.3f}")
    if conf_a:
        print(f"  Mean conf after   :  {np.mean(conf_a):.3f}  (Δ {conf_drop:+.3f})")
    else:
        print(f"  Mean conf after   :  — (no detections remaining)")
    print(f"{'─' * W}")
    assert CocoAPI, (
        "Missing coco for evaluation, install faster_coco_eval or pycocotools"
    )
    print(f"  Computing mAP@0.5:0.95 ...", end="", flush=True)
    coco_gt = build_coco_gt(annotations, sample_ids)
    map_b = compute_map(coco_gt, coco_preds_before)
    map_a = compute_map(coco_gt, coco_preds_after)
    print(f"\r  mAP@.5:.95 before :  {map_b:.4f}")
    print(f"  mAP@.5:.95 after  :  {map_a:.4f}  (Δ {map_a - map_b:+.4f})")
    print(f"{'─' * W}")
    print(f"  Pixel metrics (mean ± std):")
    print(
        f"    RMSE global : {rmse_g.mean():.2f} ± {rmse_g.std():.2f}  (budget ≤ {RMSE_GLOBAL_BUDGET:.0f})"
    )
    print(
        f"    RMSE inside : {rmse_i.mean():.2f} ± {rmse_i.std():.2f}  (budget ≤ {RMSE_INSIDE_BUDGET:.0f})"
    )
    print(
        f"    SSIM inside : {ssim_i.mean():.4f} ± {ssim_i.std():.4f}  (budget ≥ {SSIM_INSIDE_BUDGET:.2f})"
    )
    print(f"    Pass rate   : {passes.mean() * 100:.1f}%  ({passes.sum()}/{n} images)")
    if skipped:
        print(f"    Skipped     : {skipped} (server errors)")

    # ── worst-case images ──────────────────────────────────────────────────────
    failing = [r for r in rows if not r["passes"]]
    if failing:
        print(f"{'─' * W}")
        print(f"  Fairness failures ({len(failing)} images):")
        for r in sorted(failing, key=lambda x: x["rmse_inside"], reverse=True)[:5]:
            print(
                f"    {r['file']:25s}  rmse_g={r['rmse_global']:.1f}  "
                f"rmse_i={r['rmse_inside']:.1f}  ssim_i={r['ssim_inside']:.3f}"
            )

    # ── diagnosis ─────────────────────────────────────────────────────────────
    print(f"{'─' * W}")
    if abs(map_a - map_b) < 0.10:
        print("  ⚠  Attack is barely fooling the model (<10% drop).")
    elif abs(map_a - map_b) < 0.30:
        print("  ~  Attack is partially effective.")
    else:
        print("  ✓  Attack is significantly reducing detections.")
    print(f"{'═' * W}\n")


if __name__ == "__main__":
    main()
