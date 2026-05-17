from cv_dev.consts import (
    DATA_PATH,
    JSON_PATH,
    RESULTS_PATH,
    TRAIN_OUTPUT,
)

import argparse
import base64
import json
import math
import os
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence
from collections import defaultdict
from typing import Any
import itertools

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import torch
import torch.nn.functional as F
from ultralytics import RTDETR

os.environ["WANDB_DISABLED"] = "true"

DEFAULT_EPSILON = 8 / 255  # standard FGSM magnitude
BATCH_SIZE = 4


class COCOPatched(COCO):
    def __init__(self, annotations):
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        assert type(annotations) == dict, (
            f"Annotation format {type(annotations)} not supported"
        )
        print("Annotations loaded.")
        self.dataset = annotations
        self.createIndex()


def sample_generator(
    instances: Sequence[Mapping[str, Any]],
    data_dir: Path,
) -> Iterator[Mapping[str, Any]]:
    for instance in instances:
        with open(data_dir / "images" / instance["file_name"], "rb") as img_file:
            img_data = img_file.read()
            yield {
                "key": instance["id"],
                "b64": base64.b64encode(img_data).decode("ascii"),
            }


def score_cv(preds: Sequence[Mapping[str, Any]], ground_truth: Any) -> float:
    if not preds:
        return 0.0

    ground_truth = COCOPatched(ground_truth)
    results = ground_truth.loadRes(preds)
    coco_eval = COCOeval(ground_truth, results, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0].item()


def apply_fgsm(
    raw_model: torch.nn.Module,
    im: Image.Image,
    epsilon: float,
    device: str,
) -> Image.Image:
    """
    Fast Gradient Sign Method adversarial perturbation.

    Computes gradients at the model's inference resolution (1280x1280), maps the
    gradient sign back to the original image dimensions, then applies the perturbation
    there so that inference coordinate outputs stay in the original image's space.
    """
    orig_w, orig_h = im.size
    imgsz = 1280

    im_rgb = im.convert("RGB")
    im_resized = im_rgb.resize((imgsz, imgsz), Image.BILINEAR)
    img_np = np.array(im_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)

    try:
        raw_model.eval()
        with torch.enable_grad():
            out = raw_model(img_tensor)

            if isinstance(out, (list, tuple)):
                pred = next((o for o in out if isinstance(o, torch.Tensor)), None)
            elif isinstance(out, torch.Tensor):
                pred = out
            else:
                return im

            if pred is None:
                return im

            # RTDETR decoder output: [batch, num_queries, 4 + num_classes]
            # where first 4 are box coords and the rest are class logits.
            # We maximise detection confidence as the proxy loss, then step
            # opposite to the gradient to minimise it (i.e. evade detection).
            if pred.dim() == 3 and pred.shape[-1] > 4:
                confidence = pred[..., 4:].sigmoid().max(dim=-1).values.sum()
            else:
                confidence = pred.abs().sum()

            confidence.backward()
    except Exception as e:
        print(f"  [FGSM] gradient computation failed ({e}), skipping perturbation")
        return im

    if img_tensor.grad is None:
        return im

    # Resize gradient sign back to original image space
    grad_sign = img_tensor.grad.sign()  # [1, 3, imgsz, imgsz]
    if (orig_h, orig_w) != (imgsz, imgsz):
        grad_sign = F.interpolate(grad_sign, size=(orig_h, orig_w), mode="nearest")

    orig_np = np.array(im_rgb, dtype=np.float32) / 255.0
    orig_tensor = torch.from_numpy(orig_np).permute(2, 0, 1).unsqueeze(0)

    # Step in -gradient direction to minimise detection confidence
    perturbed = (orig_tensor - epsilon * grad_sign.cpu()).clamp(0, 1)

    perturbed_np = (
        (perturbed.squeeze(0).permute(1, 2, 0).numpy() * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return Image.fromarray(perturbed_np)


def run_ultralytics(model, batch):
    for b in batch:
        bytes_array = base64.b64decode(b["b64"])
        im = Image.open(BytesIO(bytes_array))
        results = model(im, verbose=False, imgsz=1280, rect=True)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            yield {
                "image_id": b["key"],
                "score": 1.0,
                "category_id": int(box.cls[0]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            }


def run_ultralytics_fgsm(model, raw_model, batch, epsilon, device):
    for b in batch:
        bytes_array = base64.b64decode(b["b64"])
        im = Image.open(BytesIO(bytes_array)).convert("RGB")

        perturbed_im = apply_fgsm(raw_model, im, epsilon, device)

        results = model(perturbed_im, verbose=False, imgsz=1280, rect=True)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            yield {
                "image_id": b["key"],
                "score": 1.0,
                "category_id": int(box.cls[0]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model robustness against FGSM adversarial attack"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"FGSM perturbation magnitude (default: {DEFAULT_EPSILON:.4f} = 8/255)",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip clean baseline evaluation and only run FGSM",
    )
    args = parser.parse_args()

    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, "r") as f:
        annotations = json.load(f)

    val_labels_path = Path("data/cv/val/labels")
    val_image_ids = set()
    if val_labels_path.exists():
        for label_file in val_labels_path.glob("*.txt"):
            val_image_ids.add(label_file.stem)

    annotations["images"] = [
        img for img in annotations["images"] if str(img["id"]) in val_image_ids
    ]
    annotations["annotations"] = [
        ann
        for ann in annotations["annotations"]
        if str(ann["image_id"]) in val_image_ids
    ]

    instances = annotations["images"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device:       {device}")
    print(f"Val images:   {len(instances)}")
    print(f"FGSM epsilon: {args.epsilon:.4f}  ({round(args.epsilon * 255)}/255)")

    model = RTDETR(str(TRAIN_OUTPUT / "rtdetr-l-finetuned-adv" / "weights" / "best.pt"))
    model.eval()
    model.to(device)
    raw_model = model.model
    raw_model.eval()

    # --- Baseline (clean images) ---
    baseline_map = None
    if not args.no_baseline:
        print("\n[1/2] Baseline inference (no attack)...")
        batch_gen = itertools.batched(
            sample_generator(instances, DATA_PATH), n=BATCH_SIZE
        )
        baseline_results = []
        for batch in tqdm(batch_gen, total=math.ceil(len(instances) / BATCH_SIZE)):
            baseline_results.extend(list(run_ultralytics(model, batch)))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Scoring baseline...")
        baseline_map = score_cv(baseline_results, annotations)
        print(f"Baseline mAP@.5:.05:.95: {baseline_map:.4f}")

    # --- FGSM attack ---
    step_label = "2/2" if not args.no_baseline else "1/1"
    print(f"\n[{step_label}] FGSM attack (epsilon={args.epsilon:.4f})...")
    fgsm_results = []
    for sample in tqdm(sample_generator(instances, DATA_PATH), total=len(instances)):
        fgsm_results.extend(
            list(run_ultralytics_fgsm(model, raw_model, [sample], args.epsilon, device))
        )

    adv_results_path = RESULTS_PATH / "cv_adv_results.json"
    with open(adv_results_path, "w") as f:
        json.dump(fgsm_results, f)
    print(f"Saved adversarial results to {adv_results_path}")

    print("Scoring FGSM results...")
    fgsm_map = score_cv(fgsm_results, annotations)
    print(f"FGSM mAP@.5:.05:.95:   {fgsm_map:.4f}")

    print("\n=== Adversarial Robustness Summary ===")
    if baseline_map is not None:
        drop = baseline_map - fgsm_map
        pct = (1 - fgsm_map / baseline_map) * 100 if baseline_map > 0 else 0.0
        print(f"Baseline mAP:  {baseline_map:.4f}")
        print(f"FGSM mAP:      {fgsm_map:.4f}")
        print(f"mAP drop:      {drop:.4f}  ({pct:.1f}% reduction)")
    else:
        print(f"FGSM mAP:      {fgsm_map:.4f}")


if __name__ == "__main__":
    main()
