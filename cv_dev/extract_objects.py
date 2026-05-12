from cv_dev.consts import IMAGE_PATH, JSON_PATH, CATEGORIES

from pathlib import Path
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

OBJECT_BANK_PATH = Path("cv_dev/object_bank")


def extract_objects() -> None:
    from ultralytics import SAM

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    file_by_id = {img["id"]: img["file_name"] for img in data["images"]}

    ann_by_image: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    for cat in CATEGORIES:
        (OBJECT_BANK_PATH / cat).mkdir(parents=True, exist_ok=True)

    # sam2_l.pt for best quality; swap to sam2_b.pt if VRAM is tight
    model = SAM("sam2_l.pt")

    for image_id, annotations in tqdm(ann_by_image.items(), "Extracting objects"):
        img_pil = Image.open(IMAGE_PATH / file_by_id[image_id]).convert("RGB")
        img_np = np.array(img_pil)
        H, W = img_np.shape[:2]

        bboxes: list[list[int]] = []
        valid_anns: list[dict] = []
        for ann in annotations:
            x, y, w, h = [int(v) for v in ann["bbox"]]
            x, y = max(0, x), max(0, y)
            w = min(w, W - x)
            h = min(h, H - y)
            if w <= 0 or h <= 0:
                continue
            bboxes.append([x, y, x + w, y + h])
            valid_anns.append(ann)

        if not bboxes:
            continue

        # SAM2 segments the full image using bbox prompts — much better than
        # rembg on tight crops because the model sees full context for edges
        results = model(img_np, bboxes=bboxes, verbose=False)

        if not results or results[0].masks is None:
            continue

        # masks.data: (N, H, W) float32 on GPU/CPU
        masks = results[0].masks.data.cpu().numpy()

        for i, (ann, (x1, y1, x2, y2)) in enumerate(zip(valid_anns, bboxes)):
            if i >= len(masks):
                continue

            mask_full = masks[i] > 0.5  # bool (H, W)
            crop_rgb = img_np[y1:y2, x1:x2]
            crop_mask = mask_full[y1:y2, x1:x2]

            alpha = (crop_mask * 255).astype(np.uint8)
            rgba = np.dstack([crop_rgb, alpha])

            category_name = CATEGORIES[ann["category_id"]]
            out_path = OBJECT_BANK_PATH / category_name / f"{ann['id']}.png"
            Image.fromarray(rgba, "RGBA").save(out_path)


if __name__ == "__main__":
    extract_objects()
