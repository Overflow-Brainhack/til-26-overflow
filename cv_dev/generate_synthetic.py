from cv_dev.consts import (
    BACKGROUNDS_PATH,
    OBJECT_BANK_PATH,
    CATEGORIES,
    CATEGORY_ANNOTATION_COUNTS,
    SYNTHETIC_IMAGE_PATH,
    SYNTHETIC_JSON_PATH,
    SYNTH_MAX_OBJECTS,
    SYNTH_MIN_SCALE,
    SYNTH_MAX_SCALE,
    SYNTH_MAX_PLACEMENT_TRIES,
)

import json
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

TRAIN_DATA_WIDTH = 1920
TRAIN_DATA_HEIGHT = 1080


def _intersects(a: list[float], b: list[float]) -> bool:
    """True if two COCO bboxes [x, y, w, h] overlap at all."""
    ax2 = a[0] + a[2]
    ay2 = a[1] + a[3]
    bx2 = b[0] + b[2]
    by2 = b[1] + b[3]
    return not (ax2 <= b[0] or bx2 <= a[0] or ay2 <= b[1] or by2 <= a[1])


def _find_placement(
    obj_w: int,
    obj_h: int,
    placed: list[list[float]],
) -> tuple[int, int] | None:
    for _ in range(SYNTH_MAX_PLACEMENT_TRIES):
        x = random.randint(0, max(0, TRAIN_DATA_WIDTH - obj_w))
        y = random.randint(0, max(0, TRAIN_DATA_HEIGHT - obj_h))
        candidate = [float(x), float(y), float(obj_w), float(obj_h)]
        if not any(_intersects(candidate, p) for p in placed):
            return x, y
    return None


def generate_synthetic() -> None:
    SYNTHETIC_IMAGE_PATH.mkdir(parents=True, exist_ok=True)

    # Build object bank: category_name → list of image paths
    bank: dict[str, list[Path]] = {}
    for cat in CATEGORIES:
        cat_path = OBJECT_BANK_PATH / cat
        if cat_path.exists():
            paths = list(cat_path.glob("*.png"))
            if paths:
                bank[cat] = paths

    available_cats = list(bank.keys())
    if not available_cats:
        print("Object bank is empty — run extract_objects.py first.")
        return

    # Inverse-frequency weights from original annotation counts (not deduped bank size)
    # so rare categories like cruise ship (205) are sampled ~12× more than fighter jet (2469)
    inv_counts = [1.0 / CATEGORY_ANNOTATION_COUNTS.get(c, 1) for c in available_cats]
    total_inv = sum(inv_counts)
    cat_weights = [w / total_inv for w in inv_counts]

    print("\nOriginal annotation counts (from dataset):")
    for cat in CATEGORIES:
        print(f"  {cat:<22} {CATEGORY_ANNOTATION_COUNTS.get(cat, '?'):>5}")

    placed_per_cat: dict[str, int] = {c: 0 for c in CATEGORIES}

    backgrounds = sorted(BACKGROUNDS_PATH.glob("*.jpg"))
    if not backgrounds:
        print("No backgrounds found — run scrape_backgrounds.py first.")
        return

    images_meta: list[dict] = []
    annotations: list[dict] = []
    ann_id = 0

    for image_id, bg_path in enumerate(
        tqdm(backgrounds, "Generating synthetic images", colour="Green")
    ):
        bg = (
            Image.open(bg_path)
            .convert("RGB")
            .resize((TRAIN_DATA_WIDTH, TRAIN_DATA_HEIGHT), Image.Resampling.LANCZOS)
        )

        n_objects = random.randint(0, SYNTH_MAX_OBJECTS)
        placed_bboxes: list[list[float]] = []

        for _ in range(n_objects):
            cat = random.choices(available_cats, weights=cat_weights, k=1)[0]
            obj = Image.open(random.choice(bank[cat])).convert("RGBA")

            # Scale object so its width is SYNTH_MIN_SCALE–SYNTH_MAX_SCALE of bg width
            scale = random.uniform(SYNTH_MIN_SCALE, SYNTH_MAX_SCALE)
            target_w = max(1, int(TRAIN_DATA_WIDTH * scale))
            aspect = obj.height / obj.width if obj.width > 0 else 1.0
            target_h = max(1, int(target_w * aspect))

            # Never upscale beyond the object's natural resolution
            if target_w > obj.width:
                target_w, target_h = obj.width, obj.height

            obj_resized = obj.resize((target_w, target_h), Image.Resampling.LANCZOS)

            placement = _find_placement(target_w, target_h, placed_bboxes)
            if placement is None:
                continue

            x, y = placement
            bg.paste(obj_resized, (x, y), mask=obj_resized.split()[3])

            bbox = [float(x), float(y), float(target_w), float(target_h)]
            placed_bboxes.append(bbox)
            placed_per_cat[cat] += 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": CATEGORIES.index(cat),
                    "area": float(target_w * target_h),
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        bg.save(SYNTHETIC_IMAGE_PATH / f"{image_id}.jpg", "JPEG", quality=95)
        images_meta.append(
            {
                "id": image_id,
                "width": TRAIN_DATA_WIDTH,
                "height": TRAIN_DATA_HEIGHT,
                "file_name": f"{image_id}.jpg",
            }
        )

    coco = {
        "images": images_meta,
        "annotations": annotations,
        "categories": [{"id": i, "name": c} for i, c in enumerate(CATEGORIES)],
    }
    with open(SYNTHETIC_JSON_PATH, "w") as f:
        json.dump(coco, f)

    print(
        f"\nDone. {len(images_meta)} images, {ann_id} annotations → {SYNTHETIC_JSON_PATH}"
    )
    print(f"\n{'Category':<22} {'Original':>8} {'Generated':>10}")
    print("-" * 42)
    for cat in CATEGORIES:
        orig = CATEGORY_ANNOTATION_COUNTS.get(cat, 0)
        gen = placed_per_cat.get(cat, 0)
        print(f"  {cat:<20} {orig:>8} {gen:>10}")


if __name__ == "__main__":
    generate_synthetic()
