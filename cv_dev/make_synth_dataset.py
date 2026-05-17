from cv_dev.consts import (
    SYNTHETIC_JSON_PATH,
    SYNTHETIC_IMAGE_PATH,
    TRAIN_PATH,
    VAL_PATH,
)

import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
import numpy as np
from PIL import Image


def _train_transform() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(
                scale=(0.7, 1.3),
                translate_percent=(-0.1, 0.1),
                rotate=(-45, 45),
                shear=(-10, 10),
                interpolation=1,
                fill=0,
                p=0.7,
            ),
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomShadow(p=0.3),
            A.Blur(blur_limit=5, p=0.15),
            A.GaussianBlur(blur_limit=5, p=0.15),
            A.MotionBlur(blur_limit=7, p=0.15),
            A.GaussNoise(p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(10, 40),
                hole_width_range=(10, 40),
                fill=0,
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["category_ids"], min_visibility=0.3
        ),
    )


def _val_transform() -> A.Compose:
    return A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["category_ids"], min_visibility=0.3
        ),
    )


def _coco_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, ...]:
    x, y, w, h = bbox
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h


def _write_split(
    imgs: list,
    anns_by_image_id: dict,
    transform: A.Compose,
    img_dir: Path,
    lbl_dir: Path,
    split_name: str,
) -> None:
    for img_meta in tqdm(imgs, f"Writing synth {split_name}"):
        stem = Path(img_meta["file_name"]).stem
        ext = Path(img_meta["file_name"]).suffix
        new_stem = f"synth_{stem}"

        img_w, img_h = img_meta["width"], img_meta["height"]
        anns = anns_by_image_id.get(img_meta["id"], [])

        bboxes = []
        category_ids = []
        for ann in anns:
            bboxes.append(list(_coco_to_yolo(ann["bbox"], img_w, img_h)))
            category_ids.append(ann["category_id"])

        img = np.array(
            Image.open(SYNTHETIC_IMAGE_PATH / img_meta["file_name"]).convert("RGB")
        )
        result = transform(image=img, bboxes=bboxes, category_ids=category_ids)

        Image.fromarray(result["image"]).save(img_dir / f"{new_stem}{ext}")

        with open(lbl_dir / f"{new_stem}.txt", "w") as f:
            for cat_id, bbox in zip(result["category_ids"], result["bboxes"]):
                f.write(f"{cat_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


if __name__ == "__main__":
    train_img_dir = TRAIN_PATH / "images"
    train_lbl_dir = TRAIN_PATH / "labels"
    val_img_dir = VAL_PATH / "images"
    val_lbl_dir = VAL_PATH / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        assert d.exists(), f"{d} does not exist — run make_yolo_dataset.py first"

    with open(SYNTHETIC_JSON_PATH) as f:
        synth_data = json.load(f)

    anns_by_image_id: dict[int, list] = {}
    for ann in synth_data["annotations"]:
        anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

    train_imgs, val_imgs = train_test_split(
        synth_data["images"], test_size=0.2, random_state=42
    )

    _write_split(
        train_imgs,
        anns_by_image_id,
        _train_transform(),
        train_img_dir,
        train_lbl_dir,
        "train",
    )
    _write_split(
        val_imgs, anns_by_image_id, _val_transform(), val_img_dir, val_lbl_dir, "val"
    )

    print(
        f"Added {len(train_imgs)} train + {len(val_imgs)} val synth images → {TRAIN_PATH.parent}"
    )
