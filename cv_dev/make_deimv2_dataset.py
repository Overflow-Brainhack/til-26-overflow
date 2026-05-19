from cv_dev.consts import (
    JSON_PATH,
    IMAGE_PATH,
    DEIMV2_DATA_PATH,
)
from cv_dev.utils import process_annotations

from pathlib import Path
from tqdm import tqdm
import json
import shutil

from sklearn.model_selection import train_test_split


def prepare_deimv2_data() -> tuple[Path, Path, Path, Path]:
    """
    Creates a raw (unaugmented) 80/20 COCO-format train/val split under
    DEIMV2_DATA_PATH, using the identical split as make_yolo_dataset.py
    (process_annotations → train_test_split with random_state=42).
    DEIMv2 runs its own Mosaic/Mixup/CopyBlend augmentation pipeline.

    Returns (train_img_dir, train_json, val_img_dir, val_json).
    """
    train_img_dir = DEIMV2_DATA_PATH / "train"
    val_img_dir = DEIMV2_DATA_PATH / "val"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH) as f:
        data = json.load(f)

    all_images = data["images"]
    all_annotations = data["annotations"]
    categories = data["categories"]
    num_images = len(all_images)

    # Mirror the exact split from make_yolo_dataset.py
    processed = process_annotations(all_annotations, num_images)
    train_processed, val_processed = train_test_split(
        processed, test_size=0.2, random_state=42
    )

    train_ids = {int(k) for p in train_processed for k in p}
    val_ids = {int(k) for p in val_processed for k in p}

    id_to_image = {img["id"]: img for img in all_images}
    train_imgs = [id_to_image[i] for i in sorted(train_ids) if i in id_to_image]
    val_imgs = [id_to_image[i] for i in sorted(val_ids) if i in id_to_image]
    train_anns = [a for a in all_annotations if a["image_id"] in train_ids]
    val_anns = [a for a in all_annotations if a["image_id"] in val_ids]

    for imgs, dst_dir in [(train_imgs, train_img_dir), (val_imgs, val_img_dir)]:
        for img in tqdm(imgs, f"Copying images → {dst_dir.name}"):
            src = IMAGE_PATH / img["file_name"]
            dst = dst_dir / img["file_name"]
            if not dst.exists():
                shutil.copy(src, dst)

    train_json = DEIMV2_DATA_PATH / "train.json"
    val_json = DEIMV2_DATA_PATH / "val.json"

    with open(train_json, "w") as f:
        json.dump(
            {
                "images": train_imgs,
                "annotations": train_anns,
                "categories": categories,
            },
            f,
        )
    with open(val_json, "w") as f:
        json.dump(
            {
                "images": val_imgs,
                "annotations": val_anns,
                "categories": categories,
            },
            f,
        )

    print(
        f"DEIMv2 split: {len(train_imgs)} train / {len(val_imgs)} val → {DEIMV2_DATA_PATH}"
    )
    return train_img_dir, train_json, val_img_dir, val_json


if __name__ == "__main__":
    prepare_deimv2_data()
