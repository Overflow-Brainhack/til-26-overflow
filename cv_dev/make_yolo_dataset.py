from cv_dev.consts import (
    CATEGORIES,
    NUM_CATEGORIES,
    IMAGE_PATH,
    JSON_PATH,
    DATA_PATH,
    TRAIN_PATH,
    VAL_PATH,
)
from cv_dev.utils import process_annotations
from cv_dev.data_types import ProcessedAnnotation

import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import shutil
import yaml
from PIL import Image
import albumentations as A
import numpy as np


def get_dimensions(path: Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        images = json.load(f)["images"]
    widths: set[int] = set()
    heights: set[int] = set()
    for file in tqdm(images, "Reading dimensions"):
        widths.add(file["width"])
        heights.add(file["height"])

    assert len(widths) == 1, Exception(
        f"Assumed widths to all be the same, got widths of {widths}"
    )
    assert len(heights) == 1, Exception(
        f"Assumed heights to all be the same, got heights of {heights}"
    )

    return {"width": list(widths)[0], "height": list(heights)[0]}


def make_dataset(
    images_path: Path,
    labels_path: Path,
    annotations: list[dict[str, list[ProcessedAnnotation]]],
    im_w: int,
    im_h: int,
    no_aug: bool = False,
) -> None:
    if no_aug:
        transform = A.Compose(
            [
                A.NoOp(),
            ],
            bbox_params=A.BboxParams(
                format="yolo", label_fields=["category_ids"], min_visibility=0.3
            ),
        )
    else:
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                # scale jitter ±30% + rotation + shift + shear; keeps output size constant
                A.Affine(
                    scale=(0.7, 1.3),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-45, 45),
                    shear=(-10, 10),
                    interpolation=1,
                    fill=0,
                    p=0.7,
                ),
                # perspective distortion — composited objects often lack this
                A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.6
                ),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                # shadows to simulate lighting mismatch common in composited images
                A.RandomShadow(p=0.3),
                A.Blur(blur_limit=5, p=0.15),
                A.GaussianBlur(blur_limit=5, p=0.15),
                A.MotionBlur(blur_limit=7, p=0.15),
                A.GaussNoise(p=0.3),
                A.CLAHE(clip_limit=4.0, p=0.2),
                # simulate partial occlusion
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

    for annotation in tqdm(
        annotations, f"Making dataset in {images_path}", colour="Blue"
    ):
        file_name = list(annotation.keys())[0]
        img = Image.open(IMAGE_PATH / (file_name + ".jpg"))
        bboxes: list[list[float]] = []
        category_ids: list[int] = []

        for i in annotation[file_name]:
            coco_x, coco_y, coco_w, coco_h = i["bbox"]

            yolo_x = (coco_x + coco_w / 2) / im_w
            yolo_y = (coco_y + coco_h / 2) / im_h
            yolo_w = coco_w / im_w
            yolo_h = coco_h / im_h
            bboxes.append([yolo_x, yolo_y, yolo_w, yolo_h])
            category_ids.append(i["category_id"])

        transformed = transform(
            image=np.array(img), category_ids=category_ids, bboxes=bboxes
        )
        img = transformed["image"]
        bboxes = transformed["bboxes"]
        category_ids = transformed["category_ids"]

        with open(labels_path / (file_name + ".txt"), "w") as f:
            for category_id, bbox in zip(category_ids, bboxes):
                yolo_x, yolo_y, yolo_w, yolo_h = bbox
                f.write(f"{category_id} {yolo_x} {yolo_y} {yolo_w} {yolo_h}\n")

        img = Image.fromarray(img)
        img.save(images_path / (file_name + ".jpg"))


def write_yaml(yaml_path: Path) -> None:
    with open(yaml_path, "w") as f:
        yaml.dump(
            {
                "path": str(DATA_PATH),
                "train": "train/images",
                "val": "val/images",
                "names": dict(enumerate(CATEGORIES)),
                "nc": NUM_CATEGORIES,
            },
            f,
        )


if __name__ == "__main__":
    dimensions = get_dimensions(JSON_PATH)
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        annotations = data["annotations"]
        num_images = len(data["images"])

    processed_annotations = process_annotations(annotations, num_images)
    train_annotations, val_annotations = train_test_split(
        processed_annotations, test_size=0.2, random_state=42
    )

    if os.path.exists(TRAIN_PATH):
        shutil.rmtree(TRAIN_PATH)
    os.mkdir(TRAIN_PATH)

    if os.path.exists(VAL_PATH):
        shutil.rmtree(VAL_PATH)
    os.mkdir(VAL_PATH)

    os.mkdir(TRAIN_PATH / "images")
    os.mkdir(TRAIN_PATH / "labels")

    make_dataset(
        TRAIN_PATH / "images",
        TRAIN_PATH / "labels",
        train_annotations,
        dimensions["width"],
        dimensions["height"],
        no_aug=False,
    )

    os.mkdir(VAL_PATH / "images")
    os.mkdir(VAL_PATH / "labels")

    make_dataset(
        VAL_PATH / "images",
        VAL_PATH / "labels",
        val_annotations,
        dimensions["width"],
        dimensions["height"],
        no_aug=True,
    )

    write_yaml(DATA_PATH / "data.yaml")
