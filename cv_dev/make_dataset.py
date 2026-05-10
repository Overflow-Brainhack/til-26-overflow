from typing import final, override
import os
import shutil
import gc

from cv_dev.consts import (
    DATASETS_PATH,
    JSON_PATH,
    IMAGE_PATH,
    FIXED_WIDTH,
    FIXED_HEIGHT,
)
from cv_dev.utils import process_annotations
from cv_dev.data_types import ImageAnnotation, ProcessedAnnotation, Label

import torch
import json
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import numpy as np


@final
class ImageDataset(torch.utils.data.Dataset[tuple[torch.Tensor, Label]]):
    def __init__(self, X: list[torch.Tensor], y: list[list[ImageAnnotation]]) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    @override
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Label]:
        image = self.X[idx].float() / 255.0
        annotations = self.y[idx]

        if len(annotations) == 0:
            # Safe fallback if no annotations are found
            return image, {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.tensor([], dtype=torch.int64),
                "area": torch.tensor([], dtype=torch.float32),
                "iscrowd": torch.tensor([], dtype=torch.int64),
                "image_id": torch.tensor([idx]),
            }

        boxes = [ann["bbox"] for ann in annotations]
        labels = [ann["category_id"] + 1 for ann in annotations]
        area = [ann["area"] for ann in annotations]
        iscrowd = [ann.get("iscrowd", 0) for ann in annotations]
        image_id = annotations[0].get("image_id", idx)

        boxes = torch.tensor(
            [[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=torch.float32
        )

        labels = torch.tensor(labels, dtype=torch.int64)
        area = torch.tensor(area, dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([image_id])

        target: Label = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": image_id,
        }

        return image, target


def make_dataset(
    annotations: list[dict[str, list[ProcessedAnnotation]]],
    dataset_type: str,
) -> ImageDataset:
    images: list[torch.Tensor] = []
    labels: list[list[ImageAnnotation]] = []

    if dataset_type == "train":
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(
                    limit=45,
                    interpolation=1,
                    border_mode=0,
                    fill=(0, 0, 0),
                    fill_mask=(0, 0, 0),
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                    p=0.5,
                ),
                A.Blur(blur_limit=3, p=0.5),
                A.GaussNoise(p=0.5),
                A.Resize(width=FIXED_WIDTH, height=FIXED_HEIGHT, p=1),
                A.ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
        )
    else:
        transform = A.Compose(
            [A.Resize(width=FIXED_WIDTH, height=FIXED_HEIGHT, p=1), A.ToTensorV2()],
            bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
        )

    for annotation in tqdm(annotations, f"Making {dataset_type} dataset"):
        file_name = str(list(annotation.keys())[0])
        file_id = int(file_name)
        img = Image.open(IMAGE_PATH / (file_name + ".jpg"))
        bboxes: list[list[float]] = []
        category_ids: list[int] = []
        label: list[ImageAnnotation] = []

        for i in annotation[file_name]:
            bboxes.append(i["bbox"])
            category_ids.append(i["category_id"])

        transformed = transform(
            image=np.array(img), bboxes=bboxes, category_ids=category_ids
        )

        images.append(torch.Tensor(transformed["image"]))
        category_ids = transformed["category_ids"]
        bboxes = transformed["bboxes"]

        for category_id, bbox in zip(category_ids, bboxes):
            x, y, w, h = bbox
            label.append(
                {
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                    "area": w * h,
                    "image_id": file_id,
                }
            )
        labels.append(label)

    return ImageDataset(images, labels)


def save_dataset(dataset: ImageDataset, path: Path) -> None:
    torch.save(dataset, path)


def load_dataset(path: Path) -> ImageDataset:
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        annotations = data["annotations"]
        num_images = len(data["images"])

    # make smaller dataset
    # annotations = list(filter(lambda ann: ann["image_id"] < 500, annotations))
    # num_images = 500

    processed_annotations = process_annotations(annotations, num_images)
    train_annotations, val_annotations = train_test_split(
        processed_annotations, test_size=0.2, random_state=42
    )

    if os.path.exists(DATASETS_PATH):
        shutil.rmtree(DATASETS_PATH)
    os.mkdir(DATASETS_PATH)

    train_dataset = make_dataset(train_annotations, "train")
    save_dataset(train_dataset, DATASETS_PATH / "train.pt")

    del train_dataset
    _ = gc.collect()

    val_dataset = make_dataset(val_annotations, "val")
    save_dataset(val_dataset, DATASETS_PATH / "val.pt")
