from cv_dev.consts import (
    SYNTHETIC_JSON_PATH,
    SYNTHETIC_DATA_PATH,
    SYNTHETIC_TRAIN_PATH,
    SYNTHETIC_VAL_PATH,
)
from cv_dev.utils import process_annotations
from cv_dev.make_yolo_dataset import make_dataset, get_dimensions, write_yaml

import json
from sklearn.model_selection import train_test_split
import os
import shutil

if __name__ == "__main__":
    dimensions = get_dimensions(SYNTHETIC_JSON_PATH)
    with open(SYNTHETIC_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        annotations = data["annotations"]
        num_images = len(data["images"])

    processed_annotations = process_annotations(annotations, num_images)
    train_annotations, val_annotations = train_test_split(
        processed_annotations, test_size=0.2, random_state=42
    )

    if os.path.exists(SYNTHETIC_TRAIN_PATH):
        shutil.rmtree(SYNTHETIC_TRAIN_PATH)
    os.mkdir(SYNTHETIC_TRAIN_PATH)

    if os.path.exists(SYNTHETIC_VAL_PATH):
        shutil.rmtree(SYNTHETIC_VAL_PATH)
    os.mkdir(SYNTHETIC_VAL_PATH)

    os.mkdir(SYNTHETIC_TRAIN_PATH / "images")
    os.mkdir(SYNTHETIC_TRAIN_PATH / "labels")

    make_dataset(
        SYNTHETIC_TRAIN_PATH / "images",
        SYNTHETIC_TRAIN_PATH / "labels",
        train_annotations,
        dimensions["width"],
        dimensions["height"],
        no_aug=False,
    )

    os.mkdir(SYNTHETIC_VAL_PATH / "images")
    os.mkdir(SYNTHETIC_VAL_PATH / "labels")

    make_dataset(
        SYNTHETIC_VAL_PATH / "images",
        SYNTHETIC_VAL_PATH / "labels",
        val_annotations,
        dimensions["width"],
        dimensions["height"],
        no_aug=True,
    )

    write_yaml(SYNTHETIC_DATA_PATH / "data.yaml", SYNTHETIC_DATA_PATH)
