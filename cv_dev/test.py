from cv_dev.consts import (
    DATA_PATH,
    JSON_PATH,
    RESULTS_PATH,
    NUM_CATEGORIES,
    TRAIN_OUTPUT,
)

import base64
import json
import math
import os
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence
from collections import defaultdict
from typing import Any
import itertools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import torch
from ultralytics import RTDETR, YOLO

os.environ["WANDB_DISABLED"] = "true"

BATCH_SIZE = 4


class COCOPatched(COCO):
    def __init__(self, annotations):
        # The varnames here are disgusting, but they're used by other
        # non-overridden methods so don't touch them.
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


def run_hf(model, processor, batch):
    for b in batch:
        bytes_array = base64.b64decode(b["b64"])
        im = Image.open(BytesIO(bytes_array))
        inputs = processor(images=im, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([im.size[::-1]])
        res = processor.post_process_object_detection(
            outputs, threshold=0.9, target_sizes=target_sizes
        )[0]
        for label, box in zip(res["labels"], res["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)
            yield {
                "image_id": b["key"],
                "score": 1.0,
                "category_id": int(label.item()),
                "bbox": [x, y, w, h],
            }


def run_ultralytics(model, batch):
    for b in batch:
        bytes_array = base64.b64decode(b["b64"])
        im = Image.open(BytesIO(bytes_array))
        results = model(im, verbose=False, imgsz=1280, rect=True)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # [x1, y1, x2, y2]
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)
            yield {
                "image_id": b["key"],
                "score": 1.0,
                "category_id": int(box.cls[0]),
                "bbox": [x, y, w, h],
            }


def main():
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, "r") as f:
        annotations = json.load(f)

    # Get list of validation images from val/labels folder
    val_labels_path = Path("data/cv/val/labels")
    val_image_ids = set()

    if val_labels_path.exists():
        for label_file in val_labels_path.glob("*.txt"):
            # Convert label filename to image id (removing extension)
            image_id = label_file.stem
            val_image_ids.add(image_id)

    # filter the whole annotations to only include validation images
    annotations["images"] = [
        img for img in annotations["images"] if str(img["id"]) in val_image_ids
    ]

    annotations["annotations"] = [
        ann
        for ann in annotations["annotations"]
        if str(ann["image_id"]) in val_image_ids
    ]

    instances = annotations["images"]

    batch_generator = itertools.batched(
        sample_generator(instances, DATA_PATH), n=BATCH_SIZE
    )

    # model = YOLO(TRAIN_OUTPUT / "yolo11x-finetuned-2" / "weights" / "best.pt")
    model = RTDETR(TRAIN_OUTPUT / "rtdetr-x-finetuned" / "weights" / "best.pt")

    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for batch in tqdm(batch_generator, total=math.ceil(len(instances) / BATCH_SIZE)):
        # results.extend(list(run_hf(model, processor, batch)))
        results.extend(list(run_ultralytics(model, batch)))

    cv_results_path = RESULTS_PATH / "cv_results.json"
    print(f"Saving test results to {str(cv_results_path)}")
    with open(cv_results_path, "w") as results_file:
        json.dump(results, results_file)

    print("Evaluating results...")
    mean_ap = score_cv(results, annotations)
    print("mAP@.5:.05:.95:", mean_ap)


if __name__ == "__main__":
    main()
