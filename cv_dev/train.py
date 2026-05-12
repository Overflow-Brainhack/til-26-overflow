from cv_dev.consts import DATA_PATH, TRAIN_OUTPUT, DATASETS_PATH, NUM_CATEGORIES
from cv_dev.make_dataset import load_dataset, ImageDataset
from transformers import (
    DetrForObjectDetection,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

from ultralytics import YOLO, RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer

import os
import torch

os.environ["WANDB_DISABLED"] = "true"

torch.set_float32_matmul_precision("medium")

DATA_YAML = str(DATA_PATH / "data.yaml")


def train_yolov11(n_epochs: int):
    model = YOLO("yolo11x.pt")
    model.train(
        data=DATA_YAML,
        epochs=n_epochs,
        batch=4,
        project=str(TRAIN_OUTPUT),
        name="yolo11x-finetuned",
        save_period=5,
        device=0,
        workers=0,
        imgsz=1280,
        rect=True,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        perspective=0.0005,
    )


def train_yolov26(n_epochs: int):
    # model = YOLO("yolo26l.pt")
    model = YOLO(TRAIN_OUTPUT / "yolo26l-finetuned" / "weights" / "last.pt")
    model.train(
        data=DATA_YAML,
        epochs=n_epochs,
        batch=32,
        project=str(TRAIN_OUTPUT),
        name="yolo26l-finetuned",
        save_period=5,
        device=0,
        workers=0,
        imgsz=640,
        rect=True,
        resume=True,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        perspective=0.0005,
    )


def train_rtdetr(n_epochs: int):
    args = dict(
        # model=TRAIN_OUTPUT / "rtdetr-x-finetuned" / "weights" / "last.pt",
        model="rtdetr-x.pt",
        data=DATA_YAML,
        epochs=n_epochs,
        batch=2,
        project=str(TRAIN_OUTPUT),
        name="rtdetr-x-finetuned",
        save_period=5,
        device=0,
        workers=0,
        imgsz=1280,
        rect=True,
        resume=True,
    )
    trainer: RTDETRTrainer = RTDETRTrainer(overrides=args)
    trainer.train()


def train_detr_hf(n_epochs: int):
    # ImageNet normalization — applied on top of the 0-1 float tensors from ImageDataset
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    class DETRDataset(Dataset):
        def __init__(self, base: ImageDataset) -> None:
            self.base = base

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, idx: int):
            image, target = self.base[idx]
            _, H, W = image.shape

            pixel_values = normalize(image, IMAGENET_MEAN, IMAGENET_STD)
            pixel_mask = torch.ones(H, W, dtype=torch.long)

            # xyxy absolute → cxcywh normalized
            boxes = target["boxes"]
            if len(boxes) > 0:
                cx = (boxes[:, 0] + boxes[:, 2]) / 2 / W
                cy = (boxes[:, 1] + boxes[:, 3]) / 2 / H
                bw = (boxes[:, 2] - boxes[:, 0]) / W
                bh = (boxes[:, 3] - boxes[:, 1]) / H
                boxes_norm = torch.stack([cx, cy, bw, bh], dim=1)
            else:
                boxes_norm = torch.zeros((0, 4), dtype=torch.float32)

            return {
                "pixel_values": pixel_values,
                "pixel_mask": pixel_mask,
                "labels": {
                    # ImageDataset adds +1 to category_id; undo it for DETR
                    "class_labels": target["labels"] - 1,
                    "boxes": boxes_norm,
                    "image_id": target["image_id"],
                    "area": target["area"],
                    "iscrowd": target["iscrowd"],
                    "orig_size": torch.tensor([H, W]),
                    "size": torch.tensor([H, W]),
                },
            }

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "pixel_mask": torch.stack([x["pixel_mask"] for x in batch]),
            "labels": [x["labels"] for x in batch],
        }

    print("loading datasets")
    train_base = load_dataset(DATASETS_PATH / "train.pt")
    val_base = load_dataset(DATASETS_PATH / "val.pt")
    print("loaded datasets")

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-101",
        num_labels=NUM_CATEGORIES,
        ignore_mismatched_sizes=True,
    )

    print("loaded model")

    output_dir = str(TRAIN_OUTPUT / "detr-resnet101-finetuned")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        weight_decay=1e-4,
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_num_workers=0,
        fp16=True,
        logging_steps=50,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=DETRDataset(train_base),
        eval_dataset=DETRDataset(val_base),
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(output_dir)


if __name__ == "__main__":
    # train_yolov11(50)
    train_yolov26(50)
    # train_rtdetr(50)
    # train_detr_hf(50)
