from cv_dev.consts import DATA_PATH, TRAIN_OUTPUT

from ultralytics import YOLO
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


def train_rtdetr(n_epochs: int):
    args = dict(
        model=TRAIN_OUTPUT / "trains" / "rtdetr-x-finetuned" / "weights" / "last.pt",
        # model="rtdetr-x.pt",
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


if __name__ == "__main__":
    # train_yolov11(50)
    train_rtdetr(50)
