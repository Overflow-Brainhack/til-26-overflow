from cv_dev.consts import DATA_PATH, DATASETS_PATH, NUM_CATEGORIES

from ultralytics.models.rtdetr.train import RTDETRTrainer

import warnings
import os
import lightning as pl
from cv_dev.make_dataset import ImageDataset
import torch

os.environ["WANDB_DISABLED"] = "true"

torch.set_float32_matmul_precision("medium")


def train_rtdetr_baidu(n_epochs: int):
    args = dict(
        # model="cv-training/trains/rtdetr-baidu-finetuned/weights/last.pt",
        model="rtdetr-l.pt",
        data=DATA_PATH / "data.yaml",
        epochs=n_epochs,
        batch=8,
        project="cv-training/trains",
        name="rtdetr-baidu-finetuned",
        save_period=5,
        device=0,
        workers=0,
        imgsz=1024,
        rect=True,
        # resume=True,
    )

    trainer: RTDETRTrainer = RTDETRTrainer(overrides=args)
    trainer.train()


if __name__ == "__main__":
    train_rtdetr_baidu(30)
