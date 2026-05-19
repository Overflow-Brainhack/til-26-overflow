from cv_dev.consts import (
    DATA_PATH,
    TRAIN_OUTPUT,
    DATASETS_PATH,
    NUM_CATEGORIES,
    SYNTHETIC_DATA_PATH,
)
from cv_dev.make_torch_dataset import load_dataset, ImageDataset
from cv_dev.adv_rtdetr import AdversarialRTDETRTrainer

from transformers import (
    DetrForObjectDetection,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

from ultralytics import YOLO, RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from pathlib import Path

import os
import shutil
import subprocess

import torch

os.environ["WANDB_DISABLED"] = "true"

torch.set_float32_matmul_precision("medium")

DATA_YAML = str(DATA_PATH / "data.yaml")
SYNTHETIC_DATA_YAML = str(SYNTHETIC_DATA_PATH / "data.yaml")


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


def train_rtdetr(
    model: Path | str,
    n_epochs: int,
    resume: bool = False,
    name: str = "rtdetr-x-finetuned",
):
    args = dict(
        model=str(model),
        data=DATA_YAML,
        epochs=n_epochs,
        batch=4,
        project=str(TRAIN_OUTPUT),
        name=name,
        save_period=5,
        device=0,
        workers=0,
        imgsz=1280,
        rect=True,
        resume=resume,
    )
    trainer: RTDETRTrainer = RTDETRTrainer(overrides=args)
    trainer.train()


def train_rtdetr_adv(
    model: Path | str,
    n_epochs: int,
    resume: bool = False,
    name: str = "rtdetr-x-finetuned",
):
    args = dict(
        model=str(model),
        data=DATA_YAML,
        epochs=n_epochs,
        batch=4,
        project=str(TRAIN_OUTPUT),
        name=name,
        save_period=5,
        device=0,
        workers=0,
        imgsz=1280,
        rect=True,
        resume=resume,
    )
    trainer: AdversarialRTDETRTrainer = AdversarialRTDETRTrainer(overrides=args)
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


def train_rtdetr_synth(
    model: Path,
    n_epochs: int,
    resume: bool = False,
    name: str = "rtdetr-x-finetuned-synth",
):
    args = dict(
        # training from rtdetr-x-40
        model=str(model),
        data=SYNTHETIC_DATA_YAML,
        epochs=n_epochs,
        batch=4,
        project=str(TRAIN_OUTPUT),
        name=name,
        save_period=5,
        device=0,
        workers=0,
        imgsz=1280,
        rect=True,
        resume=resume,
    )
    trainer: RTDETRTrainer = RTDETRTrainer(overrides=args)
    trainer.train()


def train_ec(
    ec_repo: Path,
    resume: bool = False,
) -> None:
    """
    Fine-tune ECDet-L/M on our 18-class dataset.

    Requires the EdgeCrafter repo cloned locally:
        git clone https://github.com/Intellindust-AI-Lab/EdgeCrafter <ec_repo>
        pip install -r <ec_repo>/requirements.txt

    Args:
        re_repo: path to the cloned EdgeCrafter directory
        resume: resume from the latest checkpoint in the output dir
    """

    dataset_cfg_path = ec_repo / "configs" / "dataset" / "til26_dataset.yml"
    shutil.copyfile("cv_dev/til26_dataset.yml", dataset_cfg_path)

    model_cfg_path = ec_repo / "configs" / "ecdet" / "ecdet_m_til26.yml"
    shutil.copyfile("cv_dev/ecdet_m_til26.yml", model_cfg_path)

    cmd = [
        "uv",
        "run",
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "train.py",
        "-c",
        str(model_cfg_path.relative_to(ec_repo)),
        "--use-amp",
        "--seed=0",
    ]

    if resume:
        output_dir = Path(
            "/home/dev/til-26-overflow/cv-training/trains/ecdet-m-finetuned"
        )
        checkpoints = sorted(
            output_dir.glob("checkpoint*.pth"), key=lambda p: p.stat().st_mtime
        )
        if checkpoints:
            cmd += ["--resume", str(checkpoints[-1])]
    else:
        cmd.append("-t")
        cmd.append("/home/dev/til-26-overflow/ecdet-m-pretrained.pth")

    subprocess.run(cmd, cwd=str(ec_repo), check=True)


def train_deimv2(
    deimv2_repo: Path,
    resume: bool = False,
) -> None:
    """
    Fine-tune DEIMv2-L (DINOv3 backbone) on our 18-class dataset.

    Requires the DEIMv2 repo cloned locally:
        git clone https://github.com/Intellindust-AI-Lab/DEIMv2 <deimv2_repo>
        pip install -r <deimv2_repo>/requirements.txt

    Args:
        deimv2_repo: path to the cloned DEIMv2 directory
        resume: resume from the latest checkpoint in the output dir
    """

    dataset_cfg_path = deimv2_repo / "configs" / "dataset" / "til26_dataset.yml"
    shutil.copyfile("cv_dev/til26_dataset.yml", dataset_cfg_path)

    dataloader_cfg_path = deimv2_repo / "configs" / "base" / "til26_dataloader.yml"
    shutil.copyfile("cv_dev/til26_dataloader.yml", dataloader_cfg_path)

    model_cfg_path = deimv2_repo / "configs" / "deimv2" / "deimv2_dinov3_l_til26.yml"
    shutil.copyfile("cv_dev/deimv2_dinov3_l_til26.yml", model_cfg_path)

    cmd = [
        "uv",
        "run",
        "torchrun",
        "--master_port=7777",
        "--nproc_per_node=1",
        "train.py",
        "-c",
        str(model_cfg_path.relative_to(deimv2_repo)),
        "-t",
        "/home/dev/til-26-overflow/deimv2-hf.pth",
        "--use-amp",
        "--seed=0",
    ]

    if resume:
        output_dir = "/home/dev/til-26-overflow/cv-training/trains/DEIMv2-finetuned"
        checkpoints = sorted(
            output_dir.glob("checkpoint*.pth"), key=lambda p: p.stat().st_mtime
        )
        if checkpoints:
            cmd += ["--resume", str(checkpoints[-1])]

    subprocess.run(cmd, cwd=str(deimv2_repo), check=True)


if __name__ == "__main__":
    # train_rtdetr_synth(TRAIN_OUTPUT / "rtdetr-x-finetuned/weights/epoch30.pt", 50)
    # train_rtdetr_synth(
    #     TRAIN_OUTPUT / "rtdetr-l-finetuned-2/weights/epoch20.pt",
    #     50,
    #     name="rtdetr-l-finetuned-synth",
    # )  # train rtdetr-l-50 on 50 synth images

    # train_rtdetr(
    #     # TRAIN_OUTPUT / "rtdetr-l-finetuned/weights/last.pt",
    #     "rtdetr-l.pt",
    #     80,
    #     name="rtdetr-l-finetuned",
    # )

    # train_rtdetr_adv(
    #     # TRAIN_OUTPUT / "rtdetr-l-finetuned-adv/weights/last.pt",
    #     "rtdetr-l.pt",
    #     70,
    #     name="rtdetr-l-finetuned-adv",
    #     resume=True,
    # )

    # train_deimv2(Path("/home/dev/DEIMv2/"))

    train_ec(Path("/home/dev/EdgeCrafter/ecdetseg/"), resume=True)
