"""Manages the CV model."""

from manager import Manager

from typing import Any, override
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torchvision.ops import batched_nms
from huggingface_hub import PyTorchModelHubMixin
from EdgeCrafter.engine.core import YAMLConfig


def build_model(config_path: str, resume_path: str, device: torch.device | str):
    cfg = YAMLConfig(config_path, resume=resume_path)
    cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True

    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(device)
    model.eval()

    return model


class ECDetManager(Manager):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_model(
            "EdgeCrafter/configs/ecdet/ecdet_l_til26.yml",
            "models/ecdet-l-57.pth",
            self.device,
        )
        self.transforms = self._build_transforms()

    def _build_transforms(self):
        return T.Compose(
            [
                T.Resize((640, 640)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @override
    def infer(self, image: bytes) -> list[dict[str, int | list[float]]]:
        im = Image.open(BytesIO(image)).convert("RGB")
        orig_sizes = torch.tensor([[im.size[0], im.size[1]]], device=self.device)
        tensor = self.transforms(im).unsqueeze(0).to(self.device)
        outputs = self.model(tensor, orig_sizes)

        labels, boxes, scores = outputs

        keep = scores[0] > 0.25
        class_labels = labels[0][keep]
        bboxes = boxes[0][keep]
        scores = scores[0][keep]

        if len(bboxes) > 0:
            keep = batched_nms(
                bboxes.float(), scores.float(), class_labels, iou_threshold=0.5
            )
            class_labels = class_labels[keep]
            bboxes = bboxes[keep]
            scores = scores[keep]

        preds = []
        for label, bbox in zip(class_labels, bboxes):
            preds.append(
                {
                    "category_id": label.item(),
                    "bbox": [
                        bbox[0].item(),
                        bbox[1].item(),
                        (bbox[2] - bbox[0]).item(),
                        (bbox[3] - bbox[1]).item(),
                    ],
                }
            )
        return preds
