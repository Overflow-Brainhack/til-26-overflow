"""Manages the CV model."""

from ultralytics import RTDETR, YOLO
from typing import Any
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.ops import batched_nms
from huggingface_hub import PyTorchModelHubMixin

from DEIMv2.engine.backbone import DINOv3STAs
from DEIMv2.engine.deim import HybridEncoder
from DEIMv2.engine.deim import DEIMTransformer
from DEIMv2.engine.deim.postprocessor import PostProcessor


class DEIMv2(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        self.backbone = DINOv3STAs(**config["DINOv3STAs"])
        self.encoder = HybridEncoder(**config["HybridEncoder"])
        self.decoder = DEIMTransformer(**config["DEIMTransformer"])
        self.postprocessor = PostProcessor(**config["PostProcessor"])

    def forward(self, x, orig_target_sizes):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.postprocessor(x, orig_target_sizes)

        return x


deimv2_l_config = {
    "DINOv3STAs": {
        "name": "dinov3_vits16",
        "embed_dim": 224,
        "interaction_indexes": [5, 8, 11],
        "num_heads": None,
        "conv_inplane": 32,
        "hidden_dim": 224,
    },
    "HybridEncoder": {
        "in_channels": [224, 224, 224],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 224,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 896,
        "dropout": 0.0,
        "enc_act": "gelu",
        "expansion": 1.0,
        "depth_mult": 1,
        "act": "silu",
        "version": "deim",
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "DEIMTransformer": {
        "feat_channels": [224, 224, 224],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 224,
        "num_levels": 3,
        "num_layers": 4,
        "eval_idx": -1,
        "num_queries": 300,
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "num_points": [3, 6, 3],
        "cross_attn_method": "default",
        "query_select_method": "default",
        "activation": "silu",
        "mlp_act": "silu",
        "dim_feedforward": 1792,
        "eval_spatial_size": [640, 640],
        "num_classes": 18,
    },
    "PostProcessor": {"num_top_queries": 300, "num_classes": 18},
}


class CVManager:
    def __init__(self):
        # self.model = RTDETR("models/rtdetr-l-70.pt") best
        # self.model = RTDETR("models/rtdetr-x-43.pt")
        self.model = RTDETR("models/rtdetr-l-20-s40.pt")

        # self.model = DEIMv2(deimv2_l_config)
        # state_dict = torch.load("models/DEIMv2-l-68.pth", map_location="cpu")
        # if "model" in state_dict:
        #     state_dict = state_dict["model"]
        # self.model.load_state_dict(state_dict, strict=True)
        # self.model.to("cuda")

    def run_ultralytics(self, image: bytes) -> list[dict[str, Any]]:
        im = Image.open(BytesIO(image))
        results = self.model(im, verbose=False, imgsz=1280, rect=True)
        preds = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            preds.append(
                {
                    "category_id": int(box.cls[0]),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                }
            )
        return preds

    def run_deim(self, image: bytes) -> list[dict[str, Any]]:
        im = Image.open(BytesIO(image)).convert("RGB")
        original_dimensions = im.size
        transform_dimensions = (640, 640)
        scaling_factor = (
            original_dimensions[0] / transform_dimensions[0],
            original_dimensions[1] / transform_dimensions[1],
        )
        transform = transforms.Compose(
            [
                transforms.Resize(transform_dimensions),
                transforms.ToTensor(),
            ]
        )
        input_tensor = transform(im).unsqueeze(0).to("cuda")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_tensor,
                orig_target_sizes=torch.tensor([transform_dimensions]).to("cuda"),
            )

        class_labels, bboxes, scores = (
            outputs[0]["labels"],
            outputs[0]["boxes"],
            outputs[0]["scores"],
        )

        score_mask = scores >= 0.25
        class_labels = class_labels[score_mask]
        bboxes = bboxes[score_mask]
        scores = scores[score_mask]

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
                        bbox[0].item() * scaling_factor[0],
                        bbox[1].item() * scaling_factor[1],
                        (bbox[2] - bbox[0]).item() * scaling_factor[0],
                        (bbox[3] - bbox[1]).item() * scaling_factor[1],
                    ],
                }
            )
        return preds

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image."""
        if isinstance(self.model, RTDETR) or isinstance(self.model, YOLO):
            return self.run_ultralytics(image)
        elif isinstance(self.model, DEIMv2):
            return self.run_deim(image)
        return []
