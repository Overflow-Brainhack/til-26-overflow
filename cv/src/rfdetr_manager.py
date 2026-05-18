"""Manages the CV model."""

from typing import override
from turbojpeg import TurboJPEG, TJPF_RGB
from manager import Manager
import numpy as np
from rfdetr import RFDETRLarge


class RFDETRManager(Manager):
    def __init__(self):
        super().__init__()
        self.model = RFDETRLarge(
            num_classes=18,
            resolution=1280,
            pretrain_weights="models/checkpoint_1280_best_total.pth",
        )
        self.model.optimize_for_inference()

        self.jpeg = TurboJPEG()

    @override
    def infer(self, image: bytes) -> list[dict[str, int | list[float]]]:
        im = self.jpeg.decode(image, pixel_format=TJPF_RGB)

        # 'im' is a standard numpy uint8 array
        detections = self.model.predict(im, threshold=0.5)

        # Extract indices directly without referencing descriptive metadata dictionaries
        preds = [
            {
                "category_id": int(det[3]),
                "bbox": [
                    float(det[0][0]),
                    float(det[0][1]),
                    float(det[0][2] - det[0][0]),
                    float(det[0][3] - det[0][1]),
                ],
            }
            for det in detections
        ]

        return preds
