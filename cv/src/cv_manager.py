"""Manages the CV model."""

from ultralytics import RTDETR, YOLO
from typing import Any
from io import BytesIO
from PIL import Image


class CVManager:
    def __init__(self):
        # self.model = RTDETR("models/rtdetr-l-30.pt")
        # self.model = RTDETR("models/rtdetr-x-43.pt")
        self.model = RTDETR("models/rtdetr-x-40-synth15-fixed.pt")

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

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image."""
        return self.run_ultralytics(image)
