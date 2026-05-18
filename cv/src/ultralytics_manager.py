"""Manages the CV model."""

from manager import Manager
from ultralytics import RTDETR, YOLO
from typing import Any, override
from io import BytesIO
from PIL import Image, ImageFilter


class UltralyticsManager(Manager):
    def __init__(self):
        super().__init__()
        self.model = RTDETR("models/rtdetr-l-70.pt")
        # self.model = RTDETR("models/rtdetr-l-adv-70.pt")

    @override
    def infer(self, image: bytes) -> list[dict[str, int | list[float]]]:
        im = Image.open(BytesIO(image))
        results = self.model.predict(
            im, verbose=False, imgsz=1280, rect=True, augment=True, half=True
        )
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
