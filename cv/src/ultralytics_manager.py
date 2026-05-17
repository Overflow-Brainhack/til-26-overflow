"""Manages the CV model."""

from ultralytics import RTDETR, YOLO
from typing import Any
from io import BytesIO
from PIL import Image, ImageFilter


class UltralyticsManager:
    def __init__(self):
        # self.model = RTDETR("models/rtdetr-l-70.pt")
        self.model = RTDETR("models/rtdetr-l-adv-70.pt")

    def _preprocess(self, image: bytes) -> bytes:
        """Strip adversarial perturbations with minimal accuracy impact."""
        im = Image.open(BytesIO(image)).convert("RGB")
        # Gaussian blur radius=1: kills sub-pixel adversarial noise, within YOLO blur augment range
        im = im.filter(ImageFilter.GaussianBlur(radius=1))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def run_ultralytics(self, image: bytes) -> list[dict[str, Any]]:
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

    def infer(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection for the noising docker"""
        return self.run_ultralytics(image)

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image."""
        image = self._preprocess(image)
        return self.run_ultralytics(image)
