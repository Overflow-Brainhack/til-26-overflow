"""Manages the CV model."""

from rfdetr import RFDETRLarge
from typing import Any
from io import BytesIO
from PIL import Image


class RFDETRManager:
    def __init__(self):
        self.model = RFDETRLarge(
            num_classes=18, pretrain_weights="models/checkpoint50_best_ema.pth"
        )
        self.model.optimize_for_inference()

    def xyxy_to_xywh(self, xyxy: list[int | float]) -> list[float]:
        x1, y1, x2, y2 = xyxy
        x = float(x1)
        y = float(y1)
        w = float(x2 - x1)
        h = float(y2 - y1)
        return [x, y, w, h]

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            `cv/README.md` for the expected format.
        """
        im = Image.open(BytesIO(image))
        detections = self.model.predict(im, threshold=0.5)

        preds = []

        for det in detections:
            xyxy = det[0]
            conf = det[2]
            class_id = det[3]
            class_name = det[5]["class_name"]

            preds.append(
                {
                    "category_id": int(class_id),
                    "bbox": self.xyxy_to_xywh(xyxy),
                }
            )

        return preds
