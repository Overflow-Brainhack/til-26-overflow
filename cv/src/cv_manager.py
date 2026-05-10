"""Manages the CV model."""

from ultralytics import RTDETR
from typing import Any
from io import BytesIO
from PIL import Image


class CVManager:
    def __init__(self):
        self.model = RTDETR("models/epoch15.pt")
        pass

    def xyxy_to_xywh(self, xyxy: list[int | float]) -> list[float]:
        x1, y1, x2, y2 = xyxy
        x = float(x1)
        y = float(y1)
        w = float(x2 - x1)
        h = float(y2 - y1)
        return [x, y, w, h]

    def run_ultralytics(self, image: bytes) -> list[dict[str, Any]]:
        im = Image.open(BytesIO(image))
        results = self.model(im, verbose=False, imgsz=1024, rect=True)
        preds = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # [x1, y1, x2, y2]
            preds.append(
                {
                    "category_id": int(box.cls[0]),
                    "bbox": self.xyxy_to_xywh(
                        [x1, y1, x2, y2]
                    ),  # [x_top_left, y_top_left, width, height]
                }
            )
        return preds

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            `cv/README.md` for the expected format.
        """

        return self.run_ultralytics(image)
