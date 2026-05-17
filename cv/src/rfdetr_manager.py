"""Manages the CV model."""

from turbojpeg import TurboJPEG, TJPF_RGB
import numpy as np
from rfdetr import RFDETRLarge
from typing import Any
import torch

class RFDETRManager:
    def __init__(self):
        # MOVE THESE HERE: They now run safely after Uvicorn finishes loading
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.fp32_precision = 'high'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        self.model = RFDETRLarge(num_classes=18, resolution=1280, pretrain_weights='models/checkpoint_1280_best_total.pth')
        self.model.optimize_for_inference()
        
        # Initialize TurboJPEG once during startup to avoid overhead per request
        self.jpeg = TurboJPEG()

    def _preprocess(self, image: bytes) -> bytes:
        """Strip adversarial perturbations with minimal accuracy impact."""
        im = Image.open(BytesIO(image)).convert("RGB")
        # Gaussian blur radius=1: kills sub-pixel adversarial noise, within YOLO blur augment range
        im = im.filter(ImageFilter.GaussianBlur(radius=1))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def infer(self, image: bytes) -> list[dict[str, Any]]:
        # TurboJPEG directly decodes bytes to a NumPy array in memory.
        # We explicitly pass TJPF_RGB to ensure it decodes to RGB format (matching PIL's behavior).
        im = self.jpeg.decode(image, pixel_format=TJPF_RGB)
        
        # 'im' is now a standard numpy uint8 array. 
        detections = self.model.predict(im, threshold=0.5)

        # Extract indices directly without referencing descriptive metadata dictionaries
        preds = [
            {
                "category_id": int(det[3]), 
                "bbox": [float(det[0][0]), float(det[0][1]), float(det[0][2] - det[0][0]), float(det[0][3] - det[0][1])]
            }
            for det in detections
        ]
            
        return preds

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions.
        """
        # image = self._preprocess(image)
        return self.infer(image)
