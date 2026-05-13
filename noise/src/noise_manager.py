"""Manages adversarial image noising via PGD attack on a surrogate YOLO model."""

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO


class NoiseManager:
    SURROGATE = "yolov8n.pt"
    EPSILON = 45.0   # max RMSE in pixel space [0,255]; keeps us under both L2 thresholds
    STEPS = 10
    STEP_SIZE = 5.0  # gradient step size in pixel space

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.SURROGATE)
        # Training mode gives raw anchor logits before NMS, enabling gradient flow
        self.model.model.train()
        self.model.model.to(self.device)

    @staticmethod
    def _project_rmse(delta: torch.Tensor, eps_pixel: float) -> torch.Tensor:
        """Scale delta down if its pixel-space RMSE exceeds eps_pixel."""
        rmse = (delta * 255.0).pow(2).mean().sqrt()
        if rmse > eps_pixel:
            delta = delta * (eps_pixel / rmse)
        return delta

    def _detection_loss(self, raw: torch.Tensor) -> torch.Tensor:
        """Return mean of per-anchor max class confidence. Minimising this suppresses detections."""
        # raw: [B, 4+nc, num_anchors]
        class_logits = raw[:, 4:, :]              # [B, nc, num_anchors]
        conf = torch.sigmoid(class_logits)         # [B, nc, num_anchors]
        return conf.max(dim=1).values.mean()       # scalar

    def _attack(self, orig: torch.Tensor) -> torch.Tensor:
        """Run PGD to find a perturbation that suppresses detections within the RMSE budget."""
        alpha = self.STEP_SIZE / 255.0
        init_scale = (self.EPSILON / 255.0) * 0.1

        delta = torch.empty_like(orig).uniform_(-init_scale, init_scale)
        delta.requires_grad_(True)

        for _ in range(self.STEPS):
            adv = (orig + delta).clamp(0.0, 1.0)
            x = F.interpolate(adv, (640, 640), mode="bilinear", align_corners=False)

            raw = self.model.model(x)
            if isinstance(raw, (list, tuple)):
                raw = raw[0]

            loss = self._detection_loss(raw)
            loss.backward()

            with torch.no_grad():
                delta.data -= alpha * delta.grad.sign()
                delta.data = self._project_rmse(delta.data, self.EPSILON)
                # keep adversarial image within valid [0, 1] range
                delta.data = delta.data.clamp(-orig, 1.0 - orig)

            delta.grad.zero_()

        return delta.detach()

    def noise(self, image: bytes) -> str:
        """Performs adversarial noising on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A string containing the noised image encoded in base64.
        """
        try:
            img_pil = Image.open(io.BytesIO(image)).convert("RGB")
            arr = np.array(img_pil, dtype=np.float32) / 255.0
            orig = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.enable_grad():
                delta = self._attack(orig)

            adv = (orig + delta).clamp(0.0, 1.0)
            adv_np = (adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            buf = io.BytesIO()
            Image.fromarray(adv_np).save(buf, format="JPEG", quality=95)
            return base64.b64encode(buf.getvalue()).decode("ascii")

        except Exception as e:
            print(f"Error occurred: {e}")
            return base64.b64encode(image).decode("ascii")
