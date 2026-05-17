import random

import torch
from ultralytics.models.rtdetr.train import RTDETRTrainer


def _detection_confidence(preds) -> torch.Tensor:
    """Extract mean max confidence from ultralytics eval-mode predictions.

    RT-DETR eval output: (B, num_queries, 4+nc), class scores at [..., 4:].
    Scores are already sigmoid'd by RTDETRDecoder in inference mode.
    """
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    if preds.dim() == 3:
        if preds.shape[-1] > preds.shape[1]:
            # (B, num_queries, 4+nc) — RT-DETR style
            return preds[..., 4:].max(dim=-1).values.mean()
        # (B, 4+nc, num_anchors) — YOLO style
        return torch.sigmoid(preds[:, 4:, :]).max(dim=1).values.mean()
    return preds.mean()


class AdversarialRTDETRTrainer(RTDETRTrainer):
    """RTDETRTrainer that injects FGSM adversarial examples from a given epoch.

    Training schedule:
        - Epochs 0..ADV_START_EPOCH-1: standard clean training (fast)
        - Epochs ADV_START_EPOCH..end: ADV_FRACTION of batches get FGSM perturbation
    """

    ADV_START_EPOCH: int = 40
    ADV_END_EPOCH: int = 65
    ADV_FRACTION: float = 0.25
    ADV_EPS: float = (
        40.0 / 255.0
    )  # ~40 pixel RMSE in [0,1]; matches noise attack budget
    COOLDOWN_LR_FACTOR: float = 0.1

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        if (
            self.epoch >= self.ADV_START_EPOCH
            and self.epoch < self.ADV_END_EPOCH
            and random.random() < self.ADV_FRACTION
        ):
            batch["img"] = self._fgsm(batch["img"])
        return batch

    def _fgsm(self, imgs: torch.Tensor) -> torch.Tensor:
        """Single-step FGSM perturbation.

        Uses torch.autograd.grad so only the image leaf gets a gradient —
        model parameter gradients are unaffected, keeping the optimizer clean.
        """
        imgs_leaf = imgs.detach().clone().requires_grad_(True)

        self.model.eval()
        try:
            with torch.enable_grad():
                preds = self.model(imgs_leaf)
                loss = _detection_confidence(preds)
                (img_grad,) = torch.autograd.grad(loss, imgs_leaf)
        finally:
            self.model.train()

        with torch.no_grad():
            # Negative sign: gradient descent on confidence → suppresses detections
            delta = -self.ADV_EPS * img_grad.sign()
            rmse = (delta * 255.0).pow(2).mean().sqrt()
            if rmse > self.ADV_EPS * 255.0:
                delta = delta * (self.ADV_EPS * 255.0 / rmse)
            adv = (imgs + delta).clamp(0.0, 1.0)

        return adv.detach()

    def optimizer_step(self):
        if self.epoch >= self.ADV_END_EPOCH:
            for pg in self.optimizer.param_groups:
                if "_base_lr" not in pg:
                    pg["_base_lr"] = pg["lr"]
                pg["lr"] = pg["_base_lr"] * self.COOLDOWN_LR_FACTOR
        super().optimizer_step()
