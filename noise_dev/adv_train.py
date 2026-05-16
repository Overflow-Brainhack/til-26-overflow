from __future__ import annotations

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
    """RTDETRTrainer that injects FGSM adversarial examples between two epoch bounds.

    Schedule:
        0 .. ADV_START_EPOCH-1      : clean training
        ADV_START_EPOCH .. ADV_END_EPOCH-1 : ADV_FRACTION of batches get FGSM
        ADV_END_EPOCH .. end        : clean fine-tune at reduced LR (cooldown)

    Why FGSM and not PGD inline:
        FGSM = 1 extra forward pass per affected batch (~1.5x cost).
        At ADV_FRACTION=0.25 over 5 adversarial epochs out of 15, the overall
        training overhead is ~6%. PGD (6 steps) would be ~7x per batch.

    Why not 100% adversarial batches in the adversarial phase:
        Full adversarial batches cause catastrophic forgetting of clean mAP.
        Mixing 25% adversarial + 75% clean maintains clean performance.

    FGSM direction:
        We minimise detection confidence (matching the attack server's goal),
        then train on those images with ground-truth labels so the model learns
        to detect objects even under that perturbation type.
    """

    ADV_START_EPOCH: int = 10
    ADV_END_EPOCH: int = 15
    ADV_FRACTION: float = 0.25
    ADV_EPS: float = 40.0 / 255.0  # ~40 pixel RMSE in [0,1]; consistent with noise attack
    COOLDOWN_LR_FACTOR: float = 0.1

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        if (
            self.ADV_START_EPOCH <= self.epoch < self.ADV_END_EPOCH
            and random.random() < self.ADV_FRACTION
        ):
            batch["img"] = self._fgsm(batch["img"])
        return batch

    def _fgsm(self, imgs: torch.Tensor) -> torch.Tensor:
        """Single-step FGSM: compute adversarial perturbation that suppresses detections.

        Uses eval mode so that RTDETRDecoder returns a clean (B, queries, 4+nc)
        tensor via _predict_once — training mode would invoke the raw nn.Sequential
        path and break the neck's skip-connection routing.

        torch.autograd.grad is used so only the image leaf accumulates a gradient;
        model parameter gradients are untouched and the optimizer stays clean.
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
