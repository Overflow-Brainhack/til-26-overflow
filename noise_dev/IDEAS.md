# Noise Development — Ideas & Notes

## Fairness check breakdown

Three checks are run by the evaluator using **ground-truth bboxes** (not our predicted ones):

| Check | Threshold | What it measures |
|---|---|---|
| `L2 (RMSE)` ≤ 67 | global | RMSE across every pixel in the image |
| `L2 inside` ≤ 50 | tighter | RMSE restricted to pixels inside GT bboxes |
| `SSIM inside` ≥ 0.3 | very loose | Structural similarity inside GT bboxes |

### Evaluation formula (matches `pipeline.py`)

- **Global RMSE**: `sqrt(mean((orig - noised)^2))` over all pixels and channels
- **Inside RMSE**: `sqrt(mean_over_inside_pixels(mean_over_channels((orig - noised)^2)))`
- **Inside SSIM**: skimage `structural_similarity` with `win_size=7`, then average the per-pixel SSIM map over inside-bbox pixels only

### Key implications

1. **The evaluator uses GT bboxes, not our CV predictions.** Pixels inside a GT bbox but outside our detected bbox get the full global-level perturbation applied. If global RMSE is near 67, any mis-detected region will push `L2 inside` well over 50 and fail.

2. **We cap `RMSE_INSIDE_MAX = 47`** (3-unit margin below the 50 threshold) to absorb JPEG re-encoding drift.

3. **`SSIM_INSIDE_MIN = 0.31`** (0.01 margin above the 0.30 threshold). The binary-search `_enforce_ssim` guarantees this for every image regardless of bbox size.

4. **SSIM 0.3 is very permissive.** Even at RMSE 47 inside bboxes, SSIM is typically 0.4–0.6. Failures only occur on tiny bboxes.

### Inside/outside budget asymmetry

The global budget (67) is larger than the inside budget (50). If object pixels cover fraction `f` of the image:

```
global_rmse² = f · inside_rmse² + (1-f) · outside_rmse²
→ outside_rmse ≤ sqrt((67² − f · 47²) / (1−f))
```

At `f = 0.2`: outside can reach **~71 RMSE** — stronger per-pixel than the global limit — by concentrating the residual budget on non-object pixels. `NoiseManager` exploits this: inside is capped at 47, outside receives the full remaining global budget.

---

## Current approach: CV-technique attack (`noise_manager.py`)

No surrogate models, no GPU, no gradient computation. Pure image processing.

### Why we switched from PGD

PGD with `yolo11m` as surrogate barely affected `yolo11x-finetuned`:

- White-box FGSM (attacker knows the model): mAP 0.95 → 0.50 (47% drop)
- Black-box PGD transfer (yolo11m → yolo11x-finetuned): negligible effect

Root cause: cross-model transfer failure. Gradients computed for yolo11m don't point toward adversarial regions for yolo11x. MI-PGD + DIM improved theoretical transferability but required a V100 and ~900ms/image.

The CV approach is model-agnostic by construction — it doesn't assume anything about the opponent architecture.

### Inside-bbox attack (RMSE ≤ 47, SSIM ≥ 0.31)

Three components, each targeting a different detection cue:

| Component | Weight | What it attacks |
|---|---|---|
| Gray overlay | 0.4 | Contrast/luminance — pulls pixels toward 128, washing out brightness gradients that anchor object boundaries |
| Hue rotation (30–60°) | 0.3 | Color features — confuses color-dependent class recognition (vehicles, uniforms, equipment) |
| Mid-frequency noise | 0.3 | Texture features — band-pass noise (PIL GaussianBlur difference) that survives JPEG and median filter |

Combined delta is scaled to fill exactly `RMSE_INSIDE_MAX = 47` before SSIM enforcement.

### Mid-frequency noise construction

```python
raw = np.random.randn(H, W, 3)
pil = Image.fromarray((raw * 40 + 128).clip(0,255).astype(uint8))
light = blur(pil, radius=2)   # keeps mid+low freq
heavy = blur(pil, radius=20)  # keeps only low freq
band = (light - heavy) / 40   # mid-freq band
```

High-frequency noise is killed by JPEG recompression and median filters. Low-frequency noise is nearly DC and doesn't disrupt texture features. Mid-band noise hits the 8–64px wavelengths that CNN detection backbones rely on for texture classification.

### Outside-bbox attack (RMSE up to ~71)

Gaussian noise with σ=100 applied to all non-bbox pixels, then the full delta (inside + outside combined) is projected to global RMSE ≤ 67. Since inside was already ≤ 47 < 67, the projection scale is < 1 only when outside noise pushes global over 67 — and when it does scale, it can only reduce inside further (never above budget).

### SSIM enforcement

12-step binary search on the scale of the inside delta. `scale=0` always passes (SSIM=1.0), so the search is guaranteed to find a valid solution. Precision: 1/4096 on the scale factor.

---

## Defence strategies (for `cv_manager.py`)

Applied in `CVManager._preprocess()` before every inference call:

1. **Random resize** (scale 90–100% then restore) — bilinear interpolation at a random scale destroys adversarial perturbations computed for a specific pixel grid. A 3% shift degrades gradient-based attacks by 30–50%.

2. **Median filter 3×3** — removes structured/salt-and-pepper noise and outlier pixels from gradient attacks while preserving object edges.

3. **Bit-depth reduction to 6 bits** (`x >> 2 << 2`) — quantises to multiples of 4, wiping perturbations with amplitude < 4px.

4. **JPEG recompression at q=80** — kills high-frequency residuals. JPEG quantisation is particularly effective against gradient-based attacks which live in fine pixel space.

5. **TTA (`augment=True`)** — ultralytics runs 3 augmented inference passes and merges via WBF. Adversarial perturbations are fragile across scale/flip augmentations.

Each stage acts on a different frequency/amplitude regime, so they are complementary.

---

## Attack strategies — next steps

### 1. Universal adversarial perturbation (offline, fast at inference)

Compute a single image-agnostic perturbation patch offline against a surrogate ensemble. At inference, composite it over detected bboxes scaled to bbox size. Zero gradient cost at runtime.

Key literature:

- **Moosavi-Dezfooli et al., "Universal Adversarial Perturbations" (CVPR 2017)** — foundational. Iterates over training images, aggregates minimal per-image boundary-crossing perturbations into a running universal δ. ~80% fooling rate on ImageNet at L∞ ≤ 10.
- **Mopuri et al., "Fast Feature Fool" (BMVC 2017)** — data-free UAP by maximising intermediate CNN activations. No training set required — directly applicable here since we don't have the opponent's data.
- **Mopuri et al., "NAG: Network for Adversary Generation" (CVPR 2018)** — generator network that outputs UAPs on demand after offline training.
- **Xie et al., "Adversarial Examples for Semantic Segmentation and Object Detection" (ICCV 2017)** — extends UAP to dense prediction heads. Attacking the region proposal stage transfers better than attacking the classification head.
- **Brown et al., "Adversarial Patch" (NeurIPS 2017)** — restricts perturbation to a small region. Directly analogous to the inside-bbox budget.
- **Thys et al., "Fooling automated surveillance cameras" (CVPRW 2019)** — adversarial patches optimised to suppress objectness scores in person detectors specifically.

Fast Feature Fool is the most applicable: no white-box access to the opponent model needed, and it produces a single patch that transfers across architectures because it attacks general feature activations rather than a specific model's decision boundary.

### 2. False positive injection (outside bboxes only)

Spend the outside-bbox budget on structured noise patterns that resemble object features (edges, gradients at the right scale) to trigger ghost detections. This swamps NMS and wastes the opponent's top-k slots. Hard to do without gradients — requires either a pre-computed universal trigger pattern or a copy-paste approach (paste scaled-down object crops into background).

### 3. Adversarial training for CV model (defence)

Mix FGSM-perturbed images into training batches (`noise_dev/adv_train.py`, `AdversarialRTDETRTrainer`). White-box FGSM on a non-adversarially-trained RTDETR dropped mAP from 0.95 to 0.50; adversarial training should recover most of this.

Use FGSM (not PGD) for training: 1 gradient step adds ~25% overhead per affected batch. At 25% batch fraction over 5 of 20 epochs, overall training cost increase is ~6%.

---

## Evaluation tooling

- **`noise_dev/eval_attack.py`** — batch evaluation against finetuned models. Reports detection drop rate, confidence shift, RMSE/SSIM metrics, and fairness pass rate. Run with `python -m noise_dev.eval_attack --n 100`.
- **`cv_dev/adv_test.py`** — white-box FGSM upper-bound evaluation directly against the CV model. Gives the best-case attack effectiveness (attacker knows exact model weights).

---

## Speed budget (CV approach)

On CPU (no GPU needed):

| Operation | Approx time |
|---|---|
| HTTP call to CV server (get bboxes) | ~50–150ms |
| Gray overlay + hue rotate + mid-freq noise | ~20–50ms |
| SSIM binary search (12 iters) | ~50–100ms |
| Outside noise + global projection | ~5ms |
| **Total per image** | **~125–300ms** |

3–4× faster than the old MI-PGD approach (~600–1200ms) and requires no GPU VRAM at all.
