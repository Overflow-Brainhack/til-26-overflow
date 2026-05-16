Foundational:

- Moosavi-Dezfooli et al., "Universal Adversarial Perturbations" (CVPR 2017) — the original paper. Computes a single image-agnostic δ by iterating over training images: for each image, find the minimal perturbation that pushes it across the decision boundary, and aggregate these into a running universal δ. Fooling rate ~80% on ImageNet classifiers with L∞ ≤ 10.

Data-free (relevant for black-box):

- Mopuri et al., "Fast Feature Fool" (BMVC 2017) — generates UAP without any training data by maximising intermediate CNN feature activations. No need to know the training set. Directly applicable here since you don't have the opponent's training data.
- Mopuri et al., "NAG: Network for Adversary Generation" (CVPR 2018) — trains a generator network to output UAPs on demand. Slower to set up but fast at inference.

For object detection specifically:

- Xie et al., "Adversarial Examples for Semantic Segmentation and Object Detection" (ICCV 2017) — extends the UAP concept to dense prediction heads. Key finding: attacking the region proposal stage is more transferable than attacking the classification head.
- Chow et al., "Understanding Catastrophic Overfitting in Adversarial Training" (2019) — UAPs computed against ensembles transfer much better than single-model ones.

Physical / patch variants:

- Brown et al., "Adversarial Patch" (NeurIPS 2017) — constrained to a small region rather than spread over the whole image. More practical for real-world deployment and directly analogous to your inside-bbox budget.
- Thys et al., "Fooling automated surveillance cameras" (CVPRW 2019) — applies adversarial patches specifically to fool person detectors. Optimises the patch to maximise objectness suppression across many person images.

Why this would help you:
  The core value is zero runtime cost — you compute the perturbation offline once against your surrogate ensemble, then simply add it (clamped to budget) to every input image at inference. For your constraints (RMSE ≤ 47 inside), you'd optimise the patch to sit within that budget across the training distribution, then scale to fill it per-image. The Fast Feature Fool approach is the most relevant since it doesn't require white-box access to the opponent model.
