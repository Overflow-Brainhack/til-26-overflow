# CV Challenge Context

## Task

Object detection on 18 military/vehicle/aerial categories. Images are synthetic composites: objects pasted onto random backgrounds (forests, trees, skies, etc).

## Categories (id: name)

0: cargo aircraft, 1: commercial aircraft, 2: drone, 3: fighter jet, 4: fighter plane,
5: helicopter, 6: light aircraft, 7: missile, 8: truck, 9: car, 10: tank, 11: bus,
12: van, 13: cargo ship, 14: yacht, 15: cruise ship, 16: warship, 17: sailboat

## Current model

- **RT-DETR-X** (ultralytics), imgsz=1280, rect=True, batch=2
- Training script: `cv_dev/train.py` → `train_rtdetr(50)`
- Model weights served from: `cv/models/rtdetr-x-43.pt` (or latest best.pt)

## Results

| Checkpoint | Val mAP50-95 | Test Accuracy | Test Speed |
|---|---|---|---|
| RT-DETR-L epoch 15 | 0.963 | 64.3% | 93.6% |
| RT-DETR-X (current best) | — | **72.7%** | **88.3%** |

- Large train/val vs test gap — root cause: model overfits to compositing artifacts of training data
- TTA (hflip + vflip) tested: improved accuracy but dropped speed to ~80% → removed

## Key problems to solve

1. Model overfits to training compositing artifacts, doesn't generalise to test compositing style
2. Val mAP >> test accuracy (97% vs 72.7%) — need better distribution coverage
3. Speed/accuracy tradeoff — RT-DETR-X is slower than L; need to stay above ~85% speed

## Improvement directions (priority order)

1. **Synthetic data generation** (current focus): paste extracted objects onto Mapillary backgrounds to diversify compositing artifacts
2. **Augmentations**: already improved (perspective, affine, shadows, motion blur, occlusion via CoarseDropout)
3. **Longer training**: keep running RT-DETR-X past 50 epochs if loss still declining
4. **DETR-ResNet-101** (HuggingFace): training code added in `train.py`, requires `cv_dev/datasets/train.pt` + `val.pt` from `make_dataset.py`

## Synthetic data pipeline (in progress)

Goal: 500+ synthetic images with random object-on-background compositing to bridge the distribution gap.

### Step 1 — Object bank extraction (`cv_dev/extract_objects.py`)
- Crops every annotated object using COCO bounding boxes
- Removes background via `rembg` → saved as RGBA PNG to `cv_dev/object_bank/{category}/`
- Run: `python -m cv_dev.extract_objects`
- After running: use `clean_object_bank()` in `cv_dev/utils.py` to review and delete bad rembg crops

### Step 2 — Background scraping (`cv_dev/scrape_backgrounds.py`)
- Downloads 500 ground-level images from Mapillary API (diverse global locations)
- Requires `MAPILLARY_TOKEN` env variable (free account at mapillary.com)
- Saved to `cv_dev/backgrounds/`
- Run: `python -m cv_dev.scrape_backgrounds`

### Step 3 — Synthetic image generation (`cv_dev/generate_synthetic.py`) [TODO]
- Picks a random background image
- Randomly selects 3–6 objects from the bank (different categories allowed)
- Scales, slightly rotates, and pastes objects onto the background
- Outputs COCO-format annotation JSON + images to `data/cv/synthetic/`
- Then merged with real training data for `make_yolo_dataset.py`

## File layout

- `cv_dev/train.py` — ultralytics RTDETRTrainer + DETR-HF trainer
- `cv_dev/make_yolo_dataset.py` — COCO → YOLO format, writes data/cv/train/ and data/cv/val/
- `cv_dev/test.py` — local eval, runs model against val split and reports mAP
- `cv_dev/consts.py` — shared paths, loads categories from data/cv/annotations.json
- `cv_dev/extract_objects.py` — rembg-based object bank builder
- `cv_dev/scrape_backgrounds.py` — Mapillary background downloader
- `cv_dev/utils.py` — `process_annotations`, `clean_object_bank`
- `cv/src/cv_manager.py` — inference wrapper used by the FastAPI server
