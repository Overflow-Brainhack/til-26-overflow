# CV Challenge Context

## Task
Object detection on 18 military/vehicle/aerial categories. Images are synthetic composites: objects pasted onto random backgrounds (forests, trees, skies, etc).

## Categories (id: name)
0: cargo aircraft, 1: commercial aircraft, 2: drone, 3: fighter jet, 4: fighter plane,
5: helicopter, 6: light aircraft, 7: missile, 8: truck, 9: car, 10: tank, 11: bus,
12: van, 13: cargo ship, 14: yacht, 15: cruise ship, 16: warship, 17: sailboat

## Current model
- RT-DETR-L (ultralytics), trained 30 epochs, imgsz=1024, batch=8
- Weights: `cv/models/epoch15.pt` (epoch 15 used for submission)
- Training script: `cv_dev/train.py`

## Results
- Val mAP50-95 at epoch 15: 0.963, at epoch 30: 0.975 (still improving)
- **Test server (epoch 15): 64.3% accuracy, 93.6% speed**
- Huge train/val vs test gap — root cause is distribution mismatch from synthetic compositing

## Key problems to solve
1. Model learns compositing artifacts of training data, doesn't generalize to test compositing
2. Training only 30 epochs — loss still declining
3. No perspective/affine per-object augmentation
4. No TTA at inference

## Improvement directions (priority order)
1. **Augmentations**: perspective/affine, edge blur, aggressive scale jitter, random shadows, mosaic, cutout
2. **Longer training**: 50+ epochs, loss still declining at epoch 30
3. **Model**: try RT-DETRv2-X or YOLOv11x for better generalization
4. **Inference**: TTA (hflip + original), tune confidence threshold from default 0.25
5. **Image size**: try 1280 if GPU allows

## File layout
- `cv_dev/train.py` — ultralytics RTDETRTrainer
- `cv_dev/make_yolo_dataset.py` — converts COCO annotations to YOLO format, writes data/cv/train/ and data/cv/val/
- `cv_dev/test.py` — local eval script (hardcoded path to weights, needs updating)
- `cv_dev/consts.py` — shared paths, loads categories from data/cv/annotations.json
- `cv/src/cv_manager.py` — inference wrapper used by server
