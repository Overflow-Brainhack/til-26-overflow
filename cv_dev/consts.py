from cv_dev.data_types import RawCategory

from pathlib import Path
import json

FIXED_WIDTH: int = 1280
FIXED_HEIGHT: int = 1280

DATA_PATH = Path("data/cv/")
IMAGE_PATH = Path("data/cv/images/")
JSON_PATH = Path("data/cv/annotations.json")
DATASETS_PATH = Path("cv_dev/datasets")
RESULTS_PATH = Path("cv_dev/results")
OBJECT_BANK_PATH = Path("cv_dev/object_bank")
BACKGROUNDS_PATH = Path("cv_dev/backgrounds")
SYNTHETIC_DATA_PATH = Path("data/synthetic_cv")
SYNTHETIC_IMAGE_PATH = Path("data/synthetic_cv/images")
SYNTHETIC_JSON_PATH = Path("data/synthetic_cv/annotations.json")

# generate_synthetic.py tuning
SYNTH_MAX_OBJECTS: int = 5  # max objects pasted per image (0–N chosen uniformly)
SYNTH_MIN_SCALE: float = 0.05  # object width as fraction of background width
SYNTH_MAX_SCALE: float = 0.30
SYNTH_MAX_PLACEMENT_TRIES: int = 20  # retries before skipping an object


with open(JSON_PATH, "r", encoding="utf-8") as f:
    categories: list[RawCategory] = json.load(f)["categories"]

CATEGORIES = [c["name"] for c in categories]
NUM_CATEGORIES = len(CATEGORIES)  # 18

TRAIN_PATH = Path("data/cv/train")
VAL_PATH = Path("data/cv/val")
SYNTHETIC_TRAIN_PATH = Path("data/synthetic_cv/train")
SYNTHETIC_VAL_PATH = Path("data/synthetic_cv/val")

TRAIN_OUTPUT = Path(__file__).parent.parent / "cv-training" / "trains"
