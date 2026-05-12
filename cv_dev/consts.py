from cv_dev.data_types import RawCategory

from pathlib import Path
import json

FIXED_WIDTH: int = 1024
FIXED_HEIGHT: int = 1024

DATA_PATH = Path("data/cv/")
IMAGE_PATH = Path("data/cv/images/")
JSON_PATH = Path("data/cv/annotations.json")
DATASETS_PATH = Path("cv_dev/datasets")
RESULTS_PATH = Path("cv_dev/results")
OBJECT_BANK_PATH = Path("cv_dev/object_bank")
BACKGROUNDS_PATH = Path("cv_dev/backgrounds")


with open(JSON_PATH, "r", encoding="utf-8") as f:
    categories: list[RawCategory] = json.load(f)["categories"]

CATEGORIES = [c["name"] for c in categories]
NUM_CATEGORIES = len(CATEGORIES)  # 18

TRAIN_PATH = Path("data/cv/train")
VAL_PATH = Path("data/cv/val")

TRAIN_OUTPUT = Path(__file__).parent.parent / "cv-training" / "trains"
