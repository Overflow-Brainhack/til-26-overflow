from cv_dev.consts import IMAGE_PATH, JSON_PATH, CATEGORIES

from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import json
import rembg

OBJECT_BANK_PATH = Path("cv_dev/object_bank")


def extract_objects() -> None:
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    file_by_id = {img["id"]: img["file_name"] for img in data["images"]}

    ann_by_image: dict[int, list] = defaultdict(list)
    for ann in data["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    for cat in CATEGORIES:
        (OBJECT_BANK_PATH / cat).mkdir(parents=True, exist_ok=True)

    # create session once — reusing it avoids reloading the model per crop
    session = rembg.new_session()

    for image_id, annotations in tqdm(ann_by_image.items(), "Extracting objects"):
        img = Image.open(IMAGE_PATH / file_by_id[image_id]).convert("RGB")

        for ann in annotations:
            x, y, w, h = [int(v) for v in ann["bbox"]]
            x, y = max(0, x), max(0, y)
            w, h = min(w, img.width - x), min(h, img.height - y)
            if w <= 0 or h <= 0:
                continue

            crop = img.crop((x, y, x + w, y + h))
            crop_rgba = rembg.remove(crop, session=session)

            category_name = CATEGORIES[ann["category_id"]]
            out_path = OBJECT_BANK_PATH / category_name / f"{ann['id']}.png"
            crop_rgba.save(out_path)


if __name__ == "__main__":
    extract_objects()
