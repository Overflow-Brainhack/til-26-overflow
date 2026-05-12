from cv_dev.data_types import RawAnnotation, ProcessedAnnotation
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import subprocess
import sys
import numpy as np


def _display_in_terminal(path: Path) -> None:
    """Try chafa (Arch), then kitty icat, then PIL fallback."""
    for cmd in (["chafa", "--size=60x30", str(path)], ["kitty", "+kitten", "icat", str(path)]):
        try:
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                sys.stdout.buffer.write(result.stdout)
                sys.stdout.flush()
                return
        except FileNotFoundError:
            continue
    # fallback: open in default viewer and pause
    print(f"  [cannot display inline — opening: {path}]")
    Image.open(path).show()


def _alpha_coverage(img: Image.Image) -> float:
    """Fraction of pixels with alpha > 128."""
    alpha = np.array(img)[:, :, 3]
    return float((alpha > 128).sum()) / alpha.size


def clean_object_bank(
    bank_path: Path,
    min_coverage: float = 0.04,
    max_coverage: float = 0.97,
) -> None:
    """
    Review rembg crops that look bad (too sparse or background not removed).

    min_coverage: flag if foreground alpha fraction is below this (rembg wiped the object)
    max_coverage: flag if foreground alpha fraction is above this (background wasn't removed)

    Interactive prompts per flagged image:
        y  — delete this crop
        n  — keep it
        a  — delete this and all remaining flagged without prompting
        q  — stop reviewing, keep everything remaining
    """
    pngs = sorted(bank_path.rglob("*.png"))
    flagged: list[tuple[Path, float]] = []

    print(f"Scanning {len(pngs)} crops...")
    for p in tqdm(pngs, "Checking alpha coverage"):
        try:
            img = Image.open(p).convert("RGBA")
            cov = _alpha_coverage(img)
            if cov < min_coverage or cov > max_coverage:
                flagged.append((p, cov))
        except Exception as e:
            print(f"  Error reading {p}: {e}")

    if not flagged:
        print("No bad crops found.")
        return

    print(f"\n{len(flagged)} flagged crops (coverage outside [{min_coverage:.0%}, {max_coverage:.0%}]):")
    deleted = 0
    delete_all = False

    for i, (p, cov) in enumerate(flagged):
        if delete_all:
            p.unlink()
            deleted += 1
            continue

        print(f"\n[{i+1}/{len(flagged)}] {p.parent.name}/{p.name}  coverage={cov:.1%}")
        _display_in_terminal(p)

        while True:
            choice = input("  Delete? [y]es / [n]o / [a]ll remaining / [q]uit: ").strip().lower()
            if choice in ("y", "n", "a", "q"):
                break
            print("  Enter y, n, a, or q.")

        if choice == "y":
            p.unlink()
            deleted += 1
        elif choice == "a":
            p.unlink()
            deleted += 1
            delete_all = True
        elif choice == "q":
            print(f"  Stopped. {len(flagged) - i - 1} remaining flagged crops kept.")
            break

    print(f"\nDone. Deleted {deleted}/{len(flagged)} flagged crops.")


def process_annotations(
    annotations: list[RawAnnotation], num_images: int
) -> list[dict[str, list[ProcessedAnnotation]]]:
    out: list[dict[str, list[ProcessedAnnotation]]] = [
        {str(i): []} for i in range(num_images)
    ]

    for annotation in tqdm(annotations, "Processing annotations", colour="Green"):
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        bbox = annotation["bbox"]

        assert isinstance(image_id, int), (
            f"Expected image_id to be an integer, got {type(image_id)}"
        )

        assert isinstance(category_id, int), (
            f"Expected category_id to be an integer, got {type(category_id)}"
        )

        assert isinstance(bbox, list) and all(
            isinstance(x, float | int) for x in bbox
        ), f"Expected bbox to be an list of number values, got {type(bbox)}"

        out[image_id][str(image_id)].append(
            {
                "category_id": category_id,
                "bbox": bbox,
            }
        )
        if annotation["iscrowd"] != 0:
            print(f"iscrowd non-zero at id {annotation['id']}")

    return out
