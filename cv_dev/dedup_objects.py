from cv_dev.consts import OBJECT_BANK_PATH, CATEGORIES

from pathlib import Path
from PIL import Image
import numpy as np
import hashlib
import argparse
from tqdm import tqdm


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _dhash(img: Image.Image, size: int = 8) -> int:
    """Difference hash: compare adjacent pixel brightness across a grid."""
    gray = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = np.array(gray, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_category(cat_path: Path, hamming_threshold: int) -> int:
    pngs = sorted(cat_path.glob("*.png"))
    if len(pngs) < 2:
        return 0

    # Pass 1 — exact byte duplicates via MD5
    seen_md5: dict[str, Path] = {}
    exact_dupes: set[Path] = set()
    for p in pngs:
        h = _md5(p)
        if h in seen_md5:
            exact_dupes.add(p)
        else:
            seen_md5[h] = p

    survivors = [p for p in pngs if p not in exact_dupes]

    # Pass 2 — perceptual near-duplicates via dHash
    # Greedy: keep the first image in sorted order; mark later images as dupes
    # if their Hamming distance to any kept image is within threshold.
    hashes: list[tuple[int, Path]] = []
    for p in survivors:
        try:
            hashes.append((_dhash(Image.open(p)), p))
        except Exception:
            continue

    perceptual_dupes: set[Path] = set()
    kept: list[int] = []  # dHash values of images we're keeping

    for dhash, p in hashes:
        if any(_hamming(dhash, k) <= hamming_threshold for k in kept):
            perceptual_dupes.add(p)
        else:
            kept.append(dhash)

    to_delete = exact_dupes | perceptual_dupes
    for p in to_delete:
        p.unlink()
    return len(to_delete)


def dedup_objects(hamming_threshold: int = 5) -> None:
    """
    Remove duplicate objects from every category in the object bank.

    hamming_threshold: max bit difference between two dHashes to consider images
    near-identical (0 = pixel-perfect match only, 5 = very similar, 10 = loose).
    """
    total = 0
    for cat in tqdm(CATEGORIES, "Deduplicating categories", colour="Cyan"):
        cat_path = OBJECT_BANK_PATH / cat
        if not cat_path.exists():
            continue
        before = len(list(cat_path.glob("*.png")))
        removed = dedup_category(cat_path, hamming_threshold)
        if removed:
            print(f"  {cat}: {before} → {before - removed}  (-{removed})")
        total += removed
    print(f"\nDone. Removed {total} duplicates across all categories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Hamming distance threshold for perceptual duplicates (default: 5)",
    )
    args = parser.parse_args()
    dedup_objects(args.threshold)
