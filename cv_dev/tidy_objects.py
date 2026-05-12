from cv_dev.consts import OBJECT_BANK_PATH

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import subprocess
import sys
import numpy as np


def _display_in_terminal(path: Path) -> None:
    """Use timg in kitten ssh terminal to view images"""
    cmd = ["timg", str(path)]
    try:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            sys.stdout.buffer.write(result.stdout)
            sys.stdout.flush()
            return
    except FileNotFoundError:
        pass
    print(f"  [cannot display image {path}]")


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
    Review SAM2 crops that look bad (too sparse or background not removed).

    min_coverage: flag if foreground alpha fraction is below this (object fully masked out)
    max_coverage: flag if foreground alpha fraction is above this (background not removed)

    Interactive prompts per flagged image:
        y  — delete this crop
        n  — keep it
        a  — delete this and all remaining flagged without prompting
        q  — stop reviewing, keep everything remaining
    """
    pngs = sorted(bank_path.rglob("*.png"))
    flagged: list[tuple[Path, float]] = []

    print(f"Scanning {len(pngs)} crops...")
    for p in tqdm(pngs, "Checking alpha coverage", colour="Purple"):
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

    print(
        f"\n{len(flagged)} flagged crops (coverage outside [{min_coverage:.0%}, {max_coverage:.0%}]):"
    )
    deleted = 0
    delete_all = False

    for i, (p, cov) in enumerate(flagged):
        if delete_all:
            p.unlink()
            deleted += 1
            continue

        print(
            f"\n[{i + 1}/{len(flagged)}] {p.parent.name}/{p.name}  coverage={cov:.1%}"
        )
        _display_in_terminal(p)

        while True:
            choice = (
                input("  Delete? [y]es / [n]o / [a]ll remaining / [q]uit: ")
                .strip()
                .lower()
            )
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


if __name__ == "__main__":
    clean_object_bank(OBJECT_BANK_PATH)
