"""
Takes an existing pixel art sprite (in .png format) and apply a new colour scheme to it.

Specifically, it replaces all pixels with the following 3 colours:
- #319b3d (HSL 126 0.5 0.4) -> lighter variant of base
- #158422 (HSL 126 0.7 0.3) -> base colour
- #265a2c (HSL 126 0.4 0.25) -> darker variant of base

All other colours are left untouched to preserve basic details of the original.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .palette import Palette, base_colors

ASSETS_DIR = Path("assets")


_COLOR_MAPPING: list[tuple[tuple[int, int, int, int], str]] = [
    ((49, 155, 61, 255), "light"),  # #319b3d
    ((21, 132, 34, 255), "base"),  # #158422
    ((38, 90, 44, 255), "dark"),  # #265a2c
]


def apply_palette_to_sprite_in_memory(
    img: Image.Image, palette: Palette
) -> Image.Image:
    """Apply palette recoloring to a PIL Image in memory; returns a new recolored image."""
    img = img.convert("RGBA")
    data = np.array(img)
    target_colors = palette.get_rgba()
    for original_color, key in _COLOR_MAPPING:
        mask = np.all(data == original_color, axis=-1)
        data[mask] = target_colors[key].astype(np.uint8)
    return Image.fromarray(data)


def apply_palette_to_sprite(
    sprite_path: Path,
    output_path: Path,
    palette: Palette,
):
    """Applies the given color palette to the sprite at sprite_path and saves the result to output_path."""
    img = Image.open(sprite_path)
    apply_palette_to_sprite_in_memory(img, palette).save(output_path)


def generate_colored_sprites(sprite_name: str):
    """Generates colored sprites for all base colors."""
    output_folder = ASSETS_DIR / f"{sprite_name}_colored"
    output_folder.mkdir(exist_ok=True)
    # delete existing files in output folder
    for file in output_folder.glob(f"{sprite_name}_*.png"):
        file.unlink()
    for i, (color_name, base_color) in enumerate(base_colors.items()):
        palette = Palette.from_base_color(base_color)
        apply_palette_to_sprite(
            sprite_path=ASSETS_DIR / f"{sprite_name}.png",
            output_path=output_folder
            / f"{sprite_name}_{i}_{color_name.replace(' ', '_')}.png",
            palette=palette,
        )


def generate_colored_sprite_preview(sprite_name: str):
    """Generates a preview image showing all the colored sprites."""
    output_folder = ASSETS_DIR / f"{sprite_name}_colored"
    sprite_paths = list(output_folder.glob(f"{sprite_name}_*.png"))
    # order by i value in filename
    sprite_paths.sort(key=lambda p: int(p.stem.split("_")[1]))
    sprites = [Image.open(path) for path in sprite_paths]

    # Create a grid to display the sprites
    cols = 3
    rows = (len(sprites) + cols - 1) // cols
    sprite_width, sprite_height = sprites[0].size
    preview_image = Image.new("RGBA", (cols * sprite_width, rows * sprite_height))

    for idx, sprite in enumerate(sprites):
        x = (idx % cols) * sprite_width
        y = (idx // cols) * sprite_height
        preview_image.paste(sprite, (x, y))

    preview_image.save(output_folder / "preview.png")


if __name__ == "__main__":
    # generate recolored previews of all sprites
    for sprite_name in [
        "barracks",
        "infantry",
        "medic",
        "scout",
        "heavyInfantry",
        "factory",
        "tank",
        "artillery",
        "bomber",
        "fighter",
    ]:
        generate_colored_sprites(sprite_name)
        generate_colored_sprite_preview(sprite_name)
