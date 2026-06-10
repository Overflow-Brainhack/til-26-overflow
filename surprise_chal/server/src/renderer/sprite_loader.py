"""Sprite loader: discovers assets, recolors animation frames, and builds a texture cache.

On first call to load_entity_textures(), all animation frames for every entity type are
recolored for each of the 22 palette colors and stored as arcade.Texture objects in memory.
Subsequent draw calls look up textures by (entity_type, palette_index) key.

Asset search order per entity type: assets/ → placeholders/ → letter fallback.
"""

from __future__ import annotations

from pathlib import Path

import arcade
import PIL.Image

from sprites.palette import Palette, base_colors
from sprites.recolor import apply_palette_to_sprite_in_memory

ANIM_FPS: float = 5.0

ENTITY_SPRITE_MAP: dict[str, str] = {
    "Infantry": "infantry",
    "Tank": "tank",
    "Artillery": "artillery",
    "Scout": "scout",
    "Medic": "medic",
    "Fighter": "fighter",
    "Bomber": "bomber",
    "Base": "base",
    "Mine": "mine",
    "Barracks": "barracks",
    "Factory": "factory",
    "Airbase": "airbase",
}

# (entity_type, palette_index) → list of arcade.Texture, one per animation frame
TextureCache = dict[tuple[str, int], list[arcade.Texture]]

_CANDIDATE_DIRS = [Path("assets"), Path("placeholders")]

# Pre-computed RGB tuples for each palette index, matching the sprite recolor colors.
PALETTE_COLOURS: list[tuple[int, int, int]] = [  # ty: ignore[invalid-assignment]
    tuple(round(c * 255) for c in color.convert("srgb").coords())
    for color in base_colors.values()
]


def _find_frames(sprite_name: str) -> list[PIL.Image.Image]:
    """Search candidate dirs for animated or static sprite frames; return PIL images."""
    for assets_dir in _CANDIDATE_DIRS:
        if not assets_dir.is_dir():
            continue
        # Animated unit: numbered frames in a subdirectory
        anim_dir = assets_dir / sprite_name
        if anim_dir.is_dir():
            frames: list[PIL.Image.Image] = []
            i = 1
            while True:
                frame_path = anim_dir / f"{sprite_name}{i}.png"
                if not frame_path.exists():
                    break
                frames.append(PIL.Image.open(frame_path).convert("RGBA"))
                i += 1
            if frames:
                return frames
        # Static building: single flat sprite
        flat_path = assets_dir / f"{sprite_name}.png"
        if flat_path.exists():
            return [PIL.Image.open(flat_path).convert("RGBA")]
    return []


def load_entity_textures() -> TextureCache:
    """Load and recolor all entity sprites for every palette color.

    Must be called after an arcade window (and thus OpenGL context) exists.
    """
    cache: TextureCache = {}
    palette_entries = list(base_colors.items())

    for entity_type, sprite_name in ENTITY_SPRITE_MAP.items():
        base_frames = _find_frames(sprite_name)
        if not base_frames:
            continue

        for palette_idx, (_, base_color) in enumerate(palette_entries):
            palette = Palette.from_base_color(base_color)
            textures: list[arcade.Texture] = []
            for frame_idx, frame_img in enumerate(base_frames):
                recolored = apply_palette_to_sprite_in_memory(frame_img, palette)
                name = f"{entity_type}_{palette_idx}_{frame_idx}"
                textures.append(arcade.Texture(hash=name, image=recolored))
            cache[(entity_type, palette_idx)] = textures

    return cache
