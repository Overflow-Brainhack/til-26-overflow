"""hex tile rendering utilities"""

from __future__ import annotations

import math

import arcade
import arcade.shape_list

from engine.hex_grid import HexCoord, axial_to_pixel
from engine.terrain import TerrainType

TERRAIN_COLOURS: dict[TerrainType, tuple[int, int, int]] = {
    TerrainType.NORMAL: (100, 140, 80),
    TerrainType.ELEVATED: (180, 160, 120),
    TerrainType.DIFFICULT: (80, 100, 60),
    TerrainType.CONCEALMENT: (60, 100, 90),
    TerrainType.RICH_RESOURCE: (200, 180, 80),
}

FOG_COLOUR = (30, 30, 40, 200)


def build_terrain_shape_list(
    tiles: dict,
    size: float,
    include_outlines: bool = True,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> arcade.shape_list.ShapeElementList:
    """build a batched ShapeElementList for all terrain tiles at the given hex size.

    offset_x/offset_y bake a fixed translation into every vertex so the list can be
    drawn without ever touching center_x/center_y (which triggers glCopyBufferSubData).
    """
    sl: arcade.shape_list.ShapeElementList = arcade.shape_list.ShapeElementList()
    r = size - 1
    offsets = [
        (
            r * math.cos(math.pi / 180 * (60 * i - 30)),
            r * math.sin(math.pi / 180 * (60 * i - 30)),
        )
        for i in range(6)
    ]

    for coord, tile in tiles.items():
        px, py = axial_to_pixel(coord, size=size)
        pts = [(px + dx + offset_x, py + dy + offset_y) for dx, dy in offsets]
        colour = TERRAIN_COLOURS.get(tile.terrain, (100, 100, 100))
        sl.append(arcade.shape_list.create_polygon(pts, (*colour, 255)))
        if include_outlines:
            sl.append(arcade.shape_list.create_line_loop(pts, (0, 0, 0, 80), 1))

    return sl


def draw_hex(
    x: float,
    y: float,
    size: float,
    colour: tuple[int, int, int],
    alpha: int = 255,
    outline: bool = True,
) -> None:
    """draw a pointy-top filled hexagon"""
    points = []
    for i in range(6):
        angle = math.pi / 180 * (60 * i - 30)
        points.append((x + size * math.cos(angle), y + size * math.sin(angle)))
    arcade.draw_polygon_filled(points, (*colour, alpha))
    if outline:
        arcade.draw_polygon_outline(points, (0, 0, 0, 60), 1)


def draw_tile_flash(
    coord_q: int,
    coord_r: int,
    color: tuple[int, int, int],
    alpha: int,
    cx: float,
    cy: float,
    size: float,
) -> None:
    """Draw a colored hex overlay on a tile (used for attack/damage flash effects)."""
    px, py = axial_to_pixel(HexCoord(coord_q, coord_r), size=size)
    draw_hex(cx + px, cy + py, size - 1, color, max(0, min(255, alpha)), outline=False)


def _hex_tri_points(hx: float, hy: float, r: float) -> list[tuple[float, float]]:
    """Return 12 points forming 4 triangles (fan from v0) covering a pointy-top hex."""
    vs = [
        (
            hx + r * math.cos(math.pi / 180 * (60 * i - 30)),
            hy + r * math.sin(math.pi / 180 * (60 * i - 30)),
        )
        for i in range(6)
    ]
    return [
        vs[0],
        vs[1],
        vs[2],
        vs[0],
        vs[2],
        vs[3],
        vs[0],
        vs[3],
        vs[4],
        vs[0],
        vs[4],
        vs[5],
    ]


def draw_tile_flashes_batched(
    flashes: list[tuple[int, int, tuple[int, int, int], int]],
    cx: float,
    cy: float,
    size: float,
) -> None:
    """Batch-draw all tile flash overlays in a single GL call.

    flashes is a list of (coord_q, coord_r, color, alpha).
    """
    if not flashes:
        return
    pts: list[tuple[float, float]] = []
    colors: list[tuple[int, int, int, int]] = []
    r = size - 1
    for coord_q, coord_r, color, alpha in flashes:
        px, py = axial_to_pixel(HexCoord(coord_q, coord_r), size=size)
        c: tuple[int, int, int, int] = (*color, max(0, min(255, alpha)))
        for pt in _hex_tri_points(cx + px, cy + py, r):
            pts.append(pt)
            colors.append(c)
    arcade.shape_list.create_triangles_filled_with_colors(pts, colors).draw()


def draw_beam(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: tuple[int, int, int],
    alpha: int,
    line_width: float = 2.0,
) -> None:
    """Draw an attack beam line between two world-space positions."""
    r, g, b = color
    arcade.draw_line(x1, y1, x2, y2, (r, g, b, max(0, min(255, alpha))), line_width)


def draw_terrain_tiles(
    tiles: dict,
    visible: set[HexCoord],
    cx: float,
    cy: float,
    size: float,
    window_w: int,
    window_h: int,
) -> None:
    cull = size * 2
    for coord, tile in tiles.items():
        px, py = axial_to_pixel(coord, size=size)
        sx = cx + px
        sy = cy + py
        if sx < -cull or sx > window_w + cull:
            continue
        if sy < -cull or sy > window_h + cull:
            continue
        colour = TERRAIN_COLOURS.get(tile.terrain, (100, 100, 100))
        draw_hex(sx, sy, size - 1, colour)
        if coord not in visible:
            draw_hex(sx, sy, size - 1, (30, 30, 40), alpha=190, outline=False)
