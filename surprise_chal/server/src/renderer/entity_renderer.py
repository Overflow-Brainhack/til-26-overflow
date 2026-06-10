"""entity (unit + building) rendering utilities"""

from __future__ import annotations

import math

import arcade
import arcade.shape_list

from engine.entities.building import Building
from engine.entities.unit import AirUnit, Unit
from engine.hex_grid import HexCoord, axial_to_pixel
from engine.state import GameState
from renderer import sprite_loader
from renderer.animation import MoveEvent, TurnTransition

# up to 20 player colours (used for HP bars and letter-fallback rendering)
PLAYER_COLOURS: list[tuple[int, int, int]] = [
    (220, 50, 50),
    (50, 120, 220),
    (50, 200, 80),
    (220, 180, 30),
    (180, 50, 220),
    (50, 200, 200),
    (220, 100, 50),
    (100, 220, 100),
    (220, 50, 150),
    (50, 100, 180),
    (150, 220, 50),
    (200, 120, 80),
    (80, 150, 220),
    (220, 200, 50),
    (50, 180, 150),
    (180, 100, 220),
    (220, 150, 50),
    (100, 50, 220),
    (50, 220, 150),
    (220, 80, 80),
]

_BUILDING_SHAPES: dict[str, str] = {
    "Base": "B",
    "Mine": "M",
    "Barracks": "K",
    "Factory": "F",
    "Airbase": "A",
}

# entity types that are buildings (for fallback rendering)
_BUILDING_TYPES = frozenset(_BUILDING_SHAPES.keys())

# entity types that are air units
_AIR_TYPES = frozenset(["Fighter", "Bomber"])


def entity_pixel_pos(
    coord_q: int, coord_r: int, cx: float, cy: float, size: float
) -> tuple[float, float]:
    """Convert axial hex coord to screen pixel centre."""
    px, py = axial_to_pixel(HexCoord(coord_q, coord_r), size=size)
    return cx + px, cy + py


def _hp_bar_color(frac: float) -> tuple[int, int, int]:
    if frac > 0.5:
        return (80, 200, 80)
    if frac > 0.25:
        return (200, 200, 50)
    return (200, 60, 60)


def _append_rect_tris(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: tuple[int, int, int, int],
    pts: list,
    colors: list,
) -> None:
    pts += [(x1, y1), (x2, y1), (x2, y2), (x1, y1), (x2, y2), (x1, y2)]
    colors += [color, color, color, color, color, color]


def _flush_hp_bars(
    hp_bar_data: list[tuple[float, float, int, int, int, float]],
) -> None:
    """Draw all batched HP bars in a single GL call. hp_bar_data is [(sx,sy,hp,max_hp,alpha,size)]."""
    if not hp_bar_data:
        return
    pts: list[tuple[float, float]] = []
    colors: list[tuple[int, int, int, int]] = []
    for sx, sy, hp, max_hp, alpha, size in hp_bar_data:
        bar_w = size * 0.9
        bar_h = 2.0
        bar_y = sy - size * 0.55
        frac = max(0.0, hp / max_hp)
        bar_alpha = min(200, alpha)
        _append_rect_tris(
            sx - bar_w / 2,
            bar_y - bar_h / 2,
            sx + bar_w / 2,
            bar_y + bar_h / 2,
            (60, 60, 60, bar_alpha),
            pts,
            colors,
        )
        if frac > 0:
            fill_colour = _hp_bar_color(frac)
            fill_x1 = sx - bar_w / 2
            fill_x2 = fill_x1 + bar_w * frac
            _append_rect_tris(
                fill_x1,
                bar_y - bar_h / 2,
                fill_x2,
                bar_y + bar_h / 2,
                (*fill_colour, bar_alpha),
                pts,
                colors,
            )
    arcade.shape_list.create_triangles_filled_with_colors(pts, colors).draw()


def _draw_single_entity(
    etype: str,
    colour: tuple[int, int, int],
    palette_idx: int,
    sx: float,
    sy: float,
    hp: int,
    max_hp: int,
    alpha: int,
    size: float,
    texture_cache: sprite_loader.TextureCache,
    anim_frame: int,
    sprite_list: arcade.SpriteList,
    hp_bar_data: list | None = None,
) -> None:
    """Draw one entity (sprite or fallback shape) plus its HP bar at (sx, sy).

    Sprites are appended to sprite_list for batched drawing; fallback shapes and
    HP bars are drawn immediately (they appear under sprites if called first).
    When hp_bar_data is provided, HP bar geometry is collected there instead of
    drawn immediately, for later batched rendering via _flush_hp_bars().
    """
    frames = texture_cache.get((etype, palette_idx))
    a = max(0, min(255, alpha))

    if frames:
        frame_idx = anim_frame % len(frames)
        scale = (size * 1.3) / frames[0].width
        sprite = arcade.Sprite(frames[frame_idx], scale=scale)
        sprite.textures = frames
        sprite.center_x = sx
        sprite.center_y = sy
        sprite.alpha = a
        sprite_list.append(sprite)
    else:
        if etype in _BUILDING_TYPES:
            half = size * 0.45
            arcade.draw_rect_filled(
                arcade.XYWH(sx, sy, half * 2, half * 2), (*colour, min(220, a))
            )
            arcade.draw_rect_outline(
                arcade.XYWH(sx, sy, half * 2, half * 2), (0, 0, 0, min(180, a)), 1
            )
            letter = _BUILDING_SHAPES.get(etype, etype[0] if etype else "?")
            arcade.draw_text(letter, sx - 4, sy - 5, (*arcade.color.WHITE[:3], a), 8)
        else:
            radius = size * 0.42 if etype in _AIR_TYPES else size * 0.35
            arcade.draw_circle_filled(sx, sy, radius, (*colour, min(230, a)))
            if etype in _AIR_TYPES:
                arcade.draw_circle_outline(
                    sx, sy, radius, (255, 255, 255, min(180, a)), 1
                )
            letter = etype[0] if etype else "?"
            arcade.draw_text(letter, sx - 4, sy - 5, (*arcade.color.WHITE[:3], a), 8)

    if max_hp > 0:
        if hp_bar_data is not None:
            hp_bar_data.append((sx, sy, hp, max_hp, a, size))
        else:
            bar_w = size * 0.9
            bar_h = 2.0
            bar_y = sy - size * 0.55
            frac = max(0.0, hp / max_hp)
            bar_alpha = min(200, a)
            arcade.draw_rect_filled(
                arcade.XYWH(sx, bar_y, bar_w, bar_h), (60, 60, 60, bar_alpha)
            )
            if frac > 0:
                fill_colour = _hp_bar_color(frac)
                fill_cx = sx - bar_w / 2 + bar_w * frac / 2
                arcade.draw_rect_filled(
                    arcade.XYWH(fill_cx, bar_y, bar_w * frac, bar_h),
                    (*fill_colour, bar_alpha),
                )


def draw_entities(
    state: GameState,
    visible: set[HexCoord],
    player_colour: dict[str, tuple[int, int, int]],
    player_palette_idx: dict[str, int],
    texture_cache: sprite_loader.TextureCache,
    anim_frame_by_type: dict[str, int],
    cx: float,
    cy: float,
    size: float,
    window_w: int,
    window_h: int,
) -> None:
    cull = size * 3
    sprite_list = arcade.SpriteList()
    for entity in state.entities.values():
        if entity.coord not in visible:
            continue
        sx, sy = entity_pixel_pos(entity.coord.q, entity.coord.r, cx, cy, size)
        if sx < -cull or sx > window_w + cull:
            continue
        if sy < -cull or sy > window_h + cull:
            continue
        colour = player_colour.get(entity.owner_id, (200, 200, 200))
        etype = entity.entity_type()
        palette_idx = player_palette_idx.get(entity.owner_id, 0)
        anim_frame = anim_frame_by_type.get(etype, 0)
        _draw_single_entity(
            etype,
            colour,
            palette_idx,
            sx,
            sy,
            entity.hp,
            entity.max_hp,
            255,
            size,
            texture_cache,
            anim_frame,
            sprite_list,
        )
    sprite_list.draw(pixelated=True)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _move_pos(
    evt: "MoveEvent",
    progress: float,
    cx: float,
    cy: float,
    size: float,
) -> tuple[float, float]:
    """Compute animated screen position for a MoveEvent at the given progress (0→1)."""
    fx, fy = entity_pixel_pos(evt.from_coord.q, evt.from_coord.r, cx, cy, size)

    if not evt.succeeded:
        # bounce: slide 40% toward first path step and back
        p0x, p0y = entity_pixel_pos(evt.path[0].q, evt.path[0].r, cx, cy, size)
        peak_x = _lerp(fx, p0x, 0.4)
        peak_y = _lerp(fy, p0y, 0.4)
        if progress <= 0.5:
            t = progress / 0.5
            return _lerp(fx, peak_x, t), _lerp(fy, peak_y, t)
        else:
            t = (progress - 0.5) / 0.5
            return _lerp(peak_x, fx, t), _lerp(peak_y, fy, t)

    # successful: walk full waypoint list (from_coord + each path step)
    waypoints = [evt.from_coord] + evt.path
    n = len(waypoints) - 1
    if n <= 0:
        return fx, fy
    seg = progress * n
    idx = min(int(seg), n - 1)
    t = seg - idx
    wx0, wy0 = entity_pixel_pos(waypoints[idx].q, waypoints[idx].r, cx, cy, size)
    wx1, wy1 = entity_pixel_pos(
        waypoints[idx + 1].q, waypoints[idx + 1].r, cx, cy, size
    )
    # Cross-map step: adjacent hexes are size*√3 apart; anything much larger is a
    # torus wrap — snap to the destination instead of sliding across the screen.
    thresh = size * 2.5
    if (wx1 - wx0) ** 2 + (wy1 - wy0) ** 2 > thresh * thresh:
        return wx1, wy1
    return _lerp(wx0, wx1, t), _lerp(wy0, wy1, t)


def draw_animated_entities(
    transition: "TurnTransition",
    progress: float,
    visible: set[HexCoord] | None,
    player_colour: dict[str, tuple[int, int, int]],
    player_palette_idx: dict[str, int],
    texture_cache: sprite_loader.TextureCache,
    anim_frame_by_type: dict[str, int],
    cx: float,
    cy: float,
    size: float,
    window_w: int,
    window_h: int,
    sprite_list: arcade.SpriteList | None = None,
) -> None:
    """Draw all entities with animated positions/alphas for a mid-transition frame."""
    cull = size * 3
    if sprite_list is None:
        sprite_list = arcade.SpriteList()
    else:
        sprite_list.clear()
    hp_bar_data: list[tuple[float, float, int, int, int, float]] = []

    animated_ids = (
        {e.entity_id for e in transition.moves}
        | {e.entity_id for e in transition.deaths}
        | {e.entity_id for e in transition.spawns}
    )

    # ── static entities (in to_entities, not animated) ──────────────────────
    for eid, e in transition.to_entities.items():
        if eid in animated_ids:
            continue
        coord = HexCoord(e["q"], e["r"])
        if visible is not None and coord not in visible:
            continue
        sx, sy = entity_pixel_pos(e["q"], e["r"], cx, cy, size)
        if sx < -cull or sx > window_w + cull or sy < -cull or sy > window_h + cull:
            continue
        owner = e.get("owner_id", "")
        etype = e.get("type", "")
        colour = player_colour.get(owner, (200, 200, 200))
        palette_idx = player_palette_idx.get(owner, 0)
        fe = transition.from_entities.get(eid)
        hp = e.get("hp", 0)
        max_hp = e.get("max_hp", 1)
        if fe is not None:
            hp = int(_lerp(fe.get("hp", hp), hp, progress))
        _draw_single_entity(
            etype,
            colour,
            palette_idx,
            sx,
            sy,
            hp,
            max_hp,
            255,
            size,
            texture_cache,
            anim_frame_by_type.get(etype, 0),
            sprite_list,
            hp_bar_data,
        )

    # ── moving entities ──────────────────────────────────────────────────────
    for evt in transition.moves:
        coord = evt.to_coord if evt.succeeded else evt.from_coord
        if visible is not None and coord not in visible:
            continue
        sx, sy = _move_pos(evt, progress, cx, cy, size)
        if sx < -cull or sx > window_w + cull or sy < -cull or sy > window_h + cull:
            continue
        colour = player_colour.get(evt.owner_id, (200, 200, 200))
        fe = transition.from_entities.get(evt.entity_id)
        te = transition.to_entities.get(evt.entity_id)
        from_hp = fe.get("hp", 0) if fe else 0
        to_hp = te.get("hp", from_hp) if te else from_hp
        max_hp = (te or fe or {}).get("max_hp", 1)
        hp = int(_lerp(from_hp, to_hp, progress))
        _draw_single_entity(
            evt.entity_type,
            colour,
            evt.palette_idx,
            sx,
            sy,
            hp,
            max_hp,
            255,
            size,
            texture_cache,
            anim_frame_by_type.get(evt.entity_type, 0),
            sprite_list,
            hp_bar_data,
        )

    # ── dying entities (fade out t=0.5→0.9) ─────────────────────────────────
    for evt in transition.deaths:
        if visible is not None and evt.coord not in visible:
            continue
        sx, sy = entity_pixel_pos(evt.coord.q, evt.coord.r, cx, cy, size)
        if sx < -cull or sx > window_w + cull or sy < -cull or sy > window_h + cull:
            continue
        fade = max(0.0, (0.9 - progress) / 0.4)
        alpha = int(255 * fade)
        if alpha <= 0:
            continue
        colour = player_colour.get(evt.owner_id, (200, 200, 200))
        fe = transition.from_entities.get(evt.entity_id, {})
        hp = fe.get("hp", 0)
        max_hp = fe.get("max_hp", 1)
        _draw_single_entity(
            evt.entity_type,
            colour,
            evt.palette_idx,
            sx,
            sy,
            hp,
            max_hp,
            alpha,
            size,
            texture_cache,
            anim_frame_by_type.get(evt.entity_type, 0),
            sprite_list,
            hp_bar_data,
        )

    # ── spawning entities (fade in t=0.6→1.0) ───────────────────────────────
    for evt in transition.spawns:
        if visible is not None and evt.coord not in visible:
            continue
        sx, sy = entity_pixel_pos(evt.coord.q, evt.coord.r, cx, cy, size)
        if sx < -cull or sx > window_w + cull or sy < -cull or sy > window_h + cull:
            continue
        fade = min(1.0, max(0.0, (progress - 0.6) / 0.4))
        alpha = int(255 * fade)
        if alpha <= 0:
            continue
        colour = player_colour.get(evt.owner_id, (200, 200, 200))
        te = transition.to_entities.get(evt.entity_id, {})
        hp = te.get("hp", 0)
        max_hp = te.get("max_hp", 1)
        _draw_single_entity(
            evt.entity_type,
            colour,
            evt.palette_idx,
            sx,
            sy,
            hp,
            max_hp,
            alpha,
            size,
            texture_cache,
            anim_frame_by_type.get(evt.entity_type, 0),
            sprite_list,
            hp_bar_data,
        )

    sprite_list.draw(pixelated=True)
    _flush_hp_bars(hp_bar_data)
