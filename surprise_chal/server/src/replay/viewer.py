"""interactive replay viewer using arcade"""

from __future__ import annotations

import bisect
import gc
import math
import time
import typing
from enum import Enum, auto
from math import pi, sin
from pathlib import Path

import arcade
import arcade.shape_list
import pyglet
from arcade.camera import Camera2D
from arcade.texture_atlas import DefaultTextureAtlas

from engine.constants import TREATY_CUTOFF_TURN
from engine.hex_grid import HexCoord, HexGrid, axial_to_pixel
from engine.map_gen import MapGenerator
from renderer import sprite_loader
from renderer.animation import TurnTransition, build_transition
from renderer.entity_renderer import (
    draw_animated_entities,
)
from renderer.hex_renderer import (
    build_terrain_shape_list,
    draw_beam,
    draw_tile_flashes_batched,
)
from replay.recorder import ReplayLoader


class _AnimState(Enum):
    PAUSED = auto()
    ANIMATING = auto()


# colours per player index (up to 20)
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

WINDOW_W = 1280
WINDOW_H = 720
HEX_SIZE = 12.0  # pixels per hex vertex (main view)
MINIMAP_SCALE = 0.30  # fraction of main hex size for minimap (2x each dimension)
MINIMAP_PAD = 10  # pixels from edge
MINIMAP_ALPHA = 200
MINIMAP_EMPTY_COLOUR = (70, 70, 75, 255)  # grey base for tiles with no entity
LABEL_ZOOM_MIN = 1.5  # only draw per-entity letters once zoomed in this far

# ── treaty graph constants ────────────────────────────────────────────────────
# Square node-graph panel pinned to the right edge, sitting just above the minimap.
# Teams are placed on a circle; a line between two nodes = an active peace treaty
# this turn. Per the treaty cutoff, no lines are drawn from that turn onward.
_TREATY_PANEL_SIZE = 170

# ── chat panel constants ──────────────────────────────────────────────────────
_CHAT_PANEL_W = 380  # right-side column, sits between the gold chart and the minimap
_CHAT_FONT = 10
_CHAT_CHAR_W = 6.4  # approximate px per char at font size 10
_CHAT_LINE_H = 15
_CHAT_INDENT_PX = 14.0  # continuation-line x indent
_CHAT_MAX_ROWS = 60  # generous safety cap; the available gap (fit_rows) governs height


class _TreatySource(typing.Protocol):
    """Anything that can report treaty pairs for a turn (ReplayLoader; a test stub)."""

    def treaties_for_turn(self, index: int) -> list[dict]: ...


def treaty_edges_for_display(loader: _TreatySource, turn: int) -> list[dict]:
    """Treaty pairs the graph should draw at a displayed turn — encodes the
    'no lines from the cutoff onward' rule in one testable place (pure, no arcade).
    Returns the recorded pairs for the turn, or [] once `turn >= TREATY_CUTOFF_TURN`
    (which also covers the recorder's off-by-one stale-pair on the cutoff frame)."""
    if turn >= TREATY_CUTOFF_TURN:
        return []
    return loader.treaties_for_turn(turn)


def _chat_wrap(text: str, max_chars: int) -> list[str]:
    """Word-wrap text into lines of at most max_chars characters."""
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = (line + " " + word) if line else word
        if len(candidate) <= max_chars:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [""]


def _chat_prefix_segs(
    turn: int | str,
    sender: str,
    recipient: str | None,
    player_colour: dict[str, tuple[int, int, int]],
) -> tuple[list[tuple[str, float, tuple[int, int, int, int]]], float]:
    """Build colored prefix segments for one chat row. Returns (segs, body_start_x)."""
    segs: list[tuple[str, float, tuple[int, int, int, int]]] = []
    x = 8.0

    turn_str = f"[T{turn}] "
    segs.append((turn_str, x, (130, 130, 130, 255)))
    x += len(turn_str) * _CHAT_CHAR_W

    if sender == "__system__":
        star = "★ "
        segs.append((star, x, (220, 200, 80, 255)))
        x += len(star) * _CHAT_CHAR_W
    elif recipient:
        sc: tuple[int, int, int, int] = (
            *player_colour.get(sender, (200, 200, 200)),
            255,
        )  # type: ignore[assignment]
        rc: tuple[int, int, int, int] = (
            *player_colour.get(recipient, (200, 200, 200)),
            255,
        )  # type: ignore[assignment]
        open_part = f"[{sender}"
        segs.append((open_part, x, sc))
        x += len(open_part) * _CHAT_CHAR_W
        segs.append((" → ", x, (180, 180, 180, 255)))
        x += 3 * _CHAT_CHAR_W
        close_part = f"{recipient}]"
        segs.append((close_part, x, rc))
        x += len(close_part) * _CHAT_CHAR_W
        segs.append((": ", x, (180, 180, 180, 255)))
        x += 2 * _CHAT_CHAR_W
    else:
        sc = (*player_colour.get(sender, (200, 200, 200)), 255)  # type: ignore[assignment]
        bracket = f"[{sender}]"
        segs.append((bracket, x, sc))
        x += len(bracket) * _CHAT_CHAR_W
        segs.append((": ", x, (180, 180, 180, 255)))
        x += 2 * _CHAT_CHAR_W

    return segs, x


class ReplayViewer(arcade.Window):
    def __init__(self, replay_path: str) -> None:
        super().__init__(
            WINDOW_W,
            WINDOW_H,
            title=f"replay: {Path(replay_path).name}",
            resizable=True,
        )
        # on_resize fires from inside super().__init__ before our attrs exist — the
        # guard makes the relayout a no-op until __init__ has finished wiring things up
        self._ready = False
        self.set_update_rate(
            1 / 30
        )  # 30fps — gives OS headroom to deliver mouse events
        self._loader = ReplayLoader(replay_path)
        self._header = self._loader.header
        self._map_w, self._map_h = self._loader.map_dims()
        self._grid = HexGrid(self._map_w, self._map_h)

        # rebuild map from seed
        seed = self._loader.seed()
        gen = MapGenerator(seed, self._map_w, self._map_h, 1)
        self._tiles, _ = gen.generate()

        # player id → colour index mapping built from header
        snap = self._header.get("state_snapshot", {})
        player_ids = list((snap.get("players") or {}).keys())
        # palette index = player order; colour comes from the same palette used for sprites
        self._player_palette_idx: dict[str, int] = {
            pid: i for i, pid in enumerate(player_ids)
        }
        self._player_colour: dict[str, tuple[int, int, int]] = {
            pid: sprite_loader.PALETTE_COLOURS[i % len(sprite_loader.PALETTE_COLOURS)]
            for i, pid in enumerate(player_ids)
        }

        # sprite texture cache — loaded once; requires OpenGL context (post super().__init__)
        self._texture_cache = sprite_loader.load_entity_textures()
        # nearest-neighbor atlas — filter is (re)applied in _rebuild_view_cache after
        # each sprite load, since atlas auto-resize creates a new GL texture that
        # otherwise reverts to GL_LINEAR.
        self._entity_atlas = DefaultTextureAtlas((2048, 2048))

        # animation time accumulator (real-time, independent of turn stepping)
        self._anim_time: float = 0.0

        # playback state
        self._current_turn = 0
        self._playing = False
        self._speed = 1.0  # turns per second

        # animation state machine
        self._anim_state: _AnimState = _AnimState.PAUSED
        self._transition: TurnTransition | None = None
        self._anim_progress: float = 0.0
        # visible set cached from last _rebuild_view_cache for use during animation
        self._visible_set_cache: set[HexCoord] | None = None
        # persistent SpriteList reused across animated frames — avoids per-frame VBO
        # allocation and eliminates ~9000 sprite objects of GC pressure per turn
        self._anim_sprite_list: arcade.SpriteList = arcade.SpriteList()

        # Camera2D: world (0,0) at screen centre; zoom handled by shader projection matrix
        # Using Camera2D avoids glCopyBufferSubData (broken on macOS Metal/MoltenVK)
        self._camera = Camera2D(position=(0.0, 0.0))

        # mouse debug state
        self._dbg_mouse_xy: tuple[float, float] = (0.0, 0.0)
        self._dbg_last_event: str = "none"
        self._dbg_frame_ms: float = 0.0
        self._dbg_last_draw_t: float = time.perf_counter()
        self._dbg_rebuild_ms: float = 0.0

        # pre-created Text objects for things that draw every frame — updating .text is
        # much cheaper than constructing a new arcade.Text (pyglet Label) each frame
        _DBG_LINE_H = 14
        _DBG_PW = 300
        _DBG_PX = WINDOW_W // 2 - _DBG_PW // 2
        self._dbg_texts: list[arcade.Text] = [
            arcade.Text(
                "", _DBG_PX + 4, 6 + 4 + i * _DBG_LINE_H, arcade.color.YELLOW, 10
            )
            for i in range(6)
        ]
        self._hud_text = arcade.Text("", 10, WINDOW_H - 25, arcade.color.WHITE, 12)

        # chat panel text cache — rebuilt per (last_msg_turn, channel, visible)
        self._chat_visible: bool = True
        self._chat_cache_key: tuple = (-1, "", True)
        self._chat_panel_bg: tuple[float, float, float, float] = (0, 0, 0, 0)  # xywh
        # pool of pre-allocated Text objects in a persistent batch — reused each rebuild
        self._chat_pool: list[arcade.Text] = []
        self._chat_pool_used: int = 0

        # chat: accumulate all messages per turn from the replay
        self._chat_by_turn: dict[int, list[dict]] = {}
        for record in self._loader._records:
            t = record.get("turn", 0)
            self._chat_by_turn[t] = record.get("chat", [])
        # sorted list of turns that actually have messages — used for O(log N) cache-key
        # lookup and O(messages) iteration instead of O(turns) range scan
        self._msg_turns_all: list[int] = sorted(
            t for t, msgs in self._chat_by_turn.items() if msgs
        )

        # channel list: ["All", "Global", "DM: <pid>", ...] — rebuilt per turn
        self._channels: list[str] = ["All", "Global"]
        self._chat_channel_name: str = "All"
        self._rebuild_channel_list()

        # terrain shape list — built once at HEX_SIZE, Camera2D handles pan/zoom
        self._terrain_sl = build_terrain_shape_list(self._tiles, HEX_SIZE)

        # fog base SL — all tiles in dark fog color; drawn first when fog is active,
        # then visible_terrain_sl draws real colors on top (O(visible) rebuild per turn
        # instead of O(non-visible), which is much cheaper when fog is dense)
        _r = HEX_SIZE - 1
        _fog_pts_offsets = [
            (
                _r * math.cos(math.pi / 180 * (60 * i - 30)),
                _r * math.sin(math.pi / 180 * (60 * i - 30)),
            )
            for i in range(6)
        ]
        fog_base_sl: arcade.shape_list.ShapeElementList = (
            arcade.shape_list.ShapeElementList()
        )
        for _coord in self._tiles:
            _wx, _wy = axial_to_pixel(_coord, size=HEX_SIZE)
            _pts = [(_wx + _dx, _wy + _dy) for _dx, _dy in _fog_pts_offsets]
            fog_base_sl.append(
                arcade.shape_list.create_polygon(_pts, (20, 20, 30, 240))
            )
        self._fog_base_sl = fog_base_sl

        # fog-of-war perspective: None = omniscient, str = player id
        self._fog_player: str | None = None
        self._fog_player_ids: list[str | None] = [None] + list(
            self._player_colour.keys()
        )

        # minimap geometry + grey base layer — positions baked in screen space (we never
        # touch center_x/center_y, which triggers glCopyBufferSubData on macOS Metal).
        # Set up before the first view-cache build because the per-turn team overlay
        # (built in _rebuild_view_cache) bakes minimap screen positions into its vertices.
        self._setup_minimap_geometry()

        # entity/fog/resource view cache — rebuilt in _step and on fog toggle, never in on_draw
        # sprite list for entity types that have sprites; fallback shape list for missing ones
        self._entity_sprite_list: arcade.SpriteList = arcade.SpriteList(
            atlas=self._entity_atlas
        )
        self._entity_sprite_meta: list[
            tuple[arcade.Sprite, int]
        ] = []  # (sprite, nframes)
        self._entity_sl: arcade.shape_list.ShapeElementList = (
            arcade.shape_list.ShapeElementList()
        )
        self._hp_bar_sl: arcade.shape_list.ShapeElementList = (
            arcade.shape_list.ShapeElementList()
        )
        self._entity_label_data: list[tuple[str, float, float]] = []  # (letter, wx, wy)
        # entity letters are drawn from a reusable Text pool inside a single pyglet
        # batch (one GL draw call), and only for entities inside the viewport — building
        # an arcade.Text per entity each turn cost ~2s on an 800-entity late-game map.
        self._entity_label_batch = pyglet.graphics.Batch()
        self._entity_label_pool: list[arcade.Text] = []
        self._label_active_count = 0  # how many pool entries currently show a letter
        self._label_cache_key: tuple | None = None
        self._visible_terrain_sl: arcade.shape_list.ShapeElementList = (
            arcade.shape_list.ShapeElementList()
        )
        # minimap team overlay: team-coloured hexes at every entity coord this turn
        self._minimap_entity_sl: arcade.shape_list.ShapeElementList = (
            arcade.shape_list.ShapeElementList()
        )
        # resource rows: list of (colour, alpha, label_text, is_alive) built per turn.
        # Text lives in a batch so the panel draws in one GL call regardless of player count
        # (colour, alpha, label, alive, swatch_x, row_y) per player row
        self._resource_rows: list[tuple[tuple, int, str, bool, float, float]] = []
        self._resource_panel_w: float = _CHAT_PANEL_W
        self._resource_panel_h: float = 0.0
        self._resource_text_objs: list[arcade.Text] = []
        self._resource_batch = pyglet.graphics.Batch()
        self._chat_batch = pyglet.graphics.Batch()
        self._rebuild_view_cache()

        # Hand GC scheduling to the viewer: collect once here (after the heavy __init__
        # allocations), then disable auto-collection so the cyclic GC never fires mid-frame.
        gc.collect()
        gc.disable()

        self._ready = True

    def _setup_minimap_geometry(self) -> None:
        """(Re)compute minimap screen geometry + rebuild the grey base shape list.

        Depends on self.width, so it is re-run on every window resize / fullscreen toggle.
        """
        mm_size = HEX_SIZE * MINIMAP_SCALE
        mm_px_off = math.sqrt(3) / 2 * mm_size
        mm_py_off = mm_size
        mm_px_w = math.sqrt(3) * (self._map_w + 0.5) * mm_size
        mm_px_h = (1.5 * self._map_h + 0.5) * mm_size
        self._mm_ox = self.width - mm_px_w - MINIMAP_PAD
        self._mm_oy = MINIMAP_PAD
        self._mm_px_w = mm_px_w
        self._mm_px_h = mm_px_h
        self._mm_px_off = mm_px_off
        self._mm_py_off = mm_py_off
        self._mm_size = mm_size
        # hex vertex offsets for a team-overlay marker (full mm_size so it fully covers
        # the grey base hex underneath and clusters read as solid team-coloured regions)
        self._mm_hex_offsets: list[tuple[float, float]] = [
            (
                mm_size * math.cos(math.pi / 180 * (60 * i - 30)),
                mm_size * math.sin(math.pi / 180 * (60 * i - 30)),
            )
            for i in range(6)
        ]
        # base layer: every tile is a uniform grey "empty" hex; tiles holding a
        # unit/building are overpainted with their owner's colour in the team overlay
        base_r = mm_size - 1
        base_offsets = [
            (
                base_r * math.cos(math.pi / 180 * (60 * i - 30)),
                base_r * math.sin(math.pi / 180 * (60 * i - 30)),
            )
            for i in range(6)
        ]
        minimap_sl: arcade.shape_list.ShapeElementList = (
            arcade.shape_list.ShapeElementList()
        )
        for _coord in self._tiles:
            _px, _py = axial_to_pixel(_coord, size=mm_size)
            _cx = self._mm_ox + mm_px_off + _px
            _cy = self._mm_oy + mm_py_off + _py
            _pts = [(_cx + dx, _cy + dy) for dx, dy in base_offsets]
            minimap_sl.append(
                arcade.shape_list.create_polygon(_pts, MINIMAP_EMPTY_COLOUR)
            )
        self._minimap_sl = minimap_sl

    # ── arcade callbacks ──────────────────────────────────────────────────────

    def on_update(self, delta_time: float) -> None:
        self._anim_time += delta_time
        frame_idx = int(self._anim_time * sprite_loader.ANIM_FPS)
        for sprite, nframes in self._entity_sprite_meta:
            sprite.set_texture(frame_idx % nframes)

        if self._anim_state is _AnimState.ANIMATING and self._transition is not None:
            self._anim_progress += delta_time * self._speed / self._transition.duration
            if self._anim_progress >= 1.0:
                self._anim_state = _AnimState.PAUSED
                self._transition = None
                self._anim_progress = 0.0
                if self._playing:
                    self._step(1)
        elif self._playing and self._anim_state is _AnimState.PAUSED:
            self._step(1)

    def on_draw(self) -> None:
        _mi = bisect.bisect_right(self._msg_turns_all, self._current_turn) - 1
        _last_msg_turn = self._msg_turns_all[_mi] if _mi >= 0 else -1
        _chat_key = (_last_msg_turn, self._chat_channel_name, self._chat_visible)
        if self._chat_cache_key != _chat_key:
            self._rebuild_chat_cache()
        if self._anim_state is _AnimState.PAUSED:
            self._sync_entity_labels()

        now = time.perf_counter()
        self._dbg_frame_ms = (now - self._dbg_last_draw_t) * 1000
        self._dbg_last_draw_t = now
        self.clear()
        self._draw_main()
        self._draw_minimap()
        self._draw_treaty_graph()
        self._draw_resource_panel()
        self._draw_hud()
        self._draw_chat_panel()
        self._draw_mouse_debug()

    def _sync_entity_labels(self) -> None:
        """Keep the entity-letter batch in sync with zoom/turn/camera.

        Only entities inside the current viewport get a letter, drawn from a reusable
        Text pool so we never construct hundreds of pyglet labels per frame. The whole
        batch renders in a single GL call (see _draw_main, inside the camera).
        """
        zoom = self._camera.zoom
        if zoom < LABEL_ZOOM_MIN:
            if self._label_active_count:
                for t in self._entity_label_pool[: self._label_active_count]:
                    t.text = ""
                self._label_active_count = 0
            self._label_cache_key = None
            return

        cam_x, cam_y = self._camera.position
        # quantise camera pos so small pans don't force a relayout every frame
        key = (self._current_turn, round(cam_x / 8), round(cam_y / 8), round(zoom, 1))
        if key == self._label_cache_key:
            return
        self._label_cache_key = key

        half_w = self.width / (2 * zoom) + HEX_SIZE * 2
        half_h = self.height / (2 * zoom) + HEX_SIZE * 2
        minx, maxx = cam_x - half_w, cam_x + half_w
        miny, maxy = cam_y - half_h, cam_y + half_h

        pool = self._entity_label_pool
        i = 0
        for letter, wx, wy in self._entity_label_data:
            if wx < minx or wx > maxx or wy < miny or wy > maxy:
                continue
            if i < len(pool):
                t = pool[i]
                t.text = letter
                t.x = wx
                t.y = wy
            else:
                pool.append(
                    arcade.Text(
                        letter,
                        wx,
                        wy,
                        arcade.color.WHITE,
                        9,
                        batch=self._entity_label_batch,
                    )
                )
            i += 1
        # blank any pool entries left over from a previously larger visible set
        for j in range(i, self._label_active_count):
            pool[j].text = ""
        self._label_active_count = i

    def _visible_set(self, entities_dict: dict) -> set[HexCoord] | None:
        """compute tiles visible to _fog_player from snapshot data; None = omniscient."""
        if self._fog_player is None:
            return None
        from engine.constants import BUILDING_STATS, UNIT_STATS

        visible: set[HexCoord] = set()
        for edict in entities_dict.values():
            if edict.get("owner_id") != self._fog_player:
                continue
            coord = HexCoord(edict.get("q", 0), edict.get("r", 0))
            etype = edict.get("type", "")
            vision = 0
            if etype in UNIT_STATS:
                vision = UNIT_STATS[etype].vision_range
            elif etype in BUILDING_STATS and edict.get("is_complete", True):
                vision = BUILDING_STATS[etype].vision_bonus
            for tile_coord in self._grid.disk(coord, vision):
                visible.add(tile_coord)
        return visible

    def _rebuild_view_cache(self) -> None:
        t0 = time.perf_counter()
        turn_data = self._loader.get_turn(self._current_turn)
        snap = turn_data.get("state_snapshot") or self._header.get("state_snapshot", {})
        entities_dict = snap.get("entities", {})
        visible = self._visible_set(entities_dict)
        self._visible_set_cache = visible

        hex_offsets = [
            (math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8))
            for i in range(8)
        ]

        entity_sl: arcade.shape_list.ShapeElementList = (
            arcade.shape_list.ShapeElementList()
        )
        entity_sprite_list = arcade.SpriteList(atlas=self._entity_atlas)
        entity_sprite_meta: list[tuple[arcade.Sprite, int]] = []
        label_data: list[tuple[str, float, float]] = []
        # batched geometry for minimap overlay and HP bars — assembled as triangle lists
        # and emitted in a single GL call instead of one create_* call per entity
        mm_pts: list[tuple[float, float]] = []
        mm_colors: list[tuple[int, int, int, int]] = []
        hp_pts: list[tuple[float, float]] = []
        hp_colors: list[tuple[int, int, int, int]] = []

        for edict in entities_dict.values():
            coord = HexCoord(edict.get("q", 0), edict.get("r", 0))
            if visible is not None and coord not in visible:
                continue
            wx, wy = axial_to_pixel(coord, size=HEX_SIZE)
            owner = edict.get("owner_id", "")
            colour = self._player_colour.get(owner, (200, 200, 200))
            etype = edict.get("type", "")
            palette_idx = self._player_palette_idx.get(owner, 0)
            frames = self._texture_cache.get((etype, palette_idx))

            if frames:
                scale = (HEX_SIZE * 1.3) / frames[0].width
                sprite = arcade.Sprite(frames[0], scale=scale)
                sprite.textures = frames
                sprite.center_x = wx
                sprite.center_y = wy
                entity_sprite_list.append(sprite)
                entity_sprite_meta.append((sprite, len(frames)))
            else:
                # no sprite available — fall back to circle + letter
                radius = HEX_SIZE * 0.4 if "Air" in etype else HEX_SIZE * 0.35
                pts = [(wx + radius * cx, wy + radius * cy) for cx, cy in hex_offsets]
                entity_sl.append(arcade.shape_list.create_polygon(pts, (*colour, 230)))
                label_data.append((etype[0] if etype else "?", wx - 4, wy - 5))

            # minimap overlay: fan-triangulate the 6-vertex hex into 4 triangles
            mm_px, mm_py = axial_to_pixel(coord, size=self._mm_size)
            mm_cx = self._mm_ox + self._mm_px_off + mm_px
            mm_cy = self._mm_oy + self._mm_py_off + mm_py
            vs = [(mm_cx + dx, mm_cy + dy) for dx, dy in self._mm_hex_offsets]
            mc: tuple[int, int, int, int] = (*colour, 255)  # type: ignore[assignment]
            for ti in range(1, 5):
                mm_pts += [vs[0], vs[ti], vs[ti + 1]]
                mm_colors += [mc, mc, mc]

            # HP bar (background + fill) — 2 rects = 4 triangles total
            max_hp = edict.get("max_hp", 0)
            if max_hp > 0:
                frac = max(0.0, edict.get("hp", 0) / max_hp)
                bar_w = HEX_SIZE * 0.9
                bar_h = 2.0
                bar_y = wy - HEX_SIZE * 0.55
                x1, x2 = wx - bar_w / 2, wx + bar_w / 2
                y1, y2 = bar_y - bar_h / 2, bar_y + bar_h / 2
                bg = (60, 60, 60, 200)
                for pt in [(x1, y1), (x2, y1), (x2, y2), (x1, y1), (x2, y2), (x1, y2)]:
                    hp_pts.append(pt)
                    hp_colors.append(bg)
                if frac > 0:
                    fill_colour: tuple[int, int, int, int] = (
                        (80, 200, 80, 200)
                        if frac > 0.5
                        else (200, 200, 50, 200)
                        if frac > 0.25
                        else (200, 60, 60, 200)
                    )
                    fx2 = x1 + bar_w * frac
                    for pt in [
                        (x1, y1),
                        (fx2, y1),
                        (fx2, y2),
                        (x1, y1),
                        (fx2, y2),
                        (x1, y2),
                    ]:
                        hp_pts.append(pt)
                        hp_colors.append(fill_colour)

        # build minimap entity overlay in one GL call
        if mm_pts:
            mm_entity_sl = arcade.shape_list.create_triangles_filled_with_colors(
                mm_pts, mm_colors
            )
        else:
            mm_entity_sl: arcade.shape_list.ShapeElementList = (  # type: ignore[no-redef]
                arcade.shape_list.ShapeElementList()
            )

        # build HP bar geometry in one GL call
        if hp_pts:
            hp_bar_sl = arcade.shape_list.create_triangles_filled_with_colors(
                hp_pts, hp_colors
            )
        else:
            hp_bar_sl: arcade.shape_list.ShapeElementList = (  # type: ignore[no-redef]
                arcade.shape_list.ShapeElementList()
            )

        # visible terrain SL: only the visible subset, drawn over fog_base_sl.
        # O(visible) rebuild — much cheaper than O(non-visible) when fog is dense.
        if visible is not None:
            visible_tiles = {c: self._tiles[c] for c in visible if c in self._tiles}
            visible_terrain_sl = build_terrain_shape_list(visible_tiles, HEX_SIZE)
        else:
            visible_terrain_sl = arcade.shape_list.ShapeElementList()

        self._entity_sprite_list = entity_sprite_list
        self._entity_sprite_meta = entity_sprite_meta
        self._entity_sl = entity_sl
        self._entity_label_data = label_data
        self._hp_bar_sl = hp_bar_sl
        self._label_cache_key = None  # entity positions changed → force label re-sync
        self._visible_terrain_sl = visible_terrain_sl
        self._minimap_entity_sl = mm_entity_sl
        # Re-apply nearest-neighbor filter after sprite additions — atlas auto-resize
        # creates a new GL texture that resets to the default GL_LINEAR filter.
        self._entity_atlas.texture.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
        self._dbg_rebuild_ms = (time.perf_counter() - t0) * 1000

        # resource panel rows — Text lives in a fresh batch so the panel draws in one call.
        # Laid out in TWO columns so the leaderboard's vertical height is ~halved (it then
        # sits low enough to give the chat panel below it much more room).
        players_dict = snap.get("players", {})
        items = list(players_dict.items())
        line_h = 16
        pad = 10
        n_cols = 2
        panel_w = (
            _CHAT_PANEL_W  # match the chat panel width so the two stack flush-right
        )
        col_w = panel_w / n_cols
        px = self.width - panel_w - pad
        top_y = self.height - pad - 30
        rows_per_col = math.ceil(len(items) / n_cols) if items else 0
        panel_h = rows_per_col * line_h + 6 if rows_per_col else 0
        # each row stores its own swatch_x / row_y so the draw pass needs no recompute
        rows: list[tuple[tuple, int, str, bool, float, float]] = []
        resource_text_objs: list[arcade.Text] = []
        resource_batch = pyglet.graphics.Batch()
        for i, (pid, pdata) in enumerate(items):
            name = pdata.get("name", pid)
            gold = pdata.get("resources", {}).get("gold", 0)
            alive = pdata.get("alive", True)
            colour = self._player_colour.get(pid, (200, 200, 200))
            alpha = 255 if alive else 100
            label = f"{name}: {gold}g" + ("" if alive else "  [dead]")
            text_colour = (220, 220, 220, 255) if alive else (110, 110, 110, 255)
            col = i // rows_per_col if rows_per_col else 0  # column-major fill
            row = i % rows_per_col if rows_per_col else 0
            col_x = px + col * col_w
            row_y = top_y - (row + 1) * line_h + 4
            rows.append((colour, alpha, label, alive, col_x, row_y))
            resource_text_objs.append(
                arcade.Text(
                    label, col_x + 20, row_y, text_colour, 10, batch=resource_batch
                )
            )
        self._resource_rows = rows
        self._resource_text_objs = resource_text_objs
        self._resource_batch = resource_batch
        self._resource_panel_w = panel_w
        self._resource_panel_h = panel_h

    def _rebuild_channel_list(self) -> None:
        """Rebuild channels from DM partners seen up to current turn.
        Pins selection by name so list growth doesn't shift the active tab."""
        partners: set[str] = set()
        for t in range(self._current_turn + 1):
            for msg in self._chat_by_turn.get(t, []):
                if msg.get("recipient_id") is None:
                    continue
                sender = msg.get("sender_id", "")
                recipient = msg.get("recipient_id", "")
                if sender and sender != "__system__":
                    partners.add(sender)
                if recipient and recipient != "__system__":
                    partners.add(recipient)
        self._channels = ["All", "Global"] + [f"DM: {pid}" for pid in sorted(partners)]
        if self._chat_channel_name not in self._channels:
            self._chat_channel_name = "All"

    def _draw_main(self) -> None:
        frame_idx = int(self._anim_time * sprite_loader.ANIM_FPS)
        anim_frames = {etype: frame_idx for etype in sprite_loader.ENTITY_SPRITE_MAP}
        with self._camera.activate():
            if self._fog_player is None:
                self._terrain_sl.draw()
            else:
                self._fog_base_sl.draw()
                self._visible_terrain_sl.draw()

            if (
                self._anim_state is _AnimState.ANIMATING
                and self._transition is not None
            ):
                self._draw_attack_effects(self._anim_progress)
                draw_animated_entities(
                    self._transition,
                    self._anim_progress,
                    self._visible_set_cache,
                    self._player_colour,
                    self._player_palette_idx,
                    self._texture_cache,
                    anim_frames,
                    0.0,  # world coords: Camera2D handles pan/zoom; disable cull
                    0.0,
                    HEX_SIZE,
                    1 << 30,
                    1 << 30,
                    self._anim_sprite_list,
                )
            else:
                self._entity_sprite_list.draw(pixelated=True)
                self._entity_sl.draw()
                self._hp_bar_sl.draw()
                self._entity_label_batch.draw()

    def _draw_attack_effects(self, progress: float) -> None:
        if self._transition is None:
            return
        visible = self._visible_set_cache
        flashes: list[tuple[int, int, tuple[int, int, int], int]] = []
        beams: list[tuple[float, float, float, float, tuple[int, int, int], int]] = []
        flash_t = max(0.0, (progress - 0.2) / 0.5)
        for evt in self._transition.attacks:
            attacker_vis = visible is None or evt.attacker_coord in visible
            target_vis = visible is None or evt.target_coord in visible
            if attacker_vis and target_vis:
                ax, ay = axial_to_pixel(evt.attacker_coord, size=HEX_SIZE)
                tx, ty = axial_to_pixel(evt.target_coord, size=HEX_SIZE)
                thresh = HEX_SIZE * 6
                if (tx - ax) ** 2 + (ty - ay) ** 2 <= thresh * thresh:
                    beam_alpha = int(sin(pi * min(1.0, progress / 0.6)) * 255)
                    beam_color = (255, 255, 200) if evt.succeeded else (200, 60, 60)
                    beams.append((ax, ay, tx, ty, beam_color, beam_alpha))
            if target_vis:
                flash_alpha = int(sin(pi * min(1.0, flash_t)) * 200)
                flash_color = (255, 140, 0) if evt.succeeded else (180, 40, 40)
                flashes.append(
                    (evt.target_coord.q, evt.target_coord.r, flash_color, flash_alpha)
                )
                if evt.is_splash:
                    splash_alpha = flash_alpha // 2
                    for nb in self._grid.neighbors(evt.target_coord):
                        if visible is None or nb in visible:
                            flashes.append((nb.q, nb.r, flash_color, splash_alpha))
        for coord, is_lethal in self._transition.damage_flashes:
            if visible is None or coord in visible:
                flash_alpha = int(sin(pi * min(1.0, flash_t)) * 200)
                color: tuple[int, int, int] = (
                    (200, 60, 60) if is_lethal else (255, 140, 0)
                )
                flashes.append((coord.q, coord.r, color, flash_alpha))
        draw_tile_flashes_batched(flashes, 0.0, 0.0, HEX_SIZE)
        for ax, ay, tx, ty, beam_color, beam_alpha in beams:
            draw_beam(ax, ay, tx, ty, beam_color, beam_alpha, 2.0)

    def _draw_minimap(self) -> None:
        """draw the whole-world minimap in the bottom-right corner"""
        ox = self._mm_ox
        oy = self._mm_oy
        px_w = self._mm_px_w
        px_h = self._mm_px_h

        arcade.draw_rect_filled(
            arcade.XYWH(ox + px_w / 2, oy + px_h / 2, px_w + 4, px_h + 4),
            (20, 20, 20, 180),
        )

        # single batched draw — positions baked in at init, no center_x/y setter
        self._minimap_sl.draw()
        # team presence: colour each tile that holds a unit/building this turn
        self._minimap_entity_sl.draw()

        # viewport indicator
        zoom = self._camera.zoom
        cam_x, cam_y = self._camera.position
        scale_ratio = self._mm_size / HEX_SIZE
        vp_w = self.width / zoom * scale_ratio
        vp_h = self.height / zoom * scale_ratio
        vp_cx = ox + self._mm_px_off + cam_x * scale_ratio
        vp_cy = oy + self._mm_py_off + cam_y * scale_ratio
        arcade.draw_rect_outline(
            arcade.XYWH(vp_cx, vp_cy, vp_w, vp_h), (255, 255, 255, 200), 1
        )

    def _draw_treaty_graph(self) -> None:
        """Right-side node graph of treaty status: every team is a node on a circle,
        and a line connects two teams bound by an active peace treaty this turn. Per
        the treaty cutoff, no lines are drawn from turn TREATY_CUTOFF_TURN onward."""
        ids = list(self._player_colour.keys())
        if not ids:
            return

        size = _TREATY_PANEL_SIZE
        x0 = self.width - size - MINIMAP_PAD
        y0 = self._mm_oy + self._mm_px_h + 8  # just above the minimap
        cx = x0 + size / 2
        cy = y0 + size / 2 - 6  # nudge down to leave room for the title
        radius = size * 0.36

        # panel background
        arcade.draw_rect_filled(
            arcade.XYWH(x0 + size / 2, y0 + size / 2, size, size), (20, 20, 20, 180)
        )

        # title (and a voided marker once the cutoff has passed)
        voided = self._current_turn >= TREATY_CUTOFF_TURN
        title = "Treaties — VOIDED" if voided else "Treaties"
        arcade.draw_text(title, x0 + 8, y0 + size - 16, (160, 210, 255, 255), 10)

        # node positions: evenly around the circle, starting at top, going clockwise
        n = len(ids)
        pos: dict[str, tuple[float, float]] = {}
        for i, pid in enumerate(ids):
            ang = math.pi / 2 - 2 * math.pi * i / n
            pos[pid] = (cx + radius * math.cos(ang), cy + radius * math.sin(ang))

        # alive status from the current snapshot → dim dead teams' nodes,
        # and skip treaty edges where either party has been eliminated
        snap = self._loader.get_turn(self._current_turn).get("state_snapshot", {})
        players = snap.get("players", {})

        # treaty edges (drawn under the nodes); the helper returns [] at/after the
        # cutoff so no line is ever drawn from turn TREATY_CUTOFF_TURN onward
        for t in treaty_edges_for_display(self._loader, self._current_turn):
            ka, kb = t.get("a"), t.get("b")
            a = pos.get(ka) if isinstance(ka, str) else None
            b = pos.get(kb) if isinstance(kb, str) else None
            if a is None or b is None:
                continue
            if not players.get(ka, {}).get("alive", True) or not players.get(
                kb, {}
            ).get("alive", True):
                continue
            # a treaty mid-break (breaking_in_turns set) is amber, else green
            colour = (
                (235, 170, 60, 220)
                if t.get("breaking_in_turns") is not None
                else (90, 220, 120, 200)
            )
            arcade.draw_line(a[0], a[1], b[0], b[1], colour, 1.5)

        for pid, (nx, ny) in pos.items():
            colour = self._player_colour.get(pid, (200, 200, 200))
            alive = players.get(pid, {}).get("alive", True)
            alpha = 255 if alive else 90
            arcade.draw_circle_filled(nx, ny, 5, (*colour, alpha))
            if not alive:
                arcade.draw_circle_outline(nx, ny, 5, (90, 90, 90, 160), 1)

    def _draw_hud(self) -> None:
        total = self._loader.total_turns - 1
        fog_label = "all" if self._fog_player is None else self._fog_player
        self._hud_text.text = (
            f"turn {self._current_turn}/{total}  "
            f"{'▶' if self._playing else '⏸'}  "
            f"speed {self._speed:.1f}x  "
            f"view:{fog_label}  "
            f"[space]=play/pause  [←→]=step  [+/-]=speed  scroll=zoom  "
            f"[F]=fog  [Tab]=channel  [C]=chat  [F11]=fullscreen"
        )
        self._hud_text.draw()

    def _draw_resource_panel(self) -> None:
        if not self._resource_rows:
            return
        swatch = 8
        pad = 10
        panel_w = getattr(self, "_resource_panel_w", _CHAT_PANEL_W)
        panel_h = getattr(self, "_resource_panel_h", 0)
        px = self.width - panel_w - pad
        top_y = self.height - pad - 30
        arcade.draw_rect_filled(
            arcade.XYWH(px + panel_w / 2, top_y - panel_h / 2, panel_w, panel_h),
            (10, 10, 20, 170),
        )
        # swatch per row at its stored (column) position
        for colour, alpha, _label, _alive, col_x, row_y in self._resource_rows:
            arcade.draw_rect_filled(
                arcade.XYWH(col_x + 8, row_y + swatch // 2, swatch, swatch),
                (*colour, alpha),
            )
        # all row labels render in a single batched draw
        self._resource_batch.draw()

    def _messages_for_channel(self) -> list[dict]:
        """collect all chat messages up to the current turn for the active channel"""
        channel = self._chat_channel_name
        # System events (treaty formed/break/expired) are DM'd to BOTH parties, so each
        # shows up twice with identical text. In the aggregate "All" view that reads as
        # a duplicate, so collapse exact-duplicate system messages within a turn. The
        # text already names both parties, so two distinct events never share text;
        # per-DM channels keep each party's own copy.
        dedup_system = channel == "All"
        accumulated: list[dict] = []
        n = bisect.bisect_right(self._msg_turns_all, self._current_turn)
        for t in self._msg_turns_all[:n]:
            msgs = self._chat_by_turn[t]
            if dedup_system:
                seen_sys: set[str] = set()
                for m in msgs:
                    if m.get("sender_id") == "__system__":
                        key = m.get("text", "")
                        if key in seen_sys:
                            continue
                        seen_sys.add(key)
                    accumulated.append(m)
            else:
                accumulated.extend(msgs)

        if channel == "Global":
            return [m for m in accumulated if m.get("recipient_id") is None]
        if channel.startswith("DM: "):
            partner = channel[4:]
            return [
                m
                for m in accumulated
                if m.get("recipient_id") is not None
                and (m.get("sender_id") == partner or m.get("recipient_id") == partner)
            ]
        return accumulated  # "All" or fallback

    def _rebuild_chat_cache(self) -> None:
        # Cache key uses last_msg_turn (not current_turn) so turns without new messages
        # are skipped without a rebuild.
        _mi = bisect.bisect_right(self._msg_turns_all, self._current_turn) - 1
        _last_msg_turn = self._msg_turns_all[_mi] if _mi >= 0 else -1
        self._chat_cache_key = (
            _last_msg_turn,
            self._chat_channel_name,
            self._chat_visible,
        )
        self._chat_panel_bg = (0.0, 0.0, 0.0, 0.0)

        if not self._chat_visible:
            for j in range(self._chat_pool_used):
                self._chat_pool[j].text = ""
            self._chat_pool_used = 0
            return

        # Right-side placement: a column flush with the right edge (same margin as
        # the resource panel + minimap), occupying the vertical gap ABOVE the
        # minimap and BELOW the gold/resource panel. Computed from their live
        # geometry so it tracks window resize, player count, and map size.
        pad = 10
        PX = self.width - _CHAT_PANEL_W - pad
        minimap_top = getattr(self, "_mm_oy", pad) + getattr(self, "_mm_px_h", 0.0)
        # the treaty graph sits just above the minimap; chat starts above the graph
        graph_top = minimap_top + 8 + _TREATY_PANEL_SIZE
        # bottom of the (2-column) leaderboard — read its actual pixel height
        resource_bottom = (self.height - pad - 30) - getattr(
            self, "_resource_panel_h", 0.0
        )
        gap_margin = 8
        PY = graph_top + gap_margin  # panel bottom sits just above the treaty graph
        band_top = resource_bottom - gap_margin  # top sits just below the leaderboard
        INNER = _CHAT_PANEL_W - 16  # usable inner width in px

        # Compute max_rows BEFORE fetching messages to limit word-wrap to visible rows;
        # otherwise a long game word-wraps every message on every rebuild (O(N_messages)).
        header_h = _CHAT_LINE_H + 6
        avail_h = band_top - PY
        fit_rows = int((avail_h - header_h - 8) / _CHAT_LINE_H)
        max_rows = max(1, min(_CHAT_MAX_ROWS, fit_rows))

        channel = self._chat_channel_name
        # Take only the last max_rows*4 messages — generous buffer for worst-case wrapping.
        all_messages = self._messages_for_channel()
        messages = all_messages[max(0, len(all_messages) - max_rows * 4) :]

        all_rows: list[list[tuple[str, float, tuple[int, int, int, int]]]] = []

        for msg in messages:
            sender = msg.get("sender_id", "?")
            recipient = msg.get("recipient_id")
            body = msg.get("text", "")
            turn = msg.get("turn", "?")

            prefix_segs, body_x = _chat_prefix_segs(
                turn, sender, recipient, self._player_colour
            )

            if sender == "__system__":
                body_col: tuple[int, int, int, int] = (220, 200, 80, 255)
            elif recipient:
                body_col = (160, 240, 160, 255)
            else:
                body_col = (240, 240, 240, 255)

            avail_chars = max(8, int((INNER - (body_x - 8)) / _CHAT_CHAR_W))
            # Word-wrap for layout only. Messages are already length-capped when
            # written to the replay (recorder._MAX_CHAT_CHARS), so no extra
            # per-message line truncation is needed here; _CHAT_MAX_ROWS still
            # bounds the panel as a whole.
            wrapped = _chat_wrap(body, avail_chars)

            row0: list[tuple[str, float, tuple[int, int, int, int]]] = list(prefix_segs)
            if wrapped:
                row0.append((wrapped[0], body_x, body_col))
            all_rows.append(row0)

            for cont in wrapped[1:]:
                all_rows.append([(cont, body_x, body_col)])

        visible_rows = all_rows[-max_rows:]

        panel_h = max(header_h + 8, avail_h)
        self._chat_panel_bg = (
            PX + _CHAT_PANEL_W / 2,
            PY + panel_h / 2,
            float(_CHAT_PANEL_W),
            float(panel_h),
        )

        # Update pool in-place: grow lazily on first use, reuse on subsequent rebuilds.
        pool = self._chat_pool
        slot = 0

        def _put(
            text: str, x: float, y: float, color: tuple[int, int, int, int]
        ) -> None:
            nonlocal slot
            if slot < len(pool):
                t = pool[slot]
                t.text = text
                t.x = x
                t.y = y
                t.color = color
            else:
                pool.append(
                    arcade.Text(text, x, y, color, _CHAT_FONT, batch=self._chat_batch)
                )
            slot += 1

        header_y = PY + panel_h - header_h  # header at the top of the fixed panel
        _put(
            f"Chat [{channel}]  Tab=ch  C=hide", PX + 8, header_y, (160, 210, 255, 255)
        )

        n_rows = len(visible_rows)
        for row_i, row_segs in enumerate(visible_rows):
            y = PY + 4 + (n_rows - 1 - row_i) * _CHAT_LINE_H
            for seg_text, rel_x, color in row_segs:
                _put(seg_text, PX + rel_x, y, color)

        # Blank any pool entries left over from a previously larger panel
        for j in range(slot, self._chat_pool_used):
            pool[j].text = ""
        self._chat_pool_used = slot

    def _draw_chat_panel(self) -> None:
        if not self._chat_visible:
            return
        bx, by, bw, bh = self._chat_panel_bg
        arcade.draw_rect_filled(arcade.XYWH(bx, by, bw, bh), (10, 10, 20, 170))
        # header + all rows render in a single batched draw
        self._chat_batch.draw()

    # ── input ─────────────────────────────────────────────────────────────────

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == arcade.key.SPACE:
            self._playing = not self._playing
            if not self._playing:
                gc.collect()
        elif symbol == arcade.key.RIGHT:
            self._step(1)
        elif symbol == arcade.key.LEFT:
            self._step(-1)
        elif symbol in (arcade.key.PLUS, arcade.key.EQUAL, arcade.key.NUM_ADD):
            self._speed = min(10.0, self._speed * 2)
        elif symbol in (arcade.key.MINUS, arcade.key.NUM_SUBTRACT):
            self._speed = max(0.1, self._speed / 2)
        elif symbol == arcade.key.HOME:
            self._anim_state = _AnimState.PAUSED
            self._transition = None
            self._current_turn = 0
            self._rebuild_view_cache()
            self._rebuild_channel_list()
        elif symbol == arcade.key.END:
            self._anim_state = _AnimState.PAUSED
            self._transition = None
            self._current_turn = self._loader.total_turns - 1
            self._rebuild_view_cache()
            self._rebuild_channel_list()
        elif symbol == arcade.key.C:
            self._chat_visible = not self._chat_visible
        elif symbol == arcade.key.TAB:
            try:
                idx = self._channels.index(self._chat_channel_name)
            except ValueError:
                idx = 0
            if modifiers & arcade.key.MOD_SHIFT:
                idx = (idx - 1) % len(self._channels)
            else:
                idx = (idx + 1) % len(self._channels)
            self._chat_channel_name = self._channels[idx]
        elif symbol == arcade.key.F:
            idx = self._fog_player_ids.index(self._fog_player)
            self._fog_player = self._fog_player_ids[
                (idx + 1) % len(self._fog_player_ids)
            ]
            self._rebuild_view_cache()
        elif symbol == arcade.key.F11:
            self.set_fullscreen(not self.fullscreen)

    @typing.override
    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        if not getattr(self, "_ready", False):
            return
        # keep the world camera matched to the new framebuffer, then rebuild every
        # layout-dependent cache (minimap, panels) against the new self.width/height
        self._camera.match_window()
        self._setup_minimap_geometry()
        self._reposition_hud_debug()
        self._rebuild_view_cache()
        self._rebuild_chat_cache()
        self._label_cache_key = None

    def _reposition_hud_debug(self) -> None:
        """Move the size-anchored persistent Text objects after a resize."""
        self._hud_text.x = 10
        self._hud_text.y = self.height - 25
        dbg_px = self.width // 2 - 150
        for t in self._dbg_texts:
            t.x = dbg_px + 4

    @typing.override
    def on_mouse_motion(self, x: float, y: float, dx: float, dy: float) -> None:
        self._dbg_mouse_xy = (x, y)
        self._dbg_last_event = "motion"

    @typing.override
    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        self._dbg_mouse_xy = (x, y)
        self._dbg_last_event = "press"

    @typing.override
    def on_mouse_release(self, x: float, y: float, button: int, modifiers: int) -> None:
        self._dbg_mouse_xy = (x, y)
        self._dbg_last_event = "release"

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
        factor = 1.1 if scroll_y > 0 else 0.9
        self._camera.zoom = max(0.25, min(5.0, self._camera.zoom * factor))

    def on_mouse_drag(
        self, x: float, y: float, dx: float, dy: float, buttons: int, modifiers: int
    ) -> None:
        self._dbg_mouse_xy = (x, y)
        self._dbg_last_event = "drag"
        if buttons & arcade.MOUSE_BUTTON_LEFT:
            zoom = self._camera.zoom
            cx, cy = self._camera.position
            self._camera.position = (cx - dx / zoom, cy - dy / zoom)

    def _draw_mouse_debug(self) -> None:
        lines = [
            f"last event : {self._dbg_last_event}",
            f"mouse xy   : {self._dbg_mouse_xy[0]:.0f}, {self._dbg_mouse_xy[1]:.0f}",
            f"cam pos    : {self._camera.position[0]:.1f}, {self._camera.position[1]:.1f}",
            f"cam zoom   : {self._camera.zoom:.2f}",
            f"frame ms   : {self._dbg_frame_ms:.1f}",
            f"rebuild ms : {self._dbg_rebuild_ms:.1f}",
        ]
        line_h = 14
        panel_w = 300
        panel_h = len(lines) * line_h + 8
        px = self.width // 2 - panel_w // 2
        py = 6
        arcade.draw_rect_filled(
            arcade.XYWH(px + panel_w / 2, py + panel_h / 2, panel_w, panel_h),
            (0, 0, 0, 200),
        )
        for text_obj, line in zip(self._dbg_texts, lines):
            text_obj.text = line
            text_obj.draw()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _step(self, delta: int) -> None:
        if delta > 0:
            if self._anim_state is _AnimState.ANIMATING:
                # skip current animation — jump to result state already cached
                self._anim_state = _AnimState.PAUSED
                self._transition = None
                return
            next_turn = self._current_turn + 1
            if next_turn >= self._loader.total_turns:
                self._playing = False
                gc.collect()
                return
            prev_record = self._loader.get_turn(self._current_turn)
            next_record = self._loader.get_turn(next_turn)
            self._current_turn = next_turn
            self._rebuild_view_cache()
            self._rebuild_channel_list()
            try:
                self._transition = build_transition(
                    prev_record, next_record, self._player_palette_idx
                )
            except Exception:
                self._transition = None
            self._anim_progress = 0.0
            self._anim_state = _AnimState.ANIMATING
        else:
            if self._anim_state is _AnimState.ANIMATING:
                # cancel forward animation — stay on current result turn
                self._anim_state = _AnimState.PAUSED
                self._transition = None
                return
            new_turn = max(0, self._current_turn + delta)
            if new_turn == self._current_turn:
                return
            self._current_turn = new_turn
            self._rebuild_view_cache()
            self._rebuild_channel_list()


def launch_viewer(replay_path: str) -> None:
    ReplayViewer(replay_path)
    arcade.run()
