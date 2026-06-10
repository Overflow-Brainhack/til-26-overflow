"""procedural seeded map generation with equidistant player starting positions"""

from __future__ import annotations

import math
import random

try:
    from noise import pnoise2
except ImportError:
    pnoise2 = None  # type: ignore[assignment]

from engine.constants import (
    MAP_HEIGHT_DIFFICULT_THRESHOLD,
    MAP_HEIGHT_ELEVATED_THRESHOLD,
    MAP_MOISTURE_THRESHOLD,
    MAP_RESOURCE_THRESHOLD,
    NOISE_OCTAVES,
    NOISE_SCALE,
)
from engine.hex_grid import HexCoord, HexGrid, axial_to_pixel
from engine.terrain import TerrainType, Tile


class MapGenerator:
    def __init__(
        self,
        seed: int,
        width: int,
        height: int,
        num_players: int,
    ) -> None:
        self.seed = seed
        self.width = width
        self.height = height
        self.num_players = num_players
        self._rng = random.Random(seed)
        self._grid = HexGrid(width, height)

    def generate(self) -> tuple[dict[HexCoord, Tile], list[HexCoord]]:
        """generate the tile map and player starting positions.

        returns (tiles, start_coords) where start_coords has length num_players
        and each start is guaranteed to be on a plain NORMAL tile (no terrain
        advantage — fair, neutral spawns for every player).
        """
        tiles = self._generate_terrain()
        start_coords = self._place_starts(tiles)
        return tiles, start_coords

    # ── terrain generation ────────────────────────────────────────────────────

    def _generate_terrain(self) -> dict[HexCoord, Tile]:
        """Two-channel heightmap terrain generation.

        Channel 1 (height): determines primary structure — mountain peaks,
        foothills, and plains bands.  Adjacent tiles have correlated height
        values so terrain clusters naturally instead of checkerboarding.

        Channel 2 (resource): sparse overlay on plains tiles only → RICH_RESOURCE.
        Channel 3 (moisture): overlay on remaining plains → CONCEALMENT (dense forest).
        """
        tiles: dict[HexCoord, Tile] = {}

        offset_height = (self._rng.random() * 1000, self._rng.random() * 1000)
        offset_resource = (self._rng.random() * 1000, self._rng.random() * 1000)
        offset_moisture = (self._rng.random() * 1000, self._rng.random() * 1000)

        for coord in self._grid.all_coords():
            px, py = axial_to_pixel(coord, size=1.0)
            nx = px / self.width * NOISE_SCALE
            ny = py / self.height * NOISE_SCALE
            terrain = self._pick_terrain(
                nx, ny, offset_height, offset_resource, offset_moisture
            )
            tiles[coord] = Tile(terrain=terrain)

        return tiles

    def _pick_terrain(
        self,
        nx: float,
        ny: float,
        off_h: tuple[float, float],
        off_r: tuple[float, float],
        off_m: tuple[float, float],
    ) -> TerrainType:
        """Assign terrain via correlated height bands + secondary overlays."""

        # pnoise2 amplitude is ~0.5 for octaves=4; normalise to [0,1]
        _PNOISE_AMP = 0.5

        def noise(ox: float, oy: float) -> float:
            if pnoise2 is not None:
                raw = pnoise2(nx + ox, ny + oy, octaves=NOISE_OCTAVES)
                return max(0.0, min(1.0, (raw / _PNOISE_AMP + 1.0) / 2.0))
            import hashlib

            key = f"{self.seed}:{round(nx + ox, 2)}:{round(ny + oy, 2)}"
            return (int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000) / 10000

        height = noise(*off_h)

        # ── primary structure (height bands) ──────────────────────────────────
        # High elevation → mountain peaks with attack bonus
        if height > MAP_HEIGHT_ELEVATED_THRESHOLD:
            return TerrainType.ELEVATED

        # Mid-high elevation → foothills / rough ground (costs 2 to enter)
        if height > MAP_HEIGHT_DIFFICULT_THRESHOLD:
            return TerrainType.DIFFICULT

        # ── plains zone: secondary overlays ───────────────────────────────────
        # Rich resource: sparse fertile spots (gold mines, farmland) in open ground
        if noise(*off_r) > MAP_RESOURCE_THRESHOLD:
            return TerrainType.RICH_RESOURCE

        # Concealment: dense forest / fog, only grows in accessible lowlands
        if noise(*off_m) > MAP_MOISTURE_THRESHOLD:
            return TerrainType.CONCEALMENT

        return TerrainType.NORMAL

    # ── player start placement ────────────────────────────────────────────────

    def _place_starts(self, tiles: dict[HexCoord, Tile]) -> list[HexCoord]:
        """place num_players starting positions evenly distributed on the torus.

        Uses repulsion relaxation (see `_lattice_positions`) to get equal spacing
        between players, then snaps each spawn to the nearest unused NORMAL tile.
        This preserves the spacing while ensuring the base itself never sits on
        elevated, difficult, concealment, or rich-resource terrain.
        """
        if self.num_players == 1:
            return [self._nearest_normal_free(HexCoord(0, 0), tiles, set())]

        starts: list[HexCoord] = []
        used: set[HexCoord] = set()

        for coord in self._lattice_positions():
            coord = self._nearest_normal_free(coord, tiles, used)
            starts.append(coord)
            used.add(coord)

        return starts

    def _nearest_normal_free(
        self,
        coord: HexCoord,
        tiles: dict[HexCoord, Tile],
        used: set[HexCoord],
    ) -> HexCoord:
        """nearest unused NORMAL tile around a preferred coordinate."""
        for candidate in self._grid.disk(coord, 12):
            if candidate in used:
                continue
            if tiles.get(candidate, Tile()).terrain == TerrainType.NORMAL:
                return candidate
        for candidate in self._grid.all_coords():
            if candidate in used:
                continue
            if tiles.get(candidate, Tile()).terrain == TerrainType.NORMAL:
                return candidate
        return coord

    @staticmethod
    def _best_grid(n: int, aspect: float) -> tuple[int, int]:
        """Pick (cols, rows) for n players on a map of the given aspect ratio.

        Prefers layouts that:
          1. waste few cells (cols*rows close to n — perfectly even when n factors)
          2. match the map aspect ratio (cols/rows ≈ width/height)

        Examples (landscape aspect ≈ 1.78):
          n=20 → 5×4 (exact, no partial row)
          n=12 → 4×3 (exact)
          n=7  → 4×2 (rows of 4 and 3)
          n=4  → 2×2
        """
        if n <= 1:
            return 1, 1
        best: tuple[float, int, int] | None = None
        for rows in range(1, n + 1):
            cols = math.ceil(n / rows)
            layout_aspect = cols / rows
            # symmetric (log) distance so 2x too-wide ~ 2x too-tall
            aspect_err = abs(math.log(layout_aspect / aspect))
            waste = cols * rows - n  # empty cells; 0 == perfectly even
            score = aspect_err + 0.5 * waste
            if best is None or score < best[0]:
                best = (score, cols, rows)
        assert best is not None
        return best[1], best[2]

    # number of repulsion-relaxation passes for spawn spreading
    _RELAX_ITERS = 120
    _RELAX_STEP = 0.0015

    def _lattice_positions(self) -> list[HexCoord]:
        """Evenly distributed spawn coords via toroidal repulsion relaxation.

        Seeds a grid layout (best rows×cols factorisation for the player count
        and map aspect), then runs a deterministic repulsion relaxation on the
        flat torus: every point pushes every other point away (inverse-square,
        minimum-image wrap), so the points settle into a configuration where
        each player's nearest-neighbour distance is essentially equal. This
        minimises the difference in spacing between players — the fairness goal.

        Fully deterministic (seeded only by player count + map size, no RNG).
        Coords are returned canonical (wrapped).
        """
        n = self.num_players
        w, h = self.width, self.height
        aspect = w / h

        # ── seed from the grid layout (continuous [0,1) torus coords) ──────────
        cols, rows = self._best_grid(n, aspect)
        per_row = [cols] * rows
        per_row[-1] = n - cols * (rows - 1)
        pts: list[list[float]] = []
        for row in range(rows):
            count = per_row[row]
            fy = (row + 0.5) / rows
            for j in range(count):
                fx = (j + 0.5) / count
                pts.append([fx, fy])

        # ── repulsion relaxation (x scaled by aspect → even spacing in real space) ──
        for _ in range(self._RELAX_ITERS):
            forces = [[0.0, 0.0] for _ in range(n)]
            for i in range(n):
                for k in range(n):
                    if i == k:
                        continue
                    dx = pts[i][0] - pts[k][0]
                    dy = pts[i][1] - pts[k][1]
                    dx -= round(dx)  # minimum-image wrap on torus
                    dy -= round(dy)
                    rx, ry = dx * aspect, dy
                    d2 = rx * rx + ry * ry + 1e-9
                    dist = math.sqrt(d2)
                    inv = 1.0 / d2  # inverse-square repulsion
                    forces[i][0] += (rx / dist) * inv
                    forces[i][1] += (ry / dist) * inv
            for i in range(n):
                pts[i][0] = (pts[i][0] + self._RELAX_STEP * forces[i][0] / aspect) % 1.0
                pts[i][1] = (pts[i][1] + self._RELAX_STEP * forces[i][1]) % 1.0

        # ── convert continuous coords → canonical axial hex coords ────────────
        positions: list[HexCoord] = []
        for fx, fy in pts:
            r_grid = int(fy * h) % h
            q_off = int(fx * w) % w
            q = q_off - r_grid // 2  # offset column → canonical axial q
            positions.append(self._grid.wrap(HexCoord(q, r_grid)))

        return positions
