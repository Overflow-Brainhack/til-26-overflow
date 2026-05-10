"""Tracks the agent's accumulated knowledge of the world across steps.

Static state (walls, base positions, observed tile types) is preserved
across `/reset` calls so that novice mode (fixed map layout) doesn't need
to re-explore each round. Per-round dynamic state (bombs, enemy positions)
is cleared explicitly via `reset_round()`.

In novice mode the simulator hardcodes maze seed 19 and episode seed 88
(arena.py:454, dynamics.py:304), so the map is identical every game.
`save()`/`load()` lets us capture that map once offline and bundle the
JSON into the Docker image — round 1 starts with full map knowledge
instead of wasting steps on re-exploration.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from constants import (
    BASE_VISION_RADIUS,
    DESTR_WALL_CHANNEL,
    DIR_VECTOR,
    Direction,
    GRID_SIZE,
    NUM_CHANNELS,
    VIEWCONE_LENGTH,
    VIEWCONE_WIDTH,
    ViewChannel,
    WALL_CHANNEL,
)
from observation import ParsedObs, base_view_to_world, view_to_world


# How long we trust an enemy-agent sighting before discarding it.
ENEMY_AGENT_TTL = 12

# Walls of opposite cells share an edge; we represent edges as frozensets.
Edge = frozenset


def _edge(a: tuple[int, int], b: tuple[int, int]) -> Edge:
    return frozenset((a, b))


@dataclass
class BombInfo:
    pos: tuple[int, int]
    timer_at_seen: int  # countdown value when we last saw it
    step_seen: int      # global step at which we saw it
    ally: bool

    def remaining(self, current_step: int) -> int:
        return self.timer_at_seen - max(0, current_step - self.step_seen)


@dataclass
class MapMemory:
    # ── static knowledge (preserved across rounds, serializable) ────────────
    blocked_edges: set[Edge] = field(default_factory=set)
    destructible_edges: set[Edge] = field(default_factory=set)
    known_edges: set[Edge] = field(default_factory=set)
    tile_contents: dict[tuple[int, int], str] = field(default_factory=dict)
    # All bases ever observed, regardless of team. `enemy_bases` is computed
    # from this minus our current ally_base — which lets a single saved cache
    # file work for any agent_id (we'd never know which team we'd be at
    # capture time).
    base_positions: set[tuple[int, int]] = field(default_factory=set)

    # ── per-round dynamic state ─────────────────────────────────────────────
    ally_base: Optional[tuple[int, int]] = None
    bombs: dict[tuple[int, int], BombInfo] = field(default_factory=dict)
    enemy_agents: dict[tuple[int, int], int] = field(default_factory=dict)  # pos -> last_step
    last_seen_step: dict[tuple[int, int], int] = field(default_factory=dict)
    current_step: int = 0

    # ── derived ─────────────────────────────────────────────────────────────
    @property
    def enemy_bases(self) -> set[tuple[int, int]]:
        """All known base positions excluding our own."""
        if self.ally_base is None:
            return set(self.base_positions)
        return self.base_positions - {self.ally_base}

    # ── per-round reset ─────────────────────────────────────────────────────
    def reset_round(self) -> None:
        """Clear per-round dynamic state. Static map knowledge persists."""
        self.bombs.clear()
        self.enemy_agents.clear()
        self.current_step = 0

    def update(self, obs: ParsedObs) -> None:
        self.current_step = obs.step
        self.ally_base = obs.base_location

        self._ingest_view(
            obs.agent_view,
            world_for_cell=lambda r, c: view_to_world(obs.location, obs.direction, r, c),
            view_shape=(VIEWCONE_LENGTH, VIEWCONE_WIDTH),
        )
        self._ingest_view(
            obs.base_view,
            world_for_cell=lambda r, c: base_view_to_world(obs.base_location, r, c),
            view_shape=(2 * BASE_VISION_RADIUS + 1, 2 * BASE_VISION_RADIUS + 1),
        )

        self._gc_dynamic()

    def _ingest_view(self, view: np.ndarray, world_for_cell, view_shape: tuple[int, int]) -> None:
        h, w = view_shape
        if view.shape != (h, w, NUM_CHANNELS):
            return
        for r in range(h):
            for c in range(w):
                if view[r, c, ViewChannel.VISIBLE] < 0.5:
                    continue
                wx, wy = world_for_cell(r, c)
                if not (0 <= wx < GRID_SIZE and 0 <= wy < GRID_SIZE):
                    continue
                self.last_seen_step[(wx, wy)] = self.current_step
                self._stamp_walls(view[r, c], (wx, wy))
                self._stamp_tile(view[r, c], (wx, wy))
                self._stamp_entities(view[r, c], (wx, wy))

    def _stamp_walls(self, cell_view: np.ndarray, pos: tuple[int, int]) -> None:
        wx, wy = pos
        for d in Direction:
            dx, dy = DIR_VECTOR[d]
            nx, ny = wx + dx, wy + dy
            edge = _edge((wx, wy), (nx, ny))
            self.known_edges.add(edge)

            if cell_view[WALL_CHANNEL[d]] > 0.5:
                self.blocked_edges.add(edge)
                if cell_view[DESTR_WALL_CHANNEL[d]] > 0.5:
                    self.destructible_edges.add(edge)
                else:
                    self.destructible_edges.discard(edge)
            else:
                # No wall — clear any stale belief (e.g. destructible wall was bombed).
                self.blocked_edges.discard(edge)
                self.destructible_edges.discard(edge)

    def _stamp_tile(self, cell_view: np.ndarray, pos: tuple[int, int]) -> None:
        if cell_view[ViewChannel.TILE_MISSION] > 0.5:
            self.tile_contents[pos] = "mission"
        elif cell_view[ViewChannel.TILE_RESOURCE] > 0.5:
            self.tile_contents[pos] = "resource"
        elif cell_view[ViewChannel.TILE_RECON] > 0.5:
            self.tile_contents[pos] = "recon"
        elif cell_view[ViewChannel.TILE_EMPTY] > 0.5:
            self.tile_contents[pos] = "empty"

    def _stamp_entities(self, cell_view: np.ndarray, pos: tuple[int, int]) -> None:
        if cell_view[ViewChannel.ENEMY_AGENT] > 0.5:
            self.enemy_agents[pos] = self.current_step
        # Track ALL bases (ally or enemy). `enemy_bases` filters at read time.
        if cell_view[ViewChannel.ALLY_BASE] > 0.5 or cell_view[ViewChannel.ENEMY_BASE] > 0.5:
            self.base_positions.add(pos)

        ally_bomb = cell_view[ViewChannel.ALLY_BOMB] > 0.5
        enemy_bomb = cell_view[ViewChannel.ENEMY_BOMB] > 0.5
        if ally_bomb or enemy_bomb:
            timer_chan = ViewChannel.ALLY_BOMB_TIMER if ally_bomb else ViewChannel.ENEMY_BOMB_TIMER
            timer = int(round(cell_view[timer_chan]))
            self.bombs[pos] = BombInfo(
                pos=pos,
                timer_at_seen=max(timer, 0),
                step_seen=self.current_step,
                ally=ally_bomb,
            )

    def _gc_dynamic(self) -> None:
        # Drop bombs whose timer has elapsed.
        expired = [p for p, b in self.bombs.items() if b.remaining(self.current_step) < -1]
        for p in expired:
            del self.bombs[p]

        # Drop stale enemy sightings.
        cutoff = self.current_step - ENEMY_AGENT_TTL
        for p in [p for p, s in self.enemy_agents.items() if s < cutoff]:
            del self.enemy_agents[p]

    # ── query helpers ───────────────────────────────────────────────────────
    def passable(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        """Can the agent traverse from a to b? Treats unknown edges as passable
        (optimistic — encourages exploration)."""
        return _edge(a, b) not in self.blocked_edges

    def edge_is_destructible_wall(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        return _edge(a, b) in self.destructible_edges

    def edge_is_known(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        return _edge(a, b) in self.known_edges

    def tile_value(self, pos: tuple[int, int]) -> float:
        from constants import REWARD_MISSION, REWARD_RECON, REWARD_RESOURCE
        kind = self.tile_contents.get(pos)
        if kind == "mission":
            return REWARD_MISSION
        if kind == "resource":
            return REWARD_RESOURCE
        if kind == "recon":
            return REWARD_RECON
        return 0.0

    def collectible_cells(self) -> list[tuple[int, int]]:
        return [p for p, k in self.tile_contents.items() if k in ("mission", "resource", "recon")]

    def in_bounds(self, p: tuple[int, int]) -> bool:
        return 0 <= p[0] < GRID_SIZE and 0 <= p[1] < GRID_SIZE

    # ── persistence (static state only) ─────────────────────────────────────

    def to_static_dict(self) -> dict[str, Any]:
        """Serialize the across-rounds-stable subset of state.

        Excludes: ally_base (per-team), bombs, enemy_agents, last_seen_step,
        current_step. Including those would couple a saved cache to a
        specific team or round.
        """
        def edges(s: set[Edge]) -> list[list[list[int]]]:
            out: list[list[list[int]]] = []
            for e in s:
                pts = sorted(e)  # canonical ordering for determinism
                out.append([list(pts[0]), list(pts[1])])
            out.sort()
            return out

        return {
            "version": 1,
            "blocked_edges": edges(self.blocked_edges),
            "destructible_edges": edges(self.destructible_edges),
            "known_edges": edges(self.known_edges),
            "tile_contents": [
                [list(p), v] for p, v in sorted(self.tile_contents.items())
            ],
            "base_positions": sorted(list(p) for p in self.base_positions),
        }

    @classmethod
    def from_static_dict(cls, data: dict[str, Any]) -> "MapMemory":
        m = cls()
        for key, dest in (
            ("blocked_edges", m.blocked_edges),
            ("destructible_edges", m.destructible_edges),
            ("known_edges", m.known_edges),
        ):
            for a, b in data.get(key, []):
                dest.add(_edge(tuple(a), tuple(b)))
        m.tile_contents = {
            tuple(p): v for p, v in data.get("tile_contents", [])
        }
        m.base_positions = {tuple(p) for p in data.get("base_positions", [])}
        return m

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_static_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MapMemory":
        return cls.from_static_dict(json.loads(Path(path).read_text()))

    def merge_static_from(self, other: "MapMemory") -> None:
        """Adopt another memory's static state into ours. Idempotent."""
        self.blocked_edges.update(other.blocked_edges)
        self.destructible_edges.update(other.destructible_edges)
        self.known_edges.update(other.known_edges)
        self.tile_contents.update(other.tile_contents)
        self.base_positions.update(other.base_positions)


# ── module-level singleton ─────────────────────────────────────────────────
# Shared across AEManager instances within a single Docker process. Survives
# /reset so novice-mode static knowledge accumulates across rounds.
_SHARED: Optional[MapMemory] = None


def get_shared_memory() -> MapMemory:
    global _SHARED
    if _SHARED is None:
        _SHARED = MapMemory()
    return _SHARED
