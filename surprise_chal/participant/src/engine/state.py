"""authoritative game state"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.entities.base import Entity
from engine.entities.building import Building
from engine.entities.unit import Unit
from engine.hex_grid import HexCoord, HexGrid
from engine.player import Player
from engine.terrain import Tile


@dataclass
class GameState:
    grid: HexGrid
    tiles: dict[HexCoord, Tile]
    players: dict[str, Player]
    entities: dict[str, Entity]
    turn_number: int = 0
    # map from coord → list of entity ids (derived, kept in sync)
    coord_index: dict[HexCoord, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rebuild_coord_index()

    def _rebuild_coord_index(self) -> None:
        self.coord_index = {}
        for entity in self.entities.values():
            self.coord_index.setdefault(entity.coord, []).append(entity.id)

    # ── entity helpers ────────────────────────────────────────────────────────

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity
        self.coord_index.setdefault(entity.coord, []).append(entity.id)

    def remove_entity(self, entity_id: str) -> None:
        entity = self.entities.pop(entity_id, None)
        if entity:
            ids = self.coord_index.get(entity.coord, [])
            if entity_id in ids:
                ids.remove(entity_id)

    def move_entity(self, entity_id: str, new_coord: HexCoord) -> None:
        entity = self.entities[entity_id]
        old_ids = self.coord_index.get(entity.coord, [])
        if entity_id in old_ids:
            old_ids.remove(entity_id)
        entity.coord = new_coord
        self.coord_index.setdefault(new_coord, []).append(entity_id)

    def entities_at(self, coord: HexCoord) -> list[Entity]:
        return [
            self.entities[eid]
            for eid in self.coord_index.get(coord, [])
            if eid in self.entities
        ]

    def units_at(self, coord: HexCoord) -> list[Unit]:
        return [e for e in self.entities_at(coord) if isinstance(e, Unit)]

    def buildings_at(self, coord: HexCoord) -> list[Building]:
        return [e for e in self.entities_at(coord) if isinstance(e, Building)]

    def units_for(self, player_id: str) -> list[Unit]:
        return [
            e
            for e in self.entities.values()
            if isinstance(e, Unit) and e.owner_id == player_id
        ]

    def buildings_for(self, player_id: str) -> list[Building]:
        return [
            e
            for e in self.entities.values()
            if isinstance(e, Building) and e.owner_id == player_id
        ]

    def tile(self, coord: HexCoord) -> Tile:
        return self.tiles.get(coord, Tile())

    # ── queries ───────────────────────────────────────────────────────────────

    def count_bases(self, player_id: str) -> int:
        from engine.entities.buildings.base_building import Base

        # only fully-constructed bases keep a player alive — a Base still under
        # construction does not count toward the elimination check
        return sum(
            1
            for b in self.buildings_for(player_id)
            if isinstance(b, Base) and b.is_complete
        )

    def alive_players(self) -> list[Player]:
        return [p for p in self.players.values() if p.alive]

    def is_ground_blocked(
        self, coord: HexCoord, moving_unit_id: str | None = None
    ) -> bool:
        """true if the tile is occupied by any entity other than the moving unit"""
        for entity in self.entities_at(coord):
            if entity.id != moving_unit_id:
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "turn_number": self.turn_number,
            "players": {pid: p.to_dict() for pid, p in self.players.items()},
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "tiles": {
                f"{c.q},{c.r}": {"terrain": t.terrain.name.lower()}
                for c, t in self.tiles.items()
            },
        }
