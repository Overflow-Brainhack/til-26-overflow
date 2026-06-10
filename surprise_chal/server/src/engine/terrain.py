"""terrain types and tile definition"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class TerrainType(Enum):
    NORMAL = auto()
    ELEVATED = auto()  # attack bonus for units on tile; blocks LOS for lower units
    DIFFICULT = auto()  # costs 2 movement points to enter
    CONCEALMENT = auto()  # reduces vision into this tile by 1
    RICH_RESOURCE = auto()  # resource buildings on this tile yield bonus gold


@dataclass
class Tile:
    terrain: TerrainType = TerrainType.NORMAL
    # entity ids occupying this tile (populated/maintained by GameState)
    entity_ids: list[str] = field(default_factory=list)

    def movement_cost(self) -> int:
        from engine.constants import DIFFICULT_TERRAIN_MOVE_COST

        if self.terrain == TerrainType.DIFFICULT:
            return DIFFICULT_TERRAIN_MOVE_COST
        return 1

    def is_elevated(self) -> bool:
        return self.terrain == TerrainType.ELEVATED

    def is_concealment(self) -> bool:
        return self.terrain == TerrainType.CONCEALMENT

    def is_rich_resource(self) -> bool:
        return self.terrain == TerrainType.RICH_RESOURCE
