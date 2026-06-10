from __future__ import annotations

from engine.entities.building import ResourceBuilding
from engine.hex_grid import HexCoord


class Base(ResourceBuilding):
    """player starting building; its destruction triggers elimination"""

    def __init__(
        self, owner_id: str, coord: HexCoord, entity_id: str | None = None
    ) -> None:
        super().__init__(owner_id, coord, entity_id)
        # bases are pre-built at game start
        self.construction_turns_remaining = 0

    def entity_type(self) -> str:
        return "Base"
