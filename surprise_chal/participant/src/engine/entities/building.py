"""building abstract base classes"""

from __future__ import annotations

from engine.constants import BUILDING_STATS, BuildingStats
from engine.entities.base import Entity
from engine.hex_grid import HexCoord
from engine.resources import ResourceBag


class Building(Entity):
    def __init__(
        self, owner_id: str, coord: HexCoord, entity_id: str | None = None
    ) -> None:
        stats = self._stats()
        super().__init__(
            owner_id, coord, hp=stats.hp, max_hp=stats.hp, entity_id=entity_id
        )
        self.construction_turns_remaining: int = stats.build_turns
        # production queue: (unit_type_name, turns_remaining, target_coord)
        self.production_queue: list[tuple[str, int, HexCoord]] = []

    @classmethod
    def _stats(cls) -> BuildingStats:
        return BUILDING_STATS[cls.__name__]

    @property
    def is_complete(self) -> bool:
        return self.construction_turns_remaining == 0

    @property
    def vision_bonus(self) -> int:
        return self._stats().vision_bonus

    def entity_type(self) -> str:
        return self.__class__.__name__

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(
            {
                "construction_turns_remaining": self.construction_turns_remaining,
                "is_complete": self.is_complete,
                "vision_bonus": self.vision_bonus,
            }
        )
        return d


class ResourceBuilding(Building):
    """yields resources each turn once complete"""

    def yield_resources(self, tile_is_rich: bool) -> ResourceBag:
        from engine.constants import RICH_RESOURCE_FLAT_YIELD

        if not self.is_complete:
            return ResourceBag()
        if tile_is_rich:
            return ResourceBag(gold=RICH_RESOURCE_FLAT_YIELD)
        return ResourceBag(gold=self._stats().gold_yield_per_turn)


class ProductionBuilding(Building):
    """produces units when given a produce order"""

    @property
    def producible_unit_types(self) -> tuple[str, ...]:
        return self._stats().producible_unit_types

    def can_produce(self, unit_type: str) -> bool:
        return self.is_complete and unit_type in self.producible_unit_types

    def enqueue_unit(self, unit_type: str, target: HexCoord, build_turns: int) -> None:
        self.production_queue.append((unit_type, build_turns, target))

    def tick_production(self) -> list[tuple[str, HexCoord]]:
        """advance all queued productions by 1 turn; return completed (type, target) pairs"""
        completed: list[tuple[str, HexCoord]] = []
        new_queue: list[tuple[str, int, HexCoord]] = []
        for unit_type, turns, target in self.production_queue:
            turns -= 1
            if turns <= 0:
                completed.append((unit_type, target))
            else:
                new_queue.append((unit_type, turns, target))
        self.production_queue = new_queue
        return completed
