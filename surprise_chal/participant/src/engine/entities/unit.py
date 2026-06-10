"""unit abstract base classes"""

from __future__ import annotations

from engine.constants import UNIT_STATS, UnitStats
from engine.entities.base import Entity
from engine.hex_grid import HexCoord


class Unit(Entity):
    def __init__(
        self, owner_id: str, coord: HexCoord, entity_id: str | None = None
    ) -> None:
        stats = self._stats()
        super().__init__(
            owner_id, coord, hp=stats.hp, max_hp=stats.hp, entity_id=entity_id
        )
        self.has_moved: bool = False
        self.has_attacked: bool = False

    @classmethod
    def _stats(cls) -> UnitStats:
        return UNIT_STATS[cls.__name__]

    @property
    def movement_range(self) -> int:
        return self._stats().movement_range

    @property
    def attack_range(self) -> int:
        return self._stats().attack_range

    @property
    def vision_range(self) -> int:
        return self._stats().vision_range

    @property
    def attack_power(self) -> int:
        return self._stats().attack_power

    @property
    def can_fly(self) -> bool:
        return self._stats().can_fly

    def entity_type(self) -> str:
        return self.__class__.__name__

    def reset_turn(self) -> None:
        self.has_moved = False
        self.has_attacked = False

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update(
            {
                "movement_range": self.movement_range,
                "attack_range": self.attack_range,
                "vision_range": self.vision_range,
                "attack_power": self.attack_power,
                "can_fly": self.can_fly,
                "has_moved": self.has_moved,
                "has_attacked": self.has_attacked,
            }
        )
        return d


class GroundUnit(Unit):
    """ground unit — occupies its tile exclusively (one entity per tile)"""


class AirUnit(Unit):
    """air unit. Like every entity it occupies its tile exclusively — the engine
    enforces a strict one-entity-per-tile rule via state.is_ground_blocked(), which
    does not distinguish air from ground. (Air units DO differ elsewhere: they
    ignore elevation line-of-sight blocking — see fog_of_war — and Bombers deal
    bonus damage to buildings.) There is no separate air layer / tile-sharing."""
