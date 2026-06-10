"""entity base class"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from engine.hex_grid import HexCoord


class Entity(ABC):
    def __init__(
        self,
        owner_id: str,
        coord: HexCoord,
        hp: int,
        max_hp: int,
        entity_id: str | None = None,
    ) -> None:
        self.id: str = entity_id or str(uuid.uuid4())
        self.owner_id: str = owner_id
        self.coord: HexCoord = coord
        self.hp: int = hp
        self.max_hp: int = max_hp

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    def take_damage(self, amount: int) -> None:
        self.hp = max(0, self.hp - amount)

    def heal(self, amount: int) -> None:
        self.hp = min(self.max_hp, self.hp + amount)

    @abstractmethod
    def entity_type(self) -> str:
        """returns the class name string used for serialisation"""
        ...

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "type": self.entity_type(),
            "q": self.coord.q,
            "r": self.coord.r,
            "hp": self.hp,
            "max_hp": self.max_hp,
        }
