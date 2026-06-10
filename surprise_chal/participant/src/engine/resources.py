"""resource types and resource bag — extensible to multiple resource types"""

from __future__ import annotations

from enum import Enum, auto


class ResourceType(Enum):
    GOLD = auto()
    # future resource types added here (e.g. IRON = auto())


class ResourceBag:
    """maps resource types to integer amounts; supports arithmetic"""

    def __init__(self, **kwargs: int) -> None:
        self._data: dict[ResourceType, int] = {rt: 0 for rt in ResourceType}
        for key, val in kwargs.items():
            self._data[ResourceType[key.upper()]] = val

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> ResourceBag:
        bag = cls()
        for k, v in d.items():
            bag._data[ResourceType[k.upper()]] = v
        return bag

    def get(self, rt: ResourceType) -> int:
        return self._data.get(rt, 0)

    def set(self, rt: ResourceType, amount: int) -> None:
        self._data[rt] = amount

    def __add__(self, other: ResourceBag) -> ResourceBag:
        result = ResourceBag()
        for rt in ResourceType:
            result._data[rt] = self.get(rt) + other.get(rt)
        return result

    def __sub__(self, other: ResourceBag) -> ResourceBag:
        result = ResourceBag()
        for rt in ResourceType:
            result._data[rt] = self.get(rt) - other.get(rt)
        return result

    def __iadd__(self, other: ResourceBag) -> ResourceBag:
        for rt in ResourceType:
            self._data[rt] += other.get(rt)
        return self

    def __isub__(self, other: ResourceBag) -> ResourceBag:
        for rt in ResourceType:
            self._data[rt] -= other.get(rt)
        return self

    def can_afford(self, cost: ResourceBag) -> bool:
        return all(self.get(rt) >= cost.get(rt) for rt in ResourceType)

    def to_dict(self) -> dict[str, int]:
        return {rt.name.lower(): self.get(rt) for rt in ResourceType}

    def __repr__(self) -> str:
        parts = ", ".join(f"{rt.name}={self.get(rt)}" for rt in ResourceType)
        return f"ResourceBag({parts})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourceBag):
            return NotImplemented
        return all(self.get(rt) == other.get(rt) for rt in ResourceType)
