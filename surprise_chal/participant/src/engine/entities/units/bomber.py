from __future__ import annotations

from engine.entities.unit import AirUnit


class Bomber(AirUnit):
    """heavy air unit; deals +300% (x4) bonus damage to buildings"""

    BUILDING_DAMAGE_MULTIPLIER: float = 4.0

    def entity_type(self) -> str:
        return "Bomber"
