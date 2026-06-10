from __future__ import annotations

from engine.entities.unit import GroundUnit


class Medic(GroundUnit):
    """heals adjacent friendly ground units each turn; cannot attack"""

    HEAL_AMOUNT: int = 20

    def entity_type(self) -> str:
        return "Medic"
