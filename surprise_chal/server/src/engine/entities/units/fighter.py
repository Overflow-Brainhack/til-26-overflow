from __future__ import annotations

from engine.entities.unit import AirUnit


class Fighter(AirUnit):
    """fast air unit; effective against other air units; can intercept"""

    def entity_type(self) -> str:
        return "Fighter"
