from __future__ import annotations

from engine.entities.unit import GroundUnit


class Tank(GroundUnit):
    """heavy ground unit; high HP and attack, no special mechanics"""

    def entity_type(self) -> str:
        return "Tank"
