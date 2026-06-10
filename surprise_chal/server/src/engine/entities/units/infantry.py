from __future__ import annotations

from engine.entities.unit import GroundUnit


class Infantry(GroundUnit):
    """balanced, cheap ground unit"""

    def entity_type(self) -> str:
        return "Infantry"
