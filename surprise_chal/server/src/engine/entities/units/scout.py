from __future__ import annotations

from engine.entities.unit import GroundUnit


class Scout(GroundUnit):
    """fast, high-vision unit; stealthy in concealment terrain"""

    def entity_type(self) -> str:
        return "Scout"
