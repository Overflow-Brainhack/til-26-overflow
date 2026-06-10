from __future__ import annotations

from engine.entities.unit import GroundUnit


class Artillery(GroundUnit):
    """long-range unit with splash damage around the primary target"""

    def entity_type(self) -> str:
        return "Artillery"
