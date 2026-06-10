from __future__ import annotations

from engine.entities.building import ResourceBuilding


class Mine(ResourceBuilding):
    """passive gold-generating building"""

    def entity_type(self) -> str:
        return "Mine"
