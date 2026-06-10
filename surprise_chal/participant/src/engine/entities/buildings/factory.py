from __future__ import annotations

from engine.entities.building import ProductionBuilding


class Factory(ProductionBuilding):
    """produces artillery"""

    def entity_type(self) -> str:
        return "Factory"
