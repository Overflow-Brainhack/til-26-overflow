from __future__ import annotations

from engine.entities.building import ProductionBuilding


class Airbase(ProductionBuilding):
    """produces fighters and bombers"""

    def entity_type(self) -> str:
        return "Airbase"
