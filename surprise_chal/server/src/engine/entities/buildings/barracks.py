from __future__ import annotations

from engine.entities.building import ProductionBuilding


class Barracks(ProductionBuilding):
    """produces infantry, cavalry, scouts, and medics"""

    def entity_type(self) -> str:
        return "Barracks"
