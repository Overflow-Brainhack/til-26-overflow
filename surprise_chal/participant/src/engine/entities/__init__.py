"""entity registry — maps type-name strings to classes"""

from engine.entities.base import Entity
from engine.entities.building import Building, ProductionBuilding, ResourceBuilding
from engine.entities.buildings import (
    Airbase,
    Barracks,
    Base,
    Factory,
    Mine,
)
from engine.entities.unit import AirUnit, GroundUnit, Unit
from engine.entities.units import (
    Artillery,
    Bomber,
    Fighter,
    Infantry,
    Medic,
    Scout,
    Tank,
)

UNIT_REGISTRY: dict[str, type[Unit]] = {
    "Infantry": Infantry,
    "Tank": Tank,
    "Artillery": Artillery,
    "Scout": Scout,
    "Medic": Medic,
    "Fighter": Fighter,
    "Bomber": Bomber,
}

BUILDING_REGISTRY: dict[str, type[Building]] = {
    "Base": Base,
    "Mine": Mine,
    "Barracks": Barracks,
    "Factory": Factory,
    "Airbase": Airbase,
}

ENTITY_REGISTRY: dict[str, type[Entity]] = {**UNIT_REGISTRY, **BUILDING_REGISTRY}

__all__ = [
    "Entity",
    "Unit",
    "GroundUnit",
    "AirUnit",
    "Building",
    "ResourceBuilding",
    "ProductionBuilding",
    "Infantry",
    "Tank",
    "Artillery",
    "Scout",
    "Medic",
    "Fighter",
    "Bomber",
    "Base",
    "Mine",
    "Barracks",
    "Factory",
    "Airbase",
    "UNIT_REGISTRY",
    "BUILDING_REGISTRY",
    "ENTITY_REGISTRY",
]
