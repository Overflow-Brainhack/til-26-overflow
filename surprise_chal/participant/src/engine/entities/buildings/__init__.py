"""concrete building types"""

from engine.entities.buildings.airbase import Airbase
from engine.entities.buildings.barracks import Barracks
from engine.entities.buildings.base_building import Base
from engine.entities.buildings.factory import Factory
from engine.entities.buildings.mine import Mine

__all__ = ["Base", "Mine", "Barracks", "Factory", "Airbase"]
