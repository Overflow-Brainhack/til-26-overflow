"""all game constants and configurable values"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ── server ────────────────────────────────────────────────────────────────────

RESPONSE_TIMEOUT_SECONDS: float = 10.0
PLAYER_SERVER_PORT: int = 6700
MAX_PLAYERS: int = 20

# ── game rules ────────────────────────────────────────────────────────────────

DEFAULT_MAP_WIDTH: int = 35
DEFAULT_MAP_HEIGHT: int = 30
MAX_TURNS: int = 300
# Stand-in competition map seed. 
DEFAULT_GAME_SEED: int = 67
TREATY_BREAK_DELAY_TURNS: int = 5
# From this turn onward all peace treaties are void and no new treaty may be formed
# (proposals/accepts are ignored). The final stretch of the game is forced open war,
# so a board frozen by mutual peace cannot coast to the turn limit. With MAX_TURNS=300
# this opens up the last 100 turns.
TREATY_CUTOFF_TURN: int = 200
UNIT_DECAY_PER_TURN: int = 10  # hp lost each turn after player is eliminated

# ── map generation ────────────────────────────────────────────────────────────
#
# Two-channel heightmap algorithm:
#   1. height noise  → primary land structure (mountain ridges, foothills, plains)
#   2. resource noise → sparse RICH_RESOURCE overlay on plains only
#   3. moisture noise → CONCEALMENT overlay on remaining plains
#
# Approximate resulting distribution:
#   ELEVATED      ~12%   (height > ELEVATED_THRESHOLD)
#   DIFFICULT     ~20%   (height in DIFFICULT_THRESHOLD..ELEVATED_THRESHOLD)
#   RICH_RESOURCE ~10%   (resource > RESOURCE_THRESHOLD, plains only)
#   CONCEALMENT   ~13%   (moisture > MOISTURE_THRESHOLD, non-resource plains)
#   NORMAL        ~45%   (everything else)

# Thresholds are calibrated to actual Perlin noise percentiles (NOISE_SCALE=4,
# NOISE_OCTAVES=4 produces a bell-shaped distribution from ~0.08 to ~0.89,
# NOT uniform [0,1] — so percentile values are used directly):
#   p68=0.577  p78=0.620  p85=0.661  p88=0.681
MAP_HEIGHT_ELEVATED_THRESHOLD: float = 0.681  # p88 → top 12% of tiles → mountain peaks
MAP_HEIGHT_DIFFICULT_THRESHOLD: float = (
    0.577  # p68 → next 20% of tiles → foothills/rough
)
MAP_RESOURCE_THRESHOLD: float = 0.661  # p85 → 15% of tiles, plains-only → ~10% total
MAP_MOISTURE_THRESHOLD: float = (
    0.620  # p78 → 22% of tiles, remaining plains → ~13% total
)

NOISE_SCALE: float = 4.0
NOISE_OCTAVES: int = 4

# ── unit stats ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UnitStats:
    hp: int
    movement_range: int
    attack_range: int
    vision_range: int
    attack_power: int
    gold_cost: int
    build_turns: int
    can_fly: bool = False


UNIT_STATS: dict[str, UnitStats] = {
    "Infantry": UnitStats(
        hp=100,
        movement_range=1,
        attack_range=1,
        vision_range=3,
        attack_power=30,
        gold_cost=50,
        build_turns=1,
    ),
    "Scout": UnitStats(
        hp=50,
        movement_range=3,
        attack_range=1,
        vision_range=5,
        attack_power=10,
        gold_cost=100,
        build_turns=1,
    ),
    "Medic": UnitStats(
        hp=60,
        movement_range=1,
        attack_range=0,
        vision_range=3,
        attack_power=0,
        gold_cost=100,
        build_turns=1,
    ),
    "Tank": UnitStats(
        hp=200,
        movement_range=2,
        attack_range=1,
        vision_range=3,
        attack_power=60,
        gold_cost=200,
        build_turns=1,
    ),
    "Artillery": UnitStats(
        hp=50,
        movement_range=1,
        attack_range=3,
        vision_range=4,
        attack_power=60,
        gold_cost=200,
        build_turns=2,
    ),
    "Fighter": UnitStats(
        hp=250,
        movement_range=3,
        attack_range=2,
        vision_range=4,
        attack_power=50,
        gold_cost=300,
        build_turns=2,
        can_fly=True,
    ),
    "Bomber": UnitStats(
        hp=150,
        movement_range=2,
        attack_range=1,
        vision_range=3,
        attack_power=50,
        gold_cost=350,
        build_turns=3,
        can_fly=True,
    ),
}

# ── building stats ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuildingStats:
    hp: int
    gold_cost: int
    build_turns: int
    gold_yield_per_turn: int = 0
    producible_unit_types: tuple[str, ...] = field(default_factory=tuple)
    vision_bonus: int = 0


BUILDING_STATS: dict[str, BuildingStats] = {
    "Base": BuildingStats(
        hp=300,
        gold_cost=300,
        build_turns=5,
        gold_yield_per_turn=10,
        vision_bonus=3,
    ),
    "Mine": BuildingStats(
        hp=100,
        gold_cost=200,
        build_turns=2,
        gold_yield_per_turn=20,
    ),
    "Barracks": BuildingStats(
        hp=200,
        gold_cost=100,
        build_turns=2,
        producible_unit_types=("Infantry", "Scout", "Medic"),
    ),
    "Factory": BuildingStats(
        hp=200,
        gold_cost=300,
        build_turns=3,
        producible_unit_types=("Tank", "Artillery"),
    ),
    "Airbase": BuildingStats(
        hp=200,
        gold_cost=500,
        build_turns=3,
        producible_unit_types=("Fighter", "Bomber"),
    ),
}

# ── terrain modifiers ─────────────────────────────────────────────────────────

RICH_RESOURCE_FLAT_YIELD: int = (
    50  # any resource building (Base/Mine) on a rich tile yields this flat amount
)
DIFFICULT_TERRAIN_MOVE_COST: int = 2  # movement points consumed per step
ELEVATION_ATTACK_BONUS: float = 1.25  # attacker on elevated tile deals 25% more damage

# ── artillery splash ──────────────────────────────────────────────────────────

ARTILLERY_SPLASH_RADIUS: int = 1
ARTILLERY_SPLASH_DAMAGE_RATIO: float = 0.5  # splash deals 50% of primary damage
