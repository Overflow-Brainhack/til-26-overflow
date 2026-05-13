"""Game constants — copied from til-26-ae/til_environment/bomberman_config.yaml.

Duplicated here so the Docker image doesn't need the simulator package.
If the official config changes, update these values.
"""

from enum import IntEnum


GRID_SIZE = 16
NUM_ITERS = 200

# Vision (left, right, behind, ahead) — agent_viewcone is (behind+ahead+1, left+right+1, C).
VIEWCONE = (2, 2, 2, 4)
VIEWCONE_LENGTH = VIEWCONE[2] + VIEWCONE[3] + 1  # 7
VIEWCONE_WIDTH = VIEWCONE[0] + VIEWCONE[1] + 1   # 5
AGENT_ROW_OFFSET = VIEWCONE[2]                   # row where agent sits (= 2)
AGENT_COL_OFFSET = VIEWCONE[0]                   # col where agent sits (= 2)

BASE_VISION_RADIUS = 3
BASE_VIEW_SIDE = 2 * BASE_VISION_RADIUS + 1      # 7

NUM_ACTIONS = 6
NUM_CHANNELS = 25

# Bomb stats.
BOMB_TIMER = 3
BOMB_BLAST_RADIUS = 2
BOMB_ATTACK = 20.0

# Agent / base.
AGENT_MAX_HEALTH = 60.0
BASE_MAX_HEALTH = 100.0
FREEZE_TURNS = 3

# Economy.
BOMB_COST = 1.5
BASE_RESOURCE_RATE = 0.1

# Tile reward values (for goal scoring).
REWARD_MISSION = 5.0
REWARD_RECON = 1.0
REWARD_RESOURCE = 2.0

# Reward bonuses for finishing blows (from bomberman_config.yaml rewards section).
BASE_DESTROY_BONUS = 50.0
AGENT_KILL_BONUS = 30.0


class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


# Direction → (dx, dy). y grows downward (matches simulator convention).
DIR_VECTOR = {
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.UP: (0, -1),
}


class Action(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2       # turn 90° counter-clockwise
    RIGHT = 3      # turn 90° clockwise
    STAY = 4
    PLACE_BOMB = 5


class ViewChannel(IntEnum):
    """Mirrors til_environment.observation.ViewChannel."""
    VISIBLE = 0
    WALL_RIGHT = 1
    WALL_DOWN = 2
    WALL_LEFT = 3
    WALL_UP = 4
    TILE_EMPTY = 5
    TILE_RECON = 6
    TILE_MISSION = 7
    TILE_RESOURCE = 8
    ALLY_AGENT = 9
    ENEMY_AGENT = 10
    ALLY_BASE = 11
    ENEMY_BASE = 12
    DESTR_WALL_RIGHT = 13
    DESTR_WALL_DOWN = 14
    DESTR_WALL_LEFT = 15
    DESTR_WALL_UP = 16
    ALLY_BOMB = 17
    ENEMY_BOMB = 18
    ALLY_BOMB_TIMER = 19
    ENEMY_BOMB_TIMER = 20
    ALLY_AGENT_HEALTH = 21
    ENEMY_AGENT_HEALTH = 22
    ALLY_BASE_HEALTH = 23
    ENEMY_BASE_HEALTH = 24


# Wall channel pairs by direction the wall faces (from this cell's perspective).
WALL_CHANNEL = {
    Direction.RIGHT: ViewChannel.WALL_RIGHT,
    Direction.DOWN: ViewChannel.WALL_DOWN,
    Direction.LEFT: ViewChannel.WALL_LEFT,
    Direction.UP: ViewChannel.WALL_UP,
}

DESTR_WALL_CHANNEL = {
    Direction.RIGHT: ViewChannel.DESTR_WALL_RIGHT,
    Direction.DOWN: ViewChannel.DESTR_WALL_DOWN,
    Direction.LEFT: ViewChannel.DESTR_WALL_LEFT,
    Direction.UP: ViewChannel.DESTR_WALL_UP,
}
