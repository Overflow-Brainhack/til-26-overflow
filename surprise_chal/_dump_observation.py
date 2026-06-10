"""Build one real observation from the engine and print it. Throwaway."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server", "src"))

from engine.chat import ChatLog, ChatMessage
from engine.diplomacy import DiplomacyManager, TreatyType
from engine.entities.buildings.base_building import Base
from engine.entities.units.infantry import Infantry
from engine.entities.units.scout import Scout
from engine.hex_grid import HexCoord, HexGrid
from engine.player import Player
from engine.resources import ResourceBag
from engine.state import GameState
from engine.terrain import Tile, TerrainType
from schemas.observation import build_observation

grid = HexGrid(12, 10)
tiles = {
    HexCoord(5, 3): Tile(TerrainType.ELEVATED),
    HexCoord(4, 4): Tile(TerrainType.CONCEALMENT),
    HexCoord(3, 2): Tile(TerrainType.RICH_RESOURCE),
    HexCoord(2, 4): Tile(TerrainType.DIFFICULT),
}
players = {
    "player-0": Player(id="player-0", name="P0", resources=ResourceBag(gold=340)),
    "player-1": Player(id="player-1", name="P1", resources=ResourceBag(gold=999)),
}
state = GameState(grid=grid, tiles=tiles, players=players, entities={})
state.turn_number = 5

# player-0: a base (complete), an infantry, a scout (wide vision)
state.add_entity(Base("player-0", HexCoord(3, 3)))
state.add_entity(Infantry("player-0", HexCoord(3, 4)))
state.add_entity(Scout("player-0", HexCoord(4, 3)))
# player-1: a base + infantry, in range of player-0's scout
state.add_entity(Base("player-1", HexCoord(6, 3)))
state.add_entity(Infantry("player-1", HexCoord(6, 4)))

# diplomacy: player-1 has proposed peace to player-0 (shows as incoming proposal)
diplo = DiplomacyManager()
diplo.propose("player-1", "player-0", TreatyType.PEACE)

# player-0 has "met" player-1 (saw their base)
players["player-0"].known_player_ids.add("player-1")

# chat: one global broadcast, one DM to player-0, one system DM
chat = ChatLog()
chat.post(ChatMessage(turn=2, sender_id="player-7", text="anyone want to gang up on P3?"))
chat.post(ChatMessage(turn=4, sender_id="player-1", text="truce?", recipient_id="player-0"))
chat.post(ChatMessage(turn=4, sender_id="__system__", text="peace treaty formed: P0 <-> P9", recipient_id="player-0"))

obs = build_observation(state, "player-0", diplo, chat, max_turns=300)

# Print: full structure, but trim visible_tiles to the ones holding entities + a count.
shown = dict(obs)
with_ents = [t for t in obs["visible_tiles"] if t["entities"]]
shown["visible_tiles"] = with_ents + [f"... +{len(obs['visible_tiles']) - len(with_ents)} more EMPTY visible tiles (terrain only)"]
print(json.dumps(shown, indent=2))
print(f"\n# top-level keys: {list(obs.keys())}")
print(f"# total visible tiles: {len(obs['visible_tiles'])}  (of {grid.width*grid.height} on the map)")
