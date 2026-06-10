"""A trivial random-ish baseline opponent (no API key). Useful as filler in
local games. It is deliberately weak — your agent should crush it.

It only ever produces a unit from a completed production building or builds a
Mine when it can afford one, and sends units at the nearest enemy. Just enough
to not sit completely idle.
"""

from __future__ import annotations

import random

from agent_base import PlayerAgent
from engine.actions import (
    ActionPayload,
    AttackAction,
    ConstructBuildingAction,
    MoveAction,
    ProduceUnitAction,
)
from engine.constants import BUILDING_STATS, UNIT_STATS
from engine.hex_grid import HexCoord, HexGrid


class RandomAgent(PlayerAgent):
    async def decide(self, observation: dict) -> ActionPayload:
        pid = observation["player_id"]
        turn = observation.get("turn_number", 0)
        gold = observation.get("resources", {}).get("gold", 0)
        grid = HexGrid(
            observation.get("map_width", 35), observation.get("map_height", 30)
        )
        units, buildings, enemies, occ = [], [], [], set()
        for tile in observation.get("visible_tiles", []):
            for e in tile.get("entities", []):
                occ.add((e["q"], e["r"]))
                if e.get("owner_id") == pid:
                    (buildings if e["type"] in BUILDING_STATS else units).append(e)
                else:
                    enemies.append(e)

        actions: list = []
        complete = [b for b in buildings if b.get("is_complete", True)]
        prod = [b for b in complete if b["type"] in ("Barracks", "Factory", "Airbase")]
        if prod and gold >= UNIT_STATS["Infantry"].gold_cost:
            b = prod[0]
            for nb in grid.neighbors(HexCoord(b["q"], b["r"])):
                if (nb.q, nb.r) not in occ:
                    actions.append(
                        ProduceUnitAction(
                            building_id=b["id"], unit_type="Infantry", target=nb
                        )
                    )
                    break
        elif complete and gold >= BUILDING_STATS["Mine"].gold_cost:
            b = random.choice(complete)
            for nb in grid.neighbors(HexCoord(b["q"], b["r"])):
                if (nb.q, nb.r) not in occ:
                    actions.append(ConstructBuildingAction(building_type="Mine", coord=nb))
                    break

        for u in units:
            if not enemies or u.get("movement_range", 0) < 1:
                continue
            here = HexCoord(u["q"], u["r"])
            tgt = min(enemies, key=lambda e: grid.distance(here, HexCoord(e["q"], e["r"])))
            tc = HexCoord(tgt["q"], tgt["r"])
            if u.get("attack_range", 0) >= 1 and 0 < grid.distance(here, tc) <= u["attack_range"]:
                actions.append(AttackAction(unit_id=u["id"], target=tc))
            else:
                for nb in grid.neighbors(here):
                    if (nb.q, nb.r) not in occ and grid.distance(nb, tc) < grid.distance(here, tc):
                        actions.append(MoveAction(unit_id=u["id"], path=[here, nb]))
                        break
        return ActionPayload(player_id=pid, turn_number=turn, actions=actions)
