"""TEMPLATE: a deterministic (no-LLM) agent.

A complete, readable baseline you can run with no API key. It:
  1. spends gold on economy/production (Mine → Barracks → Infantry),
  2. produces Infantry from a completed Barracks onto a free adjacent tile,
  3. attacks any enemy in range, else steps toward the nearest enemy.

It is intentionally simple — beat it. Everything you need is in the observation
dict and the `engine/` package (the real rules + stats). Costs/ranges come from
`engine.constants` so they always match the engine.

Run it with:  AGENT=algo  python -m server   (see README)
"""

from __future__ import annotations

from agent_base import PlayerAgent
from engine.actions import (
    ActionPayload,
    AttackAction,
    ConstructBuildingAction,
    HoldAction,
    MoveAction,
    ProduceUnitAction,
)
from engine.constants import BUILDING_STATS, UNIT_STATS
from engine.hex_grid import HexCoord, HexGrid

_PRODUCTION_BUILDINGS = ("Barracks", "Factory", "Airbase")


def _flatten(observation: dict, pid: str):
    """Pull own units, own buildings, enemies, and occupied tiles out of the obs."""
    own_units, own_buildings, enemies = [], [], []
    occupied: set[tuple[int, int]] = set()
    for tile in observation.get("visible_tiles", []):
        for e in tile.get("entities", []):
            occupied.add((e["q"], e["r"]))
            if e.get("owner_id") == pid:
                # buildings have no attack_range field; units do
                if e.get("type") in BUILDING_STATS:
                    own_buildings.append(e)
                else:
                    own_units.append(e)
            else:
                enemies.append(e)
    return own_units, own_buildings, enemies, occupied


class AlgoAgent(PlayerAgent):
    async def decide(self, observation: dict) -> ActionPayload:
        pid = observation["player_id"]
        turn = observation.get("turn_number", 0)
        gold = observation.get("resources", {}).get("gold", 0)
        grid = HexGrid(
            observation.get("map_width", 35), observation.get("map_height", 30)
        )
        own_units, own_buildings, enemies, occupied = _flatten(observation, pid)
        actions: list = []

        # ── economy: produce a unit, else build something ──────────────────────
        complete = [b for b in own_buildings if b.get("is_complete", True)]
        prod = [b for b in complete if b["type"] in _PRODUCTION_BUILDINGS]
        if prod and gold >= UNIT_STATS["Infantry"].gold_cost:
            b = prod[0]
            spot = self._free_neighbour(grid, (b["q"], b["r"]), occupied)
            if spot:
                actions.append(
                    ProduceUnitAction(
                        building_id=b["id"],
                        unit_type="Infantry",
                        target=HexCoord(*spot),
                    )
                )
                occupied.add(spot)
                gold -= UNIT_STATS["Infantry"].gold_cost
        else:
            # build the next economy/production piece next to a completed building
            mines = sum(1 for b in own_buildings if b["type"] == "Mine")
            have_barracks = any(b["type"] == "Barracks" for b in own_buildings)
            want = "Mine" if mines < 1 or have_barracks else "Barracks"
            cost = BUILDING_STATS[want].gold_cost
            if gold >= cost and complete:
                spot = None
                for b in complete:
                    spot = self._free_neighbour(grid, (b["q"], b["r"]), occupied)
                    if spot:
                        break
                if spot:
                    actions.append(
                        ConstructBuildingAction(
                            building_type=want, coord=HexCoord(*spot)
                        )
                    )
                    occupied.add(spot)
                    gold -= cost

        # ── combat: attack in range, else advance toward the nearest enemy ─────
        for u in own_units:
            ar = u.get("attack_range", 0)
            mr = u.get("movement_range", 0)
            here = HexCoord(u["q"], u["r"])
            target = self._nearest(grid, here, enemies)
            if target is None:
                continue  # nothing to do (units may also just hold)
            tc = HexCoord(target["q"], target["r"])
            dist = grid.distance(here, tc)
            if ar >= 1 and 0 < dist <= ar:
                actions.append(AttackAction(unit_id=u["id"], target=tc))
            elif mr >= 1:
                step = self._step_toward(grid, here, tc, occupied)
                if step is not None:
                    actions.append(MoveAction(unit_id=u["id"], path=[here, step]))
                else:
                    actions.append(HoldAction(unit_id=u["id"]))

        return ActionPayload(player_id=pid, turn_number=turn, actions=actions)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _free_neighbour(grid, coord, occupied):
        for n in grid.neighbors(HexCoord(*coord)):
            if (n.q, n.r) not in occupied:
                return (n.q, n.r)
        return None

    @staticmethod
    def _nearest(grid, here, enemies):
        best, best_d = None, 10**9
        for e in enemies:
            d = grid.distance(here, HexCoord(e["q"], e["r"]))
            if d < best_d:
                best, best_d = e, d
        return best

    @staticmethod
    def _step_toward(grid, here, target, occupied):
        """Pick the free neighbour of `here` that gets closest to `target`."""
        best, best_d = None, grid.distance(here, target)
        for n in grid.neighbors(here):
            if (n.q, n.r) in occupied:
                continue
            d = grid.distance(n, target)
            if d < best_d:
                best, best_d = n, d
        return best
