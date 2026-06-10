"""Engine-level proof of the core defensive thesis (runs the REAL turn_processor):

  1. An UNRINGED Base dies to 2 Bombers in one turn (2×200 = 400 ≥ 300 hp).
  2. A FULL 6-unit denial ring physically blocks a Bomber from the adjacency it
     needs (air units are blocked from occupied tiles) → the Base takes 0.
  3. A reactively teleported reserve does NOT intercept this turn's strike
     (attacks resolve from the pre-move tile, before movement) → ring must stand.
  4. Base redundancy: losing one complete Base is not fatal while another stands.

These are the load-bearing claims behind the agent's strategy. Throwaway.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "participant", "src"))

from engine.actions import ActionPayload, AttackAction, MoveAction  # noqa: E402
from engine.diplomacy import DiplomacyManager  # noqa: E402
from engine.entities import Base, Bomber, Infantry  # noqa: E402
from engine.hex_grid import HexCoord, HexGrid  # noqa: E402
from engine.player import Player  # noqa: E402
from engine.resources import ResourceBag  # noqa: E402
from engine.state import GameState  # noqa: E402
from engine.turn_processor import TurnProcessor  # noqa: E402

OK = lambda b: "PASS" if b else "*** FAIL ***"  # noqa: E731


def fresh():
    grid = HexGrid(35, 30)
    players = {
        "D": Player(id="D", name="Defender", resources=ResourceBag(gold=0)),
        "A": Player(id="A", name="Attacker", resources=ResourceBag(gold=0)),
    }
    # attacker keeps a far-off base so it is never eliminated mid-test
    state = GameState(grid=grid, tiles={}, players=players, entities={})
    state.add_entity(Base("A", HexCoord(30, 25)))
    return state


def run(state, *payloads):
    diplo = DiplomacyManager()
    TurnProcessor(state, diplo).process_turn(
        {p.player_id: p for p in payloads}
    )


print("=" * 70)
print("TEST 1 — unringed Base vs 2 Bombers in one turn")
state = fresh()
base = Base("D", HexCoord(10, 10))
state.add_entity(base)
ring = state.grid.neighbors(HexCoord(10, 10))
b1 = Bomber("A", ring[0])
b2 = Bomber("A", ring[1])
state.add_entity(b1)
state.add_entity(b2)
hp0 = base.hp
run(
    state,
    ActionPayload("A", 0, [
        AttackAction(unit_id=b1.id, target=HexCoord(10, 10)),
        AttackAction(unit_id=b2.id, target=HexCoord(10, 10)),
    ]),
    ActionPayload("D", 0, []),
)
dead = base.id not in state.entities
print(f"  base hp {hp0} -> {'DESTROYED' if dead else base.hp};  2x200=400 >= 300  => {OK(dead)}")

print("=" * 70)
print("TEST 2 — FULL 6-Infantry ring blocks a Bomber from reaching the Base")
state = fresh()
base = Base("D", HexCoord(10, 10))
state.add_entity(base)
for c in state.grid.neighbors(HexCoord(10, 10)):
    state.add_entity(Infantry("D", c))  # all 6 ring tiles occupied
# bomber sits two out and tries to teleport onto a ring tile, then (next turn) bomb
bomber = Bomber("A", HexCoord(10, 7))
state.add_entity(bomber)
target_ring = state.grid.neighbors(HexCoord(10, 10))[0]
run(
    state,
    ActionPayload("A", 0, [MoveAction(unit_id=bomber.id, path=[HexCoord(10, 7), target_ring])]),
    ActionPayload("D", 0, []),
)
blocked = bomber.coord != target_ring  # could not enter the occupied ring tile
print(f"  bomber tried to enter occupied ring tile {target_ring}; ended at {bomber.coord}")
print(f"  base hp = {base.hp} (full)  => {OK(blocked and base.hp == base.max_hp)}")

print("=" * 70)
print("TEST 3 — a reactive teleport canNOT intercept this turn's strike")
state = fresh()
base = Base("D", HexCoord(10, 10))
state.add_entity(base)
gap = state.grid.neighbors(HexCoord(10, 10))[0]
bomber = Bomber("A", gap)  # already adjacent (a ring gap)
state.add_entity(bomber)
# defender teleports a reserve Infantry to the bomber's tile-neighbour AND has a
# fighter elsewhere "react" — but movement resolves AFTER attacks, so the strike lands.
reserve = Infantry("D", HexCoord(20, 20))
state.add_entity(reserve)
hp0 = base.hp
run(
    state,
    ActionPayload("A", 0, [AttackAction(unit_id=bomber.id, target=HexCoord(10, 10))]),
    # defender tries to body-block by moving onto the gap this turn (too late)
    ActionPayload("D", 0, [MoveAction(unit_id=reserve.id, path=[HexCoord(20, 20), gap])]),
)
took_hit = base.hp == hp0 - 200
print(f"  base hp {hp0} -> {base.hp};  reactive move can't stop a 200 (=50x4) strike  => {OK(took_hit)}")
print("  (lesson: the ring must be STANDING, not scrambled in reaction)")

print("=" * 70)
print("TEST 4 — redundancy: losing one complete Base is not elimination")
state = fresh()
b_main = Base("D", HexCoord(10, 10))
b_redun = Base("D", HexCoord(25, 5))
state.add_entity(b_main)
state.add_entity(b_redun)
ring = state.grid.neighbors(HexCoord(10, 10))
for c in (ring[0], ring[1]):
    state.add_entity(Bomber("A", c))
run(
    state,
    ActionPayload("A", 0, [AttackAction(unit_id=e.id, target=HexCoord(10, 10))
                           for e in list(state.entities.values()) if isinstance(e, Bomber)]),
    ActionPayload("D", 0, []),
)
diplo = DiplomacyManager()
# elimination is evaluated in phase3; re-run an empty turn to trigger the check
TurnProcessor(state, diplo).process_turn({"D": ActionPayload("D", 0, []), "A": ActionPayload("A", 0, [])})
alive = state.players["D"].alive
remaining = state.count_bases("D")
print(f"  main base destroyed; D still has {remaining} complete Base; alive={alive}  => {OK(alive and remaining == 1)}")
print("=" * 70)
