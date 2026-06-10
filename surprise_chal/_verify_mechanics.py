"""Throwaway verification of suspected code-vs-RULES.md discrepancies.

Runs the REAL engine (participant/src/engine, byte-identical to server's). Delete when done.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "participant", "src"))

from engine.actions import (
    ActionPayload,
    AttackAction,
    BreakTreatyAction,
    MoveAction,
)
from engine.diplomacy import DiplomacyManager, TreatyType
from engine.entities import Base, Infantry
from engine.hex_grid import HexCoord, HexGrid
from engine.player import Player
from engine.resources import ResourceBag
from engine.state import GameState
from engine.turn_processor import TurnProcessor


def make_state():
    grid = HexGrid(35, 30)
    players = {
        "player-0": Player(id="player-0", name="P0", resources=ResourceBag(gold=500)),
        "player-1": Player(id="player-1", name="P1", resources=ResourceBag(gold=500)),
    }
    state = GameState(grid=grid, tiles={}, players=players, entities={})
    # bases so nobody is eliminated mid-test
    state.add_entity(Base("player-0", HexCoord(10, 10)))
    state.add_entity(Base("player-1", HexCoord(25, 20)))
    return state


def pl(pid, state, actions):
    return ActionPayload(pid, state.turn_number, actions)


def empty(pid, state):
    return ActionPayload(pid, state.turn_number, [])


print("=" * 70)
print("TEST A — does an ACTIVE peace treaty block a direct attack? (rules: yes)")
state = make_state()
diplo = DiplomacyManager()
atk = Infantry("player-0", HexCoord(0, 0))
tgt = Infantry("player-1", HexCoord(1, 0))
state.add_entity(atk)
state.add_entity(tgt)
diplo.propose("player-0", "player-1", TreatyType.PEACE)
diplo.accept("player-1", "player-0", TreatyType.PEACE)
proc = TurnProcessor(state, diplo)
before = tgt.hp
proc.process_turn(
    {
        "player-0": pl("player-0", state, [AttackAction(unit_id=atk.id, target=HexCoord(1, 0))]),
        "player-1": empty("player-1", state),
    }
)
print(f"  is_peace={diplo.is_peace('player-0','player-1')}  target hp {before} -> {tgt.hp}")
print(f"  => {'BLOCKED (matches rules)' if tgt.hp == before else 'ATTACK LANDED'}")

print("=" * 70)
print("TEST B — can you attack a BREAKING (counting-down) treaty partner?")
print("         (rules: NO, treaty stays fully ACTIVE for 5 turns)")
state = make_state()
diplo = DiplomacyManager()
atk = Infantry("player-0", HexCoord(0, 0))
tgt = Infantry("player-1", HexCoord(1, 0))
state.add_entity(atk)
state.add_entity(tgt)
diplo.propose("player-0", "player-1", TreatyType.PEACE)
diplo.accept("player-1", "player-0", TreatyType.PEACE)
proc = TurnProcessor(state, diplo)
# turn 1: break + attack in the same payload
before = tgt.hp
proc.process_turn(
    {
        "player-0": pl(
            "player-0",
            state,
            [
                BreakTreatyAction(partner_player_id="player-1", treaty_type="peace"),
                AttackAction(unit_id=atk.id, target=HexCoord(1, 0)),
            ],
        ),
        "player-1": empty("player-1", state),
    }
)
t = diplo.active_treaties_for("player-0")[0]
print(f"  turn of the break: hp {before} -> {tgt.hp}  (attack is phase1, break is phase3 -> still blocked this turn)")
print(f"  treaty now: status={t.status.name}  break_in_turns={t.break_in_turns}  is_peace={diplo.is_peace('player-0','player-1')}")
# turn 2: attack while treaty is visibly BREAKING with turns still on the clock
before2 = tgt.hp
proc.process_turn(
    {
        "player-0": pl("player-0", state, [AttackAction(unit_id=atk.id, target=HexCoord(1, 0))]),
        "player-1": empty("player-1", state),
    }
)
print(f"  next turn (treaty still shows breaking_in_turns>0): hp {before2} -> {tgt.hp}")
print(f"  => {'ATTACK LANDED — discrepancy CONFIRMED' if tgt.hp < before2 else 'blocked (matches rules)'}")

print("=" * 70)
print("TEST C — does a move path require ADJACENT consecutive steps?")
print("         (rules: step-by-step adjacent movement w/ A* + movement budget)")
state = make_state()
diplo = DiplomacyManager()
inf = Infantry("player-0", HexCoord(0, 0))  # movement_range = 1
state.add_entity(inf)
far = HexCoord(15, 7)
d = state.grid.distance(HexCoord(0, 0), far)
proc = TurnProcessor(state, diplo)
proc.process_turn(
    {
        "player-0": pl("player-0", state, [MoveAction(unit_id=inf.id, path=[HexCoord(0, 0), far])]),
        "player-1": empty("player-1", state),
    }
)
print(f"  Infantry(move=1) given path [(0,0) -> (15,7)] which is distance {d}")
print(f"  unit ended at: ({inf.coord.q},{inf.coord.r})")
moved = (inf.coord.q, inf.coord.r) == (far.q, far.r)
print(f"  => {'TELEPORT — no adjacency check, discrepancy CONFIRMED' if moved else 'stayed put (matches rules)'}")

print("=" * 70)
print("TEST C2 — multi-hop teleport budget: how far with a padded path?")
state = make_state()
diplo = DiplomacyManager()
sc = Infantry("player-0", HexCoord(0, 0))  # move 1; try 1 listed hop to a DIFFICULT-cost tile?
state.add_entity(sc)
# a single hop onto an arbitrary normal tile costs 1; two normal hops would need move>=2.
proc = TurnProcessor(state, diplo)
proc.process_turn(
    {
        "player-0": pl("player-0", state, [MoveAction(unit_id=sc.id, path=[HexCoord(0, 0), HexCoord(30, 25), HexCoord(3, 18)])]),
        "player-1": empty("player-1", state),
    }
)
print(f"  Infantry(move=1) given a 2-hop path (len-1=2 > move 1) -> should be REJECTED")
print(f"  unit ended at: ({sc.coord.q},{sc.coord.r}) (expect 0,0 — budget still caps #hops)")

print("=" * 70)
print("TEST D — can a unit move into a tile being vacated by another unit same turn?")
print("         (rules 'Move Collision': blocked only if occupant is NOT vacating)")
state = make_state()
diplo = DiplomacyManager()
X = Infantry("player-0", HexCoord(1, 0))  # moves (1,0)->(2,0)
Y = Infantry("player-0", HexCoord(0, 0))  # moves (0,0)->(1,0)  (X's old tile)
state.add_entity(X)
state.add_entity(Y)
proc = TurnProcessor(state, diplo)
proc.process_turn(
    {
        "player-0": pl(
            "player-0",
            state,
            [
                MoveAction(unit_id=X.id, path=[HexCoord(1, 0), HexCoord(2, 0)]),
                MoveAction(unit_id=Y.id, path=[HexCoord(0, 0), HexCoord(1, 0)]),
            ],
        ),
        "player-1": empty("player-1", state),
    }
)
print(f"  X ended at ({X.coord.q},{X.coord.r}) (expect 2,0)")
print(f"  Y ended at ({Y.coord.q},{Y.coord.r}) (expect 1,0 if follow allowed, 0,0 if blocked)")
followed = (Y.coord.q, Y.coord.r) == (1, 0)
print(f"  => {'follow ALLOWED (matches rules)' if followed else 'follow BLOCKED — discrepancy (no same-turn follow/swap)'}")
print("=" * 70)
