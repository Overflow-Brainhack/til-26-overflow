"""The deterministic (no-LLM) Surprise agent — a strong defensive turtle.

This is the robust floor the whole submission stands on: a complete, valid
ActionPayload every turn, computed in milliseconds, with no external calls. It is
also the fallback an LLM overlay would splice diplomacy/chat over.

Architecture (AE-style, see PLAN.md):
  • world.py    — WorldModel: parse the obs + carry cross-turn memory (terrain,
                  last-known enemies, our own production orders, diplomacy).
  • planner.py  — pure functions → a strategic Plan (economy/redundancy/air/stance).
  • actuator.py — all hex math → a validated ActionPayload (rings, builds, attacks).

The strategy is dictated by the load-bearing scoring fact: **binary survival, no
tiebreaker**. So we turtle — base redundancy, universal peace, standing denial
rings, economy to fund them — and treat offense only as defense.

The agent instance persists for the whole game (server.py builds it once), so the
WorldModel accumulates memory across turns; it self-resets if turn_number rewinds
(a fresh game in a reused container).
"""

from __future__ import annotations

from actuator import Actuator
from agent_base import PlayerAgent
from engine.actions import ActionPayload
from planner import plan as make_plan
from world import WorldModel


class AlgoAgent(PlayerAgent):
    def __init__(self) -> None:
        self.world = WorldModel()

    async def decide(self, observation: dict) -> ActionPayload:
        w = self.world
        w.update(observation)
        p = make_plan(w)
        actions = Actuator(w, p).act()
        return ActionPayload(player_id=w.pid, turn_number=w.turn, actions=actions)
