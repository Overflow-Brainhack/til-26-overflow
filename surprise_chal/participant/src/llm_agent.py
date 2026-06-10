"""TEMPLATE: a bare-minimal LLM agent.

The whole point of this file is to show the *interface*: turn the observation
into a short prompt, ask an LLM for actions as JSON, and turn that JSON back into
an `ActionPayload`. It is deliberately naive — a starting point, not a strong
player. Ideas to improve it (what the strong reference agents do):
  • compute unit moves/attacks in Python (LLMs are bad at hex-range math),
  • give the LLM only economy/diplomacy/chat decisions,
  • split into parallel calls, cache, add a deterministic fallback, etc.

Needs OPENROUTER_API_KEY. (see README)
"""

from __future__ import annotations

from agent_base import PlayerAgent
from engine.actions import ActionPayload, action_from_dict
from llm import call_llm, parse_json

_SYSTEM = """\
You play one team in a 20-player free-for-all hex wargame. Destroy enemy Bases;
be the last team alive. You start with 1 Base and 500 gold. Buildings earn gold; production buildings train units; units fight.

Reply with ONLY a JSON object: {"actions": [ ...actions... ]}. Valid actions:
  {"type":"move","unit_id":"ID","path":[[q,r],[q,r]]}    
  {"type":"attack","unit_id":"ID","target_q":Q,"target_r":R} target within attack_range, distance>=1
  {"type":"hold","unit_id":"ID"}
  {"type":"construct_building","building_type":"Mine|Barracks|Factory|Airbase|Base","q":Q,"r":R}
        Base: any empty tile you can SEE. Others: adjacent to a COMPLETED own building.
  {"type":"produce_unit","building_id":"ID","unit_type":"Infantry|Scout|Medic|Tank|Artillery|Fighter|Bomber","target_q":Q,"target_r":R}
        target must be adjacent to the producing building.
  {"type":"propose_treaty","target_player_id":"PID","treaty_type":"peace"}
  {"type":"respond_treaty","proposing_player_id":"PID","treaty_type":"peace","accept":true}
  {"type":"send_chat","text":"MSG","recipient_id":null}      null = global; or a player id you've met
Costs: Infantry 50, Scout 100, Medic 100, Tank 200, Artillery 200, Fighter 300, Bomber 350;
Mine 200, Barracks 100, Factory 300, Airbase 500, Base 300. Gold can't go negative. Spend it.
Invalid actions are silently dropped — don't worry about emitting a perfect set.
"""


def _brief(obs: dict) -> str:
    pid = obs["player_id"]
    lines = [
        f"TURN {obs.get('turn_number', 0)}/{obs.get('max_turns', '?')}  "
        f"gold={obs.get('resources', {}).get('gold', 0)}  you={pid}  "
        f"map={obs.get('map_width')}x{obs.get('map_height')}",
    ]
    mine_b, mine_u, foe = [], [], []
    for tile in obs.get("visible_tiles", []):
        for e in tile.get("entities", []):
            if e.get("owner_id") == pid:
                if "attack_range" in e:
                    mine_u.append(
                        f"{e['type']} id={e['id']} ({e['q']},{e['r']}) "
                        f"atk_rng={e.get('attack_range')} mv={e.get('movement_range')}"
                    )
                else:
                    mine_b.append(
                        f"{e['type']} id={e['id']} ({e['q']},{e['r']}) "
                        f"{'done' if e.get('is_complete', True) else 'building'}"
                    )
            else:
                foe.append(f"{e['type']}[{e['owner_id']}] ({e['q']},{e['r']})")
    lines.append("MY BUILDINGS: " + ("; ".join(mine_b) or "none"))
    lines.append("MY UNITS: " + ("; ".join(mine_u) or "none"))
    lines.append("VISIBLE ENEMIES: " + ("; ".join(foe[:30]) or "none"))
    lines.append("KNOWN PLAYERS (can DM/propose): " + (", ".join(obs.get("known_players", [])) or "none"))
    return "\n".join(lines)


class LLMAgent(PlayerAgent):
    async def decide(self, observation: dict) -> ActionPayload:
        pid = observation["player_id"]
        turn = observation.get("turn_number", 0)

        reply = await call_llm(_SYSTEM, _brief(observation), max_tokens=700, timeout=9.0)
        data = parse_json(reply)

        actions = []
        for raw in data.get("actions", []):
            try:
                actions.append(action_from_dict(raw))  # engine's hardened parser
            except Exception:
                pass  # skip one malformed action rather than lose the whole turn
        return ActionPayload(player_id=pid, turn_number=turn, actions=actions)
