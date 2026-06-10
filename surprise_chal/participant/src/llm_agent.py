"""LLM overlay agent.

The deterministic algo agent is the real player: it emits a complete survival
payload every turn. The LLM is only allowed to add soft-layer actions
(chat/treaty), so a slow or malformed model reply cannot lose the turn.
"""

from __future__ import annotations

from algo_agent import AlgoAgent
from agent_base import PlayerAgent
from engine.actions import (
    ActionPayload,
    ProposeTreatyAction,
    RespondTreatyAction,
    SendChatAction,
    action_from_dict,
)
from llm import call_llm, parse_json

_MAX_CHAT_MESSAGES = 4
_MAX_CHAT_CHARS = 240
_MAX_PROMPT_MESSAGES = 12

_SYSTEM = """\
You are the diplomacy layer for a deterministic survival agent in a 20-player
free-for-all hex wargame. The Python agent already handles all combat, movement,
building, production, and default peace proposals.

Return ONLY JSON: {"actions":[...]}.

Allowed actions:
  {"type":"send_chat","text":"MSG","recipient_id":null}
  {"type":"send_chat","text":"MSG","recipient_id":"player-N"}
  {"type":"propose_treaty","target_player_id":"player-N","treaty_type":"peace"}
  {"type":"respond_treaty","proposing_player_id":"player-N","treaty_type":"peace","accept":true}

Do not emit move, attack, construct_building, produce_unit, hold, or break_treaty.
Default strategic goal: survive to the turn limit. Be concise. Prefer peace.
"""


def _clip(text: object, limit: int) -> str:
    s = "" if text is None else str(text)
    s = " ".join(s.split())
    return s[:limit]


def _message_line(msg: dict) -> str:
    sender = msg.get("sender_id", msg.get("sender", "?"))
    recipient = msg.get("recipient_id", msg.get("recipient"))
    channel = "global" if recipient is None else f"dm->{recipient}"
    return f"{sender} {channel}: {_clip(msg.get('text', ''), 180)}"


def _brief(obs: dict, baseline_count: int) -> str:
    pid = obs["player_id"]
    enemies: list[str] = []
    own_bases: list[str] = []
    damaged: list[str] = []
    for tile in obs.get("visible_tiles", []):
        for e in tile.get("entities", []):
            label = f"{e.get('type')}[{e.get('owner_id')}]@({e.get('q')},{e.get('r')}) hp={e.get('hp')}"
            if e.get("owner_id") == pid:
                if e.get("type") == "Base":
                    status = "complete" if e.get("is_complete") else "building"
                    own_bases.append(f"Base@({e.get('q')},{e.get('r')}) {status} hp={e.get('hp')}")
                if e.get("hp", 0) < e.get("max_hp", e.get("hp", 0)):
                    damaged.append(label)
            else:
                enemies.append(label)

    chats = list(obs.get("private_chat", []))[-_MAX_PROMPT_MESSAGES:]
    chats += list(obs.get("global_chat", []))[-_MAX_PROMPT_MESSAGES:]
    chats = chats[-_MAX_PROMPT_MESSAGES:]

    lines = [
        f"turn={obs.get('turn_number', 0)}/{obs.get('max_turns', '?')} you={pid}",
        f"gold={obs.get('resources', {}).get('gold', 0)} baseline_actions={baseline_count}",
        "known_players=" + (", ".join(obs.get("known_players", [])) or "none"),
        "treaties=" + (", ".join(f"{t.get('partner_id')} break={t.get('breaking_in_turns')}" for t in obs.get("treaties", [])) or "none"),
        "incoming=" + (", ".join(p.get("proposer_id", "?") for p in obs.get("incoming_treaty_proposals", [])) or "none"),
        "own_bases=" + ("; ".join(own_bases) or "none"),
        "damaged_own=" + ("; ".join(damaged[:12]) or "none"),
        "visible_enemies=" + ("; ".join(enemies[:24]) or "none"),
        "recent_chat:",
    ]
    lines.extend(_message_line(m) for m in chats)
    return "\n".join(lines)


class LLMAgent(PlayerAgent):
    def __init__(self) -> None:
        self.algo = AlgoAgent()

    async def decide(self, observation: dict) -> ActionPayload:
        baseline = await self.algo.decide(observation)
        reply = await call_llm(
            _SYSTEM,
            _brief(observation, len(baseline.actions)),
            max_tokens=500,
            timeout=7.0,
        )
        additions = self._soft_actions(parse_json(reply), observation)
        if additions:
            baseline.actions.extend(additions)
        return baseline

    def _soft_actions(self, data: dict, obs: dict) -> list:
        known = set(obs.get("known_players", []))
        incoming = {p.get("proposer_id") for p in obs.get("incoming_treaty_proposals", [])}
        actions: list = []
        for raw in data.get("actions", []):
            if len(actions) >= _MAX_CHAT_MESSAGES:
                break
            try:
                action = action_from_dict(raw)
            except Exception:
                continue

            if isinstance(action, SendChatAction):
                if action.recipient_id is not None and action.recipient_id not in known:
                    continue
                text = _clip(action.text, _MAX_CHAT_CHARS)
                if not text:
                    continue
                actions.append(SendChatAction(text=text, recipient_id=action.recipient_id))
            elif isinstance(action, ProposeTreatyAction):
                if action.target_player_id in known:
                    actions.append(
                        ProposeTreatyAction(
                            target_player_id=action.target_player_id,
                            treaty_type="peace",
                        )
                    )
            elif isinstance(action, RespondTreatyAction):
                # Preserve the survival invariant: the overlay may accept peace, not
                # reject it. Baseline already accepts all incoming proposals.
                if action.proposing_player_id in incoming:
                    actions.append(
                        RespondTreatyAction(
                            proposing_player_id=action.proposing_player_id,
                            treaty_type="peace",
                            accept=True,
                        )
                    )
        return actions
