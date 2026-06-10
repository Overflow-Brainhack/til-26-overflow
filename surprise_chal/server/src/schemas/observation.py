"""observation payload: what is sent to each player server each turn"""

from __future__ import annotations

from engine.chat import ChatLog
from engine.constants import MAX_TURNS, TREATY_CUTOFF_TURN
from engine.diplomacy import DiplomacyManager
from engine.fog_of_war import filter_observation
from engine.state import GameState

# NOTE: chat is shipped UNCAPPED in the observation — the full cumulative
# `global_chat` (and the player's full `private_chat`) goes out every turn. A
# byte budget was tried but rejected: trimming to fit silently dropped other
# players' legitimate messages whenever a length-DoS flooder filled the window.
# We'd rather deliver everything — an oversized payload simply risks timing the
# reader out (the intended consequence of a length-DoS), and no honest message
# is lost. Replay size is bounded separately, per-message, in recorder.py.


def build_observation(
    state: GameState,
    player_id: str,
    diplomacy: DiplomacyManager,
    chat_log: ChatLog,
    max_turns: int = MAX_TURNS,
) -> dict:
    """build the full observation JSON payload for a single player.

    `max_turns` is the game's configured turn limit (the competition's flag). It is
    shipped so agents can size their endgame relative to the REAL deadline instead of
    assuming the default limit — defaults to MAX_TURNS for callers that don't pass it.
    """
    visible_coords, filtered_entities = filter_observation(state, player_id)
    player = state.players[player_id]

    visible_tiles = []
    for coord in sorted(visible_coords, key=lambda c: (c.r, c.q)):
        tile = state.tile(coord)
        entity_dicts = [
            filtered_entities[eid]
            for eid in (state.coord_index.get(coord) or [])
            if eid in filtered_entities
        ]
        visible_tiles.append(
            {
                "q": coord.q,
                "r": coord.r,
                "terrain": tile.terrain.name.lower(),
                "entities": entity_dicts,
            }
        )

    # Past the treaty cutoff the diplomacy channel is closed (treaties voided, none
    # can form), so the agent always sees empty treaty/proposal lists from that turn
    # on — consistent with the engine ignoring any treaty action it submits.
    treaties_open = state.turn_number < TREATY_CUTOFF_TURN

    treaty_dicts = (
        [
            {
                "partner_id": (
                    t.partner_id if t.proposer_id == player_id else t.proposer_id
                ),
                "treaty_type": t.treaty_type.name.lower(),
                "breaking_in_turns": t.break_in_turns,
            }
            for t in diplomacy.active_treaties_for(player_id)
        ]
        if treaties_open
        else []
    )

    proposal_dicts = (
        [
            {
                "proposer_id": t.proposer_id,
                "treaty_type": t.treaty_type.name.lower(),
            }
            for t in diplomacy.incoming_proposals_for(player_id)
        ]
        if treaties_open
        else []
    )

    return {
        # external state (same for all players)
        "turn_number": state.turn_number,
        "max_turns": max_turns,
        "map_width": state.grid.width,
        "map_height": state.grid.height,
        "global_chat": [m.to_dict() for m in chat_log.global_messages()],
        # internal state (per player)
        "player_id": player_id,
        "resources": player.resources.to_dict(),
        "visible_tiles": visible_tiles,
        "treaties": treaty_dicts,
        "incoming_treaty_proposals": proposal_dicts,
        "private_chat": [m.to_dict() for m in chat_log.private_messages_for(player_id)],
        "known_players": sorted(player.known_player_ids),
    }
