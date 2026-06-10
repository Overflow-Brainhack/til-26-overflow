"""action dataclasses and validator"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

from engine.hex_grid import HexCoord

if TYPE_CHECKING:
    from engine.state import GameState


# ── action dataclasses ────────────────────────────────────────────────────────


@dataclass
class MoveAction:
    type: Literal["move"] = "move"
    unit_id: str = ""
    path: list[HexCoord] = field(default_factory=list)


@dataclass
class AttackAction:
    type: Literal["attack"] = "attack"
    unit_id: str = ""
    target: HexCoord = HexCoord(0, 0)


@dataclass
class HoldAction:
    type: Literal["hold"] = "hold"
    unit_id: str = ""


@dataclass
class ConstructBuildingAction:
    type: Literal["construct_building"] = "construct_building"
    building_type: str = ""
    coord: HexCoord = HexCoord(0, 0)


@dataclass
class ProduceUnitAction:
    type: Literal["produce_unit"] = "produce_unit"
    building_id: str = ""
    unit_type: str = ""
    target: HexCoord = HexCoord(0, 0)


@dataclass
class ProposeTreatyAction:
    type: Literal["propose_treaty"] = "propose_treaty"
    target_player_id: str = ""
    treaty_type: str = "peace"


@dataclass
class RespondTreatyAction:
    type: Literal["respond_treaty"] = "respond_treaty"
    proposing_player_id: str = ""
    treaty_type: str = "peace"
    accept: bool = False


@dataclass
class BreakTreatyAction:
    type: Literal["break_treaty"] = "break_treaty"
    partner_player_id: str = ""
    treaty_type: str = "peace"


@dataclass
class SendChatAction:
    type: Literal["send_chat"] = "send_chat"
    text: str = ""
    recipient_id: str | None = None  # None = global


AnyAction = Union[
    MoveAction,
    AttackAction,
    HoldAction,
    ConstructBuildingAction,
    ProduceUnitAction,
    ProposeTreatyAction,
    RespondTreatyAction,
    BreakTreatyAction,
    SendChatAction,
]


@dataclass
class ActionPayload:
    player_id: str
    turn_number: int
    actions: list[AnyAction]


# ── deserialisation ───────────────────────────────────────────────────────────

# Case-insensitive type canonicalisation. LLMs routinely emit "mine"/"infantry"
# instead of "Mine"/"Infantry"; left raw, the ActionValidator rejects them
# (the registries are case-sensitive) and the action is silently dropped. We map
# back to the canonical key here — the single chokepoint both the player side
# (parse_actions) and the competition server (payload_from_dict) flow through.
_TYPE_CANON: dict[str, dict[str, str]] = {}


def _canon_type(kind: str, value: object) -> object:
    if not isinstance(value, str):
        return value
    cache = _TYPE_CANON.get(kind)
    if cache is None:
        from engine.constants import BUILDING_STATS, UNIT_STATS

        src = BUILDING_STATS if kind == "building" else UNIT_STATS
        cache = {k.lower(): k for k in src}
        _TYPE_CANON[kind] = cache
    return cache.get(value.strip().lower(), value)


def canonical_type(kind: str, value: object) -> object:
    """Public case-insensitive type canonicaliser (kind = "building" | "unit").
    Used by code paths that construct actions directly without action_from_dict
    (e.g. the super council's translator/roles)."""
    return _canon_type(kind, value)


def _first_str(d: dict, *keys: str) -> str:
    """First non-empty string value among ``keys`` — tolerates the many aliases
    LLMs use for "the other player" (partner_id / player_id / …) since each
    treaty action wants a *different* canonical field name."""
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v:
            return v
    return ""


def _as_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "accept", "y")
    return bool(v)


def _coord(v: object) -> int:
    """Coerce a coordinate to int, raising on null/missing/non-numeric. LLMs
    sometimes emit "target_q": null; left unchecked that builds HexCoord(None, …)
    which survives parsing and then crashes the engine's distance math (a single
    such action took down a whole competition). Raising here makes the bad action
    get dropped at the parse boundary (parse_actions / payload_from_dict)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"invalid coordinate: {v!r}")
    return int(v)


def action_from_dict(d: dict) -> AnyAction:
    t = d.get("type")
    if t == "move":
        return MoveAction(
            unit_id=d["unit_id"],
            path=[HexCoord(_coord(q), _coord(r)) for q, r in d.get("path", [])],
        )
    if t == "attack":
        return AttackAction(
            unit_id=d["unit_id"],
            target=HexCoord(_coord(d["target_q"]), _coord(d["target_r"])),
        )
    if t == "hold":
        return HoldAction(unit_id=d["unit_id"])
    if t == "construct_building":
        bt = _canon_type("building", d["building_type"])
        if not isinstance(bt, str):
            raise ValueError(f"invalid building_type: {bt!r}")
        return ConstructBuildingAction(
            building_type=bt,
            coord=HexCoord(_coord(d["q"]), _coord(d["r"])),
        )
    if t == "produce_unit":
        ut = _canon_type("unit", d["unit_type"])
        if not isinstance(ut, str):
            raise ValueError(f"invalid unit_type: {ut!r}")
        return ProduceUnitAction(
            building_id=d["building_id"],
            unit_type=ut,
            target=HexCoord(_coord(d["target_q"]), _coord(d["target_r"])),
        )
    if t == "propose_treaty":
        return ProposeTreatyAction(
            target_player_id=_first_str(
                d, "target_player_id", "partner_id", "player_id", "partner_player_id"
            ),
            treaty_type=(d.get("treaty_type") or "peace"),
        )
    if t == "respond_treaty":
        return RespondTreatyAction(
            proposing_player_id=_first_str(
                d, "proposing_player_id", "proposer_id", "partner_id", "player_id"
            ),
            treaty_type=(d.get("treaty_type") or "peace"),
            accept=_as_bool(d.get("accept", False)),
        )
    if t == "break_treaty":
        return BreakTreatyAction(
            partner_player_id=_first_str(
                d, "partner_player_id", "partner_id", "player_id", "target_player_id"
            ),
            treaty_type=(d.get("treaty_type") or "peace"),
        )
    if t == "send_chat":
        return SendChatAction(
            text=d["text"],
            recipient_id=d.get("recipient_id", d.get("recipient")),
        )
    raise ValueError(f"unknown action type: {t!r}")


def payload_from_dict(d: dict) -> ActionPayload:
    # Skip individual malformed actions rather than rejecting the whole payload (or,
    # worse, letting a bad one through to crash the engine). The competition server
    # deserialises untrusted player responses through here, so one buggy action from
    # any player must never sink the turn or discard that player's valid actions.
    actions: list[AnyAction] = []
    for a in d.get("actions", []):
        try:
            actions.append(action_from_dict(a))
        except Exception:
            pass
    return ActionPayload(
        player_id=d["player_id"],
        turn_number=d["turn_number"],
        actions=actions,
    )


def action_to_dict(action: AnyAction) -> dict:
    d: dict = {"type": action.type}
    if isinstance(action, MoveAction):
        d["unit_id"] = action.unit_id
        d["path"] = [[c.q, c.r] for c in action.path]
    elif isinstance(action, AttackAction):
        d["unit_id"] = action.unit_id
        d["target_q"] = action.target.q
        d["target_r"] = action.target.r
    elif isinstance(action, HoldAction):
        d["unit_id"] = action.unit_id
    elif isinstance(action, ConstructBuildingAction):
        d["building_type"] = action.building_type
        d["q"] = action.coord.q
        d["r"] = action.coord.r
    elif isinstance(action, ProduceUnitAction):
        d["building_id"] = action.building_id
        d["unit_type"] = action.unit_type
        d["target_q"] = action.target.q
        d["target_r"] = action.target.r
    elif isinstance(action, ProposeTreatyAction):
        d["target_player_id"] = action.target_player_id
        d["treaty_type"] = action.treaty_type
    elif isinstance(action, RespondTreatyAction):
        d["proposing_player_id"] = action.proposing_player_id
        d["treaty_type"] = action.treaty_type
        d["accept"] = action.accept
    elif isinstance(action, BreakTreatyAction):
        d["partner_player_id"] = action.partner_player_id
        d["treaty_type"] = action.treaty_type
    elif isinstance(action, SendChatAction):
        d["text"] = action.text
        d["recipient_id"] = action.recipient_id
    return d


def payload_to_dict(payload: ActionPayload) -> dict:
    return {
        "player_id": payload.player_id,
        "turn_number": payload.turn_number,
        "actions": [action_to_dict(a) for a in payload.actions],
    }


# ── validator ─────────────────────────────────────────────────────────────────


class ActionValidator:
    """validates a single action against the current game state and player knowledge.

    invalid actions are silently converted to no-ops by the turn processor.
    """

    def __init__(self, state: GameState, player_id: str) -> None:
        self._state: GameState = state
        self._player_id = player_id

    def validate_move(self, action: MoveAction) -> bool:
        from engine.entities.unit import GroundUnit, Unit

        entity = self._state.entities.get(action.unit_id)
        if entity is None or entity.owner_id != self._player_id:
            return False
        if not isinstance(entity, Unit):
            return False
        if not isinstance(entity, GroundUnit) and not entity.can_fly:
            return False
        if not action.path:
            return False
        if action.path[0] != entity.coord:
            return False
        if len(action.path) - 1 > entity.movement_range:
            return False
        return True

    def validate_attack(self, action: AttackAction) -> bool:
        from engine.entities.unit import Unit

        attacker = self._state.entities.get(action.unit_id)
        if attacker is None or attacker.owner_id != self._player_id:
            return False
        if not isinstance(attacker, Unit):
            return False
        if attacker.attack_range == 0:
            return False
        dist = self._state.grid.distance(attacker.coord, action.target)
        if dist > attacker.attack_range or dist == 0:
            return False
        return True

    def validate_construct(self, action: ConstructBuildingAction) -> bool:
        from engine.entities import BUILDING_REGISTRY

        if action.building_type not in BUILDING_REGISTRY:
            return False
        stats = __import__(
            "engine.constants", fromlist=["BUILDING_STATS"]
        ).BUILDING_STATS
        bstats = stats.get(action.building_type)
        if bstats is None:
            return False
        player = self._state.players.get(self._player_id)
        if player is None:
            return False
        from engine.resources import ResourceBag

        cost = ResourceBag(gold=bstats.gold_cost)
        if not player.resources.can_afford(cost):
            return False
        coord = self._state.grid.wrap(action.coord)
        # A Base can be founded on any tile the player can SEE (no building blind
        # in fog); every other building must be adjacent to a COMPLETED own
        # building (which is necessarily within vision; under-construction
        # buildings can't anchor).
        if action.building_type == "Base":
            from engine.fog_of_war import compute_visible

            if coord not in compute_visible(self._state, self._player_id):
                return False
        else:
            own_buildings = self._state.buildings_for(self._player_id)
            if not any(
                b.is_complete and self._state.grid.distance(b.coord, action.coord) <= 1
                for b in own_buildings
            ):
                return False
        # target tile must be empty of any entity
        if self._state.entities_at(coord):
            return False
        return True

    def validate_produce(self, action: ProduceUnitAction) -> bool:
        from engine.entities.building import ProductionBuilding
        from engine.resources import ResourceBag

        building = self._state.entities.get(action.building_id)
        if building is None or building.owner_id != self._player_id:
            return False
        if not isinstance(building, ProductionBuilding):
            return False
        if not building.can_produce(action.unit_type):
            return False
        player = self._state.players.get(self._player_id)
        if player is None:
            return False
        from engine.constants import UNIT_STATS

        ustats = UNIT_STATS.get(action.unit_type)
        if ustats is None:
            return False
        cost = ResourceBag(gold=ustats.gold_cost)
        if not player.resources.can_afford(cost):
            return False
        if self._state.grid.distance(building.coord, action.target) > 1:
            return False
        return True
