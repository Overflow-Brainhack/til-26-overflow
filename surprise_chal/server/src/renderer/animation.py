"""turn transition data and builders for animated rendering"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.hex_grid import HexCoord


@dataclass
class MoveEvent:
    entity_id: str
    owner_id: str
    entity_type: str
    palette_idx: int
    from_coord: HexCoord
    path: list[HexCoord]  # full attempted path; path[-1] == to_coord if succeeded
    to_coord: HexCoord  # == from_coord if failed
    succeeded: bool


@dataclass
class AttackEvent:
    attacker_id: str
    attacker_owner: str
    attacker_coord: HexCoord  # position before moves (phase 1 fires before moves)
    target_coord: HexCoord
    attacker_type: str
    succeeded: bool  # True if any entity at target_coord lost HP this turn
    is_splash: bool  # True for Artillery


@dataclass
class DeathEvent:
    entity_id: str
    owner_id: str
    entity_type: str
    coord: HexCoord
    palette_idx: int


@dataclass
class SpawnEvent:
    entity_id: str
    owner_id: str
    entity_type: str
    coord: HexCoord
    palette_idx: int


@dataclass
class TurnTransition:
    from_turn: int
    from_entities: dict[str, dict]  # prev state_snapshot["entities"]
    to_entities: dict[str, dict]  # next state_snapshot["entities"]
    moves: list[MoveEvent]
    attacks: list[AttackEvent]
    deaths: list[DeathEvent]
    spawns: list[SpawnEvent]
    # (coord, is_lethal) — tile flashes from build_transition_from_diff (no beam origin)
    damage_flashes: list[tuple[HexCoord, bool]] = field(default_factory=list)
    duration: float = 0.6


def _coord(e: dict) -> HexCoord:
    return HexCoord(e["q"], e["r"])


def _entity_at_coord(entities: dict[str, dict], coord: HexCoord) -> dict | None:
    for e in entities.values():
        if HexCoord(e["q"], e["r"]) == coord:
            return e
    return None


def build_transition(
    prev_record: dict,
    next_record: dict,
    palette_map: dict[str, int],
) -> TurnTransition:
    """Build a TurnTransition using full action data (replay viewer path).

    prev_record = record[N-1], next_record = record[N] (which contains the actions
    that produced its state_snapshot from prev_record's state_snapshot).
    """
    from_entities: dict[str, dict] = (prev_record.get("state_snapshot") or {}).get(
        "entities", {}
    )
    to_entities: dict[str, dict] = (next_record.get("state_snapshot") or {}).get(
        "entities", {}
    )

    from_turn: int = prev_record.get("turn", 0)
    died_ids = set(from_entities) - set(to_entities)
    spawned_ids = set(to_entities) - set(from_entities)

    deaths: list[DeathEvent] = []
    for eid in died_ids:
        e = from_entities[eid]
        deaths.append(
            DeathEvent(
                entity_id=eid,
                owner_id=e.get("owner_id", ""),
                entity_type=e.get("type", ""),
                coord=_coord(e),
                palette_idx=palette_map.get(e.get("owner_id", ""), 0),
            )
        )

    spawns: list[SpawnEvent] = []
    for eid in spawned_ids:
        e = to_entities[eid]
        spawns.append(
            SpawnEvent(
                entity_id=eid,
                owner_id=e.get("owner_id", ""),
                entity_type=e.get("type", ""),
                coord=_coord(e),
                palette_idx=palette_map.get(e.get("owner_id", ""), 0),
            )
        )

    # collect all actions across players; deduplicate move actions by unit_id
    all_actions: list[dict] = []
    seen_move_ids: set[str] = set()
    for payload in next_record.get("actions", {}).values():
        for act in payload.get("actions", []):
            act_type = act.get("type", "")
            if act_type == "move":
                uid = act.get("unit_id", "")
                if uid in seen_move_ids:
                    continue
                seen_move_ids.add(uid)
            all_actions.append(act)

    moves: list[MoveEvent] = []
    for act in all_actions:
        if act.get("type") != "move":
            continue
        uid = act.get("unit_id", "")
        raw_path = act.get("path") or []
        if not raw_path or uid not in from_entities:
            continue
        if uid in died_ids:
            continue  # DeathEvent takes priority
        path = [HexCoord(int(p[0]), int(p[1])) for p in raw_path]
        from_e = from_entities[uid]
        from_coord = _coord(from_e)
        if uid in to_entities:
            to_coord = _coord(to_entities[uid])
            succeeded = to_coord != from_coord
        else:
            to_coord = from_coord
            succeeded = False
        moves.append(
            MoveEvent(
                entity_id=uid,
                owner_id=from_e.get("owner_id", ""),
                entity_type=from_e.get("type", ""),
                palette_idx=palette_map.get(from_e.get("owner_id", ""), 0),
                from_coord=from_coord,
                path=path,
                to_coord=to_coord,
                succeeded=succeeded,
            )
        )

    attacks: list[AttackEvent] = []
    for act in all_actions:
        if act.get("type") != "attack":
            continue
        uid = act.get("unit_id", "")
        if uid not in from_entities:
            continue
        tq = act.get("target_q")
        tr = act.get("target_r")
        if tq is None or tr is None:
            continue
        attacker_e = from_entities[uid]
        attacker_coord = _coord(attacker_e)
        target_coord = HexCoord(int(tq), int(tr))

        # determine success: any entity at target lost HP
        succeeded = False
        for eid, fe in from_entities.items():
            if _coord(fe) == target_coord:
                te = to_entities.get(eid)
                if te is None or te.get("hp", 0) < fe.get("hp", 0):
                    succeeded = True
                    break

        attacks.append(
            AttackEvent(
                attacker_id=uid,
                attacker_owner=attacker_e.get("owner_id", ""),
                attacker_coord=attacker_coord,
                target_coord=target_coord,
                attacker_type=attacker_e.get("type", ""),
                succeeded=succeeded,
                is_splash=(attacker_e.get("type", "") == "Artillery"),
            )
        )

    return TurnTransition(
        from_turn=from_turn,
        from_entities=from_entities,
        to_entities=to_entities,
        moves=moves,
        attacks=attacks,
        deaths=deaths,
        spawns=spawns,
    )


def build_transition_from_diff(
    from_entities: dict[str, dict],
    to_entities: dict[str, dict],
    palette_map: dict[str, int],
    from_turn: int = 0,
) -> TurnTransition:
    """Build a TurnTransition from state diff only (live window path, no action data).

    No failed moves or attack beams — only successful position changes and damage flashes.
    """
    died_ids = set(from_entities) - set(to_entities)
    spawned_ids = set(to_entities) - set(from_entities)

    deaths: list[DeathEvent] = []
    for eid in died_ids:
        e = from_entities[eid]
        deaths.append(
            DeathEvent(
                entity_id=eid,
                owner_id=e.get("owner_id", ""),
                entity_type=e.get("type", ""),
                coord=_coord(e),
                palette_idx=palette_map.get(e.get("owner_id", ""), 0),
            )
        )

    spawns: list[SpawnEvent] = []
    for eid in spawned_ids:
        e = to_entities[eid]
        spawns.append(
            SpawnEvent(
                entity_id=eid,
                owner_id=e.get("owner_id", ""),
                entity_type=e.get("type", ""),
                coord=_coord(e),
                palette_idx=palette_map.get(e.get("owner_id", ""), 0),
            )
        )

    moves: list[MoveEvent] = []
    for eid, fe in from_entities.items():
        if eid in died_ids:
            continue
        te = to_entities.get(eid)
        if te is None:
            continue
        fc = _coord(fe)
        tc = _coord(te)
        if fc != tc:
            moves.append(
                MoveEvent(
                    entity_id=eid,
                    owner_id=fe.get("owner_id", ""),
                    entity_type=fe.get("type", ""),
                    palette_idx=palette_map.get(fe.get("owner_id", ""), 0),
                    from_coord=fc,
                    path=[tc],
                    to_coord=tc,
                    succeeded=True,
                )
            )

    # damage flashes: any entity that lost HP (and survived)
    damage_flashes: list[tuple[HexCoord, bool]] = []
    seen_coords: set[HexCoord] = set()
    for eid, fe in from_entities.items():
        te = to_entities.get(eid)
        if te is None:
            continue
        if te.get("hp", 0) < fe.get("hp", 0):
            coord = _coord(fe)
            if coord not in seen_coords:
                seen_coords.add(coord)
                is_lethal = te.get("hp", 0) <= 0
                damage_flashes.append((coord, is_lethal))

    return TurnTransition(
        from_turn=from_turn,
        from_entities=from_entities,
        to_entities=to_entities,
        moves=moves,
        attacks=[],
        deaths=deaths,
        spawns=spawns,
        damage_flashes=damage_flashes,
    )
