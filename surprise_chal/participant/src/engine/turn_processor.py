"""turn processor: resolves all three phases of a game turn"""

from __future__ import annotations

from engine.actions import (
    ActionPayload,
    AttackAction,
    BreakTreatyAction,
    ConstructBuildingAction,
    MoveAction,
    ProduceUnitAction,
    ProposeTreatyAction,
    RespondTreatyAction,
)
from engine.constants import (
    ARTILLERY_SPLASH_DAMAGE_RATIO,
    ARTILLERY_SPLASH_RADIUS,
    BUILDING_STATS,
    ELEVATION_ATTACK_BONUS,
    TREATY_BREAK_DELAY_TURNS,
    TREATY_CUTOFF_TURN,
    UNIT_DECAY_PER_TURN,
    UNIT_STATS,
)
from engine.diplomacy import DiplomacyManager, TreatyType
from engine.entities import BUILDING_REGISTRY, UNIT_REGISTRY
from engine.entities.building import Building, ProductionBuilding, ResourceBuilding
from engine.entities.buildings.base_building import Base
from engine.entities.unit import GroundUnit, Unit
from engine.entities.units.artillery import Artillery
from engine.entities.units.bomber import Bomber
from engine.entities.units.medic import Medic
from engine.hex_grid import HexCoord
from engine.resources import ResourceBag
from engine.state import GameState


def _treaty_type_or_none(value: str) -> TreatyType | None:
    """Safe enum lookup for a player-supplied treaty_type. Returns None for an
    empty/unknown value so the action is skipped (a silent no-op) instead of
    crashing the whole turn — an LLM once emitted treaty_type="" and the bare
    TreatyType[''] lookup took down a 20-player game at turn 16."""
    try:
        return TreatyType[value.upper()]
    except (KeyError, AttributeError):
        return None


class TurnProcessor:
    def __init__(self, state: GameState, diplomacy: DiplomacyManager) -> None:
        self.state = state
        self.diplomacy = diplomacy

    def process_turn(self, payloads: dict[str, ActionPayload]) -> list[str]:
        """apply all actions for one turn; returns list of system event strings"""
        events = self._phase1_units(payloads)
        events += self._phase2_buildings(payloads)
        events += self._phase3_coordination(payloads)
        self.state.turn_number += 1
        return events

    # ── phase 1: units ────────────────────────────────────────────────────────

    def _phase1_units(self, payloads: dict[str, ActionPayload]) -> list[str]:
        events: list[str] = []

        # collect all attack actions, grouped by unit
        attack_map: dict[str, AttackAction] = {}
        for payload in payloads.values():
            for action in payload.actions:
                if isinstance(action, AttackAction):
                    attack_map[action.unit_id] = action

        # resolve attacks simultaneously: compute damage first, then apply
        damage_pending: dict[str, int] = {}  # entity_id → total pending damage

        for unit_id, action in attack_map.items():
            attacker = self.state.entities.get(unit_id)
            if attacker is None or not isinstance(attacker, Unit):
                continue
            if attacker.owner_id not in payloads:
                continue

            target_coord = action.target
            dist = self.state.grid.distance(attacker.coord, target_coord)
            if dist > attacker.attack_range or attacker.attack_range == 0:
                continue

            # compute base damage
            power = attacker.attack_power
            if self.state.tile(attacker.coord).is_elevated():
                power = int(power * ELEVATION_ATTACK_BONUS)

            # reject attack if any own or allied unit is on the primary tile
            primary_entities = self.state.entities_at(target_coord)
            if any(
                e.owner_id == attacker.owner_id
                or self.diplomacy.is_peace(attacker.owner_id, e.owner_id)
                for e in primary_entities
            ):
                continue  # invalid — friendly/allied unit on target tile

            # valid attack: damage all entities on primary tile
            for target in primary_entities:
                dmg = power
                if isinstance(attacker, Bomber) and isinstance(target, Building):
                    dmg = int(dmg * Bomber.BUILDING_DAMAGE_MULTIPLIER)
                damage_pending[target.id] = damage_pending.get(target.id, 0) + dmg

            attacker.has_attacked = (
                True  # set for all valid attacks, including empty tile
            )

            # artillery splash — hits everyone in the ring, no friendly filter
            if isinstance(attacker, Artillery):
                for splash_coord in self.state.grid.ring(
                    target_coord, ARTILLERY_SPLASH_RADIUS
                ):
                    for splash_target in self.state.entities_at(splash_coord):
                        splash_dmg = int(power * ARTILLERY_SPLASH_DAMAGE_RATIO)
                        damage_pending[splash_target.id] = (
                            damage_pending.get(splash_target.id, 0) + splash_dmg
                        )

        # apply damage
        for entity_id, dmg in damage_pending.items():
            entity = self.state.entities.get(entity_id)
            if entity:
                entity.take_damage(dmg)

        # apply medic heals (after attacks, medic heals neighbours)
        medic_heals: dict[str, int] = {}  # entity_id → heal amount
        for entity in list(self.state.entities.values()):
            if not isinstance(entity, Medic):
                continue
            for nb_coord in self.state.grid.neighbors(entity.coord):
                for nb_entity in self.state.entities_at(nb_coord):
                    if (
                        isinstance(nb_entity, GroundUnit)
                        and nb_entity.owner_id == entity.owner_id
                        and not isinstance(nb_entity, Medic)
                    ):
                        medic_heals[nb_entity.id] = (
                            medic_heals.get(nb_entity.id, 0) + Medic.HEAL_AMOUNT
                        )
        for entity_id, amount in medic_heals.items():
            entity = self.state.entities.get(entity_id)
            if entity:
                entity.heal(amount)

        # remove dead entities
        dead_ids = [eid for eid, e in self.state.entities.items() if not e.is_alive]
        for eid in dead_ids:
            entity = self.state.entities[eid]
            if isinstance(entity, Base):
                events.append(f"base_{eid}_destroyed_owner_{entity.owner_id}")
            self.state.remove_entity(eid)

        # collect move actions
        move_map: dict[str, MoveAction] = {}
        for payload in payloads.values():
            for action in payload.actions:
                if isinstance(action, MoveAction):
                    move_map[action.unit_id] = action

        # validate and compute final positions for moves
        final_positions: dict[str, HexCoord] = {}
        for unit_id, action in move_map.items():
            entity = self.state.entities.get(unit_id)
            if entity is None or not isinstance(entity, Unit):
                continue
            if not action.path or action.path[0] != entity.coord:
                continue
            if len(action.path) - 1 > entity.movement_range:
                continue
            # validate each step along path (wrap each step so edge tiles are
            # looked up correctly instead of falling back to the default Tile)
            cost = 0
            for step in action.path[1:]:
                tile = self.state.tile(self.state.grid.wrap(step))
                cost += tile.movement_cost()
            if cost > entity.movement_range:
                continue
            final_positions[unit_id] = self.state.grid.wrap(action.path[-1])

        # resolve conflicts — max 1 entity per tile
        destination_count: dict[HexCoord, list[str]] = {}
        for uid, dest in final_positions.items():
            entity = self.state.entities.get(uid)
            if entity and isinstance(entity, Unit):
                destination_count.setdefault(dest, []).append(uid)

        conflicted: set[str] = set()
        for dest, uids in destination_count.items():
            if len(uids) > 1:
                conflicted.update(uids)
            elif len(uids) == 1:
                uid = uids[0]
                entity = self.state.entities.get(uid)
                if entity and self.state.is_ground_blocked(dest, uid):
                    conflicted.add(uid)

        # execute valid moves
        for unit_id, dest in final_positions.items():
            if unit_id in conflicted:
                continue
            entity = self.state.entities.get(unit_id)
            if entity is None or not isinstance(entity, Unit):
                continue
            self.state.move_entity(unit_id, dest)
            entity.has_moved = True

        return events

    # ── phase 2: buildings ────────────────────────────────────────────────────

    def _phase2_buildings(self, payloads: dict[str, ActionPayload]) -> list[str]:
        events: list[str] = []

        # yield resources from completed resource buildings.
        # A DEAD player's buildings go inert: they remain on the map as obstacles
        # (others must destroy them to reclaim the tile) but produce no gold.
        for entity in self.state.entities.values():
            if not isinstance(entity, ResourceBuilding) or not entity.is_complete:
                continue
            player = self.state.players.get(entity.owner_id)
            if player is None or not player.alive:
                continue
            tile = self.state.tile(entity.coord)
            yield_bag = entity.yield_resources(tile.is_rich_resource())
            player.resources += yield_bag

        # tick production queues — skip dead players' buildings (inert: no units).
        # (owner, type, target, building_coord)
        new_units: list[tuple[str, str, HexCoord, HexCoord]] = []
        for entity in list(self.state.entities.values()):
            if not isinstance(entity, ProductionBuilding):
                continue
            owner = self.state.players.get(entity.owner_id)
            if owner is None or not owner.alive:
                continue
            completed = entity.tick_production()
            for unit_type, target in completed:
                new_units.append((entity.owner_id, unit_type, target, entity.coord))

        # place completed units
        for owner_id, unit_type, target, building_coord in new_units:
            cls = UNIT_REGISTRY.get(unit_type)
            if cls is None:
                continue
            # spawn on the target tile, else any free tile adjacent to the BUILDING
            spawn = self._find_spawn_tile(target, building_coord, owner_id)
            if spawn is None:
                continue
            unit = cls(owner_id, spawn)
            self.state.add_entity(unit)
            events.append(f"unit_produced_{unit.id}_{unit_type}_owner_{owner_id}")

        # tick construction of buildings under construction — a dead player's
        # half-built building freezes (it stays an inert obstacle, never completes).
        for entity in list(self.state.entities.values()):
            if not isinstance(entity, Building):
                continue
            owner = self.state.players.get(entity.owner_id)
            if owner is None or not owner.alive:
                continue
            if entity.construction_turns_remaining > 0:
                entity.construction_turns_remaining -= 1

        # Cross-player build collisions resolve like move collisions: if two or
        # more DIFFERENT players target the same tile with a build this turn, ALL
        # of them fail (no one builds, no gold spent) — rather than the lowest
        # player index winning by iteration order. (A single player listing two
        # builds on one tile is unaffected here; their first still claims it via
        # the live entities_at check below.)
        build_targets: dict[HexCoord, set[str]] = {}
        for payload in payloads.values():
            for action in payload.actions:
                if isinstance(action, ConstructBuildingAction):
                    build_targets.setdefault(
                        self.state.grid.wrap(action.coord), set()
                    ).add(payload.player_id)
        contested_tiles = {c for c, owners in build_targets.items() if len(owners) > 1}

        # process construct building actions
        from engine.fog_of_war import compute_visible

        visible_cache: dict[str, set[HexCoord]] = {}  # pid → visible coords (lazy)
        for payload in payloads.values():
            pid = payload.player_id
            player = self.state.players.get(pid)
            if player is None or not player.alive:
                continue
            for action in payload.actions:
                if not isinstance(action, ConstructBuildingAction):
                    continue
                bstats = BUILDING_STATS.get(action.building_type)
                if bstats is None:
                    continue
                cost = ResourceBag(gold=bstats.gold_cost)
                if not player.resources.can_afford(cost):
                    continue
                coord = self.state.grid.wrap(action.coord)
                # contested by 2+ players this turn → all fail (move-collision parity)
                if coord in contested_tiles:
                    continue
                # A Base can be founded on any empty tile you can SEE (this is how
                # you expand to a new region) — you cannot found one blind in fog.
                # Every other building must sit adjacent to a COMPLETED own building
                # (which is necessarily within your own vision), so the explicit
                # vision gate only applies to the build-anywhere Base.
                if action.building_type == "Base":
                    if pid not in visible_cache:
                        visible_cache[pid] = compute_visible(self.state, pid)
                    if coord not in visible_cache[pid]:
                        continue
                else:
                    own_buildings = self.state.buildings_for(pid)
                    if not any(
                        b.is_complete
                        and self.state.grid.distance(b.coord, action.coord) <= 1
                        for b in own_buildings
                    ):
                        continue
                # tile must be empty of any entity
                if self.state.entities_at(coord):
                    continue
                cls = BUILDING_REGISTRY.get(action.building_type)
                if cls is None:
                    continue
                building = cls(pid, coord)
                # a constructed building starts its full build countdown (Base's
                # __init__ pre-completes it for game-start spawns, so set it here)
                building.construction_turns_remaining = bstats.build_turns
                player.resources -= cost
                self.state.add_entity(building)

        # process produce unit actions
        for payload in payloads.values():
            pid = payload.player_id
            player = self.state.players.get(pid)
            if player is None:
                continue
            for action in payload.actions:
                if not isinstance(action, ProduceUnitAction):
                    continue
                building = self.state.entities.get(action.building_id)
                if building is None or building.owner_id != pid:
                    continue
                if not isinstance(building, ProductionBuilding):
                    continue
                if not building.can_produce(action.unit_type):
                    continue
                ustats = UNIT_STATS.get(action.unit_type)
                if ustats is None:
                    continue
                cost = ResourceBag(gold=ustats.gold_cost)
                if not player.resources.can_afford(cost):
                    continue
                if self.state.grid.distance(building.coord, action.target) > 1:
                    continue
                player.resources -= cost
                building.enqueue_unit(
                    action.unit_type, action.target, ustats.build_turns
                )

        return events

    # ── phase 3: coordination ─────────────────────────────────────────────────

    def _phase3_coordination(self, payloads: dict[str, ActionPayload]) -> list[str]:
        events: list[str] = []

        # Treaty cutoff: from TREATY_CUTOFF_TURN onward the back stretch of the game
        # is forced open war — every existing treaty is voided and no propose/accept
        # can form a new one. We void ONCE (the turn the cutoff is first reached),
        # emitting expiry events so both parties are notified; thereafter there are
        # simply no treaties to void and all treaty actions below are skipped.
        treaties_locked = self.state.turn_number >= TREATY_CUTOFF_TURN
        if treaties_locked:
            for treaty in self.diplomacy.void_all():
                events.append(
                    f"treaty_expired_{treaty.treaty_type.name.lower()}_"
                    f"{treaty.proposer_id}_{treaty.partner_id}"
                )

        # process diplomacy actions (suppressed once treaties are locked)
        for payload in payloads.values():
            pid = payload.player_id
            for action in payload.actions:
                if treaties_locked:
                    break  # no diplomacy of any kind past the cutoff
                if isinstance(action, ProposeTreatyAction):
                    tt = _treaty_type_or_none(action.treaty_type)
                    if tt is None:
                        continue
                    # Fog-gate proposals exactly like chat DMs: you may only
                    # propose to a player you have MET (in your known_players —
                    # by sight or prior contact) who is currently alive. A
                    # cold-call/guessed or dead target is a silent no-op and
                    # establishes NO meeting. A delivered proposal lets the
                    # target respond / propose back — they now "know" the
                    # proposer (mirrors a delivered DM adding the sender).
                    proposer = self.state.players.get(pid)
                    target = self.state.players.get(action.target_player_id)
                    if proposer is None or target is None or not target.alive:
                        continue
                    if action.target_player_id not in proposer.known_player_ids:
                        continue
                    if self.diplomacy.propose(pid, action.target_player_id, tt):
                        target.known_player_ids.add(pid)
                elif isinstance(action, RespondTreatyAction):
                    tt = _treaty_type_or_none(action.treaty_type)
                    if tt is None:
                        continue
                    if action.accept:
                        if self.diplomacy.accept(pid, action.proposing_player_id, tt):
                            events.append(
                                f"treaty_formed_{tt.name.lower()}_{pid}_{action.proposing_player_id}"
                            )
                    else:
                        self.diplomacy.reject(pid, action.proposing_player_id, tt)
                elif isinstance(action, BreakTreatyAction):
                    tt = _treaty_type_or_none(action.treaty_type)
                    if tt is None:
                        continue
                    if self.diplomacy.break_treaty(
                        pid, action.partner_player_id, tt, TREATY_BREAK_DELAY_TURNS
                    ):
                        events.append(
                            f"treaty_breaking_{tt.name.lower()}_{pid}_{action.partner_player_id}"
                        )

        # tick treaty break countdowns
        expired = self.diplomacy.tick()
        for treaty in expired:
            events.append(
                f"treaty_expired_{treaty.treaty_type.name.lower()}_"
                f"{treaty.proposer_id}_{treaty.partner_id}"
            )

        # NOTE: "met via chat" is intentionally NOT updated here. A DM only
        # establishes a meeting when it is actually DELIVERED (the game runner adds
        # the sender to the recipient's known set at delivery time). Updating it
        # here from the raw payload would let a DROPPED cold-guess to an unmet
        # player still bootstrap a meeting — which is exactly the loophole we close.

        # check elimination: players with no bases start/continue decay
        for pid, player in self.state.players.items():
            if not player.alive:
                continue
            base_count = self.state.count_bases(pid)
            if base_count == 0:
                if player.decay_turns is None:
                    player.mark_eliminated()
                    events.append(f"player_eliminated_{pid}")

        # apply decay to units of eliminated players
        for entity in list(self.state.entities.values()):
            if not isinstance(entity, Unit):
                continue
            player = self.state.players.get(entity.owner_id)
            if player and not player.alive:
                entity.take_damage(UNIT_DECAY_PER_TURN)
                if not entity.is_alive:
                    self.state.remove_entity(entity.id)

        # update "met" relationships when players see each other's units
        self._update_met_players()

        return events

    # ── helpers ───────────────────────────────────────────────────────────────

    def _find_spawn_tile(
        self, preferred: HexCoord, building_coord: HexCoord, owner_id: str
    ) -> HexCoord | None:
        """Pick the tile a produced unit spawns on, or None if there's no room.

        Spawn on `preferred` (the produce order's target) when it's free. Otherwise
        fall back to any free tile ADJACENT TO THE PRODUCING BUILDING — the building's
        own 6 neighbours, not the target's. A produced unit therefore always appears
        next to the building that made it (never drifting two tiles out), and is only
        lost when the building is fully boxed in (its tile + all 6 neighbours occupied).
        """
        candidate = self.state.grid.wrap(preferred)
        if (
            not self.state.is_ground_blocked(candidate)
            and self.state.grid.distance(
                candidate, self.state.grid.wrap(building_coord)
            )
            <= 1
        ):
            return candidate
        for nb in self.state.grid.neighbors(self.state.grid.wrap(building_coord)):
            if not self.state.is_ground_blocked(nb):
                return nb
        return None

    def _update_met_players(self) -> None:
        """Sight-based "met" is ONE-DIRECTIONAL: the observer learns the owner of
        any entity inside its vision, but the owner does NOT automatically learn
        the observer. Vision ranges differ (a Scout sees far without being seen),
        so meeting must be asymmetric — A seeing B does not mean B has seen A.
        (Replying to a DM is handled separately: delivering a DM adds the sender to
        the recipient's known set in the game runner, so a player can always answer
        someone who reached them even if they never saw them.)
        """
        from engine.fog_of_war import compute_visible

        for pid in self.state.players:
            visible = compute_visible(self.state, pid)
            for entity in self.state.entities.values():
                if entity.owner_id != pid and entity.coord in visible:
                    self.state.players[pid].known_player_ids.add(entity.owner_id)
