"""Deterministic survival-first agent for the Surprise challenge.

The scoring rule makes survival the main objective: every player alive at the
turn limit co-wins. This agent therefore builds redundant Bases, accepts peace,
keeps a defensive army around its Bases, and only pushes when visible enemies are
close enough to threaten us.
"""

from __future__ import annotations

from agent_base import PlayerAgent
from engine.actions import (
    ActionPayload,
    AttackAction,
    ConstructBuildingAction,
    MoveAction,
    ProduceUnitAction,
    ProposeTreatyAction,
    RespondTreatyAction,
)
from engine.constants import BUILDING_STATS, UNIT_STATS
from engine.hex_grid import HexCoord, HexGrid

_PRODUCTION_BUILDINGS = ("Barracks", "Factory", "Airbase")
_TREATY_CUTOFF_TURN = 200
_AIR_THREATS = {"Airbase", "Bomber", "Fighter"}
_HIGH_VALUE_TARGETS = {"Base", "Airbase", "Factory", "Bomber", "Artillery"}


def _coord(entity: dict) -> HexCoord:
    return HexCoord(entity["q"], entity["r"])


def _is_building(entity: dict) -> bool:
    return entity.get("type") in BUILDING_STATS


def _flatten(observation: dict, pid: str):
    own_units, own_buildings, enemies = [], [], []
    occupied: set[tuple[int, int]] = set()
    terrain: dict[tuple[int, int], str] = {}
    visible: list[tuple[int, int]] = []

    for tile in observation.get("visible_tiles", []):
        pos = (tile["q"], tile["r"])
        visible.append(pos)
        terrain[pos] = tile.get("terrain", "normal")
        for entity in tile.get("entities", []):
            occupied.add((entity["q"], entity["r"]))
            if entity.get("owner_id") == pid:
                (own_buildings if _is_building(entity) else own_units).append(entity)
            else:
                enemies.append(entity)
    return own_units, own_buildings, enemies, occupied, terrain, visible


class AlgoAgent(PlayerAgent):
    def __init__(self) -> None:
        self._last_proposed: dict[str, int] = {}
        self._seen_enemy_bases: dict[str, HexCoord] = {}
        self._seen_enemy_airbases: dict[str, HexCoord] = {}
        self._seen_rich_tiles: set[HexCoord] = set()
        self._own_base_sites: dict[tuple[int, int], int] = {}
        self._all_coords_cache: dict[tuple[int, int], list[HexCoord]] = {}
        self._remote_probe_index = 0

    async def decide(self, observation: dict) -> ActionPayload:
        pid = observation["player_id"]
        turn = observation.get("turn_number", 0)
        max_turns = observation.get("max_turns", 300)
        gold = observation.get("resources", {}).get("gold", 0)
        grid = HexGrid(
            observation.get("map_width", 35), observation.get("map_height", 30)
        )
        own_units, own_buildings, enemies, occupied, terrain, visible = _flatten(
            observation, pid
        )
        self._update_memory(enemies, terrain)
        self._update_own_base_memory(turn, own_buildings, visible)

        actions: list = []
        actions.extend(self._diplomacy_actions(observation, turn))

        complete = [b for b in own_buildings if b.get("is_complete", True)]
        bases = [b for b in own_buildings if b["type"] == "Base"]
        complete_bases = [b for b in bases if b.get("is_complete", True)]
        pending_bases = [b for b in bases if not b.get("is_complete", True)]
        preassigned_units: set[str] = set()

        gold = self._plan_remote_base_probe(
            grid,
            turn,
            max_turns,
            gold,
            own_units,
            complete_bases,
            pending_bases,
            enemies,
            occupied,
            actions,
            preassigned_units,
        )

        gold = self._plan_builds(
            grid,
            turn,
            max_turns,
            gold,
            complete,
            own_buildings,
            complete_bases,
            pending_bases,
            enemies,
            occupied,
            terrain,
            visible,
            actions,
        )
        gold = self._plan_production(
            grid, turn, gold, own_units, complete, enemies, occupied, actions
        )
        self._plan_unit_actions(
            grid,
            turn,
            own_units,
            complete_bases or complete,
            enemies,
            occupied,
            terrain,
            actions,
            preassigned_units,
        )

        return ActionPayload(player_id=pid, turn_number=turn, actions=actions)

    def _update_memory(
        self, enemies: list[dict], terrain: dict[tuple[int, int], str]
    ) -> None:
        for enemy in enemies:
            if enemy["type"] == "Base":
                self._seen_enemy_bases[enemy["owner_id"]] = _coord(enemy)
            elif enemy["type"] == "Airbase":
                self._seen_enemy_airbases[enemy["owner_id"]] = _coord(enemy)
        for (q, r), kind in terrain.items():
            if kind == "rich_resource":
                self._seen_rich_tiles.add(HexCoord(q, r))

    def _update_own_base_memory(
        self, turn: int, own_buildings: list[dict], visible: list[tuple[int, int]]
    ) -> None:
        own_base_coords = set()
        for building in own_buildings:
            if building["type"] != "Base":
                continue
            pos = (building["q"], building["r"])
            own_base_coords.add(pos)
            complete_turn = turn
            if not building.get("is_complete", True):
                complete_turn = turn + building.get("construction_turns_remaining", 0)
            self._own_base_sites[pos] = min(self._own_base_sites.get(pos, complete_turn), complete_turn)

        visible_set = set(visible)
        for pos in list(self._own_base_sites):
            if pos in visible_set and pos not in own_base_coords:
                del self._own_base_sites[pos]

    def _remember_base_order(self, coord: HexCoord, turn: int) -> None:
        self._own_base_sites[(coord.q, coord.r)] = min(
            self._own_base_sites.get((coord.q, coord.r), turn + 5), turn + 5
        )

    def _remembered_base_total(
        self, complete_bases: list[dict], pending_bases: list[dict]
    ) -> int:
        observed = {(_coord(b).q, _coord(b).r) for b in complete_bases + pending_bases}
        return len(set(self._own_base_sites) | observed)

    def _diplomacy_actions(self, observation: dict, turn: int) -> list:
        if turn >= _TREATY_CUTOFF_TURN:
            return []

        actions: list = []
        treaty_partners = {
            t.get("partner_id")
            for t in observation.get("treaties", [])
            if t.get("partner_id") and t.get("breaking_in_turns") is None
        }
        in_break = {
            t.get("partner_id")
            for t in observation.get("treaties", [])
            if t.get("partner_id") and t.get("breaking_in_turns") is not None
        }

        for proposal in observation.get("incoming_treaty_proposals", []):
            proposer = proposal.get("proposer_id")
            if proposer:
                actions.append(
                    RespondTreatyAction(
                        proposing_player_id=proposer,
                        treaty_type=proposal.get("treaty_type", "peace"),
                        accept=True,
                    )
                )

        for player_id in observation.get("known_players", []):
            if player_id in treaty_partners or player_id in in_break:
                continue
            # Re-propose occasionally, but do not burn action volume every turn.
            if turn - self._last_proposed.get(player_id, -1000) < 15:
                continue
            actions.append(
                ProposeTreatyAction(target_player_id=player_id, treaty_type="peace")
            )
            self._last_proposed[player_id] = turn

        return actions

    def _plan_remote_base_probe(
        self,
        grid: HexGrid,
        turn: int,
        max_turns: int,
        gold: int,
        own_units: list[dict],
        complete_bases: list[dict],
        pending_bases: list[dict],
        enemies: list[dict],
        occupied: set[tuple[int, int]],
        actions: list,
        preassigned_units: set[str],
    ) -> int:
        early_emergency = (
            turn >= 15
            and len(complete_bases) >= 2
            and len(own_units) >= 6
            and self._remembered_base_total(complete_bases, pending_bases) < 3
        )
        late_spread = turn >= 60 and len(own_units) >= 14 and len(complete_bases) >= 2
        if not early_emergency and not late_spread:
            return gold
        desired_bases = 3
        if turn >= 80 or gold >= 1200:
            desired_bases = 4
        if turn >= min(180, max_turns * 2 // 3) or gold >= 2500:
            desired_bases = 5
        observed_total = len(complete_bases) + len(pending_bases)
        remembered_total = self._remembered_base_total(complete_bases, pending_bases)
        if len(complete_bases) >= 3 and remembered_total >= desired_bases:
            return gold
        if observed_total >= desired_bases:
            return gold
        if gold < BUILDING_STATS["Base"].gold_cost:
            return gold

        scouts = [u for u in own_units if u["type"] == "Scout"]
        if not scouts:
            return gold
        scout = scouts[turn % len(scouts)]
        if scout["id"] in preassigned_units:
            return gold

        target = self._remote_probe_target(grid, turn, complete_bases, enemies, occupied)
        if target is None:
            return gold
        base_site = self._remote_base_site(grid, target, occupied)
        if base_site is None:
            return gold

        here = _coord(scout)
        actions.append(MoveAction(unit_id=scout["id"], path=[here, target]))
        actions.append(ConstructBuildingAction(building_type="Base", coord=base_site))
        self._remember_base_order(base_site, turn)
        preassigned_units.add(scout["id"])
        occupied.add((target.q, target.r))
        occupied.add((base_site.q, base_site.r))
        self._remote_probe_index += 1
        return gold - BUILDING_STATS["Base"].gold_cost

    def _plan_builds(
        self,
        grid: HexGrid,
        turn: int,
        max_turns: int,
        gold: int,
        complete: list[dict],
        own_buildings: list[dict],
        complete_bases: list[dict],
        pending_bases: list[dict],
        enemies: list[dict],
        occupied: set[tuple[int, int]],
        terrain: dict[tuple[int, int], str],
        visible: list[tuple[int, int]],
        actions: list,
    ) -> int:
        if not complete:
            return gold

        def count(kind: str) -> int:
            return sum(1 for b in own_buildings if b["type"] == kind)

        # Survival first: redundant Bases early, then extra hidden spread later.
        desired_bases = 3
        if turn >= 80 or gold >= 1200:
            desired_bases = 4
        if turn >= min(180, max_turns * 2 // 3) or gold >= 2500:
            desired_bases = 5
        if turn >= 240 or gold >= 5000:
            desired_bases = 6
        observed_total = len(complete_bases) + len(pending_bases)
        remembered_total = self._remembered_base_total(complete_bases, pending_bases)
        base_shortage = len(complete_bases) < 3 or (
            observed_total < desired_bases and remembered_total < desired_bases
        )
        if base_shortage and gold >= 300:
            spot = self._base_site(
                grid, complete_bases or complete, enemies, occupied, terrain, visible
            )
            if spot is not None:
                actions.append(
                    ConstructBuildingAction(building_type="Base", coord=spot)
                )
                self._remember_base_order(spot, turn)
                occupied.add((spot.q, spot.r))
                gold -= BUILDING_STATS["Base"].gold_cost

        build_queue: list[str] = []

        def planned_count(kind: str) -> int:
            return count(kind) + build_queue.count(kind)

        barracks_target = 1
        if turn >= 60 or gold >= 1000:
            barracks_target = 2
        while planned_count("Barracks") < barracks_target:
            build_queue.append("Barracks")

        if planned_count("Mine") < 1:
            build_queue.append("Mine")

        factory_target = 0
        if turn >= 12 or count("Mine") >= 1:
            factory_target = 1
        if turn >= 80 or gold >= 1000:
            factory_target = 2
        if turn >= 180 or gold >= 2500:
            factory_target = 3
        while planned_count("Factory") < factory_target:
            build_queue.append("Factory")

        mine_target = 3 if turn < 80 else 6
        if gold >= 1500:
            mine_target = 8
        while planned_count("Mine") < mine_target:
            build_queue.append("Mine")

        airbase_target = 0
        if turn >= 45 or gold >= 900:
            airbase_target = 1
        air_threat = self._has_air_threat(enemies)
        if turn >= 80 or gold >= 1400 or air_threat:
            airbase_target = 2
        if turn >= 160 or gold >= 2800:
            airbase_target = 3
        while planned_count("Airbase") < airbase_target:
            build_queue.append("Airbase")

        # Add at most a few buildings per turn so production still gets funded.
        build_limit = 4 if gold < 1500 else 6
        for want in build_queue[:build_limit]:
            cost = BUILDING_STATS[want].gold_cost
            reserve = 150 if want not in ("Barracks", "Mine") else 50
            remembered_bases = max(observed_total, remembered_total)
            if (
                base_shortage
                and remembered_bases < 3
                and want not in ("Barracks", "Factory")
                and gold < cost + 300
            ):
                continue
            if gold < cost + reserve:
                continue
            spot = self._anchored_site(grid, want, complete, enemies, occupied, terrain)
            if spot is None:
                continue
            actions.append(
                ConstructBuildingAction(building_type=want, coord=spot)
            )
            occupied.add((spot.q, spot.r))
            gold -= cost

        return gold

    def _plan_production(
        self,
        grid: HexGrid,
        turn: int,
        gold: int,
        own_units: list[dict],
        complete_buildings: list[dict],
        enemies: list[dict],
        occupied: set[tuple[int, int]],
        actions: list,
    ) -> int:
        counts: dict[str, int] = {}
        for unit in own_units:
            counts[unit["type"]] = counts.get(unit["type"], 0) + 1

        complete_bases = sum(1 for b in complete_buildings if b["type"] == "Base")
        throttle_barracks = turn < 45 and complete_bases < 3 and len(own_units) >= 6 and gold < 500

        producers = [b for b in complete_buildings if b["type"] in _PRODUCTION_BUILDINGS]
        producers.sort(key=lambda b: ("Factory", "Airbase", "Barracks").index(b["type"]))

        for building in producers:
            if throttle_barracks and building["type"] == "Barracks":
                continue
            free = self._free_neighbors(grid, (building["q"], building["r"]), occupied)
            if not free:
                continue

            wanted = self._wanted_units(
                building["type"], counts, turn, gold, self._has_air_threat(enemies)
            )
            if building["type"] == "Barracks":
                per_building_limit = 3 if gold >= 600 else 2
            else:
                per_building_limit = 2 if gold >= 1000 else 1
            for unit_type in wanted[:per_building_limit]:
                if not free:
                    break
                cost = UNIT_STATS[unit_type].gold_cost
                if gold < cost:
                    continue
                q, r = free.pop(0)
                actions.append(
                    ProduceUnitAction(
                        building_id=building["id"],
                        unit_type=unit_type,
                        target=HexCoord(q, r),
                    )
                )
                occupied.add((q, r))
                counts[unit_type] = counts.get(unit_type, 0) + 1
                gold -= cost

        return gold

    def _wanted_units(
        self,
        building_type: str,
        counts: dict[str, int],
        turn: int,
        gold: int,
        air_threat: bool,
    ) -> list[str]:
        if building_type == "Barracks":
            if counts.get("Scout", 0) < 2:
                return ["Scout", "Infantry"]
            if counts.get("Medic", 0) * 5 < counts.get("Infantry", 0) + counts.get(
                "Tank", 0
            ):
                return ["Medic", "Infantry"]
            return ["Infantry", "Infantry"]

        if building_type == "Factory":
            artillery = counts.get("Artillery", 0)
            tanks = counts.get("Tank", 0)
            if tanks < 2:
                return ["Tank"]
            if artillery < tanks:
                return ["Artillery", "Tank"]
            return ["Tank", "Artillery"]

        if building_type == "Airbase":
            fighter_target = 4 if air_threat else 2
            if turn >= 160 or gold >= 2000:
                fighter_target += 2
            if counts.get("Fighter", 0) < fighter_target or gold < 800:
                return ["Fighter"]
            return ["Fighter", "Bomber"]

        return []

    def _plan_unit_actions(
        self,
        grid: HexGrid,
        turn: int,
        own_units: list[dict],
        anchors: list[dict],
        enemies: list[dict],
        occupied: set[tuple[int, int]],
        terrain: dict[tuple[int, int], str],
        actions: list,
        preassigned_units: set[str],
    ) -> None:
        move_reserved = set(occupied)
        for unit in sorted(own_units, key=lambda u: self._unit_move_order(u["type"])):
            if unit["id"] in preassigned_units:
                continue
            here = _coord(unit)
            target = self._best_attack_target(grid, unit, enemies)
            if target is not None:
                actions.append(AttackAction(unit_id=unit["id"], target=_coord(target)))

            destination = self._movement_goal(
                grid, turn, unit, own_units, anchors, enemies, move_reserved, terrain
            )
            if destination is None:
                continue
            path = self._teleport_path(
                grid,
                here,
                destination,
                unit.get("movement_range", 0),
                move_reserved,
                terrain,
            )
            if path and len(path) > 1:
                actions.append(MoveAction(unit_id=unit["id"], path=path))
                move_reserved.discard((here.q, here.r))
                move_reserved.add((path[-1].q, path[-1].r))

    @staticmethod
    def _unit_move_order(unit_type: str) -> int:
        return {
            "Scout": 0,
            "Fighter": 1,
            "Tank": 2,
            "Infantry": 3,
            "Medic": 4,
            "Artillery": 5,
            "Bomber": 6,
        }.get(unit_type, 9)

    def _best_attack_target(
        self, grid: HexGrid, unit: dict, enemies: list[dict]
    ) -> dict | None:
        attack_range = unit.get("attack_range", 0)
        if attack_range < 1:
            return None
        here = _coord(unit)
        in_range = [
            e
            for e in enemies
            if 0 < grid.distance(here, _coord(e)) <= attack_range
        ]
        if not in_range:
            return None

        def score(enemy: dict) -> tuple[int, int, int]:
            kind = enemy["type"]
            priority = {
                "Bomber": 0,
                "Artillery": 1,
                "Fighter": 2,
                "Tank": 3,
                "Base": 4,
                "Factory": 5,
                "Airbase": 6,
                "Infantry": 7,
                "Scout": 8,
                "Mine": 9,
                "Barracks": 10,
                "Medic": 11,
            }.get(kind, 12)
            return (priority, enemy.get("hp", 999), grid.distance(here, _coord(enemy)))

        return min(in_range, key=score)

    def _movement_goal(
        self,
        grid: HexGrid,
        turn: int,
        unit: dict,
        own_units: list[dict],
        anchors: list[dict],
        enemies: list[dict],
        occupied: set[tuple[int, int]],
        terrain: dict[tuple[int, int], str],
    ) -> HexCoord | None:
        here = _coord(unit)
        unit_type = unit["type"]
        nearest_enemy = self._priority_enemy(grid, here, enemies, unit_type)
        nearest_anchor = self._nearest(grid, here, anchors)
        anchor_coord = _coord(nearest_anchor) if nearest_anchor else None
        movement = unit.get("movement_range", 0)
        completed_base_count = sum(1 for a in anchors if a.get("type") == "Base")
        garrison_first = completed_base_count < 3 or turn < 80

        if unit_type == "Scout":
            if (
                nearest_enemy
                and grid.distance(here, _coord(nearest_enemy)) <= 3
                and anchor_coord
            ):
                return self._deploy_near(
                    grid, anchor_coord, movement, occupied, terrain, 2
                )
            if turn < 90 and anchor_coord and len(own_units) >= 14:
                probe = self._remote_probe_target(grid, turn, anchors, enemies, occupied)
                if probe is not None:
                    return probe
            return self._deploy_near(grid, anchor_coord, movement, occupied, terrain, 3) if anchor_coord else None

        if unit_type in ("Artillery", "Medic"):
            if (
                unit_type == "Artillery"
                and nearest_enemy
            ):
                return self._deploy_near(
                    grid, _coord(nearest_enemy), movement, occupied, terrain, 3
                )
            if garrison_first:
                garrison = self._garrison_deploy(
                    grid, unit, own_units, anchors, occupied, terrain
                )
                if garrison is not None:
                    return garrison
            if nearest_enemy and grid.distance(here, _coord(nearest_enemy)) <= 5:
                desired = 3 if unit_type == "Artillery" else 1
                return self._deploy_near(
                    grid, _coord(nearest_enemy), movement, occupied, terrain, desired
                )
            if anchor_coord:
                desired = 2 if unit_type == "Artillery" else 1
                return self._deploy_near(
                    grid, anchor_coord, movement, occupied, terrain, desired
                )
            return None

        if unit_type in ("Tank", "Infantry", "Fighter"):
            if nearest_enemy:
                desired = 2 if unit_type == "Fighter" else 1
                return self._deploy_near(
                    grid, _coord(nearest_enemy), movement, occupied, terrain, desired
                )
            if garrison_first or unit_type != "Fighter":
                garrison = self._garrison_deploy(
                    grid, unit, own_units, anchors, occupied, terrain
                )
                if garrison is not None:
                    return garrison
            if nearest_enemy:
                desired = 2 if unit_type == "Fighter" else 1
                return self._deploy_near(
                    grid, _coord(nearest_enemy), movement, occupied, terrain, desired
                )
            if unit_type == "Fighter":
                remembered_airbase = self._nearest_memory_target(
                    grid, here, self._seen_enemy_airbases
                )
                if remembered_airbase is not None:
                    return self._deploy_near(
                        grid, remembered_airbase, movement, occupied, terrain, 2
                    )
            if anchor_coord:
                desired = 2 if unit_type == "Fighter" else 1
                return self._deploy_near(
                    grid, anchor_coord, movement, occupied, terrain, desired
                )
            return None

        if unit_type == "Bomber":
            enemy_base = self._nearest_of_type(
                grid, here, enemies, {"Base"}
            ) or self._nearest_memory_target(grid, here, self._seen_enemy_bases)
            if enemy_base:
                enemy_coord = _coord(enemy_base) if isinstance(enemy_base, dict) else enemy_base
                return self._deploy_near(
                    grid, enemy_coord, movement, occupied, terrain, 1
                )
            return self._deploy_near(grid, anchor_coord, movement, occupied, terrain, 2) if anchor_coord else None

        return None

    @staticmethod
    def _enemy_threatens_anchor(
        grid: HexGrid, enemy: dict, anchors: list[dict]
    ) -> bool:
        if not anchors:
            return True
        radius = 6 if enemy["type"] in ("Bomber", "Artillery", "Tank", "Fighter") else 4
        return any(
            grid.distance(_coord(enemy), _coord(anchor)) <= radius for anchor in anchors
        )

    def _has_air_threat(self, enemies: list[dict]) -> bool:
        return any(e["type"] in _AIR_THREATS for e in enemies) or bool(
            self._seen_enemy_airbases
        )

    def _priority_enemy(
        self, grid: HexGrid, here: HexCoord, enemies: list[dict], unit_type: str
    ) -> dict | None:
        if not enemies:
            return None

        def score(enemy: dict) -> tuple[int, int]:
            kind = enemy["type"]
            if unit_type == "Fighter":
                priority = {
                    "Bomber": 0,
                    "Fighter": 1,
                    "Airbase": 2,
                    "Artillery": 3,
                    "Base": 4,
                }.get(kind, 8)
            elif unit_type == "Bomber":
                priority = {
                    "Base": 0,
                    "Airbase": 1,
                    "Factory": 2,
                    "Barracks": 3,
                    "Mine": 4,
                }.get(kind, 8)
            elif unit_type == "Artillery":
                priority = {
                    "Base": 0,
                    "Airbase": 1,
                    "Factory": 2,
                    "Artillery": 3,
                    "Tank": 4,
                    "Bomber": 5,
                }.get(kind, 8)
            else:
                priority = {
                    "Bomber": 0,
                    "Artillery": 1,
                    "Tank": 2,
                    "Fighter": 3,
                    "Base": 4,
                    "Airbase": 5,
                    "Factory": 6,
                }.get(kind, 8)
            return (priority, grid.distance(here, _coord(enemy)))

        return min(enemies, key=score)

    @staticmethod
    def _nearest_memory_target(
        grid: HexGrid, here: HexCoord, memory: dict[str, HexCoord]
    ) -> HexCoord | None:
        if not memory:
            return None
        return min(memory.values(), key=lambda c: grid.distance(here, c))

    def _deploy_near(
        self,
        grid: HexGrid,
        target: HexCoord,
        movement: int,
        occupied: set[tuple[int, int]],
        terrain: dict[tuple[int, int], str],
        desired_distance: int,
    ) -> HexCoord | None:
        if movement < 1:
            return None

        candidates: list[HexCoord] = []
        radii = [desired_distance]
        radii.extend(r for r in range(1, 5) if r != desired_distance)
        for radius in radii:
            candidates.extend(grid.ring(target, radius))

        valid = [
            c
            for c in candidates
            if (c.q, c.r) not in occupied and self._can_land(c, movement, terrain)
        ]
        if not valid:
            return None

        def score(candidate: HexCoord) -> tuple[int, int]:
            distance = grid.distance(candidate, target)
            terrain_penalty = 1 if terrain.get((candidate.q, candidate.r)) == "difficult" else 0
            return (abs(distance - desired_distance), terrain_penalty)

        return min(valid, key=score)

    def _garrison_deploy(
        self,
        grid: HexGrid,
        unit: dict,
        own_units: list[dict],
        anchors: list[dict],
        occupied: set[tuple[int, int]],
        terrain: dict[tuple[int, int], str],
    ) -> HexCoord | None:
        bases = [a for a in anchors if a.get("type") == "Base"]
        if not bases:
            return None

        unit_type = unit["type"]
        movement = unit.get("movement_range", 0)
        desired = 2 if unit_type in ("Artillery", "Fighter") else 1
        here = _coord(unit)

        def nearby_defenders(base: dict) -> int:
            base_coord = _coord(base)
            return sum(
                1
                for other in own_units
                if other["id"] != unit["id"]
                and grid.distance(_coord(other), base_coord) <= desired
            )

        # Units already in a useful defensive ring should not churn every turn.
        for base in bases:
            if grid.distance(here, _coord(base)) <= desired:
                return None

        target_base = min(bases, key=lambda b: (nearby_defenders(b), b.get("hp", 999)))
        needed = 6 if desired == 1 else 10
        if nearby_defenders(target_base) >= needed:
            return None
        return self._deploy_near(
            grid, _coord(target_base), movement, occupied, terrain, desired
        )

    @staticmethod
    def _can_land(
        coord: HexCoord, movement: int, terrain: dict[tuple[int, int], str]
    ) -> bool:
        kind = terrain.get((coord.q, coord.r))
        if kind == "difficult":
            return movement >= 2
        if kind is None and movement == 1:
            return False
        return True

    def _teleport_path(
        self,
        grid: HexGrid,
        start: HexCoord,
        goal: HexCoord,
        movement: int,
        occupied: set[tuple[int, int]],
        terrain: dict[tuple[int, int], str],
    ) -> list[HexCoord] | None:
        dest = grid.wrap(goal)
        if movement < 1 or dest == start:
            return None
        if (dest.q, dest.r) in occupied:
            return None
        if not self._can_land(dest, movement, terrain):
            return None
        return [start, dest]

    def _remote_probe_target(
        self,
        grid: HexGrid,
        turn: int,
        bases: list[dict],
        enemies: list[dict],
        occupied: set[tuple[int, int]],
    ) -> HexCoord | None:
        base_coords = [_coord(b) for b in bases]
        enemy_coords = [_coord(e) for e in enemies] + list(self._seen_enemy_bases.values())

        coords = self._all_coords_cache.setdefault(
            (grid.width, grid.height), list(grid.all_coords())
        )
        offset = (turn * 17 + self._remote_probe_index * 43) % len(coords)
        rotated = coords[offset:] + coords[:offset]

        def score(candidate: HexCoord) -> tuple[int, int, int]:
            base_dist = min((grid.distance(candidate, b) for b in base_coords), default=8)
            enemy_dist = min((grid.distance(candidate, e) for e in enemy_coords), default=8)
            occupied_penalty = 1 if (candidate.q, candidate.r) in occupied else 0
            return (-occupied_penalty, min(base_dist, 12), min(enemy_dist, 10))

        return max(rotated, key=score) if rotated else None

    def _remote_base_site(
        self, grid: HexGrid, scout_target: HexCoord, occupied: set[tuple[int, int]]
    ) -> HexCoord | None:
        options = [nb for nb in grid.neighbors(scout_target) if (nb.q, nb.r) not in occupied]
        if not options:
            return None
        known_enemy = list(self._seen_enemy_bases.values()) + list(
            self._seen_enemy_airbases.values()
        )
        if not known_enemy:
            return options[0]
        return max(
            options,
            key=lambda c: min((grid.distance(c, e) for e in known_enemy), default=8),
        )

    def _base_site(
        self,
        grid: HexGrid,
        bases: list[dict],
        enemies: list[dict],
        occupied: set[tuple[int, int]],
        terrain: dict[tuple[int, int], str],
        visible: list[tuple[int, int]],
    ) -> HexCoord | None:
        candidates = [HexCoord(q, r) for q, r in visible if (q, r) not in occupied]
        if not candidates:
            return None

        def score(coord: HexCoord) -> tuple[int, int, int]:
            rich = 1 if terrain.get((coord.q, coord.r)) == "rich_resource" else 0
            base_dist = min((grid.distance(coord, _coord(b)) for b in bases), default=0)
            enemy_dist = min((grid.distance(coord, _coord(e)) for e in enemies), default=8)
            # Prefer rich, separated, and not right beside known enemies.
            return (rich, min(base_dist, 10), min(enemy_dist, 8))

        return max(candidates, key=score)

    def _anchored_site(
        self,
        grid: HexGrid,
        building_type: str,
        complete_buildings: list[dict],
        enemies: list[dict],
        occupied: set[tuple[int, int]],
        terrain: dict[tuple[int, int], str],
    ) -> HexCoord | None:
        candidates: list[tuple[dict, HexCoord]] = []
        for building in complete_buildings:
            for nb in grid.neighbors(_coord(building)):
                if (nb.q, nb.r) not in occupied:
                    candidates.append((building, nb))
        if not candidates:
            return None

        def score(item: tuple[dict, HexCoord]) -> tuple[int, int, int]:
            anchor, coord = item
            rich = 1 if terrain.get((coord.q, coord.r)) == "rich_resource" else 0
            enemy_dist = min((grid.distance(coord, _coord(e)) for e in enemies), default=8)
            base_bias = 1 if anchor["type"] == "Base" else 0
            if building_type == "Mine":
                return (rich, min(enemy_dist, 8), base_bias)
            if building_type in ("Factory", "Airbase"):
                # Keep expensive production tucked near Bases.
                return (base_bias, min(enemy_dist, 8), rich)
            return (min(enemy_dist, 8), base_bias, rich)

        return max(candidates, key=score)[1]

    @staticmethod
    def _free_neighbors(
        grid: HexGrid, coord: tuple[int, int], occupied: set[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        return [
            (nb.q, nb.r)
            for nb in grid.neighbors(HexCoord(*coord))
            if (nb.q, nb.r) not in occupied
        ]

    @staticmethod
    def _nearest(grid: HexGrid, here: HexCoord, entities: list[dict]) -> dict | None:
        if not entities:
            return None
        return min(entities, key=lambda e: grid.distance(here, _coord(e)))

    @staticmethod
    def _nearest_of_type(
        grid: HexGrid, here: HexCoord, entities: list[dict], types: set[str]
    ) -> dict | None:
        filtered = [e for e in entities if e["type"] in types]
        if not filtered:
            return None
        return min(filtered, key=lambda e: grid.distance(here, _coord(e)))

    @staticmethod
    def _scout_goal(grid: HexGrid, here: HexCoord, anchor: HexCoord) -> HexCoord:
        best = here
        best_dist = grid.distance(here, anchor)
        for nb in grid.neighbors(here):
            dist = grid.distance(nb, anchor)
            if dist > best_dist:
                best = nb
                best_dist = dist
        return best

    @staticmethod
    def _stand_off_goal(
        grid: HexGrid, here: HexCoord, target: HexCoord, wanted: int
    ) -> HexCoord:
        best = here
        best_delta = abs(grid.distance(here, target) - wanted)
        for nb in grid.neighbors(here):
            delta = abs(grid.distance(nb, target) - wanted)
            if delta < best_delta:
                best = nb
                best_delta = delta
        return best

    def _greedy_path(
        self,
        grid: HexGrid,
        start: HexCoord,
        goal: HexCoord,
        movement: int,
        occupied: set[tuple[int, int]],
        terrain: dict[tuple[int, int], str],
    ) -> list[HexCoord] | None:
        if movement <= 0 or start == goal:
            return None

        path = [start]
        current = start
        budget = movement
        reserved = set(occupied)
        reserved.discard((start.q, start.r))

        while budget > 0 and current != goal:
            candidates = []
            for nb in grid.neighbors(current):
                if (nb.q, nb.r) in reserved:
                    continue
                cost = 2 if terrain.get((nb.q, nb.r)) == "difficult" else 1
                if cost > budget:
                    continue
                candidates.append((grid.distance(nb, goal), cost, nb))
            if not candidates:
                break
            _, cost, step = min(candidates)
            if (
                grid.distance(step, goal) >= grid.distance(current, goal)
                and len(path) > 1
            ):
                break
            current = step
            path.append(current)
            reserved.add((current.q, current.r))
            budget -= cost

        return path if len(path) > 1 else None
