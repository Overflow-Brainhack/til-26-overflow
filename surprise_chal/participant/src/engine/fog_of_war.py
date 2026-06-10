"""fog of war: compute which tiles are visible to a player each turn"""

from __future__ import annotations

from engine.entities.building import Building
from engine.entities.unit import Unit
from engine.hex_grid import HexCoord, HexGrid
from engine.state import GameState


def compute_visible(state: GameState, player_id: str) -> set[HexCoord]:
    """return the set of all coords visible to player_id this turn.

    vision rules:
    - each unit contributes vision_range tiles from its position
    - each building contributes vision_bonus tiles (scout tower adds 4)
    - elevated terrain blocks LOS for non-elevated/non-flying observers
    - concealment tiles reduce effective vision range by 1 when looking in
    - scouts in concealment terrain are themselves not revealed to enemies
    """
    grid = state.grid
    visible: set[HexCoord] = set()

    # collect all observer positions and their effective ranges
    observers: list[tuple[HexCoord, int, bool]] = []  # (coord, range, is_flying)

    for entity in state.entities.values():
        if entity.owner_id != player_id:
            continue
        coord = entity.coord
        if isinstance(entity, Unit):
            observers.append((coord, entity.vision_range, entity.can_fly))
        elif (
            isinstance(entity, Building)
            and entity.is_complete
            and entity.vision_bonus > 0
        ):
            observers.append((coord, entity.vision_bonus, False))

    for origin, vision_range, is_flying in observers:
        origin_tile = state.tile(origin)
        origin_elevated = origin_tile.is_elevated()
        # Concealment penalty: a concealment tile is harder to see INTO, so it is
        # only visible from one tile closer than normal (vision_range - 1). We
        # precompute that reduced disk once per observer and gate concealment
        # candidates on it. Flying observers are exempt (clean aerial sight, like
        # the elevation rule). Computed lazily — most observers see no concealment.
        near: set[HexCoord] | None = None

        for candidate in grid.disk(origin, vision_range):
            if candidate in visible:
                continue

            if not is_flying and state.tile(candidate).is_concealment():
                if near is None:
                    near = set(grid.disk(origin, max(0, vision_range - 1)))
                if candidate not in near:
                    continue  # concealment tile on the outer ring — not seen

            # check if LOS is blocked by elevation along the path
            if _has_line_of_sight(
                state, grid, origin, candidate, origin_elevated, is_flying
            ):
                visible.add(candidate)

    return visible


def _has_line_of_sight(
    state: GameState,
    grid: HexGrid,
    origin: HexCoord,
    target: HexCoord,
    origin_elevated: bool,
    is_flying: bool,
) -> bool:
    """true if observer at origin can see target.

    blocking rules:
    - elevated tiles block LOS unless the observer is also elevated or flying
    - concealment tiles do not block LOS (their -1 vision-range penalty is applied
      separately in compute_visible, not here)
    """
    if origin == target:
        return True

    if is_flying:
        return True  # air units have unobstructed vision

    path = grid.line(origin, target)

    for coord in path[1:]:  # skip origin itself
        tile = state.tile(coord)
        if tile.is_elevated() and not origin_elevated:
            if coord != target:
                return False  # elevated tile in the middle blocks
            # if the target itself is elevated and observer is not, still visible (adjacent/direct)
    return True


def filter_observation(state: GameState, player_id: str) -> tuple[set[HexCoord], dict]:
    """returns (visible_coords, filtered_entity_dict) for building observations.

    entities on non-visible tiles are excluded from the observation.
    enemy entities in concealment may be partially hidden.
    """
    from engine.entities.units.scout import Scout

    visible = compute_visible(state, player_id)
    filtered: dict[str, dict] = {}

    for eid, entity in state.entities.items():
        if entity.coord not in visible:
            continue

        # scouts in concealment are invisible to non-owners
        if (
            isinstance(entity, Scout)
            and entity.owner_id != player_id
            and state.tile(entity.coord).is_concealment()
        ):
            continue

        filtered[eid] = entity.to_dict()

    return visible, filtered
