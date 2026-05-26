"""Optional learned attack decision module.

The production heuristic owns movement, dodging, collection, exploration, and
base routing.  This module is intentionally narrow: it can only override the
attack phase with PLACE_BOMB, or defer back to the scripted policy.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from constants import (
    Action,
    BASE_MAX_HEALTH,
    BOMB_ATTACK,
    BOMB_BLAST_RADIUS,
    BOMB_TIMER,
    DIR_VECTOR,
    Direction,
    GRID_SIZE,
)
from map_memory import MapMemory
from observation import ParsedObs
from threat import cells_in_blast, expected_blast_hits_drift, project_danger


RL_ATTACK_FEATURE_DIM = 24
RL_ATTACK_SPATIAL_CHANNELS = 16
RL_ATTACK_SPATIAL_SHAPE = (RL_ATTACK_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE)

SPATIAL_KNOWN = 0
SPATIAL_RECENCY = 1
SPATIAL_WALL = 2
SPATIAL_DESTR_WALL = 3
SPATIAL_MISSION = 4
SPATIAL_RESOURCE = 5
SPATIAL_RECON = 6
SPATIAL_ALLY_BASE = 7
SPATIAL_ENEMY_BASE = 8
SPATIAL_ENEMY_BASE_LOW_HP = 9
SPATIAL_ENEMY_AGENT = 10
SPATIAL_ENEMY_AGENT_STALE = 11
SPATIAL_ALLY_BOMB = 12
SPATIAL_ENEMY_BOMB = 13
SPATIAL_SELF = 14
SPATIAL_FACING = 15


class AttackModule(Protocol):
    def choose_attack(self, obs: ParsedObs, memory: MapMemory) -> Optional[int]: ...


@dataclass(frozen=True)
class AttackFeatures:
    """Dense tactical state for a bomb-or-defer attacker."""

    step_frac: float
    health_frac: float
    base_health_frac: float
    bombs_frac: float
    resources_frac: float
    can_bomb: float
    sitting_on_ally_bomb: float
    direct_agent_hits: float
    direct_base_hits: float
    finish_base_hits: float
    expected_agent_hits: float
    nearest_enemy_dist: float
    nearest_enemy_age: float
    nearest_base_dist: float
    nearest_low_base_dist: float
    blast_escape_count: float
    self_danger_tick: float
    enemy_bomb_nearby: float
    ally_bomb_nearby: float
    enemy_count: float
    base_count: float
    x_frac: float
    y_frac: float
    direction_frac: float

    def to_array(self) -> np.ndarray:
        return np.asarray(tuple(asdict(self).values()), dtype=np.float32)


def extract_attack_features(obs: ParsedObs, memory: MapMemory) -> np.ndarray:
    """Build normalized features for attack-only RL inference/training."""
    blast = cells_in_blast(memory, obs.location)
    direct_agent_hits = sum(1 for p in memory.enemy_agents if p in blast)
    direct_base_hits = sum(1 for p in memory.enemy_bases if p in blast)
    finish_base_hits = sum(
        1
        for p in memory.enemy_bases
        if p in blast and memory.enemy_base_health.get(p, BASE_MAX_HEALTH) <= BOMB_ATTACK
    )
    expected = expected_blast_hits_drift(memory, blast, BOMB_TIMER)
    timeline = project_danger(memory)

    features = AttackFeatures(
        step_frac=_clip(obs.step / 200.0),
        health_frac=_clip(obs.health / 60.0),
        base_health_frac=_clip(obs.base_health / BASE_MAX_HEALTH),
        bombs_frac=_clip(obs.team_bombs / 6.0),
        resources_frac=_clip(obs.team_resources / 10.0),
        can_bomb=1.0
        if obs.action_mask[Action.PLACE_BOMB] == 1 and obs.team_bombs > 0
        else 0.0,
        sitting_on_ally_bomb=1.0
        if (bomb := memory.bombs.get(obs.location)) is not None and bomb.ally
        else 0.0,
        direct_agent_hits=_clip(direct_agent_hits / 3.0),
        direct_base_hits=_clip(direct_base_hits / 2.0),
        finish_base_hits=_clip(finish_base_hits / 2.0),
        expected_agent_hits=_clip(expected / 3.0),
        nearest_enemy_dist=_norm_dist(_nearest_distance(obs.location, memory.enemy_agents)),
        nearest_enemy_age=_norm_age(_nearest_enemy_age(obs, memory)),
        nearest_base_dist=_norm_dist(_nearest_distance(obs.location, memory.enemy_bases)),
        nearest_low_base_dist=_norm_dist(_nearest_low_base_distance(obs, memory)),
        blast_escape_count=_clip(_blast_escape_count(obs.location, blast, memory) / 4.0),
        self_danger_tick=_norm_danger_tick(_first_danger_tick(obs.location, timeline)),
        enemy_bomb_nearby=_clip(_nearby_bombs(obs.location, memory, ally=False) / 3.0),
        ally_bomb_nearby=_clip(_nearby_bombs(obs.location, memory, ally=True) / 3.0),
        enemy_count=_clip(len(memory.enemy_agents) / 5.0),
        base_count=_clip(len(memory.enemy_bases) / 5.0),
        x_frac=_clip(obs.location[0] / max(1, GRID_SIZE - 1)),
        y_frac=_clip(obs.location[1] / max(1, GRID_SIZE - 1)),
        direction_frac=_clip(obs.direction / 3.0),
    )
    arr = features.to_array()
    if arr.shape != (RL_ATTACK_FEATURE_DIM,):
        raise ValueError(f"attack feature shape mismatch: {arr.shape}")
    return arr


def extract_attack_spatial(obs: ParsedObs, memory: MapMemory) -> np.ndarray:
    """Build a stale-info world tensor for the attack CNN.

    Shape is channel-first `(16, 16, 16)`. It is intentionally global rather
    than base-view based: the tensor marks where the agent is on the map and
    fills the rest with whatever MapMemory currently knows or remembers.
    """
    spatial = np.zeros(RL_ATTACK_SPATIAL_SHAPE, dtype=np.float32)

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            pos = (x, y)
            seen_step = memory.last_seen_step.get(pos)
            if seen_step is not None:
                age = max(0, obs.step - seen_step)
                spatial[SPATIAL_KNOWN, y, x] = 1.0
                spatial[SPATIAL_RECENCY, y, x] = _clip(1.0 - age / 40.0)

            blocked = 0
            destructible = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nbr = (x + dx, y + dy)
                if not memory.in_bounds(nbr):
                    blocked += 1
                    continue
                if not memory.passable(pos, nbr):
                    blocked += 1
                if memory.edge_is_destructible_wall(pos, nbr):
                    destructible += 1
            spatial[SPATIAL_WALL, y, x] = blocked / 4.0
            spatial[SPATIAL_DESTR_WALL, y, x] = destructible / 4.0

    for (x, y), kind in memory.tile_contents.items():
        if not memory.in_bounds((x, y)):
            continue
        if kind == "mission":
            spatial[SPATIAL_MISSION, y, x] = 1.0
        elif kind == "resource":
            spatial[SPATIAL_RESOURCE, y, x] = 1.0
        elif kind == "recon":
            spatial[SPATIAL_RECON, y, x] = 1.0

    if memory.ally_base is not None and memory.in_bounds(memory.ally_base):
        x, y = memory.ally_base
        spatial[SPATIAL_ALLY_BASE, y, x] = 1.0

    for x, y in memory.enemy_bases:
        if not memory.in_bounds((x, y)):
            continue
        spatial[SPATIAL_ENEMY_BASE, y, x] = 1.0
        hp = memory.enemy_base_health.get((x, y), BASE_MAX_HEALTH)
        if hp <= BOMB_ATTACK:
            spatial[SPATIAL_ENEMY_BASE_LOW_HP, y, x] = 1.0

    for (x, y), step_seen in memory.enemy_agents.items():
        if not memory.in_bounds((x, y)):
            continue
        age = max(0, obs.step - step_seen)
        channel = SPATIAL_ENEMY_AGENT if age <= 1 else SPATIAL_ENEMY_AGENT_STALE
        spatial[channel, y, x] = _clip(1.0 - age / 12.0)

    for bomb in memory.bombs.values():
        if not memory.in_bounds(bomb.pos):
            continue
        x, y = bomb.pos
        remaining = _clip(bomb.remaining(obs.step) / max(1, BOMB_TIMER))
        channel = SPATIAL_ALLY_BOMB if bomb.ally else SPATIAL_ENEMY_BOMB
        spatial[channel, y, x] = max(spatial[channel, y, x], remaining)

    if memory.in_bounds(obs.location):
        x, y = obs.location
        spatial[SPATIAL_SELF, y, x] = 1.0
        dx, dy = DIR_VECTOR[Direction(obs.direction)]
        facing = (x + dx, y + dy)
        if memory.in_bounds(facing):
            fx, fy = facing
            spatial[SPATIAL_FACING, fy, fx] = 1.0

    return spatial


class TorchAttackModule:
    """Loads a TorchScript or state_dict DQN and returns PLACE_BOMB/defer."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        bomb_margin: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.bomb_margin = bomb_margin
        self.device = device
        self.is_actor_critic = False
        self.model = self._load_model()
        self.model.eval()

    def choose_attack(self, obs: ParsedObs, memory: MapMemory) -> Optional[int]:
        if obs.action_mask[Action.PLACE_BOMB] != 1 or obs.team_bombs <= 0:
            return None
        if (bomb := memory.bombs.get(obs.location)) is not None and bomb.ally:
            return None

        torch = _import_torch()
        scalar = torch.as_tensor(
            extract_attack_features(obs, memory), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        spatial = torch.as_tensor(
            extract_attack_spatial(obs, memory), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            out = self.model(scalar, spatial)
            q_tensor = out[0] if self.is_actor_critic else out
            q = q_tensor[0].detach().cpu().numpy()
        return int(Action.PLACE_BOMB) if q[1] > q[0] + self.bomb_margin else None

    def _load_model(self):
        torch = _import_torch()
        try:
            return torch.jit.load(str(self.model_path), map_location=self.device)
        except Exception:
            payload = torch.load(
                str(self.model_path), map_location=self.device, weights_only=False
            )
            if payload.get("algo") == "ppo":
                from rl_attack_ppo_model import AttackActorCritic

                self.is_actor_critic = True
                model = AttackActorCritic(
                    int(payload.get("feature_dim", RL_ATTACK_FEATURE_DIM)),
                    int(payload.get("hidden_dim", 96)),
                    int(payload.get("spatial_channels", RL_ATTACK_SPATIAL_CHANNELS)),
                )
            else:
                from rl_attack_model import AttackDQN

                model = AttackDQN(
                    int(payload.get("feature_dim", RL_ATTACK_FEATURE_DIM)),
                    int(payload.get("hidden_dim", 96)),
                    int(payload.get("spatial_channels", RL_ATTACK_SPATIAL_CHANNELS)),
                )
            model.load_state_dict(payload["model_state"])
            model.to(self.device)
            return model


class LinearAttackModule:
    """Tiny JSON-exported fallback for production without torch."""

    def __init__(self, model_path: Path | str, *, bomb_margin: float = 0.0) -> None:
        payload = json.loads(Path(model_path).read_text())
        self.weights = np.asarray(payload["weights"], dtype=np.float32)
        self.bias = np.asarray(payload["bias"], dtype=np.float32)
        self.bomb_margin = float(payload.get("bomb_margin", bomb_margin))
        if self.weights.shape != (2, RL_ATTACK_FEATURE_DIM) or self.bias.shape != (2,):
            raise ValueError("linear attack model has invalid shape")

    def choose_attack(self, obs: ParsedObs, memory: MapMemory) -> Optional[int]:
        if obs.action_mask[Action.PLACE_BOMB] != 1 or obs.team_bombs <= 0:
            return None
        x = extract_attack_features(obs, memory)
        q = self.weights @ x + self.bias
        return int(Action.PLACE_BOMB) if q[1] > q[0] + self.bomb_margin else None


def load_attack_module(
    model_path: str | None,
    *,
    bomb_margin: float = 0.0,
) -> Optional[AttackModule]:
    """Best-effort loader used by production policy construction."""
    if not model_path:
        return None
    path = Path(model_path)
    if not path.exists():
        return None
    if path.suffix == ".json":
        return LinearAttackModule(path, bomb_margin=bomb_margin)
    try:
        return TorchAttackModule(path, bomb_margin=bomb_margin)
    except RuntimeError:
        return None


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if math.isnan(value) or math.isinf(value):
        return lo
    return max(lo, min(hi, float(value)))


def _nearest_distance(
    pos: tuple[int, int], targets
) -> Optional[int]:
    if not targets:
        return None
    return min(abs(pos[0] - p[0]) + abs(pos[1] - p[1]) for p in targets)


def _norm_dist(dist: Optional[int]) -> float:
    if dist is None:
        return 1.0
    return _clip(dist / (2 * GRID_SIZE))


def _nearest_enemy_age(obs: ParsedObs, memory: MapMemory) -> Optional[int]:
    if not memory.enemy_agents:
        return None
    nearest = min(
        memory.enemy_agents,
        key=lambda p: abs(obs.location[0] - p[0]) + abs(obs.location[1] - p[1]),
    )
    return max(0, obs.step - memory.enemy_agents[nearest])


def _norm_age(age: Optional[int]) -> float:
    if age is None:
        return 1.0
    return _clip(age / 12.0)


def _nearest_low_base_distance(obs: ParsedObs, memory: MapMemory) -> Optional[int]:
    low = [
        p
        for p in memory.enemy_bases
        if memory.enemy_base_health.get(p, BASE_MAX_HEALTH) <= BOMB_ATTACK
    ]
    return _nearest_distance(obs.location, low)


def _blast_escape_count(
    pos: tuple[int, int],
    blast: set[tuple[int, int]],
    memory: MapMemory,
) -> int:
    count = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nbr = (pos[0] + dx, pos[1] + dy)
        if memory.in_bounds(nbr) and nbr not in blast and memory.passable(pos, nbr):
            count += 1
    return count


def _first_danger_tick(
    pos: tuple[int, int],
    timeline: dict[int, set[tuple[int, int]]],
) -> Optional[int]:
    for tick in sorted(timeline):
        if pos in timeline[tick]:
            return tick
    return None


def _norm_danger_tick(tick: Optional[int]) -> float:
    if tick is None:
        return 1.0
    return _clip(tick / (BOMB_TIMER + 1))


def _nearby_bombs(pos: tuple[int, int], memory: MapMemory, *, ally: bool) -> int:
    count = 0
    for bomb in memory.bombs.values():
        if bomb.ally != ally:
            continue
        dist = abs(pos[0] - bomb.pos[0]) + abs(pos[1] - bomb.pos[1])
        if dist <= BOMB_BLAST_RADIUS + 1:
            count += 1
    return count


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Torch is required for .pt attack models. Install requirements-rl.txt "
            "or export a JSON linear attack model."
        ) from exc
    return torch
