"""Planner: pure functions over the WorldModel → a strategic Plan.

No hex math, no action emission — that all lives in the actuator. The planner
only decides *intent*: how much economy, how many redundancy Bases, whether an
air threat warrants an Airbase, and whether to flip from turtle to hunter.

The whole strategy is driven by the load-bearing fact that **scoring is binary
survival with no tiebreaker** (game_runner: every player alive at the turn limit
co-wins). So the default is a defensive turtle; offense is only ever defense.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from world import WorldModel


@dataclass
class Plan:
    stance: str = "turtle"  # "turtle" | "hunter"

    # economy / build intent
    mine_target: int = 3
    base_target: int = 2  # desired number of Bases (complete + under construction)
    want_factory: bool = True  # default ground muscle (Tanks)
    want_airbase: bool = False  # only on observed air threat

    # garrison intent (per base, punishment to one-shot an intruding Bomber)
    garrison_tanks: int = 2
    garrison_fighters: int = 0

    # diplomacy
    accept_all: bool = True
    propose_peace: bool = True

    # hunter
    hunter_targets: list[str] = field(default_factory=list)  # player ids to decapitate


def plan(world: WorldModel) -> Plan:
    p = Plan()

    # ── clock-aware scaling ────────────────────────────────────────────────────
    # Short games (Discord eval = 50 turns): survive-early is the whole game — keep
    # it lean (one redundancy Base, modest economy) so defense comes online fast.
    # Long games: invest more in economy + a third Base for deep redundancy.
    horizon = world.max_turns
    if horizon <= 80:
        p.mine_target = 2
        p.base_target = 2
    elif horizon <= 200:
        p.mine_target = 3
        p.base_target = 2
        if world.turn > horizon // 2:
            p.base_target = 3
    else:
        p.mine_target = 4
        p.base_target = 2
        if world.turn > 60:
            p.base_target = 3

    # ── air-threat trigger ──────────────────────────────────────────────────────
    # Only complete Bases self-spot, and Bombers (teleport + ×4 vs buildings) are
    # the only thing that reliably kills a ringed/hidden Base. If we see ANY enemy
    # air capability, rush an Airbase + Fighters (range 2, out-range Bombers) to
    # one-shot an intruder; otherwise a Factory for cheaper ground stability.
    if world.air_threat:
        p.want_airbase = True
        p.garrison_fighters = 3  # ≥150 dmg one-shots a 150-hp Bomber
        p.garrison_tanks = 1
    else:
        p.want_factory = True
        p.garrison_tanks = 2

    # ── stance ──────────────────────────────────────────────────────────────────
    # Stay turtle. Flip individual players into the hunter list only when offense
    # IS defense: someone is actively warring on us (a treaty they broke, i.e. a
    # BREAKING countdown = war now) and we can see a Base of theirs to decapitate.
    # Binary survival makes a rampage a bad *primary* plan, so this stays narrow.
    if world.breaking_partners:
        seen_bases = {en.owner_id for en in world.enemies.values() if en.is_building and en.type == "Base"}
        targets = [pid for pid in world.breaking_partners if pid in seen_bases]
        if targets:
            p.hunter_targets = targets

    return p
