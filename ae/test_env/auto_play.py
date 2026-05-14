"""Auto-play visualization for the heuristic AE agent.

Adapted from til-26-ae/play.py. Every agent in the round is controlled by
its own HeuristicPolicy + isolated MapMemory, so you can watch the bots
play each other.

Usage (from repo root, with the dev environment active):
    python ae/test_env/auto_play.py
    python ae/test_env/auto_play.py --rounds 3 --seed 42 --fps 4

Keys during play:
    Q / ESC   quit
    R         reset to a new round
    T         toggle the tile-respawn-timer overlay
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pygame

# Make ae/src importable as flat top-level modules (matching Docker layout).
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from til_environment.bomberman_env import Bomberman  # noqa: E402
from til_environment.config import default_config, load_config  # noqa: E402

from ae_manager import DEFAULT_CACHE_PATH, DEFAULT_POLICY_KWARGS, AEManager  # noqa: E402
from constants import GRID_SIZE  # noqa: E402
from map_memory import MapMemory  # noqa: E402
from policy import HeuristicPolicy  # noqa: E402


def _build_managers(
    env: Bomberman,
    policy_factory,
    cache_path,
) -> dict[str, AEManager]:
    """One AEManager per agent. Each bot has an isolated MapMemory; if a
    cache_path is provided and exists, every bot pre-loads it (so round 1
    starts with full map knowledge)."""
    cached_template: MapMemory | None = None
    if cache_path is not None and cache_path.exists():
        cached_template = MapMemory.load(cache_path)

    out: dict[str, AEManager] = {}
    for agent in env.possible_agents:
        mem = MapMemory()
        if cached_template is not None:
            mem.merge_static_from(cached_template)
        out[agent] = AEManager(policy=policy_factory(), memory=mem)
    return out


def _reset_managers(managers: dict[str, AEManager]) -> None:
    for mgr in managers.values():
        mgr._memory.reset_round()


def _print_round_summary(env: Bomberman, round_idx: int) -> None:
    """Cumulative per-agent reward for the round.

    `env.rewards` is the *step* reward dict, which resets after termination.
    `env.dynamics.rewards._episode` is the cumulative reward we actually want.
    """
    episode = getattr(env.dynamics.rewards, "_episode", {})
    print(f"\n── round {round_idx} over ──")
    rewards = sorted(
        ((a, float(episode.get(a, 0.0))) for a in env.possible_agents),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for a, r in rewards:
        print(f"  {a}  reward={r:.2f}")


# ── analysis / history types ────────────────────────────────────────────────

HISTORY_MAXLEN = 300  # max render frames kept for rewind (~2-3 rounds at 4 fps)

_MODE_COLORS: dict[str, tuple[int, int, int]] = {
    "frozen":  (150, 150, 255),
    "dodge":   (255,  80,  80),
    "attack":  (255, 165,   0),
    "defend":  ( 80,  80, 255),
    "collect": ( 80, 255,  80),
    "explore": (200, 200,  80),
    "stay":    (160, 160, 160),
}


@dataclass
class AgentDebugInfo:
    mode: str
    target: Optional[tuple[int, int]]
    pos: tuple[int, int]


@dataclass
class HistoryFrame:
    surface: pygame.Surface
    agent_info: dict[str, AgentDebugInfo] = field(default_factory=dict)


def _world_to_screen(
    pos: tuple[int, int],
    tile_w: int,
    tile_h: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> tuple[int, int]:
    return (offset_x + pos[0] * tile_w + tile_w // 2,
            offset_y + pos[1] * tile_h + tile_h // 2)


def _collect_debug_info(managers: dict[str, AEManager]) -> dict[str, AgentDebugInfo]:
    out: dict[str, AgentDebugInfo] = {}
    for agent_id, mgr in managers.items():
        pol = mgr._policy
        out[agent_id] = AgentDebugInfo(
            mode=getattr(pol, "_debug_mode", "stay"),
            target=getattr(pol, "_debug_target", None),
            pos=getattr(pol, "_debug_pos", (0, 0)),
        )
    return out


def _draw_agent_panel(
    surface: pygame.Surface,
    info: AgentDebugInfo,
    agent_id: str,
    font: pygame.font.Font,
) -> None:
    lines = [
        f"Agent:  {agent_id}",
        f"Mode:   {info.mode}",
        f"Pos:    {info.pos}",
        f"Target: {info.target if info.target is not None else 'none'}",
    ]
    pad = 6
    line_h = font.get_height() + 3
    panel_w = max(font.size(ln)[0] for ln in lines) + 2 * pad
    panel_h = len(lines) * line_h + 2 * pad

    sx = surface.get_width() - panel_w - 10
    sy = surface.get_height() - panel_h - 10

    bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 190))
    surface.blit(bg, (sx, sy))
    pygame.draw.rect(surface, (180, 180, 180), (sx, sy, panel_w, panel_h), 1)

    mode_color = _MODE_COLORS.get(info.mode, (255, 255, 255))
    for i, ln in enumerate(lines):
        color = mode_color if i == 1 else (220, 220, 220)
        txt = font.render(ln, True, color)
        surface.blit(txt, (sx + pad, sy + pad + i * line_h))


def _draw_analysis_overlay(
    surface: pygame.Surface,
    frame: HistoryFrame,
    tile_w: int,
    tile_h: int,
    offset_x: int,
    offset_y: int,
    selected_agent: Optional[str],
    font: pygame.font.Font,
) -> None:
    for agent_id, info in frame.agent_info.items():
        color = _MODE_COLORS.get(info.mode, (255, 255, 255))
        ax, ay = _world_to_screen(info.pos, tile_w, tile_h, offset_x, offset_y)

        if info.target is not None:
            tx, ty = _world_to_screen(info.target, tile_w, tile_h, offset_x, offset_y)
            pygame.draw.line(surface, color, (ax, ay), (tx, ty), 2)
            pygame.draw.circle(surface, color, (tx, ty), 4, 2)

        label = info.mode[:3].upper()
        shadow = font.render(label, True, (0, 0, 0))
        text = font.render(label, True, color)
        lx = ax - text.get_width() // 2
        ly = ay - tile_h // 2 - text.get_height() - 2
        surface.blit(shadow, (lx + 1, ly + 1))
        surface.blit(text, (lx, ly))

    if selected_agent and selected_agent in frame.agent_info:
        _draw_agent_panel(surface, frame.agent_info[selected_agent], selected_agent, font)


def _draw_pause_hud(
    surface: pygame.Surface,
    history_pos: int,
    history_len: int,
    font: pygame.font.Font,
) -> None:
    label = f"PAUSED  [{history_pos + 1}/{history_len}]  ←→ step  SPACE resume"
    txt = font.render(label, True, (255, 230, 60))
    bg = pygame.Surface((txt.get_width() + 14, txt.get_height() + 8), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 170))
    sx = surface.get_width() // 2 - bg.get_width() // 2
    surface.blit(bg, (sx, 6))
    surface.blit(txt, (sx + 7, 10))


_P = DEFAULT_POLICY_KWARGS  # short alias for argparse default= expressions below


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config; defaults to bomberman_config.yaml")
    parser.add_argument("--seed", type=int, default=None,
                        help="Initial seed (random if omitted)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of rounds to play before quitting")
    parser.add_argument("--fps", type=int, default=None,
                        help="Override renderer fps")
    parser.add_argument("--novice", action="store_true", default=True,
                        help="Use the fixed novice map (default)")
    parser.add_argument("--advanced", dest="novice", action="store_false",
                        help="Use a randomized advanced map")

    # Feature toggles — defaults come from DEFAULT_POLICY_KWARGS in ae_manager.py
    # so this script always mirrors the production container configuration.
    parser.add_argument("--predictive-bomb", dest="predictive_bomb",
                        action="store_true", default=_P["predictive_bomb"],
                        help="Bomb when an enemy is *likely* to be in blast at detonation")
    parser.add_argument("--no-predictive-bomb", dest="predictive_bomb",
                        action="store_false")
    parser.add_argument("--bomb-threshold", type=float,
                        default=_P["predictive_bomb_threshold"],
                        help="Min expected enemy hits required for a predictive bomb")

    parser.add_argument("--wall-breaking", dest="wall_breaking",
                        action="store_true", default=_P["wall_breaking"],
                        help="Allow pathfinding to route through destructible walls")
    parser.add_argument("--no-wall-breaking", dest="wall_breaking",
                        action="store_false")
    parser.add_argument("--wall-break-cost", type=float, default=_P["wall_break_cost"],
                        help="Extra path cost (≈ ticks lost) to break a wall")
    parser.add_argument("--adaptive-wall-break-cost", dest="adaptive_wall_break_cost",
                        action="store_true", default=_P["adaptive_wall_break_cost"],
                        help="Scale wall-break path cost down by tile value behind the wall")
    parser.add_argument("--no-adaptive-wall-break-cost", dest="adaptive_wall_break_cost",
                        action="store_false")

    parser.add_argument("--smart-defend", dest="smart_defend",
                        action="store_true", default=_P["smart_defend"],
                        help="Coverage-based defense: navigate to the cell whose bomb blast covers "
                             "the most attack-vector cells; expand defend radius when base health is low")
    parser.add_argument("--no-smart-defend", dest="smart_defend",
                        action="store_false")
    parser.add_argument("--predictive-defend", dest="predictive_defend",
                        action="store_true", default=_P["predictive_defend"],
                        help="Bonus-score defend positions by projecting enemies along their velocity "
                             "toward the attack vector (requires --smart-defend)")
    parser.add_argument("--no-predictive-defend", dest="predictive_defend",
                        action="store_false")

    parser.add_argument("--drift-aware-bomb", dest="drift_aware_bomb",
                        action="store_true", default=_P["drift_aware_bomb"],
                        help="Use velocity-biased enemy distribution for predictive bombing")
    parser.add_argument("--no-drift-aware-bomb", dest="drift_aware_bomb",
                        action="store_false")

    parser.add_argument("--auto-tune-bomb", dest="auto_tune_bomb",
                        action="store_true", default=_P["auto_tune_bomb"],
                        help="Adaptively tune the bomb threshold via EMA of observed hit rate")
    parser.add_argument("--no-auto-tune-bomb", dest="auto_tune_bomb",
                        action="store_false")
    parser.add_argument("--bomb-tune-target", type=float, default=_P["bomb_tune_target"],
                        help="Target predictive-bomb hit rate for auto-tuning")

    parser.add_argument("--bomb-economy", dest="bomb_economy",
                        action="store_true", default=_P["bomb_economy"],
                        help="Unified value scoring: only bomb when score >= bomb_reserve_threshold")
    parser.add_argument("--no-bomb-economy", dest="bomb_economy",
                        action="store_false")
    parser.add_argument("--base-bomb-value", type=float, default=_P["base_bomb_value"],
                        help="Value of hitting an enemy base in agent-hit units")
    parser.add_argument("--agent-bomb-value", type=float, default=_P["agent_bomb_value"],
                        help="Value of a single definite agent hit")
    parser.add_argument("--bomb-reserve-threshold", type=float,
                        default=_P["bomb_reserve_threshold"],
                        help="Minimum score required to place a bomb under economy mode")
    parser.add_argument("--wall-break-tile-threshold", type=float,
                        default=_P["wall_break_tile_threshold"],
                        help="Min tile value behind wall to justify a wall-break bomb; "
                             "0.0 = always break")

    parser.add_argument("--loop-detection", dest="loop_detection",
                        action="store_true", default=_P["loop_detection"],
                        help="Detect and break 2- or 3-step (action, position) cycles")
    parser.add_argument("--no-loop-detection", dest="loop_detection",
                        action="store_false")
    parser.add_argument("--loop-window", type=int, default=_P["loop_window"],
                        help="Past (action, pos) entries retained for cycle detection; "
                             "must be >= 5 to catch period-3 loops")

    parser.add_argument("--proactive-base-routing", dest="proactive_base_routing",
                        action="store_true", default=_P["proactive_base_routing"],
                        help="Include known enemy base cells in collect scoring")
    parser.add_argument("--no-proactive-base-routing", dest="proactive_base_routing",
                        action="store_false")
    parser.add_argument("--base-route-weight", type=float, default=_P["base_route_weight"],
                        help="Synthetic tile value for enemy base routing "
                             "(comparable to MISSION=5, RESOURCE=2, RECON=1)")

    parser.add_argument("--adaptive-base-weight", dest="adaptive_base_weight",
                        action="store_true", default=_P["adaptive_base_weight"],
                        help="Auto-adjust base-route weight based on enemy aggression "
                             "(requires --proactive-base-routing)")
    parser.add_argument("--no-adaptive-base-weight", dest="adaptive_base_weight",
                        action="store_false")
    parser.add_argument("--base-weight-min", type=float, default=_P["base_weight_min"],
                        help="Floor weight after a detected attack")
    parser.add_argument("--base-weight-ramp-rate", type=float,
                        default=_P["base_weight_ramp_rate"],
                        help="Weight increase per step during the ramp phase")
    parser.add_argument("--base-weight-attack-cooldown", type=int,
                        default=_P["base_weight_attack_cooldown"],
                        help="Steps to hold defensive posture after last attack "
                             "before ramping resumes")

    parser.add_argument("--cache", dest="cache_path", type=Path,
                        default=DEFAULT_CACHE_PATH,
                        help="Pre-load this novice-map cache (default: ae/src/novice_map.json)")
    parser.add_argument("--no-cache", dest="cache_path", action="store_const", const=None,
                        help="Start with empty map memory (for benchmarking)")

    parser.add_argument("--grid-offset-x", type=int, default=0,
                        help="Pixel X offset of the grid's top-left corner (analysis overlay calibration)")
    parser.add_argument("--grid-offset-y", type=int, default=0,
                        help="Pixel Y offset of the grid's top-left corner (analysis overlay calibration)")

    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else default_config()
    cfg.env.render_mode = "human"
    cfg.env.novice = args.novice
    if args.fps is not None:
        cfg.renderer.render_fps = int(args.fps)

    env = Bomberman(cfg)
    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    env.reset(seed=seed)

    def make_policy() -> HeuristicPolicy:
        return HeuristicPolicy(
            predictive_bomb=args.predictive_bomb,
            predictive_bomb_threshold=args.bomb_threshold,
            wall_breaking=args.wall_breaking,
            wall_break_cost=args.wall_break_cost,
            adaptive_wall_break_cost=args.adaptive_wall_break_cost,
            smart_defend=args.smart_defend,
            predictive_defend=args.predictive_defend,
            drift_aware_bomb=args.drift_aware_bomb,
            auto_tune_bomb=args.auto_tune_bomb,
            bomb_tune_target=args.bomb_tune_target,
            bomb_economy=args.bomb_economy,
            base_bomb_value=args.base_bomb_value,
            agent_bomb_value=args.agent_bomb_value,
            bomb_reserve_threshold=args.bomb_reserve_threshold,
            wall_break_tile_threshold=args.wall_break_tile_threshold,
            loop_detection=args.loop_detection,
            loop_window=args.loop_window,
            proactive_base_routing=args.proactive_base_routing,
            base_route_weight=args.base_route_weight,
            adaptive_base_weight=args.adaptive_base_weight,
            base_weight_min=args.base_weight_min,
            base_weight_ramp_rate=args.base_weight_ramp_rate,
            base_weight_attack_cooldown=args.base_weight_attack_cooldown,
        )

    managers = _build_managers(env, make_policy, args.cache_path)
    selected_view = env.possible_agents[0]  # camera/highlight follows this agent

    cache_used = args.cache_path is not None and args.cache_path.exists()
    features = []
    features.append(f"predictive_bomb={'on' if args.predictive_bomb else 'off'}"
                    + (f" (≥{args.bomb_threshold})" if args.predictive_bomb else ""))
    features.append(f"wall_breaking={'on' if args.wall_breaking else 'off'}"
                    + (f" (cost={args.wall_break_cost})" if args.wall_breaking else ""))
    features.append(f"smart_defend={'on' if args.smart_defend else 'off'}")
    features.append(f"drift_aware={'on' if args.drift_aware_bomb else 'off'}")
    features.append(f"auto_tune={'on (target={args.bomb_tune_target})' if args.auto_tune_bomb else 'off'}")
    features.append(f"loop_detection={'on (window=' + str(args.loop_window) + ')' if args.loop_detection else 'off'}")
    features.append(f"proactive_base_routing={'on (weight=' + str(args.base_route_weight) + ')' if args.proactive_base_routing else 'off'}")
    if args.adaptive_base_weight:
        features.append(f"adaptive_base_weight=on (min={args.base_weight_min}, "
                        f"ramp={args.base_weight_ramp_rate}, cooldown={args.base_weight_attack_cooldown})")
    features.append(f"map_cache={'on (' + args.cache_path.name + ')' if cache_used else 'off'}")
    print(f"Auto-play: {len(env.possible_agents)} HeuristicPolicy bots, seed={seed}, "
          f"novice={args.novice}, rounds={args.rounds}")
    print(f"Features:  {' · '.join(features)}")
    print("Keys: Q/ESC quit · R reset · T respawn overlay · SPACE pause · ←→ step · A analysis overlay")

    pygame.font.init()
    font = pygame.font.SysFont("monospace", 12, bold=True)

    clock = pygame.time.Clock()
    show_respawn = False
    running = True
    rounds_done = 0

    # Analysis / history state
    history: deque[HistoryFrame] = deque(maxlen=HISTORY_MAXLEN)
    history_pos: int = 0
    paused: bool = False
    show_analysis: bool = False
    selected_agent: Optional[str] = None
    tile_w: int = 32   # updated after first render
    tile_h: int = 32
    grid_offset_x: int = args.grid_offset_x
    grid_offset_y: int = args.grid_offset_y

    while running and rounds_done < args.rounds:
        if not paused:
            # ── live mode: advance + render ──────────────────────────────────
            if env.agent_selector.is_first():
                # Snapshot debug state BEFORE rendering (reflects last tick's decisions).
                agent_info = _collect_debug_info(managers)

                overlay = env.dynamics.respawn_map if show_respawn else None
                env.render(selected_agent_id=selected_view, respawn_overlay=overlay)

                # Capture the rendered frame and update tile sizing once.
                screen = pygame.display.get_surface()
                if screen is not None:
                    if tile_w == 32 and tile_h == 32:   # first time
                        tile_w = max(1, screen.get_width() // GRID_SIZE)
                        tile_h = max(1, screen.get_height() // GRID_SIZE)
                    frame = HistoryFrame(surface=screen.copy(), agent_info=agent_info)
                    history.append(frame)
                    history_pos = len(history) - 1

                    if show_analysis:
                        _draw_analysis_overlay(
                            screen, frame,
                            tile_w, tile_h, grid_offset_x, grid_offset_y,
                            selected_agent, font,
                        )
                    pygame.display.flip()

                clock.tick(env.cfg.renderer.render_fps)
        else:
            # ── paused mode: display historical frame ─────────────────────────
            if history:
                screen = pygame.display.get_surface()
                if screen is not None:
                    frame = history[history_pos]
                    screen.blit(frame.surface, (0, 0))
                    if show_analysis:
                        _draw_analysis_overlay(
                            screen, frame,
                            tile_w, tile_h, grid_offset_x, grid_offset_y,
                            selected_agent, font,
                        )
                    _draw_pause_hud(screen, history_pos, len(history), font)
                    pygame.display.flip()
            clock.tick(env.cfg.renderer.render_fps)

        # ── event handling ────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and show_analysis and history:
                mx, my = event.pos
                gx = (mx - grid_offset_x) // tile_w
                gy = (my - grid_offset_y) // tile_h
                frame = history[history_pos]
                hit = next(
                    (ag for ag, info in frame.agent_info.items() if info.pos == (gx, gy)),
                    None,
                )
                selected_agent = hit   # None deselects
                if hit:
                    print(f"[analysis] selected {hit} at ({gx},{gy})")

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    if not paused and history:
                        history_pos = len(history) - 1  # snap back to live
                    print(f"[{'PAUSED' if paused else 'LIVE'}]")
                elif event.key == pygame.K_LEFT and paused:
                    history_pos = max(0, history_pos - 1)
                elif event.key == pygame.K_RIGHT and paused:
                    history_pos = min(len(history) - 1, history_pos + 1)
                elif event.key == pygame.K_a:
                    show_analysis = not show_analysis
                    if not show_analysis:
                        selected_agent = None
                    print(f"[analysis overlay] {'ON' if show_analysis else 'OFF'}")
                elif event.key == pygame.K_r:
                    seed = random.randint(0, 99999)
                    env.reset(seed=seed)
                    _reset_managers(managers)
                    history.clear()
                    history_pos = 0
                    paused = False
                    selected_agent = None
                    print(f"[reset] new seed={seed}")
                elif event.key == pygame.K_t:
                    show_respawn = not show_respawn
                    print(f"[respawn overlay] {'ON' if show_respawn else 'OFF'}")

        if not running:
            break

        # When paused: don't advance the simulation.
        if paused:
            continue

        agent = env.agent_selection

        # Episode-end housekeeping for this agent.
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            if all(env.terminations.values()) or all(env.truncations.values()):
                rounds_done += 1
                _print_round_summary(env, rounds_done)
                if args.auto_tune_bomb:
                    sample_policy = next(iter(managers.values()))._policy
                    print(f"  [auto-tune] threshold={sample_policy.tuned_threshold:.3f}  "
                          f"hit_ema={sample_policy._hit_ema:.3f}")
                if rounds_done < args.rounds:
                    seed = random.randint(0, 99999)
                    env.reset(seed=seed)
                    _reset_managers(managers)
            continue

        # Policy chooses; AEManager catches its own exceptions and falls back
        # to STAY, so a buggy bot can't take down the whole visualization.
        obs = env.observe(agent)
        action = managers[agent].ae(obs)
        env.step(int(action))

    env.close()
    print(f"\nDone. Played {rounds_done} round(s).")


if __name__ == "__main__":
    main()
