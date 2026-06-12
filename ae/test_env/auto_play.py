"""Auto-play and benchmark for multiple AE agent types.

Agent types
-----------
  normal              — HeuristicPolicy (balanced: dodge, attack, defend, collect, explore)
  berserker           — BerserkerPolicy (rush enemy bases, ignore self-preservation)
  berserker_base      — BerserkerBasePolicy (heuristic base, berserker-style aggression)
  random              — RandomPolicy (uniform sample over legal actions)

Visual mode (default):
    python ae/test_env/auto_play.py
    python ae/test_env/auto_play.py --agent-type berserker
    python ae/test_env/auto_play.py --agent-types berserker normal normal normal normal normal
    python ae/test_env/auto_play.py --rounds 3 --seed 42 --fps 4
    python ae/test_env/auto_play.py --action-log ae/test_env/action_logs/normal.txt

Headless benchmark (compare selected types, no window):
    python ae/test_env/auto_play.py --benchmark --rounds 30
    python ae/test_env/auto_play.py --benchmark --rounds 50 --novice

Keys during play:
    Q / ESC   quit
    R         reset to a new round
    T         toggle the tile-respawn-timer overlay
    SPACE     pause / resume
    ←→        step through history when paused
    A         toggle analysis overlay
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional

import pygame

# Make ae/src importable as flat top-level modules (matching Docker layout).
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from til_environment.bomberman_env import Bomberman  # noqa: E402
from til_environment.config import default_config, load_config  # noqa: E402

from ae_manager import DEFAULT_CACHE_PATH, DEFAULT_POLICY_KWARGS, AEManager  # noqa: E402
from policies.azbase_berserker_base_policy import (  # noqa: E402
    BerserkerBasePolicy as BerserkerBaseAzbasePolicy,
)
from policies.azbasev3_policy import BerserkerBasePolicy  # noqa: E402
from policies.berserker_base_submit_policy import BerserkerBaseSubmitPolicy  # noqa: E402
from policies.azbasev4_policy import BerserkerBaseV4Policy  # noqa: E402
from policies.berserker_policy import BerserkerPolicy  # noqa: E402
from constants import Action, GRID_SIZE  # noqa: E402
from policies.edited_policy import EditedHeuristicPolicy as HeuristicPolicy  # noqa: E402

# Comment the line below to benchmark the plain edited_policy instead of the
# experimental clone (mirrors the toggle in ae_manager.py).
from policies.edited_policy_v2 import EditedHeuristicPolicyV2 as HeuristicPolicy  # noqa: E402
from map_memory import MapMemory  # noqa: E402
from observation import ParsedObs  # noqa: E402
from policy import Policy  # noqa: E402
from policies.scoremax_policy import ScoreMaxPolicy  # noqa: E402


AGENT_TYPES = (
    "normal",
    "berserker",
    "berserker_base",
    "berserker_base_v4",
    "berserker_base_azbase",
    "berserker_base_submit",
    "scoremax",
    "random",
)

DEFAULT_BENCHMARK_TYPES = AGENT_TYPES
DEFAULT_ACTION_LOG = HERE / "action_logs" / "auto_play_actions.txt"


class RandomPolicy(Policy):
    """Picks uniformly at random from legal actions each tick."""

    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:  # noqa: ARG002
        valid = [i for i in range(len(obs.action_mask)) if obs.action_mask[i]]
        return int(random.choice(valid)) if valid else int(Action.STAY)


def _make_policy(
    agent_type: str,
    policy_kwargs: dict,
) -> Policy:
    if agent_type == "normal":
        policy: Policy = HeuristicPolicy(**policy_kwargs)
    elif agent_type == "berserker":
        policy = BerserkerPolicy()
    elif agent_type == "berserker_base":
        policy = BerserkerBasePolicy(**policy_kwargs)
    elif agent_type == "berserker_base_v4":
        policy = BerserkerBaseV4Policy(**policy_kwargs)
    elif agent_type == "berserker_base_azbase":
        policy = BerserkerBaseAzbasePolicy(**policy_kwargs)
    elif agent_type == "berserker_base_submit":
        policy = BerserkerBaseSubmitPolicy(**policy_kwargs)
    elif agent_type == "scoremax":
        policy = ScoreMaxPolicy(**policy_kwargs)
    elif agent_type == "random":
        policy = RandomPolicy()
    else:
        raise ValueError(f"unknown agent type: {agent_type}")

    return policy


def _make_factories(
    agents: list[str],
    types: list[str],
    policy_kwargs: dict,
) -> dict[str, callable]:
    """Return a per-agent-id dict of zero-arg callables that produce a Policy."""
    out: dict[str, callable] = {}
    for agent, t in zip(agents, types):
        t_ = t  # capture loop variable
        out[agent] = lambda t=t_: _make_policy(t, policy_kwargs)
    return out


def _build_managers(
    env: Bomberman,
    per_agent_factories: dict[str, callable],
    cache_path: Optional[Path],
) -> tuple[dict[str, AEManager], MapMemory | None]:
    """One AEManager per agent, each with an isolated MapMemory.

    Returns (managers, cached_template) so callers can pass the template to
    _reset_managers and restore static knowledge (base positions, walls) that
    may have been mutated during a round.
    """
    cached_template: MapMemory | None = None
    if cache_path is not None and cache_path.exists():
        cached_template = MapMemory.load(cache_path)

    out: dict[str, AEManager] = {}
    for agent in env.possible_agents:
        mem = MapMemory()
        if cached_template is not None:
            mem.merge_static_from(cached_template)
        out[agent] = AEManager(policy=per_agent_factories[agent](), memory=mem)
    return out, cached_template


def _reset_managers(
    managers: dict[str, AEManager],
    cached_template: MapMemory | None = None,
) -> None:
    """Reset per-round dynamic state and restore static knowledge from cache.

    Without the restore step, base positions that were discarded when the agent
    observed a destroyed base (ENEMY_BASE=0) stay missing in subsequent rounds,
    even though the bases respawn. The production server avoids this by
    recreating AEManager (and thus re-loading the cache) on every /reset.
    """
    for mgr in managers.values():
        mgr._memory.reset_round()
        if cached_template is not None:
            mgr._memory.merge_static_from(cached_template)


def _print_round_summary(
    env: Bomberman,
    round_idx: int,
    agent_types: dict[str, str],
) -> None:
    episode = getattr(env.dynamics.rewards, "_episode", {})
    print(f"\n── round {round_idx} over ──")
    rewards = sorted(
        ((a, float(episode.get(a, 0.0))) for a in env.possible_agents),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for a, r in rewards:
        t = agent_types.get(a, "normal")
        print(f"  {a}  [{t:9s}]  reward={r:.2f}")


def _action_name(action: int | None) -> str:
    if action is None:
        return "NONE"
    try:
        return Action(int(action)).name
    except ValueError:
        return f"UNKNOWN_{action}"


def _as_scalar(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        return _as_scalar(value[0])
    return value


def _fmt_scalar(value: Any) -> str:
    value = _as_scalar(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_pair(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f"{int(value[0])},{int(value[1])}"
    return str(value)


class ActionLogger:
    """Writes one tab-separated row for every non-terminal agent action."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._fh = None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def open(
        self,
        *,
        mode: str,
        novice: bool,
        rounds: int,
        selected_types: tuple[str, ...],
    ) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8", newline="")
        self._fh.write("# auto_play action log\n")
        self._fh.write(f"# created_at={datetime.now().isoformat(timespec='seconds')}\n")
        self._fh.write(f"# mode={mode}\n")
        self._fh.write(f"# novice={novice}\n")
        self._fh.write(f"# rounds={rounds}\n")
        self._fh.write(f"# selected_types={','.join(selected_types)}\n")
        self._fh.write(
            "round\tseed\tstep\tagent\ttype\taction\taction_name\t"
            "location\tdirection\thealth\tfrozen\tbase_health\tresources\t"
            "bombs\tmode\ttarget\n"
        )
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def log_event(self, message: str) -> None:
        if self._fh is None:
            return
        self._fh.write(f"# {message}\n")
        self._fh.flush()

    def start_round(self, round_idx: int, seed: int, label: str = "") -> None:
        suffix = f" {label}" if label else ""
        self.log_event(f"round {round_idx} seed={seed}{suffix}")

    def log_action(
        self,
        *,
        round_idx: int,
        seed: int,
        agent: str,
        agent_type: str,
        obs: dict[str, Any],
        action: int,
        manager: AEManager,
    ) -> None:
        if self._fh is None:
            return

        policy = manager._policy
        self._fh.write(
            f"{round_idx}\t"
            f"{seed}\t"
            f"{_fmt_scalar(obs.get('step', ''))}\t"
            f"{agent}\t"
            f"{agent_type}\t"
            f"{int(action)}\t"
            f"{_action_name(int(action))}\t"
            f"{_fmt_pair(obs.get('location', ''))}\t"
            f"{_fmt_scalar(obs.get('direction', ''))}\t"
            f"{_fmt_scalar(obs.get('health', ''))}\t"
            f"{_fmt_scalar(obs.get('frozen_ticks', ''))}\t"
            f"{_fmt_scalar(obs.get('base_health', ''))}\t"
            f"{_fmt_scalar(obs.get('team_resources', ''))}\t"
            f"{_fmt_scalar(obs.get('team_bombs', ''))}\t"
            f"{getattr(policy, '_debug_mode', '')}\t"
            f"{_fmt_pair(getattr(policy, '_debug_target', ''))}\n"
        )
        self._fh.flush()

    def log_reward_breakdown(
        self,
        *,
        round_idx: int,
        seed: int,
        agent_types: dict[str, str],
        tracker: "RewardBreakdownTracker",
        env: Bomberman,
    ) -> None:
        if self._fh is None:
            return
        header = (
            "# reward_breakdown_header\t"
            "round\tseed\tagent\ttype\ttotal\tattack_damage_pos\t"
            "attack_damage_neg\tattack_kill\tdestroy_enemy_base\t"
            "own_base_destroyed\tcollect_mission\tcollect_resource\tcollect_recon\t"
            "other\n"
        )
        self._fh.write(header)
        for row in tracker.rows(env, agent_types):
            self._fh.write(
                "# reward_breakdown\t"
                f"{round_idx}\t{seed}\t{row['agent']}\t{row['type']}\t"
                f"{row['total']:.3f}\t"
                f"{row['attack_damage_pos']:.3f}\t"
                f"{row['attack_damage_neg']:.3f}\t"
                f"{row['attack_kill']:.3f}\t"
                f"{row['destroy_enemy_base']:.3f}\t"
                f"{row['own_base_destroyed']:.3f}\t"
                f"{row['collect_mission']:.3f}\t"
                f"{row['collect_resource']:.3f}\t"
                f"{row['collect_recon']:.3f}\t"
                f"{row['other']:.3f}\n"
            )
        self._fh.flush()


class RewardBreakdownTracker:
    """Tracks signed reward-event buckets by wrapping env.dynamics.rewards.award."""

    TRACKED_EVENTS = (
        "attack_damage",
        "attack_kill",
        "destroy_enemy_base",
        "own_base_destroyed",
        "collect_mission",
        "collect_resource",
        "collect_recon",
    )

    def __init__(self, env: Bomberman) -> None:
        self._env = env
        self._events: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        rewards = env.dynamics.rewards
        self._original_award = rewards.award

        def award(recipient_id: str, event: str, multiplier: float = 1.0) -> float:
            value = float(self._original_award(recipient_id, event, multiplier))
            if value != 0.0:
                self._record(recipient_id, event, value)
            return value

        rewards.award = award

    def reset(self) -> None:
        self._events.clear()

    def _record(self, agent: str, event: str, value: float) -> None:
        buckets = self._events[agent]
        buckets[event] += value
        if event == "attack_damage":
            key = "attack_damage_pos" if value > 0 else "attack_damage_neg"
            buckets[key] += value

    def rows(self, env: Bomberman, agent_types: dict[str, str]) -> list[dict[str, Any]]:
        episode = getattr(env.dynamics.rewards, "_episode", {})
        rows: list[dict[str, Any]] = []
        for agent in env.possible_agents:
            buckets = self._events.get(agent, {})
            tracked_total = sum(float(buckets.get(k, 0.0)) for k in self.TRACKED_EVENTS)
            total = float(episode.get(agent, 0.0))
            rows.append(
                {
                    "agent": agent,
                    "type": agent_types.get(agent, "normal"),
                    "total": total,
                    "attack_damage_pos": float(buckets.get("attack_damage_pos", 0.0)),
                    "attack_damage_neg": float(buckets.get("attack_damage_neg", 0.0)),
                    "attack_kill": float(buckets.get("attack_kill", 0.0)),
                    "destroy_enemy_base": float(
                        buckets.get("destroy_enemy_base", 0.0)
                    ),
                    "own_base_destroyed": float(
                        buckets.get("own_base_destroyed", 0.0)
                    ),
                    "collect_mission": float(buckets.get("collect_mission", 0.0)),
                    "collect_resource": float(buckets.get("collect_resource", 0.0)),
                    "collect_recon": float(buckets.get("collect_recon", 0.0)),
                    "other": total - tracked_total,
                }
            )
        return rows


# ── headless benchmark ──────────────────────────────────────────────────────


def _run_benchmark(
    rounds: int,
    novice: bool,
    policy_kwargs: dict,
    cache_path: Optional[Path],
    action_logger: ActionLogger,
    agent_types: tuple[str, ...] = DEFAULT_BENCHMARK_TYPES,
) -> None:
    """Headless comparison: run each type for *rounds* rounds, then print table.

    All agents in a game use the same type so competition strength is equal.
    Scores are averaged across all 6 agents per round (they're all the same type).
    """
    if novice:
        print(
            "\n[WARNING] Novice mode uses a hardcoded env seed — all rounds play the\n"
            "  same fixed map with the same starting positions. Deterministic agents\n"
            "  (normal, berserker) will produce identical scores every round (stdev≈0).\n"
            "  Only random varies. Use --advanced for statistically independent rounds."
        )
    results: dict[str, list[float]] = {}

    for agent_type in agent_types:
        print(f"\nBenchmarking [{agent_type}] — {rounds} rounds …", flush=True)
        cfg = default_config()
        cfg.env.novice = novice

        env = Bomberman(cfg)
        seed = random.randint(0, 99999)
        env.reset(seed=seed)
        reward_tracker = RewardBreakdownTracker(env)

        n_agents = len(env.possible_agents)
        factories = _make_factories(
            env.possible_agents,
            [agent_type] * n_agents,
            policy_kwargs,
        )
        managers, cache_tmpl = _build_managers(env, factories, cache_path)
        agent_type_map = {agent: agent_type for agent in env.possible_agents}

        round_scores: list[float] = []
        for round_idx in range(rounds):
            reward_tracker.reset()
            action_logger.start_round(
                round_idx + 1,
                seed,
                label=f"benchmark type={agent_type}",
            )
            # Run one round.
            while True:
                agent = env.agent_selection
                if env.terminations[agent] or env.truncations[agent]:
                    env.step(None)
                    if all(env.terminations.values()) or all(env.truncations.values()):
                        break
                    continue
                obs = env.observe(agent)
                action = managers[agent].ae(obs)
                action_logger.log_action(
                    round_idx=round_idx + 1,
                    seed=seed,
                    agent=agent,
                    agent_type=agent_type_map[agent],
                    obs=obs,
                    action=int(action),
                    manager=managers[agent],
                )
                env.step(int(action))

            episode = getattr(env.dynamics.rewards, "_episode", {})
            per_agent = [float(episode.get(a, 0.0)) for a in env.possible_agents]
            avg = mean(per_agent)
            best = max(per_agent)
            action_logger.log_reward_breakdown(
                round_idx=round_idx + 1,
                seed=seed,
                agent_types=agent_type_map,
                tracker=reward_tracker,
                env=env,
            )
            round_scores.append(avg)
            print(
                f"  round {round_idx + 1:3d}  avg={avg:7.1f}  max={best:7.1f}",
                flush=True,
            )

            if round_idx < rounds - 1:
                seed = random.randint(0, 99999)
                env.reset(seed=seed)
                _reset_managers(managers, cache_tmpl)

        results[agent_type] = round_scores
        env.close()

    # ── comparison table ────────────────────────────────────────────────────
    width = 62
    print("\n" + "═" * width)
    print(f"  {'TYPE':10s}  {'MEAN':>8s}  {'MAX':>8s}  {'MIN':>8s}  {'STD':>8s}")
    print("─" * width)
    for agent_type, scores in results.items():
        if not scores:
            continue
        s_mean = mean(scores)
        s_max = max(scores)
        s_min = min(scores)
        s_std = stdev(scores) if len(scores) > 1 else 0.0
        print(
            f"  {agent_type:10s}  {s_mean:8.2f}  {s_max:8.2f}  {s_min:8.2f}  {s_std:8.2f}"
        )
    print("═" * width)

    if results:
        best_type = max(
            results, key=lambda t: mean(results[t]) if results[t] else float("-inf")
        )
        print(f"\n  Best by mean score: [{best_type}]")

        best_max_type = max(
            results, key=lambda t: max(results[t]) if results[t] else float("-inf")
        )
        print(
            f"  Best single-round score: [{best_max_type}] ({max(results[best_max_type]):.2f})"
        )


def _run_matchup_benchmark(
    rounds: int,
    novice: bool,
    policy_kwargs: dict,
    cache_path: Optional[Path],
    action_logger: ActionLogger,
    agent_types: tuple[str, ...] = DEFAULT_BENCHMARK_TYPES,
) -> None:
    """Headless cross-type matchup: for every (focus, opponent) pair run *rounds* rounds.

    agent_0 uses `focus` type; agents 1-5 use `opponent` type. Tracks the focus
    agent's score and the opponents' mean score separately, then prints a matrix.

    In novice mode the env seed is hardcoded, so all rounds share the same map and
    starting positions. Deterministic agents (normal, berserker) produce identical
    scores every round (stdev=0 in the matrix). Only random actually varies.
    Use --advanced for statistically independent rounds.

    Usage:
        python ae/test_env/auto_play.py --benchmark-matchup --rounds 20
        python ae/test_env/auto_play.py --benchmark-matchup --rounds 20 --advanced
    """
    if novice:
        print(
            "\n[WARNING] Novice mode uses a hardcoded env seed — all rounds play the\n"
            "  same fixed map. Deterministic agents will show stdev=0 in the matrix.\n"
            "  Use --advanced for independent rounds across different maps."
        )
    # matchups[(focus, opp)] = {"focus": [scores…], "opp": [mean scores…]}
    matchups: dict[tuple[str, str], dict[str, list[float]]] = {}

    for focus_type in agent_types:
        for opp_type in agent_types:
            key = (focus_type, opp_type)
            matchups[key] = {"focus": [], "opp": []}

            print(
                f"\nMatchup: 1×[{focus_type}] vs 5×[{opp_type}] — {rounds} rounds …",
                flush=True,
            )

            cfg = default_config()
            cfg.env.novice = novice
            env = Bomberman(cfg)
            seed = random.randint(0, 99999)
            env.reset(seed=seed)
            reward_tracker = RewardBreakdownTracker(env)

            n_agents = len(env.possible_agents)
            type_list = [focus_type] + [opp_type] * (n_agents - 1)
            factories = _make_factories(
                env.possible_agents,
                type_list,
                policy_kwargs,
            )
            managers, cache_tmpl = _build_managers(env, factories, cache_path)
            agent_type_map = dict(zip(env.possible_agents, type_list))

            focus_agent = env.possible_agents[0]
            opp_agents = env.possible_agents[1:]

            for round_idx in range(rounds):
                reward_tracker.reset()
                action_logger.start_round(
                    round_idx + 1,
                    seed,
                    label=f"matchup focus={focus_type} opponent={opp_type}",
                )
                while True:
                    agent = env.agent_selection
                    if env.terminations[agent] or env.truncations[agent]:
                        env.step(None)
                        if all(env.terminations.values()) or all(
                            env.truncations.values()
                        ):
                            break
                        continue
                    obs = env.observe(agent)
                    action = managers[agent].ae(obs)
                    action_logger.log_action(
                        round_idx=round_idx + 1,
                        seed=seed,
                        agent=agent,
                        agent_type=agent_type_map[agent],
                        obs=obs,
                        action=int(action),
                        manager=managers[agent],
                    )
                    env.step(int(action))

                episode = getattr(env.dynamics.rewards, "_episode", {})
                focus_score = float(episode.get(focus_agent, 0.0))
                opp_scores = [float(episode.get(a, 0.0)) for a in opp_agents]
                opp_mean = mean(opp_scores)
                action_logger.log_reward_breakdown(
                    round_idx=round_idx + 1,
                    seed=seed,
                    agent_types=agent_type_map,
                    tracker=reward_tracker,
                    env=env,
                )
                matchups[key]["focus"].append(focus_score)
                matchups[key]["opp"].append(opp_mean)
                print(
                    f"  round {round_idx + 1:3d}"
                    f"  focus={focus_score:7.1f}"
                    f"  opp_avg={opp_mean:7.1f}",
                    flush=True,
                )

                if round_idx < rounds - 1:
                    seed = random.randint(0, 99999)
                    env.reset(seed=seed)
                    _reset_managers(managers, cache_tmpl)

            env.close()

    # ── results matrix ──────────────────────────────────────────────────────
    # Each cell: "mean±std". stdev=0 means all rounds were identical (fixed seed).
    def _cell(scores: list[float]) -> str:
        if not scores:
            return "  N/A"
        m = mean(scores)
        s = stdev(scores) if len(scores) > 1 else 0.0
        return f"{m:6.1f}±{s:4.1f}"

    col_w = 14
    sep = "─" * (16 + col_w * len(agent_types))
    print(f"\n{'═' * len(sep)}")
    print("MATCHUP RESULTS  — focus-agent score  mean±std  (row=focus, col=opponents)")
    print("  stdev=0 means every round was identical (fixed novice seed).")
    print(f"  {'':14s}" + "".join(f"{t:>{col_w}s}" for t in agent_types))
    print(sep)
    for focus_type in agent_types:
        row = f"  {focus_type:<14s}"
        for opp_type in agent_types:
            row += f"{_cell(matchups[(focus_type, opp_type)]['focus']):>{col_w}s}"
        print(row)
    print(sep)
    print("  OPPONENT mean score (same layout)")
    print(f"  {'':14s}" + "".join(f"{t:>{col_w}s}" for t in agent_types))
    print(sep)
    for focus_type in agent_types:
        row = f"  {focus_type:<14s}"
        for opp_type in agent_types:
            row += f"{_cell(matchups[(focus_type, opp_type)]['opp']):>{col_w}s}"
        print(row)
    print("═" * len(sep))

    # Best performer by mean focus score (off-diagonal: hardest matchup excluded).
    best = max(
        ((f, o) for f in agent_types for o in agent_types if f != o),
        key=lambda k: (
            mean(matchups[k]["focus"]) if matchups[k]["focus"] else float("-inf")
        ),
    )
    print(
        f"\n  Best cross-type matchup: [{best[0]}] vs [{best[1]}]"
        f"  (focus mean={mean(matchups[best]['focus']):.2f})"
    )


# ── analysis / history types ────────────────────────────────────────────────

HISTORY_MAXLEN = 300

_MODE_COLORS: dict[str, tuple[int, int, int]] = {
    "frozen": (150, 150, 255),
    "dodge": (255, 80, 80),
    "attack": (255, 165, 0),
    "defend": (80, 80, 255),
    "collect": (80, 255, 80),
    "explore": (200, 200, 80),
    "stay": (160, 160, 160),
}

_TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "normal": (100, 200, 255),
    "berserker": (255, 80, 80),
    "berserker_base": (255, 150, 120),
    "berserker_base_v4": (255, 190, 150),
    "berserker_base_azbase": (255, 120, 80),
    "berserker_base_submit": (255, 175, 120),
    "scoremax": (255, 215, 90),
    "random": (200, 200, 80),
}


@dataclass
class AgentDebugInfo:
    mode: str
    target: Optional[tuple[int, int]]
    pos: tuple[int, int]
    policy_type: str = "normal"


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
    return (
        offset_x + pos[0] * tile_w + tile_w // 2,
        offset_y + pos[1] * tile_h + tile_h // 2,
    )


def _collect_debug_info(
    managers: dict[str, AEManager],
    agent_type_map: dict[str, str],
) -> dict[str, AgentDebugInfo]:
    out: dict[str, AgentDebugInfo] = {}
    for agent_id, mgr in managers.items():
        pol = mgr._policy
        out[agent_id] = AgentDebugInfo(
            mode=getattr(pol, "_debug_mode", "stay"),
            target=getattr(pol, "_debug_target", None),
            pos=getattr(pol, "_debug_pos", (0, 0)),
            policy_type=agent_type_map.get(agent_id, "normal"),
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
        f"Type:   {info.policy_type}",
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

    type_color = _TYPE_COLORS.get(info.policy_type, (255, 255, 255))
    mode_color = _MODE_COLORS.get(info.mode, (255, 255, 255))
    colors = [(220, 220, 220), type_color, mode_color, (220, 220, 220), (220, 220, 220)]
    for i, (ln, color) in enumerate(zip(lines, colors)):
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
    for _, info in frame.agent_info.items():
        mode_color = _MODE_COLORS.get(info.mode, (255, 255, 255))
        type_color = _TYPE_COLORS.get(info.policy_type, (255, 255, 255))
        ax, ay = _world_to_screen(info.pos, tile_w, tile_h, offset_x, offset_y)

        if info.target is not None:
            tx, ty = _world_to_screen(info.target, tile_w, tile_h, offset_x, offset_y)
            pygame.draw.line(surface, mode_color, (ax, ay), (tx, ty), 2)
            pygame.draw.circle(surface, mode_color, (tx, ty), 4, 2)

        label = info.mode[:3].upper()
        shadow = font.render(label, True, (0, 0, 0))
        text = font.render(label, True, type_color)
        lx = ax - text.get_width() // 2
        ly = ay - tile_h // 2 - text.get_height() - 2
        surface.blit(shadow, (lx + 1, ly + 1))
        surface.blit(text, (lx, ly))

    if selected_agent and selected_agent in frame.agent_info:
        _draw_agent_panel(
            surface, frame.agent_info[selected_agent], selected_agent, font
        )


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


_P = DEFAULT_POLICY_KWARGS


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ── game / visual options ────────────────────────────────────────────────
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Rounds to play (or benchmark rounds per type)",
    )
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--novice", action="store_true", default=True)
    parser.add_argument("--advanced", dest="novice", action="store_false")
    parser.add_argument(
        "--action-log",
        type=Path,
        default=DEFAULT_ACTION_LOG,
        help=(
            "Write every chosen agent action to this text file "
            f"(default: {DEFAULT_ACTION_LOG})"
        ),
    )
    parser.add_argument(
        "--no-action-log",
        dest="action_log",
        action="store_const",
        const=None,
        help="Disable action logging.",
    )

    # ── agent type selection ─────────────────────────────────────────────────
    parser.add_argument(
        "--agent-type",
        choices=AGENT_TYPES,
        default="normal",
        metavar="TYPE",
        help=f"Policy for all agents: {AGENT_TYPES} (default: normal)",
    )
    parser.add_argument(
        "--agent-types",
        nargs="+",
        choices=AGENT_TYPES,
        metavar="TYPE",
        default=None,
        help=(
            "Per-agent types in agent_0…agent_N order. "
            "Shorter lists are padded with --agent-type. "
            f"Choices: {AGENT_TYPES}"
        ),
    )
    # ── benchmark mode ───────────────────────────────────────────────────────
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Headless: run selected policy types for --rounds rounds, "
            "then print a score comparison table. No window is opened."
        ),
    )
    parser.add_argument(
        "--benchmark-matchup",
        action="store_true",
        help=(
            "Headless: run every (focus, opponent) type pair for --rounds rounds. "
            "agent_0 uses the focus type; agents 1-5 use the opponent type. "
            "Prints a results matrix. No window is opened."
        ),
    )
    parser.add_argument(
        "--benchmark-types",
        nargs="+",
        choices=AGENT_TYPES,
        metavar="TYPE",
        default=None,
        help=(
            "Limit --benchmark/--benchmark-matchup to these types. "
            "Default: all available policy types."
        ),
    )
    # ── heuristic (normal) policy toggles ────────────────────────────────────
    parser.add_argument(
        "--predictive-bomb",
        dest="predictive_bomb",
        action="store_true",
        default=_P["predictive_bomb"],
    )
    parser.add_argument(
        "--no-predictive-bomb", dest="predictive_bomb", action="store_false"
    )
    parser.add_argument(
        "--bomb-threshold", type=float, default=_P["predictive_bomb_threshold"]
    )

    parser.add_argument(
        "--wall-breaking",
        dest="wall_breaking",
        action="store_true",
        default=_P["wall_breaking"],
    )
    parser.add_argument(
        "--no-wall-breaking", dest="wall_breaking", action="store_false"
    )
    parser.add_argument("--wall-break-cost", type=float, default=_P["wall_break_cost"])
    parser.add_argument(
        "--adaptive-wall-break-cost",
        dest="adaptive_wall_break_cost",
        action="store_true",
        default=_P["adaptive_wall_break_cost"],
    )
    parser.add_argument(
        "--no-adaptive-wall-break-cost",
        dest="adaptive_wall_break_cost",
        action="store_false",
    )

    parser.add_argument(
        "--smart-defend",
        dest="smart_defend",
        action="store_true",
        default=_P["smart_defend"],
    )
    parser.add_argument("--no-smart-defend", dest="smart_defend", action="store_false")
    parser.add_argument(
        "--predictive-defend",
        dest="predictive_defend",
        action="store_true",
        default=_P["predictive_defend"],
    )
    parser.add_argument(
        "--no-predictive-defend", dest="predictive_defend", action="store_false"
    )

    parser.add_argument(
        "--drift-aware-bomb",
        dest="drift_aware_bomb",
        action="store_true",
        default=_P["drift_aware_bomb"],
    )
    parser.add_argument(
        "--no-drift-aware-bomb", dest="drift_aware_bomb", action="store_false"
    )

    parser.add_argument(
        "--auto-tune-bomb",
        dest="auto_tune_bomb",
        action="store_true",
        default=_P["auto_tune_bomb"],
    )
    parser.add_argument(
        "--no-auto-tune-bomb", dest="auto_tune_bomb", action="store_false"
    )
    parser.add_argument(
        "--bomb-tune-target", type=float, default=_P["bomb_tune_target"]
    )

    parser.add_argument(
        "--bomb-economy",
        dest="bomb_economy",
        action="store_true",
        default=_P["bomb_economy"],
    )
    parser.add_argument("--no-bomb-economy", dest="bomb_economy", action="store_false")
    parser.add_argument("--base-bomb-value", type=float, default=_P["base_bomb_value"])
    parser.add_argument(
        "--agent-bomb-value", type=float, default=_P["agent_bomb_value"]
    )
    parser.add_argument(
        "--bomb-reserve-threshold", type=float, default=_P["bomb_reserve_threshold"]
    )
    parser.add_argument(
        "--wall-break-tile-threshold",
        type=float,
        default=_P["wall_break_tile_threshold"],
    )

    parser.add_argument(
        "--loop-detection",
        dest="loop_detection",
        action="store_true",
        default=_P["loop_detection"],
    )
    parser.add_argument(
        "--no-loop-detection", dest="loop_detection", action="store_false"
    )
    parser.add_argument("--loop-window", type=int, default=_P["loop_window"])

    parser.add_argument(
        "--proactive-base-routing",
        dest="proactive_base_routing",
        action="store_true",
        default=_P["proactive_base_routing"],
    )
    parser.add_argument(
        "--no-proactive-base-routing",
        dest="proactive_base_routing",
        action="store_false",
    )
    parser.add_argument(
        "--base-route-weight", type=float, default=_P["base_route_weight"]
    )
    parser.add_argument(
        "--adaptive-base-weight",
        dest="adaptive_base_weight",
        action="store_true",
        default=_P["adaptive_base_weight"],
    )
    parser.add_argument(
        "--no-adaptive-base-weight", dest="adaptive_base_weight", action="store_false"
    )
    parser.add_argument("--base-weight-min", type=float, default=_P["base_weight_min"])
    parser.add_argument(
        "--base-weight-ramp-rate", type=float, default=_P["base_weight_ramp_rate"]
    )
    parser.add_argument(
        "--base-weight-attack-cooldown",
        type=int,
        default=_P["base_weight_attack_cooldown"],
    )
    parser.add_argument("--minimum-aggression", type=float, default=2.0)
    parser.add_argument(
        "--aggression-ramp-rate",
        type=float,
        default=0.08,
    )
    parser.add_argument("--defensive-force", type=float, default=0.6)
    parser.add_argument(
        "--defense-abandon-margin",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-defense-distance",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--defense-cooldown-scale",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--cache", dest="cache_path", type=Path, default=DEFAULT_CACHE_PATH
    )
    parser.add_argument(
        "--no-cache", dest="cache_path", action="store_const", const=None
    )
    parser.add_argument("--grid-offset-x", type=int, default=0)
    parser.add_argument("--grid-offset-y", type=int, default=0)

    args = parser.parse_args()

    # Build kwargs for normal (HeuristicPolicy) agents.
    policy_kwargs = dict(
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
        minimum_aggression=args.minimum_aggression,
        aggression_ramp_rate=args.aggression_ramp_rate,
        defensive_force=args.defensive_force,
        defense_abandon_margin=args.defense_abandon_margin,
        max_defense_distance=args.max_defense_distance,
        defense_cooldown_scale=args.defense_cooldown_scale,
    )

    # ── benchmark mode (headless) ─────────────────────────────────────────────
    benchmark_types = (
        tuple(args.benchmark_types) if args.benchmark_types else DEFAULT_BENCHMARK_TYPES
    )
    run_mode = (
        "benchmark-matchup"
        if args.benchmark_matchup
        else "benchmark"
        if args.benchmark
        else "visual"
    )
    selected_types = benchmark_types if (args.benchmark or args.benchmark_matchup) else (
        tuple(args.agent_types) if args.agent_types else (args.agent_type,)
    )
    action_logger = ActionLogger(args.action_log)
    action_logger.open(
        mode=run_mode,
        novice=args.novice,
        rounds=args.rounds,
        selected_types=selected_types,
    )

    if args.benchmark:
        _run_benchmark(
            args.rounds,
            args.novice,
            policy_kwargs,
            args.cache_path,
            action_logger,
            benchmark_types,
        )
        action_logger.close()
        return

    if args.benchmark_matchup:
        _run_matchup_benchmark(
            args.rounds,
            args.novice,
            policy_kwargs,
            args.cache_path,
            action_logger,
            benchmark_types,
        )
        action_logger.close()
        return

    # ── visual mode ───────────────────────────────────────────────────────────
    cfg = load_config(args.config) if args.config else default_config()
    cfg.env.render_mode = "human"
    cfg.env.novice = args.novice
    if args.fps is not None:
        cfg.renderer.render_fps = int(args.fps)

    env = Bomberman(cfg)
    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    env.reset(seed=seed)

    # Resolve per-agent type list.
    n_agents = len(env.possible_agents)
    raw_types = list(args.agent_types) if args.agent_types else []
    # Pad to n_agents with the global default.
    while len(raw_types) < n_agents:
        raw_types.append(args.agent_type)
    agent_types_list = raw_types[:n_agents]
    agent_type_map = dict(zip(env.possible_agents, agent_types_list))

    factories = _make_factories(
        env.possible_agents,
        agent_types_list,
        policy_kwargs,
    )
    managers, cached_template = _build_managers(env, factories, args.cache_path)
    reward_tracker = RewardBreakdownTracker(env)

    type_summary = "  ".join(f"{a}={t}" for a, t in agent_type_map.items())
    print(f"Auto-play: seed={seed}, novice={args.novice}, rounds={args.rounds}")
    print(f"Agents:    {type_summary}")
    print(
        "Keys: Q/ESC quit · R reset · T respawn overlay · SPACE pause · ←→ step · A analysis overlay"
    )

    pygame.font.init()
    font = pygame.font.SysFont("monospace", 12, bold=True)

    clock = pygame.time.Clock()
    show_respawn = False
    running = True
    rounds_done = 0

    history: deque[HistoryFrame] = deque(maxlen=HISTORY_MAXLEN)
    history_pos: int = 0
    paused: bool = False
    show_analysis: bool = False
    selected_agent: Optional[str] = None
    tile_w: int = 32
    tile_h: int = 32
    grid_offset_x: int = args.grid_offset_x
    grid_offset_y: int = args.grid_offset_y

    selected_view = env.possible_agents[0]
    reward_tracker.reset()
    action_logger.start_round(rounds_done + 1, seed, label="visual")

    while running and rounds_done < args.rounds:
        if not paused:
            if env.agent_selector.is_first():
                agent_info = _collect_debug_info(managers, agent_type_map)

                overlay = env.dynamics.respawn_map if show_respawn else None
                env.render(selected_agent_id=selected_view, respawn_overlay=overlay)

                screen = pygame.display.get_surface()
                if screen is not None:
                    if tile_w == 32 and tile_h == 32:
                        tile_w = max(1, screen.get_width() // GRID_SIZE)
                        tile_h = max(1, screen.get_height() // GRID_SIZE)
                    frame = HistoryFrame(surface=screen.copy(), agent_info=agent_info)
                    history.append(frame)
                    history_pos = len(history) - 1

                    if show_analysis:
                        _draw_analysis_overlay(
                            screen,
                            frame,
                            tile_w,
                            tile_h,
                            grid_offset_x,
                            grid_offset_y,
                            selected_agent,
                            font,
                        )
                    pygame.display.flip()

                clock.tick(env.cfg.renderer.render_fps)
        else:
            if history:
                screen = pygame.display.get_surface()
                if screen is not None:
                    frame = history[history_pos]
                    screen.blit(frame.surface, (0, 0))
                    if show_analysis:
                        _draw_analysis_overlay(
                            screen,
                            frame,
                            tile_w,
                            tile_h,
                            grid_offset_x,
                            grid_offset_y,
                            selected_agent,
                            font,
                        )
                    _draw_pause_hud(screen, history_pos, len(history), font)
                    pygame.display.flip()
            clock.tick(env.cfg.renderer.render_fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and show_analysis and history:
                mx, my = event.pos
                gx = (mx - grid_offset_x) // tile_w
                gy = (my - grid_offset_y) // tile_h
                frame = history[history_pos]
                hit = next(
                    (
                        ag
                        for ag, info in frame.agent_info.items()
                        if info.pos == (gx, gy)
                    ),
                    None,
                )
                selected_agent = hit
                if hit:
                    print(
                        f"[analysis] selected {hit} ({agent_type_map.get(hit, '?')}) at ({gx},{gy})"
                    )

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    if not paused and history:
                        history_pos = len(history) - 1
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
                    _reset_managers(managers, cached_template)
                    reward_tracker.reset()
                    action_logger.start_round(rounds_done + 1, seed, label="manual reset")
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

        if paused:
            continue

        agent = env.agent_selection

        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            if all(env.terminations.values()) or all(env.truncations.values()):
                rounds_done += 1
                _print_round_summary(env, rounds_done, agent_type_map)
                action_logger.log_reward_breakdown(
                    round_idx=rounds_done,
                    seed=seed,
                    agent_types=agent_type_map,
                    tracker=reward_tracker,
                    env=env,
                )
                # Print auto-tune info only for normal (HeuristicPolicy) agents.
                if args.auto_tune_bomb:
                    for mgr in managers.values():
                        if isinstance(mgr._policy, HeuristicPolicy):
                            pol = mgr._policy
                            print(
                                f"  [auto-tune] threshold={pol.tuned_threshold:.3f}  "
                                f"hit_ema={pol._hit_ema:.3f}"
                            )
                            break
                if rounds_done < args.rounds:
                    seed = random.randint(0, 99999)
                    env.reset(seed=seed)
                    _reset_managers(managers, cached_template)
                    reward_tracker.reset()
                    action_logger.start_round(rounds_done + 1, seed, label="visual")
            continue

        obs = env.observe(agent)
        action = managers[agent].ae(obs)
        action_logger.log_action(
            round_idx=rounds_done + 1,
            seed=seed,
            agent=agent,
            agent_type=agent_type_map[agent],
            obs=obs,
            action=int(action),
            manager=managers[agent],
        )
        env.step(int(action))

    env.close()
    action_logger.close()
    print(f"\nDone. Played {rounds_done} round(s).")


if __name__ == "__main__":
    main()
