"""Focused diagnostics for the AE heuristic policy.

Runs a focus agent against configurable opponent policies and prints:
  - focus reward per seed
  - reward-event breakdown for the focus agent
  - policy mode/action counts

This is intentionally smaller than auto_play.py: it is for explaining why a
policy scores where it does, not for rendering or broad parameter sweeps.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from til_environment.bomberman_env import Bomberman  # noqa: E402
from til_environment.config import default_config  # noqa: E402

from ae_manager import DEFAULT_CACHE_PATH, DEFAULT_POLICY_KWARGS, AEManager  # noqa: E402
from policies.berserker_policy import BerserkerPolicy  # noqa: E402
from constants import Action  # noqa: E402
from diagnostic_policies import PROFILES, make_diagnostic_policy  # noqa: E402
from edited_policy_conservative import EditedHeuristicPolicy as HeuristicPolicy  # noqa: E402
from map_memory import MapMemory  # noqa: E402
from observation import ParsedObs  # noqa: E402
from policies.policy import Policy  # noqa: E402


class RandomPolicy(Policy):
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:  # noqa: ARG002
        valid = [i for i, ok in enumerate(obs.action_mask) if ok]
        return int(random.choice(valid)) if valid else int(Action.STAY)


def make_policy(kind: str, policy_kwargs: dict) -> Policy:
    if kind == "normal":
        return HeuristicPolicy(**policy_kwargs)
    if kind in PROFILES:
        return make_diagnostic_policy(kind, **policy_kwargs)
    if kind == "berserker":
        return BerserkerPolicy()
    if kind == "random":
        return RandomPolicy()
    raise ValueError(kind)


def load_cache(path: Path | None) -> MapMemory | None:
    if path is None or not path.exists():
        return None
    return MapMemory.load(path)


def run_seed(
    seed: int,
    focus_type: str,
    opponent_type: str,
    novice: bool,
    cache_path: Path | None,
    policy_kwargs: dict,
) -> dict:
    cfg = default_config()
    cfg.env.render_mode = None
    cfg.env.novice = novice
    env = Bomberman(cfg)
    env.reset(seed=seed)

    cached = load_cache(cache_path)
    managers: dict[str, AEManager] = {}
    for i, agent in enumerate(env.possible_agents):
        mem = MapMemory()
        if cached is not None:
            mem.merge_static_from(cached)
        kind = focus_type if i == 0 else opponent_type
        managers[agent] = AEManager(policy=make_policy(kind, policy_kwargs), memory=mem)

    focus_agent = env.possible_agents[0]
    focus_policy = managers[focus_agent]._policy

    by_agent_event: dict[str, Counter[str]] = defaultdict(Counter)
    original_award = env.dynamics.rewards.award

    def award_spy(recipient_id: str, event: str, multiplier: float = 1.0) -> float:
        value = original_award(recipient_id, event, multiplier)
        if value != 0.0:
            by_agent_event[recipient_id][event] += value
        return value

    env.dynamics.rewards.award = award_spy

    mode_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    illegal_actions = 0

    while not (all(env.terminations.values()) or all(env.truncations.values())):
        agent = env.agent_selection
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue

        obs = env.observe(agent)
        action = managers[agent].ae(obs)
        if agent == focus_agent:
            mode_counts[getattr(focus_policy, "_debug_mode", "unknown")] += 1
            action_counts[Action(int(action)).name] += 1
            mask = obs.get("action_mask")
            if mask is not None and int(action) < len(mask) and not mask[int(action)]:
                illegal_actions += 1
        env.step(int(action))

    episode = getattr(env.dynamics.rewards, "_episode", {})
    focus_score = float(episode.get(focus_agent, 0.0))
    opp_scores = [
        float(episode.get(agent, 0.0))
        for agent in env.possible_agents
        if agent != focus_agent
    ]
    env.close()

    return {
        "seed": seed,
        "focus_score": focus_score,
        "opp_mean": mean(opp_scores) if opp_scores else 0.0,
        "events": by_agent_event[focus_agent],
        "modes": mode_counts,
        "actions": action_counts,
        "illegal_actions": illegal_actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    policy_choices = (
        "normal",
        *tuple(PROFILES),
        "berserker",
        "random",
    )
    parser.add_argument("--focus", choices=policy_choices, default="normal")
    parser.add_argument("--opponent", choices=policy_choices, default="normal")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--novice", action="store_true", default=True)
    parser.add_argument("--advanced", dest="novice", action="store_false")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--no-cache", dest="cache", action="store_const", const=None)
    parser.add_argument("--no-adaptive-base-weight", dest="adaptive_base_weight", action="store_false", default=DEFAULT_POLICY_KWARGS["adaptive_base_weight"])
    parser.add_argument("--base-weight-ramp-rate", type=float, default=DEFAULT_POLICY_KWARGS["base_weight_ramp_rate"])
    parser.add_argument("--base-route-weight", type=float, default=DEFAULT_POLICY_KWARGS["base_route_weight"])
    parser.add_argument("--no-smart-defend", dest="smart_defend", action="store_false", default=DEFAULT_POLICY_KWARGS["smart_defend"])
    parser.add_argument("--no-proactive-base-routing", dest="proactive_base_routing", action="store_false", default=DEFAULT_POLICY_KWARGS["proactive_base_routing"])
    args = parser.parse_args()

    policy_kwargs = dict(DEFAULT_POLICY_KWARGS)
    policy_kwargs.update(
        adaptive_base_weight=args.adaptive_base_weight,
        base_weight_ramp_rate=args.base_weight_ramp_rate,
        base_route_weight=args.base_route_weight,
        smart_defend=args.smart_defend,
        proactive_base_routing=args.proactive_base_routing,
    )

    rows = [
        run_seed(seed, args.focus, args.opponent, args.novice, args.cache, policy_kwargs)
        for seed in args.seeds
    ]

    print(f"focus={args.focus} opponent={args.opponent} novice={args.novice}")
    print(
        "policy="
        f"adaptive_base_weight={policy_kwargs['adaptive_base_weight']} "
        f"base_weight_ramp_rate={policy_kwargs['base_weight_ramp_rate']} "
        f"base_route_weight={policy_kwargs['base_route_weight']} "
        f"smart_defend={policy_kwargs['smart_defend']} "
        f"proactive_base_routing={policy_kwargs['proactive_base_routing']}"
    )
    print("seed  focus_score  opp_mean  illegal")
    for row in rows:
        print(
            f"{row['seed']:4d}  {row['focus_score']:11.1f}"
            f"  {row['opp_mean']:8.1f}  {row['illegal_actions']:7d}"
        )
    print(f"mean_focus={mean(r['focus_score'] for r in rows):.1f}")

    events: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    for row in rows:
        events.update(row["events"])
        modes.update(row["modes"])
        actions.update(row["actions"])

    def dump(title: str, counts: Counter[str]) -> None:
        print(f"\n{title}")
        total = sum(counts.values()) or 1
        for key, value in counts.most_common():
            print(f"  {key:22s} {value:8.1f}  {100.0 * value / total:5.1f}%")

    dump("focus reward events", events)
    dump("focus modes", modes)
    dump("focus actions", actions)


if __name__ == "__main__":
    main()
