"""Single-round benchmark: a chosen policy vs 5 idle (STAY) agents.

Run from repo root:
    python ae/test_env/bench_idle_opponents.py --policy edited
    python ae/test_env/bench_idle_opponents.py --policy berserker

Purpose: measure the theoretical maximum score each policy can achieve when
opponents offer no resistance -- all enemies stand still.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from til_environment.bomberman_env import Bomberman
from til_environment.config import default_config

from ae_manager import DEFAULT_CACHE_PATH, AEManager
from constants import Action
from map_memory import MapMemory
from observation import ParsedObs
from policies.edited_policy import EditedHeuristicPolicy
from policies.berserker_policy import BerserkerPolicy
from policies.policy import Policy


class StayPolicy(Policy):
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:
        return int(Action.STAY)


_ACTION_MODE = {
    int(Action.PLACE_BOMB): "attack",
    int(Action.FORWARD): "move",
    int(Action.BACKWARD): "move",
    int(Action.LEFT): "turn",
    int(Action.RIGHT): "turn",
    int(Action.STAY): "stay",
}


def _get_mode(policy: Policy, last_action: int) -> str:
    """Return a mode label for this step.

    Uses _debug_mode when available (EditedHeuristicPolicy); falls back to
    classifying the actual action taken (BerserkerPolicy and others).
    """
    mode = getattr(policy, "_debug_mode", None)
    if mode is not None:
        return mode
    return _ACTION_MODE.get(last_action, "stay")


def run(policy_name: str) -> None:
    cfg = default_config()
    cfg.env.novice = True

    env = Bomberman(cfg)
    env.reset(seed=42)

    agents = env.possible_agents
    focus = agents[0]

    # Build focus policy.
    if policy_name == "edited":
        focus_policy: Policy = EditedHeuristicPolicy()
        label = "EditedHeuristicPolicy"
    elif policy_name == "berserker":
        focus_policy = BerserkerPolicy()
        label = "BerserkerPolicy"
    else:
        raise ValueError(f"unknown policy: {policy_name!r}")

    focus_mem = MapMemory()
    cache = DEFAULT_CACHE_PATH
    if cache.exists():
        focus_mem.merge_static_from(MapMemory.load(cache))
    focus_mem.reset_round()
    focus_mgr = AEManager(policy=focus_policy, memory=focus_mem)

    idle_mgrs: dict[str, AEManager] = {}
    for a in agents[1:]:
        m = MapMemory()
        m.reset_round()
        idle_mgrs[a] = AEManager(policy=StayPolicy(), memory=m)

    all_mgrs = {focus: focus_mgr, **idle_mgrs}

    mode_steps: dict[str, int] = defaultdict(int)
    mode_rewards: dict[str, float] = defaultdict(float)
    prev_episode_reward: float = 0.0
    last_mode: str = "stay"
    last_action: int = int(Action.STAY)

    while True:
        agent = env.agent_selection
        done = env.terminations[agent] or env.truncations[agent]
        if done:
            env.step(None)
            if all(env.terminations.values()) or all(env.truncations.values()):
                break
            continue

        obs_raw = env.observe(agent)
        action = all_mgrs[agent].ae(obs_raw)

        if agent == focus:
            # Attribute reward from the previous step to its mode.
            ep_now = env.dynamics.rewards._episode.get(focus, 0.0)
            mode_rewards[last_mode] += ep_now - prev_episode_reward
            prev_episode_reward = ep_now
            # Record mode for THIS step (will be attributed on next iteration).
            last_action = action
            last_mode = _get_mode(focus_policy, last_action)
            mode_steps[last_mode] += 1

        env.step(int(action))

    # Capture final step reward.
    ep_final = env.dynamics.rewards._episode.get(focus, 0.0)
    mode_rewards[last_mode] += ep_final - prev_episode_reward

    episode = env.dynamics.rewards._episode
    focus_score = episode.get(focus, 0.0)
    idle_scores = [episode.get(a, 0.0) for a in agents[1:]]
    rw = env.dynamics.rewards.config

    SEP = "=" * 64
    print()
    print(SEP)
    print(f"  IDLE-OPPONENT BENCHMARK  ({label} vs 5xSTAY)")
    print(SEP)

    print(f"\n  Focus agent ({focus})  total score : {focus_score:.1f}")
    print(f"  Idle agents mean score          : {sum(idle_scores)/len(idle_scores):.1f}")
    print(f"  Idle agents range               : {min(idle_scores):.1f} - {max(idle_scores):.1f}")

    print("\n  -- Reward config --")
    print(f"    collect_mission    {rw.collect_mission:>6.1f}")
    print(f"    collect_resource   {rw.collect_resource:>6.1f}")
    print(f"    collect_recon      {rw.collect_recon:>6.1f}")
    print(f"    attack_damage      {rw.attack_damage:>6.1f}  (per bomb hit on enemy agent)")
    print(f"    attack_kill        {rw.attack_kill:>6.1f}  (killing blow bonus)")
    print(f"    destroy_enemy_base {rw.destroy_enemy_base:>6.1f}")
    print(f"    own_base_destroyed {rw.own_base_destroyed:>6.1f}")

    total_steps = sum(mode_steps.values())
    print(f"\n  -- Mode breakdown ({total_steps} focus-agent steps) --")
    if policy_name == "berserker":
        ordered = ["attack", "move", "turn", "stay"]
    else:
        ordered = ["attack", "collect", "explore", "dodge", "frozen", "stay"]
    others = sorted(set(mode_steps) - set(ordered))
    print(f"  {'Mode':<12}  {'Steps':>6}  {'%':>6}  {'Score Delta':>12}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*12}")
    for mode in ordered + others:
        steps = mode_steps.get(mode, 0)
        score = mode_rewards.get(mode, 0.0)
        pct = 100.0 * steps / total_steps if total_steps else 0.0
        print(f"  {mode:<12}  {steps:>6}  {pct:>5.1f}%  {score:>12.1f}")

    print(f"\n  -- Per-agent final scores --")
    for a in agents:
        tag = f"  <- FOCUS ({label})" if a == focus else "  (STAY)"
        print(f"    {a}  {episode.get(a, 0.0):>8.1f}{tag}")

    print()
    print(SEP)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["edited", "berserker"],
        default="edited",
        help="Which focus policy to benchmark (default: edited)",
    )
    args = parser.parse_args()
    run(args.policy)
