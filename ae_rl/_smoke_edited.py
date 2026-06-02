"""Headless crash-smoke for AzbaseV1EditedPolicy (all features on).

Runs real FFA novice games with the edited policy in every slot, calling
``choose()`` directly (NOT through ``AEManager.ae``, which swallows exceptions),
so any runtime bug in the new feature code surfaces as a traceback instead of
silently degrading to STAY. Prints the action + decision-mode distribution as a
sanity check that the new branches actually fire.

    python ae_rl/_smoke_edited.py [rounds]
"""

from __future__ import annotations

import sys
from collections import Counter

import common  # noqa: F401  path bootstrap (adds ae/src to sys.path)
from ae_manager import DEFAULT_CACHE_PATH
from constants import Action
from map_memory import MapMemory
from observation import parse_observation
from policies.azbasev1_edited_policy import AzbaseV1EditedPolicy
from rollout import make_env


def _cache_template():
    try:
        if DEFAULT_CACHE_PATH.exists():
            return MapMemory.load(DEFAULT_CACHE_PATH)
    except Exception:
        return None
    return None


def main(rounds: int = 2) -> None:
    env = make_env(True)  # novice
    agents = list(env.possible_agents)
    cache = _cache_template()

    def fresh_actor():
        mem = MapMemory()
        if cache is not None:
            mem.merge_static_from(cache)
        pol = AzbaseV1EditedPolicy(
            hp_aware_kills=True, base_siege=True, endgame_dump=True
        )
        return pol, mem

    actors = {a: fresh_actor() for a in agents}
    actions: Counter = Counter()
    modes: Counter = Counter()
    steps = 0

    for rnd in range(rounds):
        env.reset(seed=100 + rnd)
        for pol, mem in actors.values():
            mem.reset_round()
            if cache is not None:
                mem.merge_static_from(cache)
        while True:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                if all(env.terminations.values()) or all(env.truncations.values()):
                    break
                continue
            obs = env.observe(agent)
            pol, mem = actors[agent]
            parsed = parse_observation(obs)
            mem.update(parsed)
            action = int(pol.choose(parsed, mem))  # exceptions propagate here
            actions[Action(action).name] += 1
            modes[pol._debug_mode] += 1
            steps += 1
            env.step(action)

    print(f"OK: {rounds} rounds, {steps} agent-steps, no exceptions")
    print("actions:", dict(actions))
    print("modes:  ", dict(modes))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
