"""Train an attack-only DQN on top of HeuristicPolicy.

The learned policy has only two choices at the attack hook:

  0. defer to the scripted attack logic
  1. place a bomb now

Everything else stays scripted: danger dodging, pathfinding, wall breaking,
collection, base routing, and exploration are still handled by HeuristicPolicy.

Example:
    PYTHONPATH=ae/src:til-26-ae uv run python ae/test_env/train_attack_dqn.py \
      --episodes 200 --novice --out ae/models/attack_dqn.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - local training dependency only
    raise SystemExit(
        "Torch is required for attack training. Install ae/requirements-rl.txt."
    ) from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional logging dependency
    tqdm = None

# Make ae/src importable as flat top-level modules (matching Docker layout).
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from til_environment.bomberman_env import Bomberman  # noqa: E402
from til_environment.config import default_config  # noqa: E402

from ae_manager import DEFAULT_CACHE_PATH, DEFAULT_POLICY_KWARGS, AEManager  # noqa: E402
from constants import Action  # noqa: E402
from diagnostic_policies import PROFILES, make_diagnostic_policy  # noqa: E402
from map_memory import MapMemory  # noqa: E402
from observation import ParsedObs  # noqa: E402
from policy import HeuristicPolicy, Policy  # noqa: E402
from rl_attack import (  # noqa: E402
    RL_ATTACK_FEATURE_DIM,
    RL_ATTACK_SPATIAL_CHANNELS,
    RL_ATTACK_SPATIAL_SHAPE,
    extract_attack_features,
    extract_attack_spatial,
)
from rl_attack_model import AttackDQN  # noqa: E402


class StationaryPolicy(Policy):
    def choose(self, obs: ParsedObs, memory: MapMemory) -> int:  # noqa: ARG002
        if obs.action_mask[Action.STAY] == 1:
            return int(Action.STAY)
        valid = [i for i, ok in enumerate(obs.action_mask) if ok]
        return int(valid[0]) if valid else int(Action.STAY)


@dataclass
class Transition:
    state: np.ndarray
    spatial: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_spatial: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buf: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buf.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buf, batch_size)

    def __len__(self) -> int:
        return len(self.buf)


class TrainableAttackModule:
    """Epsilon-greedy bomb/defer module used inside HeuristicPolicy."""

    def __init__(
        self,
        policy_net: AttackDQN,
        replay: ReplayBuffer,
        *,
        device: torch.device,
        epsilon: float,
        shaped_rewards: bool,
    ) -> None:
        self.policy_net = policy_net
        self.replay = replay
        self.device = device
        self.epsilon = epsilon
        self.shaped_rewards = shaped_rewards
        self.pending_state: Optional[np.ndarray] = None
        self.pending_spatial: Optional[np.ndarray] = None
        self.pending_action: Optional[int] = None
        self.pending_reward: float = 0.0
        self.decisions = 0
        self.bombs = 0

    def choose_attack(self, obs: ParsedObs, memory: MapMemory) -> Optional[int]:
        state = extract_attack_features(obs, memory)
        spatial = extract_attack_spatial(obs, memory)
        self._close_pending(state, spatial, done=False)

        if random.random() < self.epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                scalar_t = torch.as_tensor(
                    state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                spatial_t = torch.as_tensor(
                    spatial, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                action = int(torch.argmax(self.policy_net(scalar_t, spatial_t), dim=1).item())

        self.pending_state = state
        self.pending_spatial = spatial
        self.pending_action = action
        self.pending_reward = self._shape_immediate(obs, memory, action)
        self.decisions += 1
        if action == 1:
            self.bombs += 1
            return int(Action.PLACE_BOMB)
        return None

    def observe_reward(self, reward_delta: float) -> None:
        if self.pending_state is not None:
            self.pending_reward += reward_delta

    def finish_episode(self) -> None:
        if self.pending_state is not None:
            self._close_pending(
                np.zeros(RL_ATTACK_FEATURE_DIM, dtype=np.float32),
                np.zeros(RL_ATTACK_SPATIAL_SHAPE, dtype=np.float32),
                True,
            )

    def _close_pending(
        self,
        next_state: np.ndarray,
        next_spatial: np.ndarray,
        done: bool,
    ) -> None:
        if (
            self.pending_state is None
            or self.pending_spatial is None
            or self.pending_action is None
        ):
            return
        self.replay.push(
            Transition(
                state=self.pending_state,
                spatial=self.pending_spatial,
                action=self.pending_action,
                reward=self.pending_reward,
                next_state=next_state,
                next_spatial=next_spatial,
                done=done,
            )
        )
        self.pending_state = None
        self.pending_spatial = None
        self.pending_action = None
        self.pending_reward = 0.0

    def _shape_immediate(
        self,
        obs: ParsedObs,
        memory: MapMemory,
        action: int,
    ) -> float:
        if not self.shaped_rewards:
            return 0.0
        features = extract_attack_features(obs, memory)
        can_bomb = features[5]
        direct_agent = features[7]
        direct_base = features[8]
        finish_base = features[9]
        expected = features[10]
        escapes = features[15]
        danger = features[16]

        if action == 0:
            # Mildly reward restraint when the bomb has no tactical signal.
            opportunity = direct_agent + direct_base + finish_base + expected
            return 0.01 if opportunity < 0.08 else -0.02

        value = 0.15 * direct_agent + 0.35 * direct_base + 0.7 * finish_base
        value += 0.08 * expected
        value += 0.03 * escapes
        value -= 0.05 if danger < 1.0 else 0.0
        value -= 0.08 if can_bomb <= 0.0 else 0.0
        return float(value)


class FrozenAttackModule:
    """Greedy attack module for fixed past-version opponents."""

    def __init__(
        self,
        state_dict: dict,
        *,
        hidden_dim: int,
        device: torch.device,
    ) -> None:
        self.device = device
        self.model = AttackDQN(
            RL_ATTACK_FEATURE_DIM,
            hidden_dim,
            RL_ATTACK_SPATIAL_CHANNELS,
        ).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def choose_attack(self, obs: ParsedObs, memory: MapMemory) -> Optional[int]:
        if obs.action_mask[Action.PLACE_BOMB] != 1 or obs.team_bombs <= 0:
            return None
        state = extract_attack_features(obs, memory)
        spatial = extract_attack_spatial(obs, memory)
        with torch.no_grad():
            scalar_t = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            spatial_t = torch.as_tensor(
                spatial, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action = int(torch.argmax(self.model(scalar_t, spatial_t), dim=1).item())
        return int(Action.PLACE_BOMB) if action == 1 else None


def train_step(
    policy_net: AttackDQN,
    target_net: AttackDQN,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    *,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> Optional[float]:
    if len(replay) < batch_size:
        return None
    batch = replay.sample(batch_size)
    states = torch.as_tensor(
        np.stack([t.state for t in batch]), dtype=torch.float32, device=device
    )
    spatials = torch.as_tensor(
        np.stack([t.spatial for t in batch]), dtype=torch.float32, device=device
    )
    actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=device)
    rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=device)
    next_states = torch.as_tensor(
        np.stack([t.next_state for t in batch]), dtype=torch.float32, device=device
    )
    next_spatials = torch.as_tensor(
        np.stack([t.next_spatial for t in batch]), dtype=torch.float32, device=device
    )
    done = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=device)

    q = policy_net(states, spatials).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target_net(next_states, next_spatials).max(dim=1).values
        target = rewards + gamma * next_q * (1.0 - done)

    loss = F.smooth_l1_loss(q, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
    optimizer.step()
    return float(loss.item())


def make_policy(
    agent_type: str,
    policy_kwargs: dict,
    attack_module,
) -> Policy:
    if agent_type == "stationary":
        return StationaryPolicy()
    kwargs = dict(policy_kwargs)
    kwargs["attack_module"] = attack_module
    kwargs["attack_module_mode"] = "replace"
    if agent_type == "normal":
        return HeuristicPolicy(**kwargs)
    if agent_type in PROFILES:
        return make_diagnostic_policy(agent_type, **kwargs)
    raise ValueError(f"unsupported training agent type: {agent_type}")


def load_cache(path: Optional[Path]) -> Optional[MapMemory]:
    if path is None or not path.exists():
        return None
    return MapMemory.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--novice", action="store_true", default=True)
    parser.add_argument("--advanced", dest="novice", action="store_false")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--agent-type", choices=("normal", *tuple(PROFILES)), default="normal")
    parser.add_argument(
        "--opponent-type",
        choices=("same", "stationary", "heuristic", "frozen-dqn"),
        default="same",
        help=(
            "For --focus-only: opponents can use the same heuristic, stay still, "
            "use normal HeuristicPolicy, or use a frozen copy of --init-model/current model."
        ),
    )
    parser.add_argument(
        "--train-all-agents",
        action="store_true",
        default=True,
        help="Attach the shared DQN to every agent. Disable with --focus-only.",
    )
    parser.add_argument("--focus-only", dest="train_all_agents", action="store_false")
    parser.add_argument("--cache", dest="cache_path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--no-cache", dest="cache_path", action="store_const", const=None)
    parser.add_argument("--out", type=Path, default=Path("models/attack_dqn.pt"))
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--target-sync", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=0.35)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=160)
    parser.add_argument("--no-shaped-rewards", dest="shaped_rewards", action="store_false")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-tqdm", dest="use_tqdm", action="store_false")
    parser.set_defaults(use_tqdm=True)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    policy_net = AttackDQN(
        RL_ATTACK_FEATURE_DIM,
        args.hidden_dim,
        RL_ATTACK_SPATIAL_CHANNELS,
    ).to(device)
    init_payload = None
    if args.init_model is not None:
        init_payload = torch.load(
            str(args.init_model), map_location=device, weights_only=False
        )
        policy_net.load_state_dict(init_payload["model_state"])
    target_net = AttackDQN(
        RL_ATTACK_FEATURE_DIM,
        args.hidden_dim,
        RL_ATTACK_SPATIAL_CHANNELS,
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay_size)

    cfg = default_config()
    cfg.env.novice = args.novice
    env = Bomberman(cfg)
    seed = args.seed if args.seed is not None else random.randint(0, 99999)
    env.reset(seed=seed)
    cache_tmpl = load_cache(args.cache_path)

    policy_kwargs = dict(DEFAULT_POLICY_KWARGS)
    policy_kwargs["auto_tune_bomb"] = False

    attack_modules: dict[str, TrainableAttackModule] = {}
    managers: dict[str, AEManager] = {}
    focus_agent = env.possible_agents[0]
    frozen_state = {
        k: v.detach().clone()
        for k, v in policy_net.state_dict().items()
    }
    for agent in env.possible_agents:
        train_this_agent = args.train_all_agents or agent == focus_agent
        module = (
            TrainableAttackModule(
                policy_net,
                replay,
                device=device,
                epsilon=args.epsilon_start,
                shaped_rewards=args.shaped_rewards,
            )
            if train_this_agent
            else None
        )
        mem = MapMemory()
        if cache_tmpl is not None:
            mem.merge_static_from(cache_tmpl)
        if train_this_agent:
            policy = make_policy(args.agent_type, policy_kwargs, module)
        else:
            opponent_type = args.agent_type
            opponent_module = None
            if args.opponent_type == "stationary":
                opponent_type = "stationary"
            elif args.opponent_type == "heuristic":
                opponent_type = "normal"
            elif args.opponent_type == "frozen-dqn":
                opponent_module = FrozenAttackModule(
                    frozen_state,
                    hidden_dim=args.hidden_dim,
                    device=device,
                )
            policy = make_policy(opponent_type, policy_kwargs, opponent_module)
        managers[agent] = AEManager(policy=policy, memory=mem)
        if module is not None:
            attack_modules[agent] = module

    episode_scores: list[float] = []
    global_updates = 0
    train_step_counter = 0
    losses: deque[float] = deque(maxlen=100)

    use_tqdm = bool(args.use_tqdm and tqdm is not None)
    episode_iter = (
        tqdm(range(args.episodes), desc="attack-dqn", unit="ep")
        if use_tqdm
        else range(args.episodes)
    )

    for episode_idx in episode_iter:
        frac = min(1.0, episode_idx / max(1, args.epsilon_decay_episodes))
        epsilon = args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)
        for module in attack_modules.values():
            module.epsilon = epsilon

        previous_episode_rewards = dict(getattr(env.dynamics.rewards, "_episode", {}))
        while True:
            agent = env.agent_selection
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                if all(env.terminations.values()) or all(env.truncations.values()):
                    break
                continue

            obs = env.observe(agent)
            action = managers[agent].ae(obs)
            env.step(int(action))

            episode_rewards = getattr(env.dynamics.rewards, "_episode", {})
            for train_agent, module in attack_modules.items():
                before = float(previous_episode_rewards.get(train_agent, 0.0))
                after = float(episode_rewards.get(train_agent, 0.0))
                module.observe_reward(after - before)
            previous_episode_rewards = dict(episode_rewards)

            train_step_counter += 1
            if (
                len(replay) >= args.warmup
                and train_step_counter % max(1, args.train_every) == 0
            ):
                loss = train_step(
                    policy_net,
                    target_net,
                    optimizer,
                    replay,
                    batch_size=args.batch_size,
                    gamma=args.gamma,
                    device=device,
                )
                if loss is not None:
                    losses.append(loss)
                    global_updates += 1
                    if global_updates % args.target_sync == 0:
                        target_net.load_state_dict(policy_net.state_dict())

        for module in attack_modules.values():
            module.finish_episode()

        episode_rewards = getattr(env.dynamics.rewards, "_episode", {})
        trained_scores = [
            float(episode_rewards.get(a, 0.0))
            for a in attack_modules
        ]
        score = mean(trained_scores) if trained_scores else 0.0
        episode_scores.append(score)
        decisions = sum(m.decisions for m in attack_modules.values())
        bombs = sum(m.bombs for m in attack_modules.values())
        avg_loss = mean(losses) if losses else 0.0
        metrics = {
            "score": f"{score:.1f}",
            "mean20": f"{mean(episode_scores[-20:]):.1f}",
            "eps": f"{epsilon:.3f}",
            "replay": len(replay),
            "updates": global_updates,
            "loss100": f"{avg_loss:.4f}",
            "decisions": decisions,
            "bombs": bombs,
        }
        if use_tqdm:
            episode_iter.set_postfix(metrics)
        else:
            print(
                f"episode={episode_idx + 1:04d}"
                f" score={score:8.2f}"
                f" mean20={mean(episode_scores[-20:]):8.2f}"
                f" eps={epsilon:.3f}"
                f" replay={len(replay):5d}"
                f" updates={global_updates:5d}"
                f" loss100={avg_loss:.4f}"
                f" decisions={decisions:5d}"
                f" bombs={bombs:5d}",
                flush=True,
            )

        if episode_idx < args.episodes - 1:
            seed = random.randint(0, 99999)
            env.reset(seed=seed)
            for mgr in managers.values():
                mgr._memory.reset_round()
                if cache_tmpl is not None:
                    mgr._memory.merge_static_from(cache_tmpl)

    env.close()
    if use_tqdm:
        episode_iter.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": policy_net.state_dict(),
            "feature_dim": RL_ATTACK_FEATURE_DIM,
            "hidden_dim": args.hidden_dim,
            "spatial_channels": RL_ATTACK_SPATIAL_CHANNELS,
            "spatial_shape": RL_ATTACK_SPATIAL_SHAPE,
            "episodes": args.episodes,
            "agent_type": args.agent_type,
        },
        args.out,
    )
    print(f"saved attack DQN to {args.out}")


if __name__ == "__main__":
    main()
