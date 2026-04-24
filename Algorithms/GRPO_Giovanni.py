import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils as bf


class EpisodicReplayBuffer:
    """Replay buffer that stores whole episodes for GRPO-style updates."""

    def __init__(self, capacity=256):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add_episode(self, episode):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = episode
        self.position = (self.position + 1) % self.capacity

    def sample(self, n):
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n)

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer:
    """Episodic buffer used for on-policy rollout collection."""

    def __init__(self):
        self.buffer = []
        self.current_episode = self._init_empty_episode()

    def _init_empty_episode(self):
        return {"obs": [], "action": [], "log_probs": [], "reward": []}

    def add(self, obs, action, log_prob, reward):
        self.current_episode["obs"].append(obs)
        self.current_episode["action"].append(action)
        self.current_episode["log_probs"].append(log_prob)
        self.current_episode["reward"].append(reward)

    def finish_episode(self):
        finished = None
        if len(self.current_episode["reward"]) > 0:
            finished = self.current_episode
            self.buffer.append(self.current_episode)
        self.current_episode = self._init_empty_episode()
        return finished

    def clear_buffer(self):
        self.buffer.clear()

    def to_tensors(self, device: torch.device):
        tensor_episodes = []
        for episode in self.buffer:
            tensor_episodes.append(
                {
                    "obs": torch.stack(episode["obs"]).to(device),
                    "action": torch.stack(episode["action"]).to(device),
                    "log_probs": torch.stack(episode["log_probs"]).to(device),
                    "reward": torch.tensor(episode["reward"], dtype=torch.float32).to(device),
                }
            )
        return tensor_episodes


class Actor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super().__init__()
        self.extractor = bf.FeatureExtractor(observation_space)

        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        self.action_dim = action_space.n if self.is_discrete else action_space.shape[0]

        self.mean_head = nn.Linear(self.extractor.feature_dim, self.action_dim)

        if not self.is_discrete:
            self.std = nn.Parameter(torch.zeros(self.action_dim))

    def sample_action(self, obs: torch.Tensor):
        x = self.extractor(obs)
        if self.is_discrete:
            logits = self.mean_head(x)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        else:
            mean = torch.tanh(self.mean_head(x))
            std = torch.exp(self.std)
            distribution = torch.distributions.Normal(mean, std)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)

        return action, log_prob

    def get_log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.extractor(obs)
        if self.is_discrete:
            logits = self.mean_head(x)
            distribution = torch.distributions.Categorical(logits=logits)
            act = action.squeeze(-1) if action.dim() > 1 else action
            return distribution.log_prob(act)

        mean = torch.tanh(self.mean_head(x))
        std = torch.exp(self.std)
        distribution = torch.distributions.Normal(mean, std)
        return distribution.log_prob(action).sum(dim=-1)

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.extractor(obs)
        if self.is_discrete:
            logits = self.mean_head(x)
            return torch.argmax(logits, dim=-1)
        return torch.tanh(self.mean_head(x))


class GRPO_Giovanni:
    """
    GRPO variant with episodic replay.

    Replay can be disabled automatically:
    - globally after a fixed number of policy updates
    - locally when recent return variance gets small
    """

    def __init__(
        self,
        env,
        hidden_dim=256,
        lr=3e-4,
        G=32,
        epsilon=0.2,
        beta=0.04,
        replay_capacity=2000,
        replay_ratio=1.0,
        replay_off_after_updates=None,
        local_std_window=30,
        local_std_threshold=None,
        max_replay_age_updates=None,
        log_replay_events=True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.G = G
        self.epsilon = epsilon
        self.beta = beta

        self.replay_ratio = max(0.0, float(replay_ratio))
        self.replay_off_after_updates = replay_off_after_updates
        self.local_std_threshold = local_std_threshold
        self.max_replay_age_updates = max_replay_age_updates
        self.log_replay_events = bool(log_replay_events)

        self.replay_disabled = False
        self.replay_disable_reason = None
        self.update_count = 0
        self.recent_returns = deque(maxlen=local_std_window)

        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

        self.actor = Actor(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.actor_old = Actor(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.buffer = ReplayBuffer()
        self.replay = EpisodicReplayBuffer(capacity=replay_capacity)

        self._cached_obs = None
        self._cached_action = None
        self._cached_logprob = None

    def select_action(self, state, evaluate=False):
        obs_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                action_t = self.actor.get_deterministic_action(obs_t)
            else:
                action_t, log_prob_t = self.actor_old.sample_action(obs_t)
                self._cached_obs = obs_t.squeeze(0).cpu()
                self._cached_action = action_t.squeeze(0).cpu()
                self._cached_logprob = log_prob_t.squeeze(0).cpu()

        action_np = action_t.squeeze(0).cpu().numpy()
        if self.is_discrete:
            return int(action_np)
        return action_np

    def step(self, state, action, reward, next_state, done):
        self.buffer.add(self._cached_obs, self._cached_action, self._cached_logprob, reward)

        if done:
            finished_episode = self.buffer.finish_episode()
            if finished_episode is not None:
                ep_return = float(np.sum(finished_episode["reward"]))
                self.recent_returns.append(ep_return)

                if not self.replay_disabled:
                    replay_episode = {
                        "obs": [x.clone() for x in finished_episode["obs"]],
                        "action": [x.clone() for x in finished_episode["action"]],
                        "log_probs": [x.clone() for x in finished_episode["log_probs"]],
                        "reward": list(finished_episode["reward"]),
                        "collected_update": self.update_count,
                    }
                    self.replay.add_episode(replay_episode)

            if len(self.buffer.buffer) >= self.G:
                self.update()

    def _check_and_disable_replay(self):
        if self.replay_disabled:
            return

        if self.replay_off_after_updates is not None and self.update_count >= self.replay_off_after_updates:
            self._disable_replay(
                reason="global_update_budget_reached",
                details=f"update_count={self.update_count}, threshold={self.replay_off_after_updates}",
            )
            return

        if self.local_std_threshold is None:
            return

        if len(self.recent_returns) == self.recent_returns.maxlen:
            recent_std = float(np.std(self.recent_returns))
            if recent_std <= float(self.local_std_threshold):
                self._disable_replay(
                    reason="local_return_std_below_threshold",
                    details=(
                        f"recent_std={recent_std:.6f}, "
                        f"threshold={float(self.local_std_threshold):.6f}, "
                        f"window={self.recent_returns.maxlen}"
                    ),
                )

    def _disable_replay(self, reason, details=""):
        if self.replay_disabled:
            return

        self.replay_disabled = True
        self.replay_disable_reason = reason

        if self.log_replay_events:
            suffix = f" | {details}" if details else ""
            print(f"[GRPO_Giovanni] Replay disabled: {reason}{suffix}")

    def _sample_replay_episodes(self):
        if self.replay_disabled or len(self.replay) == 0 or self.replay_ratio <= 0.0:
            return []

        n_replay = int(round(self.G * self.replay_ratio))
        if n_replay <= 0:
            return []

        candidates = self.replay.buffer
        if self.max_replay_age_updates is not None:
            min_allowed = self.update_count - int(self.max_replay_age_updates)
            candidates = [ep for ep in candidates if ep is not None and ep["collected_update"] >= min_allowed]

        if not candidates:
            return []

        n_replay = min(n_replay, len(candidates))
        return random.sample(candidates, n_replay)

    def _episode_to_tensors(self, episode):
        return {
            "obs": torch.stack(episode["obs"]).to(self.device),
            "action": torch.stack(episode["action"]).to(self.device),
            "log_probs": torch.stack(episode["log_probs"]).to(self.device),
            "reward": torch.tensor(episode["reward"], dtype=torch.float32).to(self.device),
        }

    def update(self):
        self._check_and_disable_replay()

        on_policy = self.buffer.to_tensors(self.device)
        replay_eps = [self._episode_to_tensors(ep) for ep in self._sample_replay_episodes()]
        trajectories = on_policy + replay_eps

        returns = torch.stack([traj["reward"].sum() for traj in trajectories])
        mean_return = returns.mean()
        std_return = returns.std()
        advantages = (returns - mean_return) / (std_return + 1e-8)

        flat_obs, flat_actions, flat_old_log_probs, flat_advantages = [], [], [], []

        for i, traj in enumerate(trajectories):
            num_steps = len(traj["reward"])
            flat_obs.append(traj["obs"])
            flat_actions.append(traj["action"])
            flat_old_log_probs.append(traj["log_probs"])
            flat_advantages.append(torch.full((num_steps,), advantages[i], device=self.device))

        flat_obs = torch.cat(flat_obs, dim=0)
        flat_actions = torch.cat(flat_actions, dim=0)
        flat_old_log_probs = torch.cat(flat_old_log_probs, dim=0)
        flat_advantages = torch.cat(flat_advantages, dim=0)

        new_log_probs = self.actor.get_log_prob(flat_obs, flat_actions)
        ratio = torch.exp(new_log_probs - flat_old_log_probs)

        surr1 = ratio * flat_advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * flat_advantages
        clipped_surrogate = torch.min(surr1, surr2)

        ratio_inv = 1.0 / ratio
        kl_penalty = ratio_inv - torch.log(ratio_inv) - 1.0
        loss = -(clipped_surrogate - self.beta * kl_penalty).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.buffer.clear_buffer()
        self.update_count += 1
