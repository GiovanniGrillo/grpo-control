import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from copy import deepcopy
import Basic_Functions as bf

# =========================================================================================
# GRPO Agent Implementation
# =========================================================================================

class ReplayBuffer:
    """Episodic Replay Buffer for GRPO"""
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
        if len(self.current_episode["reward"]) > 0:
            self.buffer.append(self.current_episode)
        self.current_episode = self._init_empty_episode()
    
    def clear_buffer(self):
        self.buffer.clear()

    def to_tensors(self, device: torch.device):
        for episode in self.buffer:
            episode["obs"] = torch.stack(episode["obs"]).to(device)
            episode["action"] = torch.stack(episode["action"]).to(device)
            episode["log_probs"] = torch.stack(episode["log_probs"]).to(device)
            episode["reward"] = torch.tensor(episode["reward"], dtype=torch.float32).to(device)
        return self.buffer  

class Actor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super().__init__()
        # 1. Inherit the common FeatureExtractor
        self.extractor = bf.FeatureExtractor(observation_space)
        
        # 2. Support both Discrete and Continuous Action Spaces
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        self.action_dim = action_space.n if self.is_discrete else action_space.shape[0]

        # 3. Networks
        self.mean_head = nn.Linear(self.extractor.feature_dim, self.action_dim)
        
        if not self.is_discrete:
            self.std = nn.Parameter(torch.zeros(self.action_dim))
            # Optional: Bind scaling if needed, using tanh limits outputs to [-1, 1] natively
            self.act_high = torch.tensor(action_space.high, dtype=torch.float32)

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
            # Ensure action shapes match for categorical log_prob
            act = action.squeeze(-1) if action.dim() > 1 else action
            return distribution.log_prob(act)
        else:
            mean = torch.tanh(self.mean_head(x))
            std = torch.exp(self.std)
            distribution = torch.distributions.Normal(mean, std)
            return distribution.log_prob(action).sum(dim=-1)
    
    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.extractor(obs)
        if self.is_discrete:
            logits = self.mean_head(x)
            return torch.argmax(logits, dim=-1)
        else:
            return torch.tanh(self.mean_head(x))


class GRPO:
    """Group Relative Policy Optimization Agent"""
    def __init__(self, env, hidden_dim=256, lr=3e-4, G=16, epsilon=0.2, beta=0.04):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G = G # Group size (Number of episodes before update)
        self.epsilon = epsilon
        self.beta = beta
        
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        
        # Networks
        self.actor = Actor(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.actor_old = Actor(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

        # Cache variables to temporarily hold state for the Simulation.py standard step() format
        self._cached_obs = None
        self._cached_action = None
        self._cached_logprob = None

    def select_action(self, state, evaluate=False):
        obs_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                action_t = self.actor.get_deterministic_action(obs_t)
                log_prob_t = None
            else:
                action_t, log_prob_t = self.actor_old.sample_action(obs_t)
                
                # Cache these specifically so step() can retrieve them to put in the buffer
                self._cached_obs = obs_t.squeeze(0).cpu()
                self._cached_action = action_t.squeeze(0).cpu()
                self._cached_logprob = log_prob_t.squeeze(0).cpu()

        action_np = action_t.squeeze(0).cpu().numpy()
        
        # Return proper types based on the environment's requirements
        if self.is_discrete:
            return int(action_np)
        return action_np

    def step(self, state, action, reward, next_state, done):
        """Standardized step function for Simulation.py"""
        
        # Add the cached data from the last select_action() call
        self.buffer.add(self._cached_obs, self._cached_action, self._cached_logprob, reward)
        
        # GRPO relies strictly on episodes for computing Advantage. 
        if done:
            self.buffer.finish_episode()
            
            # Check if we have collected enough trajectories (Group Size G)
            if len(self.buffer.buffer) >= self.G:
                self.update()

    def update(self):
        trajectories = self.buffer.to_tensors(self.device)
        
        # 1. Compute Episode Returns
        returns = torch.stack([traj["reward"].sum() for traj in trajectories])
        
        # 2. Calculate Group Advantage
        mean_return = returns.mean()
        std_return = returns.std()
        advantages = (returns - mean_return) / (std_return + 1e-8)
        
        # 3. Flatten data for PyTorch operations
        flat_obs, flat_actions, flat_old_log_probs, flat_advantages = [], [], [], []
        
        for i, traj in enumerate(trajectories):
            num_steps = len(traj["reward"])
            flat_obs.append(traj["obs"])
            flat_actions.append(traj["action"])
            flat_old_log_probs.append(traj["log_probs"])
            
            # Broadcast the episodic advantage to all steps in that episode
            expanded_adv = torch.full((num_steps,), advantages[i], device=self.device)
            flat_advantages.append(expanded_adv)
            
        flat_obs = torch.cat(flat_obs, dim=0)
        flat_actions = torch.cat(flat_actions, dim=0)
        flat_old_log_probs = torch.cat(flat_old_log_probs, dim=0)
        flat_advantages = torch.cat(flat_advantages, dim=0)

        # 4. Math Phase
        new_log_probs = self.actor.get_log_prob(flat_obs, flat_actions)
        ratio = torch.exp(new_log_probs - flat_old_log_probs)
        
        surr1 = ratio * flat_advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * flat_advantages
        clipped_surrogate = torch.min(surr1, surr2)
        
        ratio_inv = 1.0 / ratio
        kl_penalty = ratio_inv - torch.log(ratio_inv) - 1.0
        loss = - (clipped_surrogate - self.beta * kl_penalty).mean()
        
        # 5. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 6. Cleanup
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.buffer.clear_buffer()