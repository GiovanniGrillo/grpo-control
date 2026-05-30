import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import gymnasium as gym
import numpy as np
import utils as bf
import os

# =========================================================================================
# ObjectRL-Style PPO Agent
# Features: Disjoint Actor/Critic Networks, Learnable Standard Deviation (LogStd), GAE
# =========================================================================================

class RolloutBuffer:
    """Storage for PPO on-policy data."""
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.is_terminals, self.values = [], [], []

    def clear(self):
        """Clears all stored rollout data."""
        del self.states[:]; del self.actions[:]; del self.logprobs[:]
        del self.rewards[:]; del self.is_terminals[:]; del self.values[:]

class PPOActorNet(nn.Module):
    """Separate Actor network with a learnable log-standard deviation for PPO."""
    def __init__(self, obs_space, action_space, hidden_dim=512):
        super(PPOActorNet, self).__init__()
        # Use the provided feature extractor
        self.extractor = bf.FeatureExtractor(obs_space)
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        self.action_dim = action_space.n if self.is_discrete else action_space.shape[0]

        # Actor architecture (similar to MLP in ObjectRL)
        self.arch = nn.Sequential(
            nn.Linear(self.extractor.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim)
        )

        if not self.is_discrete:
            # Learnable parameter for standard deviation (ObjectRL Style)
            self.action_logstd = nn.Parameter(torch.zeros(self.action_dim))
            self.upper_clamp = 1.0  # Prevents variance explosion

    def forward(self, state, action=None, is_training=True):
        feats = self.extractor(state)
        out = self.arch(feats)

        if self.is_discrete:
            dist = Categorical(logits=out)
        else:
            # Tanh bounds output for environments like CarRacing, scaled afterwards
            action_mean = torch.tanh(out)
            # Clamp and expand the learnable logstd
            action_logstd = self.action_logstd.clamp(max=self.upper_clamp).expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            # Create a normal distribution with the mean and standard deviation
            dist = Normal(loc=action_mean, scale=action_std)

        # Sample action if not provided (during rollout)
        if action is None:
            if is_training:
                action = dist.sample()
            else:
                # Use mode/mean for evaluation
                action = torch.argmax(out, dim=-1) if self.is_discrete else action_mean

        logprob = dist.log_prob(action)
        # Sum logprobs over action dimensions for continuous spaces
        if not self.is_discrete:
            logprob = logprob.sum(dim=-1)

        return action, logprob, dist.entropy()

class PPOCriticNet(nn.Module):
    """Separate Critic network (Value Function) for disjoint architecture."""
    def __init__(self, obs_space, hidden_dim=512):
        super(PPOCriticNet, self).__init__()
        # Dedicated feature extractor! (Disjoint Networks)
        self.extractor = bf.FeatureExtractor(obs_space)
        
        self.net = nn.Sequential(
            nn.Linear(self.extractor.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        # Squeeze the last dimension to match reward shape
        return self.net(self.extractor(state)).squeeze(-1)

class PPO:
    """Proximal Policy Optimization following the ObjectRL structure."""
    def __init__(self, env, hidden_dim=512, lr_actor=3e-4, lr_critic=1e-3, 
                 gamma=0.99, gae_lambda=0.95, clip_rate=0.2, entropy_coef=0.01, 
                 update_timestep=4000, K_epochs=10, batch_size=256):
                 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_rate = clip_rate
        self.entropy_coef = entropy_coef
        self.update_timestep = update_timestep
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.time_step = 0
        
        self.buffer = RolloutBuffer()
        
        # Initialize separate networks
        self.actor = PPOActorNet(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.critic = PPOCriticNet(env.observation_space, hidden_dim).to(self.device)
        
        # Create a copy of the actor for calculating PPO ratios
        self.actor_old = PPOActorNet(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # Initialize separate optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.updated = False  # Flag to track if an update has occurred

    def consume_update_flag(self):
        if self.updated:
            self.updated = False
            return True
        return False

    def select_action(self, state, evaluate=False):
        """Selects an action based on the current policy."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action, logprob, _ = self.actor_old(state_t, is_training=not evaluate)
            value = self.critic(state_t)
            
            if not evaluate:
                self.buffer.states.append(state_t)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(logprob)
                self.buffer.values.append(value)

            if self.actor.is_discrete:
                return action.item()
            else:
                return action.cpu().numpy()[0]

    def step(self, state, action, reward, next_state, done, pos = None):
        """Processes one environment step and triggers updates if necessary."""
        self.time_step += 1
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
        
        stats = {} # Default empty stats

        # Trigger learning when the update timestep is reached
        if self.time_step % self.update_timestep == 0:
            with torch.no_grad():
                next_state_t = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
                next_value = self.critic(next_state_t)
            
            # Catch the returned dictionary from learn()
            stats = self.learn(next_value)
            self.updated = True  

        return stats

    def learn(self, next_value):
        """Learns from experience memory using PPO update rules and GAE."""
        # 1. Prepare Tensors
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(self.device)
        
        old_states = torch.cat(self.buffer.states, dim=0).detach()
        old_actions = torch.cat(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.cat(self.buffer.logprobs, dim=0).detach()
        values = torch.cat(self.buffer.values, dim=0).detach()

        # 2. Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        next_v = next_value.item()

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - is_terminals[t]
            delta = rewards[t] + self.gamma * next_v * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            next_v = values[t]

        returns = advantages + values
        # Normalize Advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = old_states.size(0)
        
        # 3. Optimization using Mini-Batches
        for _ in range(self.K_epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                # Evaluate current policy on the mini-batch
                _, new_logprobs, entropy = self.actor(old_states[batch_idx], old_actions[batch_idx])
                
                if not self.actor.is_discrete:
                    entropy = entropy.sum(dim=-1)

                # Calculate ratio: exp(log(pi) - log(pi_old))
                ratio = torch.exp(new_logprobs - old_logprobs[batch_idx])
                
                # Calculate clipped surrogate loss
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_rate, 1.0 + self.clip_rate) * advantages[batch_idx]
                
                # Combine actor loss with entropy regularization
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                
                # Perform actor update
                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) # Gradient clipping
                self.actor_optim.step()

                # Calculate and perform critic update (Separate Network)
                current_values = self.critic(old_states[batch_idx])
                critic_loss = nn.MSELoss()(current_values, returns[batch_idx])
                
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # Gradient clipping
                self.critic_optim.step()

        # Update the old policy and clear the rollout buffer
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.buffer.clear()

        # Calculate current action standard deviation for logging
        current_std = 0.0
        if not self.actor.is_discrete:
            with torch.no_grad():
                # self.actor.action_logstd is the learnable parameter
                current_std = torch.exp(self.actor.action_logstd).mean().item()

        # Return the metrics matching the plotter's expected column names
        return {
            "loss_actor": actor_loss.item(),
            "loss_critic": critic_loss.item(),
            # Mapping to elite so the specific AGRPO plotter picks it up
            "tier_elite_action_std": current_std 
        }

    def save_checkpoint(self, path, ep, eval_rewards, seed_logs):
        """Saves current state for recovery."""
        checkpoint = {
            'episode': ep,
            'eval_rewards': eval_rewards,
            'seed_logs': seed_logs,
            'time_step': self.time_step,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Loads state from file for recovery."""
        if not os.path.exists(path):
            return {'episode': 0, 'eval_rewards': [], 'seed_logs': []}
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_old.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.time_step = checkpoint['time_step']
        return checkpoint