# =========================================================================================
# PPO Agent Implementation
# =========================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
import gymnasium as gym
import utils as bf

class RolloutBuffer:
    """Save for PPO (On-Policy Data)"""
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.is_terminals, self.values = [], [], []

    def clear(self):
        del self.states[:]; del self.actions[:]; del self.logprobs[:]
        del self.rewards[:]; del self.is_terminals[:]; del self.values[:]

class ActorCritic(nn.Module):
    """Combined Network for PPO"""
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super(ActorCritic, self).__init__()
        # Sample feature extractor based on the observation space type
        self.extractor = bf.FeatureExtractor(observation_space)
        
        # Action Space Type Check: Discrete vs Continuous
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        action_dim = action_space.n if self.is_discrete else action_space.shape[0]

        # Actor (Policy)
        self.actor_base = nn.Sequential(
            nn.Linear(self.extractor.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # Std for Continuous Actions (if applicable)
        if not self.is_discrete:
            self.action_dim = action_space.shape[0]
            # We handle action_var manually in the PPO class, not as a learned nn.Parameter
            self.action_var = torch.full((self.action_dim,), 0.5).to(self.device)

        # Critic (Value Function)
        self.critic = nn.Sequential(
            nn.Linear(self.extractor.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def set_action_std(self, std):
        """Manually updates the standard deviation for the action distribution."""
        self.action_var = torch.full((self.action_dim,), std * std).to(self.device)

    def act(self, state, evaluate=False):
        x = self.extractor(state)
        
        if self.is_discrete:
            logits = self.actor_head(self.actor_base(x))
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if evaluate else dist.sample()
            logprob = dist.log_prob(action)
        else:
            mean = self.actor_head(self.actor_base(x))
            dist = MultivariateNormal(mean, torch.diag(self.action_var))
            
            if evaluate:
                u = mean
                action = torch.tanh(u)
            else:
                u = dist.sample()
                action = torch.tanh(u)
                
            logprob = dist.log_prob(u) - torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1)
            
        val = self.critic(x)
        return action.detach(), logprob.detach(), val.detach()

    def evaluate(self, state, action):
        x = self.extractor(state)
        
        if self.is_discrete:
            logits = self.actor_head(self.actor_base(x))
            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(action)
        else:
            mean = self.actor_head(self.actor_base(x))
            dist = MultivariateNormal(mean, torch.diag(self.action_var))
            u = torch.atanh(torch.clamp(action, -0.999, 0.999))
            logprobs = dist.log_prob(u) - torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=-1)
            
        return logprobs, self.critic(x), dist.entropy()

class PPO:
    """Proximal Policy Optimization Agent"""
    def __init__(self, env, hidden_dim=256, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2, update_timestep=2000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.action_std = 1            # Initial standard deviation
        self.std_decay_rate = 0.000005   # Linear decay per step
        self.min_std = 0.05              # Minimum noise floor
        
        self.update_timestep = update_timestep
        self.time_step = 0
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Copy for the update
        self.policy_old = ActorCritic(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state, evaluate=evaluate)
            
            if not evaluate:
                # PPO saves values
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.values.append(state_val)

            # Conversion for Gymnasium
            if self.policy_old.is_discrete:
                return action.item()
            else:
                return action.cpu().numpy()[0]

    def decay_action_std(self):
        """
        Linearly decreases the action noise (std) over time.
        Ensures the agent becomes more precise as training progresses.
        """
        self.action_std -= self.std_decay_rate
        if self.action_std < self.min_std:
            self.action_std = self.min_std
        
        # Sync the new std with the policy networks
        self.policy.set_action_std(self.action_std)
        self.policy_old.set_action_std(self.action_std)

    def step(self, state, action, reward, next_state, done):
        """Wird aufgerufen, um das Feedback der Umgebung zu speichern"""
        self.time_step += 1
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

        # Decay exploration noise every step
        self.decay_action_std()

        # Update the policy if it's time
        if self.time_step % self.update_timestep == 0:
            self.update()

    def update(self, entropy_coef=0.05):
        """
        Updates the policy using the collected rollout buffer.
        Implements Generalized Advantage Estimation (GAE) for variance reduction.
        """
        # 1. Convert buffer lists to tensors
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        is_terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(self.device)
        # values: V(s) predicted by the critic during the rollout
        state_values = torch.stack(self.buffer.values).squeeze().detach()
        
        advantages = torch.zeros_like(rewards).to(self.device)
        gae_lambda = 0.95  # Standard GAE smoothing parameter
        last_gae_lam = 0
        
        # 2. Backward pass to compute GAE
        # We iterate backwards because A_t depends on A_{t+1}
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # If the buffer ends, we assume next_value is 0 or bootstrap if necessary
                next_value = 0 
                next_non_terminal = 1.0 - is_terminals[t]
            else:
                next_value = state_values[t+1]
                next_non_terminal = 1.0 - is_terminals[t]

            # TD-error (delta): r + gamma * V(s_next) - V(s)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - state_values[t]
            
            # Recursive GAE calculation: A_t = delta + gamma * lambda * A_{t+1}
            advantages[t] = last_gae_lam = delta + self.gamma * gae_lambda * next_non_terminal * last_gae_lam
            
        # Returns (Targets for the Critic) = Advantages + V(s)
        returns = advantages + state_values
        
        # Standardizing advantages is CRUCIAL for PPO stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        old_states = torch.stack(self.buffer.states, dim=0).squeeze(1).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).squeeze(1).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).squeeze(1).detach()

        # 3. K_epochs of PPO updates
        batch_size = 64
        indices = torch.randperm(len(old_states))
        for start in range(0, len(old_states), batch_size):
            batch_idx = indices[start:start+batch_size]
            batch_states = old_states[batch_idx]
            batch_actions = old_actions[batch_idx]
            
            logprobs, state_values, dist_entropy = self.policy.evaluate(batch_states, batch_actions)
            state_values = torch.squeeze(state_values)
            
            # Ratio berechnen
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Surrogate Loss - NUTZE DIE GAE ADVANTAGES VON OBEN
            surr1 = ratios * advantages # <--- Hier die GAE-Variable nutzen!
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Critic Loss: Nutze die GAE-'returns' statt der rohen 'rewards'
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - entropy_coef * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Alte Policy updaten & Buffer leeren
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save_checkpoint(self, path, ep, eval_rewards):
        """
        Saves the current state of the PPO agent.
        Since PPO is on-policy, we don't need to save the RolloutBuffer.
        """
        checkpoint = {
            'episode': ep,
            'eval_rewards': eval_rewards,
            'time_step': self.time_step,
            'action_std': self.action_std,
            # Networks
            'policy_state_dict': self.policy.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            # Optimizer
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        # print(f"PPO Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """
        Loads the agent state from a checkpoint file.
        Returns the full dictionary to allow the simulation loop to resume.
        """
        import os
        if not os.path.exists(path):
            return {'episode': 0, 'eval_rewards': []}

        checkpoint = torch.load(path, map_location=self.device)

        # Restore network weights
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        
        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore exploration parameters and counters
        self.time_step = checkpoint['time_step']
        self.action_std = checkpoint['action_std']
        
        # Ensure the loaded action_std is applied to the distribution logic
        self.policy.set_action_std(self.action_std)
        self.policy_old.set_action_std(self.action_std)

        return checkpoint