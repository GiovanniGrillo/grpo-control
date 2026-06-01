import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import utils as bf

# =========================================================================================
# SAC Agent Implementation
# =========================================================================================

class ReplayBuffer:
    """Experience Replay Buffer to store and sample transitions."""
    def __init__(self, capacity = 10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        # Store as lightweight numpy arrays with explicit dtypes to minimize
        # work when converting to torch tensors later.
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        s = np.asarray(state, dtype=np.float32)
        a = np.asarray(action, dtype=np.float32)
        r = float(reward)
        s_next = np.asarray(next_state, dtype=np.float32)
        d = bool(done)
        self.buffer[self.position] = (s, a, r, s_next, d)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    """Critic Network: Estimates the soft Q-value Q(s, a)."""
    def __init__(self, observation_space, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.extractor = bf.FeatureExtractor(observation_space)
        
        self.fc = nn.Sequential(
            nn.Linear(self.extractor.feature_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = self.extractor(state)
        return self.fc(torch.cat([x, action], 1))

class PolicyNetwork(nn.Module):
    """Actor Network: Outputs a stochastic policy (Gaussian distribution)."""
    def __init__(self, observation_space, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.extractor = bf.FeatureExtractor(observation_space)
        
        self.fc = nn.Sequential(
            nn.Linear(self.extractor.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.extractor(state)
        x = self.fc(x)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

class SAC:
    """Soft Actor-Critic Agent."""
    def __init__(self, env, hidden_dim=512, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_capacity=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Enable cuDNN autotuner for potential CNN speedups on fixed-size inputs
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.gamma, self.tau = gamma, tau
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.total_steps = 0
        self.updated = False  # [PPO] Flag to track if an update has occurred

        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]
        self.action_dim = action_dim

        obs_space = env.observation_space

        # Initialize Networks
        self.actor = PolicyNetwork(obs_space, action_dim, hidden_dim).to(self.device)
        self.q1 = QNetwork(obs_space, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(obs_space, action_dim, hidden_dim).to(self.device)
        self.q1_target = QNetwork(obs_space, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(obs_space, action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        # Automatic Entropy Tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

    # [PPO] Consume-and-reset flag: returns True once after each update
    def consume_update_flag(self):
        if self.updated:
            self.updated = False
            return True
        return False

    def select_action(self, state, evaluate=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if evaluate:
            with torch.no_grad():
                mu, _ = self.actor(state)
                return torch.tanh(mu).detach().cpu().numpy()[0]
        action, _ = self.actor.sample(state)
        action_array = action.detach().cpu().numpy()[0]
        if self.is_discrete:
            return np.argmax(action_array)
        return action_array

    def step(self, state, action, reward, next_state, done, pos=None):  # [PPO] pos parameter added
        """Processes one environment step and triggers updates if necessary."""
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1

        stats = None
        update_interval = 50
  
        if len(self.memory) > 1000 and self.total_steps % update_interval == 0:
            for _ in range(update_interval):
                stats = self.update(self.memory, 256)
            # Trigger evaluation and logging every 2000 steps to prevent massive overhead
            if self.total_steps % 2000 == 0:
                self.updated = True

        return stats

    def update(self, buffer, batch_size):
        s, a, r, s_next, done = buffer.sample(batch_size)
        # Convert to tensors directly on the target device to avoid extra copies
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        s_next = torch.as_tensor(s_next, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        r, done = r.unsqueeze(1), done.unsqueeze(1)

        if self.is_discrete:
            a = F.one_hot(a.long(), num_classes=self.action_dim).float()
        elif len(a.shape) == 1:
            a = a.unsqueeze(1)

        # Target Q calculation
        with torch.no_grad():
            next_a, next_log_p = self.actor.sample(s_next)
            target_q = r + (1 - done) * self.gamma * (
                torch.min(self.q1_target(s_next, next_a), self.q2_target(s_next, next_a))
                - self.log_alpha.exp() * next_log_p
            )

        # Critic update with gradient clipping [PPO]
        q_loss = F.mse_loss(self.q1(s, a), target_q) + F.mse_loss(self.q2(s, a), target_q)
        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 0.5)  # [PPO]
        self.q_opt.step()

        # Actor update with gradient clipping [PPO]
        new_a, log_p = self.actor.sample(s)
        q_min = torch.min(self.q1(s, new_a), self.q2(s, new_a))
        actor_loss = (self.log_alpha.exp() * log_p - q_min).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # [PPO]
        self.actor_opt.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_p + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft update of target networks
        for t, s_net in zip([self.q1_target, self.q2_target], [self.q1, self.q2]):
            for target_param, param in zip(t.parameters(), s_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # --- NEW: Calculate and return metrics for logging ---
        current_std = 0.0
        with torch.no_grad():
            # Get the current standard deviation from the actor network for the batch
            _, log_std = self.actor(s)
            current_std = log_std.exp().mean().item()

        # Return the metrics matching the plotter's expected column names
        return {
            "loss_critic": q_loss.item(),
            "loss_actor": actor_loss.item(),
            "loss_alpha": alpha_loss.item(), # Specific to SAC
            "tier_elite_action_std": current_std
        }

    def save_checkpoint(self, path, ep, eval_rewards, seed_logs):  # [PPO] seed_logs parameter added
        """Saves everything needed to resume training and preserve research data."""
        checkpoint = {
            'episode': ep,
            'eval_rewards': eval_rewards,
            'seed_logs': seed_logs,              # [PPO]
            'total_steps': self.total_steps,
            'log_alpha': self.log_alpha.detach().cpu().item(),
            # Networks
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            # Optimizers
            'q_opt_state_dict': self.q_opt.state_dict(),
            'actor_opt_state_dict': self.actor_opt.state_dict(),
            'alpha_opt_state_dict': self.alpha_opt.state_dict(),
            # Replay Buffer (crucial for off-policy)
            'buffer': self.memory.buffer
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Loads the checkpoint and returns the metadata for the simulation loop."""
        if not os.path.exists(path):
            return {'episode': 0, 'eval_rewards': [], 'seed_logs': []}  # [PPO] seed_logs default

        ckpt = torch.load(path, map_location=self.device)

        # Restore Networks
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.q1.load_state_dict(ckpt['q1_state_dict'])
        self.q2.load_state_dict(ckpt['q2_state_dict'])

        # Restore Optimizers
        self.q_opt.load_state_dict(ckpt['q_opt_state_dict'])
        self.actor_opt.load_state_dict(ckpt['actor_opt_state_dict'])
        self.alpha_opt.load_state_dict(ckpt['alpha_opt_state_dict'])

        # Restore Scalars & Tensors
        # `log_alpha` saved as scalar; restore to tensor safely on device
        try:
            self.log_alpha.data.copy_(torch.tensor(ckpt['log_alpha'], device=self.device))
        except Exception:
            # Fallback if ckpt stores tensor-like object
            self.log_alpha.data.copy_(torch.as_tensor(ckpt['log_alpha']).to(self.device))
        self.total_steps = ckpt['total_steps']
        self.memory.buffer = ckpt['buffer']

        # Sync Target Networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        return ckpt  # Full dict so Simulation.py gets 'episode', 'eval_rewards', 'seed_logs'