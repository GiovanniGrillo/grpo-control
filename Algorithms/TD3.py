import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from copy import deepcopy
import Basic_Functions as bf

# =========================================================================================
# TD3 Agent Implementation
# =========================================================================================

class ReplayBuffer:
    """Experience Replay Buffer (Identical to SAC_Robin.py for consistency)"""
    def __init__(self, capacity=100000):
        self.buffer = []                                           
        self.capacity = capacity                                   
        self.position = 0                                          

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)                               
        self.buffer[self.position] = (state, action, reward, next_state, done) 
        self.position = (self.position + 1) % self.capacity        

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)             
        state, action, reward, next_state, done = map(np.stack, zip(*batch)) 
        return state, action, reward, next_state, done             

    def __len__(self):
        return len(self.buffer)                                    

class Actor(nn.Module):
    def __init__(self, observation_space, act_dim, act_high, hidden=256):
        super().__init__()
        # Inherit the common FeatureExtractor for image/vector inputs
        self.extractor = bf.FeatureExtractor(observation_space)
        self.net = nn.Sequential(
            nn.Linear(self.extractor.feature_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.register_buffer("act_scale", torch.tensor(act_high, dtype=torch.float32))

    def forward(self, state):
        x = self.extractor(state)
        out = self.net(x)
        return torch.tanh(out) * self.act_scale

class Critic(nn.Module):
    """Twin Q-networks."""
    def __init__(self, observation_space, act_dim, hidden=256):
        super().__init__()
        # Q1 Architecture
        self.extractor1 = bf.FeatureExtractor(observation_space)
        self.q1_net = nn.Sequential(
            nn.Linear(self.extractor1.feature_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        
        # Q2 Architecture
        self.extractor2 = bf.FeatureExtractor(observation_space)
        self.q2_net = nn.Sequential(
            nn.Linear(self.extractor2.feature_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state, action):
        x1 = self.extractor1(state)
        q1 = self.q1_net(torch.cat([x1, action], dim=-1))
        
        x2 = self.extractor2(state)
        q2 = self.q2_net(torch.cat([x2, action], dim=-1))
        return q1, q2

    def Q1(self, state, action):
        x1 = self.extractor1(state)
        return self.q1_net(torch.cat([x1, action], dim=-1))

class TD3:
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent"""
    def __init__(self, env, hidden_dim=256, lr=3e-4, gamma=0.99, tau=5e-3, 
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2, expl_noise=0.1, buffer_capacity=100000):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.expl_noise = expl_noise
        self._update_count = 0
        
        # Check if action space is continuous
        if isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("TD3 requires a continuous action space (Box).")

        self.action_dim = env.action_space.shape[0]
        self.act_high_np = env.action_space.high.astype(np.float32)
        self.act_high = torch.tensor(self.act_high_np, device=self.device)

        self.memory = ReplayBuffer(capacity=buffer_capacity)
        obs_space = env.observation_space

        # Networks
        self.actor = Actor(obs_space, self.action_dim, self.act_high_np, hidden_dim).to(self.device)
        self.critic = Critic(obs_space, self.action_dim, hidden_dim).to(self.device)
        
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).cpu().numpy()
        
        if not evaluate:
            noise = np.random.normal(0, self.expl_noise * self.act_high_np, size=self.action_dim).astype(np.float32)
            action = np.clip(action + noise, -self.act_high_np, self.act_high_np)
            
        return action

    def step(self, state, action, reward, next_state, done):
        """Unified step function called by Simulation.py"""
        self.memory.push(state, action, reward, next_state, done)
        
        # Start updating once we have enough samples
        if len(self.memory) > 256: 
            self.update(batch_size=256)

    def update(self, batch_size):
        self._update_count += 1
        s, a, r, s_next, done = self.memory.sample(batch_size)
        
        s, a, r, s_next, done = map(lambda x: torch.FloatTensor(x).to(self.device), [s, a, r, s_next, done])
        r, done = r.unsqueeze(1), done.unsqueeze(1)

        # ---- Critic update ----
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.actor_target(s_next) + noise).clamp(-self.act_high, self.act_high)

            q1_tgt, q2_tgt = self.critic_target(s_next, next_act)
            q_tgt = r + self.gamma * (1.0 - done) * torch.min(q1_tgt, q2_tgt)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, q_tgt) + F.mse_loss(q2, q_tgt)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---- Delayed actor update ----
        if self._update_count % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(s, self.actor(s)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft updates
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)