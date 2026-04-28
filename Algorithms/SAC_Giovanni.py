import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import utils as bf

# ---------------------------------------------------------------
# SAC — Soft Actor-Critic
# ---------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
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


class QNetwork(nn.Module):
    def __init__(self, observation_space, action_dim, hidden_dim):
        super().__init__()
        self.extractor = bf.FeatureExtractor(observation_space)

        input_dim = self.extractor.feature_dim + action_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        features = self.extractor(state)
        x = torch.cat([features, action], dim=1)
        return self.fc(x)


class PolicyNetwork(nn.Module):
    def __init__(self, observation_space, action_dim, hidden_dim,
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.extractor = bf.FeatureExtractor(observation_space)

        self.fc = nn.Sequential(
            nn.Linear(self.extractor.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu      = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.extractor(state)
        x        = self.fc(features)
        mu       = self.mu(x)
        log_std  = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std    = log_std.exp()
        normal = Normal(mu, std)
        z      = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)


class SAC:
    def __init__(self, env, hidden_dim=256, lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, buffer_capacity=10000):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma  = gamma
        self.tau    = tau
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        self.total_steps = 0

        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

        if self.is_discrete:
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]
        self.action_dim = action_dim

        obs_space = env.observation_space

        # Reti
        self.actor     = PolicyNetwork(obs_space, action_dim, hidden_dim).to(self.device)
        self.q1        = QNetwork(obs_space, action_dim, hidden_dim).to(self.device)
        self.q2        = QNetwork(obs_space, action_dim, hidden_dim).to(self.device)
        self.q1_target = QNetwork(obs_space, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(obs_space, action_dim, hidden_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Ottimizzatori
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt     = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        # Entropia automatica
        self.target_entropy = -action_dim
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt      = optim.Adam([self.log_alpha], lr=lr)

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1

        update_interval = 50
        if len(self.memory) > 1000 and self.total_steps % update_interval == 0:
            for _ in range(update_interval):
                self.update(self.memory, 256)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            mu, _ = self.actor(state)
            return torch.tanh(mu).detach().cpu().numpy()[0]

        action, _ = self.actor.sample(state)
        action_array = action.detach().cpu().numpy()[0]

        if self.is_discrete:
            return np.argmax(action_array)
        return action_array

    def update(self, buffer, batch_size):
        s, a, r, s_next, done = buffer.sample(batch_size)
        s, a, r, s_next, done = map(
            lambda x: torch.FloatTensor(x).to(self.device),
            [s, a, r, s_next, done]
        )
        r, done = r.unsqueeze(1), done.unsqueeze(1)

        if self.is_discrete:
            a = F.one_hot(a.long(), num_classes=self.action_dim).float()
        elif len(a.shape) == 1:
            a = a.unsqueeze(1)

        # Target Q
        with torch.no_grad():
            next_a, next_log_p = self.actor.sample(s_next)
            q_target_next = torch.min(
                self.q1_target(s_next, next_a),
                self.q2_target(s_next, next_a)
            )
            target_q = r + (1 - done) * self.gamma * (
                q_target_next - self.log_alpha.exp() * next_log_p
            )

        # Critic
        q_loss = F.mse_loss(self.q1(s, a), target_q) + \
                 F.mse_loss(self.q2(s, a), target_q)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # Actor
        new_a, log_p = self.actor.sample(s)
        q_min        = torch.min(self.q1(s, new_a), self.q2(s, new_a))
        actor_loss   = (self.log_alpha.exp() * log_p - q_min).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Alpha
        alpha_loss = -(self.log_alpha * (log_p + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft update
        for t_net, s_net in zip([self.q1_target, self.q2_target], [self.q1, self.q2]):
            for t_p, s_p in zip(t_net.parameters(), s_net.parameters()):
                t_p.data.copy_(t_p.data * (1.0 - self.tau) + s_p.data * self.tau)