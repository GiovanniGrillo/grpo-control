import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import random
import matplotlib.pyplot as plt
import os
import pandas as pd

# =========================================================================================
# SAC Agent Implementation
# =========================================================================================

class ReplayBuffer:
    """Experience Replay Buffer to store and sample transitions."""
    def __init__(self, capacity):
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

class FeatureExtractor(nn.Module):
    """
    Automated Feature Extractor: 
    Detects if the input is an image (3D) or a vector (1D) and applies CNN or Identity.
    """
    def __init__(self, observation_space):
        super(FeatureExtractor, self).__init__()
        
        # Check for image input (Height, Width, Channels)
        if len(observation_space.shape) == 3:
            self.is_image = True
            h, w, c = observation_space.shape # Gymnasium standard: (H, W, C)
            
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten()
            )
            
            # Compute output dimension of CNN
            with torch.no_grad():
                dummy_input = torch.zeros(1, c, h, w)
                self.feature_dim = self.cnn(dummy_input).shape[1]
        else:
            self.is_image = False
            self.feature_dim = observation_space.shape[0]

    def forward(self, x):
        if self.is_image:
            # Permute from (Batch, H, W, C) to (Batch, C, H, W) and normalize pixels to [0, 1]
            x = x.permute(0, 3, 1, 2) / 255.0
            return self.cnn(x)
        return x 

class QNetwork(nn.Module):
    """Critic Network: Estimates the soft Q-value Q(s, a)."""
    def __init__(self, observation_space, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.extractor = FeatureExtractor(observation_space) # Image/Vector processing
        
        self.fc = nn.Sequential(
            nn.Linear(self.extractor.feature_dim + action_dim, hidden_dim), nn.ReLU(),
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
        self.extractor = FeatureExtractor(observation_space)
        
        self.fc = nn.Sequential(
            nn.Linear(self.extractor.feature_dim, hidden_dim), nn.ReLU(),
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
        z = normal.rsample() # Reparameterization trick
        action = torch.tanh(z)
        
        # Enforce action bounds and calculate log probability
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)

class SAC:
    """Soft Actor-Critic Agent."""
    def __init__(self, env, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma, self.tau = gamma, tau
        
        # Robust action_dim detection
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
        else:
            # For Box / Continuous spaces
            action_dim = env.action_space.shape[0]

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

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            mu, _ = self.actor(state)
            return torch.tanh(mu).detach().cpu().numpy()[0]
        action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, buffer, batch_size):
        s, a, r, s_next, done = buffer.sample(batch_size)
        s, a, r, s_next, done = map(lambda x: torch.FloatTensor(x).to(self.device), [s, a, r, s_next, done])
        r, done = r.unsqueeze(1), done.unsqueeze(1)

        # Target Q calculation
        with torch.no_grad():
            next_a, next_log_p = self.actor.sample(s_next)
            target_q = r + (1 - done) * self.gamma * (
                torch.min(self.q1_target(s_next, next_a), self.q2_target(s_next, next_a)) 
                - self.log_alpha.exp() * next_log_p
            )

        # Critic update
        q_loss = F.mse_loss(self.q1(s, a), target_q) + F.mse_loss(self.q2(s, a), target_q)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

        # Actor update
        new_a, log_p = self.actor.sample(s)
        q_min = torch.min(self.q1(s, new_a), self.q2(s, new_a))
        actor_loss = (self.log_alpha.exp() * log_p - q_min).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_p + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # Soft update of target networks
        for t, s_net in zip([self.q1_target, self.q2_target], [self.q1, self.q2]):
            for target_param, param in zip(t.parameters(), s_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

# =========================================================================================
# Visualization and Training Loop
# =========================================================================================

def save_and_plot_results(rewards_dict, folder="plots"):
    if not os.path.exists(folder): os.makedirs(folder)
    
    # Save CSV data
    pd.DataFrame.from_dict(rewards_dict, orient='index').transpose().to_csv(os.path.join(folder, "data.csv"))

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, (env_name, rewards) in enumerate(rewards_dict.items()):
        axs[i].plot(rewards, alpha=0.3, label='Raw')
        if len(rewards) > 10:
            axs[i].plot(pd.Series(rewards).rolling(10).mean(), label='MA10', linewidth=2)
        axs[i].set_title(env_name); axs[i].legend()
    
    plt.savefig(os.path.join(folder, "learning_curves.png"))
    plt.close()

# Main Loop
ENV_NAMES = ["CarRacing-v3", "CartPole-v1", "Acrobot-v1"]
all_results = {}

for env_name in ENV_NAMES:
    env = gym.make(env_name)
    agent = SAC(env)
    buffer = ReplayBuffer(50000) # Smaller buffer for RAM safety with images 50000
    env_rewards = []

    for ep in range(1000):                                       # Episode loop 1000
        state, _ = env.reset()                                  # Reset environment
        ep_reward = 0
        for t in range(1000):                                     # Step loop 1000
            action = agent.select_action(state)
            
            # Action Rescaling for CarRacing
            if env_name == "CarRacing-v3":
                actual_action = np.array([action[0], (action[1]+1)/2, (action[2]+1)/2])
            else:
                actual_action = np.argmax(action)

            next_state, reward, done, truncated, _ = env.step(actual_action)
            buffer.push(state, action, reward, next_state, done)
            
            if len(buffer) > 128: agent.update(buffer, 128)
            state, ep_reward = next_state, ep_reward + reward
            if done or truncated: break
        
        env_rewards.append(ep_reward)
        if ep % 5 == 0: print(f"{env_name} Ep {ep}: {ep_reward}")

    all_results[env_name] = env_rewards
    env.close()

save_and_plot_results(all_results)