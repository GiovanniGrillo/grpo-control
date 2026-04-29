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
    def __init__(self, capacity = 10000):                                           # Buffer-Capacity
        self.buffer = []                                                            # List to store transitions
        self.capacity = capacity                                                    # Maximum number of transitions to store
        self.position = 0                                                           # Pointer to the current position in the buffer for overwriting old data

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:                                        # If buffer is not full, append a new transition
            self.buffer.append(None)                                                # Placeholder for new transition
        self.buffer[self.position] = (state, action, reward, next_state, done)      # Store the new transition at the current position
        self.position = (self.position + 1) % self.capacity                         # Move the position pointer, wrapping around to the start if it exceeds capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)                              # Randomly sample a batch of transitions from the buffer
        state, action, reward, next_state, done = map(np.stack, zip(*batch))        # Unzip the batch of transitions into separate arrays for states, actions, rewards, next states, and done flags
        return state, action, reward, next_state, done                              # Return the sampled transitions as separate arrays

    def __len__(self):
        return len(self.buffer)                                                     # Return the current number of transitions stored in the buffer

class QNetwork(nn.Module):
    """Critic Network: Estimates the soft Q-value Q(s, a)."""
    def __init__(self, observation_space, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.extractor = bf.FeatureExtractor(observation_space) # Image/Vector processing
        
        self.fc = nn.Sequential(
            nn.Linear(self.extractor.feature_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )                                                                           # Output is a single Q-value for the given state-action pair
        
    def forward(self, state, action):
        x = self.extractor(state)                                                   # Extract features from the state using the shared feature extractor
        return self.fc(torch.cat([x, action], 1))                                   # Concatenate the extracted state features with the action and pass through the fully connected layers to get the Q-value

class PolicyNetwork(nn.Module):
    """Actor Network: Outputs a stochastic policy (Gaussian distribution)."""
    def __init__(self, observation_space, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):              # log_std_min and log_std_max are used to clamp the standard deviation of the action distribution for numerical stability
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min                                              # Minimum log standard deviation to prevent the policy from becoming too deterministic
        self.log_std_max = log_std_max                                              # Maximum log standard deviation to prevent the policy from becoming too stochastic
        self.extractor = bf.FeatureExtractor(observation_space)                     # Image/Vector processing
        
        self.fc = nn.Sequential(
            nn.Linear(self.extractor.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )                                                                           # Shared fully connected layers for the policy network
        self.mu = nn.Linear(hidden_dim, action_dim)                                 # Output layer for the mean of the action distribution
        self.log_std = nn.Linear(hidden_dim, action_dim)                            # Output layer for the log standard deviation of the action distribution

    def forward(self, state):
        x = self.extractor(state)                                                       # Extract features from the state using the shared feature extractor
        x = self.fc(x)                                                                  # Pass the extracted features through the shared fully connected layers
        mu = self.mu(x)                                                                 # Compute the mean of the action distribution from the output of the fully connected layers
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)      # Compute the log standard deviation of the action distribution and clamp it to the specified range for numerical stability
        return mu, log_std                                                              # Return the mean and log standard deviation of the action distribution

    def sample(self, state):
        mu, log_std = self.forward(state)                               # Get the mean and log standard deviation from the forward pass
        std = log_std.exp()                                             # Convert log standard deviation to standard deviation
        normal = Normal(mu, std)                                        # Create a normal distribution with the computed mean and standard deviation
        z = normal.rsample()                                            # Reparameterization trick
        action = torch.tanh(z)                                          # Apply tanh to enforce action bounds between -1 and 1
        
        # Enforce action bounds and calculate log probability
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)             # Adjust log probability for the tanh transformation to ensure correct gradients during backpropagation
        return action, log_prob.sum(-1, keepdim=True)                                   # Return the sampled action and its log probability

class SAC:
    """Soft Actor-Critic Agent."""
    def __init__(self, env, hidden_dim=256, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, buffer_capacity=10000):              # SAC hyperparameters: hidden_dim for the neural networks, learning rate, discount factor gamma, soft update coefficient tau, initial entropy coefficient alpha, and replay buffer capacity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                      # Use GPU if available, otherwise fall back to CPU
        self.gamma, self.tau = gamma, tau                                                               # Store the discount factor and soft update coefficient
        self.memory = ReplayBuffer(capacity=buffer_capacity)                                            # Initialize the replay buffer with the specified capacity
        self.total_steps = 0
        # Determine if the environment uses discrete or continuous actions                  
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)                            

        # Robust action_dim detection
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
        else:
            # For Box / Continuous spaces
            action_dim = env.action_space.shape[0]
        self.action_dim = action_dim

        obs_space = env.observation_space

        # Initialize Networks
        self.actor = PolicyNetwork(obs_space, action_dim, hidden_dim).to(self.device)       # Actor network for learning the policy
        self.q1 = QNetwork(obs_space, action_dim, hidden_dim).to(self.device)               # First critic network for estimating Q-values
        self.q2 = QNetwork(obs_space, action_dim, hidden_dim).to(self.device)               # Second critic network for estimating Q-values (used for the Clipped Double Q-Learning technique in SAC)
        self.q1_target = QNetwork(obs_space, action_dim, hidden_dim).to(self.device)        # Target network for the first critic, used for stable Q-value updates
        self.q2_target = QNetwork(obs_space, action_dim, hidden_dim).to(self.device)        # Target network for the second critic, used for stable Q-value updates
        self.q1_target.load_state_dict(self.q1.state_dict())                                # Initialize the target networks with the same weights as the critic networks
        self.q2_target.load_state_dict(self.q2.state_dict())                                # Initialize the target networks with the same weights as the critic networks

        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        # Automatic Entropy Tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)                   # Store the transition in the replay buffer
        self.total_steps += 1                                                       # Increment the total step count

        update_interval = 50
        if len(self.memory) > 1000 and self.total_steps % update_interval == 0:     # If there are enough samples in the replay buffer and it's time to update
            for _ in range(update_interval):                                        # Perform multiple updates at once to improve sample efficiency
                self.update(self.memory, 256)                                       # Update the SAC agent using a batch of transitions sampled from the replay buffer

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)           # Convert the state to a PyTorch tensor and add a batch dimension
        if evaluate:
            mu, _ = self.actor(state)                                           # Get the mean action from the policy network for evaluation (deterministic action selection)
            return torch.tanh(mu).detach().cpu().numpy()[0]                     # Apply tanh to enforce action bounds and convert to numpy array for the environment
        action, _ = self.actor.sample(state)                                    # Sample an action from the policy network for training (stochastic action selection)
        # Convert tensor to numpy array
        action_array = action.detach().cpu().numpy()[0]             
        # The agent formats the action itself to match the environment's requirements
        if self.is_discrete:
            return np.argmax(action_array)
        return action_array
        

    def update(self, buffer, batch_size):
        s, a, r, s_next, done = buffer.sample(batch_size)                                                           # Sample a batch of transitions from the replay buffer
        s, a, r, s_next, done = map(lambda x: torch.FloatTensor(x).to(self.device), [s, a, r, s_next, done])        # Convert the sampled transitions to PyTorch tensors and move them to the appropriate device (GPU or CPU)
        r, done = r.unsqueeze(1), done.unsqueeze(1)                                                                 # Reshape rewards and done flags to have a batch dimension for proper broadcasting during calculations

        if self.is_discrete:
            # Convert 1D integer action to 2D one-hot vector (batch_size, action_dim)
            a = F.one_hot(a.long(), num_classes=self.action_dim).float()
        elif len(a.shape) == 1:
            # Catch case where 1D continuous action lost its second dimension
            a = a.unsqueeze(1)

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
    
    def save_checkpoint(self, path, ep, eval_rewards):
        """Saves everything needed to resume training and preserve research data."""
        checkpoint = {
            'episode': ep,
            'eval_rewards': eval_rewards,
            'total_steps': self.total_steps,
            'log_alpha': self.log_alpha,
            # Networks
            'encoder_state_dict': self.encoder.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            # Optimizers
            'q_opt_state_dict': self.q_opt.state_dict(),
            'actor_opt_state_dict': self.actor_opt.state_dict(),
            'alpha_opt_state_dict': self.alpha_opt.state_dict(),
            # Replay Buffer (Crucial for Off-Policy)
            'buffer': self.memory.buffer 
        }
        torch.save(checkpoint, path)
        # Optional: print(f"Checkpoint saved: {path} (Buffer size: {len(self.memory)})")

    def load_checkpoint(self, path):
        """Loads the checkpoint and returns the metadata for the simulation loop."""
        if not os.path.exists(path):
            return {'episode': 0, 'eval_rewards': []}
        
        ckpt = torch.load(path, map_location=self.device)
        
        # Restore Networks
        self.encoder.load_state_dict(ckpt['encoder_state_dict'])
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.q1.load_state_dict(ckpt['q1_state_dict'])
        self.q2.load_state_dict(ckpt['q2_state_dict'])
        
        # Restore Optimizers
        self.q_opt.load_state_dict(ckpt['q_opt_state_dict'])
        self.actor_opt.load_state_dict(ckpt['actor_opt_state_dict'])
        self.alpha_opt.load_state_dict(ckpt['alpha_opt_state_dict'])
        
        # Restore Scalars & Tensors
        self.log_alpha.data.copy_(ckpt['log_alpha'])
        self.total_steps = ckpt['total_steps']
        self.memory.buffer = ckpt['buffer']
        
        # Sync Target Networks
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        return ckpt # Return full dict so Simulation.py gets 'episode' and 'eval_rewards'