# =========================================================================================
# PPO Agent Implementation
# =========================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, MultivariateNormal
import gymnasium as gym
import Basic_Functions as bf

class RolloutBuffer:
    """Speicher für PPO (On-Policy Daten)"""
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.is_terminals, self.values = [], [], []

    def clear(self):
        del self.states[:]; del self.actions[:]; del self.logprobs[:]
        del self.rewards[:]; del self.is_terminals[:]; del self.values[:]

class ActorCritic(nn.Module):
    """Kombiniertes Netzwerk für PPO"""
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super(ActorCritic, self).__init__()
        # Wir nutzen denselben FeatureExtractor wie bei deinem SAC
        self.extractor = bf.FeatureExtractor(observation_space)
        
        # Action Space Typ erkennen
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        action_dim = action_space.n if self.is_discrete else action_space.shape[0]

        # Actor (Policy)
        self.actor_base = nn.Sequential(
            nn.Linear(self.extractor.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # Für kontinuierliche Aktionen brauchen wir eine Standardabweichung
        if not self.is_discrete:
            self.action_var = nn.Parameter(torch.full((action_dim,), 0.5))

        # Critic (Value Function)
        self.critic = nn.Sequential(
            nn.Linear(self.extractor.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def act(self, state):
        x = self.extractor(state)
        
        if self.is_discrete:
            action_logits = self.actor_head(self.actor_base(x))
            dist = Categorical(logits=action_logits)
        else:
            action_mean = torch.tanh(self.actor_head(self.actor_base(x)))
            dist = MultivariateNormal(action_mean, torch.diag(self.action_var))
            
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(x)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        x = self.extractor(state)
        
        if self.is_discrete:
            action_logits = self.actor_head(self.actor_base(x))
            dist = Categorical(logits=action_logits)
        else:
            action_mean = torch.tanh(self.actor_head(self.actor_base(x)))
            dist = MultivariateNormal(action_mean, torch.diag(self.action_var))
            
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(x)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    """Proximal Policy Optimization Agent"""
    def __init__(self, env, hidden_dim=256, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2, update_timestep=2000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.update_timestep = update_timestep
        self.time_step = 0
        
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Kopie für das Update
        self.policy_old = ActorCritic(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)
            
            # PPO muss diese Werte intern speichern
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.values.append(state_val)

            # Umwandlung für Gymnasium
            if self.policy_old.is_discrete:
                return action.item()
            else:
                return action.cpu().numpy()[0]

    def step(self, state, action, reward, next_state, done):
        """Wird aufgerufen, um das Feedback der Umgebung zu speichern"""
        self.time_step += 1
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

        # Führe Update durch, wenn genügend Daten gesammelt wurden
        if self.time_step % self.update_timestep == 0:
            self.update()

    def update(self):
        # 1. Monte Carlo Schätzung der Rewards (Returns)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 2. Buffer in Tensoren umwandeln
        # Use .squeeze(1) to safely only remove the inserted batch dimension, 
        # preserving the action dimension even if it is 1.
        old_states = torch.stack(self.buffer.states, dim=0).squeeze(1).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).squeeze(1).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).squeeze(1).detach()

        # 3. K_epochs lang optimieren
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # Ratio berechnen (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Finaler Loss
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Alte Policy updaten & Buffer leeren
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()