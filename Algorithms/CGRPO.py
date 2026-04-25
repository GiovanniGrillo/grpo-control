import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import utils as bf

# =========================================================================================
# CGRPO Agent Implementation
# =========================================================================================

class PopulationBuffer:
    """Buffer to hold trajectories for N different policies"""
    def __init__(self, N):
        self.N = N
        self.buffers = [[] for _ in range(N)]                                                       # every policy has its own buffer of trajectories (episodes)
        self.current_episodes = [self._init_empty_episode() for _ in range(N)]                      # current episode being collected for each policy (reset after each episode ends)

    def _init_empty_episode(self):
        return {"obs": [], "feature": [], "action": [], "log_probs": [], "reward": []}              # Initialize an empty episode structure to store trajectory data for each policy during collection phase
    
    def add(self, policy_idx, obs, feature, action, log_prob, reward):                              
        self.current_episodes[policy_idx]["obs"].append(obs)                                        # Store raw observation for this step
        self.current_episodes[policy_idx]["feature"].append(feature)                                # Store latent feature for this step (used for DBSCAN clustering)
        self.current_episodes[policy_idx]["action"].append(action)                                  # Store action taken at this step
        self.current_episodes[policy_idx]["log_probs"].append(log_prob)                             # Store log probability of the action (used for PPO loss)
        self.current_episodes[policy_idx]["reward"].append(reward)                                  # Store reward received at this step

    def finish_episode(self, policy_idx):
        if len(self.current_episodes[policy_idx]["reward"]) > 0:                                    # Only save the episode if it has at least one step (avoid saving empty episodes)
            self.buffers[policy_idx].append(self.current_episodes[policy_idx])                      # Add the completed episode to the policy's buffer
        self.current_episodes[policy_idx] = self._init_empty_episode()                              # Reset the current episode for this policy to start fresh for the next episode
    
    def clear_buffer(self):
        self.buffers = [[] for _ in range(self.N)]                                                  # Clear all buffers after an update cycle (we only keep the latest episode for each policy, so this is safe)

class Actor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super().__init__()
        self.extractor = bf.FeatureExtractor(observation_space)
        self.action_dim = action_space.shape[0]

        # Actor Network for Continuous Actions
        self.mean_head = nn.Linear(self.extractor.feature_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def forward_features(self, obs):
        return self.extractor(obs)

    def get_distribution(self, features):
        mean = torch.tanh(self.mean_head(features))
        std = torch.exp(self.log_std).clamp(min=1e-3, max=1.0)
        return torch.distributions.Normal(mean, std)

    def sample_action(self, obs: torch.Tensor):
        feat = self.forward_features(obs)
        dist = self.get_distribution(feat)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, feat
    
    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(obs)
        return torch.tanh(self.mean_head(feat))

class CGRPO:
    """Continuous Group Relative Policy Optimization"""
    def __init__(self, env, seed = 42, hidden_dim=256, lr=3e-4, N=10, K=2, epsilon=0.2, 
                 tau=0.5, lam_s=0.01, lam_d=0.01, gamma=0.99, 
                 dbscan_eps=0.4):#, dbscan_min_samples=10, dbscan_cluster_size=150):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.N = N 
        self.K = K 
        self.epsilon = epsilon
        self.tau = tau
        self.lam_s = lam_s 
        self.lam_d = lam_d 
        self.gamma = gamma
        self.dbscan_eps = dbscan_eps               
        # self.dbscan_min_samples = dbscan_min_samples 
        # self.dbscan_cluster_size = dbscan_cluster_size

        # Population of Actors
        self.actors = nn.ModuleList([Actor(env.observation_space, env.action_space, hidden_dim).to(self.device) for _ in range(N)])
        self.old_actors = nn.ModuleList([Actor(env.observation_space, env.action_space, hidden_dim).to(self.device) for _ in range(N)])
        
        # Reference Policy
        self.ref_actor = Actor(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.ref_actor.load_state_dict(self.actors[0].state_dict()) # Initiale Zuweisung

        for i in range(self.N):
            self.old_actors[i].load_state_dict(self.actors[i].state_dict())
            
        self.optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.buffer = PopulationBuffer(N)

        self.current_policy_idx = 0
        self._cached_obs, self._cached_action, self._cached_logprob, self._cached_feat = None, None, None, None

    def select_action(self, state, evaluate=False):
        obs_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                # In eval mode, we just use the reference (best) policy
                action_t = self.ref_actor.get_deterministic_action(obs_t)
            else:
                action_t, log_prob_t, feat_t = self.old_actors[self.current_policy_idx].sample_action(obs_t)
                
                self._cached_obs = obs_t.squeeze(0).cpu()
                self._cached_action = action_t.squeeze(0).cpu()
                self._cached_logprob = log_prob_t.squeeze(0).cpu()
                self._cached_feat = feat_t.squeeze(0).cpu() # Cache latent feature for DBSCAN

        return action_t.squeeze(0).cpu().numpy()

    def step(self, state, action, reward, next_state, done):
        self.buffer.add(self.current_policy_idx, self._cached_obs, self._cached_feat, self._cached_action, self._cached_logprob, reward)

        if done:
            self.buffer.finish_episode(self.current_policy_idx)
            self.current_policy_idx += 1

            # If all N policies have collected an episode, trigger update
            if self.current_policy_idx >= self.N:
                self.update()
                self.current_policy_idx = 0 # Reset for next round

    def _update_reference_policy(self):
        # Simply copies the first actor for now. In advanced versions, this blends the top K policies.
        self.ref_actor.load_state_dict(self.actors[0].state_dict())

    def _update_reference_policy_mixture(self, all_returns):
            num_top = max(1, self.N // 5)                               # Select top 20% of policies based on returns to form the reference policy mixture
            top_indices = np.argsort(all_returns)[-num_top:]             # Get the indices of the top performing policies
            
            tau = 0.1                                                   # Soft update parameter for blending the top policies into the reference policy
            for name, param in self.ref_actor.named_parameters():
                avg_param = torch.stack([self.actors[idx].state_dict()[name] for idx in top_indices]).mean(dim=0)  # Average the parameters of the top policies for this parameter
                param.data.copy_(tau * avg_param + (1 - tau) * param.data)                                          # Soft update the reference policy's parameters towards the average of the top policies (this creates a more robust reference policy that represents a strong strategy for others to learn from) 


    def update(self):        
        all_phi = []                                                                            # store feature vector for policy phi_i
        all_returns = []
        all_features_np = []
        
        # 1. Gather trajectory returns and features for clustering
        for i in range(self.N):
            traj = self.buffer.buffers[i][-1]                                                   # last trajectory of policy i
            obs = torch.stack(traj["obs"]).to(self.device)                                      # observations for this trajectory
            actions = torch.stack(traj["action"]).to(self.device)                               # actions taken in this trajectory

            with torch.no_grad():
                # Forward pass of current policy
                feat = self.actors[i].forward_features(obs)                                     # latent features for DBSCAN clustering
                dist = self.actors[i].get_distribution(feat)                                    # action distribution for PPO loss

                # reference policy forward pass for KL divergence
                ref_feat = self.ref_actor.forward_features(obs)                                 # latent features for reference policy
                ref_dist = self.ref_actor.get_distribution(ref_feat)                            # reference action distribution for KL divergence

                # Feature vector components for policy phi_i (used for clustering)
                # Average reward
                avg_reward = sum(traj["reward"]) / len(traj["reward"])
                # entropy of the action distribution (encourages exploration) normal distribution entropy formula: 0.5 * log(2 * pi * e * std^2)
                entropy = dist.entropy().mean().item()
                # Average action variance sigma_a
                avg_action_var = dist.variance.mean().item()
                # KL divergence to reference policy (measures how different this policy is from the reference)
                kl_div = torch.distributions.kl_divergence(dist, ref_dist).mean().item()


            # Construct the feature vector for this policy (used for clustering)
            phi_i = np.array([avg_reward, entropy, avg_action_var, kl_div])
            all_phi.append(phi_i)

            # Raw returns and Features for ranking and clustering
            ret = sum(traj["reward"])
            all_returns.append(ret)
            feat_stack = torch.stack(traj["feature"]).numpy()
            all_features_np.append(feat_stack)

        # Policy Clustering based on phi_i
        phi_array = np.array(all_phi)                                                           # convert list of feature vectors to numpy array for clustering

        phi_norm = (phi_array - phi_array.mean(axis=0)) / (phi_array.std(axis=0) + 1e-8)        # Normalize features for better clustering performance

        #K-means clustering
        kmeans = KMeans(n_clusters=min(self.K, self.N), n_init='auto', random_state=self.seed)  # Cluster policies into K groups based on their feature vectors
        self.policy_groups = kmeans.fit_predict(phi_norm)                                       # Assign each policy to a cluster group based on K-means results

        # State Clustering with DBSCAN
        all_states_flat = np.concatenate(all_features_np, axis=0)                               # Flatten all features across policies for DBSCAN clustering
        trajectory_lengths = [len(f) for f in all_features_np]                                  # Length of each trajectory (number of steps) for indexing
        start_indices = np.cumsum([0] + trajectory_lengths[:-1])                                # Starting index of each trajectory in the flattened feature array

        #### State clustering with DBSCAN
        scaler = StandardScaler()
        all_states_scaled = scaler.fit_transform(all_states_flat)

        # Reduce to 2D for visualization using PCA
        pca_clustering = PCA(n_components=min(10, all_states_flat.shape[1])) # Keep enough components to retain most variance for clustering
        all_states_pca = pca_clustering.fit_transform(all_states_scaled) # Apply PCA to the scaled features for better clustering performance and visualization (reducing dimensionality while retaining most of the

        # DBSCAN parameters
        total_points = all_states_pca.shape[0]
        dynamic_cluster_size = max(50, int(total_points * 0.01)) 
        dynamic_min_samples = max(10, int(dynamic_cluster_size / 5))

        #dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(all_states_flat) # Cluster states into contexts based on their latent features using DBSCAN
        # dbscan = HDBSCAN(min_cluster_size=self.dbscan_cluster_size, min_samples = self.dbscan_min_samples, cluster_selection_epsilon = 0.5,copy = True) # Cluster states into contexts based on their latent features using DBSCAN
        dbscan = HDBSCAN(min_cluster_size=dynamic_cluster_size, min_samples = dynamic_min_samples, cluster_selection_epsilon = self.dbscan_eps,copy = True) # Cluster states into contexts based on their latent features using DBSCAN

        #global_labels = dbscan.labels_                                                          # Cluster labels for each state in the flattened feature array
        global_labels = dbscan.fit_predict(all_states_pca) # Cluster states into contexts based on their PCA-reduced features using HDBSCAN (this often gives better clusters in high-dimensional spaces)

        # Visualize all_states_flat via PCA scatter plot colored by DBSCAN clusters
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(all_states_pca[:, 0], all_states_pca[:, 1], c=global_labels,
                            cmap='tab20', alpha=0.7, s=30, edgecolors='k', linewidth=0.5)
        plt.colorbar(scatter, label='Cluster Label (-1 = noise)')
        plt.xlabel(f'PC1 ({pca_clustering.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca_clustering.explained_variance_ratio_[1]:.2%})')
        plt.title(f'State Feature Space (PCA) - HDBSCAN Clusters')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('state_space_clustered.png', dpi=150)
        plt.close()

        n_clusters = len(set(global_labels)) - (1 if -1 in global_labels else 0)
        n_noise = list(global_labels).count(-1)
        print(f"HDBSCAN Clustering: {n_clusters} clusters, {n_noise} noise points")

        state_clusters_per_policy = []                                                          # List to hold cluster labels for each policy's trajectory
        for i in range(self.N):
            start = start_indices[i]
            end = start + trajectory_lengths[i]
            state_clusters_per_policy.append(global_labels[start:end])                            # Extract cluster labels for the states in this policy's trajectory and store them separately for advantage calculation


        # Advantage Computation
        all_returns_to_go = []                                                                        # store returns-to-go for each policy's trajectory
        for i in range(self.N):
            traj_rewards = self.buffer.buffers[i][-1]["reward"]                                       # rewards for this policy's trajectory
            returns_list = []
            discounted_reward = 0
            for reward in reversed(traj_rewards):
                discounted_reward = reward + self.gamma * discounted_reward
                returns_list.insert(0, discounted_reward)
            all_returns_to_go.append(torch.tensor(returns_list, dtype=torch.float32).to(self.device)) # store returns-to-go as a tensor for this policy's trajectory

        flat_returns = torch.cat(all_returns_to_go).to(self.device)                                   # Flatten returns-to-go across all policies for normalization
        global_labels_tensor = torch.from_numpy(global_labels).to(self.device)                        # Convert global cluster labels to tensor for indexing

        unique_clusters = torch.unique(global_labels_tensor)                                          # Get unique cluster labels (contexts) across all states
        cluster_means = {}

        for cluster in unique_clusters:
            mask = (global_labels_tensor == cluster)                                                  # Create a mask to select states belonging to this cluster
            if mask.any():
                cluster_means[cluster.item()] = flat_returns[mask].mean()                             # Calculate and store the mean return for this cluster (context) for advantage normalization

        all_advantages = []                                                                           # store advantages for each policy's trajectory
        for i in range(self.N):
            policy_returns = all_returns_to_go[i].to(self.device)                                     # returns-to-go for this policy's trajectory
            policy_labels = state_clusters_per_policy[i]                                              # cluster labels for the states in this policy's trajectory

            advantages = torch.zeros_like(policy_returns)                                             # Initialize advantages tensor for this policy's trajectory
            for t, label in enumerate(policy_labels):
                advantages[t] = policy_returns[t] - cluster_means[label.item()]                       # Calculate advantage for this time step by subtracting the cluster mean return from the return-to-go
            all_advantages.append(advantages)                                                         # Store the calculated advantages for this policy's trajectory                                                         


        # Group Normalization
        normalized_advantages = [None] * self.N                                                         # Initialize list to hold normalized advantages for each policy
        unique_groups = np.unique(self.policy_groups)                                                   # Get unique cluster groups for policies
        delta = 1e-8                                                                                    # Small constant to prevent division by zero during normalization

        for g in unique_groups:
            group_indices = np.where(self.policy_groups == g)[0]                                        # Get indices of policies belonging to this group
            group_advantages = torch.cat([all_advantages[i] for i in group_indices])                    # Concatenate advantages from all policies in this group for normalization
            mean_adv = group_advantages.mean()                                                          # Calculate mean advantage for this group
            std_adv = group_advantages.std() + delta                                                    # Calculate standard deviation of advantages for this group (add small delta to prevent division by zero)

            for i in group_indices:
                normalized_advantages[i] = (all_advantages[i] - mean_adv) / (std_adv + delta)           # Normalize advantages for each policy in this group using the group's mean and std

        # Policy Updates
        all_advantages_flat = torch.cat(all_advantages)
        sigma_global = all_advantages_flat.std() + 1e-8                                                 # Global standard deviation of advantages for scaling the loss

        all_mus = []                                                                                    # Store mean actions of all policies for diversity regularization (used to encourage different policies to explore different strategies by penalizing high similarity in their action distributions)
        if self.lam_d > 0:                                                                              # Only compute mean actions for diversity regularization if lambda_d is greater than 0 to save computation when diversity regularization is not used
            for j in range(self.N):                                                                     
                traj_j = self.buffer.buffers[j][-1]                                                     # last trajectory of policy j (used to compute mean action for diversity regularization)
                obs_j = torch.stack(traj_j["obs"]).to(self.device)                                      # observations for this trajectory of policy j (used to compute mean action for diversity regularization)
                with torch.no_grad():
                    dist_j = self.actors[j].get_distribution(self.actors[j].forward_features(obs_j))    # Get action distribution for this trajectory of policy j to compute mean action for diversity regularization
                    all_mus.append(dist_j.mean)                                                         # Store mean action of this policy j for diversity regularization

        for i in range(self.N):
            traj = self.buffer.buffers[i][-1]                                                           # last trajectory of policy i
            obs = torch.stack(traj["obs"]).to(self.device)                                              # observations for this trajectory
            actions = torch.stack(traj["action"]).to(self.device)                                       # actions taken in this trajectory
            old_log_probs = torch.stack(traj["log_probs"]).to(self.device)                              # log probabilities of actions taken (from the old policy)
            advantages = normalized_advantages[i].to(self.device)                                       # normalized advantages for this policy's trajectory (used for loss)

            group_idx = self.policy_groups[i]                                                                # Get the cluster group index for this policy
            group_advantages = torch.cat([all_advantages[j] for j in np.where(self.policy_groups == group_idx)[0]]) # Get advantages for all policies in the same group for potential use in regularization
            sigma_k = group_advantages.std() + 1e-8                                                                # Standard deviation of advantages for this group for scaling the loss

            epsilon_i = self.epsilon * torch.clamp(sigma_k / sigma_global, min=1.0)                     # Adaptive clipping parameter based on group advantage variance

            # Surrogate Loss
            actor = self.actors[i]                                                                      # Get the current actor for this policy index
            feat = actor.forward_features(obs)                                                          # Forward pass to get latent features for this trajectory's observations
            dist = actor.get_distribution(feat)                                                         # Get the action distribution for this trajectory based on the current policy
            new_log_probs = dist.log_prob(actions).sum(dim=-1)                                          # Calculate new log probabilities of the actions taken in this trajectory under the current policy

            ratio = torch.exp(new_log_probs - old_log_probs)                                            # Calculate the probability ratio for loss
            surr1 = ratio * advantages                                                                  # Calculate the unclipped surrogate loss 
            surr2 = torch.clamp(ratio, 1 - epsilon_i, 1 + epsilon_i) * advantages                       # Calculate the clipped surrogate loss using the adaptive epsilon_i for this policy's group
            actor_loss = -torch.min(surr1, surr2).mean()                                                # Final actor loss is the negative of the minimum of the two surrogate losses

            # L smooth Regularization
            l_smooth = torch.tensor(0.0).to(self.device)                                                # Initialize temporal smoothness regularization term
            if len(dist.mean) > 1:                                                                      # Only calculate temporal smoothness if there are multiple time steps in the trajectory
                l_smooth = torch.mean((dist.mean[1:] - dist.mean[:-1]) ** 2)                            # Temporal smoothness regularization to encourage similar actions in consecutive time steps
            
            # Diversity Regularization (L_diversity)
            l_diversity = torch.tensor(0.0).to(self.device)                                             # Initialize diversity regularization term
            if self.lam_d > 0:                                                                          # Only calculate diversity regularization if lambda_d is greater than 0
                mu_i = dist.mean                                                                        # Mean action of the current policy i (used for diversity regularization)
                for j in range(self.N):
                    if i == j:
                        continue
                    mu_j = all_mus[j]                                                                   # Mean action of policy j
                    cos_sim = torch.cosine_similarity(mu_i, mu_j, dim=-1).mean()                        # Calculate cosine similarity between the mean actions of policy i and policy j
                    l_diversity += torch.max(torch.tensor(0.0).to(self.device), cos_sim - self.tau)     # Diversity regularization to encourage different policies to explore different strategies (penalize high similarity above a threshold tau)

            # Total Loss Zusammenführung
            loss = actor_loss + self.lam_s * l_smooth + self.lam_d * l_diversity   # Combine all loss components into the final loss for this policy update

            # Optimization
            self.optimizers[i].zero_grad()
            loss.backward()                                                                             # Backpropagate the loss to compute gradients
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)                            # Clip gradients to prevent exploding gradients (max norm of 0.5)
            self.optimizers[i].step()                                                                   # Update the actor's parameters using the optimizer

        self._update_reference_policy_mixture(all_returns)                        # Update the reference policy to be a mixture of the top K performing policies based on their returns (this encourages the reference policy to represent a strong strategy that other policies can learn from)

        
        # Cleanup & Sync
        for i in range(self.N):
            self.old_actors[i].load_state_dict(self.actors[i].state_dict())
        self.buffer.clear_buffer()