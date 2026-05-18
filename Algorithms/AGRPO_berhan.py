import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
import utils as bf
import collections

class AGRPO:
    """
    Advanced Group Relative Policy Optimization (AGRPO).
    100% Pure GRPO formulation for Continuous Control.
    """
    def __init__(self, env, seed=42, hidden_dim=256, lr=3e-4, N=10, K=2, epsilon=0.2,
                 tau=0.5, lam_s=0.0, lam_d=0.0, lam_t=0.05, gamma=0.99, dbscan_eps=0.4,
                 warmup_episodes=100, initial_threshold=-1.0, plot_interval=10):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.N = N
        self.K = K
        self.epsilon = epsilon
        self.tau = tau
        self.lam_s = lam_s      
        self.lam_d = lam_d      
        self.lam_t = lam_t      
        self.gamma = gamma
        self.dbscan_eps = dbscan_eps
        self.plot_interval = plot_interval  

        self.running_reward_std = 1.0 
        self.warmup_episodes = warmup_episodes
        self.std_history = []
        self.target_min_std_history = []
        self.target_max_std_history = []
        
        self.return_mean_history = collections.deque(maxlen=20)
        self.return_std_history = collections.deque(maxlen=20)

        self.dynamic_threshold = initial_threshold
        self.trauma_centers = []
        self.current_track_data = None

        self.obs_space = env.observation_space
        self.action_space = env.action_space
        self.hidden_dim = hidden_dim
        self.base_lr = lr

        self.actors = nn.ModuleList([bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device) for _ in range(N)])
        self.old_actors = nn.ModuleList([bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device) for _ in range(N)])
        self.ref_actor = bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.ref_actor.load_state_dict(self.actors[0].state_dict())

        for i in range(self.N):
            self.old_actors[i].load_state_dict(self.actors[i].state_dict())

        self.optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.buffer = bf.PopulationBuffer(N)
        self.current_policy_idx = 0

        self.scaler = StandardScaler()
        self.pca = None

    def select_action(self, state, evaluate=False):
        obs_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                action_t = self.ref_actor.get_deterministic_action(obs_t)
            else:
                action_t, log_prob_t, feat_t = self.old_actors[self.current_policy_idx].sample_action(obs_t)
                self._cached_obs = obs_t.squeeze(0).cpu()
                self._cached_action = action_t.squeeze(0).cpu()
                self._cached_logprob = log_prob_t.squeeze(0).cpu()
                self._cached_feat = feat_t.squeeze(0).cpu()
        return action_t.squeeze(0).cpu().numpy()

    def step(self, state, action, reward, next_state, done, pos=(0.0, 0.0)):
        self.buffer.add(self.current_policy_idx, self._cached_obs, self._cached_feat, 
                        self._cached_action, self._cached_logprob, reward, pos=pos)
        if done:
            self.buffer.finish_episode(self.current_policy_idx)
            self.current_policy_idx += 1
            if self.current_policy_idx >= len(self.actors):
                stats = self.update()
                self.current_policy_idx = 0
                return stats
        return None

    def _update_reference_policy_mixture(self, all_returns):
        num_top = max(1, int(len(self.actors) // 10))
        top_indices = np.argsort(all_returns)[-num_top:]
        self.last_elite_rewards = [all_returns[i] for i in top_indices]
        self.last_elite_indices = top_indices

        best_idx = top_indices[-1]
        best_return = all_returns[best_idx]

        if not hasattr(self, 'historical_best_return'):
            self.historical_best_return = -float('inf')
            self.stagnant_updates = 0

        # Only update the reference policy if we have a new historical best return, or if we've been stagnant for too long without surpassing a reasonable threshold.
        if best_return >= self.historical_best_return:
            self.ref_actor.load_state_dict(self.old_actors[best_idx].state_dict())
            self.historical_best_return = best_return
            self.stagnant_updates = 0
        else:
            self.stagnant_updates += 1
            if self.stagnant_updates > 5 and self.historical_best_return < 250.0:
                self.ref_actor.load_state_dict(self.old_actors[best_idx].state_dict())
                self.historical_best_return = best_return
                self.stagnant_updates = 0
                
        self.historical_best_return *= 0.995 # Slight decay to encourage continual improvement
        
        return self.last_elite_rewards

    def _gather_metrics(self):
        all_phi, all_features_np, all_pos_np = [], [], [] 
        for i in range(len(self.actors)):
            traj = self.buffer.get_latest_trajectory(i)
            obs = torch.stack(traj["obs"]).to(self.device)

            with torch.no_grad():
                feat = self.actors[i].forward_features(obs)
                dist = self.actors[i].get_distribution(feat)
                # Re-evaluate observations through the stable Reference Policy
                ref_feat = self.ref_actor.forward_features(obs)
                ref_dist = self.ref_actor.get_distribution(ref_feat)
                reward_sum = sum(traj["reward"])
                
                phi_i = np.array([
                    reward_sum / len(traj["reward"]),
                    dist.entropy().mean().item(),
                    dist.variance.mean().item(),
                    torch.distributions.kl_divergence(dist, ref_dist).mean().item()
                ])

            all_phi.append(phi_i)
            # Use the stable ref_feat for clustering
            all_features_np.append(ref_feat.cpu().numpy())
            all_pos_np.append(np.array(traj["pos"]))
            
        return np.array(all_phi), all_features_np, all_pos_np

    def _cluster_states(self, all_features_np, all_pos_np):
        flat_features = np.concatenate(all_features_np, axis=0)
        scaled_features = self.scaler.fit_transform(flat_features)

        self.pca = PCA(n_components=0.95) # Keep 95% variance
        
        # Subsample to speed up HDBSCAN (take max 5000 points)
        step = max(1, len(flat_features) // 5000)
        subsample_idx = np.arange(0, len(flat_features), step)
        scaled_sub = scaled_features[subsample_idx]
        
        pca_features_sub = self.pca.fit_transform(scaled_sub)
        pca_features_sub = np.nan_to_num(pca_features_sub, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        pca_features_sub = normalize(pca_features_sub, norm='l2')

        # use euclidean distance with l2 normalization to get cosine similarity clustering without the instability of actual cosine distance in HDBSCAN
        c_size = 25 # fixed minimum cluster size
        dbscan = HDBSCAN(min_cluster_size=c_size, 
                         min_samples=max(5, c_size // 2), 
                         metric='euclidean',
                         cluster_selection_epsilon=self.dbscan_eps, 
                         core_dist_n_jobs=1)
        hdbscan_labels = dbscan.fit_predict(pca_features_sub)

        # If HDBSCAN fails to find meaningful clusters (e.g., too much noise), fall back to KMeans for stability.
        noise_ratio = (hdbscan_labels == -1).mean()
        if noise_ratio > 0.8 or len(set(hdbscan_labels) - {-1}) <= 1:
            kmeans = KMeans(n_clusters=min(10, len(pca_features_sub)), n_init='auto', random_state=self.seed)
            hdbscan_labels = kmeans.fit_predict(pca_features_sub)

        # Prepare full dataset for label propagation with l2 and same PCA transformation
        full_pca = self.pca.transform(scaled_features)
        full_pca = np.nan_to_num(full_pca, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        full_pca = normalize(full_pca, norm='l2')

        # Propagate cluster labels to the full dataset using KNN, but only if we have some valid clusters from HDBSCAN
        valid_mask = hdbscan_labels != -1
        if np.any(valid_mask):
            valid_features = pca_features_sub[valid_mask]
            valid_labels = hdbscan_labels[valid_mask]

            knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
            knn.fit(valid_features, valid_labels)

            distances, _ = knn.kneighbors(full_pca, n_neighbors=1)
            train_distances, _ = knn.kneighbors(valid_features, n_neighbors=1)
            thresh = np.percentile(train_distances, 95) if len(train_distances) > 0 else 1.0

            predicted_labels = knn.predict(full_pca)
            labels = np.where(distances.flatten() < thresh, predicted_labels, -1)
        else:
            labels = np.zeros(len(flat_features), dtype=int)
        
        traj_lengths = [len(f) for f in all_features_np]
        
        # LOGGING: Capture cluster stats
        self.last_num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.last_noise_ratio = (labels == -1).sum() / len(labels) if len(labels) > 0 else 0.0

        ep = getattr(self, 'current_episode', 0)
        if (ep + 1) % self.plot_interval == 0:
            self._plot_clusters(full_pca, labels)
            self._plot_spatial_only(np.concatenate(all_pos_np, axis=0), labels)

        return labels, traj_lengths

    def _compute_advantages(self, labels, traj_lengths, all_features_np, all_pos_np,
                        dynamic_threshold, reward_scale, returns):
        N_agents = len(self.actors)
        all_returns_to_go = []

        # Calculate the discounted sum of future rewards (Returns-To-Go) for each trajectory.
        for i in range(N_agents):
            rewards = self.buffer.get_latest_trajectory(i)["reward"]
            rtg = bf.compute_returns_to_go(rewards, self.gamma, self.device)
            all_returns_to_go.append(rtg)


        # To avoid late-episode states being unfairly penalized for having lower RTGs than early states
        # calculate an average baseline across the population at each exact timestep (T).
        max_len = max(traj_lengths)
        
        # Create padded matrices to handle trajectories of varying lengths cleanly
        rtg_matrix = torch.zeros((N_agents, max_len), device=self.device)
        mask = torch.zeros((N_agents, max_len), device=self.device)
        
        # Populate the matrices and masks
        for i in range(N_agents):
            length = traj_lengths[i]
            rtg_matrix[i, :length] = all_returns_to_go[i]
            mask[i, :length] = 1.0
            
        # Calculate the mean RTG at each timestep 't' across all N agents.
        baseline_t = rtg_matrix.sum(dim=0) / mask.sum(dim=0).clamp(min=1.0)
        
        # Calculate advantages
        adv_matrix = rtg_matrix - baseline_t.unsqueeze(0)
        adv_matrix = adv_matrix * mask
        
        # Flatten the valid (unmasked) advantages back into a continuous list
        flat_advantages_list = []
        for i in range(N_agents):
            length = traj_lengths[i]
            flat_advantages_list.append(adv_matrix[i, :length])
            
        flat_advantages = torch.cat(flat_advantages_list)

        # Normalize advantages across the entire batch
        adv_mean = flat_advantages.mean()
        adv_std = flat_advantages.std() + 1e-8
        
        normalized_flat_adv = (flat_advantages - adv_mean) / adv_std
        
        # Split the flattened, normalized advantages back to their respective agents
        advantages = list(torch.split(normalized_flat_adv, traj_lengths))

        dim = self.actors[0].extractor.feature_dim
        max_weight = 100.0 * reward_scale 
        self.dynamic_threshold = dynamic_threshold

        # Track catastrophic failures to penalize the latent states that lead to them.
        self.trauma_new_count = 0
        self.trauma_reinforced_count = 0

        best_record = getattr(self, 'historical_best_return', 0.0)

        # If the actor can already balance reliably (>300 points), disable trauma tracking.
        if best_record < 300.0:
            for i in range(len(self.actors)):
                traj_return = returns[i]
                
                # If an agent scores below the dynamic failure threshold...
                if traj_return < self.dynamic_threshold:
                    rewards = np.array(self.buffer.get_latest_trajectory(i)["reward"])
                    positive_indices = np.where(rewards > 0)[0]

                    # Isolate the exact window of states where the failure began.
                    # We grab states right around the last time the agent received a positive reward.
                    if len(positive_indices) > 0:
                        last_good_step = positive_indices[-1]
                        start_idx = max(0, last_good_step - 5)
                        end_idx = min(len(rewards), last_good_step + 15)
                    else:
                        # If no positive rewards were found, grab the start of the episode
                        start_idx = 0
                        end_idx = min(len(rewards), 20)

                    # Extract the latent features and spatial coordinates for this "failure window"
                    trauma_points = all_features_np[i][start_idx:end_idx]
                    trauma_pos = all_pos_np[i][start_idx:end_idx]

                    if len(trauma_points) > 2:
                        # Compute the Gaussian centroid (mean/std) of this failure state
                        mu_feat = torch.tensor(trauma_points.mean(axis=0), dtype=torch.float32).to(self.device)
                        sigma_feat = torch.tensor(trauma_points.std(axis=0), dtype=torch.float32).to(self.device).clamp(min=0.1)
                        severity = abs(traj_return - self.dynamic_threshold)
                        
                        # Compute spatial position bounds for visualization/logging
                        mu_pos = trauma_pos.mean(axis=0)
                        sigma_pos_x = max(float(trauma_pos[:, 0].std()), 0.1)
                        sigma_pos_y = max(float(trauma_pos[:, 1].std()), 0.1)

                        new_trauma_data = {
                            'mu': mu_feat, 'sigma': sigma_feat, 'mu_pos': mu_pos,          
                            'sigma_pos_x': sigma_pos_x, 'sigma_pos_y': sigma_pos_y,   
                            'weight': min(severity, max_weight)
                        }

                        merged = False
                        
                        # Distance Check: Compare this new failure to existing Trauma Centers
                        if self.trauma_centers:
                            trauma_mus = torch.stack([t['mu'] for t in self.trauma_centers])
                            trauma_sigmas = torch.stack([t['sigma'] for t in self.trauma_centers])
                            
                            # Calculate normalized squared distance (Mahalanobis-like) to all centers
                            dist_sq = torch.sum(((mu_feat.unsqueeze(0) - trauma_mus) / trauma_sigmas) ** 2, dim=-1) / dim
                            min_idx = torch.argmin(dist_sq).item()

                            # If it's very close (< 1.0) to an existing center, merge them
                            # instead of creating a duplicate cluster.
                            if dist_sq[min_idx] < 1.0: 
                                existing = self.trauma_centers[min_idx]
                                # Reinforce the weight (severity) of this specific roadblock
                                existing['weight'] += min(existing['weight'] + severity, max_weight)
                                # Update running averages for features and spatial positions
                                existing['mu'] = (existing['mu'] + mu_feat) / 2
                                existing['mu_pos'] = (existing['mu_pos'] + mu_pos) / 2
                                existing['sigma'] = (existing['sigma'] + sigma_feat) / 2
                                existing['sigma_pos_x'] = (existing['sigma_pos_x'] + sigma_pos_x) / 2
                                existing['sigma_pos_y'] = (existing['sigma_pos_y'] + sigma_pos_y) / 2
                                merged = True

                        # If it's a unique failure state, create a brand new center
                        if not merged:
                            self.trauma_centers.append(new_trauma_data)
                            self.trauma_new_count += 1
                        else:
                            self.trauma_reinforced_count += 1

                        action_str = "reinforced" if merged else "Newly created"
                        print(f"   [Trauma] Agent {i} | Return: {traj_return:.1f} (Limit: {self.dynamic_threshold:.1f}) | Severity: {severity:.1f} -> {action_str}")

                        # Prevent memory leaks by only keeping the 200 most recent centers
                        if len(self.trauma_centers) > 200:
                            self.trauma_centers.pop(0)

        return advantages
    
    def _compute_trauma_penalty(self, feat):
        if not self.trauma_centers:
            return torch.tensor(0.0).to(self.device)

        total_penalty = torch.tensor(0.0).to(self.device)
        bandwidth = 20.0 
        dim = self.actors[0].extractor.feature_dim

        for center in self.trauma_centers:
            mu, sigma, weight = center['mu'], center['sigma'], center['weight']
            dist_sq = torch.sum(((feat - mu) / sigma) ** 2, dim=-1)
            normalized_dist = dist_sq / dim
            gauss_penalty = torch.exp(-normalized_dist / (2 * (bandwidth ** 2)))
            total_penalty += (gauss_penalty * weight).mean()

        return total_penalty
    
    def _compute_exp_diversity(self, i, all_mus, current_lam_d):
        if current_lam_d <= 0:
            return torch.tensor(0.0).to(self.device)
        exp_scale = 5.0
        min_len = min(mu.shape[0] for mu in all_mus.values())
        if min_len == 0:
            return torch.tensor(0.0).to(self.device)
        mu_i_trunc = all_mus[i][:min_len]
        other_indices = [j for j in range(len(self.actors)) if j != i]
        if not other_indices:
            return torch.tensor(0.0).to(self.device)
        other_mus_trunc = torch.stack([all_mus[j][:min_len] for j in other_indices])

        sims = torch.nn.functional.cosine_similarity(
            mu_i_trunc.unsqueeze(0),
            other_mus_trunc,
            dim=-1
        ).mean(dim=1)
        penalties = torch.clamp(torch.exp(exp_scale * (sims - self.tau)) - 1.0, min=0.0)
        return penalties.mean()

    def _plot_exploration_health(self):
        import os
        os.makedirs('plots', exist_ok=True)
        plt.figure(figsize=(10, 6))
        x = np.arange(len(self.std_history))
        plt.plot(x, self.std_history, label='Actual Policy STD (Avg)', color='#1f77b4', linewidth=2)
        plt.plot(x, self.target_min_std_history, label='Min Std Floor', color='#d62728', linestyle='--')
        if len(self.target_max_std_history) == len(self.std_history):
            plt.plot(x, self.target_max_std_history, label='Max Std Ceiling', color='#2ca02c', linestyle='--')
            plt.fill_between(x, self.target_min_std_history, self.target_max_std_history, color='gray', alpha=0.1)
        plt.title(f"Exploration Health Analysis (Ep {getattr(self, 'current_episode', 0)})")
        plt.xlabel("Update Steps")
        plt.ylabel("Standard Deviation")
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.savefig('plots/exploration_health.png')
        plt.close()

    def update(self):
        ep = getattr(self, 'current_episode', 0)
        best_record = getattr(self, 'historical_best_return', 0.0)

        # Scale down learning rate as mastery increases to ensure stability at the top.
        if best_record > 400:
            lr_factor = 0.2 # stability at the top
        elif best_record > 200:
            lr_factor = 0.5  # slow down as we approach mastery
        else:
            lr_factor = 1.0  # full learning rate during early exploration and swing-up phase
        
        current_lr = self.base_lr * lr_factor
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = current_lr
        
        # adaptive std scheduling based on performance to encourage appropriate exploration throughout training
        if best_record > 250.0:
            std_final_floor = 0.01
            decay_lambda = 0.01 
        elif best_record > 150.0:
            std_final_floor = 0.05
            decay_lambda = 0.005 
        else:
            std_final_floor = 0.25 
            decay_lambda = 0.002

        std_start, std_warmup_end = 0.8, 0.6
        
        if ep < self.warmup_episodes:
            min_std = max(std_warmup_end, std_start - (ep/self.warmup_episodes) * (std_start - std_warmup_end)) 
        else:
            time_passed = ep - self.warmup_episodes
            min_std = std_final_floor + (std_warmup_end - std_final_floor) * np.exp(-decay_lambda * time_passed)
        
        max_warmup = 4 * self.warmup_episodes
        std_max_floor = std_final_floor + 0.45

        if ep < max_warmup:
            max_std = std_start 
        else:
            time_passed_max = ep - max_warmup
            max_std = std_max_floor + (std_start - std_max_floor) * np.exp(-decay_lambda * time_passed_max)
            
        max_std = max(max_std, min_std + 1e-3)
        current_lam_d = self.lam_d * (min_std / 0.5) if ep >= self.warmup_episodes else 0.0

        all_rtg_lists = []
        returns_for_ranking = [sum(self.buffer.get_latest_trajectory(idx)["reward"]) for idx in range(self.N)]

        sorted_indices = np.argsort(returns_for_ranking)
        n_scouts = max(1, int(self.N * 0.10))
        n_elite = max(1, int(self.N * 0.10))

        n_reset = max(1, int(self.N * 0.10)) # 10% worst performers get reset with genetic jitter to maintain diversity and prevent stagnation

        elite_idx = sorted_indices[-n_elite:]
        scout_idx = sorted_indices[:n_scouts]
        reset_idx = sorted_indices[n_scouts : n_scouts + n_reset]
        mid_idx = sorted_indices[n_scouts + n_reset : -n_elite]
        
        with torch.no_grad():
            sum_actual_std = 0
            for i in range(len(self.actors)):
                self.old_actors[i].load_state_dict(self.actors[i].state_dict())
                if i in scout_idx:
                    self.actors[i].log_std.clamp_(min=np.log(0.6), max=np.log(0.8))
                elif i in elite_idx:
                    self.actors[i].log_std.clamp_(min=np.log(1e-3), max=np.log(max_std))
                else:
                    self.actors[i].log_std.clamp_(min=np.log(min_std), max=np.log(max_std))
                sum_actual_std += torch.exp(self.actors[i].log_std).mean().item()
                
                traj_rewards = self.buffer.get_latest_trajectory(i)["reward"]
                all_rtg_lists.append(bf.compute_returns_to_go(traj_rewards, self.gamma, self.device))

            self.std_history.append(sum_actual_std / self.N)
            self.target_min_std_history.append(min_std)
            self.target_max_std_history.append(max_std)

        phi, features_np, all_pos_np = self._gather_metrics()
        labels, traj_lengths = self._cluster_states(features_np, all_pos_np)

        current_returns = np.array(returns_for_ranking)
        self.return_mean_history.append(np.mean(current_returns))
        self.return_std_history.append(np.std(current_returns))

        if ep < self.warmup_episodes:
            dynamic_threshold = -float('inf') 
        else:
            roll_mean = np.mean(self.return_mean_history)
            roll_std = np.mean(self.return_std_history)
            dynamic_threshold = roll_mean - (2.0 * roll_std)

        current_std = np.std(current_returns)
        if current_std > 1e-4:
            self.running_reward_std = 0.9 * self.running_reward_std + 0.1 * current_std
        reward_scale = self.running_reward_std 

        for center in self.trauma_centers:
            center['weight'] *= 0.99
        forget_limit = reward_scale / 10.0
        self.trauma_centers = [c for c in self.trauma_centers if c['weight'] > forget_limit]

        advantages = self._compute_advantages(labels, traj_lengths, features_np, all_pos_np, 
                                              dynamic_threshold, reward_scale, returns_for_ranking)

        actor_features, all_mus = {}, {}
        for j in range(len(self.actors)):
            obs = torch.stack(self.buffer.get_latest_trajectory(j)["obs"]).to(self.device)
            feat = self.actors[j].forward_features(obs)
            actor_features[j], all_mus[j] = feat, self.actors[j].get_distribution(feat).mean.detach()

        loss_stats = {"actor_loss": 0.0, "smooth_loss": 0.0, "div_loss": 0.0, "trauma_loss": 0.0}
        updated_agents_count = 0

        for i in range(len(self.actors)):
            if ep >= self.warmup_episodes and i in reset_idx: continue
            updated_agents_count += 1

            if i in elite_idx:
                role_lam_d, role_lam_t = 0.0, 0.0 
            elif i in scout_idx:
                role_lam_d, role_lam_t = current_lam_d * 30.0, self.lam_t 
            else:
                role_lam_d, role_lam_t = current_lam_d, self.lam_t 

            traj = self.buffer.get_latest_trajectory(i)
            obs = torch.stack(traj["obs"]).to(self.device)
            actions = torch.stack(traj["action"]).to(self.device)
            old_log_probs = torch.stack(traj["log_probs"]).to(self.device).detach()
            adv = advantages[i].to(self.device).detach()

            # ppo style multiple epochs with early stopping based on KL divergence to ensure stable updates
            # important when using the trauma penalty which can create sharp gradients
            K_epochs = 10
            for epoch in range(K_epochs):
                feat = self.actors[i].forward_features(obs)
                dist = self.actors[i].get_distribution(feat)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)

                log_ratio = new_log_probs - old_log_probs
                approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item()
                
                if approx_kl > 0.025:
                    break

                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                l_smooth = torch.mean((feat[1:] - feat[:-1]) ** 2) if feat.shape[0] > 1 else torch.tensor(0.0).to(self.device)
                l_div = self._compute_exp_diversity(i, all_mus, role_lam_d) / reward_scale
                l_trauma = role_lam_t * self._compute_trauma_penalty(feat) / reward_scale

                total_loss = actor_loss + (self.lam_s * l_smooth) + l_div + l_trauma

                self.optimizers[i].zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                self.optimizers[i].step()

                if epoch == K_epochs - 1 or approx_kl > 0.025:
                    loss_stats["actor_loss"] += actor_loss.item()
                    loss_stats["smooth_loss"] += l_smooth.item() if torch.is_tensor(l_smooth) else l_smooth
                    loss_stats["div_loss"] += l_div.item() if torch.is_tensor(l_div) else l_div
                    loss_stats["trauma_loss"] += l_trauma.item() if torch.is_tensor(l_trauma) else l_trauma

        if ep >= self.warmup_episodes:
            survivor_actors, survivor_old_actors, survivor_optimizers = [], [], []
            for idx in range(len(self.actors)):
                if idx not in reset_idx:
                    survivor_actors.append(self.actors[idx]); survivor_old_actors.append(self.old_actors[idx])
                    survivor_optimizers.append(self.optimizers[idx])
            
            for _ in range(len(reset_idx)):
                new_actor = bf.ContinuousActor(self.obs_space, self.action_space, self.hidden_dim).to(self.device)
                new_old_actor = bf.ContinuousActor(self.obs_space, self.action_space, self.hidden_dim).to(self.device)
                
                # dynamic mutation scale based on performance to maintain diversity without destabilizing training as we approach mastery
                mutation_scale = 0.01 if best_record < 400.0 else 0.002

                new_dict = {}
                for k, v in self.ref_actor.state_dict().items():
                    new_dict[k] = v + torch.randn_like(v) * mutation_scale
                    
                new_actor.load_state_dict(new_dict)
                new_old_actor.load_state_dict(new_dict)
                
                survivor_actors.append(new_actor); survivor_old_actors.append(new_old_actor)
                survivor_optimizers.append(optim.Adam(new_actor.parameters(), lr=current_lr))
                
            self.actors, self.old_actors, self.optimizers = nn.ModuleList(survivor_actors), nn.ModuleList(survivor_old_actors), survivor_optimizers
            self.N = len(self.actors)

        elite_stats = self._update_reference_policy_mixture(returns_for_ranking)
        self.buffer = bf.PopulationBuffer(self.N) 

        if (ep + 1) % self.plot_interval == 0:
            self._plot_exploration_health()

        for key in loss_stats: loss_stats[key] /= max(1, updated_agents_count)
        print(f"[Update] Actor: {loss_stats['actor_loss']:.4f}, Smooth: {loss_stats['smooth_loss']:.4f}, Div: {loss_stats['div_loss']:.4f}, Trauma: {loss_stats['trauma_loss']:.4f}, Elite Return: {np.mean(elite_stats):.2f}")
        
        actual_stds = [torch.exp(self.actors[idx].log_std).mean().item() for idx in range(self.N)]
        def safe_mean(idx_list): return np.mean(current_returns[idx_list]) if len(idx_list) > 0 else 0.0
        def safe_std(idx_list): return np.std(current_returns[idx_list]) if len(idx_list) > 0 else 0.0
        def safe_mean_std_action(idx_list): return np.mean([actual_stds[i] for i in idx_list]) if len(idx_list) > 0 else 0.0

        stats_dict = {
            "loss_actor": loss_stats["actor_loss"],
            "loss_smooth": loss_stats["smooth_loss"],
            "loss_div": loss_stats["div_loss"],
            "loss_trauma": loss_stats["trauma_loss"],
            "trauma_threshold": getattr(self, 'dynamic_threshold', 0.0),
            "trauma_reward_scale": reward_scale,
            "trauma_centers_count": len(self.trauma_centers),
            "trauma_new": getattr(self, 'trauma_new_count', 0),
            "trauma_reinforced": getattr(self, 'trauma_reinforced_count', 0),
            "cluster_count": getattr(self, 'last_num_clusters', 0),
            "cluster_noise_ratio": getattr(self, 'last_noise_ratio', 0.0),
            "tier_elite_return_avg": safe_mean(elite_idx),
            "tier_elite_return_std": safe_std(elite_idx),
            "tier_mid_return_avg": safe_mean(mid_idx),
            "tier_mid_return_std": safe_std(mid_idx),
            "tier_scout_return_avg": safe_mean(scout_idx),
            "tier_scout_return_std": safe_std(scout_idx),
            "tier_elite_action_std": safe_mean_std_action(elite_idx),
            "tier_mid_action_std": safe_mean_std_action(mid_idx),
            "tier_scout_action_std": safe_mean_std_action(scout_idx),
            "target_min_std": min_std,       
            "target_max_std": max_std,       
            "elite_mean": np.mean(elite_stats),
            "current_lr": current_lr
        }
        return stats_dict

    def _plot_spatial_only(self, pos_data, labels):
        os.makedirs('plots', exist_ok=True)
        plt.figure(figsize=(10, 10))
        if hasattr(self, 'current_track_data') and self.current_track_data is not None:
            track_x = [p[0] for p in self.current_track_data]
            track_y = [p[1] for p in self.current_track_data]
            track_x.append(track_x[0]); track_y.append(track_y[0])
            plt.plot(track_x, track_y, color='darkgray', linewidth=35, alpha=0.5, label='Road')
            plt.plot(track_x, track_y, color='white', linewidth=2, linestyle='--', alpha=0.8)
        
        plt.scatter(pos_data[:, 0], pos_data[:, 1], c=labels, cmap='tab20', s=8, alpha=0.8, zorder=6)
        ep = getattr(self, 'current_episode', 0)
        plt.title(f"Spatial Cluster Distribution (Ep {ep})")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True, alpha=0.3)
        plt.axis('equal') 
        plt.savefig(f'plots/carracing_spatial_ep{ep}.png')
        plt.close()

    def _plot_clusters(self, data, labels):
        plt.figure(figsize=(10, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', s=10)
        plt.title("Latent State Clusters (PCA)")
        plt.savefig('state_space_clustered.png')
        plt.close()

    def save_checkpoint(self, path, ep, eval_rewards, seed_logs=None):
        checkpoint = {
            'episode': ep,
            'eval_rewards': eval_rewards,
            'seed_logs': seed_logs if seed_logs is not None else [],
            'current_policy_idx': self.current_policy_idx,
            'actors_state_dict': self.actors.state_dict(),
            'old_actors_state_dict': self.old_actors.state_dict(),
            'ref_actor_state_dict': self.ref_actor.state_dict(),
            'optimizers_state_dict': [opt.state_dict() for opt in self.optimizers],
            'buffer_data': self.buffer.buffers,
            'current_episodes': getattr(self.buffer, 'current_episodes', None),
            'trauma_centers': self.trauma_centers,
            'scaler': self.scaler,
            'pca': self.pca
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return {'episode': 0, 'eval_rewards': [], 'seed_logs': []}
        
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actors.load_state_dict(ckpt['actors_state_dict'])
        self.old_actors.load_state_dict(ckpt['old_actors_state_dict'])
        self.ref_actor.load_state_dict(ckpt['ref_actor_state_dict'])
        for opt, state in zip(self.optimizers, ckpt['optimizers_state_dict']):
            opt.load_state_dict(state)
        self.current_policy_idx = ckpt['current_policy_idx']
        self.buffer.buffers = ckpt['buffer_data']
        
        if ckpt.get('current_episodes') is not None:
            self.buffer.current_episodes = ckpt['current_episodes']
            
        self.trauma_centers = ckpt.get('trauma_centers', [])
        self.scaler = ckpt.get('scaler', StandardScaler())
        self.pca = ckpt.get('pca', None)
        
        return ckpt