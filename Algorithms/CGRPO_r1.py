import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents Tkinter thread crashes during multiprocessing
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utils as bf


class AGRPO:
    """
    Continuous Group Relative Policy Optimization (CGRPO).
    Maintains a population of N agents, clusters their visited states into contexts,
    and optimizes policies relative to the context baselines.
    """
    def __init__(self, env, seed=42, hidden_dim=256, lr=5e-4, N=20, K=2, epsilon=0.2,
                 tau=0.5, lam_s=0.01, lam_d=0.0001, lam_t=0.01, gamma=0.99, dbscan_eps=0.4, 
                 TRAUMA_THRESHOLD = -20.0):
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
        self.Trauma_Threshold = TRAUMA_THRESHOLD
        self.trauma_forgeting_threshold = self.Trauma_Threshold / 3.0
        self.current_track_data = None

        self.trauma_centers = []

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
        self.buffer.add(self.current_policy_idx, self._cached_obs, self._cached_feat, self._cached_action, self._cached_logprob, reward, pos=pos)

        if done:
            self.buffer.finish_episode(self.current_policy_idx)
            self.current_policy_idx += 1

            if self.current_policy_idx >= len(self.actors):
                stats = self.update()
                self.current_policy_idx = 0
                return stats
        return None

    def _update_reference_policy_mixture(self, all_returns):
        """
        Updates the reference policy to represent the top K performing agents.
        Acts as a strong baseline/teacher for the entire population.
        """
        num_top = max(1, int(len(self.actors) // 10))
        top_indices = np.argsort(all_returns)[-num_top:]

        self.last_elite_rewards = [all_returns[i] for i in top_indices]
        self.last_elite_indices = top_indices

        with torch.no_grad():
            avg_state_dict = {}
            for name in self.actors[0].state_dict():
                avg_state_dict[name] = torch.stack(
                    [self.actors[idx].state_dict()[name] for idx in top_indices]
                ).mean(dim=0)
            self.ref_actor.load_state_dict(avg_state_dict) 
        return self.last_elite_rewards

    # --- UPDATE HELPER METHODS ---

    def _gather_metrics(self):
        all_phi, all_returns, all_features_np, all_pos_np = [], [], [], []
        for i in range(len(self.actors)):
            traj = self.buffer.get_latest_trajectory(i)
            obs = torch.stack(traj["obs"]).to(self.device)

            with torch.no_grad():
                feat = self.actors[i].forward_features(obs)
                dist = self.actors[i].get_distribution(feat)
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
            all_returns.append(reward_sum)
            all_features_np.append(torch.stack(traj["feature"]).cpu().numpy())
            all_pos_np.append(np.array(traj["pos"]))

        return np.array(all_phi), all_returns, all_features_np, all_pos_np

    def _cluster_states(self, all_features_np, all_pos_np):
        """Clustering states into contexts using a 3-step process: 
        1) PCA for dimensionality reduction, 
        2) HDBSCAN for core cluster detection, 
        3) KNN for assigning all points to clusters."""
        flat_features = np.concatenate(all_features_np, axis=0)
        
        subsample_idx = np.arange(0, len(flat_features), 20)
        features_subsampled = flat_features[subsample_idx]
        
        scaled_sub = self.scaler.fit_transform(features_subsampled)

        pca_dims = 20
        pca_comps = min(pca_dims, flat_features.shape[1])
        if self.pca is None or self.pca.n_components != pca_comps:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=pca_comps)
        
        pca_features_sub = self.pca.fit_transform(scaled_sub)
        pca_features_sub = np.nan_to_num(pca_features_sub, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

        c_size = int(max(10, len(pca_features_sub) * 0.005))
        min_samples = int(max(5, c_size / 2))
        
        dbscan = HDBSCAN(
            min_cluster_size=c_size, 
            min_samples=min_samples, 
            cluster_selection_epsilon=self.dbscan_eps, 
            core_dist_n_jobs=1
        )
        hdbscan_labels = dbscan.fit_predict(pca_features_sub)

        from sklearn.neighbors import KNeighborsClassifier
        
        valid_mask = hdbscan_labels != -1
        valid_features = pca_features_sub[valid_mask]
        valid_labels = hdbscan_labels[valid_mask]

        if len(valid_labels) == 0:
            print("[Clustering] Warning: HDBSCAN found only noise. Defaulting to single fallback cluster.")
            labels = np.zeros(len(flat_features), dtype=int)
        else:
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(valid_features, valid_labels)

            full_scaled = self.scaler.transform(flat_features)
            full_pca = self.pca.transform(full_scaled)
            full_pca = np.nan_to_num(full_pca, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
            
            labels = knn.predict(full_pca)
        
        traj_lengths = [len(f) for f in all_features_np]
        self._plot_clusters(full_pca if len(valid_labels) > 0 else pca_features_sub, labels)

        flat_pos = np.concatenate(all_pos_np, axis=0)
        self._plot_spatial_only(flat_pos, labels)
        
        return labels, traj_lengths
    
    
    def _plot_spatial_only(self, pos_data, labels):
        import os
        os.makedirs('plots', exist_ok=True)
        
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        if hasattr(self, 'current_track_data') and self.current_track_data is not None:
            track_x = [p[0] for p in self.current_track_data]
            track_y = [p[1] for p in self.current_track_data]
            
            track_x.append(track_x[0])
            track_y.append(track_y[0])
            
            plt.plot(track_x, track_y, color='darkgray', linewidth=35, alpha=0.5, label='Road')
            plt.plot(track_x, track_y, color='white', linewidth=2, linestyle='--', alpha=0.8)
        
        plt.scatter(pos_data[:, 0], pos_data[:, 1], c=labels, cmap='tab20', s=8, alpha=0.8, zorder=6)
        
        ep = getattr(self, 'current_episode', 0)
        plt.title(f"Spatial Cluster Distribution (Ep {ep})")
        plt.xlabel("X Position (World)")
        plt.ylabel("Y Position (World)")
        plt.grid(True, alpha=0.3)
        plt.axis('equal') 
        
        if hasattr(self, 'current_track_data') and self.current_track_data is not None:
            plt.legend()
        print("plot created")
        plt.savefig(f'plots/carracing_spatial_ep{ep}.png')
        plt.close()

    def _plot_clusters(self, data, labels):
        """Generates a headless PCA scatter plot of identified state clusters."""
        plt.figure(figsize=(10, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', s=10)
        plt.savefig('state_space_clustered.png')
        plt.close()

    # def _compute_advantages(self, labels, traj_lengths, all_features_np, all_pos_np):
    #     all_returns_to_go = []
    #     for i in range(len(self.actors)):
    #         rewards = self.buffer.get_latest_trajectory(i)["reward"]
    #         rtg = bf.compute_returns_to_go(rewards, self.gamma, self.device)
    #         all_returns_to_go.append(rtg)

    #     flat_returns = torch.cat(all_returns_to_go)
    #     labels_t = torch.from_numpy(labels).to(self.device)

    #     unique_labels = torch.unique(labels_t)
    #     cluster_means = {}

    #     global_mean = flat_returns.mean()
    #     for c in unique_labels:
    #         c_item = c.item()
    #         if c_item != -1:
    #             cluster_means[c_item] = flat_returns[labels_t == c].mean()

    #     cluster_means[-1] = global_mean

    #     advantages = []
    #     start = 0
    #     for i, length in enumerate(traj_lengths):
    #         policy_labels = labels[start:start+length]
    #         policy_labels_t = labels_t[start:start+length]
    #         baseline = torch.tensor([cluster_means.get(int(l), cluster_means.get(-1, 0.0)) for l in policy_labels], dtype=torch.float32).to(self.device)
    #         adv = all_returns_to_go[i] - baseline
    #         advantages.append(adv)
    #         start += length

    #     flat_features = np.concatenate(all_features_np, axis=0)
    #     flat_pos = np.concatenate(all_pos_np, axis=0)
    #     labels_np = labels  

    #     unique_labels = np.unique(labels_np)
    #     for c in unique_labels:
    #         if c != -1 and cluster_means.get(c, 0) < self.Trauma_Threshold:
                
    #             mask = (labels_np == c)
    #             trauma_points = flat_features[mask]

    #             if len(trauma_points) > 5:
    #                 mu = torch.tensor(trauma_points.mean(axis=0), dtype=torch.float32).to(self.device)
    #                 # + 1e-4 für numerische Stabilität
                    
    #                 sigma = torch.tensor(trauma_points.std(axis=0) + 1e-4, dtype=torch.float32).to(self.device)
                    
    #                 severity = abs(cluster_means[c])
    #                 mu_pos = flat_pos[mask].mean(axis=0)
                    
    #                 sigma_pos_x = flat_pos[mask][:, 0].std()
    #                 sigma_pos_y = flat_pos[mask][:, 1].std()
                    
    #                 if sigma_pos_x < 0.1: sigma_pos_x = 0.1
    #                 if sigma_pos_y < 0.1: sigma_pos_y = 0.1

    #                 self.trauma_centers.append({
    #                     'mu': mu,
    #                     'sigma': sigma,
    #                     'mu_pos': mu_pos,          
    #                     'sigma_pos_x': sigma_pos_x,
    #                     'sigma_pos_y': sigma_pos_y,   
    #                     'weight': severity
    #                 })
    #                 # print(f"[Memory] Trauma saved! Weight: {severity:.2f}, Score: {len(trauma_points)}")

    #                 if len(self.trauma_centers) > 200:
    #                     self.trauma_centers.pop(0)
    #                 else:
    #                     print(f"[Memory] Trauma center added. Total centers: {len(self.trauma_centers)}")
        
    #     return advantages

    # def _compute_advantages(self, labels, traj_lengths, all_features_np, all_pos_np):
        # 1. Collect returns on GPU
        all_returns_to_go = []
        for i in range(len(self.actors)):
            rewards = self.buffer.get_latest_trajectory(i)["reward"]
            rtg = bf.compute_returns_to_go(rewards, self.gamma, self.device)
            all_returns_to_go.append(rtg)

        flat_returns = torch.cat(all_returns_to_go) # Shape: (Total_Steps,)
        labels_t = torch.from_numpy(labels).to(self.device).long() # Shape: (Total_Steps,)

        # 2. Vectorized Cluster Means calculation on GPU
        # We map labels: -1 -> Index 0, 0 -> Index 1, 1 -> Index 2, etc.
        max_label = int(labels.max())
        means_vec = torch.zeros(max_label + 2, device=self.device)
        
        # Calculate global mean (for noise/label -1)
        means_vec[0] = flat_returns.mean()
        
        # Calculate each cluster mean using GPU masking
        unique_labels = torch.unique(labels_t)
        for c in unique_labels:
            if c == -1: continue
            means_vec[c + 1] = flat_returns[labels_t == c].mean()

        # 3. Vectorized Baseline Expansion (The Speed Boost)
        # Instead of a Python loop, we use labels_t as an index array.
        # This creates the full baseline tensor in one O(1) GPU operation.
        flat_baselines = means_vec[labels_t + 1]
        
        # 4. Advantage Calculation
        flat_advantages = flat_returns - flat_baselines
        
        # Split back into individual trajectories for the agents
        advantages = list(torch.split(flat_advantages, traj_lengths))

        # 5. Trauma Identification (Keep on CPU as it involves small loops and NumPy/Logging)
        flat_features = np.concatenate(all_features_np, axis=0)
        flat_pos = np.concatenate(all_pos_np, axis=0)
        dim = self.actors[0].extractor.feature_dim
        
        # Use NumPy for the cluster loop to avoid unnecessary .item() calls
        labels_np = labels)
        for c in np.unique(labels_np):
            if c != -1:
                # Check mean from our GPU vector
                c_mean = means_vec[c + 1].item()
                if c_mean < self.Trauma_Threshold:
                    mask = (labels_np == c)
                    trauma_points = flat_features[mask]

                    if len(trauma_points) > 5:
                        mu = torch.tensor(trauma_points.mean(axis=0), dtype=torch.float32).to(self.device)
                        sigma = torch.tensor(trauma_points.std(axis=0) + 1e-4, dtype=torch.float32).to(self.device)
                        
                        severity = abs(c_mean)
                        mu_pos = flat_pos[mask].mean(axis=0)
                        sigma_pos_x = max(float(flat_pos[mask][:, 0].std()), 0.1)
                        sigma_pos_y = max(float(flat_pos[mask][:, 1].std()), 0.1)

                        self.trauma_centers.append({
                            'mu': mu, 'sigma': sigma, 'mu_pos': mu_pos,          
                            'sigma_pos_x': sigma_pos_x, 'sigma_pos_y': sigma_pos_y,   
                            'weight': severity
                        })
                        
                        if len(self.trauma_centers) > 200:
                            self.trauma_centers.pop(0)
                        else:
                            print(f"[Memory] Trauma center added. Total: {len(self.trauma_centers)}")
        
        return advantages

    def _compute_advantages(self, labels, traj_lengths, all_features_np, all_pos_np):
        # 1. Compute Returns-to-Go on GPU
        all_returns_to_go = []
        for i in range(len(self.actors)):
            rewards = self.buffer.get_latest_trajectory(i)["reward"]
            rtg = bf.compute_returns_to_go(rewards, self.gamma, self.device)
            all_returns_to_go.append(rtg)

        flat_returns = torch.cat(all_returns_to_go)
        labels_t = torch.from_numpy(labels).to(self.device).long()

        # 2. Vectorized Cluster Means (Returns) on GPU
        # Index 0: Noise (-1), Index 1+: Clusters (0, 1, ...)
        max_label = int(labels.max())
        means_vec = torch.zeros(max_label + 2, device=self.device)
        
        # Calculate global mean for noise (label -1)
        means_vec[0] = flat_returns.mean()
        
        # Calculate mean return for each cluster
        unique_labels_t = torch.unique(labels_t)
        cluster_means_dict = {} # Keep a copy for trauma filtering
        
        for c in unique_labels_t:
            c_val = int(c.item())
            # GPU Masking for mean calculation
            m = flat_returns[labels_t == c_val].mean()
            means_vec[c_val + 1] = m
            if c_val != -1:
                cluster_means_dict[c_val] = m.item()

        # 3. Vectorized Baseline Expansion and Advantages
        # O(1) mapping of labels to their respective cluster means on GPU
        flat_baselines = means_vec[labels_t + 1]
        flat_advantages = flat_returns - flat_baselines
        
        # Split back into individual trajectories
        advantages = list(torch.split(flat_advantages, traj_lengths))

        # 4. Trauma Management (Identification & Merging)
        flat_features = np.concatenate(all_features_np, axis=0)
        flat_pos = np.concatenate(all_pos_np, axis=0)
        dim = self.actors[0].extractor.feature_dim
        
        labels_np = labels
        unique_labels_np = np.unique(labels_np)
        
        for c in unique_labels_np:
            # Check if cluster mean return is below threshold
            if c != -1 and cluster_means_dict.get(int(c), 0.0) < self.Trauma_Threshold:
                mask = (labels_np == c)
                trauma_points = flat_features[mask]

                if len(trauma_points) > 5:
                    # Calculate new trauma candidates (Latent & Spatial)
                    mu_feat = torch.tensor(trauma_points.mean(axis=0), dtype=torch.float32).to(self.device)
                    sigma_feat = torch.tensor(trauma_points.std(axis=0) + 1e-4, dtype=torch.float32).to(self.device)
                    severity = abs(cluster_means_dict[int(c)])
                    
                    mu_pos = flat_pos[mask].mean(axis=0)
                    sigma_pos_x = max(float(flat_pos[mask][:, 0].std()), 0.1)
                    sigma_pos_y = max(float(flat_pos[mask][:, 1].std()), 0.1)

                    new_trauma_data = {
                        'mu': mu_feat, 
                        'sigma': sigma_feat, 
                        'mu_pos': mu_pos,          
                        'sigma_pos_x': sigma_pos_x, 
                        'sigma_pos_y': sigma_pos_y,   
                        'weight': severity
                    }

                    # Check for overlap with existing centers (Latent Space Reinforcement)
                    merged = False
                    for existing in self.trauma_centers:
                        # Distance in latent space normalized by feature dimensions
                        dist_sq = torch.sum(((mu_feat - existing['mu']) / existing['sigma']) ** 2) / dim
                        
                        # If within 1.0 standard deviation in feature space, we update the old one
                        if dist_sq < 1.0:
                            existing.update(new_trauma_data)
                            merged = True
                            # Optional: Log the reinforcement
                            # print(f"[Memory] Trauma reinforced. New Weight: {severity:.2f}")
                            break
                    
                    if not merged:
                        self.trauma_centers.append(new_trauma_data)
                        print(f"[Memory] New trauma center added. Total: {len(self.trauma_centers)}")

                    # Maintain memory cap
                    if len(self.trauma_centers) > 200:
                        self.trauma_centers.pop(0)
        
        return advantages

    def _compute_trauma_penalty(self, feat):
        """Calculates a penalty based on the distance of the current state features to known trauma centers.
        The penalty is a weighted sum of Gaussian functions centered at each trauma point, where the weight is determined by the severity of the trauma (e.g., how low the returns were in that cluster)."""
        if not self.trauma_centers:
            return torch.tensor(0.0).to(self.device)

        total_penalty = torch.tensor(0.0).to(self.device)
        bandwidth = 20.0                                             # Controls how quickly the penalty falls off with distance
        dim = self.actors[0].extractor.feature_dim

        for center in self.trauma_centers:
            mu = center['mu']
            sigma = center['sigma']
            weight = center['weight']

            dist_sq = torch.sum(((feat - mu) / sigma) ** 2, dim=-1)

            normalized_dist = dist_sq / dim
            gauss_penalty = torch.exp(-normalized_dist / (2 * (bandwidth ** 2)))

            total_penalty += (gauss_penalty * weight).mean()

        return total_penalty

    def _get_trajectory_obs(self, policy_idx):
        traj = self.buffer.get_latest_trajectory(policy_idx)
        return torch.stack(traj["obs"]).to(self.device)

    def _compute_exp_diversity(self, i, all_mus, current_lam_d):
        if current_lam_d <= 0:
            return torch.tensor(0.0).to(self.device)

        penalties = []
        mu_i = self.actors[i].get_distribution(self.actors[i].forward_features(
            self._get_trajectory_obs(i)
        )).mean

        exp_scale = 5.0
        for j in range(len(self.actors)):
            if i == j: continue

            min_len = min(mu_i.shape[0], all_mus[j].shape[0])
            step_sim = torch.cosine_similarity(mu_i[:min_len], all_mus[j][:min_len], dim=-1).mean()
            #step_sim = torch.cosine_similarity(mu_i, all_mus[j], dim=-1).mean()
            penalty = torch.exp(exp_scale * (step_sim - self.tau)) - 1.0
            penalties.append(torch.clamp(penalty, min=0.0))

        return torch.stack(penalties).mean() if penalties else torch.tensor(0.0).to(self.device)

    # --- MAIN OPTIMIZATION ---

    def update(self):
        ep = getattr(self, 'current_episode', 0)
        current_lam_d = self.lam_d if ep >= 100 else 0.0
        min_std = max(0.05, 0.5 - (min(1.0, ep/200.0) * 0.45)) if ep < 160 else 0.00005

        loss_stats = {
            "actor_loss": 0.0,
            "smooth_loss": 0.0,
            "div_loss": 0.0,
            "trauma_loss": 0.0
        }

        # 1. Decay existing traumas by 20%
        for center in self.trauma_centers:
            center['weight'] *= 0.8
            
        # 2. Clean up "forgotten" traumas (e.g., weight below 1.0)
        self.trauma_centers = [c for c in self.trauma_centers if c['weight'] > self.trauma_forgeting_threshold]

        for i in range(len(self.actors)):
            self.old_actors[i].load_state_dict(self.actors[i].state_dict())
            self.actors[i].log_std.data = torch.clamp(self.actors[i].log_std.data, min=np.log(min_std))

        phi, returns, features_np, all_pos_np = self._gather_metrics()
        labels, traj_lengths = self._cluster_states(features_np, all_pos_np)

        advantages = self._compute_advantages(labels, traj_lengths, features_np, all_pos_np)

        advantages_raw_flat = torch.cat(advantages) 
        labels_t = torch.from_numpy(labels).to(self.device)

        noise_mask = (labels_t == -1)
        if noise_mask.any():
            avg_noise_adv = advantages_raw_flat[noise_mask].mean().item()
            avg_cluster_adv = advantages_raw_flat[~noise_mask].mean().item()
            print(f"[Debug] Raw Adv - Noise: {avg_noise_adv:.4f} | Clusters: {avg_cluster_adv:.4f}")

        phi_norm = (phi - phi.mean(axis=0)) / (phi.std(axis=0) + 1e-8)
        groups = KMeans(n_clusters=min(self.K, len(self.actors)), n_init='auto').fit_predict(phi_norm)

        normalized_advantages = bf.normalize_advantages_by_group(advantages, groups, self.device)

        all_advantages_flat = torch.cat(normalized_advantages)
        sigma_global = all_advantages_flat.std() + 1e-8

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"[Update] Episode {ep + 1} | Clusters: {n_clusters}, Noise: {n_noise}")

        actor_features = {}
        all_mus = {}
        for j in range(len(self.actors)):
            obs = self._get_trajectory_obs(j)
            feat = self.actors[j].forward_features(obs)
            actor_features[j] = feat
            dist = self.actors[j].get_distribution(feat)
            all_mus[j] = dist.mean.detach()

        group_members = {g: np.where(groups == g)[0] for g in np.unique(groups)}

        for i in range(len(self.actors)):
            traj = self.buffer.get_latest_trajectory(i)
            obs = torch.stack(traj["obs"]).to(self.device)
            actions = torch.stack(traj["action"]).to(self.device)
            old_log_probs = torch.stack(traj["log_probs"]).to(self.device)
            adv = normalized_advantages[i].to(self.device)

            group_idx = group_members[groups[i]]
            g_adv = torch.cat([advantages[j] for j in group_idx])
            epsilon_i = self.epsilon * torch.clamp(g_adv.std() / sigma_global, min=1.0)

            feat = actor_features[i]
            dist = self.actors[i].get_distribution(feat)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - epsilon_i, 1 + epsilon_i) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            l_smooth = torch.mean((feat[1:] - feat[:-1]) ** 2) if feat.shape[0] > 1 else 0
            l_div = self._compute_exp_diversity(i, all_mus, current_lam_d)
            l_trauma = self._compute_trauma_penalty(feat)

            total_loss = actor_loss + self.lam_s * l_smooth + current_lam_d * l_div + self.lam_t * l_trauma

            loss_stats["actor_loss"] += actor_loss.item()
            loss_stats["smooth_loss"] += l_smooth.item() if torch.is_tensor(l_smooth) else l_smooth
            loss_stats["div_loss"] += l_div.item()
            loss_stats["trauma_loss"] += l_trauma.item()

            self.optimizers[i].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.optimizers[i].step()

        elite_stats = self._update_reference_policy_mixture(returns)
        self.buffer.clear_buffer()

        for key in loss_stats:
            loss_stats[key] /= len(self.actors)
        print(f"  [Loss Analysis] Actor: {loss_stats['actor_loss']:.4f} | Div: {loss_stats['div_loss']:.4f} | Trauma: {loss_stats['trauma_loss']:.4f}")

        return {
            "actor": loss_stats["actor_loss"],
            "div": loss_stats["div_loss"],
            "trauma": loss_stats["trauma_loss"],
            "elite_mean": np.mean(elite_stats)
        }

    def save_checkpoint(self, path, ep, eval_rewards):
        import pickle
        checkpoint = {
            'episode': ep,
            'eval_rewards': eval_rewards,
            'current_policy_idx': self.current_policy_idx,
            'actors_state_dict': self.actors.state_dict(),
            'old_actors_state_dict': self.old_actors.state_dict(),
            'ref_actor_state_dict': self.ref_actor.state_dict(),
            'optimizers_state_dict': [opt.state_dict() for opt in self.optimizers],
            'buffer_data': self.buffer.buffers,
            'current_episodes': self.buffer.current_episodes if hasattr(self.buffer, 'current_episodes') else None,
            'trauma_centers': self.trauma_centers,
            'scaler': self.scaler,
            'pca': self.pca
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        import os
        if not os.path.exists(path):
            return {'episode': 0, 'eval_rewards': []}

        ckpt = torch.load(path, map_location=self.device)

        self.actors.load_state_dict(ckpt['actors_state_dict'])
        self.old_actors.load_state_dict(ckpt['old_actors_state_dict'])
        self.ref_actor.load_state_dict(ckpt['ref_actor_state_dict'])

        for opt, state in zip(self.optimizers, ckpt['optimizers_state_dict']):
            opt.load_state_dict(state)

        self.current_policy_idx = ckpt['current_policy_idx']
        self.buffer.buffers = ckpt['buffer_data']

        if ckpt['current_episodes'] is not None:
            self.buffer.current_episodes = ckpt['current_episodes']
            
        self.trauma_centers = ckpt.get('trauma_centers', [])
        self.scaler = ckpt.get('scaler', StandardScaler())
        self.pca = ckpt.get('pca', None)

        return ckpt