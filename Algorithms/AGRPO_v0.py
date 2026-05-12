import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents Tkinter thread crashes during multiprocessing
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utils as bf

class AGRPO:
    """
    Advanced Group Relative Policy Optimization (AGRPO).
    
    This algorithm maintains a population of N agents, clusters their visited states 
    into contexts (latent space), and optimizes policies relative to these context baselines.
    It features a dynamic "Trauma Memory" system that identifies, records, and reinforces 
    highly penalized areas in the latent space, forcing agents to avoid persistent hazards.
    """
    def __init__(self, env, seed=42, hidden_dim=256, lr=5e-4, N=10, K=2, epsilon=0.2,
                 tau=0.5, lam_s=0.01, lam_d=0.05, lam_t=0.05, gamma=0.99, dbscan_eps=0.4,
                 warmup_episodes=100, initial_threshold=-1.0, plot_interval=20): # lam_t=0.005, lam_d=0.00005

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.N = N
        self.K = K
        self.epsilon = epsilon
        self.tau = tau
        self.lam_s = lam_s      # Smoothness penalty weight
        self.lam_d = lam_d      # Diversity penalty weight
        self.lam_t = lam_t      # Trauma penalty weight
        self.gamma = gamma
        self.dbscan_eps = dbscan_eps
        self.plot_interval = plot_interval  # Plot frequency (every N updates)

        # Warmup and Exploration Control
        self.running_reward_std = 1.0 # Initial guess
        self.warmup_episodes = warmup_episodes
        self.std_history = []
        self.target_min_std_history = []
        self.target_max_std_history = []
        
        # Trauma Management Parameters
        self.dynamic_threshold = initial_threshold
        self.trauma_centers = []
        self.current_track_data = None

        # Population Control
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        self.hidden_dim = hidden_dim
        self.base_lr = lr

        # Neural Networks
        self.actors = nn.ModuleList([bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device) for _ in range(N)])
        self.old_actors = nn.ModuleList([bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device) for _ in range(N)])
        self.ref_actor = bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.ref_actor.load_state_dict(self.actors[0].state_dict())

        for i in range(self.N):
            self.old_actors[i].load_state_dict(self.actors[i].state_dict())

        # Optimization & Memory
        self.optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.buffer = bf.PopulationBuffer(N)
        self.current_policy_idx = 0

        # Clustering utilities
        self.scaler = StandardScaler()
        self.pca = None

    # =========================================================================
    # ENVIRONMENT INTERACTION
    # =========================================================================

    def select_action(self, state, evaluate=False):
        """Selects an action using the current policy or the reference policy during evaluation."""
        obs_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                action_t = self.ref_actor.get_deterministic_action(obs_t)
            else:
                action_t, log_prob_t, feat_t = self.old_actors[self.current_policy_idx].sample_action(obs_t)
                
                # Cache data for the buffer step
                self._cached_obs = obs_t.squeeze(0).cpu()
                self._cached_action = action_t.squeeze(0).cpu()
                self._cached_logprob = log_prob_t.squeeze(0).cpu()
                self._cached_feat = feat_t.squeeze(0).cpu()

        return action_t.squeeze(0).cpu().numpy()

    def step(self, state, action, reward, next_state, done, pos=(0.0, 0.0)):
        """Stores transition in the buffer and triggers a population update if the batch is complete."""
        self.buffer.add(self.current_policy_idx, self._cached_obs, self._cached_feat, 
                        self._cached_action, self._cached_logprob, reward, pos=pos)

        if done:
            self.buffer.finish_episode(self.current_policy_idx)
            self.current_policy_idx += 1

            # Once all agents have collected an episode, trigger the optimization step
            if self.current_policy_idx >= len(self.actors):
                stats = self.update()
                self.current_policy_idx = 0
                return stats
        return None

    # =========================================================================
    # POPULATION DYNAMICS & CLUSTERING
    # =========================================================================

    def _update_reference_policy_mixture(self, all_returns):
        """
        Updates the reference (teacher) policy to represent the top K performing agents.
        This provides a strong baseline for KL-divergence calculations.
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

    def _gather_metrics(self):
        """Collects trajectories, performance metrics (phi), and features across all agents."""
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
                
                # Phi represents behavioral and performance characteristics for agent grouping
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
        """
        Clusters visited states into contexts using a 3-step process: 
        1) PCA for dimensionality reduction.
        2) HDBSCAN for core cluster detection (noise robust).
        3) KNN to assign all remaining points to the established clusters.
        """
        flat_features = np.concatenate(all_features_np, axis=0)
        
        # Subsample for faster clustering
        subsample_idx = np.arange(0, len(flat_features), 20) # Standard value 20
        features_subsampled = flat_features[subsample_idx]
        scaled_sub = self.scaler.fit_transform(features_subsampled)

        # PCA
        pca_dims = 20
        pca_comps = min(pca_dims, flat_features.shape[1])
        if self.pca is None or self.pca.n_components != pca_comps:
            self.pca = PCA(n_components=pca_comps)
        
        pca_features_sub = self.pca.fit_transform(scaled_sub)
        pca_features_sub = np.nan_to_num(pca_features_sub, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

        # HDBSCAN
        c_size = int(max(10, len(pca_features_sub) * 0.005))
        min_samples = int(max(5, c_size / 2))
        
        dbscan = HDBSCAN(min_cluster_size=c_size, min_samples=min_samples, 
                         cluster_selection_epsilon=self.dbscan_eps, core_dist_n_jobs=1)
        hdbscan_labels = dbscan.fit_predict(pca_features_sub)

        # KNN assignment
        valid_mask = hdbscan_labels != -1
        valid_features = pca_features_sub[valid_mask]
        valid_labels = hdbscan_labels[valid_mask]

        full_scaled = self.scaler.transform(flat_features)
        full_pca = self.pca.transform(full_scaled)
        full_pca = np.nan_to_num(full_pca, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

        if len(valid_labels) == 0:
            print("[Clustering] Warning: HDBSCAN found only noise. Defaulting to single fallback cluster.")
            labels = np.zeros(len(flat_features), dtype=int)
        else:
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(valid_features, valid_labels)

            labels = knn.predict(full_pca)
        
        traj_lengths = [len(f) for f in all_features_np]

        # Generate diagnostic plots (periodic to save computation)
        ep = getattr(self, 'current_episode', 0)
        if (ep + 1) % self.plot_interval == 0:
            self._plot_clusters(full_pca, labels)
            self._plot_spatial_only(np.concatenate(all_pos_np, axis=0), labels)

        return labels, traj_lengths

    # =========================================================================
    # ADVANTAGES & TRAUMA MANAGEMENT
    # =========================================================================

    def _compute_advantages(self, labels, traj_lengths, all_features_np, all_pos_np, pop_mean, reward_scale):
        """
        Calculates context-specific advantages using GPU vectorization and manages the Trauma memory.
        Trauma centers are recorded or reinforced if a context yields severe negative returns.
        """
        K_selector = 0.8 # Multiplier to determine trauma severity threshold based on population performance
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
        means_vec[0] = flat_returns.mean() # Global mean for noise
        
        cluster_means_dict = {} 
        for c in torch.unique(labels_t):
            c_val = int(c.item())
            m = flat_returns[labels_t == c_val].mean()
            means_vec[c_val + 1] = m
            if c_val != -1:
                cluster_means_dict[c_val] = m.item()

        # 3. Vectorized Baseline Expansion and Advantages
        flat_baselines = means_vec[labels_t + 1]
        flat_advantages = flat_returns - flat_baselines
        advantages = list(torch.split(flat_advantages, traj_lengths))

        # 4. Trauma Management (Identification & Merging)
        flat_features = np.concatenate(all_features_np, axis=0)
        flat_pos = np.concatenate(all_pos_np, axis=0)
        dim = self.actors[0].extractor.feature_dim
        
        self.dynamic_threshold = pop_mean - (K_selector * reward_scale)
        max_weight = 100 * reward_scale # 1.0 / max(1e-6, self.lam_t)
        labels_np = labels
        for c in np.unique(labels_np):
            # Check if cluster mean return crosses the severity threshold
            c_return = cluster_means_dict.get(int(c), 0.0)
            if c != -1 and c_return < self.dynamic_threshold:
                mask = (labels_np == c)
                trauma_points = flat_features[mask]

                if len(trauma_points) > 5:
                    mu_feat = torch.tensor(trauma_points.mean(axis=0), dtype=torch.float32).to(self.device)
                    sigma_feat = torch.tensor(trauma_points.std(axis=0) + 1e-4, dtype=torch.float32).to(self.device)
                    severity = abs(c_return - self.dynamic_threshold)
                    
                    mu_pos = flat_pos[mask].mean(axis=0)
                    sigma_pos_x = max(float(flat_pos[mask][:, 0].std()), 0.1)
                    sigma_pos_y = max(float(flat_pos[mask][:, 1].std()), 0.1)

                    new_trauma_data = {
                        'mu': mu_feat, 'sigma': sigma_feat, 'mu_pos': mu_pos,          
                        'sigma_pos_x': sigma_pos_x, 'sigma_pos_y': sigma_pos_y,   
                        'weight': min(severity, max_weight)
                    }

                    # Reinforce existing traumas if the new hazard is in the same latent area
                    merged = False
                    if self.trauma_centers:
                        trauma_mus = torch.stack([t['mu'] for t in self.trauma_centers])
                        trauma_sigmas = torch.stack([t['sigma'] for t in self.trauma_centers])
                        dist_sq = torch.sum(((mu_feat.unsqueeze(0) - trauma_mus) / trauma_sigmas) ** 2, dim=-1) / dim
                        min_idx = torch.argmin(dist_sq).item()

                        if dist_sq[min_idx] < 1.0:
                            existing = self.trauma_centers[min_idx]
                            # Accumulate weights (Gravity effect)
                            existing['weight'] += min(existing['weight'] + severity, max_weight)
                            # Average positions and features (Centering)
                            existing['mu'] = (existing['mu'] + mu_feat) / 2
                            existing['mu_pos'] = (existing['mu_pos'] + mu_pos) / 2
                            # Average standard deviations (Smoothing)
                            existing['sigma'] = (existing['sigma'] + sigma_feat) / 2
                            existing['sigma_pos_x'] = (existing['sigma_pos_x'] + sigma_pos_x) / 2
                            existing['sigma_pos_y'] = (existing['sigma_pos_y'] + sigma_pos_y) / 2
                            merged = True

                    if not merged:
                        self.trauma_centers.append(new_trauma_data)
                        print(f"[Trauma] New trauma center added at cluster {c} with severity {severity:.2f} and weight {new_trauma_data['weight']:.2f}. Total centers: {len(self.trauma_centers)}")

                    # Maintain memory cap
                    if len(self.trauma_centers) > 200:
                        self.trauma_centers.pop(0)
        
        return advantages

    def _compute_trauma_penalty(self, feat):
        """
        Calculates a repulsion penalty based on the Gaussian distance of current features 
        to known trauma centers in the latent space.
        """
        if not self.trauma_centers:
            return torch.tensor(0.0).to(self.device)

        total_penalty = torch.tensor(0.0).to(self.device)
        bandwidth = 20.0 # Controls the reach of the trauma 'repulsion' field
        dim = self.actors[0].extractor.feature_dim

        for center in self.trauma_centers:
            mu, sigma, weight = center['mu'], center['sigma'], center['weight']

            dist_sq = torch.sum(((feat - mu) / sigma) ** 2, dim=-1)
            normalized_dist = dist_sq / dim
            # print(normalized_dist)

            gauss_penalty = torch.exp(-normalized_dist / (2 * (bandwidth ** 2)))
            total_penalty += (gauss_penalty * weight).mean()

        return total_penalty

    def _compute_exp_diversity(self, i, all_mus, current_lam_d):
        """Calculates an exponential penalty if agent policies become too similar (collapsing)."""
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
        """
        Visualizes the health of exploration.
        Blue: What the agents actually do.
        Red (Dashed): The minimum safety net we provide.
        """
        import os
        os.makedirs('plots', exist_ok=True)
        plt.figure(figsize=(10, 6))

        x = np.arange(len(self.std_history))
        plt.plot(x, self.std_history, label='Actual Policy STD (Avg)', color='#1f77b4', linewidth=2)
        plt.plot(x, self.target_min_std_history, label='Min Std Floor (Safety Net)', color='#d62728', linestyle='--')

        if len(self.target_max_std_history) == len(self.std_history):
            plt.plot(x, self.target_max_std_history, label='Max Std Ceiling', color='#2ca02c', linestyle='--')
            plt.fill_between(x, self.target_min_std_history, self.target_max_std_history,
                             color='gray', alpha=0.1, label='Allowed Std Corridor')

        plt.title(f"Exploration Health Analysis (Ep {getattr(self, 'current_episode', 0)})")
        plt.xlabel("Update Steps")
        plt.ylabel("Standard Deviation")
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.savefig('plots/exploration_health.png')
        plt.close()

    # =========================================================================
    # MAIN OPTIMIZATION
    # =========================================================================

    def update(self):
        '''
        Core optimization loop with Tiered Population Strategy.
        - Calculates exploration decay (Half-life 100).
        - Balances penalties (Trauma/Diversity) against Reward Scale via EMA.
        - Triggers trauma memory based on statistical Z-score outliers.
        '''
        ep = getattr(self, 'current_episode', 0)
        
        # 1. EXPONENTIAL DECAY (Half-life 100 episodes)
        std_start, std_warmup_end, std_final_floor = 0.5, 0.2, 0.05
        if ep < self.warmup_episodes:
            # Linear decay during warmup
            min_std = max(std_warmup_end, std_start - (ep/self.warmup_episodes) * (std_start - std_warmup_end)) 
        else:
            # Exponential decay: lambda for 100 ep half-life = 0.006931
            decay_lambda = 0.006931
            time_passed = ep - self.warmup_episodes
            min_std = std_final_floor + (std_warmup_end - std_final_floor) * np.exp(-decay_lambda * time_passed)
        
        double_warmup = 2 * self.warmup_episodes
        std_max_floor = std_final_floor + 0.05 

        if ep < double_warmup:
            max_std = std_start 
        else:
            time_passed_max = ep - double_warmup
            max_std = std_max_floor + (std_start - std_max_floor) * np.exp(-decay_lambda * time_passed_max)
            
        max_std = max(max_std, min_std + 1e-3)

        # Diversity scales with current exploration noise to maintain relative impact
        current_lam_d = self.lam_d * (min_std / 0.5) if ep >= self.warmup_episodes else 0.0

        # 2. PREPARATION LOOP (Reusing existing loop for RTG & Logging)
        all_rtg_lists = []

        returns_for_ranking = [sum(self.buffer.get_latest_trajectory(idx)["reward"]) for idx in range(self.N)]
        elite_idx_prep = np.argsort(returns_for_ranking)[-max(1, int(self.N * 0.10)):]

        with torch.no_grad():
            sum_actual_std = 0
            for i in range(len(self.actors)):
                # Save old policy weights
                self.old_actors[i].load_state_dict(self.actors[i].state_dict())
                # Clamp log_std to our dynamic exploration floor
                if i not in elite_idx_prep:
                    self.actors[i].log_std.clamp_(min=np.log(min_std), max=np.log(max_std))
                sum_actual_std += torch.exp(self.actors[i].log_std).mean().item()
                
                # OPTIMIZATION: Calculate RTGs here to get global stats without extra loop
                traj_rewards = self.buffer.get_latest_trajectory(i)["reward"]
                all_rtg_lists.append(bf.compute_returns_to_go(traj_rewards, self.gamma, self.device))

            self.std_history.append(sum_actual_std / self.N)
            self.target_min_std_history.append(min_std)
            self.target_max_std_history.append(max_std)

        # 3. GATHER METRICS & CLUSTERING
        phi, returns, features_np, all_pos_np = self._gather_metrics()
        labels, traj_lengths = self._cluster_states(features_np, all_pos_np)

        # 4. STATISTICAL CALIBRATION (Reward Scaling & Trauma Trigger)
        # Global RTG stats for the Trauma-Z-Score-Trigger
        flat_rtgs = torch.cat(all_rtg_lists)
        global_rtg_mean = flat_rtgs.mean().item()
        global_rtg_std = flat_rtgs.std().item()

        # EMA Reward Scaler (Relative signal strength of the environment)
        current_returns = np.array(returns)
        current_std = np.std(current_returns)
        if current_std > 1e-4:
            self.running_reward_std = 0.9 * self.running_reward_std + 0.1 * current_std
        reward_scale = self.running_reward_std # This keeps penalties in sync with rewards

        # 5. DYNAMIC TRAUMA MEMORY MANAGEMENT
        for center in self.trauma_centers:
            center['weight'] *= 0.9                     # 0.9 standard value

        # Forget trauma if its potential loss impact falls below 1/100 of reward scale
        forget_limit = reward_scale / 100.0              # Standard value: 1/10th of reward scale
        self.trauma_centers = [c for c in self.trauma_centers if c['weight'] > forget_limit]

        # Advantages with statistical baseline (using global RTG mean for Z-Score Trigger)
        advantages = self._compute_advantages(labels, traj_lengths, features_np, all_pos_np, 
                                              pop_mean=global_rtg_mean, reward_scale=reward_scale)

        # 6. POPULATION STRATEGY (Tier Ranking)
        sorted_indices = np.argsort(returns) 
        n_scouts = max(1, int(self.N * 0.10))
        n_elite = max(1, int(self.N * 0.10))
        n_mid = int((self.N - n_scouts - n_elite) / 2)
        
        scout_idx = sorted_indices[:n_scouts]
        reset_idx = sorted_indices[n_scouts : n_scouts + (self.N - n_scouts - n_mid - n_elite)]
        mid_idx = sorted_indices[-(n_elite + n_mid) : -n_elite]
        elite_idx = sorted_indices[-n_elite:]

        # Normalization and Grouping
        phi_norm = (phi - phi.mean(axis=0)) / (phi.std(axis=0) + 1e-8)
        groups = KMeans(n_clusters=min(self.K, len(self.actors)), n_init='auto').fit_predict(phi_norm)
        normalized_advantages = bf.normalize_advantages_by_group(advantages, groups, self.device)
        sigma_global = torch.cat(normalized_advantages).std() + 1e-8
        group_members = {g: np.where(groups == g)[0] for g in np.unique(groups)}

        # Pre-compute clipping bounds (epsilon)
        group_epsilons = {}
        for g in group_members:
            g_adv = torch.cat([advantages[j] for j in group_members[g]])
            group_epsilons[g] = self.epsilon * torch.clamp(g_adv.std() / sigma_global, min=1.0)

        # Pre-calculate distributions
        actor_features, all_mus = {}, {}
        for j in range(len(self.actors)):
            obs = torch.stack(self.buffer.get_latest_trajectory(j)["obs"]).to(self.device)
            feat = self.actors[j].forward_features(obs)
            actor_features[j], all_mus[j] = feat, self.actors[j].get_distribution(feat).mean.detach()

        # 7. OPTIMIZATION LOOP
        loss_stats = {"actor_loss": 0.0, "smooth_loss": 0.0, "div_loss": 0.0, "trauma_loss": 0.0}
        updated_agents_count = 0

        for i in range(len(self.actors)):
            if ep >= self.warmup_episodes and i in reset_idx: continue
            updated_agents_count += 1

            # Role-Specific Penalty Logic
            if i in elite_idx:
                role_lam_d, role_lam_t = 0.0, 0.0 # Elites explore without penalty
            elif i in scout_idx:
                role_lam_d, role_lam_t = current_lam_d * 30.0, self.lam_t # Extreme scouts
            else:
                role_lam_d, role_lam_t = current_lam_d, self.lam_t # Standard mid-tier/reset

            # Data prep
            traj = self.buffer.get_latest_trajectory(i)
            obs = torch.stack(traj["obs"]).to(self.device)
            actions = torch.stack(traj["action"]).to(self.device)
            old_log_probs = torch.stack(traj["log_probs"]).to(self.device)
            adv = normalized_advantages[i].to(self.device)

            feat = actor_features[i]
            dist = self.actors[i].get_distribution(feat)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            # PPO Core
            ratio = torch.exp(new_log_probs - old_log_probs)
            epsilon_i = group_epsilons[groups[i]]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - epsilon_i, 1 + epsilon_i) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # Balanced Penalties (scaled by Reward Volatility)
            l_smooth = torch.mean((feat[1:] - feat[:-1]) ** 2) if feat.shape[0] > 1 else 0
            l_div = self._compute_exp_diversity(i, all_mus, role_lam_d) / reward_scale
            l_trauma = role_lam_t * self._compute_trauma_penalty(feat) / reward_scale

            total_loss = actor_loss + (self.lam_s * l_smooth) + l_div + l_trauma

            # Step
            self.optimizers[i].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.optimizers[i].step()

            loss_stats["actor_loss"] += actor_loss.item()
            loss_stats["smooth_loss"] += l_smooth.item() if torch.is_tensor(l_smooth) else l_smooth
            loss_stats["div_loss"] += l_div.item() if torch.is_tensor(l_div) else l_div
            loss_stats["trauma_loss"] += l_trauma.item() if torch.is_tensor(l_trauma) else l_trauma

        # 8. ANNIHILATION & SPAWNING (Population Refactoring)
        if ep >= self.warmup_episodes:
            survivor_actors, survivor_old_actors, survivor_optimizers = [], [], []
            for idx in range(len(self.actors)):
                if idx not in reset_idx:
                    survivor_actors.append(self.actors[idx]); survivor_old_actors.append(self.old_actors[idx])
                    survivor_optimizers.append(self.optimizers[idx])
            
            for _ in range(len(reset_idx)):
                target_elite = np.random.choice(elite_idx)
                new_actor = bf.ContinuousActor(self.obs_space, self.action_space, self.hidden_dim).to(self.device)
                new_old_actor = bf.ContinuousActor(self.obs_space, self.action_space, self.hidden_dim).to(self.device)
                
                # DNA Mutated Cloning
                new_dict = {k: v + torch.randn_like(v) * 0.02 for k, v in self.actors[target_elite].state_dict().items()}
                new_actor.load_state_dict(new_dict); new_old_actor.load_state_dict(new_dict)
                
                survivor_actors.append(new_actor); survivor_old_actors.append(new_old_actor)
                survivor_optimizers.append(optim.Adam(new_actor.parameters(), lr=self.base_lr))
                
            self.actors, self.old_actors, self.optimizers = nn.ModuleList(survivor_actors), nn.ModuleList(survivor_old_actors), survivor_optimizers
            self.N = len(self.actors)

        # 9. FINALIZE
        elite_stats = self._update_reference_policy_mixture(returns)
        self.buffer = bf.PopulationBuffer(self.N) # Fresh buffer for potentially resized population

        if (ep + 1) % self.plot_interval == 0:
            self._plot_exploration_health()

        for key in loss_stats: loss_stats[key] /= max(1, updated_agents_count)
        print(f"[Update] Actor Loss: {loss_stats['actor_loss']:.4f}, Smooth: {loss_stats['smooth_loss']:.4f}, Div: {loss_stats['div_loss']:.4f}, Trauma: {loss_stats['trauma_loss']:.4f}, Elite Mean Return: {np.mean(elite_stats):.2f}")
        return {"actor": loss_stats["actor_loss"], "div": loss_stats["div_loss"], "trauma": loss_stats["trauma_loss"], "elite_mean": np.mean(elite_stats)}
    
    # =========================================================================
    # PLOTTING & UTILS
    # =========================================================================

    def _plot_spatial_only(self, pos_data, labels):
        """Plots the agent trajectories and overlaid trauma zones (ellipses) in 2D space."""
        os.makedirs('plots', exist_ok=True)
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        # Plot track data if available
        if hasattr(self, 'current_track_data') and self.current_track_data is not None:
            track_x = [p[0] for p in self.current_track_data]
            track_y = [p[1] for p in self.current_track_data]
            track_x.append(track_x[0]); track_y.append(track_y[0])
            
            plt.plot(track_x, track_y, color='darkgray', linewidth=35, alpha=0.5, label='Road')
            plt.plot(track_x, track_y, color='white', linewidth=2, linestyle='--', alpha=0.8)
        
        # Plot agent positions
        plt.scatter(pos_data[:, 0], pos_data[:, 1], c=labels, cmap='tab20', s=8, alpha=0.8, zorder=6)
        
        ep = getattr(self, 'current_episode', 0)
        plt.title(f"Spatial Cluster Distribution (Ep {ep})")
        plt.xlabel("X Position (World)")
        plt.ylabel("Y Position (World)")
        plt.grid(True, alpha=0.3)
        plt.axis('equal') 
        
        if hasattr(self, 'current_track_data') and self.current_track_data is not None:
            plt.legend()
            
        plt.savefig(f'plots/carracing_spatial_ep{ep}.png')
        plt.close()

    def _plot_clusters(self, data, labels):
        """Generates a PCA scatter plot of identified state clusters in the latent space."""
        plt.figure(figsize=(10, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab20', s=10)
        plt.title("Latent State Clusters (PCA)")
        plt.savefig('state_space_clustered.png')
        plt.close()

    def save_checkpoint(self, path, ep, eval_rewards):
        checkpoint = {
            'episode': ep,
            'eval_rewards': eval_rewards,
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
            return {'episode': 0, 'eval_rewards': []}

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