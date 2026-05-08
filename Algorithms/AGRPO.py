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
                 tau=0.5, lam_s=0.01, lam_d=0.00005, lam_t=0.005, gamma=0.99, dbscan_eps=0.4, 
                 TRAUMA_THRESHOLD=-20.0, warmup_episodes=100):
        
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
        self.warmup_episodes = warmup_episodes
        self.std_history = []
        
        # Trauma Management Parameters
        self.Trauma_Threshold = TRAUMA_THRESHOLD
        self.trauma_forgeting_threshold = self.Trauma_Threshold / 3.0
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
        subsample_idx = np.arange(0, len(flat_features), 20)
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
        
        # Generate diagnostic plots
        self._plot_clusters(full_pca if len(valid_labels) > 0 else pca_features_sub, labels)
        self._plot_spatial_only(np.concatenate(all_pos_np, axis=0), labels)
        
        return labels, traj_lengths

    # =========================================================================
    # ADVANTAGES & TRAUMA MANAGEMENT
    # =========================================================================

    def _compute_advantages(self, labels, traj_lengths, all_features_np, all_pos_np):
        """
        Calculates context-specific advantages using GPU vectorization and manages the Trauma memory.
        Trauma centers are recorded or reinforced if a context yields severe negative returns.
        """
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
        
        max_weight = 1.0 / max(1e-6, self.lam_t)
        labels_np = labels
        for c in np.unique(labels_np):
            # Check if cluster mean return crosses the severity threshold
            if c != -1 and cluster_means_dict.get(int(c), 0.0) < self.Trauma_Threshold:
                mask = (labels_np == c)
                trauma_points = flat_features[mask]

                if len(trauma_points) > 5:
                    mu_feat = torch.tensor(trauma_points.mean(axis=0), dtype=torch.float32).to(self.device)
                    sigma_feat = torch.tensor(trauma_points.std(axis=0) + 1e-4, dtype=torch.float32).to(self.device)
                    severity = abs(cluster_means_dict[int(c)])
                    
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
                    for existing in self.trauma_centers:
                        dist_sq = torch.sum(((mu_feat - existing['mu']) / existing['sigma']) ** 2) / dim
                        
                        if dist_sq < 1.0:
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
                            break
                    
                    if not merged:
                        self.trauma_centers.append(new_trauma_data)

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
            
            gauss_penalty = torch.exp(-normalized_dist / (2 * (bandwidth ** 2)))
            total_penalty += (gauss_penalty * weight).mean()

        return total_penalty

    def _compute_exp_diversity(self, i, all_mus, current_lam_d):
        """Calculates an exponential penalty if agent policies become too similar (collapsing)."""
        if current_lam_d <= 0:
            return torch.tensor(0.0).to(self.device)

        penalties = []
        mu_i = self.actors[i].get_distribution(self.actors[i].forward_features(
            torch.stack(self.buffer.get_latest_trajectory(i)["obs"]).to(self.device)
        )).mean

        exp_scale = 5.0
        for j in range(len(self.actors)):
            if i == j: continue

            min_len = min(mu_i.shape[0], all_mus[j].shape[0])
            step_sim = torch.cosine_similarity(mu_i[:min_len], all_mus[j][:min_len], dim=-1).mean()
            penalty = torch.exp(exp_scale * (step_sim - self.tau)) - 1.0
            penalties.append(torch.clamp(penalty, min=0.0))

        return torch.stack(penalties).mean() if penalties else torch.tensor(0.0).to(self.device)

    def _plot_exploration_health(self):
        """
        Visualizes the health of exploration. 
        Blue: What the agents actually do.
        Red (Dashed): The minimum safety net we provide.
        """
        import os
        os.makedirs('plots', exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.std_history, label='Actual Policy STD (Avg)', color='#1f77b4', linewidth=2)
        plt.plot(self.target_min_std_history, label='Min Std Floor (Safety Net)', color='#d62728', linestyle='--')
        
        plt.fill_between(range(len(self.std_history)), self.target_min_std_history, self.std_history, 
                         where=(np.array(self.std_history) > np.array(self.target_min_std_history)),
                         color='#1f77b4', alpha=0.1, label='Learned Exploration Buffer')

        plt.title(f"Exploration Health Analysis (Ep {getattr(self, 'current_episode', 0)})")
        plt.xlabel("Update Steps")
        plt.ylabel("Standard Deviation")
        plt.yscale('log') # Log scale is crucial to see the collapse at small values
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.savefig('plots/exploration_health.png')
        plt.close()

    # =========================================================================
    # MAIN OPTIMIZATION
    # =========================================================================

    def update(self):
        '''
        Core optimization loop implementing a Tiered Population Strategy:
        1. Elite (Top 10%): Pure reward optimization (No Diversity penalty).
        2. Mid-Tier (40%): Standard training (Reward + Diversity + Trauma).
        3. Reset-Tier (40%): Standard training, but periodically cloned from Elite + Noise.
        4. Scouts (Bottom 10%): Maximum diversity exploration to find new paths.
        '''
        ep = getattr(self, 'current_episode', 0)
        
        std_start = 0.5
        std_warmup_end = 0.2
        std_final_floor = 0.05

        if ep < self.warmup_episodes:
            min_std = max(std_warmup_end, std_start - (ep/self.warmup_episodes) * 0.3) 
        else:
            decay_lambda = 0.006931
            time_passed = ep - self.warmup_episodes
            
            decay_range = std_warmup_end - std_final_floor
            min_std = std_final_floor + decay_range * np.exp(-decay_lambda * time_passed)
        
        current_lam_d = self.lam_d * (min_std / 0.5) if ep >= self.warmup_episodes else 0.0

        with torch.no_grad():
            # What the actors actually have learned/feel right now
            actual_stds = torch.stack([torch.exp(a.log_std).mean() for a in self.actors])
            avg_actual_std = actual_stds.mean().item()
            
            self.std_history.append(avg_actual_std)
            self.target_min_std_history.append(min_std)

        for i in range(len(self.actors)):
            self.old_actors[i].load_state_dict(self.actors[i].state_dict())
            with torch.no_grad():
                # We allow the network to be MORE noisy than min_std, but not less.
                self.actors[i].log_std.clamp_(min=np.log(min_std))

        loss_stats = {"actor_loss": 0.0, "smooth_loss": 0.0, "div_loss": 0.0, "trauma_loss": 0.0}

        # 1. Manage Memory: Decay old traumas and remove forgotten ones
        for center in self.trauma_centers:
            center['weight'] *= 0.8
        self.trauma_centers = [c for c in self.trauma_centers if c['weight'] > self.trauma_forgeting_threshold]

        # Prevent std collapse
        for i in range(len(self.actors)):
            self.old_actors[i].load_state_dict(self.actors[i].state_dict())
            self.actors[i].log_std.data = torch.clamp(self.actors[i].log_std.data, min=np.log(min_std))

        # 2. Gather data and calculate advantages
        phi, returns, features_np, all_pos_np = self._gather_metrics()
        labels, traj_lengths = self._cluster_states(features_np, all_pos_np)
        advantages = self._compute_advantages(labels, traj_lengths, features_np, all_pos_np)

        # 3. POPULATION STRATEGY: Rank agents based on returns
        sorted_indices = np.argsort(returns) # 0 is worst, N-1 is best
        
        n_scouts = max(1, int(self.N * 0.10))
        n_elite = max(1, int(self.N * 0.10))
        n_mid = int((self.N - n_scouts - n_elite) / 2)
        n_reset = self.N - n_scouts - n_elite - n_mid

        # Assign indices to tiers
        scout_idx = sorted_indices[:n_scouts]                     # Bottom performers become scouts
        reset_idx = sorted_indices[n_scouts : n_scouts+n_reset]   # Lower middle
        mid_idx = sorted_indices[n_scouts+n_reset : -n_elite]     # Upper middle
        elite_idx = sorted_indices[-n_elite:]                     # Top performers

        # Group agents by performance profile (phi) to normalize advantages relatively
        phi_norm = (phi - phi.mean(axis=0)) / (phi.std(axis=0) + 1e-8)
        groups = KMeans(n_clusters=min(self.K, len(self.actors)), n_init='auto').fit_predict(phi_norm)
        normalized_advantages = bf.normalize_advantages_by_group(advantages, groups, self.device)

        sigma_global = torch.cat(normalized_advantages).std() + 1e-8
        group_members = {g: np.where(groups == g)[0] for g in np.unique(groups)}

        # Pre-calculate distributions for diversity comparisons
        actor_features, all_mus = {}, {}
        for j in range(len(self.actors)):
            obs = torch.stack(self.buffer.get_latest_trajectory(j)["obs"]).to(self.device)
            feat = self.actors[j].forward_features(obs)
            actor_features[j] = feat
            all_mus[j] = self.actors[j].get_distribution(feat).mean.detach()

        # 4. Optimize each actor individually based on their ROLE
        # ==========================================
        warmup_episodes = self.warmup_episodes 
        updated_agents_count = 0

        for i in range(len(self.actors)):
            # EFFICIENCY FIX: Skip backpropagation for agents that will be annihilated anyway.
            # Their trajectory data was already used for clustering and baseline calculations.
            if ep >= warmup_episodes and i in reset_idx:
                continue
                
            updated_agents_count += 1

            if i in elite_idx:
                # Elites are "Fearless Pioneers": No trauma, no diversity pressure
                role_lam_d = 0.0
                role_lam_t = 0.0
            elif i in scout_idx:
                # Scouts are "Explorers": Normal trauma, but massive diversity pressure
                role_lam_d = current_lam_d * 10.0
                role_lam_t = self.lam_t
            elif i in mid_idx:
                # Mid-Tier are "Stable Performers": Normal trauma, normal diversity
                role_lam_d = current_lam_d
                role_lam_t = self.lam_t
            else:
                # Fallback (mostly for warmup phase before reset logic kicks in)
                role_lam_d = current_lam_d
                role_lam_t = self.lam_t 

            traj = self.buffer.get_latest_trajectory(i)
            obs = torch.stack(traj["obs"]).to(self.device)
            actions = torch.stack(traj["action"]).to(self.device)
            old_log_probs = torch.stack(traj["log_probs"]).to(self.device)
            adv = normalized_advantages[i].to(self.device)

            # Adaptive clipping bounds based on group variance
            g_adv = torch.cat([advantages[j] for j in group_members[groups[i]]])
            epsilon_i = self.epsilon * torch.clamp(g_adv.std() / sigma_global, min=1.0)

            feat = actor_features[i]
            dist = self.actors[i].get_distribution(feat)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            # PPO Clipping objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - epsilon_i, 1 + epsilon_i) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            # Assign Role-Specific Diversity Multipliers
            if i in elite_idx:
                role_lam_d = 0.0                  # Elite: Pure reward, ignore diversity
            elif i in scout_idx:
                role_lam_d = current_lam_d * 10.0 # Scouts: Heavy diversity to force exploration
            else:
                role_lam_d = current_lam_d        # Mid: Normal diversity

            # Auxiliary penalties
            l_smooth = torch.mean((feat[1:] - feat[:-1]) ** 2) if feat.shape[0] > 1 else 0
            l_div = self._compute_exp_diversity(i, all_mus, role_lam_d)
            l_trauma = role_lam_t * self._compute_trauma_penalty(feat)

            total_loss = actor_loss + (self.lam_s * l_smooth) + l_div + l_trauma

            # Gradient Step
            self.optimizers[i].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.optimizers[i].step()

            # Track stats
            loss_stats["actor_loss"] += actor_loss.item()
            loss_stats["smooth_loss"] += l_smooth.item() if torch.is_tensor(l_smooth) else l_smooth
            loss_stats["div_loss"] += l_div.item() if torch.is_tensor(l_div) else l_div
            loss_stats["trauma_loss"] += l_trauma.item() if torch.is_tensor(l_trauma) else l_trauma


        # ==========================================
        # 5. ANNIHILATION & SPAWNING (Dynamic Population Architecture)
        # ==========================================
        if ep >= warmup_episodes:
            # 5a. Identify Survivors
            survivor_actors = []
            survivor_old_actors = []
            survivor_optimizers = []
            
            for idx in range(len(self.actors)):
                if idx not in reset_idx:
                    survivor_actors.append(self.actors[idx])
                    survivor_old_actors.append(self.old_actors[idx])
                    survivor_optimizers.append(self.optimizers[idx])
            
            # 5b. Spawn New Agents (Clones with Noise)
            num_to_spawn = len(reset_idx) 
            
            for _ in range(num_to_spawn):
                target_elite = np.random.choice(elite_idx)
                
                # Birth entirely new neural networks
                new_actor = bf.ContinuousActor(self.obs_space, self.action_space, self.hidden_dim).to(self.device)
                new_old_actor = bf.ContinuousActor(self.obs_space, self.action_space, self.hidden_dim).to(self.device)
                
                # Extract elite DNA and mutate
                elite_state_dict = self.actors[target_elite].state_dict()
                new_state_dict = {}
                for name, param in elite_state_dict.items():
                    noise = torch.randn_like(param) * 0.02 
                    new_state_dict[name] = param + noise
                    
                new_actor.load_state_dict(new_state_dict)
                new_old_actor.load_state_dict(new_state_dict)
                
                # Give the newborn a fresh optimizer
                new_opt = optim.Adam(new_actor.parameters(), lr=self.base_lr)
                
                # Add to the new generation
                survivor_actors.append(new_actor)
                survivor_old_actors.append(new_old_actor)
                survivor_optimizers.append(new_opt)
                
            # 5c. Overwrite population with the new generation
            self.actors = nn.ModuleList(survivor_actors)
            self.old_actors = nn.ModuleList(survivor_old_actors)
            self.optimizers = survivor_optimizers
            self.N = len(self.actors)

        # ==========================================
        # 6. Finalize Update
        # ==========================================
        elite_stats = self._update_reference_policy_mixture(returns)
        
        # Re-instantiate the buffer to match the current population size
        self.buffer = bf.PopulationBuffer(self.N)

        for key in loss_stats: 
            # Divide by the actual number of trained agents to get accurate monitoring metrics
            loss_stats[key] /= max(1, updated_agents_count) 

        if ep % 20 == 0:
            self._plot_exploration_health()

        return {
            "actor": loss_stats["actor_loss"],
            "div": loss_stats["div_loss"],
            "trauma": loss_stats["trauma_loss"],
            "elite_mean": np.mean(elite_stats)
        }

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

        ckpt = torch.load(path, map_location=self.device)

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