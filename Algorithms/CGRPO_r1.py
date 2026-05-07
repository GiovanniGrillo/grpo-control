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
                 tau=0.5, lam_s=0.01, lam_d=0.0001, gamma=0.99, dbscan_eps=0.4):
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
                self.update()
                self.current_policy_idx = 0

    def _update_reference_policy_mixture(self, all_returns):
        """
        Updates the reference policy to represent the top K performing agents.
        Acts as a strong baseline/teacher for the entire population.
        """
        num_top = max(1, int(len(self.actors) // 10))
        top_indices = np.argsort(all_returns)[-num_top:]

        with torch.no_grad():
            avg_state_dict = {}
            for name in self.actors[0].state_dict():
                avg_state_dict[name] = torch.stack(
                    [self.actors[idx].state_dict()[name] for idx in top_indices]
                ).mean(dim=0)
            self.ref_actor.load_state_dict(avg_state_dict) 

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

    # def _cluster_states(self, all_features_np):
        flat_features = np.concatenate(all_features_np, axis=0)
        scaled = self.scaler.fit_transform(flat_features)

        pca_dims = 20
        print(f"pca_dims={pca_dims} | flat_features.shape={flat_features.shape} | scaled.shape={scaled.shape}")
        pca_comps = min(pca_dims, flat_features.shape[1])
        if self.pca is None or self.pca.n_components != pca_comps:
            self.pca = PCA(n_components=pca_comps)
        pca_features = self.pca.fit_transform(scaled)

        pca_features = np.nan_to_num(pca_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

        total_points = pca_features.shape[0]
        c_size = int(max(20, total_points * 0.005))
        min_samples = int(max(5, c_size / 10))
        print(f"DBSCAN params: c_size={c_size}, min_samples={min_samples}, dbscan_eps={self.dbscan_eps}")

        dbscan = HDBSCAN(min_cluster_size=c_size, min_samples=min_samples, cluster_selection_epsilon=self.dbscan_eps, core_dist_n_jobs=1)
        labels = dbscan.fit_predict(pca_features)

        # n_state_clusters = 10  # Fixed number of state clusters
        # kmeans_state = KMeans(n_clusters=n_state_clusters, n_init='auto')
        # labels = kmeans_state.fit_predict(pca_features)

        self._plot_clusters(pca_features, labels)
        return labels, [len(f) for f in all_features_np]

    def _cluster_states(self, all_features_np, all_pos_np):
        """Clustering states into contexts using a 3-step process: 
        1) PCA for dimensionality reduction, 
        2) HDBSCAN for core cluster detection, 
        3) KNN for assigning all points to clusters."""
        flat_features = np.concatenate(all_features_np, axis=0)
        
        subsample_idx = np.arange(0, len(flat_features), 10)
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

        num_clusters = len(np.unique(labels))
        print(f"[Clustering] HDBSCAN found {num_clusters} core clusters in subsample.")
        print(f"[Clustering] KNN assigned all {len(labels)} points to existing clusters (0 Noise).")

        self._plot_clusters(full_pca if len(valid_labels) > 0 else pca_features_sub, labels)

        flat_pos = np.concatenate(all_pos_np, axis=0)
        self._plot_spatial_only(flat_pos, labels)
        
        return labels, [len(f) for f in all_features_np]
    
    
    def _plot_spatial_only(self, pos_data, labels):
        import os
        os.makedirs('plots', exist_ok=True)
        
        plt.figure(figsize=(10, 10))
        
        if hasattr(self, 'current_track_data') and self.current_track_data is not None:
            track_x = [p[0] for p in self.current_track_data]
            track_y = [p[1] for p in self.current_track_data]
            
            track_x.append(track_x[0])
            track_y.append(track_y[0])
            
            plt.plot(track_x, track_y, color='darkgray', linewidth=35, alpha=0.5, label='Road')
            plt.plot(track_x, track_y, color='white', linewidth=2, linestyle='--', alpha=0.8)
        # ----------------------------------

        plt.scatter(pos_data[:, 0], pos_data[:, 1], c=labels, cmap='tab20', s=8, alpha=0.8, zorder=5)
        
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

    def _compute_advantages(self, labels, traj_lengths):
        all_returns_to_go = []
        for i in range(len(self.actors)):
            rewards = self.buffer.get_latest_trajectory(i)["reward"]
            rtg = bf.compute_returns_to_go(rewards, self.gamma, self.device)
            all_returns_to_go.append(rtg)

        flat_returns = torch.cat(all_returns_to_go)
        labels_t = torch.from_numpy(labels).to(self.device)

        unique_labels = torch.unique(labels_t)
        cluster_means = {}

        global_mean = flat_returns.mean()
        for c in unique_labels:
            c_item = c.item()
            if c_item != -1:
                cluster_means[c_item] = flat_returns[labels_t == c].mean()

        cluster_means[-1] = global_mean

        advantages = []
        start = 0
        for i, length in enumerate(traj_lengths):
            policy_labels = labels[start:start+length]
            policy_labels_t = labels_t[start:start+length]
            baseline = torch.tensor([cluster_means.get(int(l), cluster_means.get(-1, 0.0)) for l in policy_labels], dtype=torch.float32).to(self.device)
            adv = all_returns_to_go[i] - baseline
            advantages.append(adv)
            start += length
        return advantages

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

        for i in range(len(self.actors)):
            self.old_actors[i].load_state_dict(self.actors[i].state_dict())
            self.actors[i].log_std.data = torch.clamp(self.actors[i].log_std.data, min=np.log(min_std))

        phi, returns, features_np, all_pos_np = self._gather_metrics()
        labels, traj_lengths = self._cluster_states(features_np, all_pos_np)

        advantages = self._compute_advantages(labels, traj_lengths)

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

            total_loss = actor_loss + self.lam_s * l_smooth + current_lam_d * l_div

            self.optimizers[i].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
            self.optimizers[i].step()

        self._update_reference_policy_mixture(returns)
        self.buffer.clear_buffer()

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
            'current_episodes': self.buffer.current_episodes if hasattr(self.buffer, 'current_episodes') else None
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

        return ckpt