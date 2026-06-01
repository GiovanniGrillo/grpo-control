import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import utils as bf
import collections

class AGRPO:
    """
    Advanced Group Relative Policy Optimization (AGRPO).
    """
    def __init__(self, env, seed=42, hidden_dim=512, lr=5e-4, N=50, K=3, epsilon=0.4,
                 tau=0.6, lam_s=0.005, lam_d=0.01, gamma=0.99, dbscan_eps=0.2,
                 warmup_episodes=100): #tau = 0.5

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
        self.plot_interval = N  

        self.running_reward_std = 1.0 
        self.warmup_episodes = warmup_episodes
        self.std_history = []
        self.target_min_std_history = []
        self.target_max_std_history = []
        
        self.return_mean_history = collections.deque(maxlen=20)
        self.return_std_history = collections.deque(maxlen=20)

        # Off-Policy Archive for stable HDBSCAN clustering and robust baselines
        self.history_buffer = collections.deque(maxlen=10) 

        self.current_track_data = None

        self.obs_space = env.observation_space
        self.action_space = env.action_space
        self.hidden_dim = hidden_dim
        self.base_lr = lr

        self.actors = nn.ModuleList([bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device) for _ in range(N)])
        self.old_actors = nn.ModuleList([bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device) for _ in range(N)])

        self.elite_return_history = collections.deque(maxlen=10) 
        self.stagnant_updates = 0

        self.num_elites = max(1, int(self.N // 10))
        self.ref_actors = nn.ModuleList([bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device) for _ in range(self.num_elites)])
        for ref_actor in self.ref_actors:
            ref_actor.load_state_dict(self.actors[0].state_dict())
        self.ref_actor = self.ref_actors[0]

        for i in range(self.N):
            self.old_actors[i].load_state_dict(self.actors[i].state_dict())

        self.optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.buffer = bf.PopulationBuffer(N)
        self.current_policy_idx = 0
        self.updated = False

        self.scaler = StandardScaler()
        self.action_scaler = StandardScaler()
        self.pca = None

    def select_action(self, state, evaluate=False):
        obs_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                actions = torch.stack([actor.get_deterministic_action(obs_t) for actor in self.ref_actors])
                action_t = actions.mean(dim=0)
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
                self.updated = True
                return stats
        return None

    def consume_update_flag(self):
        if self.updated:
            self.updated = False
            return True
        return False

    def _update_ensemble(self, top_indices):
        with torch.no_grad():
            for i, idx in enumerate(top_indices):
                if i < len(self.ref_actors):
                    self.ref_actors[i].load_state_dict(self.old_actors[idx].state_dict())
            self.ref_actor = self.ref_actors[-1]

    def _update_reference_policy_mixture(self, all_returns):
        num_top = max(1, int(len(self.actors) // 10))
        top_indices = np.argsort(all_returns)[-num_top:]
        self.last_elite_rewards = [all_returns[i] for i in top_indices]
        self.last_elite_indices = top_indices

        best_idx = top_indices[-1]
        best_return = all_returns[best_idx]

        if len(self.elite_return_history) < 5:
            self._update_ensemble(top_indices)
            self.elite_return_history.append(best_return)
            return self.last_elite_rewards

        elite_mean = np.mean(self.elite_return_history)
        elite_std = np.std(self.elite_return_history)
        dynamic_elite_threshold = elite_mean - (0.5 * elite_std)

        if best_return >= dynamic_elite_threshold:
            self._update_ensemble(top_indices)
            self.elite_return_history.append(best_return)
            self.stagnant_updates = 0
        else:
            self.stagnant_updates += 1

            if self.stagnant_updates > 5:
                self._update_ensemble(top_indices)
                self.elite_return_history.append(best_return)
                self.stagnant_updates = 0

        return self.last_elite_rewards

    def _gather_metrics(self):
        all_phi, all_features_np, all_pos_np, all_actions_np = [], [], [], []

        for i in range(len(self.actors)):
            traj = self.buffer.get_latest_trajectory(i)
            obs = torch.stack(traj["obs"]).to(self.device)
            actions = torch.stack(traj["action"]).to(self.device)

            if actions.dim() == 1:
                actions = actions.unsqueeze(-1)

            with torch.no_grad():
                feat = self.actors[i].forward_features(obs)
                dist = self.actors[i].get_distribution(feat)
                
                feat_mean = feat.mean(dim=0).cpu().numpy().flatten()
                feat_std = feat.std(dim=0).cpu().numpy().flatten()
                
                u = torch.atanh(torch.clamp(actions, -0.999999, 0.999999))
                squash_correction = torch.log(1.0 - actions.pow(2) + 1e-6).sum(dim=-1)
                
                log_prob_i = dist.log_prob(u).sum(dim=-1) - squash_correction
                
                elite_lps = []
                for ref_act in self.ref_actors:
                    ref_feat = ref_act.forward_features(obs)
                    ref_lp = ref_act.get_distribution(ref_feat).log_prob(u).sum(dim=-1) - squash_correction
                    elite_lps.append(ref_lp)
                
                stacked_lps = torch.stack(elite_lps)
                log_prob_ref = torch.logsumexp(stacked_lps, dim=0) - np.log(len(self.ref_actors))
                
                kl_div = (log_prob_i - log_prob_ref).mean().item()
                
                reward_sum = sum(traj["reward"])
                phi_i = np.concatenate([
                    feat_mean, 
                    feat_std, 
                    np.array([reward_sum / len(traj["reward"]), kl_div])
                ])

            all_phi.append(phi_i)
            all_features_np.append(torch.stack(traj["feature"]).cpu().numpy())
            all_pos_np.append(np.array(traj["pos"]))
            all_actions_np.append(actions.cpu().numpy())

        return np.array(all_phi), all_features_np, all_pos_np, all_actions_np
    
    def _cluster_states(self, flat_features, flat_actions, flat_rtg):
        """
        Clusters across the entire historical off-policy archive to build stable manifolds.
        Takes flattened arrays representing multiple epochs of experience.
        """
        scaled_features = self.scaler.fit_transform(flat_features)
        pca_dims = 30
        pca_comps = min(pca_dims, flat_features.shape[1])
        if self.pca is None or self.pca.n_components != pca_comps:
            self.pca = PCA(n_components=pca_comps)
        pca_features = np.nan_to_num(self.pca.fit_transform(scaled_features), nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Scale Actions
        scaled_actions = np.nan_to_num(self.action_scaler.fit_transform(flat_actions), nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Scale RTG (Standardize to match state/action scale)
        scaled_returns = self.scaler.fit_transform(flat_rtg.reshape(-1, 1))

        # 4. Feature Injection: State + Action + FUTURE RETURN
        combined = np.concatenate([
            pca_features, 
            scaled_actions * 2.0,  
            scaled_returns * 2.0   # Gewichtung des langfristigen Erfolgs!
        ], axis=1)

        # 5. HDBSCAN Clustering
        c_size = int(np.clip(len(combined) * 0.005, 10, 200))
        min_samples = int(max(5, c_size / 2))
        
        dbscan = HDBSCAN(min_cluster_size=c_size, min_samples=min_samples, 
                         cluster_selection_epsilon=self.dbscan_eps, core_dist_n_jobs=1)
        
        labels = dbscan.fit_predict(combined)
        
        # Logging applies to the entire historical manifold
        self.last_num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.last_noise_ratio = (labels == -1).sum() / len(labels) if len(labels) > 0 else 0.0

        return labels
    
    def _compute_advantages(self, current_traj_lengths, full_labels, flat_rtg):
        """
        Calculates baselines from the entire historical archive, but only evaluates 
        advantages for the current generation to ensure valid On-Policy gradients.
        """
        flat_rtg_t = torch.from_numpy(flat_rtg).to(self.device).float()
        labels_t = torch.from_numpy(full_labels).to(self.device).long()

        # 1. Build robust baselines using ALL historical data in the clusters
        max_label = int(full_labels.max())
        means_vec = torch.zeros(max_label + 2, device=self.device)
        means_vec[0] = flat_rtg_t.mean() # Global fallback for noise (-1)
        
        for c in torch.unique(labels_t):
            c_val = int(c.item())
            if c_val != -1:
                m = flat_rtg_t[labels_t == c_val].mean()
                means_vec[c_val + 1] = m

        # 2. Extract labels and returns ONLY for the current generation
        # The current generation is appended at the very end of the flattened arrays
        current_total_steps = sum(current_traj_lengths)
        current_labels_t = labels_t[-current_total_steps:]
        current_rtg_t = flat_rtg_t[-current_total_steps:]

        # 3. Calculate advantages for the current generation against historical baselines
        flat_baselines = means_vec[current_labels_t + 1]
        flat_advantages = current_rtg_t - flat_baselines
        
        advantages = list(torch.split(flat_advantages, current_traj_lengths))
        return advantages
    
    def _compute_exp_diversity(self, i, current_mu, all_mus, current_lam_d):
        if current_lam_d <= 0:
            return torch.tensor(0.0).to(self.device)
        exp_scale = 5.0
        min_len = min(mu.shape[0] for mu in all_mus.values())
        if min_len == 0:
            return torch.tensor(0.0).to(self.device)
        
        # Use the un-detached current_mu to allow gradient flow for the active agent
        mu_i_trunc = current_mu[:min_len]
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

    def _set_action_std_for_role(self, actor, role, ep, min_std_mids, max_std):
        actor.log_std.requires_grad = False
        if ep < self.warmup_episodes:
            target_std = 0.6
        elif role == "scout":
            target_std = min(0.6, max_std)
        elif role == "elite":
            target_std = min_std_mids / 3.0
        else:
            target_std = min_std_mids

        actor.log_std.data.fill_(np.log(target_std))

    def update(self):
        ep = getattr(self, 'current_episode', 0)

        std_start, std_warmup_end, std_final_floor = 0.8, 0.6, 0.05
        if ep < self.warmup_episodes:
            mid_std = max(std_warmup_end, std_start - (ep/self.warmup_episodes) * (std_start - std_warmup_end))
        else:
            decay_lambda = 0.069315
            time_passed = ep - self.warmup_episodes
            mid_std = std_final_floor + (std_warmup_end - std_final_floor) * np.exp(-decay_lambda * time_passed)

        max_warmup = 2 * self.warmup_episodes
        std_max_floor = std_final_floor + 0.45

        if ep < max_warmup:
            max_std = std_start
        else:
            time_passed_max = ep - max_warmup
            max_std = std_max_floor + (std_start - std_max_floor) * np.exp(-decay_lambda * time_passed_max)

        # max_std = max(max_std, mid_std + 1e-3)
        current_lam_d = self.lam_d * (mid_std / 0.5) if ep >= self.warmup_episodes else 0.0

        all_rtg_lists = []
        returns_for_ranking = [sum(self.buffer.get_latest_trajectory(idx)["reward"]) for idx in range(self.N)]
        elite_stats = self._update_reference_policy_mixture(returns_for_ranking)

        sorted_indices = np.argsort(returns_for_ranking)
        n_scouts = max(1, int(self.N * 0.10))
        n_elite = max(1, int(self.N * 0.10))

        elite_idx = sorted_indices[-n_elite:]
        scout_idx = sorted_indices[:n_scouts]
        n_mid = int((self.N - n_scouts - n_elite) / 2)

        reset_idx = sorted_indices[n_scouts : n_scouts + (self.N - n_scouts - n_mid - n_elite)]
        mid_idx = sorted_indices[-(n_elite + n_mid) : -n_elite]

        with torch.no_grad():
            sum_actual_std = 0
            for i in range(len(self.actors)):
                self.old_actors[i].load_state_dict(self.actors[i].state_dict())

                if i in scout_idx:
                    self._set_action_std_for_role(self.actors[i], "scout", ep, mid_std, max_std)
                elif i in elite_idx:
                    self._set_action_std_for_role(self.actors[i], "elite", ep, mid_std, max_std)
                else:
                    self._set_action_std_for_role(self.actors[i], "mid", ep, mid_std, max_std)

                sum_actual_std += torch.exp(self.actors[i].log_std).mean().item()
                traj_rewards = self.buffer.get_latest_trajectory(i)["reward"]
                all_rtg_lists.append(bf.compute_returns_to_go(traj_rewards, self.gamma, self.device))

            self.std_history.append(sum_actual_std / self.N)
            self.target_min_std_history.append(mid_std)
            self.target_max_std_history.append(max_std)

        ################################################################################
        # 3. METRIC EXTRACTION & CLUSTERING
        ################################################################################
        phi, features_np, all_pos_np, all_actions_np = self._gather_metrics()
        
        # Calculate Returns-To-Go for current generation
        current_rtgs = []
        for i in range(len(features_np)):
            rewards = self.buffer.get_latest_trajectory(i)["reward"]
            rtg = bf.compute_returns_to_go(rewards, self.gamma, self.device).cpu().numpy()
            current_rtgs.append(rtg)

        # Store the current generation's raw behavior in the Off-Policy Archive
        self.history_buffer.append({
            'features': features_np,
            'actions': all_actions_np,
            'rtg': current_rtgs
        })

        # Flatten the entire archive for stabilized historical clustering
        archived_features, archived_actions, archived_rtgs = [], [], []
        for hist in self.history_buffer:
            archived_features.extend(hist['features'])
            archived_actions.extend(hist['actions'])
            archived_rtgs.extend(hist['rtg'])

        flat_features = np.concatenate(archived_features, axis=0)
        flat_actions = np.concatenate(archived_actions, axis=0)
        flat_rtg = np.concatenate(archived_rtgs, axis=0)

        # Cluster across time
        full_labels = self._cluster_states(flat_features, flat_actions, flat_rtg)

        current_returns = np.array(returns_for_ranking)
        self.return_mean_history.append(np.mean(current_returns))
        self.return_std_history.append(np.std(current_returns))

        current_std = np.std(current_returns)
        if current_std > 1e-4:
            self.running_reward_std = 0.9 * self.running_reward_std + 0.1 * current_std
        reward_scale = max(1.0, self.running_reward_std)

        ################################################################################
        # 4. ADVANTAGE NORMALIZATION & PRE-COMPUTATION
        ################################################################################
        traj_lengths = [len(f) for f in features_np]
        advantages = self._compute_advantages(traj_lengths, full_labels, flat_rtg)

        phi_norm = (phi - phi.mean(axis=0)) / (phi.std(axis=0) + 1e-8)
        current_K = max(1, int(self.N / 12))                                                                            #15
        groups = KMeans(n_clusters=min(current_K, len(self.actors)), n_init='auto').fit_predict(phi_norm)
        normalized_advantages = bf.normalize_advantages_by_group(advantages, groups, self.device)
        sigma_global = torch.cat(normalized_advantages).std() + 1e-8
        group_members = {g: np.where(groups == g)[0] for g in np.unique(groups)}

        group_epsilons = {}
        for g in group_members:
            g_adv = torch.cat([advantages[j] for j in group_members[g]])
            group_epsilons[g] = self.epsilon * torch.clamp(g_adv.std() / sigma_global, min=1.0)

        all_mus = {}
        with torch.no_grad(): 
            for j in range(len(self.actors)):
                obs = torch.stack(self.buffer.get_latest_trajectory(j)["obs"]).to(self.device)
                feat = self.actors[j].forward_features(obs)
                all_mus[j] = self.actors[j].get_distribution(feat).mean.detach()
                
            cached_trajectories = {}
            for i in range(len(self.actors)):
                traj = self.buffer.get_latest_trajectory(i)
                cached_trajectories[i] = {
                    'obs': torch.stack(traj["obs"]).cpu(),
                    'actions': torch.stack(traj["action"]).cpu(),
                    'old_log_probs': torch.stack(traj["log_probs"]).cpu(),
                    'adv': normalized_advantages[i].cpu()
                }
        torch.cuda.empty_cache()

        ################################################################################
        # 5. PPO REPLAY: MULTIPLE OPTIMIZATION EPOCHS (BATCHED FOR GPU EFFICIENCY)
        ################################################################################
        loss_stats = {"actor_loss": 0.0, "smooth_loss": 0.0, "div_loss": 0.0, "ppo_ratio": 0.0, "grad_norm": 0.0}
        updated_agents_count = 0
        PPO_EPOCHS = 5
        BATCH_SIZE = 10
        div_violators_indices = []

        for batch_start in range(0, len(self.actors), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(self.actors))
            batch_indices = list(range(batch_start, batch_end))

            batch_cached = {}
            for i in batch_indices:
                batch_cached[i] = {
                    'obs': cached_trajectories[i]['obs'].to(self.device),
                    'actions': cached_trajectories[i]['actions'].to(self.device),
                    'old_log_probs': cached_trajectories[i]['old_log_probs'].to(self.device),
                    'adv': cached_trajectories[i]['adv'].to(self.device)
                }

            for epoch in range(PPO_EPOCHS):
                epoch_violators = []
                for i in batch_indices:
                    if ep >= self.warmup_episodes and i in reset_idx: continue
                    if epoch == 0: updated_agents_count += 1

                    if i in elite_idx:
                        role_lam_d = 0.0
                    elif i in scout_idx:
                        role_lam_d = current_lam_d * 10.0
                    else:
                        role_lam_d = current_lam_d

                    cached = batch_cached[i]
                    obs = cached['obs']
                    actions = cached['actions']
                    old_log_probs = cached['old_log_probs']
                    adv = cached['adv']

                    feat = self.actors[i].forward_features(obs)
                    dist = self.actors[i].get_distribution(feat)
                    
                    u = torch.atanh(torch.clamp(actions, -0.999999, 0.999999))
                    squash_correction = torch.log(1.0 - actions.pow(2) + 1e-6).sum(dim=-1)
                    new_log_probs = dist.log_prob(u).sum(dim=-1) - squash_correction

                    ratio = torch.exp(new_log_probs - old_log_probs)
                    loss_stats["ppo_ratio"] += ratio.mean().item() / PPO_EPOCHS

                    epsilon_i = group_epsilons[groups[i]]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - epsilon_i, 1 + epsilon_i) * adv
                    actor_loss = -torch.min(surr1, surr2).mean()

                    l_smooth = torch.mean((feat[1:] - feat[:-1]) ** 2) if feat.shape[0] > 1 else torch.tensor(0.0).to(self.device)
                    l_div = self._compute_exp_diversity(i, dist.mean, all_mus, role_lam_d) / reward_scale

                    if l_div.item() > 1e-4:
                        epoch_violators.append(i)

                    total_loss = actor_loss + (self.lam_s * l_smooth) + l_div

                    self.optimizers[i].zero_grad()
                    total_loss.backward(retain_graph=False)
                    
                    grad_norm = 0.0
                    for p in self.actors[i].parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    loss_stats["grad_norm"] += grad_norm / PPO_EPOCHS

                    torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 0.5)
                    self.optimizers[i].step()

                    loss_stats["actor_loss"] += actor_loss.item() / PPO_EPOCHS
                    loss_stats["smooth_loss"] += (l_smooth.item() if torch.is_tensor(l_smooth) else l_smooth) / PPO_EPOCHS
                    loss_stats["div_loss"] += (l_div.item() if torch.is_tensor(l_div) else l_div) / PPO_EPOCHS

                if epoch == 0:
                    div_violators_indices.extend(epoch_violators)

            torch.cuda.empty_cache()

        div_violators_count = len(div_violators_indices)
        actual_stds = [torch.exp(self.actors[idx].log_std).mean().item() for idx in range(self.N)]

        def safe_mean(idx_list): return np.mean(current_returns[idx_list]) if len(idx_list) > 0 else 0.0
        def safe_std(idx_list): return np.std(current_returns[idx_list]) if len(idx_list) > 0 else 0.0
        def safe_mean_std_action(idx_list): return np.mean([actual_stds[i] for i in idx_list]) if len(idx_list) > 0 else 0.0

        ################################################################################
        # 6. DYNAMIC POPULATION CULLING & LOGGING
        ################################################################################
        reduce_population = False
        if len(mid_idx) > 0:
            mid_tier_violators = len([i for i in div_violators_indices if i in mid_idx])
            div_pressure = mid_tier_violators / len(mid_idx) if len(mid_idx) > 0 else 0.0
            if div_pressure > 0.25 and self.N > 20:
                reduce_population = True
        
        if ep >= self.warmup_episodes:
            survivor_actors, survivor_old_actors, survivor_optimizers = [], [], []
            for idx in range(len(self.actors)):
                if idx not in reset_idx:
                    survivor_actors.append(self.actors[idx])
                    survivor_old_actors.append(self.old_actors[idx])
                    survivor_optimizers.append(self.optimizers[idx])

            num_to_replace = len(reset_idx)
            
            if reduce_population:
                num_to_replace = int(len(reset_idx) * 0.90)  
                print(f"[Annihilation] Diversity pressure high. Population shrinking to {len(survivor_actors) + num_to_replace}.")

            for _ in range(num_to_replace):
                target_elite = np.random.choice(elite_idx)
                new_actor = bf.ContinuousActor(self.obs_space, self.action_space, self.hidden_dim).to(self.device)
                new_old_actor = bf.ContinuousActor(self.obs_space, self.action_space, self.hidden_dim).to(self.device)
                
                new_dict = {k: v + torch.randn_like(v) * 0.02 for k, v in self.actors[target_elite].state_dict().items()}
                new_actor.load_state_dict(new_dict)
                new_old_actor.load_state_dict(new_dict)
                
                survivor_actors.append(new_actor)
                survivor_old_actors.append(new_old_actor)
                survivor_optimizers.append(optim.Adam(new_actor.parameters(), lr=self.base_lr))

            self.actors = nn.ModuleList(survivor_actors)
            self.old_actors = nn.ModuleList(survivor_old_actors)
            self.optimizers = survivor_optimizers
            self.N = len(self.actors)
            print(f"[Population Update] New population size: {self.N}")

        self.buffer = bf.PopulationBuffer(self.N) 

        if (ep + 1) % self.plot_interval == 0:
            self._plot_exploration_health()

        for key in loss_stats: loss_stats[key] /= max(1, updated_agents_count)
        noise_pct = getattr(self, 'last_noise_ratio', 0.0) * 100
        cluster_cnt = getattr(self, 'last_num_clusters', 0)
        
        print(f"[Update] Actor: {loss_stats['actor_loss']:.4f}, Ratio: {loss_stats['ppo_ratio']:.3f}, Grad: {loss_stats['grad_norm']:.4f}, Smooth: {loss_stats['smooth_loss']:.4f}, Elite Return: {np.mean(elite_stats):.2f} | Clusters: {cluster_cnt} (Noise: {noise_pct:.1f}%)")

        stats_dict = {
            "loss_actor": loss_stats["actor_loss"],
            "ppo_ratio": loss_stats["ppo_ratio"],
            "loss_smooth": loss_stats["smooth_loss"],
            "loss_div": loss_stats["div_loss"],
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
            "target_min_std": mid_std,       
            "target_max_std": max_std,       
            "elite_mean": np.mean(elite_stats),
            "div_violators": div_violators_count,
            "population_size": self.N
        }
        return stats_dict

    def _plot_spatial_only(self, pos_data, labels):
        os.makedirs('plots', exist_ok=True)
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        if hasattr(self, 'current_track_data') and self.current_track_data is not None:
            track_x = [p[0] for p in self.current_track_data]
            track_y = [p[1] for p in self.current_track_data]
            track_x.append(track_x[0]); track_y.append(track_y[0])
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
            'ref_actors_state_dict': self.ref_actors.state_dict(),
            'optimizers_state_dict': [opt.state_dict() for opt in self.optimizers],
            'buffer_data': self.buffer.buffers,
            'current_episodes': getattr(self.buffer, 'current_episodes', None),
            
            'scaler': self.scaler,
            'action_scaler': self.action_scaler, 
            'pca': self.pca,
            'history_buffer': self.history_buffer,
            
            'elite_return_history': self.elite_return_history,
            'return_mean_history': self.return_mean_history,
            'return_std_history': self.return_std_history,
            'running_reward_std': self.running_reward_std
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return {'episode': 0, 'eval_rewards': [], 'seed_logs': []}

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actors.load_state_dict(ckpt['actors_state_dict'])
        self.old_actors.load_state_dict(ckpt['old_actors_state_dict'])

        if 'ref_actors_state_dict' in ckpt:
            self.ref_actors.load_state_dict(ckpt['ref_actors_state_dict'])
            self.ref_actor = self.ref_actors[0]
        elif 'ref_actor_state_dict' in ckpt:
            self.ref_actor.load_state_dict(ckpt['ref_actor_state_dict'])

        for opt, state in zip(self.optimizers, ckpt['optimizers_state_dict']):
            opt.load_state_dict(state)
            
        self.current_policy_idx = ckpt['current_policy_idx']
        self.buffer.buffers = ckpt['buffer_data']

        if ckpt.get('current_episodes') is not None:
            self.buffer.current_episodes = ckpt['current_episodes']

        # --- Wiederherstellung der Pipeline ---
        self.scaler = ckpt.get('scaler', StandardScaler())
        self.action_scaler = ckpt.get('action_scaler', StandardScaler())
        self.pca = ckpt.get('pca', None)
        self.history_buffer = ckpt.get('history_buffer', collections.deque(maxlen=10))
        
        # --- Wiederherstellung der Metriken ---
        self.elite_return_history = ckpt.get('elite_return_history', collections.deque(maxlen=10))
        self.return_mean_history = ckpt.get('return_mean_history', collections.deque(maxlen=20))
        self.return_std_history = ckpt.get('return_std_history', collections.deque(maxlen=20))
        self.running_reward_std = ckpt.get('running_reward_std', 1.0)
        
        return ckpt