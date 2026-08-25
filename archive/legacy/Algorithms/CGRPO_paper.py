# =========================================================================================
# Continuous Group Relative Policy Optimization (CGRPO)
# Implementation of "Algorithm 1" from Khanda et al. (2025)
# =========================================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import math
import gymnasium as gym
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
import utils as bf

class GroupBuffer:
    """Stores trajectories for the N policies before updating."""
    def __init__(self):
        self.episodes = []
        self.reset_current_episode()

    def reset_current_episode(self):
        self.current_ep = {"obs": [], "action": [], "logprob": [], "reward": [], "feature": [], "policy_idx": -1}

    def add(self, obs, action, logprob, reward, feature, policy_idx):
        self.current_ep["obs"].append(obs)
        self.current_ep["action"].append(action)
        self.current_ep["logprob"].append(logprob)
        self.current_ep["reward"].append(reward)
        self.current_ep["feature"].append(feature)
        self.current_ep["policy_idx"] = policy_idx

    def finish_episode(self):
        if len(self.current_ep["reward"]) > 0:
            self.episodes.append(self.current_ep)
        self.reset_current_episode()

    def clear(self):
        self.episodes.clear()
        self.reset_current_episode()


class CGRPO:
    """CGRPO Algorithm matching the theoretical framework of Khanda et al."""
    
    def __init__(self, env, hidden_dim=256, lr=3e-4, gamma=0.99, K_epochs=10, dbscan_eps=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.dbscan_eps = dbscan_eps
        
        # ---------------------------------------------------------
        # CGRPO PAPER PARAMETERS (Sec 3.2 & 7.1)
        # ---------------------------------------------------------
        self.num_policies = 2            # N=2 as per HalfCheetah experiment
        self.episodes_per_policy = 2     # Collects ~2000 steps per policy per update (Cartpole)
        self.total_update_episodes = self.num_policies * self.episodes_per_policy
        
        self.epsilon_base = 0.2          # Base clipping parameter for PPO objective
        self.lam_s = 0.01                # Temporal Smoothness Regularization weight
        self.lam_d = 0.01                # Inter-Group Diversity Regularization weight
        self.tau_div = 0.5               # Cosine similarity threshold for diversity penalty
        
        # Standard Action STD decay to allow continuous control to converge
        self.action_std = 0.6
        self.min_std = 0.05
        self.std_decay = 0.005
        
        self.buffer = GroupBuffer()
        self.current_episode = 0
        self._update_flag = False
        self.champion_idx = 0            # Tracks the best policy for evaluation
        
        # 1. Initialize N policies
        self.policies = nn.ModuleList([
            bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device) 
            for _ in range(self.num_policies)
        ])
        self.optimizers = [optim.Adam(p.parameters(), lr=lr) for p in self.policies]
        
        # 2. Global Reference Policy (\pi_{ref})
        # Serves as the anchor for knowledge distillation between the N policies.
        self.policy_ref = bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device)
        self.policy_ref.load_state_dict(self.policies[0].state_dict())
        
        self._sync_action_stds()

        self.scaler = StandardScaler()
        self.pca = None

    def _sync_action_stds(self):
        for p in self.policies:
            if hasattr(p, 'set_action_std'): p.set_action_std(self.action_std)
        if hasattr(self.policy_ref, 'set_action_std'): 
            self.policy_ref.set_action_std(self.action_std)

    def set_eval_mode(self):
        self.policies.eval()

    def set_train_mode(self):
        self.policies.train()

    def _get_active_policy_idx(self):
        """Rotates data collection among the N policies."""
        return (self.current_episode % self.total_update_episodes) // self.episodes_per_policy

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                # Evaluate ONLY the current champion to avoid destructive action averaging
                action = self.policies[self.champion_idx].get_deterministic_action(state_t)
                return action.squeeze(0).cpu().numpy()
            else:
                active_idx = self._get_active_policy_idx()
                action, logprob, feat = self.policies[active_idx].sample_action(state_t)
                
                self._cached_obs = state_t.squeeze(0).cpu()
                self._cached_action = action.squeeze(0).cpu()
                self._cached_logprob = logprob.squeeze(0).cpu()
                self._cached_feat = feat.squeeze(0).cpu()
                
                return action.squeeze(0).cpu().numpy()

    def step(self, state, action, reward, next_state, done, **kwargs):
        active_idx = self._get_active_policy_idx()
        
        self.buffer.add(
            self._cached_obs, 
            self._cached_action, 
            self._cached_logprob, 
            reward, 
            self._cached_feat,
            active_idx
        )
        
        if done:
            self.buffer.finish_episode()
            if len(self.buffer.episodes) >= self.total_update_episodes:
                stats = self.update()
                self._update_flag = True
                return stats
        return None

    def consume_update_flag(self):
        flag = self._update_flag
        self._update_flag = False
        return flag

    def _cluster_states(self, flat_features_np):
        """Eq. 7: State Clustering using DBSCAN to define relative situations."""
        scaled_features = self.scaler.fit_transform(flat_features_np)
        pca_comps = min(30, flat_features_np.shape[1])
        if self.pca is None or self.pca.n_components != pca_comps:
            self.pca = PCA(n_components=pca_comps)
            
        pca_features = np.nan_to_num(self.pca.fit_transform(scaled_features), nan=0.0, posinf=0.0, neginf=0.0)
        
        c_size = int(np.clip(len(pca_features) * 0.01, 10, 80))
        min_samples = int(max(2, c_size / 3))
        
        dbscan = HDBSCAN(min_cluster_size=c_size, min_samples=min_samples, 
                         cluster_selection_epsilon=self.dbscan_eps, core_dist_n_jobs=1)
        return dbscan.fit_predict(pca_features)

    def update(self):
        # ---------------------------------------------------------
        # PRE-COMPUTATION & CHAMPION SELECTION
        # ---------------------------------------------------------
        policy_returns = [[] for _ in range(self.num_policies)]
        all_obs, all_actions, all_features, all_rtg, all_policy_ids = [], [], [], [], []

        for ep_data in self.buffer.episodes:
            rewards = ep_data["reward"]
            p_idx = ep_data["policy_idx"]
            policy_returns[p_idx].append(sum(rewards))
            
            rtg = bf.compute_returns_to_go(rewards, self.gamma, self.device)
            
            all_obs.append(torch.stack(ep_data["obs"]))
            all_actions.append(torch.stack(ep_data["action"]))
            all_features.append(torch.stack(ep_data["feature"]))
            all_rtg.append(rtg)
            all_policy_ids.append(torch.full((len(rewards),), p_idx, dtype=torch.long))

        # Identify champion policy (to update reference policy at the end)
        avg_returns = [np.mean(ret) if len(ret) > 0 else -np.inf for ret in policy_returns]
        self.champion_idx = int(np.argmax(avg_returns))

        flat_obs = torch.cat(all_obs).to(self.device)
        flat_actions = torch.cat(all_actions).to(self.device)
        flat_rtg = torch.cat(all_rtg).to(self.device)
        flat_policy_ids = torch.cat(all_policy_ids).to(self.device)
        flat_features_np = torch.cat(all_features).numpy()

        # ---------------------------------------------------------
        # Eq 6: STATE-AWARE ADVANTAGE ESTIMATION
        # A_i(s_t, a_t) = G_i(s_t) - mean(G_cluster(s_t))
        # ---------------------------------------------------------
        labels = self._cluster_states(flat_features_np)
        labels_t = torch.tensor(labels, device=self.device)
        
        advantages = torch.zeros_like(flat_rtg)
        unique_labels = torch.unique(labels_t)
        
        for c in unique_labels:
            c_val = c.item()
            if c_val == -1: continue # Ignore noise
            
            mask = (labels_t == c_val)
            c_rtg = flat_rtg[mask]
            
            # Subtract baseline (mean of state cluster)
            if len(c_rtg) > 1:
                advantages[mask] = c_rtg - c_rtg.mean()

        # ---------------------------------------------------------
        # Eq 8: GROUP-NORMALIZED ADVANTAGES
        # ---------------------------------------------------------
        # Note: Since N=2 effectively forms one global policy group for 
        # this experiment, we normalize across the entire collected batch.
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        advantages = torch.clamp(advantages, -4.0, 4.0)

        # Eq 9: Group-specific clipping (simplifies to base since group is global)
        epsilon_g = self.epsilon_base * max(1.0, float(adv_std) / float(adv_std)) 

        # ---------------------------------------------------------
        # Eq 12: POLICY UPDATES (L_CGRPO + L_smooth + L_diversity)
        # ---------------------------------------------------------
        batch_size = 256
        avg_loss, avg_l_smooth, avg_l_div = 0.0, 0.0, 0.0
        num_batches = 0

        for epoch in range(self.K_epochs):
            for p_idx in range(self.num_policies):
                # Strict On-Policy isolation: Policy only learns from its own data
                p_mask = (flat_policy_ids == p_idx)
                if not p_mask.any(): continue
                
                p_obs = flat_obs[p_mask]
                p_actions = flat_actions[p_mask]
                p_adv = advantages[p_mask]
                
                indices = torch.randperm(len(p_obs))
                
                for start in range(0, len(p_obs), batch_size):
                    batch_idx = indices[start:start+batch_size]
                    
                    b_obs = p_obs[batch_idx]
                    b_actions = p_actions[batch_idx]
                    b_adv = p_adv[batch_idx]
                    
                    feat = self.policies[p_idx].forward_features(b_obs)
                    dist = self.policies[p_idx].get_distribution(feat)
                    
                    # Unbound action for normal distribution evaluation
                    u = torch.atanh(torch.clamp(b_actions, -0.999999, 0.999999))
                    new_log_probs = dist.log_prob(u) - torch.log(1.0 - b_actions.pow(2) + 1e-6)
                    new_log_probs = new_log_probs.sum(dim=-1)
                    
                    # Log-probs of old actions for the ratio
                    with torch.no_grad():
                        old_dist = self.policy_ref.get_distribution(self.policy_ref.forward_features(b_obs))
                        old_log_probs = old_dist.log_prob(u) - torch.log(1.0 - b_actions.pow(2) + 1e-6)
                        old_log_probs = old_log_probs.sum(dim=-1)

                    ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    # Objective 1: Clipped CGRPO Objective (Eq 8)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1.0 - epsilon_g, 1.0 + epsilon_g) * b_adv
                    L_CGRPO = -torch.min(surr1, surr2).mean()

                    # Objective 2: Temporal Smoothness (Eq 10)
                    # Penalizes erratic jumps in feature space between sequential states
                    if len(feat) > 1:
                        L_smooth = self.lam_s * torch.mean((feat[1:] - feat[:-1])**2)
                    else:
                        L_smooth = torch.tensor(0.0).to(self.device)

                    # Objective 3: Inter-Group Diversity (Eq 11)
                    # Penalizes policies if their mean actions become too similar to the reference
                    mean_action = dist.mean
                    with torch.no_grad():
                        ref_mean_action = old_dist.mean
                        
                    sim = torch.nn.functional.cosine_similarity(mean_action, ref_mean_action, dim=-1)
                    L_diversity = self.lam_d * torch.mean(torch.clamp(sim - self.tau_div, min=0.0))

                    # Total Loss (Eq 12)
                    total_loss = L_CGRPO + L_smooth + L_diversity

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        continue

                    self.optimizers[p_idx].zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policies[p_idx].parameters(), 0.5)
                    self.optimizers[p_idx].step()
                    
                    avg_loss += L_CGRPO.item()
                    avg_l_smooth += L_smooth.item()
                    avg_l_div += L_diversity.item()
                    num_batches += 1

        self.buffer.clear()
        
        # ---------------------------------------------------------
        # Eq 29-30: REFERENCE POLICY UPDATE
        # Update \pi_{ref} as mixture of top-performing policies
        # ---------------------------------------------------------
        # We perform hard-distillation by copying the champion's weights
        self.policy_ref.load_state_dict(self.policies[self.champion_idx].state_dict())

        # Exploration Decay
        self.action_std = max(self.min_std, self.action_std - self.std_decay)
        self._sync_action_stds()

        noise_ratio = (labels == -1).sum() / len(labels) if len(labels) > 0 else 0.0
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return {
            "loss_actor": avg_loss / max(1, num_batches),
            "loss_smooth": avg_l_smooth / max(1, num_batches),
            "loss_div": avg_l_div / max(1, num_batches),
            "cluster_count": num_clusters,
            "cluster_noise_ratio": round(noise_ratio * 100, 1),
            "group_adv_mean": advantages.abs().mean().item(),
            "tier_elite_action_std": self.action_std,
            "population_size": self.num_policies
        }

    def save_checkpoint(self, path, ep, eval_rewards, seed_logs=None):
        checkpoint = {
            'episode': ep,
            'eval_rewards': eval_rewards,
            'seed_logs': seed_logs if seed_logs is not None else [],
            'policies_state_dict': self.policies.state_dict(),
            'policy_ref_state_dict': self.policy_ref.state_dict(),
            'optimizers_state_dict': [opt.state_dict() for opt in self.optimizers],
            'action_std': self.action_std,
            'scaler': self.scaler,
            'pca': self.pca
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return {'episode': 0, 'eval_rewards': [], 'seed_logs': []}

        ckpt = torch.load(path, map_location=self.device)
        self.policies.load_state_dict(ckpt['policies_state_dict'])
        self.policy_ref.load_state_dict(ckpt['policy_ref_state_dict'])
        
        for opt, state in zip(self.optimizers, ckpt['optimizers_state_dict']):
            opt.load_state_dict(state)
        
        self.action_std = ckpt.get('action_std', 0.6)
        self._sync_action_stds()
            
        self.scaler = ckpt.get('scaler', StandardScaler())
        self.pca = ckpt.get('pca', None)
        
        return ckpt