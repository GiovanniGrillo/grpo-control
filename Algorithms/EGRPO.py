# =========================================================================================
# Ensemble Continuous GRPO Agent (Adaptive KL-Penalty & Hybrid-Exploration)
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
    """Speichert Episoden für das Ensemble Update"""
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

class EGRPO:
    def __init__(self, env, hidden_dim=256, lr=3e-4, gamma=0.99, K_epochs=4, beta_kl=0.01, dbscan_eps=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.K_epochs = K_epochs       
        self.beta_kl = beta_kl         
        self.dbscan_eps = dbscan_eps   
        
        # ---------------------------------------------------------
        # ENSEMBLE PARAMETER (Dynamisch anpassbar)
        # ---------------------------------------------------------
        self.num_policies = 1
        self.episodes_per_policy = 10
        self.total_update_episodes = self.num_policies * self.episodes_per_policy # = 20
        
        # ---------------------------------------------------------
        # HYPERPARAMETER: KL-Penalty & Exploration
        # ---------------------------------------------------------
        self.target_kl = 0.015         
        
        self.warmup_episodes = 100     
        self.action_std_init = 0.8     
        self.action_std_min = 0.05     
        self.std_decay_rate = 0.01     
        
        self.action_std = self.action_std_init
        self.best_policy_idx = 0

        self.buffer = GroupBuffer()
        self.current_episode = 0
        self._update_flag = False
        
        # 1. Ensemble Policies
        self.policies = nn.ModuleList([
            bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device) 
            for _ in range(self.num_policies)
        ])
        self.optimizers = [optim.Adam(p.parameters(), lr=lr) for p in self.policies]
        
        # 2. Reference Policies (Kopie für KL-Divergenz)
        self.policies_ref = nn.ModuleList([
            bf.ContinuousActor(env.observation_space, env.action_space, hidden_dim).to(self.device) 
            for _ in range(self.num_policies)
        ])
        for i in range(self.num_policies):
            self.policies_ref[i].load_state_dict(self.policies[i].state_dict())
            if hasattr(self.policies[i], 'set_action_std'):
                self.policies[i].set_action_std(self.action_std)
                self.policies_ref[i].set_action_std(self.action_std)

        self.scaler = StandardScaler()
        self.pca = None

    def set_eval_mode(self):
        self.policies.eval()

    def set_train_mode(self):
        self.policies.train()

    def _get_active_policy_idx(self):
        """Bestimmt, welche Policy in der aktuellen Episode fahren darf"""
        return (self.current_episode % self.total_update_episodes) // self.episodes_per_policy

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                action = self.policies[self.best_policy_idx].get_deterministic_action(state_t)
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
            
            # Update triggern, wenn ALLE Policies ihre Episoden gesammelt haben
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
        scaled_features = self.scaler.fit_transform(flat_features_np)
        
        pca_dims = 30
        pca_comps = min(pca_dims, flat_features_np.shape[1])
        if self.pca is None or self.pca.n_components != pca_comps:
            self.pca = PCA(n_components=pca_comps)
            
        pca_features = np.nan_to_num(self.pca.fit_transform(scaled_features), nan=0.0, posinf=0.0, neginf=0.0)

        c_size = int(np.clip(len(pca_features) * 0.01, 10, 80))
        min_samples = int(max(2, c_size / 3))
        
        dbscan = HDBSCAN(min_cluster_size=c_size, min_samples=min_samples, 
                         cluster_selection_epsilon=self.dbscan_eps, core_dist_n_jobs=1)
        
        labels = dbscan.fit_predict(pca_features)
        return labels

    def update(self):
        policy_returns = [[] for _ in range(self.num_policies)]
        for ep_data in self.buffer.episodes:
            policy_returns[ep_data["policy_idx"]].append(sum(ep_data["reward"]))
            
        avg_returns = [np.mean(ret) if len(ret) > 0 else -np.inf for ret in policy_returns]
        self.best_policy_idx = int(np.argmax(avg_returns))

        # 1. Sync Reference Policies
        for i in range(self.num_policies):
            self.policies_ref[i].load_state_dict(self.policies[i].state_dict())

        all_obs, all_actions, all_features, all_rtg, all_policy_ids = [], [], [], [], []

        # 2. Returns-to-go berechnen (Global über alle Episoden)
        for ep_data in self.buffer.episodes:
            rewards = ep_data["reward"]
            rtg = bf.compute_returns_to_go(rewards, self.gamma, self.device)
            p_idx = ep_data["policy_idx"]
            
            all_obs.append(torch.stack(ep_data["obs"]))
            all_actions.append(torch.stack(ep_data["action"]))
            all_features.append(torch.stack(ep_data["feature"]))
            all_rtg.append(rtg)
            all_policy_ids.append(torch.full((len(rewards),), p_idx, dtype=torch.long))

        flat_obs = torch.cat(all_obs).to(self.device)
        flat_actions = torch.cat(all_actions).to(self.device)
        flat_rtg = torch.cat(all_rtg).to(self.device)
        flat_policy_ids = torch.cat(all_policy_ids).to(self.device)
        flat_features_np = torch.cat(all_features).numpy()

        # 3. Globales State-Clustering
        labels = self._cluster_states(flat_features_np)
        labels_t = torch.tensor(labels, device=self.device)

        # 4. Advantage Berechnung (Group Relative Normalization)
        advantages = torch.zeros_like(flat_rtg)
        unique_labels = torch.unique(labels_t)
        
        for c in unique_labels:
            c_val = c.item()
            if c_val == -1: continue
                
            mask = (labels_t == c_val)
            c_rtg = flat_rtg[mask]
            
            if len(c_rtg) > 1:
                advantages[mask] = (c_rtg - c_rtg.mean()) / (c_rtg.std() + 1e-8)
            else:
                advantages[mask] = 0.0

        advantages = torch.clamp(advantages, -4.0, 4.0)

        # 5. Lokales Policy Training (Jedes Netz lernt nur aus seinen eigenen Aktionen)
        batch_size = 128
        avg_loss, avg_kl = 0.0, 0.0
        num_batches = 0

        for epoch in range(self.K_epochs):
            for p_idx in range(self.num_policies):
                # Isoliere die Daten der aktuellen Policy
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
                    
                    u = torch.atanh(torch.clamp(b_actions, -0.999999, 0.999999))
                    new_log_probs = dist.log_prob(u) - torch.log(1.0 - b_actions.pow(2) + 1e-6)
                    new_log_probs = new_log_probs.sum(dim=-1)
                    
                    with torch.no_grad():
                        feat_ref = self.policies_ref[p_idx].forward_features(b_obs)
                        dist_ref = self.policies_ref[p_idx].get_distribution(feat_ref)
                        ref_log_probs = dist_ref.log_prob(u) - torch.log(1.0 - b_actions.pow(2) + 1e-6)
                        ref_log_probs = ref_log_probs.sum(dim=-1)

                    log_ratio = new_log_probs - ref_log_probs
                    log_ratio_clamped = torch.clamp(log_ratio, min=-20.0, max=5.0)
                    ratio = torch.exp(log_ratio_clamped)
                    
                    approx_kl = -log_ratio 
                    
                    grpo_objective = (ratio * b_adv) - (self.beta_kl * approx_kl)
                    loss = -grpo_objective.mean()

                    if torch.isnan(loss) or torch.isinf(loss):
                        continue

                    self.optimizers[p_idx].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policies[p_idx].parameters(), 0.5)
                    self.optimizers[p_idx].step()
                    
                    avg_loss += loss.item()
                    avg_kl += approx_kl.mean().item()
                    num_batches += 1

        self.buffer.clear()
        
        true_avg_kl = avg_kl / max(1, num_batches)
        
        # ---------------------------------------------------------
        # ADAPTIVER KL-PENALTY & EXPLORATION DECAY
        # ---------------------------------------------------------
        if true_avg_kl > self.target_kl * 1.5:
            self.beta_kl = min(self.beta_kl * 1.5, 0.1)  
        elif true_avg_kl < self.target_kl / 1.5:
            self.beta_kl = max(self.beta_kl / 1.5, 0.001)

        ep = getattr(self, 'current_episode', 0)
        
        if ep < self.warmup_episodes:
            self.action_std = self.action_std_init
        else:
            ep_post_warmup = ep - self.warmup_episodes
            self.action_std = self.action_std_min + (self.action_std_init - self.action_std_min) * math.exp(-self.std_decay_rate * ep_post_warmup)
            
        for i in range(self.num_policies):
            if hasattr(self.policies[i], 'set_action_std'):
                self.policies[i].set_action_std(self.action_std)
                self.policies_ref[i].set_action_std(self.action_std)

        noise_ratio = (labels == -1).sum() / len(labels) if len(labels) > 0 else 0.0
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return {
            "loss_actor": avg_loss / max(1, num_batches),
            "kl_div": true_avg_kl,
            "kl_beta": self.beta_kl,
            "cluster_count": num_clusters,
            "cluster_noise_ratio": round(noise_ratio * 100, 1),
            "group_adv_mean": advantages.abs().mean().item(),
            "tier_elite_action_std": self.action_std,
            "target_min_std": self.action_std_min,
            "target_max_std": self.action_std_init,
            "population_size": self.num_policies # Zeigt an, dass wir im Ensemble-Modus sind
        }

    def save_checkpoint(self, path, ep, eval_rewards, seed_logs=None):
        checkpoint = {
            'episode': ep,
            'eval_rewards': eval_rewards,
            'seed_logs': seed_logs if seed_logs is not None else [],
            'policies_state_dict': self.policies.state_dict(),
            'policies_ref_state_dict': self.policies_ref.state_dict(),
            'optimizers_state_dict': [opt.state_dict() for opt in self.optimizers],
            'beta_kl': self.beta_kl,
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
        self.policies_ref.load_state_dict(ckpt['policies_ref_state_dict'])
        
        for opt, state in zip(self.optimizers, ckpt['optimizers_state_dict']):
            opt.load_state_dict(state)
        
        self.beta_kl = ckpt.get('beta_kl', 0.01)
        self.action_std = ckpt.get('action_std', self.action_std_init)
        
        for i in range(self.num_policies):
            if hasattr(self.policies[i], 'set_action_std'):
                self.policies[i].set_action_std(self.action_std)
                self.policies_ref[i].set_action_std(self.action_std)
            
        self.scaler = ckpt.get('scaler', StandardScaler())
        self.pca = ckpt.get('pca', None)
        
        return ckpt