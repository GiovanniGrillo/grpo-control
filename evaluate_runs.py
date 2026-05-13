import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import glob

def plot_with_variance(ax, episodes, mean_series, std_series, color, label):
    """Hilfsfunktion für das Plotten von Linien mit Standardabweichungs-Schatten."""
    
    # FIX 1: Filtere alle NaN-Werte heraus, damit Matplotlib eine durchgehende Linie 
    # zwischen den Update-Schritten (alle N Episoden) ziehen kann.
    mask = ~mean_series.isna()
    
    if not mask.any():
        return # Keine Daten zum Plotten vorhanden
        
    valid_episodes = episodes[mask]
    valid_mean = mean_series[mask]
    
    ax.plot(valid_episodes, valid_mean, color=color, label=label, linewidth=2)
    
    if not std_series.isna().all():
        valid_std = std_series[mask]
        ax.fill_between(valid_episodes, valid_mean - valid_std, valid_mean + valid_std, color=color, alpha=0.2)

def generate_plots_for_run(run_dir):
    metrics_path = os.path.join(run_dir, "metrics.csv")
    
    if not os.path.exists(metrics_path):
        print(f"Skipping {run_dir} - no metrics.csv found yet.")
        return
        
    try:
        df = pd.read_csv(metrics_path)
    except pd.errors.EmptyDataError:
        print(f"Skipping {run_dir} - metrics.csv is empty.")
        return

    if df.empty or 'episode' not in df.columns:
        print(f"Skipping {run_dir} - no episode data found.")
        return

    print(f"Generating plots for: {os.path.basename(run_dir)} ...")
    
    grouped = df.groupby('episode')
    df_mean = grouped.mean()
    df_std = grouped.std()
    
    # FIX 2: Ersetze -inf (aus dem Trauma Warmup) durch NaN, damit die Y-Achse nicht explodiert.
    # Der Schwellenwert taucht dann im Plot einfach erst ab Episode 100 auf, wenn er real wird.
    df_mean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    episodes = df_mean.index
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ==========================================
    # 1. Learning Curve (Eval Reward)
    # ==========================================
    if 'eval_reward' in df_mean.columns:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        plot_with_variance(ax, episodes, df_mean['eval_reward'], df_std['eval_reward'], '#1f77b4', 'Eval Reward (Total)')
        plt.title('Learning Curve (Average over Seeds)')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, '1_learning_curve.png'), dpi=300)
        plt.close()

    # ==========================================
    # 2. Losses (Actor, Trauma, Div, Smooth)
    # ==========================================
    expected_losses = ['loss_actor', 'loss_trauma', 'loss_div', 'loss_smooth']
    available_losses = [l for l in expected_losses if l in df_mean.columns]
    
    if available_losses:
        fig, axs = plt.subplots(len(available_losses), 1, figsize=(10, 3 * len(available_losses)), sharex=True)
        if len(available_losses) == 1: axs = [axs] 
        
        colors = ['#d62728', '#9467bd', '#2ca02c', '#ff7f0e']
        for ax, loss_col, color in zip(axs, available_losses, colors):
            plot_with_variance(ax, episodes, df_mean[loss_col], df_std[loss_col], color, loss_col.replace('loss_', '').capitalize() + ' Loss')
            ax.set_ylabel('Loss Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        axs[-1].set_xlabel('Episodes')
        fig.suptitle('Optimization Losses', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, '2_losses.png'), dpi=300)
        plt.close()

    # ==========================================
    # 3. Tier Returns (Elite vs Mid vs Scout)
    # ==========================================
    tiers = ['tier_elite_return_avg', 'tier_mid_return_avg', 'tier_scout_return_avg']
    if all(t in df_mean.columns for t in tiers):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        plot_with_variance(ax, episodes, df_mean['tier_elite_return_avg'], df_std['tier_elite_return_avg'], '#2ca02c', 'Elite Tier')
        plot_with_variance(ax, episodes, df_mean['tier_mid_return_avg'], df_std['tier_mid_return_avg'], '#1f77b4', 'Mid Tier')
        plot_with_variance(ax, episodes, df_mean['tier_scout_return_avg'], df_std['tier_scout_return_avg'], '#ff7f0e', 'Scout Tier')
        
        plt.title('Performance by Population Tier')
        plt.xlabel('Episodes')
        plt.ylabel('Return')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, '3_tier_returns.png'), dpi=300)
        plt.close()

    # ==========================================
    # 4. Action STD (Exploration Health)
    # ==========================================
    std_tiers = ['tier_elite_action_std', 'tier_mid_action_std', 'tier_scout_action_std']
    if all(s in df_mean.columns for s in std_tiers):
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        mask = ~df_mean['tier_elite_action_std'].isna()
        valid_episodes = episodes[mask]
        
        if mask.any():
            ax.plot(valid_episodes, df_mean.loc[mask, 'tier_elite_action_std'], color='#2ca02c', label='Elite STD', linewidth=2)
            ax.plot(valid_episodes, df_mean.loc[mask, 'tier_mid_action_std'], color='#1f77b4', label='Mid STD', linewidth=2)
            ax.plot(valid_episodes, df_mean.loc[mask, 'tier_scout_action_std'], color='#ff7f0e', label='Scout STD', linewidth=2)
            
            if 'target_min_std' in df_mean.columns and 'target_max_std' in df_mean.columns:
                ax.plot(valid_episodes, df_mean.loc[mask, 'target_min_std'], color='#d62728', linestyle='--', label='Min Floor')
                ax.plot(valid_episodes, df_mean.loc[mask, 'target_max_std'], color='#7f7f7f', linestyle='--', label='Max Ceiling')
                ax.fill_between(valid_episodes, df_mean.loc[mask, 'target_min_std'], df_mean.loc[mask, 'target_max_std'], color='gray', alpha=0.1)

            plt.title('Action Variance (Exploration vs Exploitation)')
            plt.xlabel('Episodes')
            plt.ylabel('Standard Deviation of Actions')
            plt.yscale('log')
            plt.grid(True, which="both", alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(plot_dir, '4_action_stds.png'), dpi=300)
        plt.close()

    # ==========================================
    # 5. Trauma System Health
    # ==========================================
    if 'trauma_centers_count' in df_mean.columns and 'trauma_threshold' in df_mean.columns:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = '#d62728'
        mask1 = ~df_mean['trauma_centers_count'].isna()
        
        if mask1.any():
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Active Trauma Centers', color=color1)
            ax1.plot(episodes[mask1], df_mean.loc[mask1, 'trauma_centers_count'], color=color1, label='Memory Size', linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color1)
            
            ax2 = ax1.twinx()  
            color2 = '#7f7f7f'
            
            mask2 = ~df_mean['trauma_threshold'].isna()
            if mask2.any():
                ax2.set_ylabel('Trauma Trigger Threshold', color=color2)  
                ax2.plot(episodes[mask2], df_mean.loc[mask2, 'trauma_threshold'], color=color2, linestyle='--', label='Threshold Limit', alpha=0.7)
                ax2.tick_params(axis='y', labelcolor=color2)

            fig.suptitle('Trauma Memory Dynamics')
            fig.tight_layout()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plot_dir, '5_trauma_dynamics.png'), dpi=300)
        plt.close()

    # ==========================================
    # 6. Cluster Health (Latent Space Stability)
    # ==========================================
    if 'cluster_count' in df_mean.columns and 'cluster_noise_ratio' in df_mean.columns:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = '#17becf'
        mask = ~df_mean['cluster_count'].isna()
        
        if mask.any():
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Number of Clusters', color=color1)
            ax1.plot(episodes[mask], df_mean.loc[mask, 'cluster_count'], color=color1, linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color1)
            
            ax2 = ax1.twinx()  
            color2 = '#bcbd22'
            ax2.set_ylabel('Noise Ratio (-1)', color=color2)  
            ax2.plot(episodes[mask], df_mean.loc[mask, 'cluster_noise_ratio'], color=color2, linestyle='-.', linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color2)

            fig.suptitle('Latent Space Clustering Health')
            fig.tight_layout()
            plt.savefig(os.path.join(plot_dir, '6_cluster_health.png'), dpi=300)
        plt.close()

    print(f"-> Successfully created up to 6 plots in {plot_dir}/")

if __name__ == "__main__":
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        print(f"Directory '{runs_dir}' not found. Ensure you are running this from the main folder.")
    else:
        all_runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        
        if not all_runs:
            print("No experiment folders found in 'runs/'.")
        else:
            print(f"Found {len(all_runs)} runs. Starting analysis...")
            for run_path in all_runs:
                generate_plots_for_run(run_path)
            
            print("\nAll plotting complete!")