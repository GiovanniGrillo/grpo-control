import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import glob

def plot_with_variance(ax, episodes, mean_series, std_series, color, label):
    """Hilfsfunktion für das Plotten von Linien mit Standardabweichungs-Schatten."""
    
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

    # ==========================================
    # 7. Diversity (Tau) Violations & Population
    # ==========================================
    if 'div_violators' in df_mean.columns:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        mask = ~df_mean['div_violators'].isna()
        valid_episodes = episodes[mask]
        
        if mask.any():
            plot_with_variance(ax1, valid_episodes, df_mean.loc[mask, 'div_violators'], df_std.loc[mask, 'div_violators'], '#9467bd', 'Agents > Tau')
            
            plt.title('Diversity Pressure vs Population Size')
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Number of Agents', color='#9467bd')
            ax1.tick_params(axis='y', labelcolor='#9467bd')
            
            from matplotlib.ticker import MaxNLocator
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            # NEU: Zweite Y-Achse für Populationsgröße
            if 'population_size' in df_mean.columns:
                ax2 = ax1.twinx()
                color_pop = '#333333'
                mask_pop = ~df_mean['population_size'].isna()
                if mask_pop.any():
                    ax2.plot(episodes[mask_pop], df_mean.loc[mask_pop, 'population_size'], color=color_pop, linestyle=':', linewidth=2.5, label='Population (N)')
                    ax2.set_ylabel('Total Population Size', color=color_pop)
                    ax2.tick_params(axis='y', labelcolor=color_pop)
                    
                    # Legenden zusammenführen
                    lines_1, labels_1 = ax1.get_legend_handles_labels()
                    lines_2, labels_2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
            else:
                ax1.legend()
            
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            plt.savefig(os.path.join(plot_dir, '7_diversity_violations.png'), dpi=300)
        plt.close()

    print(f"-> Successfully created up to 7 plots in {plot_dir}/")

def compare_runs(runs_dict, output_dir="runs/Comparison"):
    """
    Compare multiple runs across the 7 predefined plot categories.
    runs_dict: Dictionary in the form {"Desired Label Name": "path/to/run_folder"}
    """
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.tab10.colors 
    
    run_data_cache = {}
    
    # 1. Load all data and cache it
    for label, run_dir in runs_dict.items():
        metrics_path = os.path.join(run_dir, "metrics.csv")
        if not os.path.exists(metrics_path):
            print(f"Warning: No metrics.csv found in {run_dir}. Skipping.")
            continue
            
        df = pd.read_csv(metrics_path)
        if df.empty or 'episode' not in df.columns:
            continue
            
        grouped = df.groupby('episode')
        df_mean = grouped.mean()
        df_std = grouped.std()
        df_mean.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        run_data_cache[label] = {
            'mean': df_mean,
            'std': df_std
        }

    if not run_data_cache:
        print("No valid data found to compare.")
        return

    # Helper function to plot a specific metric for all cached runs onto a given axis
    def plot_metric_on_ax(ax, metric, title, y_label, log_scale=False):
        data_found = False
        for i, (label, data) in enumerate(run_data_cache.items()):
            df_mean = data['mean']
            df_std = data['std']
            
            if metric in df_mean.columns:
                episodes = df_mean.index
                color = colors[i % len(colors)]
                plot_with_variance(ax, episodes, df_mean[metric], df_std[metric], color, label)
                data_found = True
                
        if data_found:
            ax.set_title(title)
            ax.set_xlabel('Episodes')
            ax.set_ylabel(y_label)
            if log_scale:
                ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        return data_found

    # ==========================================
    # 1. Learning Curve (Eval Reward)
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    if plot_metric_on_ax(ax, 'eval_reward', 'Learning Curve (Eval Reward)', 'Reward'):
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_compare_learning_curve.png'), dpi=300)
    plt.close()

    # ==========================================
    # 2. Losses
    # ==========================================
    losses = ['loss_actor', 'loss_trauma', 'loss_div', 'loss_smooth']
    fig, axs = plt.subplots(len(losses), 1, figsize=(10, 3 * len(losses)), sharex=True)
    if len(losses) == 1: axs = [axs]
    
    any_loss = False
    for ax, loss in zip(axs, losses):
        if plot_metric_on_ax(ax, loss, f"{loss.replace('loss_', '').capitalize()} Loss", 'Loss Value'):
            any_loss = True
            
    if any_loss:
        fig.suptitle('Comparison: Optimization Losses', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_compare_losses.png'), dpi=300)
    plt.close()

    # ==========================================
    # 3. Tier Returns
    # ==========================================
    tiers = [('tier_elite_return_avg', 'Elite Tier Return'), 
             ('tier_mid_return_avg', 'Mid Tier Return'), 
             ('tier_scout_return_avg', 'Scout Tier Return')]
    fig, axs = plt.subplots(len(tiers), 1, figsize=(10, 3 * len(tiers)), sharex=True)
    
    any_tier = False
    for ax, (metric, title) in zip(axs, tiers):
        if plot_metric_on_ax(ax, metric, title, 'Return'):
            any_tier = True
            
    if any_tier:
        fig.suptitle('Comparison: Performance by Population Tier', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_compare_tier_returns.png'), dpi=300)
    plt.close()

    # ==========================================
    # 4. Action STD
    # ==========================================
    stds = [('tier_elite_action_std', 'Elite Action STD'), 
            ('tier_mid_action_std', 'Mid Action STD'), 
            ('tier_scout_action_std', 'Scout Action STD')]
    fig, axs = plt.subplots(len(stds), 1, figsize=(10, 3 * len(stds)), sharex=True)
    
    any_std = False
    for ax, (metric, title) in zip(axs, stds):
        if plot_metric_on_ax(ax, metric, title, 'Standard Deviation', log_scale=True):
            any_std = True
            
    if any_std:
        fig.suptitle('Comparison: Action Variance (Exploration vs Exploitation)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_compare_action_stds.png'), dpi=300)
    plt.close()

    # ==========================================
    # 5. Trauma System Health
    # ==========================================
    trauma_metrics = [('trauma_centers_count', 'Active Trauma Centers'), 
                      ('trauma_threshold', 'Trauma Trigger Threshold')]
    fig, axs = plt.subplots(len(trauma_metrics), 1, figsize=(10, 6), sharex=True)
    
    any_trauma = False
    for ax, (metric, title) in zip(axs, trauma_metrics):
        if plot_metric_on_ax(ax, metric, title, 'Value'):
            any_trauma = True
            
    if any_trauma:
        fig.suptitle('Comparison: Trauma Memory Dynamics', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '5_compare_trauma_dynamics.png'), dpi=300)
    plt.close()

    # ==========================================
    # 6. Cluster Health
    # ==========================================
    cluster_metrics = [('cluster_count', 'Number of Clusters'), 
                       ('cluster_noise_ratio', 'Noise Ratio (-1)')]
    fig, axs = plt.subplots(len(cluster_metrics), 1, figsize=(10, 6), sharex=True)
    
    any_cluster = False
    for ax, (metric, title) in zip(axs, cluster_metrics):
        if plot_metric_on_ax(ax, metric, title, 'Value'):
            any_cluster = True
            
    if any_cluster:
        fig.suptitle('Comparison: Latent Space Clustering Health', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '6_compare_cluster_health.png'), dpi=300)
    plt.close()

    # ==========================================
    # 7. Diversity Violations & Population
    # ==========================================
    div_metrics = [('div_violators', 'Agents Exceeding Tau'), 
                   ('population_size', 'Total Population Size')]
    fig, axs = plt.subplots(len(div_metrics), 1, figsize=(10, 6), sharex=True)
    
    any_div = False
    for ax, (metric, title) in zip(axs, div_metrics):
        if plot_metric_on_ax(ax, metric, title, 'Count'):
            # Force y-axis to integer values for population and agent counts
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            any_div = True
            
    if any_div:
        fig.suptitle('Comparison: Diversity Pressure vs Population Size', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '7_compare_diversity_annihilation.png'), dpi=300)
    plt.close()

    print(f"-> Successfully created 7 comparison plots in {output_dir}/")
    
# ==========================================
# Main Run
# ==========================================
if __name__ == "__main__":
    runs_dir = "runs"
    
    # --- 1. Individual Plots ---
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
            
            print("\nAll individual plotting complete!")

    # --- 2. Comparison-Plots ---
    """print("\nStarting comparison plots...")

    runs_to_compare_Cartpole = {
        "AGRPO_N100_popcull": "runs/N100_pop_cull/AGRPO_dm_control_cartpole-swingup-v0_20260515_134323",  
        "AGRPO_mem_N50": "runs/N50_memory/AGRPO_dm_control_cartpole-swingup-v0_20260515_185707",
        "AGRPO_baseline_N50": "runs/N50_all_loss/AGRPO_dm_control_cartpole-swingup-v0_20260513_171600"
    }
    
    compare_runs(runs_to_compare_Cartpole, output_dir="runs/comparison_cartpole_plots")

    runs_to_compare_Acrobot = {
        "AGRPO_N100_popcull": "runs/N100_pop_cull/AGRPO_dm_control_acrobot-swingup-v0_20260515_142111",  
        "AGRPO_mem_N50": "runs/N50_memory/AGRPO_dm_control_acrobot-swingup-v0_20260515_194206",
        "AGRPO_baseline_N50": "runs/N50_all_loss/AGRPO_dm_control_acrobot-swingup-v0_20260513_175454"
    }
    
    compare_runs(runs_to_compare_Acrobot, output_dir="runs/comparison_acrobot_plots")

    runs_to_compare_Carracing = {
        "AGRPO_N100_popcull": "runs/N100_pop_cull/AGRPO_CarRacing-v3_20260515_145522",  
        "AGRPO_mem_N50": "runs/N50_memory/AGRPO_CarRacing-v3_20260516_171029",
        "AGRPO_baseline_N50": "runs/N50_all_loss/AGRPO_CarRacing-v3_20260513_182950"
    }
    
    compare_runs(runs_to_compare_Carracing, output_dir="runs/comparison_carracing_plots")"""