import matplotlib.pyplot as plt 
import os
import pandas as pd
import numpy as np

# =========================================================================================
# Visualization
# =========================================================================================

def save_and_plot_results(rewards_dict, folder="plots", window_size=10):
    """
    Saves and plots the training results for multiple environments and algorithms.
    Expects rewards_dict in the following format: 
    {env_name: {algo_name: [rewards_list]}}
    """
    if not os.path.exists(folder): 
        os.makedirs(folder)
    
    # 1. Flatten the data for CSV export (column names e.g., "CartPole-v1_SAC")
    flat_data = {}
    for env_name, algos in rewards_dict.items():
        for algo_name, rewards in algos.items():
            column_name = f"{env_name}_{algo_name}"
            flat_data[column_name] = rewards
            
    pd.DataFrame.from_dict(flat_data, orient='index').transpose().to_csv(os.path.join(folder, "data.csv"))

    # 2. Dynamic plot creation (automatically scales with the number of environments)
    num_envs = len(rewards_dict)
    fig, axs = plt.subplots(1, num_envs, figsize=(6 * num_envs, 5))
    
    # Ensure axs is iterable even if there's only one environment
    if num_envs == 1:
        axs = [axs]
        
    for i, (env_name, algos) in enumerate(rewards_dict.items()):
        ax = axs[i]
        ax.set_title(env_name)
        
        # DOWNWARD COMPATIBILITY: If it's a flat list, wrap it in a dictionary
        if isinstance(algos, list):
            algos = {"Agent": algos}
            
        for algo_name, rewards in algos.items():
            series = pd.Series(rewards)

        for algo_name, rewards in algos.items():
            series = pd.Series(rewards)
            
            # Only calculate and plot rolling stats if we have enough data points
            if len(rewards) >= window_size:
                rolling_mean = series.rolling(window=window_size).mean()
                rolling_std = series.rolling(window=window_size).std()
                
                # Plot the moving average (returns the line object to extract its color)
                line, = ax.plot(rolling_mean, label=f'{algo_name} (MA{window_size})', linewidth=2)
                
                # Plot the shaded area for the standard deviation using the exact same color
                ax.fill_between(
                    range(len(rewards)), 
                    rolling_mean - rolling_std, 
                    rolling_mean + rolling_std, 
                    color=line.get_color(), 
                    alpha=0.2 # 20% opacity for better visibility
                )
            else:
                # Fallback plot for extremely short runs
                ax.plot(rewards, label=f'{algo_name} (Raw)', linewidth=2)
                
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Reward")
        ax.legend()
    
    plt.tight_layout() # Prevents overlapping of axis labels
    plt.savefig(os.path.join(folder, "learning_curves.png"), dpi=300) # High-res output
    plt.close()