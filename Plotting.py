import matplotlib.pyplot as plt 
import os
import pandas as pd
import numpy as np
import json

"""
Data Management and Visualization Module

This module separates the processes of data storage and visualization. 
It utilizes a centralized JSON file to store training results, allowing for 
incremental updates, adding new environments, or overwriting specific 
algorithm runs without losing previously computed data.
"""

# =========================================================================================
# Visualization & Data Management
# =========================================================================================

def save_data(new_rewards_dict, folder="plots", filename="all_results.json"):
    """
    Saves and updates the training results in a central JSON document.
    
    It intelligently merges new data with existing data:
    - Appends new algorithms or environments if they do not exist yet.
    - Overwrites existing algorithm data for a specific environment if a new run is executed.
    - Preserves all other unrelated data in the document.
    
    Args:
        new_rewards_dict (dict): A nested dictionary containing the rewards of the current session. 
                                 Expected format: {env_name: {algo_name: [rewards_list]}}.
        folder (str, optional): The target directory for the output files. Defaults to "plots".
        filename (str, optional): The name of the central JSON master file. Defaults to "all_results.json".
        
    Outputs:
        - all_results.json: The main database holding all nested runs.
        - data.csv: A flattened CSV version of the JSON data (e.g., column 'CartPole-v1_PPO'), 
                    generated automatically for easy inspection.
    """
    if not os.path.exists(folder): 
        os.makedirs(folder)
        
    filepath = os.path.join(folder, filename)
    
    # 1. Load existing data if the file already exists
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                all_data = {} # Handle corrupted or empty files gracefully
    else:
        all_data = {}

    # 2. Update the master dictionary with the new data
    for env_name, algos in new_rewards_dict.items():
        if env_name not in all_data:
            all_data[env_name] = {}
            
        for algo_name, rewards in algos.items():
            # Overwrite or add the specific run
            all_data[env_name][algo_name] = rewards

    # 3. Save the updated master document back to JSON
    with open(filepath, 'w') as f:
        json.dump(all_data, f, indent=4)
        
    # 4. Optional: Export a flattened CSV version for spreadsheet applications
    flat_data = {}
    for env_name, algos in all_data.items():
        for algo_name, rewards in algos.items():
            column_name = f"{env_name}_{algo_name}"
            flat_data[column_name] = rewards
            
    # Orient='index' and transpose() fill unequal list lengths with NaN automatically
    pd.DataFrame.from_dict(flat_data, orient='index').transpose().to_csv(os.path.join(folder, "data.csv"), index=False)
    
    print(f"Data successfully saved/updated in {filepath}")


def plot_data(folder="plots", filename="all_results.json", window_size=10):
    """
    Reads the centralized JSON database and generates a comprehensive grid plot 
    comparing all stored algorithms across all recorded environments.
    
    Features:
    - Automatically scales the number of subplots based on the environments found.
    - Calculates and plots a Moving Average (MA) for smooth learning curves.
    - Adds a shaded area representing the standard deviation (rolling STD) for variance.
    - Falls back to plotting raw data if a run has fewer episodes than the window_size.
    
    Args:
        folder (str, optional): The directory where the JSON file is located and where 
                                the plot will be saved. Defaults to "plots".
        filename (str, optional): The name of the central JSON master file to read from. 
                                  Defaults to "all_results.json".
        window_size (int, optional): The rolling window size for calculating the moving 
                                     average and standard deviation. Defaults to 10.
                                     
    Outputs:
        - learning_curves.png: A high-resolution (300 DPI) image containing all subplots.
    """
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        print(f"Error: Could not find {filepath}. Please save data first.")
        return

    # Load data from the master document
    with open(filepath, 'r') as f:
        rewards_dict = json.load(f)
        
    num_envs = len(rewards_dict)
    if num_envs == 0:
        print("The data document is empty.")
        return
        
    # Create the dynamic plot grid
    fig, axs = plt.subplots(1, num_envs, figsize=(6 * num_envs, 5))
    
    # Ensure axs is iterable even if there is only one environment
    if num_envs == 1:
        axs = [axs]
        
    for i, (env_name, algos) in enumerate(rewards_dict.items()):
        ax = axs[i]
        ax.set_title(env_name)
        
        for algo_name, runs in algos.items():
            # 'runs' is a list of lists (e.g., 5 seeds x 100 episodes).
            # Convert to a 2D numpy array for efficient vector operations.
            runs_array = np.array(runs) 
            
            if runs_array.ndim == 1:
                # If only one run is present
                mean_across_seeds = runs_array
                std_across_seeds = np.zeros_like(runs_array) # No variance with only one run
                label_suffix = "(1 Seed)"
            else:
                # Multi seed data
                mean_across_seeds = np.mean(runs_array, axis=0)
                std_across_seeds = np.std(runs_array, axis=0)
                label_suffix = f"({runs_array.shape[0]} Seeds)"
            
            # 2. Smooth the curves over time (episodes) using a rolling window.
            series_mean = pd.Series(mean_across_seeds)
            series_std = pd.Series(std_across_seeds)
            
            if len(mean_across_seeds) >= window_size:
                # Apply moving average to both the mean and the standard deviation
                # to reduce noise in the visualization.
                plot_mean = series_mean.rolling(window=window_size).mean()
                plot_std = series_std.rolling(window=window_size).mean()
            else:
                # Use raw data if the number of episodes is smaller than the window size.
                plot_mean = series_mean
                plot_std = series_std
                
            # Plot the mean curve (represents the average performance across seeds).
            line, = ax.plot(plot_mean, label=f'{algo_name}', linewidth=2)
            
            # Add a shaded area representing the standard deviation (confidence interval).
            # This visualizes the performance variance across different random seeds.
            ax.fill_between(
                range(len(plot_mean)), 
                plot_mean - plot_std, 
                plot_mean + plot_std, 
                color=line.get_color(), 
                alpha=0.2 # Set transparency to 20%
            )
                
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Reward")
        ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(folder, "learning_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot successfully updated and saved to {plot_path}")