import os
import json
import time
import shutil
import datetime
import pandas as pd

class ExperimentLogger:
    """
    Handles the creation of unique experiment folders, backs up the source code,
    and logs all metrics incrementally to JSON and CSV.
    """
    def __init__(self, env_name, algo_name, base_dir="runs"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_env = env_name.replace("/", "_")
        
        # Create a unique folder for this specific run
        self.run_dir = os.path.join(base_dir, f"{algo_name}_{safe_env}_{timestamp}")
        self.source_dir = os.path.join(self.run_dir, "source_code")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.source_dir, exist_ok=True)
        
        self.metrics = []
        self.env_name = env_name
        self.algo_name = algo_name
        
    def save_config(self, config_dict):
        """Saves hyperparameters and environment setup to a JSON file."""
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
    def copy_source_code(self, files_to_copy):
        """Creates a hard copy of the used source scripts for reproducibility."""
        for file_path in files_to_copy:
            if os.path.exists(file_path):
                # Handle files in subdirectories (like Algorithms/AGRPO.py)
                filename = os.path.basename(file_path)
                shutil.copy(file_path, os.path.join(self.source_dir, filename))
            else:
                print(f"[Logger] Warning: Could not find {file_path} to backup.")
                
    def log_episode(self, seed, episode, eval_reward, step_time, update_stats):
        """Appends a new row of metrics."""
        log_entry = {
            "seed": seed,
            "episode": episode,
            "eval_reward": eval_reward,
            "time_elapsed_s": round(step_time, 2)
        }
        # Merge the detailed statistics from the AGRPO update step
        if update_stats:
            log_entry.update(update_stats)
            
        self.metrics.append(log_entry)
        
    def save_metrics(self):
        """Exports the collected metrics exclusively to a flat CSV."""
        csv_path = os.path.join(self.run_dir, "metrics.csv")
        
        df = pd.DataFrame(self.metrics)
        df.to_csv(csv_path, index=False)
        
    def log_total_time(self, run_time):
        time_path = os.path.join(self.run_dir, "run_time.txt")
        with open(time_path, 'w') as f:
            f.write(f"Total Run Time for {self.algo_name} on {self.env_name}: {run_time:.2f} seconds\n")