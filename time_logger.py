# time_logger.py
import json
import os

class TimeTracker:
    def __init__(self, filename="run_times.json"):
        self.filename = filename
        self.data = []

    def log(self, env_name, algo_name, run_time):
        self.data.append({
            "env": env_name,
            "algo": algo_name,
            "time": run_time
        })

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=4)

    def get_total_time(self, env_name, algo_name):
        return sum(item['time'] for item in self.data if item['env'] == env_name and item['algo'] == algo_name)