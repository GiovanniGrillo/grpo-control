import gymnasium as gym
import shimmy
import torch
import os
import Algorithms.SAC_Robin as SAC
import Algorithms.PPO as PPO
import Algorithms.TD3 as TD3
import Algorithms.GRPO as GRPO
import Algorithms.CGRPO as CGRPO
import Algorithms.GRPO_Giovanni as GRPO_Giovanni
import time
import numpy as np
import Plotting as plot
import utils as bf
import random
import time_logger as tl

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Main Loop
DEFAULT_ENV_NAMES = ["dm_control/cartpole-swingup-v0", "dm_control/acrobot-swingup-v0"]#["dm_control/cartpole-swingup-v0", "dm_control/acrobot-swingup-v0", "CarRacing-v3"]
AGENT_REGISTRY = {
    "TD3": TD3.TD3,
    "PPO": PPO.PPO,
    "SAC": SAC.SAC,
    "GRPO": GRPO.GRPO,
    "CGRPO": CGRPO.CGRPO,
    "GRPO_Giovanni": GRPO_Giovanni.GRPO_Giovanni,
}

RUN_MODE = os.getenv("RUN_MODE", "full").strip().lower()  # quick | full
MAX_EPISODES = int(os.getenv("MAX_EPISODES", "500" if RUN_MODE == "full" else "60"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "1000" if RUN_MODE == "full" else "400"))
NUM_SEEDS = int(os.getenv("NUM_SEEDS", "3" if RUN_MODE == "full" else "2"))

env_override = os.getenv("ENV_NAMES", "").strip()
if env_override:
    ENV_NAMES = [name.strip() for name in env_override.split(",") if name.strip()]
else:
    ENV_NAMES = DEFAULT_ENV_NAMES

agent_override = os.getenv("AGENTS", "").strip()
if agent_override:
    requested = [name.strip() for name in agent_override.split(",") if name.strip()]
    unknown = [name for name in requested if name not in AGENT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown AGENTS requested: {unknown}. Available: {list(AGENT_REGISTRY.keys())}")
    agents = [AGENT_REGISTRY[name] for name in requested]
else:
    agents = [CGRPO.CGRPO]#TD3.TD3, PPO.PPO, SAC.SAC, CGRPO.CGRPO, GRPO_Giovanni.GRPO_Giovanni]

np.random.seed(42) # Set a global seed for reproducibility of the master seeds
master_seeds = np.random.randint(size=NUM_SEEDS, low=0, high=10000) # Generate random seeds for reproducibility across runs

print(f"RUN_MODE={RUN_MODE} | MAX_EPISODES={MAX_EPISODES} | MAX_STEPS={MAX_STEPS} | NUM_SEEDS={NUM_SEEDS}")
print(f"ENV_NAMES={ENV_NAMES}")
print(f"AGENTS={[cls.__name__ for cls in agents]}")

all_results = {}

# Set seed for global reproducibility so that all libraries are seeded consistently.

def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

tracker = tl.TimeTracker(filename=os.path.join("plots", "run_times.json"))


for env_name in ENV_NAMES:
    env = gym.make(env_name, max_episode_steps=MAX_STEPS)                # Create the training environment with a max episode length of 1000 steps to ensure that episodes terminate
    eval_env = gym.make(env_name, max_episode_steps=MAX_STEPS)           # Separate environment for evaluation to ensure that training and evaluation are independent and do not interfere with each other

    """
    Special handling for CarRacing-v3 to apply the action wrapper and ensure compatibility with the agents.
    See the added methods in Basic_Functions.py for more explanation.
    """
    
    if env_name == "CarRacing-v3":
        env = bf.CarRacingWrapper(env)
        env = bf.CarRacingActionWrapper(env) # Apply the action wrapper to convert continuous actions to discrete for CarRacing-v3a
        eval_env = bf.CarRacingWrapper(eval_env)
        eval_env = bf.CarRacingActionWrapper(eval_env) # Apply the action wrapper to convert continuous actions to discrete for CarRacing-v3a
    
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if isinstance(eval_env.observation_space, gym.spaces.Dict):
        eval_env = gym.wrappers.FlattenObservation(eval_env)

    # Initialize the nested dictionary for this environment right away
    if env_name not in all_results:
        all_results[env_name] = {}

    for AgentClass in agents: # Loop over agents 
        start_time = time.time()
        # Get the name directly from the class
        algo_name = AgentClass.__name__
        
        if algo_name not in all_results[env_name]:
            all_results[env_name][algo_name] = []

        print(f"\n--- Training {algo_name} on {env_name} ---")

        for seed in master_seeds:
            set_seed(seed) # Set the seed for reproducibility for this run
            
            seed_time = time.time()
            agent = AgentClass(env) # Initialize agent with environment
            
            print(f"\n--- Run: {algo_name} | Seed: {seed} ---")

            eval_rewards = []

            for ep in range(MAX_EPISODES):
                # ==========================================
                # 1. TRAINING (Collect data & learn)
                # ==========================================
                state, _ = env.reset()                                  # Reset environment
                # ep_reward = 0

                for t in range(MAX_STEPS):
                    action = agent.select_action(state)

                    next_state, reward, done, truncated, info = env.step(action)
                    
                    # For dm_control: treat truncated (time limit) as episode end
                    episode_done = done or truncated
                    agent.step(state, action, reward, next_state, episode_done)

                    state = next_state
                    # state, ep_reward = next_state, ep_reward + reward
                    if episode_done:
                        break
                
                # ==========================================
                # 2. EVALUATION
                # ==========================================
                eval_state, _ = eval_env.reset()
                eval_ep_reward = 0
                for t in range(MAX_STEPS):
                    eval_action = agent.select_action(eval_state, evaluate=True) 
                    eval_state, r, d, trunc, _ = eval_env.step(eval_action)
                    eval_ep_reward += r
                    if d or trunc: break

                eval_rewards.append(eval_ep_reward)

                step_time = time.time() - seed_time
                Environment_name = env_name.split("/")[-1] # Extract the environment name for cleaner logging
                if ep % 10 == 0: print(f"{Environment_name} Ep {ep}: {eval_ep_reward:.4f} Time: {step_time:.2f}s")

            # Store the rewards under the specific algorithm's name
            all_results[env_name][algo_name].append(eval_rewards)
        
        current_data = {
            env_name: {
                algo_name: all_results[env_name][algo_name]
            }
        }
        plot.save_data(current_data)            # Save the data after each algorithm to ensure that we have intermediate results even if the process is interrupted
        
        run_time = time.time() - start_time
        tracker.log(env_name, algo_name, run_time)
        tracker.save() # Save the timing data after each algorithm to ensure we have intermediate timing results
        print(f"--- Runningtime for {algo_name} in {env_name}: {run_time:.2f}s ---")

        plot.plot_data()
    
    env.close()
    eval_env.close()



plot.plot_data()