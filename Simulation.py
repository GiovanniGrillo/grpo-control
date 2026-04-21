import gymnasium as gym
import shimmy
import torch
import Algorithms.SAC_Robin as SAC
import Algorithms.PPO as PPO
import Algorithms.TD3 as TD3
import Algorithms.GRPO as GRPO
import Algorithms.CGRPO as CGRPO
import time
import numpy as np
import Plotting as plot
import utils as bf
import random

# Main Loop
ENV_NAMES = ["dm_control/cartpole-swingup-v0", "dm_control/acrobot-swingup-v0", "CarRacing-v3"] # List of environments ["dm_control/cartpole-swingup-v0"]
agents = [TD3.TD3, PPO.PPO, SAC.SAC, CGRPO.CGRPO] # List of agents GRPO.GRPO

# ENV_NAMES = ["dm_control/cartpole-swingup-v0"]
# agents = [CGRPO.CGRPO]

master_seeds = np.random.randint(size = 5, low=0, high=10000) # Generate 5 random seeds for reproducibility across runs

all_results = {}

# Set seed for global reproducibility so that all libraries are seeded consistently.

def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

for env_name in ENV_NAMES:
    env = gym.make(env_name)
    eval_env = gym.make(env_name) # Separate environment for evaluation to ensure that training and evaluation are independent and do not interfere with each other

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
        # Get the name directly from the class
        algo_name = AgentClass.__name__
        
        if algo_name not in all_results[env_name]:
            all_results[env_name][algo_name] = []

        print(f"\n--- Training {algo_name} on {env_name} ---")

        for seed in master_seeds:
            set_seed(seed) # Set the seed for reproducibility for this run
            
            start_time = time.time()
            agent = AgentClass(env) # Initialize agent with environment
            
            print(f"\n--- Run: {algo_name} | Seed: {seed} ---")

            eval_rewards = []

            for ep in range(1000):                                       # Episode loop 500
                # ==========================================
                # 1. TRAINING (Collect data & learn)
                # ==========================================
                state, _ = env.reset()                                  # Reset environment
                # ep_reward = 0

                for t in range(1000):                                     # Step loop 1000
                    action = agent.select_action(state)

                    next_state, reward, done, truncated, _ = env.step(action)

                    agent.step(state, action, reward, next_state, done or truncated)  # Pass done or truncated to step() for proper episode handling with grpo
                    
                    state = next_state
                    # state, ep_reward = next_state, ep_reward + reward
                    if done or truncated: break
                
                # ==========================================
                # 2. EVALUATION
                # ==========================================
                eval_state, _ = eval_env.reset()
                eval_ep_reward = 0
                for t in range(1000):
                    eval_action = agent.select_action(eval_state, evaluate=True) 
                    eval_state, r, d, trunc, _ = eval_env.step(eval_action)
                    eval_ep_reward += r
                    if d or trunc: break

                eval_rewards.append(eval_ep_reward)

                step_time = time.time() - start_time
                Environment_name = env_name.split("/")[-1] # Extract the environment name for cleaner logging
                if ep % 5 == 0: print(f"{Environment_name} Ep {ep}: {eval_ep_reward:.4f} Time: {step_time:.2f}s")

            # Store the rewards under the specific algorithm's name
            all_results[env_name][algo_name].append(eval_rewards)
        
        current_data = {
            env_name: {
                algo_name: all_results[env_name][algo_name]
            }
        }
        plot.save_data(current_data)            # Save the data after each algorithm to ensure that we have intermediate results even if the process is interrupted
        plot.plot_data()
    
    env.close()
    eval_env.close()

plot.plot_data()