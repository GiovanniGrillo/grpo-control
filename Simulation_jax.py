import os
# JAX-Konfiguration: Verhindert Segfaults und GPU-Überbelegung
os.environ["MUJOCO_GL"] = "egl"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Reduziert XLA-Spam
os.environ["JAX_PLATFORMS"] = "cpu"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import shimmy
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import time
import random

import Algorithms.td3_jax.TD3_jax as TD3_jax # Dein neuer JAX-TD3
import Plotting as plot
import utils as bf
import time_logger as tl



# Registry nur mit JAX-Algorithmen
AGENT_REGISTRY = {
    "TD3_jax": TD3_jax.TD3,
}

# Konfiguration
RUN_MODE = os.getenv("RUN_MODE", "full").strip().lower()
MAX_EPISODES = int(os.getenv("MAX_EPISODES", "500" if RUN_MODE == "full" else "60"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "1000" if RUN_MODE == "full" else "400"))
NUM_SEEDS = int(os.getenv("NUM_SEEDS", "5" if RUN_MODE == "full" else "2"))
RECOVERY = os.getenv("RECOVERY", "false").strip().lower() == "true"

DEFAULT_ENV_NAMES = ["dm_control/cartpole-swingup-v0", "dm_control/acrobot-swingup-v0", "CarRacing-v3"]
ENV_NAMES = [name.strip() for name in os.getenv("ENV_NAMES", "").split(",")] if os.getenv("ENV_NAMES") else DEFAULT_ENV_NAMES

# Falls keine Agents spezifiziert, nehmen wir TD3_jax
agents = [TD3_jax.TD3] 

# Globale Master-Seeds
np.random.seed(42)
master_seeds = np.random.randint(size=NUM_SEEDS, low=0, high=10000)

def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    # JAX-Seeding passiert innerhalb der Agenten über PRNGKeys

all_results = {}
tracker = tl.TimeTracker(filename=os.path.join("plots", "run_times.json"))

for env_name in ENV_NAMES:
    env = gym.make(env_name, max_episode_steps=MAX_STEPS)
    eval_env = gym.make(env_name, max_episode_steps=MAX_STEPS)

    # Wrapper für CarRacing
    if env_name == "CarRacing-v3":
        env = bf.CarRacingWrapper(env)
        env = bf.CarRacingActionWrapper(env)
        eval_env = bf.CarRacingWrapper(eval_env)
        eval_env = bf.CarRacingActionWrapper(eval_env)
    
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
        eval_env = gym.wrappers.FlattenObservation(eval_env)

    if env_name not in all_results:
        all_results[env_name] = {}

    for AgentClass in agents:
        start_time = time.time()
        algo_name = AgentClass.__name__
        
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        if algo_name not in all_results[env_name]:
            all_results[env_name][algo_name] = []

        print(f"\n--- Training {algo_name} on {env_name} ---")

        for seed_idx, seed in enumerate(master_seeds):
            print(f"\n--- Starting run with seed {seed_idx + 1}/{len(master_seeds)} ---")
            ckpt_path = os.path.join(checkpoint_dir, f"{env_name}_{algo_name}_s{seed}_last.jax")
            
            set_seed(seed)
            seed_time = time.time()
            
            # Initialisierung des JAX-Agenten-Wrappers
            agent = AgentClass(env)
            
            start_episode = 0
            eval_rewards = []

            # Recovery
            if RECOVERY and os.path.exists(ckpt_path):
                checkpoint_data = agent.load_checkpoint(ckpt_path)
                start_episode = checkpoint_data.get('episode', 0) + 1
                eval_rewards = checkpoint_data.get('eval_rewards', [])

            for ep in range(start_episode, MAX_EPISODES):
                # 1. TRAINING
                state, _ = env.reset()
                for t in range(MAX_STEPS):
                    action = agent.select_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    
                    episode_done = done or truncated
                    agent.step(state, action, reward, next_state, episode_done)
                    
                    state = next_state
                    if episode_done: break
                
                # 2. EVALUATION
                eval_state, _ = eval_env.reset()
                eval_ep_reward = 0
                for t in range(MAX_STEPS):
                    eval_action = agent.select_action(eval_state, evaluate=True) 
                    eval_state, r, d, trunc, _ = eval_env.step(eval_action)
                    eval_ep_reward += r
                    if d or trunc: break

                eval_rewards.append(eval_ep_reward)

                # Logging & Zeitmessung (Wichtig: block_until_ready für JAX)
                if ep % 10 == 0:
                    # Wir erzwingen eine Synchronisation für die korrekte Zeitmessung
                    jax.random.PRNGKey(0).block_until_ready() 
                    step_time = time.time() - seed_time
                    env_short = env_name.split("/")[-1]
                    print(f"{env_short} Ep {ep}: {eval_ep_reward:.2f} | Time: {step_time:.2f}s")

            all_results[env_name][algo_name].append(eval_rewards)
        
        # JAX Synchronisation vor dem Speichern der Gesamtlaufzeit
        jax.random.PRNGKey(0).block_until_ready()
        run_time = time.time() - start_time
        tracker.log(env_name, algo_name, run_time)
        tracker.save()
        
        print(f"--- Runningtime for {algo_name} in {env_name}: {run_time:.2f}s ---")
        plot.save_data({env_name: {algo_name: all_results[env_name][algo_name]}})
    
    env.close()
    eval_env.close()

plot.plot_data()