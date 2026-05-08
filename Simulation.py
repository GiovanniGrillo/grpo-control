import gymnasium as gym
import shimmy
import torch
import os
import Algorithms.SAC_Robin as SAC
import Algorithms.SAC_Giovanni as SAC_Giovanni
import Algorithms.PPO as PPO
import Algorithms.TD3 as TD3
import Algorithms.GRPO as GRPO
import Algorithms.CGRPO as CGRPO
import Algorithms.GRPO_Giovanni as GRPO_Giovanni
import Algorithms.PPO as PPO
import Algorithms.CGRPO_r1 as AGRPO
import time
import numpy as np
import Plotting as plot
import utils as bf
import random
import time_logger as tl
import torch.multiprocessing as mp

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Helper functions

def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_single_seed(seed, seed_idx, total_seeds, env_name, AgentClass, algo_name, MAX_EPISODES, MAX_STEPS, skip_steps, RECOVERY, checkpoint_dir):
    
    final_path = os.path.join(checkpoint_dir, f"{env_name}_{algo_name}_s{seed}_final.pth")
    if RECOVERY and os.path.exists(final_path):
        print(f"Skipping Seed {seed} - already finished.")
        try:
            data = torch.load(final_path)
            return data.get('eval_rewards', [])
        except:
            pass

    set_seed(seed) # Set the seed for reproducibility for this run
    seed_time = time.time()

    env_max_steps = MAX_STEPS
    if env_name == "CarRacing-v3":
        env_max_steps = MAX_STEPS * skip_steps 

    env = gym.make(env_name, max_episode_steps=env_max_steps)
    eval_env = gym.make(env_name, max_episode_steps=env_max_steps)

    if env_name == "CarRacing-v3":
        env = bf.SkipFrame(env, skip=skip_steps) # Apply frame skipping to speed up training for CarRacing-v3
        env = bf.CarRacingWrapper(env)
        env = bf.CarRacingActionWrapper(env) # Apply the action wrapper to convert continuous actions to discrete for CarRacing-v3a
        eval_env = bf.CarRacingWrapper(eval_env)
        eval_env = bf.CarRacingActionWrapper(eval_env) # Apply the action wrapper to convert continuous actions to discrete for CarRacing-v3a
    
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if isinstance(eval_env.observation_space, gym.spaces.Dict):
        eval_env = gym.wrappers.FlattenObservation(eval_env)
    
    print(f"\n--- Starting run with seed {seed_idx}/{total_seeds} ---")

    safe_env_name = env_name.replace("/", "_")
    ckpt_path = os.path.join(checkpoint_dir, f"{safe_env_name}_{algo_name}_s{seed}_last.pth")
    final_path = os.path.join(checkpoint_dir, f"{safe_env_name}_{algo_name}_s{seed}_final.pth")
    
    if algo_name in ["CGRPO", "AGRPO"]:
        agent = AgentClass(env, N=20, K=2) # Initialize CGRPO with specific parameters
    else:
        agent = AgentClass(env) # Initialize agent with environment
    
    start_episode = 0
    eval_rewards = []

    current_elite_score = 0.0

    if RECOVERY and os.path.exists(ckpt_path):
        print(f"--- Recovery: Loading Checkpoint for Seed {seed} ---")
        # Load the checkpoint and extract the episode number and evaluation rewards to resume training from where it left off
        checkpoint_data = agent.load_checkpoint(ckpt_path)
        start_episode = checkpoint_data.get('episode', 0) + 1
        eval_rewards = checkpoint_data.get('eval_rewards', [])
        print(f"Resuming from Episode {start_episode}")

    print(f"\n--- Run: {algo_name} | Seed: {seed} ---")

    for ep in range(start_episode, MAX_EPISODES):
        # ==========================================
        # 1. TRAINING (Collect data & learn)
        # ==========================================

        track_data = None
        if env_name == "CarRacing-v3" and hasattr(env.unwrapped, 'track'):
            track_seed = 42 + (ep // 20) 
            state, _ = env.reset(seed=track_seed)
            track_data = [(t[2], t[3]) for t in env.unwrapped.track]
            #if hasattr(agent, 'current_track_data'):
            agent.current_track_data = track_data
            
        else:
            state, _ = env.reset()                                  # Reset environment
        # ep_reward = 0
        
        agent.current_episode = ep

        for t in range(MAX_STEPS):
            action = agent.select_action(state)

            next_state, reward, done, truncated, info = env.step(action)
            
            if env_name == "CarRacing-v3":
                # Safely get the car's physical position
                if hasattr(env.unwrapped, 'car') and env.unwrapped.car is not None:
                    pos = (env.unwrapped.car.hull.position[0], env.unwrapped.car.hull.position[1])
                else:
                    pos = (0.0, 0.0)
            else:
                # Fallback for Acrobot
                pos = (np.arctan2(next_state[1], next_state[0]), np.arctan2(next_state[3], next_state[2]))

            # For dm_control: treat truncated (time limit) as episode end
            episode_done = done or truncated
            stats = agent.step(state, action, reward, next_state, episode_done, pos = pos)
            if stats is not None:
                current_elite_score = stats.get("elite_mean", 0)
                print(f"  [Loss] Actor: {stats['actor']:.4f} | Div: {stats['div']:.4f} | Trauma: {stats['trauma']:.4f}")

            state = next_state
            # state, ep_reward = next_state, ep_reward + reward                    
            if episode_done:
                break

        # ==========================================
        # 2. EVALUATION
        # ==========================================
        if algo_name == "AGRPO":
            eval_ep_reward = current_elite_score
        else:
            eval_state, _ = eval_env.reset()
            eval_ep_reward = 0
            for t in range(MAX_STEPS):
                eval_action = agent.select_action(eval_state, evaluate=True) 
                eval_state, r, d, trunc, _ = eval_env.step(eval_action)
                eval_ep_reward += r
                if d or trunc: break

        eval_rewards.append(eval_ep_reward)

        # Checkpointing
        if ep > 0 and ep % 100 == 0:
            agent.save_checkpoint(ckpt_path, ep=ep, eval_rewards=eval_rewards)
            print(f"Checkpoint saved at Episode {ep}")

        # Logging
        step_time = time.time() - seed_time
        Environment_name = env_name.split("/")[-1] # Extract the environment name for cleaner logging
        if ep % 10 == 0: print(f"{Environment_name} Ep {ep}: {eval_ep_reward:.4f} Time: {step_time:.2f}s")

    # Store the rewards under the specific algorithm's name
    agent.save_checkpoint(final_path, ep=MAX_EPISODES-1, eval_rewards=eval_rewards) # Save the final model checkpoint after training is complete
    env.close()
    eval_env.close()
    return eval_rewards

# Main Loop
if __name__ == '__main__':
    
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    tracker = tl.TimeTracker(filename=os.path.join("plots", "run_times.json"))

    DEFAULT_ENV_NAMES = ["dm_control/cartpole-swingup-v0", "dm_control/acrobot-swingup-v0", "CarRacing-v3"]
    ENVS_REGISTRY = {
        "dm_control/cartpole-swingup-v0": "Cartpole",
        "dm_control/acrobot-swingup-v0": "Acrobot",
        "CarRacing-v3": "CarRacing",
    }
    AGENT_REGISTRY = {
        "TD3": TD3.TD3,
        "PPO": PPO.PPO,
        "SAC": SAC.SAC,
        "SAC_Giovanni": SAC_Giovanni.SAC,
        "GRPO": GRPO.GRPO,
        "CGRPO": CGRPO.CGRPO,
        "AGRPO": AGRPO.AGRPO,
        "GRPO_Giovanni": GRPO_Giovanni.GRPO_Giovanni,
    }

    RUN_MODE = os.getenv("RUN_MODE", "full").strip().lower()  # quick | full
    MAX_EPISODES = int(os.getenv("MAX_EPISODES", "500" if RUN_MODE == "full" else "60"))
    MAX_STEPS = int(os.getenv("MAX_STEPS", "1000" if RUN_MODE == "full" else "400"))
    SKIP_STEPS = int(os.getenv("SKIP_STEPS", "4" if RUN_MODE == "full" else "2"))
    NUM_SEEDS = int(os.getenv("NUM_SEEDS", "5" if RUN_MODE == "full" else "2"))
    RECOVERY = os.getenv("RECOVERY", "false").strip().lower() == "true"
    MULTIPROCESSING = os.getenv("MULTIPROCESSING", "false").strip().lower() == "true"

    SHORT_TO_FULL = {v.lower(): k for k, v in ENVS_REGISTRY.items()}

    env_override = os.getenv("ENV_NAMES", "").strip()

    if env_override:
        requested = [name.strip() for name in env_override.split(",") if name.strip()]
        
        ENV_NAMES = []
        for r in requested:
            if r.lower() in SHORT_TO_FULL:
                ENV_NAMES.append(SHORT_TO_FULL[r.lower()])
            else:
                ENV_NAMES.append(r)
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
        agents = [TD3.TD3, PPO.PPO, SAC.SAC, CGRPO.CGRPO, GRPO_Giovanni.GRPO_Giovanni, AGRPO.AGRPO]

    print(f"RUN_MODE={RUN_MODE} | MAX_EPISODES={MAX_EPISODES} | MAX_STEPS={MAX_STEPS} | NUM_SEEDS={NUM_SEEDS}")
    print(f"ENV_NAMES={ENV_NAMES}")
    print(f"AGENTS={[cls.__name__ for cls in agents]}")

    # Set seed for global reproducibility so that all libraries are seeded consistently.
    np.random.seed(42) # Set a global seed for reproducibility of the master seeds
    master_seeds = np.random.randint(size=NUM_SEEDS, low=0, high=10000) # Generate random seeds for reproducibility across runs

    all_results = {}

    for env_name in ENV_NAMES:

        """
        Special handling for CarRacing-v3 to apply the action wrapper and ensure compatibility with the agents.
        See the added methods in Basic_Functions.py for more explanation.
        """

        # Initialize the nested dictionary for this environment right away
        if env_name not in all_results:
            all_results[env_name] = {}

        for AgentClass in agents: # Loop over agents 
            start_time = time.time()
            # Get the name directly from the class
            algo_name = AgentClass.__name__
            
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

            if algo_name not in all_results[env_name]:
                all_results[env_name][algo_name] = []

            print(f"\n--- Training {algo_name} on {env_name} ---")

            if MULTIPROCESSING:
                print(f"\n--- Training {algo_name} on {env_name} (Parallel Seeds) ---")

                args = [
                    (seed, idx, NUM_SEEDS, env_name, AgentClass, algo_name, MAX_EPISODES, MAX_STEPS, SKIP_STEPS, RECOVERY, checkpoint_dir)
                    for idx, seed in enumerate(master_seeds)
                ]

                with mp.Pool(processes=2) as pool:
                    results = pool.starmap(run_single_seed, args)

                all_results[env_name][algo_name] = results
            else:
                for seed_idx, seed in enumerate(master_seeds):
                    eval_rewards = run_single_seed(seed, seed_idx, NUM_SEEDS, env_name, AgentClass, algo_name, MAX_EPISODES, MAX_STEPS, SKIP_STEPS, RECOVERY, checkpoint_dir)
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



    plot.plot_data()