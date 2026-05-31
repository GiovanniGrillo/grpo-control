from re import A
from tabnanny import check

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
import Algorithms.EGRPO as EGRPO
import Algorithms.AGRPO_archive as AGRPO
import Algorithms.AGRPO_memory as AGRPO_mem
import time
import numpy as np
import utils as bf
import random
import inspect
import logger # Replaces Plotting and time_logger
import torch.multiprocessing as mp
import warnings
import platform

os.environ["USE_NNPACK"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_single_seed(seed, seed_idx, total_seeds, env_name, AgentClass, algo_name, MAX_EPISODES, MAX_STEPS, skip_steps, RECOVERY, checkpoint_dir):
    safe_env_name = env_name.replace("/", "_")
    final_path = os.path.join(checkpoint_dir, f"{safe_env_name}_{algo_name}_s{seed}_final.pth")
    if RECOVERY and os.path.exists(final_path):
        print(f"Skipping Seed {seed} - already finished.")
        try:
            data = torch.load(final_path)
            return data.get('eval_rewards', []), []
        except:
            pass

    set_seed(seed) 
    seed_time = time.time()

    env_max_steps = MAX_STEPS
    if env_name == "CarRacing-v3":
        env_max_steps = MAX_STEPS * skip_steps 

    env = gym.make(env_name, max_episode_steps=env_max_steps)
    eval_env = gym.make(env_name, max_episode_steps=env_max_steps)

    if env_name == "CarRacing-v3":
        env = bf.SkipFrame(env, skip=skip_steps) 
        env = bf.CarRacingWrapper(env)
        env = bf.CarRacingActionWrapper(env) 
        eval_env = bf.CarRacingWrapper(eval_env)
        eval_env = bf.CarRacingActionWrapper(eval_env) 
    
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    if isinstance(eval_env.observation_space, gym.spaces.Dict):
        eval_env = gym.wrappers.FlattenObservation(eval_env)
    
    print(f"\n--- Starting run with seed {seed_idx}/{total_seeds} ---")

    ckpt_path = os.path.join(checkpoint_dir, f"{safe_env_name}_{algo_name}_s{seed}_last.pth")
    
    agent = AgentClass(env) 
    
    start_episode = 0
    eval_rewards = []
    seed_logs = [] 

    current_elite_score = 0.0

    if RECOVERY and os.path.exists(ckpt_path):
        print(f"--- Recovery: Loading Checkpoint for Seed {seed} ---")
        checkpoint_data = agent.load_checkpoint(ckpt_path)
        start_episode = checkpoint_data.get('episode', 0) + 1
        eval_rewards = checkpoint_data.get('eval_rewards', [])
        
        seed_logs = checkpoint_data.get('seed_logs', []) 
        print(f"Resuming from Episode {start_episode}")

    print(f"\n--- Run: {algo_name} on {env_name} | Seed: {seed} ---")

    early_stop = False

    checkpoint_counter = 0
        
    for ep in range(start_episode, MAX_EPISODES):
        
        checkpoint_counter += 1

        # track_data = None
        # if env_name == "CarRacing-v3" and hasattr(env.unwrapped, 'track'):
        #     track_seed = 42 + (ep // 20) 
        #     state, _ = env.reset(seed=track_seed)
        #     track_data = [(t[2], t[3]) for t in env.unwrapped.track]
        #     agent.current_track_data = track_data
        # else:
        #     state, _ = env.reset()
        
        state, _ = env.reset()
        
        agent.current_episode = ep
        update_stats = None # Set to None initially to avoid logging empty dictionaries
        step_time = 0.0

        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            if env_name == "CarRacing-v3":
                if hasattr(env.unwrapped, 'car') and env.unwrapped.car is not None:
                    pos = (env.unwrapped.car.hull.position[0], env.unwrapped.car.hull.position[1])
                else:
                    pos = (0.0, 0.0)
            else:
                pos = (np.arctan2(next_state[1], next_state[0]), np.arctan2(next_state[3], next_state[2]))

            episode_done = done or truncated
            stats = agent.step(state, action, reward, next_state, episode_done, pos=pos)
            
            step_time = time.time() - seed_time

            # Capture stats if the population update was triggered this step
            if stats is not None:
                update_stats = stats 
                current_elite_score = stats.get("elite_mean", 0)
                print(f"\rEpisode {ep} | Time: {step_time:.2f}s         ", end="", flush=True)

                agent_n = getattr(agent, 'N', 0)
                if agent_n > 0 and (MAX_EPISODES - ep - 1) < agent_n:
                    early_stop = True

            state = next_state
            if episode_done:
                break
        
        # Check the flag to see if an update occurred at the end of this episode
        updated = agent.consume_update_flag()

        # ---------------------------------------------------------------------
        # EVALUATION, LOGGING & CHECKPOINTING (Executes ONLY after a full generation)
        # ---------------------------------------------------------------------
        if updated and update_stats is not None: 
            
            # 1. Evaluation
            if hasattr(agent, 'set_eval_mode'):
                agent.set_eval_mode()
            elif hasattr(agent, 'ref_actors'):
                agent.ref_actors.eval() # Fallback für deine alten GRPO Agenten
            
            eval_ep_reward = 0
            num_eval_episodes = 5 
            
            for _ in range(num_eval_episodes):
                # FIX: Reset MUST happen inside the loop for each new evaluation run
                # NOTE: If you want to evaluate on the exact memorized track, add seed=seed here.
                eval_state, _ = eval_env.reset() 
                
                ep_reward = 0
                for t in range(MAX_STEPS):
                    eval_action = agent.select_action(eval_state, evaluate=True) 
                    eval_state, r, d, trunc, _ = eval_env.step(eval_action)
                    ep_reward += r
                    if d or trunc: break
                
                eval_ep_reward += ep_reward
                
            true_eval_score = eval_ep_reward / num_eval_episodes 
            eval_rewards.append(true_eval_score)
            
            if hasattr(agent, 'set_train_mode'):
                agent.set_train_mode()
            elif hasattr(agent, 'ref_actors'):
                agent.ref_actors.train()
                
            print(f"\n   [Evaluation] Champion Average Score ({num_eval_episodes} runs): {true_eval_score:.2f}")
            
            # 2. Checkpointing
            if checkpoint_counter >= 100:
                checkpoint_counter = 0
                agent.save_checkpoint(ckpt_path, ep=ep, eval_rewards=eval_rewards, seed_logs=seed_logs)
                print(f"   [System] Checkpoint saved at Episode {ep}\n")
            
            # 3. Telemetry Logging
            # This ensures logs are strictly tied to updates, reducing file size drastically
            seed_logs.append({
                "seed": seed,
                "episode": ep,
                "eval_reward": true_eval_score, # FIX: Log the average, not the sum
                "step_time_s": round(step_time, 2),
                **update_stats
            })
    
        if early_stop:
            print(f"Ending run early at episode {ep}: Not enough episodes left ({MAX_EPISODES - ep - 1}) for another full update cycle (N={getattr(agent, 'N', 0)}).")
            break

    agent.save_checkpoint(final_path, ep=MAX_EPISODES-1, eval_rewards=eval_rewards, seed_logs=seed_logs) 
    env.close()
    eval_env.close()
    return eval_rewards, seed_logs

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    DEFAULT_ENV_NAMES = ["dm_control/cartpole-swingup-v0", 
        "dm_control/acrobot-swingup-v0", "CarRacing-v3"]
    ENVS_REGISTRY = {
        "dm_control/cartpole-swingup-v0": "Cartpole",
        "dm_control/acrobot-swingup-v0": "Acrobot",
        "CarRacing-v3": "CarRacing",
    }
    AGENT_REGISTRY = {
        "TD3": TD3.TD3,
        "PPO": PPO.PPO,
        "SAC": SAC.SAC,
        "CGRPO": CGRPO.CGRPO,
        "EGRPO": EGRPO.EGRPO,
        "AGRPO": AGRPO.AGRPO,
        "AGRPO_mem": AGRPO_mem.AGRPO
    }

    RUN_MODE = os.getenv("RUN_MODE", "full").strip().lower() 
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
        ENV_NAMES = [SHORT_TO_FULL.get(r.lower(), r) for r in requested]
    else:
        ENV_NAMES = DEFAULT_ENV_NAMES

    agent_override = os.getenv("AGENTS", "").strip()
    if agent_override:
        requested = [name.strip() for name in agent_override.split(",") if name.strip()]
        agents = [AGENT_REGISTRY[name] for name in requested]
    else:
        agents = [EGRPO.AGRPO]

    print(f"RUN_MODE={RUN_MODE} | AGENTS={agents} | MAX_EPISODES={MAX_EPISODES} | MAX_STEPS={MAX_STEPS} | NUM_SEEDS={NUM_SEEDS}")

    np.random.seed(42) 
    master_seeds = np.random.randint(size=NUM_SEEDS, low=0, high=10000) 

    for env_name in ENV_NAMES:
        for AgentClass in agents: 
            start_time = time.time()
            algo_name = AgentClass.__name__
            
            exp_logger = logger.ExperimentLogger(env_name, algo_name)
            
            # Fetch parameters dynamically from the agent, default to 0 if they don't exist
            sig = inspect.signature(AgentClass.__init__)

            def get_default(param_name, fallback=0):
                if param_name in sig.parameters:
                    val = sig.parameters[param_name].default
                    if val is not inspect.Parameter.empty:
                        return val
                return fallback

            agent_params = {
                "N": get_default('N', 0),
                "K": get_default('K', 0),
                "lr": get_default('lr', 0),
                "epsilon": get_default('epsilon', 0),
                "lam_s": get_default('lam_s', 0),
                "lam_d": get_default('lam_d', 0),
                "lam_t": get_default('lam_t', 0),
                "gamma": get_default('gamma', 0),
                "dbscan_eps": get_default('dbscan_eps', 0),
                "warmup_episodes": get_default('warmup_episodes', 0)
            }
            
            config_dict = {
                "Env": env_name,
                "Algo": algo_name,
                "Master_Seeds": master_seeds.tolist(),
                "Max_Episodes": MAX_EPISODES,
                "Max_Steps": MAX_STEPS
            }
            
            # Merge and save
            config_dict.update(agent_params)
            exp_logger.save_config(config_dict)
            
            # Backup Source Code 
            files_to_backup = ["Simulation.py", "utils.py", f"Algorithms/{algo_name}.py"]
            exp_logger.copy_source_code(files_to_backup)
            
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)

            print(f"\n--- Training {algo_name} on {env_name} ---")

            if MULTIPROCESSING:
                args = [
                    (seed, idx, NUM_SEEDS, env_name, AgentClass, algo_name, MAX_EPISODES, MAX_STEPS, SKIP_STEPS, RECOVERY, checkpoint_dir)
                    for idx, seed in enumerate(master_seeds)
                ]
                with mp.Pool(processes=2) as pool:
                    results = pool.starmap(run_single_seed, args)
                
                # Unpack and log results
                for (eval_rewards, seed_logs) in results:
                    exp_logger.metrics.extend(seed_logs)
            else:
                for seed_idx, seed in enumerate(master_seeds):
                    eval_rewards, seed_logs = run_single_seed(seed, seed_idx, NUM_SEEDS, env_name, AgentClass, algo_name, MAX_EPISODES, MAX_STEPS, SKIP_STEPS, RECOVERY, checkpoint_dir)
                    exp_logger.metrics.extend(seed_logs)

            # Save all metrics to the experiment folder
            exp_logger.save_metrics()
            
            run_time = time.time() - start_time
            exp_logger.log_total_time(run_time)
            print(f"--- Runningtime for {algo_name} in {env_name}: {run_time:.2f}s ---")