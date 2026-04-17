import gymnasium as gym
import Algorithms.SAC_Robin as SAC
import Algorithms.PPO as PPO
import time
import numpy as np
import Plotting as plot

# Main Loop
ENV_NAMES = ["CartPole-v1", "Acrobot-v1"]#, "CarRacing-v3"]
agents = [PPO.PPO, SAC.SAC] # List of agents


all_results = {}

np.random.seed(42)
start_time = time.time()

for env_name in ENV_NAMES:
    env = gym.make(env_name)
    
    # Initialize the nested dictionary for this environment right away
    if env_name not in all_results:
        all_results[env_name] = {}

    for AgentClass in agents: # Loop over agents 
        
        agent = AgentClass(env) # Initialize agent with environment

        # Get the name directly from the class
        algo_name = AgentClass.__name__
        print(f"\n--- Training {algo_name} on {env_name} ---")
        
        env_rewards = []

        for ep in range(500):                                       # Episode loop 500
            state, _ = env.reset()                                  # Reset environment
            ep_reward = 0

            for t in range(10000):                                     # Step loop 1000
                action = agent.select_action(state)

                next_state, reward, done, truncated, _ = env.step(action)

                agent.step(state, action, reward, next_state, done)
                
                state, ep_reward = next_state, ep_reward + reward
                if done or truncated: break
            
            env_rewards.append(ep_reward)
            if ep % 5 == 0: print(f"{env_name} Ep {ep}: {ep_reward} Time: {time.time() - start_time:.2f}s")

        # Store the rewards under the specific algorithm's name
        all_results[env_name][algo_name] = env_rewards        
    
    env.close()

plot.save_and_plot_results(all_results)