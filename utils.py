import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

class CarRacingActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    
    def action(self, action):
        steering = np.clip(action[0], -1.0, 1.0)

        # Map [-1, 1] to [0, 1]. Zero network output = zero throttle.
        gas = np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0)

        # Map [-1, 1] to [0, 1]. Zero network output = zero brake.
        brake = np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0)

        return np.array([steering, gas, brake])
    
    

class FeatureExtractor(nn.Module):
    #Automated Feature Extractor: 
    #Detects if the input is an image (3D) or a vector (1D) and applies CNN or Identity.
    def __init__(self, observation_space):
        super(FeatureExtractor, self).__init__()
        
        # Check for image input (3 dimensions)
        if len(observation_space.shape) == 3:
            self.is_image = True
            
            # Auto-detect channel position: (C, H, W) from wrappers OR (H, W, C) from raw Gym
            if observation_space.shape[0] <= 4: 
                c, h, w = observation_space.shape
                self.is_chw = True
            else:
                h, w, c = observation_space.shape
                self.is_chw = False
            
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten()
            )
            
            # Compute output dimension of CNN dynamically
            with torch.no_grad():
                dummy_input = torch.zeros(1, c, h, w)
                self.feature_dim = self.cnn(dummy_input).shape[1]
        else:
            self.is_image = False
            self.feature_dim = observation_space.shape[0]

    def forward(self, x):
        if self.is_image:
            # If the input is HWC (raw gym), permute it to PyTorch's preferred CHW
            if not self.is_chw:
                x = x.permute(0, 3, 1, 2)
            
            # If pixel values are still 0-255, normalize them to 0-1
            if x.max() > 1.0:
                x = x / 255.0
                
            return self.cnn(x)
        
        # If it's just a vector (like in CartPole), return it as is
        return x

class CarRacingWrapper(gym.ObservationWrapper):
    #Custom wrapper for CarRacing-v3 to convert RGB images to grayscale and stack the last 4 frames.
    #Turning RGB into grayscale reduces dimensionality and also helps with computational efficiency.
    #Also stacking 4(subject to change) frames allows the agent to capture motion information which can be crucial for a game like CarRacing.
    def __init__(self, env, n_stack: int = 4):
        super().__init__(env)
        self.n_stack = n_stack
        c, h, w = n_stack, 96, 96
        self._frames = np.zeros((n_stack, h, w), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(c, h, w),
            dtype=np.float32,
        )

    def _preprocess(self, obs):
        # obs: (96, 96, 3) uint8
        gray = (
            np.float32(0.299) * obs[..., 0]
            + np.float32(0.587) * obs[..., 1]
            + np.float32(0.114) * obs[..., 2]
        ) * np.float32(1.0 / 255.0)
        return gray.astype(np.float32, copy=False)  # (96, 96)

    def observation(self, obs):
        frame = self._preprocess(obs)
        self._frames[:-1] = self._frames[1:]
        self._frames[-1] = frame
        return self._frames.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self._preprocess(obs)
        self._frames[...] = frame
        return self._frames.copy(), info
    

class SkipFrame(gym.Wrapper):
    """
    Skips frames to speed up training. 
    The agent chooses an action, and we repeat it for 'skip' frames, summing the reward.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
                
        return obs, total_reward, terminated, truncated, info