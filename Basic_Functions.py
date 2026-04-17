import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

class CarRacingActionWrapper(gym.ActionWrapper):
    """
    Translates the standard agent output [-1, 1] into the format expected by CarRacing-v3.
    CarRacing expects:
    - Steering: [-1, 1]
    - Gas: [0, 1]
    - Brake: [0, 1]
    """
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action):
        # Scale gas (action[1]) and brake (action[2]) from [-1, 1] to [0, 1]
        return np.array([action[0], (action[1] + 1) / 2, (action[2] + 1) / 2])

class FeatureExtractor(nn.Module):
    """
    Automated Feature Extractor: 
    Detects if the input is an image (3D) or a vector (1D) and applies CNN or Identity.
    """
    def __init__(self, observation_space):
        super(FeatureExtractor, self).__init__()
        
        # Check for image input (Height, Width, Channels)
        if len(observation_space.shape) == 3:
            self.is_image = True
            h, w, c = observation_space.shape # Gymnasium standard: (H, W, C)
            
            self.cnn = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten()
            )
            
            # Compute output dimension of CNN
            with torch.no_grad():
                dummy_input = torch.zeros(1, c, h, w)
                self.feature_dim = self.cnn(dummy_input).shape[1]
        else:
            self.is_image = False
            self.feature_dim = observation_space.shape[0]

    def forward(self, x):
        """
        Defines the forward pass of the neural network.
        """
        if self.is_image:
            # Permute from (Batch, H, W, C) to (Batch, C, H, W) and normalize pixels to [0, 1]
            x = x.permute(0, 3, 1, 2) / 255.0
            return self.cnn(x)
        
        # If it's just a vector (like in CartPole), return it as is
        return x