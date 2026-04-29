from dataclasses import dataclass, field

from Algorithms.jax_support import BaseModelConfig
from Algorithms.jax_support import BaseOptimizerConfig

@dataclass
class TD3OptimizerConfig(BaseOptimizerConfig):
    pass

@dataclass
class TD3Config(BaseModelConfig):
    policy_noise: float = 0.1
    
    target_noise: float = 0.2
    noise_clip: float = 0.5
    
    policy_delay: int = 2
    
    action_space_low: float = -1.0
    action_space_high: float = 1.0
    
    optimizer: TD3OptimizerConfig = field(default_factory=TD3OptimizerConfig)