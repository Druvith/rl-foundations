import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import tyro
from dataclasses import dataclass
import os
import sys
import time
import ale_py

# Add project root to sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# The Q-Network architecture must match the one used for training in dqn_atari.py
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), # 7*7*64
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        # The input tensor from the environment is (0-255). The training script
        # implicitly scales this to (0-1) by converting the uint8 numpy array to a float tensor.
        # We do the same here.
        return self.network(x / 255.0)

@dataclass
class Args:
    """Args for the evaluation script."""
    env_id: str = "ALE/Breakout-v5"
    """the id of the environment"""
    seed: int = 1
    """seed of the experiment"""
    model_path: str = "checkpoints/ALE/Breakout-v5__dqn_atari__1__1762918893/dqn_atari.pt"
    """path to the saved model. Replace the timestamp with your specific model."""
    num_episodes: int = 10
    """the number of episodes to run"""

def make_env(env_id, seed):
    """
    Creates and wraps the Atari environment for evaluation.
    """
    env = gym.make(env_id, render_mode="human")
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    #env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

def main():
    """
    Main function to run the agent evaluation.
    """
    args = tyro.cli(Args)

    # Setup environment
    env = make_env(args.env_id, args.seed)

    # Load Q-Network
    q_network = QNetwork(env).to("cpu") # Inference on CPU
    
    # Check for placeholder timestamp in model path
    if "__" not in os.path.basename(args.model_path):
         print(f"Warning: The model path '{args.model_path}' seems to be a placeholder.")
         print("Please provide a valid path to a trained model.")
         # Proceeding with an untrained network for demonstration.
    elif os.path.exists(args.model_path):
        q_network.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Warning: Model not found at {args.model_path}. Using an untrained network.")
    
    q_network.eval() # Set to evaluation mode

    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + episode)
        
        done = False
        total_reward = 0

        while not done:
            # Get action from Q-network (greedy policy)
            with torch.no_grad():
                # The observation from the env is a LazyFrame. Convert it to a numpy array,
                # then a tensor, and add a batch dimension.
                obs_tensor = torch.Tensor(np.array(obs)).unsqueeze(0)
                q_values = q_network(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            time.sleep(1/60) # Slow down rendering to make it watchable

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
