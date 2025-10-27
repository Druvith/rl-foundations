import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import tyro
from dataclasses import dataclass
import os

# Assuming QNetwork is defined here or imported from a common module
# For simplicity, copying it from dqn.py for now.
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n)
        )

    def forward(self, x):
        return self.network(x)

@dataclass
class Args:
    env_id: str = "CartPole-v1"
    seed: int = 1
    model_path: str = "checkpoints/CartPole-v1__dqn__1__1761538491_dqn_499999.pt" # Path to your saved model
    num_episodes: int = 10

def main():
    args = tyro.cli(Args)

    # Setup environment
    env = gym.make(args.env_id, render_mode="human")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    # Load Q-Network
    q_network = QNetwork(env).to("cpu") # Run on CPU for watching
    if os.path.exists(args.model_path):
        q_network.load_state_dict(torch.load(args.model_path, map_location="cpu"))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Warning: Model not found at {args.model_path}. Using untrained network.")
    q_network.eval() # Set to evaluation mode

    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0

        while not done:
            # Get action from Q-network (greedy)
            with torch.no_grad():
                # Ensure obs has the correct shape (batch_size, obs_dim)
                obs_tensor = torch.Tensor(obs).unsqueeze(0) # Add batch dimension
                q_values = q_network(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            env.render() # Render the environment

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
