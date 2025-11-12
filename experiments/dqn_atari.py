import os
import random
import time
from dataclasses import dataclass
from typing import Optional
import sys

from networkx import config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"Adding project root to sys.path: {project_root}")
sys.path.insert(0, project_root)

import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro 
import wandb
from src.core import ReplayBuffer   

from src.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    mps: bool = True
    wandb_project_name: str = "rl-foundations"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = True 

    env_id: str = "ALE/Breakout-v5"
    total_timesteps: int = 5000000
    learning_rate: float = 1e-4
    num_envs: int = 1
    buffer_size: int = 1000000
    gamma: float = 0.99
    #tau: float = 0.005
    target_update_frequency: int = 10000
    batch_size: int = 32
    start_e: float = 1.0
    end_e: float = 0.1
    exploration_fraction: float = 0.5
    learning_starts: int = 20000
    train_frequency: int = 4
    load_weights_path: Optional[str] = "checkpoints/ALE/Breakout-v5__dqn_atari__1__1762918893/dqn_atari.pt"

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = NoopResetEnv(env, noop_max=30)            # 1. random no-ops at start
        env = MaxAndSkipEnv(env, skip=4)                     # 2. frame-skip + max-pool
        env = EpisodicLifeEnv(env)                           # 3. life-loss -> terminal for learning
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = ClipRewardEnv(env)

        env = gym.wrappers.FrameStackObservation(env, 4)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env.action_space.seed(seed)
        return env
    
    return thunk

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
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(),
                nn.Linear(512, env.single_action_space.n)
        )

    def forward(self, x):
        return self.network(x / 255.0)  # Normalize pixel values
    

    
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(end_e, slope * t + start_e) # floor at end_e. We dont way to end the exploration entirely.

def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0, reduction: str = 'mean') -> torch.Tensor:
    err = pred - target
    abs_err = err.abs()
    quadratic = 0.5 * err.pow(2)
    linear = delta * (abs_err - 0.5 * delta)
    loss = torch.where(abs_err <= delta, quadratic, linear)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "Only single environment is supported in this implementation."
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        monitor_gym=True,
        save_code=True,
    )
    from collections import deque
    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    device = torch.device("mps" if torch.backends.mps.is_available() and args.mps else "cpu")
    print(f"Using device: {device}")

    # create envs
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    
    # create Q-network and target Q-network
    q_network = QNetwork(envs).to(device)
    if args.load_weights_path is not None:
        q_network.load_state_dict(torch.load(args.load_weights_path))
        print(f"Loaded weights from {args.load_weights_path}")

    target_q_network = QNetwork(envs).to(device)
    target_q_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    # create replay buffer
    replay_buffer = ReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
    )

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)
    #print(f"DEBUG: obs shape = {obs.shape}")  # Should be (1, 4, 84, 84)
    for global_step in range(args.total_timesteps):
        # Sample action
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        else:
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).to(device)
                q_values = q_network(obs_tensor)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Step envs
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)

        if "episode" in infos and "_episode" in infos:
            i = infos["_episode"]          # env index that ended
            ep = infos["episode"]
            ep_ret = float(ep["r"][i])
            ep_len = int(ep["l"][i])
            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)
            if global_step % 100 == 0:
                wandb.log({
                    "episode_return_mean_100": np.mean(episode_returns),
                    "episode_length_mean_100": np.mean(episode_lengths),
                    "epsilon": epsilon,
                    "sps": int(global_step / (time.time() - start_time)),
                }, step=global_step)


        dones = np.logical_or(terminateds, truncateds)

        # Store in replay buffer
        replay_buffer.add(obs, next_obs, actions, rewards, dones, infos)

        wandb.log({
            "replay_buffer_size": len(replay_buffer)
        }, step=global_step
        )
        obs = next_obs

        # Train agent after collecting sufficient data
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            batch = replay_buffer.sample(args.batch_size)
            obs_batch = batch.observations
            actions_batch = batch.actions.long()
            rewards_batch = batch.rewards
            next_obs_batch = batch.next_observations
            dones_batch = batch.dones

            # Compute current Q values
            current_q_values = q_network(obs_batch).gather(1, actions_batch).squeeze(1)

            # Compute target Q values
            with torch.no_grad():
                next_q_online = q_network(next_obs_batch)
                next_actions = torch.argmax(next_q_online, dim=1, keepdim=True)
                next_q_target = target_q_network(next_obs_batch)
                next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
                target_q_values = rewards_batch.flatten() + args.gamma * next_q_values * (1 - dones_batch.flatten())
            # Compute loss
            loss = huber_loss(current_q_values, target_q_values)

            if global_step % 1000 == 0:
                wandb.log({
                    "td_loss": loss.item(),
                    "q_value_mean": current_q_values.mean().item(),
                    "q_value_std": current_q_values.std().item(),
                    "q_value_max": current_q_values.max().item(),
                    "q_value_min": current_q_values.min().item(),
                }, step=global_step)
                print(f"Step: {global_step}, TD Loss: {loss.item():.4f}, Q-Value: {current_q_values.mean().item():.4f}")

            # Optimize the Q-network
            optimizer.zero_grad()
            loss.backward()
            # ADD CLIP HERE
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
            # Log gradient norm
            if global_step % 1000 == 0:
                grad_norm = 0.0
                for p in q_network.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** 0.5
                wandb.log({"gradient_norm": grad_norm}, step=global_step)

            optimizer.step()

        # Update target network
        if global_step % 10000 == 0:
            target_q_network.load_state_dict(q_network.state_dict())


    if args.save_model:    
        model_path = f"checkpoints/{run_name}/{args.exp_name}.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(q_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    envs.close()
    wandb.finish()
    replay_buffer.reset()





   