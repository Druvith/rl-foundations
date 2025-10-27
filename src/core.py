import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from collections import deque
import random
import psutil
import sys
import os


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards (np.ndarray): Rewards for each timestep.
        values (np.ndarray): Value estimates for each timestep.
        gamma (float): Discount factor.
        lam (float): GAE lambda parameter.
    
    Returns:
        np.ndarray: Advantage estimates for each timestep.
    """
    advantages = np.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_adv = delta + gamma * lam * last_adv
    return advantages

def reward_to_go(rewards, gamma=0.99, last_val=0):
    """
    Compute rewards-to-go for each timestep in a trajectory.
    
    Args:
        rewards (np.ndarray): Rewards for each timestep.
        gamma (float): Discount factor.
    
    Returns:
        np.ndarray: Rewards-to-go for each timestep.
    """
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running_sum = last_val
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        rtg[t] = running_sum
    return rtg


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError
    
    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError
    
    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
    

class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
    
class MLPGuassianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)
    

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Ensure V has the right shape
    

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on the type of action-space
        if isinstance(action_space, Box):
            self.pi = MLPGuassianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]            
    
class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, handle_timeout_termination=True):
        self.obs_buf = deque(maxlen=buffer_size)
        self.next_obs_buf = deque(maxlen=buffer_size)
        self.action_buf = deque(maxlen=buffer_size)
        self.reward_buf = deque(maxlen=buffer_size)
        self.done_buf = deque(maxlen=buffer_size)
        self.device = device
        self.handle_timeout_termination = handle_timeout_termination    


    def avail_memory(self):
        return psutil.virtual_memory().available  # in bytes
    
    def add(self, obs, next_obs, action, reward, done, info):
        if self.avail_memory() < obs.nbytes + next_obs.nbytes + action.nbytes + reward.nbytes + done.nbytes:
            Warning("Not enough memory to add new experience to ReplayBuffer.")
        else:
            self.obs_buf.append(obs)
            self.next_obs_buf.append(next_obs)
            self.action_buf.append(action)
            self.reward_buf.append(reward)
            self.done_buf.append(done)

    def sample(self, batch_size):
        batch_indices = random.sample(range(len(self.obs_buf)), batch_size)
        obs = torch.tensor(np.array([self.obs_buf[i] for i in batch_indices]), dtype=torch.float32).to(self.device).squeeze(1)
        next_obs = torch.tensor(np.array([self.next_obs_buf[i] for i in batch_indices]), dtype=torch.float32).to(self.device).squeeze(1)
        actions = torch.tensor(np.array([self.action_buf[i] for i in batch_indices]), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array([self.reward_buf[i] for i in batch_indices]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array([self.done_buf[i] for i in batch_indices]), dtype=torch.float32).to(self.device)
        return Batch(obs, next_obs, actions, rewards, dones)

    def reset(self):
        self.obs_buf.clear()
        self.next_obs_buf.clear()
        self.action_buf.clear()
        self.reward_buf.clear()
        self.done_buf.clear()

    def __len__(self):
        return len(self.obs_buf)
    
class Batch:
    def __init__(self, observations, next_observations, actions, rewards, dones):
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones