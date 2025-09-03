# This script implements Vanilla-policy-gradient with Generalized-Advantage-estimation 
# in pure pytorch. Agent here is deployed and trained on a gymnasium environment.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
from src.core import compute_gae, reward_to_go, MLPActorCritic
import wandb
from collections import deque

class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.9, lam=0.95):
        self.obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, *act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of the agent-environment interaction to the buffer
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE Calculation
        adv = compute_gae(rews[:-1], vals, gamma=self.gamma, lam=self.lam)
        self.adv_buf[path_slice] = adv

        # reward-to-go calculation
        self.ret_buf[path_slice] = reward_to_go(self.rew_buf[path_slice], self.gamma, last_val)
        self.path_start_idx = self.ptr


    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size # buffer has to be full before you can get 
        self.ptr, self.path_start_idx = 0, 0
        # we also need to normalise the advantages, so we've a smoother gradient flow and updates
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, 
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
    
def vpg(env_fn, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=10,
        steps_per_epoch=4000, epochs=100, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=100, lam=0.97, max_ep_len=1000,
        save_freq=10):
    
    # Set up wandb
    wandb.init(project="rl-foundations", config=locals())

    # Create checkpoint directory
    os.makedirs('experiments/checkpoints', exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Count variables in policy and value networks
    pi_vars = sum(p.numel() for p in ac.pi.parameters())
    v_vars = sum(p.numel() for p in ac.v.parameters())

    print(f"Number of parameters in policy network (ac.pi): {pi_vars}")
    print(f"Number of parameters in value network (ac.v): {v_vars}")

    # Set up experience buffer
    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # Set up the loss function
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        loss_pi = -(logp * adv).mean()

        approx_kl = (logp_old - logp).mean().item() # useful info to monitor the policy changes after the update
        ent = pi.entropy().mean().item()
        pi_info = dict(approx_kl=approx_kl, ent=ent)

        return loss_pi, pi_info
    
    # function to compute the value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()
    
    # Set up optimizers for both policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()

        # Policy update
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Value update
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()
        
        return loss_pi.item(), loss_v.item(), pi_info

    # Prepare for interaction with environment
    start_time = time.time()
    o, _ = env.reset()
    ep_ret, ep_len = 0, 0
    ep_ret_100_deque = deque(maxlen=100)

    # Collect the experience in each env and perform updates
    for epoch in range(epochs):
        epoch_ep_rets = []
        epoch_ep_lens = []
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and store
            buf.store(o, a, r, v, logp)

            # Update obs 
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = terminated or timeout or truncated
            epoch_ended = t == steps_per_epoch - 1
            
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # If trajectory didn't reach the terminal state, then bootstrap the value target
                v = 0
                if not terminated:
                    # Bootstrap the value estimate if the episode was cut short.
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                buf.finish_path(v)
                if terminal:
                    epoch_ep_rets.append(ep_ret)
                    epoch_ep_lens.append(ep_len)
                    ep_ret_100_deque.append(ep_ret)
                o, _ = env.reset()
                ep_ret, ep_len = 0, 0


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            torch.save(ac.state_dict(), f'experiments/checkpoints/vpg_epoch_{epoch}.pt')

        # Perform VPG update
        loss_pi, loss_v, pi_info = update()

        # Log info about epoch
        wandb.log({
            'epoch': epoch,
            'loss_pi': loss_pi,
            'loss_v': loss_v,
            'ep_ret_mean': np.mean(epoch_ep_rets) if epoch_ep_rets else 0,
            'ep_len_mean': np.mean(epoch_ep_lens) if epoch_ep_lens else 0,
            'ep_ret_100_mean': np.mean(ep_ret_100_deque) if ep_ret_100_deque else 0,
            'ep_ret_std': np.std(epoch_ep_rets) if epoch_ep_lens else 0,
            'ep_len_std': np.std(epoch_ep_lens) if epoch_ep_lens else 0,
            'ep_ret_max': np.max(epoch_ep_rets) if epoch_ep_lens else 0,
            'ep_len_max': np.max(epoch_ep_lens) if epoch_ep_lens else 0,
            'ep_ret_min': np.min(epoch_ep_rets) if epoch_ep_lens else 0,
            'ep_len_min': np.min(epoch_ep_lens) if epoch_ep_lens else 0,
            'approx_kl': pi_info['approx_kl'],
            'entropy': pi_info['ent'],
            'time': time.time() - start_time
        })


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    vpg(lambda : gym.make(args.env), actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)
