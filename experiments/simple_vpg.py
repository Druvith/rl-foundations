import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam    # question: why not use AdamW?
import numpy as np
import gymnasium as gym     # gymanisum is not compatible with python 3.9   
import wandb
# from gym.spaces import Discrete, Box


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):      # tanh is replaced by relu -- tanh's outputs are bounded [-1,1] whereas rely gradients are rich and saturated only on one side
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
    
def train(env_name='CartPole-v1', hidden_sizes=[64], 
          lr=1e-3, epochs=50, batch_size=5000, render=False, log_wandb=True):
    
    # make environment, check spaces, get obs / act dims
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name, render_mode='rgb_array')

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Initialize wandb
    if log_wandb:
        wandb.init(
            project="rl-foundations",
            name="Simple VPG-01",
            config={
                "env_name": env_name,
                "hidden_sizes": hidden_sizes,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "obs_dim": obs_dim,
                "n_acts": n_acts
            }
        )

    #make the core policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+hidden_sizes+[n_acts])

    # function to make action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean() # single scalar value gradient for the whole batch

    optimizer = Adam(logits_net.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = [] # R(tau) - episode length * episode reward
        batch_rets = []   # episode return 
        batch_lens = []   # length of each episode

        obs, _ = env.reset() 
        done = False        # signal from the env that the episode is over
        ep_rews = []

        # render first episode for each epoch
        finished_rendering_this_epoch = False

        while True:

            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _, _ = env.step(act)

            # save action and rewards
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:    # signal from the env that the episode is over
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # The weight for each lop_prob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len  # here the episode return is scaled by the length of the episode

                # reset episode specific variable
                obs, _ = env.reset()
                done, ep_rews = False, []

                # end experience if we have enough of it
                if len(batch_obs) > batch_size:
                    break   


        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=torch.as_tensor(batch_weights, dtype=torch.float32))
        batch_loss.backward()    # computes the gradients for each parameters
        optimizer.step()        # updates the parameters according to our objective func

        return batch_loss, batch_rets, batch_lens
    
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        
        # Calculate metrics
        avg_return = np.mean(batch_rets)
        avg_ep_len = np.mean(batch_lens)
        
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
              (i, batch_loss, avg_return, avg_ep_len))
        
        # Log to wandb
        if log_wandb:
            wandb.log({
                "epoch": i,
                "loss": batch_loss.item(),
                "avg_return": avg_return,
                "avg_episode_length": avg_ep_len,
                "max_return": np.max(batch_rets),
                "min_return": np.min(batch_rets),
                "std_return": np.std(batch_rets)
            })
    
    env.close()
    
    if log_wandb:
        wandb.finish()
        

if __name__ == '__main__':
    train(env_name='CartPole-v1', hidden_sizes=[64], 
          lr=1e-3, epochs=50, batch_size=5000, render=False, log_wandb=True)


