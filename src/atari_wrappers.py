import gymnasium as gym
import numpy as np
from gymnasium import spaces

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        for _ in range(np.random.randint(1, self.noop_max + 1)):
            obs, reward, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info
    
class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, reward, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, reward, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, skip=4):
        super().__init__(env)
        assert env.observation_space.dtype is not None, "Observation space must have a defined dtype."
        self.obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype) 
        self.skip = skip

    def step(self, action):
        total_reward = 0
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self.skip - 2:
                self.obs_buffer[0] = obs
            if i == self.skip - 1:
                self.obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        max_frame = self.obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info
    
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(float(reward))
    
class EpisodicLifeEnv(gym.Wrapper):

    def __init__(self, env : gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info
