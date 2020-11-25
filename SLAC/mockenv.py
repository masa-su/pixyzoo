import numpy as np


class DummySpace:
    def __init__(self, shape):
        self.shape = shape

    def sample(self):
        return np.random.rand(*self.shape)


class MockEnv:
    def __init__(self, obs_shape, action_shape, dtype='float64', terminal_step=100):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.dtype = dtype
        self.num_step = 0
        self.terminal_step = terminal_step
        self.observation_space = DummySpace(shape=obs_shape)
        self.action_space = DummySpace(shape=action_shape)
        self.action_repeat = 4
        self._max_episode_steps = 1E5

    def step(self, action):
        assert action.shape == self.action_shape
        assert self.num_step < self.terminal_step
        obs = np.random.rand(*self.obs_shape).astype(self.dtype)
        self.num_step += 1
        return obs, 0, self.num_step == self.terminal_step, {}

    def reset(self):
        self.num_step = 0
        return np.random.rand(*self.obs_shape).astype(self.dtype)

    def seed(self, seed):
        pass
