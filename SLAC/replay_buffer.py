from collections import deque
import numpy as np
import torch


class LazyFrames:
    """
    Stacked frames which never allocate memory to the same frame.
    """

    def __init__(self, frames):
        self._frames = list(frames)

    def __array__(self, dtype):
        return np.array(self._frames, dtype=dtype)

    def __len__(self):
        return len(self._frames)


class SequenceBuffer():
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.clear()
        self._reset_episode = False

    def clear(self):
        self._reset_episode = False
        self.length = 0
        self.state_ = deque(maxlen=self.capacity + 1)
        self.action_ = deque(maxlen=self.capacity)
        self.reward_ = deque(maxlen=self.capacity)
        self.done_ = deque(maxlen=self.capacity)

    def push(self, action, reward, done, next_state):
        assert self._reset_episode
        self.state_.append(next_state)
        self.action_.append(action)
        self.reward_.append([reward])
        self.done_.append([done])
        self.length += 1

    def get(self):
        state_ = LazyFrames(self.state_)
        action_ = np.array(self.action_, dtype=np.float32)
        reward_ = np.array(self.reward_, dtype=np.float32)
        done_ = np.array(self.done_, dtype=np.float32)
        return state_, action_, reward_, done_

    def reset_episode(self, state):
        assert not self._reset_episode
        self._reset_episode = True
        self.state_.append(state)

    def __len__(self):
        return self.length

    def is_empty(self):
        return self.length == 0

    def is_full(self):
        return self.length == self.capacity


class ReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(self, buffer_size, num_sequences, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.num_sequences = num_sequences
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device

        # Store the sequence of images as a list of LazyFrames on CPU. It can store images with 9 times less memory.
        self.state_ = [None] * buffer_size
        # Store other data on GPU to reduce workloads.
        self.action_ = torch.empty(
            buffer_size, num_sequences, *action_shape, device=device)
        self.reward_ = torch.empty(
            buffer_size, num_sequences, 1, device=device)
        self.done_ = torch.empty(buffer_size, num_sequences, 1, device=device)
        # Buffer to store a sequence of trajectories.
        self.buff = SequenceBuffer(capacity=num_sequences)

    def reset_episode(self, state):
        """
        Reset the buffer and set the initial observation. This has to be done before every episode starts.
        """
        self.buff.reset_episode(state)

    def append(self, action, reward, done, next_state, episode_done):
        """
        Store trajectory in the buffer. If the buffer is full, the sequence of trajectories is stored in replay buffer.
        Please pass 'masked' and 'true' done so that we can assert if the start/end of an episode is handled properly.
        """
        self.buff.push(action, reward, done, next_state)

        if self.buff.is_full():
            state_, action_, reward_, done_ = self.buff.get()
            self._append(state_, action_, reward_, done_)

        if episode_done:
            self.buff.clear()

    def _append(self, state_, action_, reward_, done_):
        self.state_[self._p] = state_
        self.action_[self._p].copy_(torch.from_numpy(action_))
        self.reward_[self._p].copy_(torch.from_numpy(reward_))
        self.done_[self._p].copy_(torch.from_numpy(done_))

        self._n = min(self._n + 1, self.buffer_size)
        self._p = (self._p + 1) % self.buffer_size

    def sample_latent(self, batch_size):
        """
        Sample trajectories for updating latent variable model.
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = np.empty((batch_size, self.num_sequences +
                           1, *self.state_shape), dtype=np.uint8)
        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state_[idx]
        state_ = torch.tensor(state_, dtype=torch.uint8,
                              device=self.device).float().div_(255.0)
        return state_, self.action_[idxes], self.reward_[idxes], self.done_[idxes]

    def sample_sac(self, batch_size):
        """
        Sample trajectories for updating SAC.
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = np.empty((batch_size, self.num_sequences +
                           1, *self.state_shape), dtype=np.uint8)
        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state_[idx]
        state_ = torch.tensor(state_, dtype=torch.uint8,
                              device=self.device).float().div_(255.0)
        return state_, self.action_[idxes], self.reward_[idxes, -1], self.done_[idxes, -1]

    def __len__(self):
        return self._n
