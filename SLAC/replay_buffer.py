from collections import deque


class SequentialBuffer():
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.length = 0
        self.state = deque(maxlen=self.capacity + 1)
        self.action = deque(maxlen=self.capacity)
        self.reward = deque(maxlen=self.capacity)
        self.done = deque(maxlen=self.capacity)

    def push(self, action, reward, done, next_state):
        self.state.append(next_state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.length += 1

    def get(self):
        return self.state, self.action, self.reward, self.done

    def __len__(self):
        return self.length
