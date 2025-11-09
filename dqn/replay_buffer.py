import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, capacity=50000, batch_size=64):
        self.capacity = capacity
        self.batch_size = batch_size
        self.state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity,), dtype=np.int64)
        self.reward_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        i = self.ptr % self.capacity
        self.state_buf[i] = state
        self.action_buf[i] = action
        self.reward_buf[i] = reward
        self.next_state_buf[i] = next_state
        self.done_buf[i] = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self):
        return self.size >= self.batch_size

    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return (self.state_buf[idxs],
                self.action_buf[idxs],
                self.reward_buf[idxs],
                self.next_state_buf[idxs],
                self.done_buf[idxs])
