import numpy as np

class RolloutBuffer:
    def __init__(self, buffer_size, state_shape, action_shape):
        self.buffer_size = buffer_size

        self.states   = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.actions  = np.zeros((buffer_size, *action_shape), dtype=np.float32)
        self.rewards  = np.zeros(buffer_size, dtype=np.float32)
        self.logprobs = np.zeros(buffer_size, dtype=np.float32)
        self.values   = np.zeros(buffer_size, dtype=np.float32)
        self.dones    = np.zeros(buffer_size, dtype=np.float32)

        # To be filled after rollout
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns    = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0

    def store(self, state, action, reward, logprob, value, done):
        self.states[self.ptr]   = state
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.logprobs[self.ptr] = logprob
        self.values[self.ptr]   = value
        self.dones[self.ptr]    = done
        self.ptr += 1

    def clear(self):
        self.ptr = 0

    def get_batches(self, batch_size):
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)

        for start in range(0, self.ptr, batch_size):
            yield indices[start:start + batch_size]
