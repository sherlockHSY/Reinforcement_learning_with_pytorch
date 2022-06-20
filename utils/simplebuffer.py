
from collections import deque, namedtuple
import random
import torch
import numpy as np


class SimpleBuffer:
    """
    buffer Replay Buffer，fixed-size buffer to store buffer tuples.
    """
    def __init__(self, buffer_size, device):
        """
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param device: cpu or gpu
        """
        self.device = device
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.buffer = namedtuple("buffer", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new buffer to memory."""
        e = self.buffer(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """sample all buffers from memory."""
        all_transitions = list(self.memory)
        state, action, reward, next_state, done = zip(*all_transitions)
        states = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(action)).unsqueeze(dim=1).to(self.device)
        rewards = torch.tensor(np.array(reward), dtype=torch.float).unsqueeze(dim=1).to(self.device)
        next_states = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(done), dtype=torch.float).unsqueeze(dim=1).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def clear(self):
        self.memory.clear()


