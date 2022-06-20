
from collections import deque, namedtuple
import random
import torch
import numpy as np


class ReplayBuffer:
    """
    Experience Replay Buffer，fixed-size buffer to store experience tuples.
    """
    def __init__(self, buffer_size, batch_size, device):
        """
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param device: cpu or gpu
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if self.batch_size > len(self.memory):
            return self.return_all_samples()
        else:

            experiences = random.sample(self.memory, k=self.batch_size)

            states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).to(self.device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
                self.device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).to(
                self.device)

            return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def size(self):
        return len(self.memory)

    def return_all_samples(self):
        all_transitions = list(self.memory)
        state, action, reward, next_state, done = zip(*all_transitions)
        states = torch.from_numpy(np.array(state)).float().to(self.device)
        actions = torch.from_numpy(np.array(action)).unsqueeze(dim=1).to(self.device)
        rewards = torch.from_numpy(np.array(reward)).unsqueeze(dim=1).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_state)).float().to(
            self.device)
        dones = torch.from_numpy(np.array(done).astype(np.uint8)).unsqueeze(dim=1).to(
            self.device)
        return states, actions, rewards, next_states, dones

    def clear(self):
        self.memory.clear()
