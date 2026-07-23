import random
import pickle
from collections import deque
from pathlib import Path

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.random = random.Random(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = self.random.sample(self.buffer, self.count)
        else:
            batch = self.random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def return_buffer(self):
        s = np.array([_[0] for _ in self.buffer])
        a = np.array([_[1] for _ in self.buffer])
        r = np.array([_[2] for _ in self.buffer]).reshape(-1, 1)
        t = np.array([_[3] for _ in self.buffer]).reshape(-1, 1)
        s2 = np.array([_[4] for _ in self.buffer])

        return s, a, r, t, s2

    def clear(self):
        self.buffer.clear()
        self.count = 0

    def save(self, path):
        """Persist experiences and sampling RNG for resumable training."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as file:
            pickle.dump(
                {
                    "buffer_size": self.buffer_size,
                    "buffer": self.buffer,
                    "count": self.count,
                    "random_state": self.random.getstate(),
                },
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary_path.replace(path)

    def load(self, path):
        path = Path(path)
        with path.open("rb") as file:
            state = pickle.load(file)
        self.buffer_size = state["buffer_size"]
        self.buffer = state["buffer"]
        self.count = state["count"]
        self.random.setstate(state["random_state"])
        return self
