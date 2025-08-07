# from typing import Any

import gymnasium
import numpy as np


class OneHotEncoding(gymnasium.Space):
    """
    {0,...,1,...,0}

    Example usage:
    self.observation_space = OneHotEncoding(size=4)
    """

    def __init__(self, size=None, seed: int | np.random.Generator | None = None):
        assert isinstance(size, int) and size > 0
        self.size = size
        gymnasium.Space.__init__(self, (), np.uint8, seed=seed)

    # def sample(self, mask: Any | None = None, probability: Any | None = None):
    #     if mask is not None or probability is not None:
    #         raise NotImplementedError("Masking and probability sampling not supported")

    #     one_hot_vector = np.zeros(self.size)
    #     one_hot_vector[self.np_random.integers(self.size)] = 1
    #     return one_hot_vector

    # def contains(self, x):
    #     if isinstance(x, (list, tuple, np.ndarray)):
    #         number_of_zeros = list(x).count(0)
    #         number_of_ones = list(x).count(1)
    #         return (number_of_zeros == (self.size - 1)) and (number_of_ones == 1)
    #     else:
    #         return False

    # def __repr__(self):
    #     return f"OneHotEncoding({self.size})"

    # def __eq__(self, other):
    #     return self.size == other.size
