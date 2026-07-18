"""One function that tames every source of randomness we use."""
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)         # Python's built-in dice
    np.random.seed(seed)      # NumPy's dice
    torch.manual_seed(seed)   # PyTorch's dice (weight init, shuffles, ...)
    # Refuse any operation that has no deterministic version:
    # a loud crash now beats a silently different number later.
    torch.use_deterministic_algorithms(True)
