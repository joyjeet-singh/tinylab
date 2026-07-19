"""One function that tames every source of randomness we use."""
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)         # Python's built-in dice
    np.random.seed(seed)      # NumPy's dice
    torch.manual_seed(seed)   # PyTorch's dice (weight init, shuffles, ...)
    # CUDA's matrix library only offers deterministic kernels when this is
    # set; without it, the line below crashes at the first GPU matmul.
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Refuse any operation that has no deterministic version:
    # a loud crash now beats a silently different number later.
    torch.use_deterministic_algorithms(True)
