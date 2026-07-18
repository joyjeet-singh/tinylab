"""Load CIFAR-10 and carve out the fixed subsets named in the config.

House rule: WHICH images we use depends only on data_seed, so every
run races on identical data. The run seed never touches this file.
"""
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

# ToTensor: pixels 0..255 -> floats 0..1, shaped (3, 32, 32)
# Normalize: shift/scale to roughly -1..1; nets learn best near zero
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def load_subsets(cfg):
    d = cfg["data"]

    # download=True fetches CIFAR-10 into data/ the first time (~170 MB),
    # then reuses the local copy forever. data/ is gitignored: by-product.
    full_train = datasets.CIFAR10("data", train=True, download=True, transform=TRANSFORM)
    full_test = datasets.CIFAR10("data", train=False, download=True, transform=TRANSFORM)

    # The PRIVATE dice cup: a local generator seeded with data_seed.
    # Immune to set_seed and everything else in the program.
    g = torch.Generator().manual_seed(d["data_seed"])

    train_idx = torch.randperm(len(full_train), generator=g)[: d["train_size"]]
    test_idx = torch.randperm(len(full_test), generator=g)[: d["test_size"]]

    # Subset = "the same dataset, restricted to these positions"
    return Subset(full_train, train_idx.tolist()), Subset(full_test, test_idx.tolist())
