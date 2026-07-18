"""Proof the randomness is tamed. Run twice: output must match exactly."""
import yaml
import torch
from torch.utils.data import DataLoader

from seed import set_seed
from data import load_subsets


def fingerprint(cfg_path, run_seed):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(run_seed)

    train_set, _ = load_subsets(cfg)

    # Batch order shuffled by a generator tied to the RUN seed
    g = torch.Generator().manual_seed(run_seed)
    loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"],
                        shuffle=True, generator=g)

    images, labels = next(iter(loader))  # first batch of the run
    print(f"run_seed={run_seed}")
    print("  subset fingerprint (sum of chosen indices):",
          int(torch.tensor(train_set.indices).sum()))
    print("  first 8 labels this run sees:", labels[:8].tolist())
    print("  batch checksum:", round(float(images.sum()), 4))


if __name__ == "__main__":
    fingerprint("configs/mlp.yaml", run_seed=0)
    fingerprint("configs/mlp.yaml", run_seed=1)
