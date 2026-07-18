"""The two racers. Each takes a 32x32 colour image, returns 10 scores
(one per CIFAR-10 class); highest score is the model's guess.

Every size below comes from the recipe card. This file just obeys.
"""
import torch.nn as nn

IMAGE_NUMBERS = 3 * 32 * 32   # 3 colour channels x 32x32 grid = 3072 numbers
NUM_CLASSES = 10


def build_mlp(cfg):
    """Shred the grid into a list, then stack mixing desks."""
    layers = [nn.Flatten()]                # grid -> flat list, neighbours forgotten
    in_size = IMAGE_NUMBERS
    for out_size in cfg["model"]["hidden_sizes"]:
        layers.append(nn.Linear(in_size, out_size))  # every output blends every input
        layers.append(nn.ReLU())                     # the kink: keep positives, zero negatives
        in_size = out_size
    layers.append(nn.Linear(in_size, NUM_CLASSES))   # final 10 scores
    return nn.Sequential(*layers)


def build_cnn(cfg):
    """Slide small pattern-detectors across the image, zooming out as we go."""
    c1, c2 = cfg["model"]["channels"]
    return nn.Sequential(
        # c1 stencils, each 3x3, slid over every position (padding=1 adds a
        # one-pixel border of zeros so the stencil can sit on the edges too)
        nn.Conv2d(3, c1, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),                   # keep strongest in each 2x2: 32x32 -> 16x16
        nn.Conv2d(c1, c2, kernel_size=3, padding=1),   # patterns-of-patterns
        nn.ReLU(),
        nn.MaxPool2d(2),                   # 16x16 -> 8x8
        nn.Flatten(),                      # only NOW unroll to a list
        nn.Linear(c2 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, NUM_CLASSES),
    )


def build_model(cfg):
    name = cfg["model"]["name"]
    if name == "mlp":
        return build_mlp(cfg)
    if name == "cnn":
        return build_cnn(cfg)
    raise ValueError(f"unknown model: {name}")


if __name__ == "__main__":
    import torch
    import yaml

    for path in ["configs/mlp.yaml", "configs/cnn.yaml"]:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg)
        fake_batch = torch.randn(4, 3, 32, 32)   # 4 fake "images" of pure noise
        scores = model(fake_batch)
        knobs = sum(p.numel() for p in model.parameters())
        print(f"{cfg['model']['name']:>3}: scores shape {tuple(scores.shape)}, knobs: {knobs:,}")
