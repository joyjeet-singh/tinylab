"""
save_eyeball_frames.py -- the look-with-your-eyes companion to Check B.

Check B measured toy renders landing ~25x the real cloud's own spacing away
from any real latent. This saves the two art styles side by side so the gap
can be SEEN: top row, three real frames from the training file; bottom row,
three toy renders at arbitrary arena positions.

Run from the tinylab folder:
    python3 save_eyeball_frames.py
    python3 save_eyeball_frames.py --h5 ~/Downloads/tworoom.h5 --img-size 224
Writes eyeball_real_vs_toy.png next to the scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from make_toy_tworoom import ToyTwoRoom


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="~/Downloads/tworoom.h5")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)
    with h5py.File(str(Path(args.h5).expanduser()), "r") as f:
        picks = np.sort(rng.choice(f["pixels"].shape[0], size=3, replace=False))
        real = f["pixels"][picks]

    env = ToyTwoRoom()
    lo, hi = env.margin, env.size - env.margin
    toy = [env.render(rng.uniform(lo, hi, 2).astype(np.float32), args.img_size)
           for _ in range(3)]

    fig, ax = plt.subplots(2, 3, figsize=(9, 6.4))
    for i in range(3):
        ax[0, i].imshow(real[i]); ax[0, i].set_title(f"real frame {int(picks[i])}")
        ax[1, i].imshow(toy[i]); ax[1, i].set_title("toy render")
    for a in ax.flat:
        a.axis("off")
    fig.suptitle("training world (top) vs planner-eval world (bottom)")
    fig.tight_layout()
    out = "eyeball_real_vs_toy.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
