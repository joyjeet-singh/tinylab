"""
verify_repeat_encoding.py -- check the corrected action encoding BEFORE
spending 40 minutes on a rerun.

WHAT CHANGED
------------
Reading the reference's data buffer settled what their 10-wide action is:

    action_idx = base + arange(history_len * frameskip)   # DENSE, every action
    clip['action'] = clip['action'].reshape(history_len, -1)
                                          # -> (history_len, frameskip * action_dim)

So the ten numbers are the FIVE RAW ACTIONS of each block, concatenated --
5 x 2 = 10. Not a padded two-vector. Our driving spec put the action in two
slots and zeroed the rest, which approximated the right total displacement
well enough to score 0.448 and 84%, but it is the wrong shape.

For a planner that holds one action for five environment steps, the correct
input is that action REPEATED five times, filling all ten slots.

THREE ENCODINGS, MEASURED ON REAL DATA
--------------------------------------
    slots+scale : the two dims in slots [0,1] x 5, zeros elsewhere
                  -- what the 84% run used
    repeat      : the block's mean action tiled five times
                  -- what a constant-action planner should send
    dense (true): the five raw actions concatenated, exactly as their
                  training pipeline stores them
                  -- the ceiling; no planner can do better than this, because
                     this IS the input the model was trained on

Score is one-step prediction error over the frozen-world baseline; below 1
means the model is genuinely predicting. If "repeat" beats "slots+scale", the
rerun is worth it. If "dense (true)" is far below both, that gap is the price
a constant-action planner pays for not being able to vary the action within a
block -- worth knowing and worth a sentence in the paper.

Usage:
    python3 verify_repeat_encoding.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="authors_driving_spec.json")
    ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    import torch
    import h5py
    import hdf5plugin  # noqa: F401
    from authors_adapter import load_authors_model

    model = load_authors_model(args.spec, verbose=True)
    inner = model.inner
    HS = model.history_size
    A = model.action_width
    fs = args.frameskip
    scale = model.action_scale if model.action_scale != 1.0 else 5.0

    rng = np.random.default_rng(args.seed)
    need = fs * HS + fs
    with h5py.File(str(Path(args.h5).expanduser()), "r") as f:
        off = np.asarray(f["ep_offset"][:])
        ln = np.asarray(f["ep_len"][:])
        ok = np.where(ln > need + 2)[0]
        eps = rng.choice(ok, size=min(args.samples, len(ok)), replace=False)
        frames, dense, means = [], [], []
        for ep in eps:
            s = int(off[ep]) + int(rng.integers(0, ln[ep] - need - 1))
            idx = [s + fs * i for i in range(HS + 1)]
            frames.append(np.stack([f["pixels"][i] for i in idx]))
            blocks = [np.asarray(f["action"][s + fs * i: s + fs * (i + 1)],
                                 dtype=np.float32) for i in range(HS)]
            dense.append(np.stack([b.reshape(-1) for b in blocks]))
            means.append(np.stack([b.mean(0) for b in blocks]))
    frames = np.stack(frames).astype(np.float32) / 255.0
    dense = np.stack(dense)      # (N, HS, frameskip*action_dim) = the true input
    means = np.stack(means)      # (N, HS, 2)
    N = len(frames)

    px = torch.from_numpy(frames).permute(0, 1, 4, 2, 3)
    px = (px - model._mean) / model._std
    with torch.no_grad():
        emb_all = torch.cat([inner.encode({"pixels": px[i:i + 8]})["emb"]
                             for i in range(0, N, 8)], 0)
    ctx, tgt = emb_all[:, :HS], emb_all[:, HS]
    static = torch.norm(ctx[:, -1] - tgt, dim=-1).mean()

    a_mean = torch.from_numpy(means)
    slots = a_mean.new_zeros(N, HS, A)
    slots[..., model.slots] = a_mean * scale
    repeat = a_mean.repeat(1, 1, A // a_mean.shape[-1])[..., :A]
    variants = {"slots + scale (the 84% run)": slots,
                "repeat (constant-action planner)": repeat,
                "dense, true actions (ceiling)": torch.from_numpy(dense)}

    print(f"\n{N} windows; frozen-world baseline {float(static):.3f}")
    print(f"  {'encoding':<36}{'err':>9}{'ratio':>9}")
    print("  " + "-" * 54)
    out = {}
    for name, v in variants.items():
        if v.shape[-1] != A:
            print(f"  {name:<36}  shape {tuple(v.shape)} != action width {A}")
            continue
        with torch.no_grad():
            pred = inner.predict(ctx, inner.action_encoder(v.float()))[:, -1]
        err = torch.norm(pred - tgt, dim=-1).mean()
        out[name] = float(err / static)
        print(f"  {name:<36}{float(err):>9.3f}{out[name]:>9.3f}")
    print("  " + "-" * 54)

    if len(out) == 3:
        s_, r_, d_ = out.values()
        print("\nREAD")
        if r_ < s_ - 0.01:
            print(f"  repeat beats slots+scale ({r_:.3f} vs {s_:.3f}) -- rerun "
                  f"the calibration with action_encoding 'repeat'.")
        elif r_ > s_ + 0.01:
            print(f"  slots+scale is better ({s_:.3f} vs {r_:.3f}) -- surprising; "
                  f"keep the 84% run and tell me.")
        else:
            print(f"  the two are equivalent ({s_:.3f} vs {r_:.3f}); the rerun "
                  f"would not change the number. Skip it and note the")
            print("  encoding question as settled either way.")
        print(f"  the true dense input scores {d_:.3f}. The distance from")
        print(f"  {min(s_, r_):.3f} to {d_:.3f} is what a planner gives up by")
        print("  holding one action constant across the whole block -- it")
        print("  cannot express within-block variation that the model can use.")


if __name__ == "__main__":
    main()
