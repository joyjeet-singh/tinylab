"""
action_scale_check.py -- did we feed the authors' model actions at the wrong
scale, and are we feeding OUR model at the right one?

WHY THIS EXISTS (a mistake of mine, caught by the result's shape)
------------------------------------------------------------------
The calibration run gave the authors' checkpoint 46.0% against our 72.0%. But
the pattern is wrong for "their model is worse":

    their successes are FAST   -- median 5 steps, ours 18
    their misses are RUNAWAYS  -- 22 of 27 ended farther than they started,
                                  averaging 1.59x the start-goal distance,
                                  several at 130-164 units

That is the exact signature of a model whose imagined dynamics UNDERSTATE how
far an action moves the agent. CEM compensates by choosing near-maximal
actions; in the real environment those move much further than imagined. Short
goals get hit quickly by accident, everything else overshoots badly.

And there is a concrete reason to expect it. When I measured how to drive
their model, I summed the five raw actions in each frameskip block. But the
evaluator feeds the planner's single action, which is then executed five
times. For the same displacement those differ by a factor of five:

    block displacement from data     = 5 * (sum of the 5 raw actions)
    block displacement from action a = 5 * 5a
    => the evaluator's action corresponds to the MEAN of the block, not the SUM

So the driving spec was validated at one scale and the evaluation runs at
another, five times smaller. If their model was trained on sums, the 46% is an
artifact of my error and not a property of their checkpoint.

WHAT THIS MEASURES
------------------
For each model, one-step prediction error divided by the frozen-world baseline
(<1 means genuinely predicting), with the block action set to k x mean, swept
over k. k=1 is exactly what the evaluator feeds; k=5 is exactly what the
driving spec was validated at. The k that minimises the ratio is the scale
that model was trained for.

OUR model is included deliberately. If ours also prefers k>1, then our own 72%
is under-driven too and the number could be higher -- which would matter far
more than the calibration experiment.

Usage:
    python3 action_scale_check.py --run $RUN2
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
    ap.add_argument("--run", required=True, help="our run dir (for ckpt_best)")
    ap.add_argument("--authors-spec", default="authors_driving_spec.json")
    ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
    ap.add_argument("--samples", type=int, default=120)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    import torch
    import h5py
    import hdf5plugin  # noqa: F401

    models = {}

    # ---- our checkpoint ---------------------------------------------------
    try:
        from toy_model import ToyJEPA
        run_dir = Path(args.run)
        cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
        m = cfg["model"]
        ours = ToyJEPA(embed_dim=m["embed_dim"], action_dim=m["action_dim"],
                       history_size=m["history_size"], depth=m["depth"],
                       heads=m["heads"], dim_head=m["dim_head"],
                       mlp_dim=m["mlp_dim"], proj_hidden=m["proj_hidden"],
                       dropout=m["dropout"], enc_width=m["enc_width"],
                       encoder=m.get("encoder", "cnn"),
                       img_size=m.get("img_size", 32),
                       patch_size=m.get("patch_size", 4),
                       enc_depth=m.get("enc_depth", 12),
                       enc_heads=m.get("enc_heads", 3))
        ck = torch.load(run_dir / "ckpt_best.pt", map_location="cpu",
                        weights_only=False)
        ours.load_state_dict(ck["model"])
        ours.eval()
        ours.requires_grad_(False)
        models["ours (ckpt_best)"] = ours
    except Exception as ex:
        print(f"could not load our checkpoint: {type(ex).__name__}: {ex}")

    # ---- their checkpoint -------------------------------------------------
    try:
        from authors_adapter import load_authors_model
        models["authors (released)"] = load_authors_model(args.authors_spec,
                                                          verbose=False)
    except Exception as ex:
        print(f"could not load the authors' checkpoint: "
              f"{type(ex).__name__}: {ex}")

    if not models:
        raise SystemExit("no models loaded")

    # ---- shared real-data windows -----------------------------------------
    rng = np.random.default_rng(args.seed)
    fs = args.frameskip
    maxHS = max(int(getattr(mo, "history_size", 1)) for mo in models.values())
    need = fs * maxHS + fs
    with h5py.File(str(Path(args.h5).expanduser()), "r") as f:
        off = np.asarray(f["ep_offset"][:])
        ln = np.asarray(f["ep_len"][:])
        ok = np.where(ln > need + 2)[0]
        eps = rng.choice(ok, size=min(args.samples, len(ok)), replace=False)
        frames, means = [], []
        for ep in eps:
            s = int(off[ep]) + int(rng.integers(0, ln[ep] - need - 1))
            idx = [s + fs * i for i in range(maxHS + 1)]
            frames.append(np.stack([f["pixels"][i] for i in idx]))
            means.append(np.stack([
                np.asarray(f["action"][s + fs * i: s + fs * (i + 1)],
                           dtype=np.float32).mean(0) for i in range(maxHS)]))
    frames = np.stack(frames).astype(np.float32) / 255.0
    means = np.stack(means)          # (N, maxHS, 2)  block MEAN actions
    N = len(frames)
    px_all = torch.from_numpy(frames).permute(0, 1, 4, 2, 3)
    print(f"\n{N} windows, context spaced {fs} raw steps; block action = "
          f"k x mean of the block's raw actions")
    print("k=1 is what the evaluator feeds; k=5 is the block SUM, which is "
          "what\nthe driving spec was validated at.\n")

    ks = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
    print(f"  {'model':<22}{'baseline':>10}" +
          "".join(f"{'k=' + str(k):>9}" for k in ks))
    print("  " + "-" * (32 + 9 * len(ks)))
    verdict = {}
    for name, mo in models.items():
        HS = int(getattr(mo, "history_size", 1))
        px = px_all[:, maxHS - HS:maxHS + 1]
        with torch.no_grad():
            chunks = [mo.encode({"pixels": px[i:i + 8]})["emb"]
                      for i in range(0, N, 8)]
            emb_all = torch.cat(chunks, 0)
        ctx, tgt = emb_all[:, :HS], emb_all[:, HS]
        static = torch.norm(ctx[:, -1] - tgt, dim=-1).mean()
        a_mean = torch.from_numpy(means[:, maxHS - HS:maxHS])
        row, best = [], (None, 9e9)
        for k in ks:
            with torch.no_grad():
                ae = mo.action_encoder(a_mean * k)
                pred = mo.predict(ctx, ae)[:, -1]
            r = float(torch.norm(pred - tgt, dim=-1).mean() / static)
            row.append(r)
            if r < best[1]:
                best = (k, r)
        verdict[name] = best
        print(f"  {name:<22}{float(static):>10.3f}" +
              "".join(f"{r:>9.3f}" for r in row))
        if float(static) < 0.05:
            print(f"     WARNING: frozen-world baseline for {name} is "
                  f"{float(static):.4f} -- consecutive frames are nearly")
            print("     identical in latent space, so this row means nothing.")
    print("  " + "-" * (32 + 9 * len(ks)))

    print("\nBEST SCALE PER MODEL")
    for name, (k, r) in verdict.items():
        flag = "" if k == 1.0 else "   <- NOT what the evaluator feeds"
        print(f"  {name:<22} k={k:<5g} ratio {r:.3f}{flag}")

    print("\nHOW TO READ")
    print("  If the authors' model bottoms out near k=5 while ours bottoms")
    print("  out near k=1, then the evaluator feeds ours correctly and theirs")
    print("  five times too small -- the 46% is my error, not their")
    print("  checkpoint, and the calibration run must be repeated with the")
    print("  action scaled.")
    print("  If BOTH prefer k>1, our own 72% is under-driven too and that is")
    print("  the more important finding of the two.")
    print("  If both bottom out at k=1, the scale is fine and the 46% stands")
    print("  as measured -- and we look elsewhere for why their checkpoint")
    print("  overshoots.")


if __name__ == "__main__":
    main()
