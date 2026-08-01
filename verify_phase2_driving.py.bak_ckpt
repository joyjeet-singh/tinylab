"""
verify_phase2_driving.py -- measure how the new checkpoint must be driven,
before a single planning number is taken from it.

WHY THIS EXISTS
---------------
The authors'-checkpoint calibration produced 46% on its first attempt. The run
completed, the guard passed, the control replicated exactly -- and the number
was an artifact of feeding the model actions at the wrong scale. It was caught
by the SHAPE of the failures, not by any check, and only then measured.

A phase2 checkpoint has the same exposure: it expects ImageNet pixels, ten-wide
dense actions, and z-scoring, while the planner emits two-wide raw actions
against [0,1] pixels. `dense_action_adapter.py` proposes an encoding. This
script tests it rather than trusting it.

WHAT IS MEASURED
----------------
One-step prediction error over the frozen-world baseline (below 1 means the
model is genuinely predicting), on real clips, for:

    repeat + z-score      what the adapter does -- the planner's action held
                          for frameskip steps, normalised as in training
    repeat, no z-score    isolates whether the normalisation matters
    first-2 padded        the encoding that was wrong for the authors' model
    zeros                 no action information -- should be no better than
                          standing still
    random                should be no better than standing still
    dense true actions    the five real actions of the block, z-scored: the
                          input the model was actually trained on, and the
                          ceiling no constant-action planner can beat

It also cross-checks the adapter's action statistics against the loader's own
`_action_stats`, so the two cannot silently diverge.

READING IT
----------
  repeat + z-score clearly below 1 and close to the dense ceiling
      -> drive it that way; the planning evaluation can proceed
  zeros or random competitive with repeat
      -> the model is not using our actions; STOP, do not plan
  a large gap from repeat to the dense ceiling
      -> that gap is what a constant-action planner gives up, and belongs in
         the paper as a stated limitation rather than a surprise

Usage:
    python3 verify_phase2_driving.py --run runs/<phase2 run dir>
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
    ap.add_argument("--run", required=True)
    ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    import h5py
    import hdf5plugin  # noqa: F401
    from toy_model import ToyJEPA
    from dense_action_adapter import (action_stats, convention_from_manifest,
                                      wrap_if_needed)

    run_dir = Path(args.run)
    h5 = str(Path(args.h5).expanduser())
    cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
    m = cfg["model"]

    print("=" * 70)
    print(f"HOW MUST {run_dir.name} BE DRIVEN?")
    print("=" * 70)
    conv = convention_from_manifest(run_dir)
    print(f"  manifest loader_convention: "
          f"{ {k: v for k, v in conv.items() if not k.startswith('_')} } "
          f"({conv['_source']})")

    model = ToyJEPA(embed_dim=m["embed_dim"], action_dim=m["action_dim"],
                    history_size=m["history_size"], depth=m["depth"],
                    heads=m["heads"], dim_head=m["dim_head"],
                    mlp_dim=m["mlp_dim"], proj_hidden=m["proj_hidden"],
                    dropout=m["dropout"], enc_width=m["enc_width"],
                    encoder=m.get("encoder", "cnn"),
                    img_size=m.get("img_size", 32),
                    patch_size=m.get("patch_size", 4),
                    enc_depth=m.get("enc_depth", 12),
                    enc_heads=m.get("enc_heads", 3))
    ck = torch.load(run_dir / ("ckpt_best.pt"
                               if (run_dir / "ckpt_best.pt").exists()
                               else "ckpt.pt"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    model.requires_grad_(False)
    print(f"  model: action_dim {m['action_dim']}, history_size "
          f"{m['history_size']}, epoch field {ck.get('epoch')}")

    adapter = wrap_if_needed(model, run_dir, h5, frameskip=args.frameskip)
    if adapter is model:
        print("\n  No transforms recorded -- this checkpoint is driven "
              "directly and\n  needs no measurement. Nothing to do.")
        return
    HS, A, fs = adapter.history_size, adapter.action_width, args.frameskip

    # ---- statistics cross-check ------------------------------------------
    print("\nSTATISTICS CROSS-CHECK (adapter vs the loader's own)")
    mu, sd = action_stats(h5)
    try:
        import tworoom_data as td
        spec = td.ClipSpec(history=HS, frameskip=fs)
        idx = td.TwoRoomIndex(h5, spec)
        cl = td.TwoRoomClips(h5, idx, keys=("action",))
        with h5py.File(h5, "r") as f:
            lmu, lsd = cl._action_stats(f)
        ok = np.allclose(mu, lmu) and np.allclose(sd, lsd)
        print(f"  adapter mu {np.round(mu, 4)} sd {np.round(sd, 4)}")
        print(f"  loader  mu {np.round(lmu, 4)} sd {np.round(lsd, 4)}")
        print(f"  identical: {ok}")
        if not ok:
            print("  STOP -- the adapter would normalise differently from "
                  "training.")
            return
    except Exception as ex:
        print(f"  could not cross-check ({type(ex).__name__}: {ex}); "
              f"proceeding on the adapter's own statistics")

    # ---- real windows -----------------------------------------------------
    rng = np.random.default_rng(args.seed)
    need = fs * HS + fs
    with h5py.File(h5, "r") as f:
        off = np.asarray(f["ep_offset"][:])
        ln = np.asarray(f["ep_len"][:])
        ok_eps = np.where(ln > need + 2)[0]
        eps = rng.choice(ok_eps, size=min(args.samples, len(ok_eps)),
                         replace=False)
        frames, dense, first = [], [], []
        for ep in eps:
            s = int(off[ep]) + int(rng.integers(0, ln[ep] - need - 1))
            frames.append(np.stack([f["pixels"][s + fs * i]
                                    for i in range(HS + 1)]))
            blocks = [np.asarray(f["action"][s + fs * i: s + fs * (i + 1)],
                                 dtype=np.float32) for i in range(HS)]
            dense.append(np.stack([b.reshape(-1) for b in blocks]))
            first.append(np.stack([b[0] for b in blocks]))
    frames = np.stack(frames).astype(np.float32) / 255.0
    dense = np.stack(dense)
    first = np.stack(first)
    N = len(frames)
    if np.isnan(dense).any():
        print("\n  NaN in a sampled action block -- the NaN audit missed "
              "something. STOP.")
        return

    px = torch.from_numpy(frames).permute(0, 1, 4, 2, 3)
    with torch.no_grad():
        emb_all = torch.cat([adapter.encode({"pixels": px[i:i + 8]})["emb"]
                             for i in range(0, N, 8)], 0)
    ctx, tgt = emb_all[:, :HS], emb_all[:, HS]
    static = torch.norm(ctx[:, -1] - tgt, dim=-1).mean()

    def zscore(x):
        reps = x.shape[-1] // len(mu)
        return (x - np.tile(mu, reps)) / np.tile(sd, reps)

    rep_raw = np.tile(first, (1, 1, A // first.shape[-1]))
    padded = np.zeros((N, HS, A), dtype=np.float32)
    padded[:, :, :2] = first
    variants = {
        "repeat + z-score (the adapter)": zscore(rep_raw),
        "repeat, no z-score": rep_raw,
        "first-2 padded + z-score": zscore(padded),
        "zeros": np.zeros((N, HS, A), dtype=np.float32),
        "random": rng.normal(0, 1, (N, HS, A)).astype(np.float32),
        "dense true actions (ceiling)": zscore(dense),
    }

    print(f"\n{N} windows; frozen-world baseline {float(static):.3f}")
    if not np.isfinite(float(static)) or float(static) < 0.05:
        print("  STOP -- the frozen-world baseline has collapsed, so every")
        print("  ratio below would be meaningless. Consecutive frames are")
        print("  nearly identical in latent space; check the checkpoint.")
        return
    print(f"  {'encoding':<34}{'err':>9}{'ratio':>9}")
    print("  " + "-" * 52)
    res = {}
    for name, v in variants.items():
        with torch.no_grad():
            ae = adapter.inner.action_encoder(
                torch.as_tensor(v, dtype=torch.float32))
            pred = adapter.predict(ctx, ae)[:, -1]
        err = torch.norm(pred - tgt, dim=-1).mean()
        res[name] = float(err / static)
        print(f"  {name:<34}{float(err):>9.3f}{res[name]:>9.3f}")
    print("  " + "-" * 52)

    bad = [k for k, r in res.items() if not np.isfinite(r)]
    if bad:
        print("\nSTOP -- non-finite ratios for: " + ", ".join(bad))
        print("  A NaN here means the checkpoint or the inputs are broken. No")
        print("  verdict is possible; never read GO off a table with NaN in it.")
        return

    adp = res["repeat + z-score (the adapter)"]
    ceil = res["dense true actions (ceiling)"]
    noise = max(res["zeros"], res["random"])
    print("\nVERDICT")
    if adp >= noise - 0.02:
        print(f"  STOP. The adapter's encoding ({adp:.3f}) is no better than "
              f"no action at all ({noise:.3f}).")
        print("  The model is not reading our actions. Do not plan with it; "
              "send me this table.")
        return
    if adp > 0.9:
        print(f"  CAUTION. {adp:.3f} beats the no-action controls but barely "
              f"beats standing still.")
        print("  Planning numbers from this checkpoint will be weak for a "
              "reason that is the")
        print("  model's, not the harness's. Worth knowing before, not after.")
    else:
        print(f"  GO. repeat + z-score scores {adp:.3f}, clearly better than "
              f"the no-action controls ({noise:.3f}).")
    print(f"  Dense-true ceiling {ceil:.3f}. The gap {adp - ceil:+.3f} is what "
          f"a constant-action")
    print("  planner gives up by not varying the action within a block -- a "
          "stated limitation,")
    print("  not a surprise.")
    zs = res["repeat, no z-score"]
    print(f"  Z-scoring is worth {zs - adp:+.3f} (no-z-score {zs:.3f} vs "
          f"{adp:.3f}); if that is ~0,\n  the model is insensitive to action "
          f"scale and the normalisation row can be relaxed.")


if __name__ == "__main__":
    main()
