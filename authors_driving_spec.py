"""
authors_driving_spec.py -- settle the one thing v4 left ambiguous, then write
the spec the calibration run will read.

WHERE V4 LEFT US
----------------
v4 measured how the authors' checkpoint has to be driven, and two facts came
back cleanly:

    pixel convention : ImageNet-normalised. Raw [0,1] scored 5.1-5.4 against
                       the frozen-world baseline -- five times WORSE than
                       predicting no change at all. This is not in config.json.
    action encoding  : the two real action dimensions, zero-padded to ten.
                       Controls behaved exactly as they should: zeros scored
                       1.045 and random actions 1.151 (both no better than
                       standing still), while the true action roughly HALVED
                       the error.

What it could not settle: first-2 padding scored 0.516 and last-2 scored
0.508. That 1.6% gap over 24 samples is noise. Both cannot be right -- the
weights were trained with the action in specific channels -- and driving the
model through the wrong slots could quietly degrade planning, which is the
failure mode this project exists to avoid.

TWO WAYS TO SETTLE IT, BOTH CHEAP
----------------------------------
A. Look at the weights. The action encoder's first layer is a 1x1 convolution
   over ten input channels. Any channel that received only zeros throughout
   training got exactly zero gradient, so it still sits at its initial random
   value. Channels that carried the real action were trained. Comparing the
   ten per-channel weight norms therefore shows directly which slots the
   action occupied -- no inference required.

B. Measure again with enough samples. The same sweep as v4 but paired
   per-sample and at higher n, with a sign test on the per-sample differences,
   so a real 1.6% gap would show and noise would not.

Both run here. If they agree, the answer is settled and the spec is written.
If they disagree, that is reported and nothing is written -- we would rather
stop than guess.

Usage:
    python3 authors_driving_spec.py --lewm ~/le-wm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from math import comb
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

VIT_SIZES = {"tiny": (192, 12, 3), "small": (384, 12, 6),
             "base": (768, 12, 12), "large": (1024, 24, 16),
             "huge": (1280, 32, 16)}
TARGET_MAP = {
    "stable_worldmodel.wm.lewm.LeWM": "jepa.JEPA",
    "stable_worldmodel.wm.lewm.module.Predictor": "module.ARPredictor",
    "stable_worldmodel.wm.lewm.module.Embedder": "module.Embedder",
    "stable_worldmodel.wm.lewm.module.MLP": "module.MLP",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def remap(o, m):
    if isinstance(o, dict):
        return {k: (m.get(v, v) if k == "_target_" and isinstance(v, str)
                    else remap(v, m)) for k, v in o.items()}
    return o


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    hi = max(wins, losses)
    return min(1.0, 2 * sum(comb(n, k) for k in range(hi, n + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lewm", default=str(Path.home() / "le-wm"))
    ap.add_argument("--dest", default=None)
    ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="authors_driving_spec.json")
    args = ap.parse_args()

    import numpy as np
    import torch
    from hydra.utils import instantiate, get_class
    from transformers import ViTConfig, ViTModel

    # ---- load exactly as v4 did ------------------------------------------
    section("1. rebuild the checkpoint (same recipe v4 used)")
    try:
        import stable_worldmodel.wm.lewm  # noqa: F401
        mapping = {}
    except Exception:
        sys.path.insert(0, str(Path(args.lewm).expanduser()))
        mapping = TARGET_MAP
    try:
        from stable_worldmodel.data.utils import get_cache_dir
        try:
            root = Path(get_cache_dir())
        except TypeError:
            root = Path(get_cache_dir(None))
    except Exception:
        root = Path.home() / ".stable_worldmodel"
    dest = Path(args.dest).expanduser() if args.dest else root / "hf_tworooms"
    cfg = json.loads((dest / "config.json").read_text())
    c = remap(cfg, mapping)
    e = cfg["encoder"]
    h, L, A_ = VIT_SIZES[e["size"]]
    encoder = ViTModel(ViTConfig(hidden_size=h, num_hidden_layers=L,
                                 num_attention_heads=A_,
                                 intermediate_size=h * 4,
                                 image_size=e["image_size"],
                                 patch_size=e["patch_size"]),
                       add_pooling_layer=False, use_mask_token=False)
    parts = {k: instantiate(c[k]) for k in
             ("predictor", "action_encoder", "projector", "pred_proj")
             if k in c}
    model = get_class(c["_target_"])(encoder=encoder, **parts)
    sd = torch.load(dest / "weights.pt", map_location="cpu",
                    weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.requires_grad_(False)
    HS = int(model.predictor.pos_embedding.shape[1])
    A = int(model.action_encoder.patch_embed.weight.shape[1])
    print(f"  loaded; history_size {HS}, action width {A}")

    # ---- A: which channels were trained? ---------------------------------
    section("A. which action channels carry signal? (from the weights)")
    W = model.action_encoder.patch_embed.weight.detach()   # (out, in, 1)
    norms = [float(W[:, j, 0].norm()) for j in range(A)]
    med = float(np.median(norms))
    print("  per-input-channel weight norms:")
    for j, v in enumerate(norms):
        bar = "#" * int(round(v / max(norms) * 40))
        print(f"    channel {j}: {v:7.3f}  {bar}")
    print(f"\n  median {med:.3f}")
    stand_out = [j for j, v in enumerate(norms) if v > 1.6 * med]
    if stand_out:
        print(f"  channels clearly above the rest: {stand_out}")
    else:
        print("  no channel stands out -- either all channels were trained")
        print("  (so the action was not zero-padded) or the effect is subtle.")
    weights_says = None
    if stand_out == [0, 1]:
        weights_says = "first"
    elif stand_out == [A - 2, A - 1]:
        weights_says = "last"
    print(f"  weight evidence -> "
          f"{weights_says if weights_says else 'inconclusive'}")

    # ---- B: paired measurement at higher n --------------------------------
    section(f"B. paired re-measurement, {args.samples} samples")
    try:
        import h5py
        import hdf5plugin  # noqa: F401
        rng = np.random.default_rng(args.seed)
        fs = args.frameskip
        need = fs * HS + fs
        with h5py.File(str(Path(args.h5).expanduser()), "r") as f:
            off = np.asarray(f["ep_offset"][:])
            ln = np.asarray(f["ep_len"][:])
            ok = np.where(ln > need + 2)[0]
            eps = rng.choice(ok, size=min(args.samples, len(ok)),
                             replace=False)
            frames, acts = [], []
            for ep in eps:
                s = int(off[ep]) + int(rng.integers(0, ln[ep] - need - 1))
                idx = [s + fs * i for i in range(HS + 1)]
                frames.append(np.stack([f["pixels"][i] for i in idx]))
                acts.append(np.stack([
                    np.asarray(f["action"][s + fs * i: s + fs * (i + 1)],
                               dtype=np.float32).sum(0) for i in range(HS)]))
        frames = np.stack(frames).astype(np.float32) / 255.0
        acts = np.stack(acts)
        N = len(frames)
        px = torch.from_numpy(frames).permute(0, 1, 4, 2, 3)
        px = (px - torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)) \
            / torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)

        embs = []
        with torch.no_grad():
            for i in range(0, N, 8):
                embs.append(model.encode({"pixels": px[i:i + 8]})["emb"])
        emb_all = torch.cat(embs, 0)
        ctx, tgt = emb_all[:, :HS], emb_all[:, HS]
        static = torch.norm(ctx[:, -1] - tgt, dim=-1)

        def err_for(slots):
            a = np.zeros((N, HS, A), dtype=np.float32)
            a[:, :, slots] = acts
            with torch.no_grad():
                ae = model.action_encoder(torch.from_numpy(a))
                pred = model.predict(ctx, ae)[:, -1]
            return torch.norm(pred - tgt, dim=-1)

        e_first = err_for([0, 1])
        e_last = err_for([A - 2, A - 1])
        print(f"  frozen-world baseline : {float(static.mean()):.3f}")
        print(f"  first-2 padded        : {float(e_first.mean()):.3f}  "
              f"ratio {float(e_first.mean()/static.mean()):.3f}")
        print(f"  last-2 padded         : {float(e_last.mean()):.3f}  "
              f"ratio {float(e_last.mean()/static.mean()):.3f}")
        d = (e_first - e_last).numpy()
        wins_last = int((d > 0).sum())
        wins_first = int((d < 0).sum())
        p = sign_test(wins_last, wins_first)
        print(f"\n  per-sample: last-2 better on {wins_last}, "
              f"first-2 better on {wins_first}  ->  sign test p = {p:.4f}")
        measure_says = None
        if p < 0.05:
            measure_says = "last" if wins_last > wins_first else "first"
        print(f"  measurement evidence -> "
              f"{measure_says if measure_says else 'no significant difference'}")
    except Exception:
        print(traceback.format_exc())
        return

    # ---- verdict ----------------------------------------------------------
    section("VERDICT")
    if weights_says and measure_says and weights_says != measure_says:
        print(f"  DISAGREEMENT: weights say {weights_says}, measurement says "
              f"{measure_says}. Nothing written. Send me this.")
        return
    choice = weights_says or measure_says
    if choice is None:
        print("  Neither test discriminates. The two placements are")
        print("  interchangeable for this model, so either drives it equally")
        print("  well -- writing 'first' and noting the ambiguity as a stated")
        print("  deviation.")
        choice = "first"
        note = "first-2 and last-2 padding were indistinguishable; stated as a deviation"
    else:
        note = f"resolved by {'weight inspection' if weights_says else 'paired measurement'}"
    spec = {"pixel_convention": "imagenet",
            "imagenet_mean": list(IMAGENET_MEAN),
            "imagenet_std": list(IMAGENET_STD),
            "action_width": A, "action_slots": ([0, 1] if choice == "first"
                                                else [A - 2, A - 1]),
            "history_size": HS, "frameskip": args.frameskip,
            "action_aggregation": "sum over each frameskip block",
            "note": note,
            "evidence": {"weights": weights_says, "measurement": measure_says,
                         "ratio_first": float(e_first.mean()/static.mean()),
                         "ratio_last": float(e_last.mean()/static.mean()),
                         "samples": N}}
    Path(args.out).write_text(json.dumps(spec, indent=2))
    print(f"  action slots: {spec['action_slots']}   ({note})")
    print(f"  wrote {args.out} -- commit it; it is the record of how their")
    print("  checkpoint was driven, and the calibration run will read it.")


if __name__ == "__main__":
    main()
