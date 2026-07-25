"""
authors_ckpt_fetch.py (v4, final attempt) -- load the authors' released TwoRoom
checkpoint AND measure how to drive it, in one run.

WHY V4 IS THE LAST ONE
----------------------
Agreed cap: if this run does not get us to a usable model, we stop and move to
extracting the gate scripts and putting seed-1 through the launch protocol.
So this script does everything a successful load would unlock, rather than
stopping at "it loaded" and needing another round.

THE BLOCKER V3 HIT, AND THE FIX
--------------------------------
config.json's `_target_` fields name classes in `stable_worldmodel.wm.lewm`,
which does not exist in your 0.0.6. The identical classes live in the le-wm
repo under different names, so v4 rewrites the targets before handing the
config to hydra:

    stable_worldmodel.wm.lewm.LeWM              -> jepa.JEPA
    stable_worldmodel.wm.lewm.module.Predictor  -> module.ARPredictor
    stable_worldmodel.wm.lewm.module.Embedder   -> module.Embedder
    stable_worldmodel.wm.lewm.module.MLP        -> module.MLP

Proven equivalent in my sandbox, where both versions are available: building
from the package classes and from the le-wm classes gives 18.034M parameters
either way, with IDENTICAL state-dict keys in identical order and identical
shapes. So a strict load either way is the same load.

THE MEASUREMENT (section 7)
---------------------------
Their action encoder is 10 wide; TwoRoom's actions are 2. And their training
pixel convention is not documented in config.json. Both have to be settled
before their weights can be scored, and guessing either would produce a
meaningless number -- the exact failure that cost three runs.

So section 7 measures. On real data it builds a 3-frame context spaced by the
frameskip, predicts the next latent, and compares against the encoded true
next frame -- for every combination of:

    pixel convention : raw [0,1]  |  ImageNet-normalised
    action encoding  : zeros | first-2 padded | last-2 padded | tiled | random

The score is prediction error divided by the frozen-world baseline (predicting
no change), the same ratio used in the rollout-horizon evaluation. Below 1.0
means the model is genuinely predicting. The winning combination -- if one
clearly wins -- is how their checkpoint must be driven. If nothing beats 1.0,
their model cannot be driven correctly from what we know, and we stop.

Frames are encoded once per pixel convention and reused across action
candidates, so the sweep is cheap.

Usage:
    python3 authors_ckpt_fetch.py --lewm ~/le-wm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

REPO = "quentinll/lewm-tworooms"
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


def remap(o, mapping):
    if isinstance(o, dict):
        return {k: (mapping.get(v, v)
                    if k == "_target_" and isinstance(v, str)
                    else remap(v, mapping))
                for k, v in o.items()}
    if isinstance(o, list):
        return [remap(x, mapping) for x in o]
    return o


def build_vit(size, patch_size, image_size):
    from transformers import ViTConfig, ViTModel
    h, layers, heads = VIT_SIZES[size]
    return ViTModel(ViTConfig(hidden_size=h, num_hidden_layers=layers,
                              num_attention_heads=heads,
                              intermediate_size=h * 4,
                              image_size=image_size, patch_size=patch_size),
                    add_pooling_layer=False, use_mask_token=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lewm", default=str(Path.home() / "le-wm"))
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-measure", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch

    # ---- 0 ---------------------------------------------------------------
    section("0. environment health (gates everything)")
    print(f"  torch {torch.__version__}   numpy {np.__version__}")
    try:
        _ = torch.from_numpy(np.zeros(4, dtype=np.float32))
        print("  torch <-> numpy bridge: WORKING")
    except Exception as ex:
        print(f"  BROKEN ({type(ex).__name__}). Fix: pip install \"numpy<2\", "
              f"then re-verify R1's three zeros.")
        return

    # ---- 1 ---------------------------------------------------------------
    section("1. prerequisites and which classes we will use")
    lewm = Path(args.lewm).expanduser()
    for mod, hint in (("transformers", "pip install transformers"),
                      ("hydra", "pip install hydra-core")):
        try:
            __import__(mod)
            print(f"  {mod:<14} present")
        except Exception:
            print(f"  {mod:<14} MISSING -> {hint}")
            return
    try:
        import stable_worldmodel.wm.lewm  # noqa: F401
        mapping = {}
        print("  classes        : from stable-worldmodel (wm.lewm present)")
    except Exception:
        if not (lewm / "jepa.py").exists():
            print(f"  STOP: wm.lewm is absent from your stable-worldmodel and")
            print(f"  {lewm} has no jepa.py. Clone it:")
            print(f"    git clone https://github.com/lucas-maes/le-wm {lewm}")
            return
        sys.path.insert(0, str(lewm))
        mapping = TARGET_MAP
        print(f"  classes        : from {lewm} (wm.lewm absent from 0.0.6)")
        print("                   proven identical: same keys, shapes, "
              "18.034M params")

    # ---- 2 ---------------------------------------------------------------
    section("2. files")
    try:
        from stable_worldmodel.data.utils import get_cache_dir
        try:
            root = Path(get_cache_dir())
        except TypeError:
            root = Path(get_cache_dir(None))
    except Exception:
        root = Path.home() / ".stable_worldmodel"
    dest = Path(args.dest).expanduser() if args.dest else root / "hf_tworooms"
    cfg_path, w_path = dest / "config.json", dest / "weights.pt"
    if not (cfg_path.exists() and w_path.exists()):
        try:
            from huggingface_hub import snapshot_download
            dest.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=args.repo, local_dir=str(dest))
        except Exception:
            print(f"  download them from https://huggingface.co/{args.repo} "
                  f"into {dest}")
            return
    print(f"  {dest}  (weights {w_path.stat().st_size/1e6:.1f} MB)")
    cfg = json.loads(cfg_path.read_text())

    # ---- 3 ---------------------------------------------------------------
    section("3. build and load")
    try:
        from hydra.utils import instantiate, get_class
        c = remap(cfg, mapping)
        e = cfg["encoder"]
        encoder = build_vit(e["size"], e["patch_size"], e["image_size"])
        parts = {k: instantiate(c[k]) for k in
                 ("predictor", "action_encoder", "projector", "pred_proj")
                 if k in c}
        model = get_class(c["_target_"])(encoder=encoder, **parts)
        sd = torch.load(w_path, map_location="cpu", weights_only=False)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        model.load_state_dict(sd, strict=True)
        model.eval()
        model.requires_grad_(False)
        HS = int(model.predictor.pos_embedding.shape[1])
        A = int(model.action_encoder.patch_embed.weight.shape[1])
        model.history_size = HS
        print(f"  built + STRICT LOAD OK: "
              f"{sum(q.numel() for q in model.parameters())/1e6:.2f} M params")
        print(f"  history_size {HS}   action width {A}")
        gates = [v for k, v in model.state_dict().items()
                 if "adaLN_modulation" in k]
        live = sum(float(g.abs().sum()) for g in gates)
        print(f"  conditioning path (AdaLN-zero at init): "
              f"{'LIVE, trained' if live > 0 else 'ALL ZERO -- untrained'} "
              f"(|w| sum {live:.3f})")
        if live == 0:
            print("  With a zero conditioning path the model ignores actions")
            print("  entirely. Stop -- these are not usable trained weights.")
            return
    except Exception:
        print(traceback.format_exc())
        print("\n  Send me this and we stop here as agreed.")
        return

    if args.skip_measure:
        return

    # ---- 4: the measurement ----------------------------------------------
    section("4. how must this model be driven? (measured on real data)")
    try:
        import h5py
        import hdf5plugin  # noqa: F401
        rng = np.random.default_rng(args.seed)
        fs = args.frameskip
        need = fs * HS + fs          # context span plus one target step

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
                               dtype=np.float32).sum(0)
                    for i in range(HS)]))
        frames = np.stack(frames).astype(np.float32) / 255.0   # (N,HS+1,H,W,3)
        acts = np.stack(acts)                                  # (N,HS,2)
        N = len(frames)
        print(f"  {N} samples, context {HS} frames spaced {fs} raw steps, "
              f"actions summed over each block")

        px_raw = torch.from_numpy(frames).permute(0, 1, 4, 2, 3)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
        conventions = {"raw [0,1]": px_raw,
                       "imagenet-norm": (px_raw - mean) / std}

        def action_variants(a):           # a: (N, HS, 2) -> (N, HS, A)
            z = np.zeros((N, HS, A), dtype=np.float32)
            first = z.copy(); first[:, :, :2] = a
            last = z.copy(); last[:, :, -2:] = a
            tiled = np.tile(a, (1, 1, (A + 1) // 2))[:, :, :A].astype(np.float32)
            rand = rng.normal(0, 1, (N, HS, A)).astype(np.float32)
            return {"zeros (no action)": z, "first-2 padded": first,
                    "last-2 padded": last, "tiled": tiled,
                    "random (control)": rand}

        print(f"\n  {'pixels':<15}{'action encoding':<20}"
              f"{'err':>9}{'baseline':>10}{'ratio':>8}")
        print("  " + "-" * 62)
        best = (None, 9e9)
        all_ratios, static_ref = [], None
        for pname, px in conventions.items():
            with torch.no_grad():
                emb_all = model.encode({"pixels": px})["emb"]   # (N,HS+1,D)
            ctx, tgt = emb_all[:, :HS], emb_all[:, HS]
            static = torch.norm(ctx[:, -1] - tgt, dim=-1).mean()
            static_ref = float(static) if static_ref is None else static_ref
            for aname, av in action_variants(acts).items():
                with torch.no_grad():
                    ae = model.action_encoder(torch.from_numpy(av))
                    pred = model.predict(ctx, ae)[:, -1]
                err = torch.norm(pred - tgt, dim=-1).mean()
                ratio = float(err / static)
                print(f"  {pname:<15}{aname:<20}{float(err):>9.3f}"
                      f"{float(static):>10.3f}{ratio:>8.3f}")
                all_ratios.append(ratio)
                if ratio < best[1]:
                    best = ((pname, aname), ratio)
        print("  " + "-" * 62)
        if static_ref is not None and static_ref < 0.05:
            print(f"\n  WARNING: the frozen-world baseline is {static_ref:.4f}"
                  f" -- consecutive frames are nearly identical in latent")
            print("  space, so every ratio above is meaningless. Send me this.")
            return
        if len(set(round(r, 4) for r in all_ratios)) == 1:
            print("\n  WARNING: every action encoding scored identically, so")
            print("  actions are not reaching the prediction at all. Send me")
            print("  this -- it means the conditioning is not wired as we")
            print("  think, and we stop as agreed.")
            return
        print(f"\n  best: {best[0][0]} + {best[0][1]}  (ratio {best[1]:.3f})")
        print("\n  HOW TO READ")
        print("   ratio below ~0.9 for one clear winner -> that is how their")
        print("     checkpoint must be driven; the adapter is then trivial and")
        print("     the calibration run can go ahead.")
        print("   everything near or above 1.0, including 'zeros' -> we cannot")
        print("     drive their model correctly from what is documented. STOP")
        print("     here as agreed and move to the gate scripts and seed-1.")
        print("   'zeros' winning -> the model ignores our action encoding")
        print("     entirely, which is the same STOP.")
    except Exception:
        print(traceback.format_exc())
        print("\n  Send me this and we stop here as agreed.")


if __name__ == "__main__":
    main()
