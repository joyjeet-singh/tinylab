"""
bridge_step0_check.py -- v2. Diagnostics for the Run 2 contradiction:
train-time step0 error ~5.67 on real validation clips vs ~56.55 on toy-rendered
planner states.

WHAT CHANGED FROM v1 (and why)
------------------------------
v1's Check B paired real frames with toy renders "at the same position" and
crashed when zero recorded positions fell inside the toy arena. That crash was
a v1 bug (the empty case wasn't handled), but it also exposed a real finding:
the real file's pos_agent coordinates span roughly [21, 199] on both axes --
NOT the toy world's [0, 40] units. The two worlds do not share a coordinate
convention, so "render the toy at the same position" was the wrong design.
v2 therefore:

  B  is now convention-free: it asks how far toy-rendered frames land from the
     real-data latent cloud (nearest-neighbor distance to real latents), with
     the real cloud's own thickness as the yardstick. No position mapping
     needed.
  M  (new, cheap) measures the real data's per-step motion directly from
     pos_agent and episode boundaries, in the file's own units, and compares
     the RELATIVE speed (fraction of the arena covered per step) with the toy
     world's. If the file's [21,199] span holds, the toy dot covers ~5-6x more
     of its arena per step than the real dot does -- a dynamics-scale mismatch
     on top of the rendering-style mismatch.
  C  unchanged: the training script's own step0_latent_error() -- imported,
     not reimplemented -- on toy-rendered clips cut exactly like the training
     loader, episodes driven by the dataset-style heuristic policy.
  C2 (new) is C with the toy dot slowed to the REAL relative speed (actions
     unchanged, world step scaled), separating "the toy LOOKS different"
     from "the toy MOVES differently":
        C2 ~ Check A            -> rendering style is fine; the motion-scale
                                    mismatch is what breaks the toy-side error.
        C2 ~ C ~ planner-side   -> rendering style alone reproduces it.
  A  unchanged (it already ran and reproduced the train log on your machine);
     pass --skip-a to go straight to the new checks.

Each check runs independently: a failure prints its error and the rest still run.

Run from the tinylab folder (venv active):
    python3 bridge_step0_check.py --run runs/<run_dir> --skip-a
    python3 bridge_step0_check.py --run runs/<run_dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from make_toy_tworoom import ACTION_CLIP, ACTION_SCALE, ToyTwoRoom, heuristic_action
from tworoom_data import ClipSpec, TwoRoomClips, TwoRoomIndex
from toy_model import ToyJEPA
from train_toy_lewm import data_fingerprint, split_indices, step0_latent_error


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _build_model(m: dict) -> ToyJEPA:
    return ToyJEPA(embed_dim=m["embed_dim"], action_dim=m["action_dim"],
                   history_size=m["history_size"], depth=m["depth"],
                   heads=m["heads"], dim_head=m["dim_head"], mlp_dim=m["mlp_dim"],
                   proj_hidden=m["proj_hidden"], dropout=m["dropout"],
                   enc_width=m["enc_width"],
                   encoder=m.get("encoder", "cnn"),
                   img_size=m.get("img_size", 32),
                   patch_size=m.get("patch_size", 4),
                   enc_depth=m.get("enc_depth", 12),
                   enc_heads=m.get("enc_heads", 3))


def _load_weights(model: ToyJEPA, ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return ck


@torch.no_grad()
def _encode_frames(model, frames_uint8, batch=32):
    """iterable of (H,W,3) uint8 frames -> (N,D) latents, batched for CPU."""
    out = []
    n = len(frames_uint8)
    for s in range(0, n, batch):
        chunk = np.asarray(frames_uint8[s:s + batch], dtype=np.float32) / 255.0
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).unsqueeze(1)   # (B,1,3,H,W)
        out.append(model.encode({"pixels": x})["emb"][:, 0].cpu().numpy())
    if not out:
        return np.zeros((0, model.embed_dim), dtype=np.float32)
    return np.concatenate(out, 0)


def _nn_dist(a: np.ndarray, b: np.ndarray, exclude_self=False) -> np.ndarray:
    """for each row of a, distance to its nearest row of b."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


# ---------------------------------------------------------------------------
# CHECK C / C2 clip source: toy episodes cut EXACTLY like the training loader
# ---------------------------------------------------------------------------
class ToyClipSet:
    """
    Clips from ToyTwoRoom episodes, shaped exactly like TwoRoomClips items:
    {"pixels": (T,3,H,W) float 0..1, "action": (T,A) float}, frames and actions
    at strided raw indices (stride = frameskip), the same slicing rule as
    tworoom_data.clip_indices. Episodes are driven by heuristic_action.

    step_factor scales the WORLD's response to an action (the recorded action
    itself is untouched): 1.0 reproduces the toy's own dynamics; smaller values
    slow the dot, letting Check C2 match the real data's relative speed while
    keeping actions in-distribution for the action encoder.

    Frames are rendered lazily on access to keep memory flat.
    """

    def __init__(self, env: ToyTwoRoom, spec: ClipSpec, n_clips: int,
                 img_size: int, seed: int, step_factor: float = 1.0,
                 ep_len: int = 92):
        self.env, self.spec, self.img = env, spec, img_size
        rng = np.random.default_rng(seed)
        lo, hi = env.margin, env.size - env.margin

        def sample(on_left):
            x = (rng.uniform(lo, env.wall_x - 4) if on_left
                 else rng.uniform(env.wall_x + 4, hi))
            return np.array([x, rng.uniform(lo, hi)], dtype=np.float32)

        def step(pos, a):
            a = np.clip(a, -ACTION_CLIP, ACTION_CLIP)
            nxt = pos + a * ACTION_SCALE * step_factor
            nxt = np.clip(nxt, lo, hi)
            if env.blocked(pos, nxt):
                return pos.copy()
            return nxt

        self.clips = []                 # list of (positions (T,2), actions (T,A))
        moves = []
        while len(self.clips) < n_clips:
            left = rng.random() < 0.5
            pos, target = sample(left), sample(not left)
            P, A = [], []
            for _ in range(ep_len):
                a = heuristic_action(env, pos, target, rng)
                P.append(pos.copy()); A.append(a)
                nxt = step(pos, a)
                moves.append(float(np.linalg.norm(nxt - pos)))
                pos = nxt
                if np.linalg.norm(pos - target) < 3.0:
                    target = sample(rng.random() < 0.5)
            P, A = np.array(P), np.array(A, dtype=np.float32)
            for s in range(0, ep_len - spec.span + 1, spec.frameskip):
                idx = s + np.arange(spec.num_steps) * spec.frameskip
                self.clips.append((P[idx], A[idx]))
                if len(self.clips) >= n_clips:
                    break
        self.mean_step = float(np.mean(moves)) if moves else 0.0

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i: int) -> dict:
        P, A = self.clips[int(i)]
        px = np.stack([self.env.render(p, self.img) for p in P]).astype(np.float32) / 255.0
        return {"pixels": np.transpose(px, (0, 3, 1, 2)), "action": A}


# ---------------------------------------------------------------------------
# CHECK M: the real data's own motion, in its own units
# ---------------------------------------------------------------------------
def real_motion_stats(f, frameskip: int, max_eps: int = 200):
    """
    Per-step and per-frameskip displacement of pos_agent inside episodes, plus
    the observed position span. All in the FILE'S OWN units -- no convention
    assumed. Needs ep_offset/ep_len; returns None (with a note) if absent.
    """
    if "ep_offset" not in f or "ep_len" not in f:
        print("  (ep_offset/ep_len not in the file; motion stats skipped)")
        return None
    off = np.asarray(f["ep_offset"][:max_eps])
    ln = np.asarray(f["ep_len"][:max_eps])
    d1, dk = [], []
    xs, ys = [], []
    for o, l in zip(off, ln):
        p = np.asarray(f["pos_agent"][o:o + l], dtype=np.float64)
        if len(p) < frameskip + 1:
            continue
        d1.append(np.linalg.norm(np.diff(p, axis=0), axis=1))
        dk.append(np.linalg.norm(p[frameskip:] - p[:-frameskip], axis=1))
        xs.append(p[:, 0]); ys.append(p[:, 1])
    if not d1:
        return None
    d1 = np.concatenate(d1); dk = np.concatenate(dk)
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    span = float(np.mean([xs.max() - xs.min(), ys.max() - ys.min()]))
    return {"step_med": float(np.median(d1)), "step_mean": float(d1.mean()),
            "fs_med": float(np.median(dk)),
            "x": (float(xs.min()), float(xs.max())),
            "y": (float(ys.min()), float(ys.max())),
            "span": span, "n_eps": int(len(off))}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run dir with manifest.json + checkpoints")
    p.add_argument("--h5", default="~/Downloads/tworoom.h5",
                   help="the real data file this run trained on")
    p.add_argument("--max-batches", type=int, default=4,
                   help="4 mirrors the training-time metric exactly; 2 for a quick pass")
    p.add_argument("--pairs", type=int, default=200,
                   help="frames per side for the latent-cloud check; 100 for a quick pass")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-a", action="store_true",
                   help="skip Check A (use once A has already reproduced the train log)")
    args = p.parse_args()

    run_dir = Path(args.run)
    h5_path = str(Path(args.h5).expanduser())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["config"]
    d, t, m = cfg["data"], cfg["training"], cfg["model"]
    img_size = m.get("img_size", 32)

    print("=" * 70)
    print(f"BRIDGE + DOMAIN CHECKS v2  --  {run_dir.name}")
    print("=" * 70)

    print("\ncheckpoint files in the run dir:")
    for name in ("ckpt.pt", "ckpt_best.pt", "ckpt_final.pt"):
        fpath = run_dir / name
        print(f"  {name:14s} " + (f"md5 {_md5(fpath)}" if fpath.exists() else "MISSING"))

    spec = ClipSpec(history=m["history_size"], num_preds=t["num_preds"],
                    frameskip=d["frameskip"])
    index = TwoRoomIndex(h5_path, spec)
    ds = TwoRoomClips(h5_path, index)
    train_idx, val_idx = split_indices(len(index.starts), d["train_split"],
                                       d["data_seed"])
    print(f"\ndata: {h5_path}")
    print(f"  clips {len(index.starts)}  train {len(train_idx)}  val {len(val_idx)}")
    fp = data_fingerprint(index, h5_path)
    fp_train = _find_key(manifest, "data_sha256")
    if fp_train is None:
        print(f"  data fingerprint (nothing in manifest to compare): {fp}")
    elif fp == fp_train:
        print("  data fingerprint MATCHES the training manifest -- same clips, same order.")
    else:
        print("  data fingerprint MISMATCH vs training manifest -- STOP and investigate:")
        print(f"    here    : {fp}\n    training: {fp_train}")

    # -- CHECK A --------------------------------------------------------------
    if args.skip_a:
        print("\nCHECK A skipped (--skip-a).")
    else:
        n_val_used = min(len(val_idx), args.max_batches * t["batch_size"])
        print(f"\nCheck A uses the first {n_val_used} validation clips "
              f"({args.max_batches} batches of {t['batch_size']}), same slices as the train log.")
        for name in ("ckpt_best.pt", "ckpt_final.pt"):
            path = run_dir / name
            if not path.exists():
                print(f"\nCHECK A on {name}: file missing, skipped.")
                continue
            try:
                model = _build_model(m)
                ck = _load_weights(model, path)
                print(f"\nCHECK A on {name}  (epoch field {ck.get('epoch')} -- "
                      f"stores epoch+1 -- step {ck.get('step')})")
                s0 = step0_latent_error(model, ds, val_idx, cfg,
                                        max_batches=args.max_batches)
                model.eval()   # step0_latent_error returns with the model in train mode
                print(f"  step0_err {s0['step0_err']}   real_step {s0['real_step']}   "
                      f"ratio {s0['err_over_step']}")
            except Exception:
                print(f"  CHECK A on {name} failed:")
                traceback.print_exc()

    # model for the remaining checks
    best = run_dir / ("ckpt_best.pt" if (run_dir / "ckpt_best.pt").exists() else "ckpt.pt")
    model = _build_model(m)
    _load_weights(model, best)
    print(f"\nremaining checks use {best.name}")

    env = ToyTwoRoom()
    rng = np.random.default_rng(args.seed)
    import h5py
    import hdf5plugin  # noqa: F401

    # -- CHECK M: real motion, real units ------------------------------------
    stats = None
    print("\nCHECK M: the real data's own motion (file units, no convention assumed)")
    try:
        with h5py.File(h5_path, "r") as f:
            stats = real_motion_stats(f, d["frameskip"])
        if stats is not None:
            print(f"  positions span x [{stats['x'][0]:.1f}, {stats['x'][1]:.1f}] "
                  f"y [{stats['y'][0]:.1f}, {stats['y'][1]:.1f}] "
                  f"(first {stats['n_eps']} episodes)")
            print(f"  per-step move: median {stats['step_med']:.2f}  "
                  f"mean {stats['step_mean']:.2f}; per-{d['frameskip']}-frame move: "
                  f"median {stats['fs_med']:.2f}")
            rel_real = stats["step_med"] / stats["span"] if stats["span"] > 0 else 0
            print(f"  relative speed: {rel_real*100:.1f}% of the arena per step")
            toy_span = env.size - 2 * env.margin
            print(f"  toy world for contrast: ~{ACTION_SCALE:.0f} units/step over a "
                  f"{toy_span:.0f}-unit arena = {ACTION_SCALE/toy_span*100:.1f}% per step")
    except Exception:
        print("  CHECK M failed:")
        traceback.print_exc()

    # -- CHECK B: do toy renders land on the real latent manifold? -----------
    print(f"\nCHECK B: toy renders vs the real latent cloud ({args.pairs} frames per side)")
    try:
        with h5py.File(h5_path, "r") as f:
            N = f["pixels"].shape[0]
            picks = np.sort(rng.choice(N, size=min(args.pairs, N), replace=False))
            real_frames = f["pixels"][picks]
        lo, hi = env.margin, env.size - env.margin
        toy_pos = rng.uniform(lo, hi, size=(args.pairs, 2)).astype(np.float32)
        toy_frames = [env.render(q, img_size) for q in toy_pos]
        z_real = _encode_frames(model, real_frames)
        z_toy = _encode_frames(model, toy_frames)
        centroid = float(np.linalg.norm(z_real.mean(0) - z_toy.mean(0)))
        toy_to_real = _nn_dist(z_toy, z_real)
        real_to_real = _nn_dist(z_real, z_real, exclude_self=True)
        toy_to_toy = _nn_dist(z_toy, z_toy, exclude_self=True)
        print(f"  distance between the two clouds' centers : {centroid:.2f}")
        print(f"  toy frame -> nearest real latent : median {np.median(toy_to_real):.2f}  "
              f"mean {toy_to_real.mean():.2f}")
        print(f"  yardsticks -- real -> nearest other real: {np.median(real_to_real):.2f}; "
              f"toy -> nearest other toy: {np.median(toy_to_toy):.2f}")
    except Exception:
        print("  CHECK B failed:")
        traceback.print_exc()

    # -- CHECK C: verbatim metric on toy clips, toy's own speed --------------
    n_toy = args.max_batches * t["batch_size"]
    print(f"\nCHECK C: step0_latent_error() verbatim on {n_toy} toy clips, toy dynamics")
    c_set = None
    try:
        c_set = ToyClipSet(env, spec, n_toy, img_size, seed=args.seed, step_factor=1.0)
        s0 = step0_latent_error(model, c_set, np.arange(len(c_set)), cfg,
                                max_batches=args.max_batches)
        model.eval()
        print(f"  step0_err {s0['step0_err']}   real_step {s0['real_step']}   "
              f"ratio {s0['err_over_step']}   (toy mean move/step "
              f"{c_set.mean_step:.2f} units)")
    except Exception:
        print("  CHECK C failed:")
        traceback.print_exc()

    # -- CHECK C2: same, dot slowed to the REAL relative speed ---------------
    print("\nCHECK C2: same clips recipe, toy dot slowed to the real relative speed")
    if stats is None or c_set is None or stats["span"] <= 0 or c_set.mean_step <= 0:
        print("  skipped (needs Check M motion stats and Check C).")
    else:
        try:
            toy_span = env.size - 2 * env.margin
            rel_real = stats["step_med"] / stats["span"]
            rel_toy = c_set.mean_step / toy_span
            factor = rel_real / rel_toy
            print(f"  speed factor = real relative speed / toy relative speed = "
                  f"{factor:.3f}")
            c2_set = ToyClipSet(env, spec, n_toy, img_size, seed=args.seed,
                                step_factor=factor)
            s0 = step0_latent_error(model, c2_set, np.arange(len(c2_set)), cfg,
                                    max_batches=args.max_batches)
            model.eval()
            print(f"  step0_err {s0['step0_err']}   real_step {s0['real_step']}   "
                  f"ratio {s0['err_over_step']}   (mean move/step "
                  f"{c2_set.mean_step:.2f} units)")
        except Exception:
            print("  CHECK C2 failed:")
            traceback.print_exc()

    # -- how to read ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("HOW TO READ")
    print("=" * 70)
    print("M : if the real dot covers a much smaller fraction of its arena per step")
    print("    than the toy dot does, the eval world's dynamics are scaled wrong for")
    print("    this model, independent of how the frames look.")
    print("B : toy->nearest-real far beyond the real cloud's own spacing means toy")
    print("    frames land OFF the latent region the predictor was trained on --")
    print("    a rendering-style gap measured with no coordinate assumptions.")
    print("C : the training-time metric on toy clips at toy speed. Compare with the")
    print("    train log's best value and with the planner-side ~56.")
    print("C2: the same at real speed. C2 ~ train log -> motion scale was the killer;")
    print("    C2 ~ C ~ 56 -> rendering style alone reproduces the planner-side error.")
    ds.close()


if __name__ == "__main__":
    main()
