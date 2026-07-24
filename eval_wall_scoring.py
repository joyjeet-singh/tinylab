"""
eval_wall_scoring.py -- does straight-line latent distance ignore the wall, on
REAL data, with OUR encoder?

WHY
---
The published explanation for LeWM's 87% (vs 97-100 for baselines) is the
"Beyond Euclidean Proximity" argument: two states on opposite sides of the
wall can look CLOSE in summary space while being far apart in reality (the
agent must travel to the door). CEM scores plans by exactly that straight-line
latent distance. This tests the claim in the training domain: sample real
frames, encode them, and compare latent distances for position-matched pairs
that are (a) in the same room vs (b) across the wall and far from the door.
If (b) ~ (a), the encoder's geometry really does ignore the wall, and even a
perfect predictor in the real environment would be steered by a misleading
score -- the paper's own mechanism, confirmed for our model.

The wall and door are DETECTED from the data, not assumed: side-change events
inside episodes give the wall's x and the door's y span in the file's own
units.

Also writes wall_scoring_scatter.png: latent distance vs position distance,
same-room pairs in gray, cross-wall far-from-door pairs in red.

Run from the tinylab folder (venv active):
    python3 eval_wall_scoring.py --run runs/<run_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from toy_model import ToyJEPA


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


def detect_geometry(f, max_eps=300):
    """Wall x and door y span, estimated from side-change events in episodes."""
    off = np.asarray(f["ep_offset"][:max_eps])
    ln = np.asarray(f["ep_len"][:max_eps])
    xs = []
    pos_all = []
    for o, l in zip(off, ln):
        p = np.asarray(f["pos_agent"][o:o + l], dtype=np.float64)
        pos_all.append(p)
    pos_cat = np.concatenate(pos_all)
    wall0 = (pos_cat[:, 0].min() + pos_cat[:, 0].max()) / 2
    mids = []
    for p in pos_all:
        s = np.sign(p[:, 0] - wall0)
        cross = np.where(s[:-1] * s[1:] < 0)[0]
        for c in cross:
            mids.append((p[c] + p[c + 1]) / 2)
    mids = np.array(mids) if mids else np.zeros((0, 2))
    if len(mids) < 10:
        return None
    wall_x = float(np.median(mids[:, 0]))
    door_lo, door_hi = np.percentile(mids[:, 1], [5, 95])
    span = float(np.mean([pos_cat[:, 0].max() - pos_cat[:, 0].min(),
                          pos_cat[:, 1].max() - pos_cat[:, 1].min()]))
    return {"wall_x": wall_x, "door_lo": float(door_lo), "door_hi": float(door_hi),
            "door_y": float(np.median(mids[:, 1])), "n_cross": int(len(mids)),
            "span": span}


@torch.no_grad()
def _encode(model, frames, batch=32):
    out = []
    for s in range(0, len(frames), batch):
        chunk = np.asarray(frames[s:s + batch], dtype=np.float32) / 255.0
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).unsqueeze(1)
        out.append(model.encode({"pixels": x})["emb"][:, 0].cpu().numpy())
    return np.concatenate(out, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--h5", default="~/Downloads/tworoom.h5")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--band-frac", type=float, nargs=2, default=(0.05, 0.15),
                   help="'nearby pair' band as fractions of the arena span")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
    m = cfg["model"]
    h5_path = str(Path(args.h5).expanduser())

    import h5py
    import hdf5plugin  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print(f"WALL-GEOMETRY SCORING TEST (in-domain)  --  {run_dir.name}")
    print("=" * 70)

    with h5py.File(h5_path, "r") as f:
        geo = detect_geometry(f)
        if geo is None:
            print("could not detect the wall from episode crossings; stopping.")
            return
        print(f"detected from data: wall x ~ {geo['wall_x']:.1f}; door y in "
              f"[{geo['door_lo']:.1f}, {geo['door_hi']:.1f}] "
              f"(median {geo['door_y']:.1f}; {geo['n_cross']} crossings; "
              f"arena span ~ {geo['span']:.0f})")
        rng = np.random.default_rng(args.seed)
        N = f["pixels"].shape[0]
        picks = np.sort(rng.choice(N, size=min(args.frames, N), replace=False))
        frames = f["pixels"][picks]
        pos = np.asarray(f["pos_agent"][picks], dtype=np.float64)

    ckpt = run_dir / ("ckpt_best.pt" if (run_dir / "ckpt_best.pt").exists()
                      else "ckpt.pt")
    model = _build_model(m)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"encoder: {m.get('encoder','cnn')} at {m.get('img_size',32)}px "
          f"({ckpt.name}); encoding {len(frames)} real frames...")
    z = _encode(model, frames).astype(np.float64)

    # pairwise distances via the gram trick (N x N, small)
    sq = (z * z).sum(1)
    ld = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0, None))
    pd_ = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))

    iu = np.triu_indices(len(z), k=1)
    ld, pd_ = ld[iu], pd_[iu]
    side = pos[:, 0] < geo["wall_x"]
    cross = side[iu[0]] != side[iu[1]]
    door_half = (geo["door_hi"] - geo["door_lo"]) / 2
    door_margin = door_half + 0.05 * geo["span"]
    far = (np.abs(pos[iu[0], 1] - geo["door_y"]) > door_margin) \
        & (np.abs(pos[iu[1], 1] - geo["door_y"]) > door_margin)

    lo, hi = args.band_frac[0] * geo["span"], args.band_frac[1] * geo["span"]
    band = (pd_ >= lo) & (pd_ <= hi)
    g_same = band & ~cross
    g_cross = band & cross & far

    print(f"\n'nearby' band: position distance {lo:.1f}-{hi:.1f} file units "
          f"({args.band_frac[0]*100:.0f}-{args.band_frac[1]*100:.0f}% of span)")
    print(f"  same-room pairs in band            : {g_same.sum():6d}  "
          f"latent dist median {np.median(ld[g_same]):.2f}")
    print(f"  cross-wall, far-from-door, in band : {g_cross.sum():6d}  "
          f"latent dist median {np.median(ld[g_cross]):.2f}")
    if g_same.sum() and g_cross.sum():
        ratio = np.median(ld[g_cross]) / np.median(ld[g_same])
        print(f"  cross/same ratio                   : {ratio:.2f}")
    r = np.corrcoef(pd_, ld)[0, 1]
    print(f"  overall correlation, latent vs position distance: {r:.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for g, c, lab, a in ((g_same, "0.6", "same room", 0.15),
                         (g_cross, "crimson", "cross-wall, far from door", 0.3)):
        idx = np.where(g)[0]
        if len(idx) > 4000:
            idx = np.random.default_rng(0).choice(idx, 4000, replace=False)
        ax.scatter(pd_[idx], ld[idx], s=4, c=c, alpha=a, label=lab)
    ax.axvspan(lo, hi, color="gold", alpha=0.12)
    ax.set_xlabel("true position distance (file units)")
    ax.set_ylabel("latent distance")
    ax.set_title("does latent distance see the wall? (real data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("wall_scoring_scatter.png", dpi=120)
    print("\nwrote wall_scoring_scatter.png")

    print("\nHOW TO READ")
    print("  ratio ~ 1 : the encoder's straight-line distance IGNORES the wall --")
    print("    a plan ending just across the wall scores as good as one that is")
    print("    genuinely near the goal. The published failure mechanism holds for")
    print("    our encoder, in-domain.")
    print("  ratio clearly > 1 : the encoder separates the rooms; straight-line")
    print("    scoring is less misleading than the published account suggests.")


if __name__ == "__main__":
    main()
