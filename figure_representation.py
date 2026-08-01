"""
figure_representation.py -- Figure 1 for §4.1.

Three panels, one argument: the latent space is a near-linear encoding of
position, transitions within it carry the action that produced them, and no run
collapsed. Everything a predictor needs is present; §4.3-4.4 show it is not
learned.

  (a) probe-decoded position against true position, in arena coordinates, with
      the wall and door drawn. A scatter of predictions against targets would
      show the same R², but drawing it in the arena shows the encoder recovers
      the geometry, including both rooms, rather than just a correlated scalar.
  (b) summed action decoded from a pair of consecutive embeddings.
  (c) mean embedding spread per epoch for both runs -- claim 14, no collapse.

Probes are refitted here rather than read from a file so the figure and the
reported numbers cannot drift apart. Same protocol as
probe_encoder_comparison.py: ridge, 80/20 held out, seed 0.

Usage:
    python3 figure_representation.py \
        --run-a runs/<run0 dir> --run-b runs/<phase2 dir>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

WALL_X, WALL_HALF = 112.0, 5.0
DOOR_LO, DOOR_HI = 35.0, 63.0
ARENA_LO, ARENA_HI = 14.0, 208.0
EPOCH_RE = re.compile(r"epoch\s+(\d+):.*?spread\s+([0-9.eE+-]+)")


def ridge_fit_predict(X, y, lam=1.0, seed=0):
    """Held-out predictions and R², same protocol as the probe script."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr = np.hstack([(X[tr] - mu) / sd, np.ones((cut, 1))])
    Xte = np.hstack([(X[te] - mu) / sd, np.ones((len(te), 1))])
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ y[tr])
    pred = Xte @ W
    ss_res = ((y[te] - pred) ** 2).sum(0)
    ss_tot = ((y[te] - y[te].mean(0)) ** 2).sum(0)
    return pred, y[te], float(np.mean(1 - ss_res / ss_tot))


def spread_series(run_dir: Path):
    """(epochs, spread) from the run's own log — run dir first, never parent."""
    for name in ("log.jsonl", "metrics.jsonl"):
        f = run_dir / name
        if not f.exists():
            continue
        ep, sp = [], []
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            # eval records ONLY. The train records are per-step (thousands of
            # them, all carrying an "epoch" field), and including them turns
            # this panel into a spike plot of within-batch variation rather
            # than the per-epoch held-out spread the caption claims.
            if r.get("kind") not in (None, "eval"):
                continue
            if "epoch" in r and r.get("spread") is not None:
                ep.append(int(r["epoch"]))
                sp.append(float(r["spread"]))
        if len(set(ep)) >= 2:
            o = np.argsort(ep)
            return np.array(ep)[o], np.array(sp)[o]
    for cand in sorted(run_dir.glob("*.log")):
        hits = EPOCH_RE.findall(cand.read_text(errors="ignore"))
        if hits:
            ep = np.array([int(h[0]) for h in hits])
            sp = np.array([float(h[1]) for h in hits])
            o = np.argsort(ep)
            return ep[o], sp[o]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="reference-faithful (Run 0)")
    ap.add_argument("--run-b", required=True, help="corrected pipeline (phase2)")
    ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--show", type=int, default=800,
                    help="points drawn in panel (a); all are used for the fit")
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--out", default="fig1_representation.png")
    args = ap.parse_args()

    import torch
    import h5py
    import hdf5plugin  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from probe_encoder_comparison import load_encoder

    IMEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    ISTD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    rng = np.random.default_rng(0)
    fs = args.frameskip

    with h5py.File(str(Path(args.h5).expanduser()), "r") as f:
        off, ln = np.asarray(f["ep_offset"][:]), np.asarray(f["ep_len"][:])
        ok = np.where(ln > fs + 2)[0]
        eps = rng.choice(ok, size=args.frames, replace=True)
        starts = np.array([int(off[e]) + int(rng.integers(0, ln[e] - fs - 1))
                           for e in eps])
        px0 = np.stack([f["pixels"][s] for s in starts])
        px1 = np.stack([f["pixels"][s + fs] for s in starts])
        pos = np.stack([np.asarray(f["pos_agent"][s]) for s in starts])
        act = np.stack([np.asarray(f["action"][s:s + fs],
                                   dtype=np.float64).sum(0) for s in starts])
    keep = ~np.isnan(act).any(axis=1)
    px0, px1, pos, act = px0[keep], px1[keep], pos[keep], act[keep]

    model, conv, _ = load_encoder(args.run_a)
    img_norm = bool(conv.get("imagenet_pixels"))

    def embed(px_u8):
        out = []
        for i in range(0, len(px_u8), 32):
            x = torch.from_numpy(px_u8[i:i + 32].astype(np.float32) / 255.0
                                 ).permute(0, 3, 1, 2)
            if img_norm:
                x = (x - IMEAN) / ISTD
            with torch.no_grad():
                out.append(model.encode({"pixels": x.unsqueeze(1)})
                           ["emb"][:, 0].numpy())
        return np.concatenate(out, 0)

    Z0, Z1 = embed(px0), embed(px1)
    pos_pred, pos_true, r2_pos = ridge_fit_predict(Z0, pos)
    act_pred, act_true, r2_act = ridge_fit_predict(np.hstack([Z0, Z1]), act)
    print(f"  position R² {r2_pos:.4f}   action R² {r2_act:.4f}   "
          f"(held-out, ridge, 80/20)")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) decoded position over the arena
    n = min(args.show, len(pos_true))
    a = ax[0]
    a.add_patch(plt.Rectangle((WALL_X - WALL_HALF, ARENA_LO), 2 * WALL_HALF,
                              DOOR_LO - ARENA_LO, color="0.75", zorder=1))
    a.add_patch(plt.Rectangle((WALL_X - WALL_HALF, DOOR_HI), 2 * WALL_HALF,
                              ARENA_HI - DOOR_HI, color="0.75", zorder=1))
    for i in range(n):
        a.plot([pos_true[i, 0], pos_pred[i, 0]], [pos_true[i, 1], pos_pred[i, 1]],
               color="0.6", lw=0.4, zorder=2)
    a.scatter(pos_true[:n, 0], pos_true[:n, 1], s=5, c="tab:blue", zorder=3,
              label="true")
    a.scatter(pos_pred[:n, 0], pos_pred[:n, 1], s=5, c="tab:orange", zorder=4,
              label="decoded")
    a.set_xlim(ARENA_LO - 5, ARENA_HI + 5)
    a.set_ylim(ARENA_LO - 5, ARENA_HI + 5)
    a.set_aspect("equal")
    a.set_title(f"(a) position decoded from the embedding\n"
                f"held-out R² = {r2_pos:.4f}", fontsize=10)
    a.set_xlabel("arena x"); a.set_ylabel("arena y")
    a.legend(fontsize=8, loc="lower right")

    # (b) decoded action
    b = ax[1]
    lim = np.abs(act_true).max() * 1.05
    b.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.8, ls=":")
    b.scatter(act_true[:, 0], act_pred[:, 0], s=4, alpha=0.35, label="x")
    b.scatter(act_true[:, 1], act_pred[:, 1], s=4, alpha=0.35, label="y")
    b.set_xlim(-lim, lim); b.set_ylim(-lim, lim); b.set_aspect("equal")
    b.set_title(f"(b) summed action decoded from $(z_t, z_{{t+k}})$\n"
                f"held-out R² = {r2_act:.4f}", fontsize=10)
    b.set_xlabel("true summed action"); b.set_ylabel("decoded")
    b.legend(fontsize=8, loc="upper left")

    # (c) spread, both runs
    c = ax[2]
    drew = False
    for run, label in ((args.run_a, "reference-faithful"),
                       (args.run_b, "corrected pipeline")):
        ep, sp = spread_series(Path(run))
        if ep is None:
            print(f"  no spread series found for {run} — panel (c) incomplete")
            continue
        c.plot(ep, sp, marker="o", ms=4, label=f"{label} ({sp.min():.2f}–"
                                               f"{sp.max():.2f})")
        drew = True
    c.set_ylim(0, max(1.2, c.get_ylim()[1]))
    c.axhspan(0, 0.3, color="tab:red", alpha=0.08)
    c.text(0.02, 0.04, "collapse region", transform=c.transAxes, fontsize=8,
           color="tab:red")
    c.set_xlabel("epoch"); c.set_ylabel("mean embedding spread")
    c.set_title("(c) no collapse under either configuration", fontsize=10)
    if drew:
        c.legend(fontsize=8, loc="lower right")
    c.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"  wrote {args.out}")
    print(f"\n  Caption numbers to use: position R² {r2_pos:.4f}, "
          f"action R² {r2_act:.4f}, {len(pos_true)} held-out frames "
          f"of {len(Z0)} encoded.")


if __name__ == "__main__":
    main()
