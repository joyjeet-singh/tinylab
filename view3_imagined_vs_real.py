"""
view3_imagined_vs_real.py -- does the planner's imagination diverge from reality
everywhere (global predictor failure) or specifically at the doorway (local)?

The planner scores plans by rolling the predictor forward in latent space. We
saved that imagined LATENT rollout per episode (`imagined` in latents_cem.npz).
Here we decode both the imagined rollout AND the real trajectory to (x, y) using
the SAME position probe (per prereg_views.md), and overlay them.

Given View 2's finding (predictor ~uniformly wrong, ratio 0.93), the PREDICTION
for View 3 is: the imagined path peels away from the real path steadily and
roughly everywhere -- NOT a clean track that only breaks at the room boundary.
A consistent, growing gap = global predictor failure (confirms Views 1-2).

We plot several missed episodes (the failure mode) plus the one reached episode
for contrast, and quantify divergence-vs-step.

Run:  python3 view3_imagined_vs_real.py --run runs/<dir>
Writes: runs/<dir>/view3_imagined_vs_real.png  + prints divergence stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def fit_probe(latent, pos):
    """latents -> (x,y), same as toy_plan.probe_position. Returns predict fn + R^2."""
    n = len(latent); cut = int(n * 0.75)
    r = Ridge(alpha=1.0).fit(latent[:cut], pos[:cut])
    return r, float(r.score(latent[cut:], pos[cut:]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n-show", type=int, default=6, help="how many episodes to plot")
    args = p.parse_args()
    run = Path(args.run)

    z = np.load(run / "latents_cem.npz", allow_pickle=True)
    latent = z["latent"]; pos = z["pos"]; ep_id = z["ep_id"]
    imagined = z["imagined"]              # object array, per-episode (H, D) or None
    reached = z["ep_reached"].astype(bool)
    goals = z["ep_goal"]

    # probe trained on ALL real states -> maps any latent (real or imagined) to (x,y)
    probe, r2 = fit_probe(latent, pos)
    print(f"probe R^2 (latent->pos, used to decode imagined rollouts): {r2:.4f}")

    # choose episodes: prefer missed ones with a real imagined rollout
    eps = [i for i in range(len(imagined))
           if imagined[i] is not None and not reached[i]]
    reached_eps = [i for i in range(len(imagined))
                   if imagined[i] is not None and reached[i]]
    show = eps[:args.n_show - len(reached_eps[:1])] + reached_eps[:1]
    show = show[:args.n_show]

    # --- divergence stat: imagined-vs-real gap per rollout step, across all missed eps ---
    per_step = {}   # step -> list of gaps
    for i in eps:
        img_xy = probe.predict(imagined[i])          # (H,2) imagined positions
        real_xy = pos[ep_id == i]                     # real positions this episode
        H = min(len(img_xy), len(real_xy))
        for t in range(H):
            per_step.setdefault(t, []).append(
                float(np.linalg.norm(img_xy[t] - real_xy[t])))
    steps = sorted(per_step)
    mean_gap = [np.mean(per_step[t]) for t in steps]
    print("\nimagined-vs-real gap by rollout step (mean over missed episodes):")
    for t in steps:
        print(f"  step {t}: {mean_gap[t]:.2f}  (n={len(per_step[t])})")
    if len(mean_gap) >= 2:
        print(f"\n  gap at step 0 : {mean_gap[0]:.2f}")
        print(f"  gap at last   : {mean_gap[-1]:.2f}")
        print("  steadily growing from a small start -> global predictor drift (confirms Views 1-2)")
        print("  flat/only-late-jump -> different story; report honestly")

    # --- figure: trajectory overlays + the divergence curve ---
    ncol = 3
    nrow = (len(show) + ncol - 1) // ncol + 1
    fig = plt.figure(figsize=(4.3 * ncol, 3.6 * nrow))

    for k, i in enumerate(show):
        ax = fig.add_subplot(nrow, ncol, k + 1)
        img_xy = probe.predict(imagined[i])
        real_xy = pos[ep_id == i]
        g = goals[i]
        ax.plot(real_xy[:, 0], real_xy[:, 1], "o-", color="#2c7fb8",
                ms=3, lw=1.4, label="real path")
        ax.plot(img_xy[:, 0], img_xy[:, 1], "s--", color="#d95f0e",
                ms=3, lw=1.4, label="imagined (planner)")
        ax.scatter([g[0]], [g[1]], marker="*", s=160, color="green",
                   edgecolors="k", zorder=5, label="goal")
        ax.scatter([real_xy[0, 0]], [real_xy[0, 1]], marker="^", s=60,
                   color="black", zorder=5, label="start")
        tag = "REACHED" if reached[i] else "missed"
        ax.set_title(f"ep {i} ({tag})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.legend(fontsize=6, loc="best")

    # divergence curve spanning the bottom row
    axd = fig.add_subplot(nrow, 1, nrow)
    axd.plot(steps, mean_gap, "o-", color="crimson")
    axd.set_xlabel("rollout step"); axd.set_ylabel("mean |imagined - real|")
    axd.set_title("imagined-vs-real divergence over the rollout (mean over missed episodes)\n"
                  "steady growth from small = global predictor drift", fontsize=9)

    fig.suptitle(f"View 3 — imagined vs real trajectories  (decode probe R²={r2:.3f})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = run / "view3_imagined_vs_real.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
