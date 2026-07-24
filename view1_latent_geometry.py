"""
view1_latent_geometry.py -- is the latent space smooth or folded?

Reads latents_cem.npz (raw latents + true positions), fits the SAME position
probe toy_plan uses, and projects the latents onto the probe's two coefficient
directions -- the pre-registered "share the rule not the matrix" basis, so the
axes mean "direction that encodes true x / true y" and folding cannot be a basis
artifact.

Colors each point by true position. Per prereg_views.md:
  SMOOTH  -> color gradient is monotone across the axes (planning-should-work)
  FOLDED  -> gradient reverses/tears, esp. at the doorway (planning-fails)

Also draws the doorway-band states in outline so you can see whether the tear
(if any) sits at the room boundary, as predicted.

Run (in the folder with the run + venv):
  python3 view1_latent_geometry.py --run runs/<dir>
Writes: runs/<dir>/view1_latent_geometry.png  (+ prints the basis diagnostics)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def probe_axes(latent, pos):
    """
    Fit latents -> (x, y) with Ridge (same as toy_plan.probe_position), and
    return the two coefficient directions as an orthonormalized 2D basis, plus
    the held-out R^2 so we can report how well these axes actually encode position.
    """
    n = len(latent)
    cut = int(n * 0.75)
    r = Ridge(alpha=1.0).fit(latent[:cut], pos[:cut])
    r2 = float(r.score(latent[cut:], pos[cut:]))
    W = r.coef_                       # (2, D): row 0 -> x, row 1 -> y
    # orthonormalize the two direction vectors (Gram-Schmidt) so the plot isn't skewed
    a = W[0] / (np.linalg.norm(W[0]) + 1e-9)
    b = W[1] - (W[1] @ a) * a
    b = b / (np.linalg.norm(b) + 1e-9)
    B = np.stack([a, b])              # (2, D) orthonormal
    return B, r2, W


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    run = Path(args.run)

    z = np.load(run / "latents_cem.npz", allow_pickle=True)
    latent = z["latent"]              # (N, D)
    pos = z["pos"]                    # (N, 2) true (x, y)
    door = z["is_doorway"].astype(bool)

    B, r2, W = probe_axes(latent, pos)
    proj = latent @ B.T               # (N, 2) in probe axes

    print(f"probe R^2 on held-out (these axes' quality): {r2:.4f}")
    print(f"projected {len(latent)} states onto probe axes")
    print(f"doorway states: {int(door.sum())}")

    # figure: two panels, colored by true x and by true y, doorway states outlined
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, ci, name in [(axes[0], 0, "true x"), (axes[1], 1, "true y")]:
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=pos[:, ci], cmap="viridis",
                        s=14, alpha=0.85, linewidths=0)
        # outline doorway-band states so a tear-at-the-boundary is visible
        ax.scatter(proj[door, 0], proj[door, 1], s=42, facecolors="none",
                   edgecolors="crimson", linewidths=0.8, label="doorway band")
        ax.set_xlabel("latent axis ≈ encodes true x")
        ax.set_ylabel("latent axis ≈ encodes true y")
        ax.set_title(f"latent geometry, colored by {name}")
        cb = fig.colorbar(sc, ax=ax); cb.set_label(name)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"View 1 — latent space in probe-derived axes  (probe R²={r2:.3f})\n"
        f"smooth gradient = geometry preserved; reversal/tear (esp. at doorway) = folded",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = run / "view1_latent_geometry.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    print("\nread it against prereg_views.md View 1:")
    print("  monotone color across the cloud  -> SMOOTH (folding hypothesis falsified)")
    print("  color reverses / cloud folds back, esp. where crimson doorway ring sits")
    print("     between two color regions      -> FOLDED (hypothesis supported)")


if __name__ == "__main__":
    main()
