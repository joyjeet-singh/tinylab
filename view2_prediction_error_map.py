"""
view2_prediction_error_map.py -- WHERE does the predictor fail?

Basis-independent (unlike View 1): paints one-step prediction error (pred_err)
onto the true room floor plan (x, y). This directly tests the pre-registered
prediction:

  CONCENTRATED at the doorway band -> failure is localized to cross-room
      transitions; supports the "torn at the doorway" reading from View 1.
  UNIFORM across the room -> predictor is GLOBALLY weak, not geometry-specific;
      the View-1 doorway split was likely an artifact -> different finding.

pred_err was ~79 median, tightly clustered (states 71-82), so we expect strong
error everywhere -- the QUESTION is whether it is *structured* (peaks at the
doorway) or *flat*.

Run:  python3 view2_prediction_error_map.py --run runs/<dir>
Writes: runs/<dir>/view2_prediction_error_map.png  + prints doorway-vs-interior stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    run = Path(args.run)

    z = np.load(run / "latents_cem.npz", allow_pickle=True)
    pos = z["pos"]                       # (N,2) true position
    pe = z["pred_err"]                   # (N,) one-step prediction error (nan on final steps)
    door = z["is_doorway"].astype(bool)

    ok = np.isfinite(pe)
    pos, pe_ok, door_ok = pos[ok], pe[ok], door[ok]
    print(f"states with finite pred_err: {ok.sum()}/{len(pe)}")

    # --- the numeric test (basis-independent, does not depend on any plot) ---
    door_err = pe_ok[door_ok]
    int_err = pe_ok[~door_ok]
    print(f"\ndoorway-band error : n={len(door_err):3d}  mean={door_err.mean():.2f}  median={np.median(door_err):.2f}")
    print(f"interior error     : n={len(int_err):3d}  mean={int_err.mean():.2f}  median={np.median(int_err):.2f}")
    ratio = door_err.mean() / int_err.mean()
    print(f"doorway / interior mean ratio : {ratio:.3f}")
    print("  >1.15  -> error concentrates at the doorway (localized failure)")
    print("  ~1.0   -> uniform (global predictor weakness)")

    # error vs x-position (the doorway is a band in x) -- binned means
    xbins = np.linspace(pos[:,0].min(), pos[:,0].max(), 13)
    idx = np.digitize(pos[:,0], xbins)
    bx, be = [], []
    for k in range(1, len(xbins)):
        m = idx == k
        if m.sum() >= 3:
            bx.append(0.5*(xbins[k-1]+xbins[k])); be.append(pe_ok[m].mean())

    # --- figure: scatter heatmap on floor plan + error-vs-x profile ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    sc = axes[0].scatter(pos[:,0], pos[:,1], c=pe_ok, cmap="magma", s=26, alpha=0.9)
    axes[0].scatter(pos[door_ok,0], pos[door_ok,1], s=60, facecolors="none",
                    edgecolors="cyan", linewidths=0.9, label="doorway band")
    # mark the doorway x-band
    if door_ok.any():
        dxlo, dxhi = pos[door_ok,0].min(), pos[door_ok,0].max()
        axes[0].axvspan(dxlo, dxhi, color="cyan", alpha=0.08)
    axes[0].set_xlabel("true x (doorway = cyan band)"); axes[0].set_ylabel("true y")
    axes[0].set_title("one-step prediction error on the floor plan")
    cb = fig.colorbar(sc, ax=axes[0]); cb.set_label("pred_err")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(bx, be, "o-", color="crimson")
    if door_ok.any():
        axes[1].axvspan(dxlo, dxhi, color="cyan", alpha=0.15, label="doorway band")
    axes[1].set_xlabel("true x"); axes[1].set_ylabel("mean pred_err")
    axes[1].set_title("error vs x-position\n(peak over the cyan band = doorway-localized)")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"View 2 — prediction error map   (doorway/interior ratio = {ratio:.2f})",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = run / "view2_prediction_error_map.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    print("\nverdict (per prereg_views.md View 2):")
    if ratio > 1.15:
        print("  -> DOORWAY-CONCENTRATED: supports localized cross-room failure (View-1 tear is real).")
    elif ratio < 1.05:
        print("  -> ~UNIFORM: global predictor weakness; the View-1 doorway split was likely artifact.")
    else:
        print("  -> WEAK/AMBIGUOUS concentration: report the ratio honestly, don't over-claim either way.")


if __name__ == "__main__":
    main()
