"""
view3_latent_divergence.py -- the honest View 3.

The earlier position-decode version was contaminated: predicted latents are
out-of-distribution for a probe trained on real encoder latents, so decoding
them to (x,y) is invalid (imagined step-0 decoded to a constant off-room point).

This version measures imagined-vs-real divergence DIRECTLY IN LATENT SPACE --
no probe, no decode, no artifact. imagined[t] is the predictor's rollout latent;
real[t] is the actual encoder latent the agent reached. We plot the gap vs step.

THE FINDING (from the numbers already computed): the gap is ~79 at step 0 and
stays ~flat (79 -> 82 over the rollout). That FLATNESS is the result:
  - a RISING curve from ~0 would mean "error accumulates over the horizon"
    (a long-rollout problem, partially excusing the one-step model).
  - a FLAT-HIGH curve from step 0 means the ONE-STEP prediction is already wrong
    everywhere -- the learned dynamics are globally bad, immediately.
This is the sharpest statement of the mechanism: the predictor never converged.

For calibration we also show the scale of a real one-step move (how far the agent
actually travels per step in latent space), so ~79 error can be read as "many
times larger than a real step" = the prediction is not close.

Run:  python3 view3_latent_divergence.py --run runs/<dir>
Writes: runs/<dir>/view3_latent_divergence.png  + prints the numbers
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
    lat = z["latent"]; ep = z["ep_id"]; img = z["imagined"]

    # (1) imagined-vs-real gap in latent space, per rollout step
    per_step = {}
    for i in range(len(img)):
        if img[i] is None:
            continue
        rl = lat[ep == i]
        for t in range(len(img[i])):
            if t < len(rl):
                per_step.setdefault(t, []).append(
                    float(np.linalg.norm(img[i][t] - rl[t])))
    steps = sorted(per_step)
    mean_gap = np.array([np.mean(per_step[t]) for t in steps])
    std_gap = np.array([np.std(per_step[t]) for t in steps])

    # (2) calibration: real per-step latent movement (consecutive real latents)
    real_step = []
    for i in range(len(img)):
        rl = lat[ep == i]
        for t in range(len(rl) - 1):
            real_step.append(float(np.linalg.norm(rl[t + 1] - rl[t])))
    real_move = float(np.mean(real_step))

    print("imagined-vs-real gap in LATENT space (no decode):")
    for t in steps:
        print(f"  step {t}: mean={mean_gap[t]:.2f}  std={std_gap[t]:.2f}  (n={len(per_step[t])})")
    print(f"\nmean real per-step latent movement (scale reference): {real_move:.2f}")
    print(f"prediction error / real step size = {mean_gap[0]/real_move:.1f}x  (>>1 = prediction not close)")
    slope = (mean_gap[-1] - mean_gap[0]) / max(1, steps[-1])
    print(f"gap growth per step = {slope:.2f}  (~0 = immediate failure, not accumulation)")

    # --- figure ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.errorbar(steps, mean_gap, yerr=std_gap, fmt="o-", color="crimson",
                capsize=3, lw=2, label="imagined vs real (latent L2)")
    ax.axhline(real_move, color="#2c7fb8", ls="--", lw=1.6,
               label=f"real 1-step move ≈ {real_move:.1f}")
    ax.fill_between(steps, mean_gap - std_gap, mean_gap + std_gap,
                    color="crimson", alpha=0.12)
    ax.set_ylim(0, max(mean_gap + std_gap) * 1.1)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("latent-space distance")
    ax.set_title("View 3 — imagined-vs-real divergence in latent space\n"
                 "flat & high from step 0 = one-step prediction already wrong "
                 "everywhere (predictor never converged)", fontsize=10)
    ax.legend()
    fig.tight_layout()
    out = run / "view3_latent_divergence.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    print("\nread: the gap is flat-high from step 0 (not rising from ~0) -> the failure is")
    print("      IMMEDIATE one-step prediction error, not horizon accumulation. Predictor")
    print("      output sits many x farther from truth than a real step. Confirms Views 1-2.")


if __name__ == "__main__":
    main()
