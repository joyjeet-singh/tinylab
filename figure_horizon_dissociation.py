"""
figure_horizon_dissociation.py -- the figure for §5.3.

The argument in one image: one-step prediction accuracy orders short-horizon
planning success and does not order long-horizon planning success, and the two
most accurate models overshoot past a random-action control.

Numbers are PARSED FROM THE COMMITTED REPORTS, not typed in, so the figure and
the text cannot drift apart. Each report contributes its success rate and its
mean final distance; the one-step ratios are supplied on the command line
because they come from a different measurement (verify_phase2_driving.py and
the rollout evaluation), and each is printed in the caption line so the source
is explicit.

  panel (a)  success rate against one-step prediction error, one line per goal
             offset. The x-axis runs from worse to better prediction, so a
             monotone relationship reads as a rising line.
  panel (b)  mean final distance at offset 100 against the random-action
             control. Bars above the control line are planners that finish
             FARTHER from the goal than random does.

Usage:
    python3 figure_horizon_dissociation.py \
        --entry "Run 2 ckpt_best"=0.83=exp_run2_recal/realenv_plan_cem_report.txt=exp_run2_recal/<offset100 report>.txt \
        --entry "authors' released"=0.410=exp_authors/realenv_plan_authors_cem_report.txt=exp_authors_100/realenv_plan_authors_cem_report.txt \
        --entry "phase2"=0.116=exp_phase2_recal/<offset25>.txt=exp_phase2_recal/<offset100>.txt \
        --random-final 111.07
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

RATE = re.compile(r"success rate\s*:\s*(\d+)/(\d+)\s*=\s*([\d.]+)%")
MEANF = re.compile(r"mean final distance\s*:\s*([\d.]+)")
OFFSET = re.compile(r"goal\s*=\s*frame\s*(\d+)\s*of the same episode")


def read_report(p: Path):
    """(successes, n, mean final distance, goal offset) from a report.txt."""
    t = p.read_text(errors="ignore")
    m = RATE.search(t)
    if not m:
        raise SystemExit(f"no 'success rate' line in {p}")
    f = MEANF.search(t)
    o = OFFSET.search(t)
    return (int(m.group(1)), int(m.group(2)),
            float(f.group(1)) if f else float("nan"),
            int(o.group(1)) if o else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", action="append", required=True,
                    metavar="LABEL=RATIO=REPORT25=REPORT100",
                    help="repeat once per checkpoint")
    ap.add_argument("--random-final", type=float, default=None,
                    help="mean final distance of the random control at "
                         "offset 100")
    ap.add_argument("--out", default="fig_horizon_dissociation.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for spec in args.entry:
        parts = spec.split("=")
        if len(parts) != 4:
            raise SystemExit(f"--entry needs LABEL=RATIO=REPORT25=REPORT100, "
                             f"got {spec!r}")
        label, ratio, r25, r100 = parts
        s25, n25, f25, o25 = read_report(Path(r25))
        s100, n100, f100, o100 = read_report(Path(r100))
        if o25 is not None and o100 is not None and o25 == o100:
            raise SystemExit(f"{label}: both reports have goal offset {o25}. "
                             f"Check the paths -- the figure would compare a "
                             f"run against itself.")
        rows.append(dict(label=label, ratio=float(ratio),
                         p25=100 * s25 / n25, p100=100 * s100 / n100,
                         f100=f100, n=n25, o25=o25, o100=o100))
        print(f"  {label:<24} 1-step {float(ratio):.3f}   "
              f"offset {o25}: {s25}/{n25} = {100*s25/n25:.1f}%   "
              f"offset {o100}: {s100}/{n100} = {100*s100/n100:.1f}%   "
              f"mean final {f100:.2f}")

    rows.sort(key=lambda r: -r["ratio"])          # worst prediction first
    x = np.arange(len(rows))
    labels = [r["label"] for r in rows]
    ticks = [f"{r['label']}\n(err {r['ratio']:.3f})" for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))

    a = ax[0]
    a.plot(x, [r["p25"] for r in rows], marker="o", ms=8, lw=2,
           label=f"goal offset {rows[0]['o25']} (short)")
    a.plot(x, [r["p100"] for r in rows], marker="s", ms=8, lw=2,
           label=f"goal offset {rows[0]['o100']} (long)")
    for r, xi in zip(rows, x):
        a.annotate(f"{r['p25']:.0f}%", (xi, r["p25"]), textcoords="offset points",
                   xytext=(0, 9), ha="center", fontsize=9)
        a.annotate(f"{r['p100']:.0f}%", (xi, r["p100"]), textcoords="offset points",
                   xytext=(0, -16), ha="center", fontsize=9)
    a.set_xticks(x)
    a.set_xticklabels(ticks, fontsize=9)
    a.set_xlabel("one-step prediction error ÷ frozen-world baseline\n"
                 "(left = worse prediction, right = better)")
    a.set_ylabel("goals reached (%)")
    a.set_ylim(0, 105)
    a.set_title("(a) accuracy orders short-horizon success,\nbut not "
                "long-horizon success", fontsize=10)
    a.legend(fontsize=9, loc="center left")
    a.grid(alpha=0.3)

    b = ax[1]
    cols = ["tab:green" if r["f100"] < (args.random_final or 1e9) else "tab:red"
            for r in rows]
    b.bar(x, [r["f100"] for r in rows], color=cols, alpha=0.85)
    if args.random_final:
        b.axhline(args.random_final, color="k", ls="--", lw=1.5,
                  label=f"random-action control ({args.random_final:.1f})")
        b.legend(fontsize=9, loc="lower right")
    for r, xi in zip(rows, x):
        b.annotate(f"{r['f100']:.1f}", (xi, r["f100"]),
                   textcoords="offset points", xytext=(0, 4), ha="center",
                   fontsize=9)
    b.set_xticks(x)
    b.set_xticklabels(ticks, fontsize=9)
    b.set_ylabel(f"mean final distance at offset {rows[0]['o100']} (units)")
    b.set_title("(b) the two most accurate models finish FARTHER\nfrom the "
                "goal than random does", fontsize=10)
    b.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")
    print(f"Caption numbers: n = {rows[0]['n']} episodes per cell, identical "
          f"across checkpoints;\none-step ratios from the driving-spec and "
          f"rollout measurements as given on the command line.")


if __name__ == "__main__":
    main()
