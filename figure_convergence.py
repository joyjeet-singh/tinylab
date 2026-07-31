"""
figure_convergence.py -- the four-run convergence figure.

The claim it carries: convergence in this reimplementation is
learning-rate-governed, and the action-aggregation deviation changed the trend
without producing convergence.

Absolute prediction losses are NOT comparable across runs -- the latent scale
moves with history_size, the action encoding and the input normalisation. So
each curve is divided by its own mean over epochs 1..N, which leaves the SHAPE
(does it settle, does it trend, how violently does it swing) and discards the
level. That normalisation is stated on the figure so nobody mistakes it for a
raw comparison.

Two panels:
  left   normalised prediction loss per epoch, one line per run
  right  the summary that makes the claim -- each run's volatility (largest
         epoch-to-epoch swing as a fraction of its mean) against its trend
         (second-half mean over first-half mean), with the learning rate
         annotated

Logs are read from each run directory: log.jsonl if present, otherwise any
*.log file containing "epoch N: pred ...". Nothing is hardcoded.

Usage:
    python3 figure_convergence.py \
        --run "Run 0 (lr 5e-5, subsampled)"=runs/<run0> \
        --run "Run 1 (lr 1e-5, bundle)"=runs/<run1> \
        --run "Run 2 (cyclic 1e-5..1e-7)"=runs/<run2> \
        --run "phase2 (lr 5e-5, dense)"=runs/<phase2>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

EPOCH_RE = re.compile(
    r"epoch\s+(\d+):\s*pred\s+([0-9.eE+-]+)"
    r"(?:\s+bell\s+([0-9.eE+-]+))?"
    r"(?:\s+spread\s+([0-9.eE+-]+))?"
    r"(?:\s+R2\s+([0-9.eE+-]+))?")


def read_epochs(run_dir: Path):
    """(source, epochs, pred, r2). Searches the RUN DIRECTORY first and only
    falls back to its parent if the run directory has no log at all.

    The previous version sorted run-dir and parent candidates together, so a
    stray runs/live.log sorted ahead of runs/<name>/live.log and every run read
    the SAME file -- four identical curves. The source is now printed for each
    run so that failure can never be silent again.
    """
    for name in ("log.jsonl", "metrics.jsonl"):
        jl = run_dir / name
        if not jl.exists():
            continue
        ep, pr, r2 = [], [], []
        for line in jl.read_text().splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "epoch" not in rec:
                continue
            val = rec.get("pred", rec.get("pred_loss"))
            if val is None or rec.get("split") not in (None, "val", "eval"):
                continue
            ep.append(int(rec["epoch"]))
            pr.append(float(val))
            r2.append(float(rec.get("R2", rec.get("r2", np.nan))))
        if len(set(ep)) >= 2:
            o = np.argsort(ep)
            return str(jl), np.array(ep)[o], np.array(pr)[o], np.array(r2)[o]

    for where in (sorted(run_dir.glob("*.log")),
                  sorted(run_dir.parent.glob("*.log"))):
        for cand in where:
            hits = EPOCH_RE.findall(cand.read_text(errors="ignore"))
            if hits:
                ep = np.array([int(h[0]) for h in hits])
                pr = np.array([float(h[1]) for h in hits])
                r2 = np.array([float(h[4]) if h[4] else np.nan for h in hits])
                o = np.argsort(ep)
                return str(cand), ep[o], pr[o], r2[o]
        if where is not None and len(where) and where[0].parent == run_dir:
            break
    raise SystemExit(f"no epoch lines found for {run_dir} -- put that run's "
                     f"log inside its own directory")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    metavar='LABEL=DIR',
                    help="repeat once per run; e.g. --run \"Run 0\"=runs/x")
    ap.add_argument("--out", default="fig_convergence.png")
    ap.add_argument("--skip-first", type=int, default=1,
                    help="epochs to exclude from the normalisation (epoch 0 is "
                         "initialisation transient, not training behaviour)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run needs LABEL=DIR, got {spec!r}")
        label, d = spec.split("=", 1)
        src, ep, pr, r2 = read_epochs(Path(d))
        keep = ep >= args.skip_first
        e, p = ep[keep], pr[keep]
        norm = p / p.mean()
        swing = float(np.abs(np.diff(p)).max() / p.mean()) if len(p) > 1 else np.nan
        half = len(p) // 2
        trend = float(p[half:].mean() / p[:half].mean()) if half else np.nan
        runs.append(dict(label=label, src=src, ep=ep, pred=pr, r2=r2, e=e, norm=norm,
                         swing=swing, trend=trend))
        print(f"{label:<34} {len(ep)} ep  volatility {swing:5.2f}  "
              f"trend {trend:5.2f}  raw {p.min():.3f}-{p.max():.3f}")
        print(f"{'':<34} source: {src}")

    srcs = [r["src"] for r in runs]
    if len(set(srcs)) < len(srcs):
        raise SystemExit("two runs resolved to the SAME log file:\n  " +
                         "\n  ".join(srcs) +
                         "\nEach run needs its own log inside its own "
                         "directory. Refusing to draw a figure of one curve "
                         "four times.")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for r in runs:
        ax[0].plot(r["e"], r["norm"], marker="o", ms=4, label=r["label"])
    ax[0].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("prediction loss ÷ that run's own mean")
    ax[0].set_title("Shape, not level\n(absolute losses are not comparable "
                    "across runs)", fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for r in runs:
        ax[1].scatter(r["trend"], r["swing"], s=90)
        ax[1].annotate(r["label"], (r["trend"], r["swing"]), fontsize=8,
                       xytext=(6, 4), textcoords="offset points")
    ax[1].axvline(1.0, color="k", lw=0.6, ls=":")
    ax[1].set_xlabel("trend  (second-half mean ÷ first-half mean)")
    ax[1].set_ylabel("volatility  (largest epoch-to-epoch swing ÷ mean)")
    ax[1].set_title("Left of the dotted line = improving.\nLow on the axis = "
                    "settling.", fontsize=10)
    ax[1].grid(alpha=0.3)

    fig.suptitle("Convergence across the four paid runs "
                 f"(epoch {args.skip_first} onward)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")
    print("\nREAD: a run that converges sits LEFT (improving) and LOW "
          "(settled).")
    print("Run 0 and phase2 share a learning rate; if phase2 sits left of "
          "Run 0 but")
    print("no lower, the deviations changed the trend without buying "
          "stability.")


if __name__ == "__main__":
    main()
