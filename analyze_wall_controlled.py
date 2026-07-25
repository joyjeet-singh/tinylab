"""
analyze_wall_controlled.py -- does crossing the wall cost anything ONCE YOU
ACCOUNT FOR HOW FAR THE GOAL IS?

WHY THIS EXISTS
---------------
Two runs gave contradictory readings on the wall:

    goal offset  25 : same-room 29/43 = 67.4%   cross-wall  5/5  = 100.0%
    goal offset 100 : same-room  7/11 = 63.6%   cross-wall 17/39 =  43.6%

Neither is significant on its own (Fisher exact p = 0.30 and p = 0.31), each
has one small cell, and they point in opposite directions. Worse, the two
variables are tangled: a goal in the other room is usually a FARTHER goal, so
a "wall effect" and a "distance effect" look identical in a raw split.

This script separates them using only data already on disk. It reports:

  1. composition and the confound itself -- the distance distribution for
     same-room vs cross-wall episodes, so the overlap (or lack of it) is
     visible before any test is read;
  2. the raw geometry split per run, with an exact Fisher test;
  3. success rate by distance band, ignoring geometry;
  4. the STRATIFIED comparison -- same-room vs cross-wall WITHIN distance
     bands, pooled across bands and runs by Mantel-Haenszel, with a
     stratified permutation test (geometry labels shuffled within band, so
     distance cannot leak into the result);
  5. a rank test of whether distance alone separates successes from misses.

No scipy required. If the stratified test comes back non-significant with
wide bands, the honest conclusion is that these samples cannot separate the
two explanations, and a geometry-balanced evaluation is needed.

Usage (one or more runs; --offsets must match --cem in order and length):
    python3 analyze_wall_controlled.py \
        --cem realenv_plan_cem.json exp_offset100/realenv_plan_cem.json \
        --offsets 25 100
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

WALL_X = 112.0


def fisher_2x2(a, b, c, d):
    """Two-sided Fisher exact for [[a,b],[c,d]] (rows=groups, cols=succ,fail)."""
    n, r1, r2, c1 = a + b + c + d, a + b, c + d, a + c
    if min(r1, r2, c1, n - c1) < 0 or n == 0:
        return 1.0

    def pr(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    p0, tot = pr(a), 0.0
    for x in range(max(0, c1 - r2), min(r1, c1) + 1):
        p = pr(x)
        if p <= p0 * (1 + 1e-9):
            tot += p
    return min(1.0, tot)


def mann_whitney_z(x, y):
    """Normal-approximation z for whether x tends to exceed y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    u = sum((xi > y).sum() + 0.5 * (xi == y).sum() for xi in x)
    n1, n2 = len(x), len(y)
    mu = n1 * n2 / 2
    sd = (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5
    return (u - mu) / sd if sd > 0 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cem", nargs="+", required=True)
    p.add_argument("--offsets", nargs="+", type=int, required=True)
    p.add_argument("--h5", default="~/Downloads/tworoom.h5")
    p.add_argument("--bands", type=int, default=3,
                   help="distance bands per run for the stratified test")
    p.add_argument("--perm", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if len(args.cem) != len(args.offsets):
        raise SystemExit("--cem and --offsets must have the same length")

    import h5py
    import hdf5plugin  # noqa: F401

    rows = []   # (run_label, cross:bool, dist:float, success:bool)
    with h5py.File(str(Path(args.h5).expanduser()), "r") as f:
        off = np.asarray(f["ep_offset"][:])
        ln = np.asarray(f["ep_len"][:])
        for idx, (path, goff) in enumerate(zip(args.cem, args.offsets)):
            d = json.loads(Path(path).read_text())
            label = f"[{idx}] offset {goff} ({Path(path).parent.name or '.'})"
            for r in d["results"]:
                e = r["episode"]
                if e >= len(off) or ln[e] <= goff:
                    continue
                s = np.asarray(f["pos_agent"][off[e]], dtype=np.float64)
                g = np.asarray(f["pos_agent"][off[e] + goff], dtype=np.float64)
                if r.get("trivial_start"):
                    continue
                rows.append((label, (s[0] - WALL_X) * (g[0] - WALL_X) < 0,
                             float(np.linalg.norm(s - g)), bool(r["success"])))

    if not rows:
        raise SystemExit("no usable episodes found -- check --h5 and --offsets")

    labels = sorted({r[0] for r in rows})
    print("=" * 70)
    print("IS IT THE WALL, OR IS IT THE DISTANCE?")
    print("=" * 70)

    # ---- 1-3: per-run picture -----------------------------------------
    strata = []   # list of (name, list-of-rows) for the stratified test
    for lab in labels:
        R = [r for r in rows if r[0] == lab]
        cr = [r for r in R if r[1]]
        sr = [r for r in R if not r[1]]
        print(f"\n--- {lab}  ({len(R)} non-trivial episodes) ---")
        print(f"  composition: cross-wall {len(cr)}/{len(R)} "
              f"= {len(cr)/len(R)*100:.0f}% of episodes")
        for nm, G in (("same-room", sr), ("cross-wall", cr)):
            if not G:
                continue
            dd = [g[2] for g in G]
            print(f"  {nm:<11} n={len(G):<3} success "
                  f"{sum(g[3] for g in G)}/{len(G)} "
                  f"= {sum(g[3] for g in G)/len(G)*100:5.1f}%   "
                  f"goal distance median {np.median(dd):6.1f} "
                  f"[{min(dd):.0f}-{max(dd):.0f}]")
        if sr and cr:
            a, b = sum(g[3] for g in sr), len(sr) - sum(g[3] for g in sr)
            c, dd_ = sum(g[3] for g in cr), len(cr) - sum(g[3] for g in cr)
            print(f"  raw geometry split: Fisher exact two-sided "
                  f"p = {fisher_2x2(a, b, c, dd_):.3f}")
            zc = mann_whitney_z([g[2] for g in cr], [g[2] for g in sr])
            print(f"  CONFOUND CHECK: are cross-wall goals farther? "
                  f"rank z = {zc:+.2f} "
                  f"({'yes, strongly' if zc > 1.96 else 'not clearly'})")
        zs = mann_whitney_z([g[2] for g in R if g[3]],
                            [g[2] for g in R if not g[3]])
        print(f"  does distance alone predict failure? rank z = {zs:+.2f} "
              f"({'yes' if abs(zs) > 1.96 else 'not significant'})")

        dists = np.array([g[2] for g in R])
        edges = np.quantile(dists, np.linspace(0, 1, args.bands + 1))
        edges[-1] += 1e-6
        print(f"  success by distance band (geometry ignored):")
        for i in range(args.bands):
            B = [g for g in R if edges[i] <= g[2] < edges[i + 1]]
            if B:
                print(f"    {edges[i]:5.0f}-{edges[i+1]:5.0f}: "
                      f"{sum(g[3] for g in B)}/{len(B)} "
                      f"= {sum(g[3] for g in B)/len(B)*100:5.1f}%")
                strata.append((f"{lab}|{i}", B))

    # ---- 4: stratified test -------------------------------------------
    print("\n" + "=" * 70)
    print("STRATIFIED: same-room vs cross-wall WITHIN distance bands")
    print("=" * 70)
    num = den = 0.0
    usable = 0
    for name, B in strata:
        sr = [g for g in B if not g[1]]
        cr = [g for g in B if g[1]]
        if not sr or not cr:
            continue
        usable += 1
        a = sum(g[3] for g in sr); b = len(sr) - a
        c = sum(g[3] for g in cr); d = len(cr) - c
        n = len(B)
        num += a * d / n
        den += b * c / n
        print(f"  {name:<14} same-room {a}/{len(sr)}   cross-wall {c}/{len(cr)}")
    if usable == 0 or den == 0:
        print("  no band contains both geometries -- the two variables are")
        print("  completely confounded in this data. A geometry-balanced")
        print("  evaluation is required to answer the question.")
        return
    print(f"\n  Mantel-Haenszel odds ratio (same-room vs cross-wall, "
          f"pooled over {usable} bands): {num/den:.2f}")
    print("  (>1 means same-room easier once distance is held fixed)")

    rng = np.random.default_rng(args.seed)
    obs = num / den

    def stat(assign):
        n_, d_ = 0.0, 0.0
        k = 0
        for _, B in strata:
            lab = assign[k:k + len(B)]
            k += len(B)
            sr = [g for g, L in zip(B, lab) if not L]
            cr = [g for g, L in zip(B, lab) if L]
            if not sr or not cr:
                continue
            a = sum(g[3] for g in sr); b = len(sr) - a
            c = sum(g[3] for g in cr); d = len(cr) - c
            n_ += a * d / len(B)
            d_ += b * c / len(B)
        return n_ / d_ if d_ > 0 else float("nan")

    base = np.concatenate([np.array([g[1] for g in B]) for _, B in strata])
    sizes = [len(B) for _, B in strata]
    cnt = 0
    ok = 0
    for _ in range(args.perm):
        perm = []
        k = 0
        for s in sizes:
            block = base[k:k + s].copy()
            rng.shuffle(block)
            perm.append(block)
            k += s
        v = stat(np.concatenate(perm))
        if not np.isnan(v):
            ok += 1
            if (v >= obs) if obs >= 1 else (v <= obs):
                cnt += 1
    if ok:
        pval = min(1.0, 2 * cnt / ok)
        print(f"  stratified permutation test (labels shuffled within band, "
              f"{ok} draws): two-sided p = {pval:.3f}")
        print("\n  READ: p below 0.05 -> the wall costs something beyond "
              "distance.")
        print("  p well above 0.05 with small bands -> these samples cannot "
              "separate")
        print("  the two explanations; run a geometry-balanced evaluation "
              "before")
        print("  making any claim about the wall in the paper.")


if __name__ == "__main__":
    main()
