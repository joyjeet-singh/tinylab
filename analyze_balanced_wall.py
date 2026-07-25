"""
analyze_balanced_wall.py -- the PRE-REGISTERED analysis of the geometry-balanced
wall experiment. Implements the primary test named in
docs/prereg_2026-07-25_wall_balanced_eval.md and nothing else, so there is no
question afterwards about which test was chosen when.

Order of business:
  0. integrity: confirm the episodes analysed are the committed ones and that
     the distance matching actually held (if it did not, the design failed and
     no test is reported);
  1. PRIMARY: matched-pair exact test. Each pair is one same-room and one
     cross-wall episode at the same goal distance. Pairs where the two
     episodes disagree are the informative ones; under the null they split
     50/50. Exact binomial, two-sided.
  2. secondary: unpaired rates with a difference and its confidence interval,
     for readers who prefer the simple table.
  3. control decomposition: the same split for the random-action arm. If the
     random control ALSO shows a geometry gap, the gap belongs to the task
     (goals across the wall are intrinsically harder to stumble into), not to
     the planner. This distinction is the reason the control is run on the
     identical episode set.

Usage, after both arms have run on the balanced set:
    python3 analyze_balanced_wall.py --cem realenv_plan_cem.json \
        --random realenv_plan_random.json --episodes ../balanced_episodes.json
"""
from __future__ import annotations

import argparse
import json
from math import comb, sqrt
from pathlib import Path


def exact_binom_two_sided(k, n):
    """P(|X - n/2| >= |k - n/2|) for X ~ Binom(n, 1/2)."""
    if n == 0:
        return 1.0
    hi = max(k, n - k)
    return min(1.0, 2 * sum(comb(n, j) for j in range(hi, n + 1)) / 2 ** n)


def wilson_diff(a, na, b, nb):
    """Crude normal-approximation CI for a difference of proportions."""
    if na == 0 or nb == 0:
        return (float("nan"), float("nan"))
    pa, pb = a / na, b / nb
    se = sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    d = pa - pb
    return (d - 1.96 * se, d + 1.96 * se)


def by_pair(results):
    out = {}
    for r in results:
        pid, g = r.get("pair_id"), r.get("geometry")
        if pid is None or g is None:
            continue
        out.setdefault(pid, {})[g] = r
    return {k: v for k, v in out.items() if "same" in v and "cross" in v}


def report_arm(name, results, primary):
    pairs = by_pair(results)
    same_s = sum(v["same"]["success"] for v in pairs.values())
    cross_s = sum(v["cross"]["success"] for v in pairs.values())
    n = len(pairs)
    print(f"\n--- {name} ---")
    print(f"  complete pairs: {n}")
    print(f"  same-room  {same_s}/{n} = {same_s/max(n,1)*100:5.1f}%")
    print(f"  cross-wall {cross_s}/{n} = {cross_s/max(n,1)*100:5.1f}%")
    b = sum(1 for v in pairs.values()
            if v["same"]["success"] and not v["cross"]["success"])
    c = sum(1 for v in pairs.values()
            if v["cross"]["success"] and not v["same"]["success"])
    p = exact_binom_two_sided(b, b + c)
    lo, hi = wilson_diff(same_s, n, cross_s, n)
    print(f"  discordant pairs: same-only {b}, cross-only {c} "
          f"-> exact two-sided p = {p:.4f}")
    print(f"  rate difference (same - cross): "
          f"{(same_s-cross_s)/max(n,1)*100:+.1f} points "
          f"[95% CI {lo*100:+.1f}, {hi*100:+.1f}]")
    if primary:
        print("  ^ THIS IS THE PRE-REGISTERED PRIMARY TEST")
    return p, (same_s - cross_s) / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cem", default="realenv_plan_cem.json")
    ap.add_argument("--random", default="realenv_plan_random.json")
    ap.add_argument("--episodes", default="balanced_episodes.json")
    ap.add_argument("--caliper-tolerance", type=float, default=0.01)
    args = ap.parse_args()

    spec = json.loads(Path(args.episodes).read_text())
    cem = json.loads(Path(args.cem).read_text())

    print("=" * 70)
    print("BALANCED WALL EXPERIMENT -- pre-registered analysis")
    print("=" * 70)

    # ---- 0. integrity ----------------------------------------------------
    committed = {(r["episode"], r["geometry"], r["pair_id"])
                 for r in spec["episodes"]}
    seen = {(r["episode"], r.get("geometry"), r.get("pair_id"))
            for r in cem["results"]}
    print(f"\nintegrity:")
    print(f"  committed episodes {len(committed)}, evaluated {len(seen)}, "
          f"exact match: {committed == seen}")
    if committed != seen:
        print("  MISMATCH -- the evaluation did not run the committed set. "
              "Stopping; fix the run before reading any test.")
        return
    print(f"  guard: {cem.get('guard', {}).get('status')}   "
          f"goal offset: {cem.get('protocol', {}).get('goal_offset')}   "
          f"design power: {spec.get('power')}")

    gaps = {}
    for r in spec["episodes"]:
        gaps.setdefault(r["pair_id"], []).append(r["sel_dist"])
    worst = max((abs(v[0] - v[1]) for v in gaps.values() if len(v) == 2),
                default=0.0)
    cal = float(spec.get("caliper", 0.0))
    ok = worst <= cal + args.caliper_tolerance
    print(f"  matching held: worst within-pair distance gap {worst:.2f} "
          f"vs caliper {cal:g} -> {'yes' if ok else 'NO'}")
    if not ok:
        print("  the design's distance control failed; do not report the "
              "primary test.")
        return

    # ---- 1-2. primary on the planner -------------------------------------
    p_cem, d_cem, n = report_arm("CEM planner", cem["results"], primary=True)

    # ---- 3. control decomposition ----------------------------------------
    p_rnd = d_rnd = None
    if Path(args.random).exists():
        rnd = json.loads(Path(args.random).read_text())
        p_rnd, d_rnd, _ = report_arm("random control", rnd["results"],
                                     primary=False)
    else:
        print(f"\n(no {args.random} found -- run the control arm on the same "
              f"episode set for the decomposition)")

    # ---- verdict against the pre-registered decision rule ----------------
    print("\n" + "=" * 70)
    print("VERDICT (against the pre-registered decision rule)")
    print("=" * 70)
    if p_cem < 0.05 and d_cem > 0:
        print("  The planner does worse on cross-wall goals than on same-room")
        print("  goals at matched distance. The wall costs something beyond")
        print("  distance.")
    elif p_cem < 0.05 and d_cem < 0:
        print("  The planner does BETTER across the wall at matched distance.")
        print("  Surprising; report it and look for what distinguishes")
        print("  cross-wall goals other than the wall.")
    else:
        print("  No difference of the pre-registered size. State in the paper")
        print("  that an effect as large as the one this study was designed to")
        print("  detect is ruled out, and that smaller effects are NOT.")
    if p_rnd is not None:
        if p_rnd < 0.05 and d_rnd > 0:
            print("\n  NOTE: the random control shows the same geometry gap, "
                  "so at least part of it belongs to the task rather than to "
                  "the planner. Compare the two effect sizes before "
                  "attributing anything to planning.")
        else:
            print("\n  The random control shows no geometry gap, so any "
                  "planner effect above is attributable to planning rather "
                  "than to the episodes being intrinsically easier.")


if __name__ == "__main__":
    main()
