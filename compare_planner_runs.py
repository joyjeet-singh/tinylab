"""
compare_planner_runs.py -- compare two planner evaluations that share the same
episode set (e.g. receding-5 vs receding-1), episode by episode.

Two runs of realenv_r2_planner_eval.py with the same --seed and --goal-offset
sample the SAME 50 episodes, so their outcomes are paired and a paired test is
the right one. This script reports:

  - the 2x2 paired table (both / A-only / B-only / neither)
  - exact McNemar (binomial on the discordant pairs, two-sided)
  - "runaway" misses: episodes that ended FARTHER from the goal than they
    started -- the signature of confident travel in a wrong direction
  - the distribution of final distances among misses, which distinguishes
    "went the wrong way fast" from "drifted and stalled"
  - the individual episodes that flipped, with their start-goal distances

Usage (paths, in either order -- A is the baseline):
    python3 compare_planner_runs.py --a ../realenv_plan_cem.json \
                                    --b realenv_plan_cem.json \
                                    --label-a receding5 --label-b receding1
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from math import comb
from pathlib import Path


def load(path):
    d = json.loads(Path(path).read_text())
    return {r["episode"]: r for r in d["results"]}, d


def exact_mcnemar(a_only, b_only):
    n = a_only + b_only
    if n == 0:
        return 1.0
    hi = max(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, k) for k in range(hi, n + 1)) / 2 ** n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    args = p.parse_args()

    A, dA = load(args.a)
    B, dB = load(args.b)
    shared = sorted(set(A) & set(B))
    if not shared:
        print("NO SHARED EPISODES -- these runs used different samples; a "
              "paired test is not valid. Compare rates only.")
        return
    if len(shared) != len(A) or len(shared) != len(B):
        print(f"WARNING: {len(A)} vs {len(B)} episodes, {len(shared)} shared; "
              f"paired analysis uses the shared set only.")

    la, lb = args.label_a, args.label_b
    both = sum(A[e]["success"] and B[e]["success"] for e in shared)
    aonly = sum(A[e]["success"] and not B[e]["success"] for e in shared)
    bonly = sum(B[e]["success"] and not A[e]["success"] for e in shared)
    neither = len(shared) - both - aonly - bonly

    print("=" * 68)
    print(f"PAIRED COMPARISON over {len(shared)} shared episodes")
    print("=" * 68)
    print(f"  {la}: {sum(A[e]['success'] for e in shared)}/{len(shared)}"
          f"   {lb}: {sum(B[e]['success'] for e in shared)}/{len(shared)}")
    print(f"  both {both}   {la}-only {aonly}   {lb}-only {bonly}   "
          f"neither {neither}")
    print(f"  exact McNemar two-sided p = {exact_mcnemar(aonly, bonly):.2e}")

    for lab, R in ((la, A), (lb, B)):
        miss = [R[e] for e in shared if not R[e]["success"]]
        run = [m for m in miss if m["final_dist"] > m["start_goal_dist"]]
        d = [m["final_dist"] for m in miss]
        print(f"\n  {lab} misses: {len(miss)}; ended farther than they "
              f"started: {len(run)}")
        if d:
            print(f"    final distance among misses: median {st.median(d):.1f}"
                  f"  range {min(d):.1f}-{max(d):.1f}")
            print(f"    READ: wide range with large values = wrong-direction "
                  f"travel; narrow band = drift/stall.")

    flips_b = [e for e in shared if B[e]["success"] and not A[e]["success"]]
    flips_a = [e for e in shared if A[e]["success"] and not B[e]["success"]]
    print(f"\n  episodes {lb} won that {la} lost ({len(flips_b)}): "
          + ", ".join(f"{e}(d={A[e]['start_goal_dist']:.0f})" for e in flips_b))
    print(f"  episodes {la} won that {lb} lost ({len(flips_a)}): "
          + ", ".join(f"{e}(d={A[e]['start_goal_dist']:.0f})" for e in flips_a))

    for lab, d in ((la, dA), (lb, dB)):
        pr = d.get("protocol", {})
        print(f"\n  {lab} protocol: receding {pr.get('receding')}, "
              f"goal_offset {pr.get('goal_offset')}, budget {pr.get('budget')},"
              f" seed {pr.get('seed')}, guard {d.get('guard', {}).get('status')}")


if __name__ == "__main__":
    main()
