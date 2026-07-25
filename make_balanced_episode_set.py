"""
make_balanced_episode_set.py -- build the episode set for the wall experiment,
and REFUSE to green-light it if the set cannot answer the question.

THE PROBLEM THIS SOLVES
-----------------------
Two evaluations gave contradictory readings on whether the dividing wall costs
the planner anything:

    goal offset  25 : same-room 29/43 = 67.4%   cross-wall  5/5  = 100.0%
    goal offset 100 : same-room  7/11 = 63.6%   cross-wall 17/39 =  43.6%

Neither is significant (Fisher p = 0.30, 0.31), and the distance-stratified
test over both runs gave a Mantel-Haenszel odds ratio of 1.26 with a
permutation p of 0.56. The obstacle is not confounding -- at offset 100 the
two geometries already have near-identical goal distances (rank z = +0.11).
The obstacle is SAMPLE SIZE: eleven same-room episodes cannot resolve a
20-point difference.

So: choose episodes deliberately instead of at random -- equal numbers of
same-room and cross-wall, matched one-to-one on goal distance so that
distance is held fixed BY CONSTRUCTION rather than by statistical adjustment.

WHAT THIS SCRIPT DOES (and does not do)
---------------------------------------
1. Scans every episode long enough for the goal offset and labels it
   same-room or cross-wall, with its start-to-goal distance.
2. Prints the availability table -- how many of each geometry exist in each
   distance band. If one geometry is missing at some distances, matching
   there is impossible and the script says so.
3. Builds one-to-one matched pairs within a distance caliper.
4. Runs a power simulation FOR THE NUMBER OF PAIRS ACTUALLY ACHIEVED and
   prints a GO / NO-GO verdict against the pre-registered target.
5. Writes balanced_episodes.json -- the exact, auditable episode list that
   the evaluation will consume, so the selection is fixed before any planning
   happens and the run is exactly reproducible.

It runs NO planner and produces NO success rates. Selection is blind to
outcome by construction: nothing here touches the model.

Usage:
    python3 make_balanced_episode_set.py --pairs 110 --goal-offset 100
"""
from __future__ import annotations

import argparse
import bisect
import json
from math import comb
from pathlib import Path

import numpy as np

WALL_X = 112.0
SECONDS_PER_EPISODE = 21.0   # measured: 50 episodes in 17.5 min at receding 5


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    hi = max(b, c)
    return min(1.0, 2 * sum(comb(n, k) for k in range(hi, n + 1)) / 2 ** n)


def power_sim(n_pairs, p_same, p_cross, rng, trials=4000):
    hits = 0
    for _ in range(trials):
        s = rng.random(n_pairs) < p_same
        c = rng.random(n_pairs) < p_cross
        if mcnemar_p(int((s & ~c).sum()), int((~s & c).sum())) < 0.05:
            hits += 1
    return hits / trials


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="~/Downloads/tworoom.h5")
    p.add_argument("--goal-offset", type=int, default=100)
    p.add_argument("--pairs", type=int, default=110,
                   help="target matched pairs per arm (110 => ~82%% power "
                        "for the observed 20-point difference)")
    p.add_argument("--caliper", type=float, default=6.0,
                   help="max goal-distance difference within a matched pair")
    p.add_argument("--target-power", type=float, default=0.80)
    p.add_argument("--p-same", type=float, default=0.64,
                   help="assumed same-room success rate (offset-100 estimate)")
    p.add_argument("--p-cross", type=float, default=0.44)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default="balanced_episodes.json")
    args = p.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401

    rng = np.random.default_rng(args.seed)
    goff = args.goal_offset

    # ---- 1. scan ----------------------------------------------------------
    print("=" * 70)
    print(f"BALANCED EPISODE SET  --  goal offset {goff}")
    print("=" * 70)
    with h5py.File(str(Path(args.h5).expanduser()), "r") as f:
        off = np.asarray(f["ep_offset"][:])
        ln = np.asarray(f["ep_len"][:])
        cand = np.where(ln > goff)[0]
        print(f"episodes long enough for offset {goff}: {len(cand)} "
              f"of {len(ln)}")
        S = np.asarray(f["pos_agent"][:], dtype=np.float64) \
            if False else None   # never load the whole column
        starts = np.stack([np.asarray(f["pos_agent"][off[e]],
                                      dtype=np.float64) for e in cand])
        goals = np.stack([np.asarray(f["pos_agent"][off[e] + goff],
                                     dtype=np.float64) for e in cand])

    cross = (starts[:, 0] - WALL_X) * (goals[:, 0] - WALL_X) < 0
    dist = np.linalg.norm(starts - goals, axis=1)
    print(f"  same-room candidates : {int((~cross).sum()):5d}   "
          f"distance median {np.median(dist[~cross]):6.1f} "
          f"[{dist[~cross].min():.0f}-{dist[~cross].max():.0f}]")
    print(f"  cross-wall candidates: {int(cross.sum()):5d}   "
          f"distance median {np.median(dist[cross]):6.1f} "
          f"[{dist[cross].min():.0f}-{dist[cross].max():.0f}]")

    # ---- 2. availability table -------------------------------------------
    print("\navailability by distance band (matching needs BOTH columns "
          "non-empty):")
    edges = np.linspace(dist.min(), dist.max(), 9)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (dist >= lo) & (dist < hi if i < len(edges) - 2 else dist <= hi)
        ns, nc = int((sel & ~cross).sum()), int((sel & cross).sum())
        flag = "" if (ns and nc) else "   <- cannot match here"
        print(f"  {lo:6.0f}-{hi:6.0f} : same-room {ns:5d}   "
              f"cross-wall {nc:5d}{flag}")

    # ---- 3. one-to-one caliper matching ----------------------------------
    ci = np.where(cross)[0]
    si = np.where(~cross)[0]
    rng.shuffle(ci)
    order = np.argsort(dist[si])
    s_sorted = si[order]
    s_dist = dist[s_sorted]
    used = np.zeros(len(s_sorted), dtype=bool)

    pairs = []
    for c in ci:
        if len(pairs) >= args.pairs:
            break
        j = bisect.bisect_left(list(s_dist), dist[c])
        best, bestd = -1, np.inf
        for k in range(max(0, j - 60), min(len(s_sorted), j + 60)):
            if used[k]:
                continue
            dd = abs(s_dist[k] - dist[c])
            if dd < bestd:
                best, bestd = k, dd
        if best >= 0 and bestd <= args.caliper:
            used[best] = True
            pairs.append((int(cand[c]), int(cand[s_sorted[best]]),
                          float(dist[c]), float(s_dist[best])))

    n = len(pairs)
    print(f"\nmatched pairs built: {n} (target {args.pairs}, "
          f"caliper {args.caliper:g} units)")
    if n:
        gaps = [abs(a - b) for _, _, a, b in pairs]
        cd = [a for _, _, a, _ in pairs]
        sd = [b for _, _, _, b in pairs]
        print(f"  within-pair distance gap: median {np.median(gaps):.2f}, "
              f"max {max(gaps):.2f}")
        print(f"  cross-wall goal distance : median {np.median(cd):6.1f}")
        print(f"  same-room  goal distance : median {np.median(sd):6.1f}   "
              f"<- matched by construction")

    # ---- 4. power gate ----------------------------------------------------
    print("\n" + "=" * 70)
    print("POWER GATE")
    print("=" * 70)
    if n == 0:
        print("  no pairs -- NO-GO.")
        return
    pw = power_sim(n, args.p_same, args.p_cross, rng)
    mins = 2 * n * SECONDS_PER_EPISODE / 60
    print(f"  assumed rates: same-room {args.p_same:.2f}, "
          f"cross-wall {args.p_cross:.2f} "
          f"({abs(args.p_same-args.p_cross)*100:.0f}-point difference)")
    print(f"  simulated power at {n} pairs: {pw:.0%} "
          f"(target {args.target_power:.0%})")
    print(f"  cost: {2*n} episodes, about {mins:.0f} min of CEM "
          f"(+ seconds for the random control)")
    if pw >= args.target_power:
        print("\n  GO. This set can detect a difference of the observed size.")
        print("  A null result would then rule out effects that large -- but")
        print("  NOT smaller ones; state that limit in the paper.")
    else:
        need = int(np.ceil(n * (args.target_power / max(pw, 0.01))))
        print(f"\n  NO-GO at this size. Re-run with --pairs {need} (roughly), "
              f"or widen --caliper if the availability table shows the "
              f"shortfall is matching, not data.")
        print("  Running underpowered would produce another inconclusive "
              "result and burn the question.")

    # ---- 5. write the auditable set --------------------------------------
    eps = []
    for k, (ce, se, cdist, sdist) in enumerate(pairs):
        eps.append({"episode": ce, "geometry": "cross", "pair_id": k,
                    "sel_dist": round(cdist, 2)})
        eps.append({"episode": se, "geometry": "same", "pair_id": k,
                    "sel_dist": round(sdist, 2)})
    out = {"goal_offset": goff, "seed": args.seed, "caliper": args.caliper,
           "n_pairs": n, "power": round(pw, 3),
           "assumed_rates": [args.p_same, args.p_cross],
           "selection": "one-to-one caliper matching on start-goal distance; "
                        "blind to model and outcome",
           "episodes": eps}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out} ({len(eps)} episodes). Commit this file BEFORE "
          f"running the evaluation -- it is the record that selection "
          f"preceded results.")


if __name__ == "__main__":
    main()
