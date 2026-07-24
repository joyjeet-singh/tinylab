"""
analyze_realenv_by_wall.py -- read the R2 planner result by geometry: does the
planner's failure mode live at the wall?

WHY
---
R2 gave CEM 36/50 vs random 9/50 (exact McNemar on the shared episodes:
32 CEM-only vs 5 random-only successes, p ~ 7e-6). But 10 of CEM's 14 misses
ENDED FARTHER from the goal than they started -- confident travel in a wrong
direction. The in-domain wall test showed the encoder separates the rooms by
only 1.79x; if that separation is not always enough to steer CEM, the misses
should concentrate on episodes whose goal lies ACROSS the wall. This script
classifies every episode (start/goal same room vs opposite rooms, from the
recorded positions) and breaks both CEM and random results down by that
geometry. It also draws the arena: every episode as a start->goal arrow,
green = reached, red = missed, wall and door drawn where the environment
defines them.

Run from the tinylab folder AFTER realenv_r2_planner_eval.py (both modes):
    python3 analyze_realenv_by_wall.py
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

WALL_X = 112.0            # the registered env's constant (validated vs data)
DOOR_LO, DOOR_HI = 35.0, 63.0   # door opening: center 49, half-extent 14


def load(tag):
    return json.loads(Path(f"realenv_plan_{tag}.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="~/Downloads/tworoom.h5")
    p.add_argument("--goal-offset", type=int, default=25)
    args = p.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cem, rnd = load("cem"), load("random")
    by_ep = {r["episode"]: {"cem": r} for r in cem["results"]}
    for r in rnd["results"]:
        by_ep.setdefault(r["episode"], {})["random"] = r
    eps = sorted(by_ep)

    with h5py.File(str(Path(args.h5).expanduser()), "r") as f:
        off = np.asarray(f["ep_offset"][:])
        for e in eps:
            s = np.asarray(f["pos_agent"][off[e]], dtype=np.float32)
            g = np.asarray(f["pos_agent"][off[e] + args.goal_offset],
                           dtype=np.float32)
            by_ep[e]["start"], by_ep[e]["goal"] = s, g
            by_ep[e]["cross"] = (s[0] - WALL_X) * (g[0] - WALL_X) < 0

    def table(tag):
        rows = {}
        for grp, sel in (("same-room", lambda d: not d["cross"]),
                         ("cross-wall", lambda d: d["cross"])):
            sub = [by_ep[e] for e in eps
                   if sel(by_ep[e]) and not by_ep[e][tag]["trivial_start"]]
            n = len(sub)
            s = sum(d[tag]["success"] for d in sub)
            rows[grp] = (s, n)
        return rows

    print("=" * 66)
    print("R2 RESULT BY GEOMETRY (non-trivial episodes; wall x = 112)")
    print("=" * 66)
    t_cem, t_rnd = table("cem"), table("random")
    print(f"{'group':<12} {'CEM':>14} {'random':>14}")
    for grp in ("same-room", "cross-wall"):
        c, r = t_cem[grp], t_rnd[grp]
        print(f"{grp:<12} {c[0]:>3}/{c[1]:<3} {c[0]/max(c[1],1)*100:5.1f}%"
              f"   {r[0]:>3}/{r[1]:<3} {r[0]/max(r[1],1)*100:5.1f}%")

    # exact McNemar over ALL shared episodes
    conly = sum(by_ep[e]["cem"]["success"] and not by_ep[e]["random"]["success"]
                for e in eps)
    ronly = sum(by_ep[e]["random"]["success"] and not by_ep[e]["cem"]["success"]
                for e in eps)
    n = conly + ronly
    p2 = min(1.0, 2 * sum(comb(n, k) for k in range(max(conly, ronly), n + 1))
             / 2 ** n) if n else 1.0
    print(f"\npaired discordant: CEM-only {conly}, random-only {ronly}; "
          f"exact McNemar two-sided p = {p2:.2e}")

    miss = [by_ep[e] for e in eps if not by_ep[e]["cem"]["success"]]
    run_away = [d for d in miss
                if d["cem"]["final_dist"] > d["cem"]["start_goal_dist"]]
    cross_miss = sum(d["cross"] for d in miss)
    print(f"CEM misses: {len(miss)} total; {cross_miss} are cross-wall; "
          f"{len(run_away)} ended farther from the goal than they started")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for a, tag, title in ((ax[0], "cem", "CEM"), (ax[1], "random", "random")):
        a.add_patch(plt.Rectangle((14, 14), 196, 196, fill=False, lw=1.5))
        a.plot([WALL_X, WALL_X], [14, DOOR_LO], "k-", lw=4)
        a.plot([WALL_X, WALL_X], [DOOR_HI, 210], "k-", lw=4)
        for e in eps:
            d = by_ep[e]
            col = "tab:green" if d[tag]["success"] else "tab:red"
            a.annotate("", xy=d["goal"], xytext=d["start"],
                       arrowprops=dict(arrowstyle="->", color=col,
                                       lw=1.4, alpha=0.85))
        a.set_xlim(0, 224); a.set_ylim(224, 0)
        a.set_aspect("equal"); a.set_title(f"{title}: start -> goal")
    fig.suptitle("real-environment planning, 50 shared episodes "
                 "(green = reached, red = missed)")
    fig.tight_layout()
    fig.savefig("realenv_by_wall.png", dpi=120)
    print("\nwrote realenv_by_wall.png")

    print("\nHOW TO READ")
    print("  cross-wall CEM far below same-room CEM -> the planner's residual")
    print("  failure mode is the wall: the 1.79x latent separation steers it")
    print("  most of the time but not reliably through the door -- the honest,")
    print("  quantified version of the published scoring critique.")
    print("  cross-wall ~ same-room -> misses are not wall-shaped; look at the")
    print("  red arrows in the figure for what they share instead.")


if __name__ == "__main__":
    main()
