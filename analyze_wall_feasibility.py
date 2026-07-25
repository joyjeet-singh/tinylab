"""
analyze_wall_feasibility.py -- how much of the 39-point wall effect is the
planner, and how much is the task being impossible?

THIS IS A POST-HOC ANALYSIS. It is NOT the pre-registered test and does not
revise it. The registered result stands exactly as reported:

    same-room 87/110 = 79.1%   cross-wall 44/110 = 40.0%
    discordant 53 vs 10  ->  exact two-sided p = 3.4e-08
    difference +39.1 points [95% CI +27.2, +51.0]

That answers "does the wall cost the planner anything beyond distance?" -- yes.
It does NOT answer "why", and two facts discovered after the run make the why
urgent:

  1. The wall blocks completely. Pushing straight into it from x=100 at y=150
     leaves the agent stuck at x=99.5 forever. So a cross-wall goal cannot be
     reached by steering at it; it requires routing through the door.
  2. The door sits high in the arena (opening y 35-63, agent centre must be
     within roughly 42-56 to pass) while the arena spans y 14-208. An episode
     whose start and goal are both low must travel UP to the door, across, and
     back DOWN. That detour can be two or three times the straight-line
     distance.
  3. The budget is 50 environment steps, but the goal is where the data-
     collection policy stood 100 steps later. Same-room goals are reachable in
     far fewer steps than the policy took, so 50 is generous. Cross-wall goals
     may need more than 50 no matter how good the planner is.

If a large share of cross-wall episodes are geometrically unreachable within
the budget, then part of the 39-point gap is the task, not the world model,
and the paper must say which part.

WHAT THIS COMPUTES
------------------
  A. Geometric minimum steps per episode. Movement is action x 5 with each
     action component clipped to +/-1, so travelling along a unit direction
     (u,v) covers 5/max(|u|,|v|) per step -- diagonal is faster than axis-
     aligned. Same-room: one straight segment. Cross-wall: the best two-segment
     path through the door aperture, minimised over the aperture point.
  B. An empirical DOOR-ROUTING ORACLE run in the real environment with true
     positions and no model at all: head for the aperture, cross, then head for
     the goal. This is the achievable ceiling under the actual dynamics and
     collision handling, and it also settles a second worry -- that same-room
     long goals are mostly vertical while cross-wall goals are mostly
     horizontal, so any left/right asymmetry would masquerade as a wall effect.
  C. The primary comparison restricted to pairs where BOTH episodes are
     reachable, plus each arm's score as a fraction of its own ceiling.

Usage:
    python3 analyze_wall_feasibility.py --episodes ../balanced_episodes.json \
        --cem realenv_plan_cem.json
"""
from __future__ import annotations

import argparse
import json
import os
from math import comb
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

WALL_X = 112.0
WALL_HALF = 5.0          # WALL_WIDTH_DEFAULT 10 -> half-thickness
AGENT_R = 7.0
DOOR_LO, DOOR_HI = 35.0, 63.0
SPEED = 5.0


def steps_for(p, q):
    """Minimum steps to travel p->q in a straight line at maximum action."""
    d = np.asarray(q, float) - np.asarray(p, float)
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return 0.0
    u = np.abs(d) / L
    return L * float(u.max()) / SPEED


def min_steps(start, goal, cross):
    """Geometric lower bound on steps, routing through the door if needed."""
    if not cross:
        return steps_for(start, goal), None
    lo, hi = DOOR_LO + AGENT_R, DOOR_HI - AGENT_R
    best, best_y = np.inf, None
    for y in np.linspace(lo, hi, 29):
        w = np.array([WALL_X, y])
        t = steps_for(start, w) + steps_for(w, goal)
        if t < best:
            best, best_y = t, float(y)
    return float(best), best_y


def oracle_episode(u, start, goal, cross, aperture_y, budget):
    """Door-routing oracle: true positions, no model. Returns (ok, steps)."""
    u._set_state(np.asarray(start, dtype=np.float32))
    u._set_goal_state(np.asarray(goal, dtype=np.float32))
    side = np.sign(start[0] - WALL_X)
    clear = WALL_HALF + AGENT_R + 2.0
    waypoints = []
    if cross:
        y = aperture_y if aperture_y is not None else 49.0
        waypoints = [np.array([WALL_X - side * 0.0 - side * clear, y]),
                     np.array([WALL_X + side * -1 * clear, y])]
    waypoints.append(np.asarray(goal, float))

    steps = 0
    wi = 0
    while steps < budget:
        pos = np.asarray(u._get_info()["proprio"], dtype=float)
        tgt = waypoints[min(wi, len(waypoints) - 1)]
        d = tgt - pos
        if np.linalg.norm(d) < 4.0 and wi < len(waypoints) - 1:
            wi += 1
            continue
        m = max(abs(d[0]), abs(d[1]), 1e-9)
        a = np.clip(d / m, -1.0, 1.0).astype(np.float32)
        _, _, term, _, info = u.step(a)
        steps += 1
        if term:
            return True, steps
        # advance past the door once the wall is behind us
        if cross and wi < len(waypoints) - 1:
            px = float(np.asarray(info["proprio"])[0])
            if np.sign(px - WALL_X) == -side:
                wi = len(waypoints) - 1
    return False, steps


def exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    hi = max(b, c)
    return min(1.0, 2 * sum(comb(n, k) for k in range(hi, n + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="balanced_episodes.json")
    ap.add_argument("--cem", default="realenv_plan_cem.json")
    ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
    ap.add_argument("--budget", type=int, default=50)
    ap.add_argument("--no-oracle", action="store_true")
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401

    spec = json.loads(Path(args.episodes).read_text())
    goff = int(spec["goal_offset"])
    cem = {r["episode"]: r for r in
           json.loads(Path(args.cem).read_text())["results"]}

    print("=" * 72)
    print("POST-HOC: is the wall effect the planner, or the task?")
    print("(the pre-registered result is unchanged by anything below)")
    print("=" * 72)

    rows = []
    with h5py.File(str(Path(args.h5).expanduser()), "r") as f:
        off = np.asarray(f["ep_offset"][:])
        for r in spec["episodes"]:
            e = r["episode"]
            s = np.asarray(f["pos_agent"][off[e]], dtype=float)
            g = np.asarray(f["pos_agent"][off[e] + goff], dtype=float)
            cross = r["geometry"] == "cross"
            ms, ay = min_steps(s, g, cross)
            rows.append({"e": e, "pair": r["pair_id"], "cross": cross,
                         "start": s, "goal": g, "dist": float(np.linalg.norm(g - s)),
                         "min_steps": ms, "aperture_y": ay,
                         "success": bool(cem[e]["success"]) if e in cem else None,
                         "vertical": abs(g[1] - s[1]) > abs(g[0] - s[0])})

    # ---- A: geometric reachability ---------------------------------------
    print("\nA. GEOMETRIC MINIMUM STEPS (budget "
          f"{args.budget})")
    for nm, sel in (("same-room", lambda r: not r["cross"]),
                    ("cross-wall", lambda r: r["cross"])):
        G = [r for r in rows if sel(r)]
        ms = np.array([r["min_steps"] for r in G])
        feas = ms <= args.budget
        print(f"  {nm:<11} n={len(G):3d}  straight distance median "
              f"{np.median([r['dist'] for r in G]):6.1f}")
        print(f"              minimum steps needed: median {np.median(ms):5.1f}"
              f"  90th pct {np.percentile(ms, 90):5.1f}  max {ms.max():5.1f}")
        print(f"              reachable within budget: {int(feas.sum())}/"
              f"{len(G)} = {feas.mean()*100:.1f}%")
        print(f"              goals mostly vertical: "
              f"{sum(r['vertical'] for r in G)}/{len(G)}")

    # ---- B: oracle --------------------------------------------------------
    if not args.no_oracle:
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        env = gym.make("swm/TwoRoom-v1", disable_env_checker=True)
        u = env.unwrapped
        env.reset(seed=0)
        print("\nB. DOOR-ROUTING ORACLE (true positions, no world model) -- "
              "the achievable ceiling")
        for r in rows:
            ok, st = oracle_episode(u, r["start"], r["goal"], r["cross"],
                                    r["aperture_y"], args.budget)
            r["oracle"] = ok
            r["oracle_steps"] = st
        for nm, sel in (("same-room", lambda r: not r["cross"]),
                        ("cross-wall", lambda r: r["cross"])):
            G = [r for r in rows if sel(r)]
            o = sum(r["oracle"] for r in G)
            print(f"  {nm:<11} oracle reaches {o}/{len(G)} = "
                  f"{o/len(G)*100:5.1f}%")
        print("  READ: if the oracle also drops sharply on cross-wall, the")
        print("  budget/detour is doing the work and the planner's gap is")
        print("  partly the task. If the oracle is near 100% on both, the")
        print("  gap is the planner's -- and the direction worry is dead too.")

    key = "oracle" if not args.no_oracle else None

    # ---- C: ceiling-relative + restricted test ---------------------------
    print("\nC. THE COMPARISON AGAINST WHAT IS ACHIEVABLE")
    for nm, sel in (("same-room", lambda r: not r["cross"]),
                    ("cross-wall", lambda r: r["cross"])):
        G = [r for r in rows if sel(r)]
        reach = [r for r in G if (r[key] if key else r["min_steps"] <= args.budget)]
        s_all = sum(bool(r["success"]) for r in G)
        s_reach = sum(bool(r["success"]) for r in reach)
        print(f"  {nm:<11} CEM {s_all}/{len(G)} = {s_all/len(G)*100:5.1f}% overall"
              f"   |   among reachable: {s_reach}/{len(reach)} = "
              f"{s_reach/max(len(reach),1)*100:5.1f}%")

    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair"], {})["cross" if r["cross"] else "same"] = r
    both = [v for v in pairs.values()
            if len(v) == 2 and all((x[key] if key else x["min_steps"] <= args.budget)
                                   for x in v.values())]
    b = sum(1 for v in both if v["same"]["success"] and not v["cross"]["success"])
    c = sum(1 for v in both if v["cross"]["success"] and not v["same"]["success"])
    ss = sum(v["same"]["success"] for v in both)
    cc = sum(v["cross"]["success"] for v in both)
    print(f"\n  matched pairs where BOTH sides are reachable: {len(both)} "
          f"of {len(pairs)}")
    if both:
        print(f"    same-room {ss}/{len(both)} = {ss/len(both)*100:.1f}%   "
              f"cross-wall {cc}/{len(both)} = {cc/len(both)*100:.1f}%   "
              f"difference {(ss-cc)/len(both)*100:+.1f} points")
        print(f"    discordant {b} vs {c} -> exact two-sided p = "
              f"{exact_p(b, c):.3e}")
        print("\n  READ: if this difference stays near +39 points, the wall")
        print("  effect is the planner's and the registered claim needs no")
        print("  qualification. If it shrinks a lot, the paper reports both")
        print("  numbers and attributes the remainder to reachability.")


if __name__ == "__main__":
    main()
