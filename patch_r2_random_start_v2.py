"""
patch_r2_random_start_v2.py -- retargeted against the evaluator's actual
episode-selection block.

WHY v1 FAILED
-------------
I guessed the start index was computed as a scalar (`s = int(off[ep])`). It is
not: the evaluator reads the start and goal positions directly out of the
dataset inside one loop, using `off[e]` and `off[e] + args.goal_offset` as
indices. There was nothing for v1's patterns to match, so it wrote nothing and
printed the block instead. This version targets that block exactly.

WHAT IT CHANGES
---------------
App. F.1 of the reference: "The initial state is chosen by randomly sampling a
state from a trajectory in the dataset, while the goal state corresponds to a
state occurring several timesteps later in the same trajectory."

With `--random-start`, the start index becomes a uniform draw from
[episode start, episode end - goal_offset - 1] and the goal stays goal_offset
frames later in the same trajectory. Default behaviour is unchanged.

ONE GUARD
---------
`--random-start` is refused together with `--episodes`. The committed episode
sets (the matched-pair wall experiment) fix their starts as part of the
pre-registration; drawing new ones would silently invalidate the matching that
the design depends on.

Usage:
    python3 patch_r2_random_start_v2.py
"""
from __future__ import annotations

import py_compile
import shutil
import sys
from pathlib import Path

TARGET = Path("realenv_r2_planner_eval.py")
MARKER = "--random-start"

EDITS = [
    ("the flag",
     '''    p.add_argument("--ckpt", default=None,''',
     '''    p.add_argument("--random-start", action="store_true",
                   help="draw the start state uniformly from within the "
                        "trajectory rather than using its first frame, as "
                        "App. F.1 of the reference describes")
    p.add_argument("--ckpt", default=None,'''),

    ("the start and goal indices",
     '''        starts, goals = [], []
        for e in eps:
            starts.append(np.asarray(f["pos_agent"][off[e]], dtype=np.float32))
            goals.append(np.asarray(f["pos_agent"][off[e] + args.goal_offset],
                                    dtype=np.float32))''',
     '''        starts, goals = [], []
        for e in eps:
            # App. F.1: the initial state is sampled from within the
            # trajectory, not fixed at its first frame. For the offset-100
            # population -- only episodes longer than 100 steps -- frame 0 is
            # the hardest available segment, so this is not a neutral choice.
            s0 = int(off[e])
            if args.random_start:
                span = int(ln[e]) - args.goal_offset - 1
                if span > 0:
                    s0 += int(rng.integers(0, span))
            starts.append(np.asarray(f["pos_agent"][s0], dtype=np.float32))
            goals.append(np.asarray(f["pos_agent"][s0 + args.goal_offset],
                                    dtype=np.float32))'''),

    ("the report banner",
     '''    say(f"\\nepisodes: {len(eps)} sampled (seed {args.seed}); start = episode "
        f"frame 0; goal = frame {args.goal_offset} of the same episode")''',
     '''    say(f"\\nepisodes: {len(eps)} sampled (seed {args.seed}); start = "
        f"{'a uniformly random frame' if args.random_start else 'episode frame 0'}"
        f"; goal = {args.goal_offset} frames later in the same episode")'''),

    ("the guard against a committed episode set",
     '''    with h5py.File(h5_path, "r") as f:
        off = np.asarray(f["ep_offset"][:])''',
     '''    if args.random_start and getattr(args, "episodes", None):
        raise SystemExit(
            "--random-start cannot be combined with --episodes: a committed "
            "episode set fixes its starts as part of the pre-registration, and "
            "redrawing them would invalidate the matching the design rests on.")
    with h5py.File(h5_path, "r") as f:
        off = np.asarray(f["ep_offset"][:])'''),
]


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found — run from the tinylab folder")
    src = TARGET.read_text()
    if MARKER in src:
        print("already applied — no change made.")
        return

    out, bad = src, []
    for name, old, new in EDITS:
        n = out.count(old)
        if n != 1:
            bad.append(f"{name} ({n} matches)")
            continue
        out = out.replace(old, new)
        print(f"  [x] {name}")

    if bad:
        print(f"\nNOTHING WRITTEN — {len(bad)} anchor(s) did not match: "
              f"{', '.join(bad)}")
        print("  Paste the surrounding lines and I will retarget again.")
        return

    shutil.copy(TARGET, TARGET.with_suffix(".py.bak_rs"))
    TARGET.write_text(out)
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except Exception as ex:
        TARGET.write_text(src)
        sys.exit(f"REVERTED — did not compile: {ex}")
    print(f"\nOK: patched and compiles. Backup at {TARGET.name}.bak_rs")

    print("\nVERIFY BEFORE THE REAL RUN — the two must differ:")
    print("  the start-goal distances printed per episode should change when")
    print("  --random-start is passed. If they are identical, the flag is not")
    print("  reaching the index computation and nothing below is meaningful.")


if __name__ == "__main__":
    main()
