"""
patch_r2_goal_target.py -- evaluate the task the environment actually defines.

WHAT THE DATA SHOWED
--------------------
Episode lengths cap at exactly 101. Episodes that end earlier finish a mean of
13.6 units from the target — inside the environment's own 16-unit success
radius, so they terminated on success. Episodes at the 101 cap finish 73.1 units
away: they timed out. The cap is a truncation, not a length.

At goal offset 100 only episodes longer than 100 qualify, so our long-horizon
evaluation has been drawn **exclusively from the data policy's failures**, with
the goal set to the state at which that policy ran out of time — 73 units from
the target it was pursuing. That is not the environment's task, and the
reference's reported figure cannot be measured that way by anyone.

WHAT THIS ADDS
--------------
`--goal-target`      the goal is the episode's recorded target position rather
                     than a state N frames later. This is the environment's
                     native task and matches its own success criterion.

`--successful-only`  restrict to episodes in which the data policy reached the
                     target (terminated rather than timed out). With
                     --goal-target this asks: can the planner do what the data
                     policy did?

Both default off, so every committed result reproduces. `--goal-target` ignores
`--goal-offset`, and the report records which mode ran.

Usage:
    python3 patch_r2_goal_target.py
"""
from __future__ import annotations

import py_compile
import shutil
import sys
from pathlib import Path

TARGET = Path("realenv_r2_planner_eval.py")
MARKER = "--goal-target"

EDITS = [
    ("the flags",
     '''    p.add_argument("--ckpt", default=None,''',
     '''    p.add_argument("--goal-target", action="store_true",
                   help="use the episode's recorded target position as the "
                        "goal, rather than a state --goal-offset frames later; "
                        "this is the environment's own task")
    p.add_argument("--successful-only", action="store_true",
                   help="restrict to episodes in which the data policy reached "
                        "the target (terminated rather than timed out)")
    p.add_argument("--ckpt", default=None,'''),

    ("episode eligibility",
     '''        if meta is None:
            ok = np.where(ln > args.goal_offset)[0]''',
     '''        if meta is None:
            if args.goal_target:
                ok = np.arange(len(ln))     # every episode has a target
            else:
                ok = np.where(ln > args.goal_offset)[0]
            if args.successful_only:
                term = np.asarray(f["terminated"][:])
                reached = np.array([bool(term[int(o) + int(l) - 1])
                                    for o, l in zip(off, ln)])
                ok = np.array([e for e in ok if reached[e]])
                if len(ok) == 0:
                    raise SystemExit("no episodes satisfy --successful-only")'''),

    ("the goal",
     '''            goals.append(np.asarray(f["pos_agent"][s0 + args.goal_offset],
                                    dtype=np.float32))''',
     '''            goals.append(np.asarray(
                f["pos_target"][s0] if args.goal_target
                else f["pos_agent"][s0 + args.goal_offset],
                dtype=np.float32))'''),

    ("the report banner",
     '''    say(f"\\nepisodes: {len(eps)} sampled (seed {args.seed}); start = episode "
        f"frame 0; goal = frame {args.goal_offset} of the same episode")''',
     '''    say(f"\\nepisodes: {len(eps)} sampled (seed {args.seed})"
        f"{' [policy-successful only]' if args.successful_only else ''}; "
        f"start = episode frame 0; goal = "
        + ("the episode's recorded TARGET position"
           if args.goal_target
           else f"frame {args.goal_offset} of the same episode"))'''),
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
        print("  The goal edit assumes the random-start patch is applied (it")
        print("  introduced `s0`). Apply that first, or paste the block.")
        return

    shutil.copy(TARGET, TARGET.with_suffix(".py.bak_gt"))
    TARGET.write_text(out)
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except Exception as ex:
        TARGET.write_text(src)
        sys.exit(f"REVERTED — did not compile: {ex}")
    print(f"\nOK: patched and compiles. Backup at {TARGET.name}.bak_gt")
    print("\nVERIFY FIRST — three episodes, ten steps, one minute:")
    print("  the banner must read \"goal = the episode's recorded TARGET\"")
    print("  and the start-goal distances must differ from the offset run.")
    print("  If they do not, the flag is not reaching the goal computation.")


if __name__ == "__main__":
    main()
