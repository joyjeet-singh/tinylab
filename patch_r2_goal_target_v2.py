"""
patch_r2_goal_target_v2.py -- retargeted banner.

WHY v1 WROTE NOTHING
--------------------
Three of four anchors matched; the report banner did not, because the
random-start patch had already rewritten that line. The patch is all-or-nothing
by design, so it wrote nothing rather than leaving the evaluator half-changed —
which is why the flags did not exist and both subsequent commands failed at
argument parsing. That is the intended behaviour; only the anchor was stale.

WHAT IT ADDS (unchanged from v1)
--------------------------------
`--goal-target`      the goal is the episode's recorded target position rather
                     than a state N frames later. Episode lengths cap at 101
                     and capped episodes end a mean of 73.1 units from their
                     target while early-terminating ones end 13.6 units away —
                     inside the success radius. So the cap is a timeout, and
                     goal offset 100 selects the data policy's failures and
                     asks the planner to reach the point where that policy ran
                     out of time. This flag asks the environment's own question
                     instead.

`--successful-only`  restrict to episodes in which the data policy reached the
                     target. With --goal-target this asks whether the planner
                     can do what the data policy demonstrably did.

Both default off, so every committed result reproduces.

Usage:
    python3 patch_r2_goal_target_v2.py
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

    ("the report banner (as rewritten by the random-start patch)",
     '''    say(f"\\nepisodes: {len(eps)} sampled (seed {args.seed}); start = "
        f"{'a uniformly random frame' if args.random_start else 'episode frame 0'}"
        f"; goal = {args.goal_offset} frames later in the same episode")''',
     '''    say(f"\\nepisodes: {len(eps)} sampled (seed {args.seed})"
        f"{' [policy-successful only]' if args.successful_only else ''}"
        f"; start = "
        f"{'a uniformly random frame' if args.random_start else 'episode frame 0'}"
        f"; goal = "
        + ("the episode's recorded TARGET position" if args.goal_target
           else f"{args.goal_offset} frames later in the same episode"))'''),
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
        print("  Paste the say() that prints the 'episodes: ... sampled' line")
        print("  and I will retarget again.")
        return

    shutil.copy(TARGET, TARGET.with_suffix(".py.bak_gt"))
    TARGET.write_text(out)
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except Exception as ex:
        TARGET.write_text(src)
        sys.exit(f"REVERTED — did not compile: {ex}")
    print(f"\nOK: patched and compiles. Backup at {TARGET.name}.bak_gt")

    print("\nVERIFY WITHOUT PIPING THROUGH grep — a pipe hides argparse errors,")
    print("which is why the last check appeared to produce nothing at all:")
    print("  python3 realenv_r2_planner_eval.py --run <run> --ckpt <ckpt> \\")
    print("      --goal-target --successful-only --num-eval 3 --budget 10")
    print("\n  The banner must read \"[policy-successful only]\" and")
    print("  \"goal = the episode's recorded TARGET position\", and the")
    print("  start-goal distances must differ from the offset runs.")


if __name__ == "__main__":
    main()
