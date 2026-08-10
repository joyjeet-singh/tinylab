"""
patch_r2_random_start.py -- sample the start state the way the paper says it
does.

THE DEVIATION
-------------
App. F.1: "The initial state is chosen by **randomly sampling a state from a
trajectory** in the dataset, while the goal state corresponds to a state
occurring several timesteps later in the same trajectory."

Our evaluator always starts at frame 0. For the offset-100 episode population —
which is length-biased, since only episodes longer than 100 steps qualify —
frame 0 to frame 100 is the hardest available segment: the data policy's full
journey from its start position, through the door, toward the target. A random
start inside the same trajectory will often land mid-journey.

This is the one documented deviation we have never tested, and it is the
remaining candidate for the authors'-checkpoint discrepancy that we can
actually run.

WHAT THIS DOES
--------------
Adds `--random-start`. When passed, the start index is drawn uniformly from
[episode_start, episode_end - goal_offset - 1] instead of being fixed at the
episode's first frame. The goal remains `goal_offset` steps after the start, in
the same trajectory, as the paper describes. Default behaviour is unchanged, so
every committed result reproduces.

IF THE ANCHORS DO NOT MATCH
---------------------------
The script prints the episode-selection block with line numbers and changes
nothing. Paste that block and the patch can be retargeted exactly.

Usage:
    python3 patch_r2_random_start.py
"""
from __future__ import annotations

import py_compile
import re
import shutil
import sys
from pathlib import Path

TARGET = Path("realenv_r2_planner_eval.py")
MARKER = "--random-start"

ARGPARSE = (
    '''    p.add_argument("--ckpt", default=None,''',
    '''    p.add_argument("--random-start", action="store_true",
                   help="draw the start state uniformly from the trajectory "
                        "instead of using its first frame, as App. F.1 of the "
                        "reference describes")
    p.add_argument("--ckpt", default=None,''')

# plausible shapes of the start computation, most specific first
START_PATTERNS = [
    (r"^(\s*)s = int\(off\[ep\]\)\s*$",
     '''\\1s = int(off[ep])
\\1if args.random_start:
\\1    _span = int(ln[ep]) - args.goal_offset - 1
\\1    if _span > 0:
\\1        s += int(rng.integers(0, _span))'''),
    (r"^(\s*)start = int\(off\[ep\]\)\s*$",
     '''\\1start = int(off[ep])
\\1if args.random_start:
\\1    _span = int(ln[ep]) - args.goal_offset - 1
\\1    if _span > 0:
\\1        start += int(rng.integers(0, _span))'''),
    (r"^(\s*)starts = off\[eps\]\s*$",
     '''\\1starts = off[eps]
\\1if args.random_start:
\\1    starts = np.array([int(o) + int(rng.integers(
\\1        0, max(1, int(l) - args.goal_offset - 1)))
\\1        for o, l in zip(off[eps], ln[eps])])'''),
]

BANNER = (
    'start = episode frame 0; goal = frame',
    'start = episode frame {"random" if args.random_start else 0}; '
    'goal = frame',
)


def show_context(src: str):
    lines = src.splitlines()
    hit = next((i for i, l in enumerate(lines)
                if "start = episode frame 0" in l), None)
    if hit is None:
        hit = next((i for i, l in enumerate(lines)
                    if re.search(r"episodes:.*sampled", l)), None)
    if hit is None:
        print("  could not locate the episode-selection block at all")
        return
    lo, hi = max(0, hit - 18), min(len(lines), hit + 4)
    print(f"\n  the block, lines {lo+1}-{hi}:\n")
    for i in range(lo, hi):
        print(f"  {i+1:>5} | {lines[i]}")


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found — run from the tinylab folder")
    src = TARGET.read_text()
    if MARKER in src:
        print("already applied — no change made.")
        return

    out = src
    ok = True

    # 1. the flag
    if out.count(ARGPARSE[0]) == 1:
        out = out.replace(*ARGPARSE)
        print("  [x] --random-start flag added")
    else:
        print(f"  [ ] argparse anchor appears {out.count(ARGPARSE[0])} times")
        ok = False

    # 2. the start computation
    done = False
    for pat, rep in START_PATTERNS:
        if len(re.findall(pat, out, re.M)) == 1:
            out = re.sub(pat, rep, out, flags=re.M)
            print(f"  [x] start computation patched "
                  f"(matched {pat.strip('^$')[:34]})")
            done = True
            break
    if not done:
        print("  [ ] could not find the start computation")
        ok = False

    # 3. the banner, so the report says which was used
    if out.count(BANNER[0]) == 1 and "f\"episodes:" in out:
        out = out.replace(*BANNER)
        print("  [x] report banner updated")
    elif out.count(BANNER[0]) == 1:
        print("  [~] banner found but the say() may not be an f-string; "
              "update it by hand so the report records which mode ran")
    else:
        print("  [ ] banner not found")

    if not ok:
        print("\nNOTHING WRITTEN. The evaluator's episode selection is not "
              "shaped as expected.")
        show_context(src)
        print("\n  Paste the block above and the patch can be retargeted.")
        return

    shutil.copy(TARGET, TARGET.with_suffix(".py.bak_rs"))
    TARGET.write_text(out)
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except Exception as ex:
        TARGET.write_text(src)
        sys.exit(f"REVERTED — did not compile: {ex}")
    print(f"\nOK: patched and compiles. Backup at {TARGET.name}.bak_rs")
    print("\nSanity-check on two episodes before the real run:")
    print("  the two runs below must select DIFFERENT start frames")
    print("    python3 realenv_r2_planner_eval.py --run <run> --goal-offset 100 \\")
    print("        --num-eval 2 --budget 10")
    print("    python3 realenv_r2_planner_eval.py --run <run> --goal-offset 100 \\")
    print("        --num-eval 2 --budget 10 --random-start")
    print("  Compare the 'start-goal' distances printed per episode. If they "
          "are\n  identical, the flag is not reaching the start computation.")


if __name__ == "__main__":
    main()
