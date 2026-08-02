"""
patch_r2_deviations_text.py -- stop the reports misdescribing themselves.

THE PROBLEM
-----------
The DEVIATIONS block printed at the end of every planner report is a set of
hardcoded strings. As a result:

  * every offset-100 report states "goal = frame 25" and "goal_offset 25"
  * every run of the AUTHORS' released checkpoint states "OUR reproduction
    checkpoint, not the authors' released weights"
  * the 220-episode balanced-wall runs state "random 50 eps" and "num_eval 50"

These reports are the artifacts that go into the appendix and the repository. A
deviations table that misstates the deviations is worse than no table, and a
reader checking one report against another would find them contradicting the
episode line printed twenty lines above.

THE FIX
-------
Interpolate the values the run actually used: goal offset, budget, evaluation
count, receding horizon, whether a committed episode set was supplied, whether
the authors' checkpoint is in use, and whether an action scale was applied.

Nothing about the runs changes -- only what the report says about them. Existing
result files are unaffected; re-run any report you intend to cite.

Usage (from the tinylab folder):
    python3 patch_r2_deviations_text.py
"""
from __future__ import annotations

import difflib
import py_compile
import sys
from pathlib import Path

TARGET = Path("realenv_r2_planner_eval.py")
MARKER = "args.goal_offset}); reference"

EDITS = [
    ("episode selection line 1",
     '''    say("  - episode/start selection ours (random 50 eps seed 42, start =")''',
     '''    say("  - episode/start selection ours ("
        + (f"committed set {Path(args.episodes).name}"
           if getattr(args, "episodes", None)
           else f"random {args.num_eval} eps seed {args.seed}")
        + ", start =")'''),

    ("episode selection line 2",
     '''    say("    frame 0, goal = frame 25); reference's exact selection unknown")''',
     '''    say(f"    frame 0, goal = frame {args.goal_offset}); reference's "
        f"exact selection unknown")'''),

    ("receding horizon line",
     '''    say("  - receding_horizon 5 read as: execute 5 planned actions, replan")''',
     '''    say(f"  - receding_horizon {args.receding} read as: execute "
        f"{args.receding} planned actions, replan")'''),

    ("which checkpoint",
     '''    say("  - OUR reproduction checkpoint, not the authors' released weights")''',
     '''    say("  - the AUTHORS' released weights, driven through our harness"
        if getattr(args, "authors_spec", None)
        else "  - OUR reproduction checkpoint, not the authors' released weights")
    if getattr(args, "action_scale", 1.0) != 1.0:
        say(f"  - action scaled by {args.action_scale:g} before reaching the "
            f"model (the environment receives the unscaled action)")'''),

    ("matched line",
     '''    say("  budget 50, num_eval 50, goal_offset 25, img 224, goal-image")''',
     '''    say(f"  budget {args.budget}, goal_offset {args.goal_offset}, "
        f"img 224, goal-image")'''),
]

def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()
    if MARKER in src:
        print("already applied -- the deviations block is parameterised. "
              "No change made.")
        return

    out, done, bad = src, [], []
    for name, old, new in EDITS:
        c = out.count(old)
        if c != 1:
            bad.append(f"{name} ({c}x)")
            continue
        out = out.replace(old, new)
        done.append(name)
    if not done:
        sys.exit("ABORT: no target matched. Nothing written. Send me lines "
                 "360-380 and I will retarget.")
    if bad:
        print(f"NOTE: {len(bad)} line(s) did not match and were left alone: "
              f"{', '.join(bad)}")
        print("Each say() line is independent, so the rest are still fixed.")

    TARGET.with_suffix(".py.bak_dev").write_text(src)
    TARGET.write_text(out)
    print("--- diff applied ---")
    for line in difflib.unified_diff(src.splitlines(), out.splitlines(),
                                     "before", "after", lineterm="", n=1):
        print(line)
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except Exception as ex:
        TARGET.write_text(src)
        sys.exit(f"REVERTED -- did not compile: {ex}")
    print(f"\nOK: patched and compiles. Backup at {TARGET}.bak_dev")
    print("\nSanity-check it on a 2-episode run at a non-default offset:")
    print("  python3 realenv_r2_planner_eval.py --run <run> --goal-offset 100 \\")
    print("      --num-eval 2 --budget 10")
    print("The deviations block should now say goal_offset 100, episodes 2.")
    print("\nNOTE: the module docstring (lines ~17, ~33, ~37) also states the")
    print("defaults as though they were fixed. That is documentation rather")
    print("than output, but worth a line saying they are defaults.")


if __name__ == "__main__":
    main()
