"""
patch_toy_plan_diagnostic.py -- fix the last known-incorrect string in the repo.

WHAT IS WRONG
-------------
`toy_plan.py`'s DIAGNOSTIC block tells the reader that a high probe R² plus a
low success rate means the SCORING is at fault -- "straight-line distance
between summaries is a poor measure of real distance" -- and calls that "the
published explanation for LeWM's 87% on TwoRoom."

Three things this project has since measured say otherwise:

1. **It is invalid under domain shift.** The probe was 0.9916 while every
   planner evaluation ran in the 32-pixel fixture, 25x off-manifold. High probe
   and low success meant the EVALUATION WORLD was wrong, not the scoring. The
   text confidently pointed away from the actual bug for three paid runs.

2. **The scoring explanation is not supported in-domain.** Rooms are separated
   by 1.79x at matched distance, and the strong form of the Euclidean critique
   fails at the representation level.

3. **A high probe does not imply a usable model.** Run 0's encoder decodes
   position at R² 0.9977 and the action from a latent pair at 0.9207, yet its
   predictor never converged; phase2's predictor never beat a frozen-world
   baseline under any input. The bottleneck was the predictor, which this
   diagnostic never mentions.

THE REPLACEMENT
---------------
The new text lists what a high probe with poor planning actually implicates, in
the order this project found them, and refuses to name a cause it has not
checked. Same structure, same trigger, no false confidence.

Usage (from the tinylab folder):
    python3 patch_toy_plan_diagnostic.py
"""
from __future__ import annotations

import difflib
import py_compile
import sys
from pathlib import Path

TARGET = Path("toy_plan.py")
MARKER = "narrow it down before naming a cause"

OLD = '''        else:
            print("    -> HIGH. The encoder sees fine. If planning still fails, the")
            print("       problem is the SCORING: straight-line distance between")
            print("       summaries is a poor measure of real distance. That is the")
            print("       published explanation for LeWM's 87% on TwoRoom.")'''

NEW = '''        else:
            print("    -> HIGH. The encoder sees fine. If planning still fails,")
            print("       narrow it down before naming a cause -- in the order")
            print("       this project actually found them:")
            print("       1. THE EVALUATION WORLD. Is the domain guard passing?")
            print("          A probe of 0.99 stayed high while every evaluation")
            print("          ran 25x off-manifold in a look-alike fixture, and it")
            print("          cost three paid runs.")
            print("       2. THE ACTION CONVENTION. Does the model receive actions")
            print("          in the form it was trained on? A mis-scaled action")
            print("          produced a confident 46% that was pure artifact.")
            print("       3. THE PREDICTOR. Does one-step imagined error beat a")
            print("          frozen-world baseline? A good encoder does not imply")
            print("          a usable world model -- one of our checkpoints scored")
            print("          0.9977 on this probe and never beat standing still.")
            print("       4. ONLY THEN the scoring. Note that the strong form of")
            print("          the Euclidean critique is NOT supported in-domain for")
            print("          our encoder (rooms separate by 1.79x at matched")
            print("          distance), so do not assume it.")'''


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()
    if MARKER in src:
        print(f"already applied -- {TARGET} no longer asserts the scoring "
              f"explanation. No change made.")
        return
    cnt = src.count(OLD)
    if cnt != 1:
        sys.exit(f"ABORT: the diagnostic else-branch appears {cnt} times "
                 f"(need exactly 1). Nothing written.")
    out = src.replace(OLD, NEW)
    TARGET.with_suffix(".py.bak").write_text(src)
    TARGET.write_text(out)
    print("--- diff applied ---")
    for line in difflib.unified_diff(src.splitlines(), out.splitlines(),
                                     "before", "after", lineterm="", n=1):
        print(line)
    py_compile.compile(str(TARGET), doraise=True)
    print(f"\nOK: patched and byte-compiles. Backup at {TARGET}.bak "
          f"(gitignored).")


if __name__ == "__main__":
    main()
