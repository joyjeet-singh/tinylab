"""
patch_r2_planner_action_dim.py -- stop the planner searching a 10-dimensional
action space the environment cannot execute.

THE ERROR
---------
    RuntimeError: Sizes of tensors must match except in dimension 1.
    Expected size 2 but got size 10 for tensor number 1 in the list.
    (toy_plan.py:100, act = torch.cat([act, cand[:, t:t+1]], dim=1))

The planner takes its action width from the manifest's `model.action_dim`,
which for a corrected-pipeline run is **10** -- because that is the width of the
model's *dense* action input, five raw actions concatenated. But:

  * the environment accepts a 2-D action,
  * the action history the evaluator carries is 2-D,
  * and the adapter's job is precisely to widen the planner's 2-D action into
    the model's 10-D dense encoding at the moment it is fed in.

So CEM was sampling 10-wide candidates that the environment cannot execute and
that the adapter would then widen a second time. The concatenation against the
2-D history is what caught it.

THE FIX
-------
When the adapter is active, pin the planner's action space to 2. The adapter
handles the rest. This is the same correction already made for the authors'
checkpoint (`patch_r2_authors_fix.py` pins `m = {"action_dim": 2}`); it simply
was not carried over to our own corrected-pipeline checkpoints.

Nothing changes for a run without the adapter, so every committed result
reproduces.

Usage (from the tinylab folder, after patch_r2_dense_adapter.py):
    python3 patch_r2_planner_action_dim.py
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

TARGET = Path("realenv_r2_planner_eval.py")
MARKER = "planner action space pinned"

OLD = '''            model = _wrapped'''

NEW = '''            model = _wrapped
            # The planner must search the ENVIRONMENT's action space, which is
            # 2-D. The manifest's action_dim of 10 is the width of the model's
            # DENSE input (frameskip x 2), and the adapter produces that from a
            # 2-D action. Leaving it at 10 makes CEM sample candidates the
            # environment cannot execute.
            m = {**m, "action_dim": 2}
            say("  planner action space pinned to 2-D (the environment's); "
                "the adapter widens each action to the model's 10-D input")'''


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()
    if MARKER in src:
        print(f"already applied -- no change made.")
        return
    if "wrap_if_needed" not in src:
        sys.exit("run patch_r2_dense_adapter.py first; this builds on it.")
    c = src.count(OLD)
    if c != 1:
        sys.exit(f"ABORT: target appears {c} times (need exactly 1). "
                 f"Nothing written.")
    out = src.replace(OLD, NEW)
    TARGET.with_suffix(".py.bak_actdim").write_text(src)
    TARGET.write_text(out)
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except Exception as ex:
        TARGET.write_text(src)
        sys.exit(f"REVERTED -- did not compile: {ex}")
    print(f"OK: patched and compiles. Backup at {TARGET}.bak_actdim")
    print("\nThe banner will now print both ADAPTER ACTIVE and the pinned")
    print("action space. If you do not see both, stop and tell me.")


if __name__ == "__main__":
    main()
