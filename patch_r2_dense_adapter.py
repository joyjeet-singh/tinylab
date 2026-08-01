"""
patch_r2_dense_adapter.py -- make the planner evaluator drive a
corrected-pipeline checkpoint correctly.

THE GAP THIS CLOSES
-------------------
`dense_action_adapter.wrap_if_needed()` exists, is tested, and is never called
by the evaluator. A phase2 checkpoint expects ImageNet-normalised pixels and
ten-wide z-scored actions; the evaluator supplies [0,1] pixels and two-wide raw
actions. Running it as-is would feed the model inputs it has never seen -- the
exact mistake that produced the 46% artifact with the authors' checkpoint.

WHAT THIS PATCH DOES
--------------------
After our model is built and its weights loaded, the evaluator now calls
wrap_if_needed(), which reads `loader_convention` from the run's manifest and:

  * wraps the model iff that run was trained with the new pipeline, or
  * returns it untouched otherwise.

So Run 0, Run 1 and Run 2 checkpoints are driven exactly as before -- every
committed result reproduces byte-for-byte -- while a phase2 checkpoint gets
ImageNet pixels, the planner's action repeated frameskip times, and the
dataset z-score, all read from the manifest rather than from a flag anyone has
to remember.

It also refuses to combine with --action-scale, whose rescaling would compose
with the adapter's own and silently double-transform the action.

Usage (from the tinylab folder, after patch_add_ckpt_flag.py):
    python3 patch_r2_dense_adapter.py
"""
from __future__ import annotations

import difflib
import py_compile
import sys
from pathlib import Path

TARGET = Path("realenv_r2_planner_eval.py")
MARKER = "wrap_if_needed"

OLD = '''        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()'''

NEW = '''        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        # A checkpoint trained with the corrected pipeline expects
        # ImageNet-normalised pixels and dense z-scored actions. The manifest
        # records which; wrap_if_needed() returns the model untouched when the
        # run predates the change, so older results are unaffected.
        from dense_action_adapter import wrap_if_needed
        _wrapped = wrap_if_needed(model, run_dir, args.h5,
                                  frameskip=cfg["data"]["frameskip"])
        if _wrapped is not model:
            if args.action_scale != 1.0:
                raise SystemExit(
                    "--action-scale cannot be combined with a corrected-"
                    "pipeline checkpoint: the adapter already applies the "
                    "dataset z-score, and the two rescalings would compose.")
            model = _wrapped'''


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()
    if MARKER in src:
        print(f"already applied -- {TARGET} calls the adapter. No change made.")
        return
    c = src.count(OLD)
    if c != 1:
        sys.exit(f"ABORT: the checkpoint-load block appears {c} times "
                 f"(need exactly 1). Nothing written.\n"
                 f"Send me the lines around `model.load_state_dict` and I "
                 f"will retarget.")
    out = src.replace(OLD, NEW)
    TARGET.with_suffix(".py.bak_adapter").write_text(src)
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
    print(f"\nOK: patched and compiles. Backup at {TARGET}.bak_adapter")
    print("\nThe banner will now print ADAPTER ACTIVE for a phase2 checkpoint")
    print("and nothing for the older ones. Check it before trusting a number.")


if __name__ == "__main__":
    main()
