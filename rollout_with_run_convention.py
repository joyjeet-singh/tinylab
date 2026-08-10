"""Run eval_rollout_horizon.py under a run's OWN loader convention.

`tworoom_data` carries the loader convention as module-level constants, and
they currently sit at phase2's settings (dense actions, ImageNet pixels,
z-scored actions). Run 0 and Run 2 were trained with all three off, so calling
the rollout evaluation on them as-is feeds a 10-wide action array to a 2-wide
action encoder and dies on a shape mismatch.

This sets the three constants from the run's own manifest before the loader is
constructed -- the same thing verify_phase2_driving.py does automatically, and
the same rule: a checkpoint is driven the way its manifest says it was trained.
It changes no tracked source file and no measurement code.

eval_rollout_horizon.py's own note says horizon 1's `err/step` is the
scale-free ratio "Check A" reported, so that cell is the one-step prediction
error the paper's Table 3 quotes.

Usage:
    ./.venv/bin/python rollout_with_run_convention.py --run <dir> --ckpt <name>
"""
import json
import sys
from pathlib import Path

import tworoom_data

FLAGS = ("dense_actions", "imagenet_pixels", "zscore_actions")
CONST = {"dense_actions": "DENSE_ACTIONS",
         "imagenet_pixels": "IMAGENET_PIXELS",
         "zscore_actions": "ZSCORE_ACTIONS"}

run = Path(sys.argv[sys.argv.index("--run") + 1])
manifest = json.loads((run / "manifest.json").read_text())
conv = manifest.get("loader_convention")
if conv is None:
    conv = {k: False for k in FLAGS}
    where = "absent from the manifest -- assumed all off, as the run predates the flags"
else:
    where = "read from the manifest"

print("=" * 70)
print(f"LOADER CONVENTION for {run.name}")
print("=" * 70)
print(f"  {where}")
for k in FLAGS:
    v = bool(conv.get(k, False))
    setattr(tworoom_data, CONST[k], v)
    print(f"  {CONST[k]:<18} {v}")
    assert getattr(tworoom_data, CONST[k]) is v, f"failed to set {CONST[k]}"
print()

import importlib  # noqa: E402  (must follow the flag assignment)

# --script picks which measurement to drive; it is consumed here rather than
# passed on to that measurement's own argument parser.
target = "eval_rollout_horizon"
if "--script" in sys.argv:
    i = sys.argv.index("--script")
    target = sys.argv[i + 1]
    del sys.argv[i:i + 2]

sys.argv[0] = f"{target}.py"
print(f"driving {target}.py under the convention above\n")
mod = importlib.import_module(target)
mod.main()
