"""
patch_r2_action_scale.py -- add --action-scale to realenv_r2_planner_eval.py so
OUR checkpoint can be driven at the scale it was actually trained to expect.

WHY
---
Our data loader stored ONE raw action per clip step, subsampled at the
frameskip stride, while the reference keeps every action and reshapes them to
(history_len, frameskip * action_dim). So our predictor learned to associate a
single one-step action with a five-step displacement -- it under-predicts the
movement a held action produces by roughly the frameskip factor.

Measured (action_scale_check.py, one-step error over the frozen-world
baseline, 120 windows):

    k        0.5     1.0     2.0     3.0     5.0     7.0
    ours    0.820   0.737   0.590   0.491   0.513   0.676

k=1 is what the evaluator has been feeding. Use k=5, not the empirical argmin
of 3: five is the semantically justified value -- the planner holds action a
for five environment steps, producing displacement 25a, and a model trained on
single actions predicts 5a, so feeding 5a makes it predict correctly. Choosing
3 because it minimises the metric would be tuning a headline number to that
metric. Report 5 as primary and 3 as a sensitivity row.

Caveat to state in the paper: 5a exceeds the range of single raw actions the
model saw during training, so the correction is partly extrapolation.

The flag wraps the model's action_encoder, so it works for any model. It
defaults to 1.0, meaning every previously committed result is reproduced
byte-for-byte unless the flag is passed. Do NOT combine it with
--authors-spec, whose adapter applies its own encoding from the driving spec.

Same construction as the earlier patches: marker check first, exact-once
assertions, backup, diff, byte-compile.

Usage (from the tinylab folder):
    python3 patch_r2_action_scale.py
"""
from __future__ import annotations

import difflib
import py_compile
import sys
from pathlib import Path

TARGET = Path("realenv_r2_planner_eval.py")
MARKER = "--action-scale"

EDITS = [
    ("argparse flag",
     '''    p.add_argument("--unsafe-skip-guard", action="store_true",''',
     '''    p.add_argument("--action-scale", type=float, default=1.0,
                   help="multiply the action before it reaches the model "
                        "(NOT the environment). Our loader subsampled one "
                        "action per clip step, so the model under-predicts "
                        "displacement by the frameskip factor; 5.0 corrects "
                        "it. Default 1.0 reproduces every committed result.")
    p.add_argument("--unsafe-skip-guard", action="store_true",'''),

    ("wrap the action encoder",
     '''    @torch.no_grad()
    def emb_of(img_hwc_uint8):''',
     '''    if args.action_scale != 1.0:
        if args.authors_spec:
            raise SystemExit(
                "--action-scale and --authors-spec both rescale actions; the "
                "authors' encoding comes from the driving spec. Pass only one.")
        class _ScaledActionEncoder(torch.nn.Module):
            """Wraps the real encoder; must be a Module to be assignable."""

            def __init__(self, inner, k):
                super().__init__()
                self.inner, self.k = inner, float(k)

            def forward(self, a):
                return self.inner(a * self.k)

        _k = float(args.action_scale)
        model.action_encoder = _ScaledActionEncoder(model.action_encoder, _k)
        say(f"ACTION SCALE {_k:g} applied to the model input only "
            f"(the environment still receives the unscaled action)")

    @torch.no_grad()
    def emb_of(img_hwc_uint8):'''),

    ("record it in the outputs",
     '''           "protocol": {"num_eval": args.num_eval, "budget": args.budget,''',
     '''           "action_scale": args.action_scale,
           "protocol": {"num_eval": args.num_eval, "budget": args.budget,'''),
]


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()

    if MARKER in src:
        print(f"'{MARKER}' already present in {TARGET}. No change made.")
        return

    out = src
    for name, old, new in EDITS:
        cnt = out.count(old)
        if cnt != 1:
            sys.exit(f"ABORT: target for '{name}' appears {cnt} times "
                     f"(need exactly 1). Nothing written.")
        out = out.replace(old, new)

    TARGET.with_suffix(".py.bak4").write_text(src)
    TARGET.write_text(out)
    print("--- diff applied ---")
    for line in difflib.unified_diff(src.splitlines(), out.splitlines(),
                                     "before", "after", lineterm="", n=1):
        print(line)
    py_compile.compile(str(TARGET), doraise=True)
    print(f"\nOK: patched and byte-compiles. Backup at {TARGET}.bak4")
    print("Default is 1.0, so existing results reproduce unchanged.")


if __name__ == "__main__":
    main()
