"""
patch_verify_displacement_matched.py -- fix a flaw in the driving-spec test.

THE FLAW
--------
verify_phase2_driving.py builds its "repeat" encoding from the FIRST raw action
of each block, tiled frameskip times. For a transition actually caused by the
block [a1..a5] with sum S, that encoding implies a summed action of 5*a1, which
differs from S by 2.17 units on average against a true spread of 1.29 -- more
than the entire signal.

So the test asked: "can the model predict this transition when handed action
information that is wrong for it?" The answer, unsurprisingly, was no, and the
script reported STOP.

That is not the planning question. A planner emits one action `a` and the
environment holds it for frameskip steps, producing summed action 5a. The block
it corresponds to is one whose MEAN is `a` -- and repeating the mean gives
exactly the right sum.

THE FIX
-------
Use the block mean, tiled, as the primary "constant-action planner" encoding.
It is displacement-matched by construction, so the model is asked to predict a
transition using action information consistent with that transition, differing
from the true dense input only in that the within-block variation has been
flattened.

Both encodings are now reported. The difference between them is itself
informative: it separates "the model cannot handle constant actions" from "the
model was handed the wrong action".

NOTE: `dense_action_adapter.py` was always correct -- it repeats the planner's
action, which is right. Only this test was wrong.

Usage (from the tinylab folder):
    python3 patch_verify_displacement_matched.py
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

TARGET = Path("verify_phase2_driving.py")
MARKER = "displacement-matched"

EDITS = [
    ("collect the block mean",
     '''        frames, dense, first = [], [], []''',
     '''        frames, dense, first, meanb = [], [], [], []'''),

    ("store it",
     '''            first.append(np.stack([b[0] for b in blocks]))''',
     '''            first.append(np.stack([b[0] for b in blocks]))
            meanb.append(np.stack([b.mean(0) for b in blocks]))'''),

    ("stack it",
     '''    dense = np.stack(dense)
    first = np.stack(first)''',
     '''    dense = np.stack(dense)
    first = np.stack(first)
    meanb = np.stack(meanb)'''),

    ("build both encodings",
     '''    rep_raw = np.tile(first, (1, 1, A // first.shape[-1]))''',
     '''    # A planner emits one action `a` held for frameskip steps, so its block
     # sums to 5a -- which matches a data block whose MEAN is a. Repeating the
     # mean is therefore displacement-matched to the transition being
     # predicted; repeating the FIRST action is not, and differs from the true
     # summed action by more than the whole signal.
    rep_mean = np.tile(meanb, (1, 1, A // meanb.shape[-1]))
    rep_raw = np.tile(first, (1, 1, A // first.shape[-1]))'''),

    ("report both, mean first",
     '''    variants = {
        "repeat + z-score (the adapter)": zscore(rep_raw),
        "repeat, no z-score": rep_raw,''',
     '''    variants = {
        "repeat of MEAN (displacement-matched)": zscore(rep_mean),
        "repeat of mean, no z-score": rep_mean,
        "repeat of FIRST (not matched)": zscore(rep_raw),'''),

    ("verdict reads the matched one",
     '''    adp = res["repeat + z-score (the adapter)"]''',
     '''    adp = res["repeat of MEAN (displacement-matched)"]'''),
]


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()
    if MARKER in src:
        print(f"already applied -- {TARGET} tests the displacement-matched "
              f"encoding. No change made.")
        return
    out, bad = src, []
    for name, old, new in EDITS:
        c = out.count(old)
        if c != 1:
            bad.append(f"{name} appears {c}x")
            continue
        out = out.replace(old, new)
    if bad:
        sys.exit("ABORT: targets did not match: " + "; ".join(bad) +
                 "\nNothing written.")
    TARGET.with_suffix(".py.bak_dm").write_text(src)
    TARGET.write_text(out)
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except Exception as ex:
        TARGET.write_text(src)
        sys.exit(f"REVERTED -- patched file did not compile: {ex}")
    print(f"OK: {TARGET} patched and compiles. Backup at {TARGET}.bak_dm")
    print("\nRe-run on the recalibrated checkpoint. The table will now show")
    print("both encodings; the MEAN one is the planning question, the FIRST")
    print("one is what the previous (flawed) run measured.")


if __name__ == "__main__":
    main()
