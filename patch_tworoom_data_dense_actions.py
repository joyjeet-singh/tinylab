"""
patch_tworoom_data_dense_actions.py -- make our clip loader keep actions DENSE,
matching the reference.

WHAT IS WRONG TODAY
-------------------
`TwoRoomClips.__getitem__` reads every key at the strided clip indices:

    idx = start + arange(num_steps) * frameskip
    out[k] = np.asarray(f[k][idx], dtype=np.float32)

That is right for pixels and wrong for actions. It stores ONE raw action per
clip step, sampled every fifth frame, while the frame at the next clip step is
five raw steps later. So the predictor is asked to explain a five-step
displacement from a one-step action, with the other four actions never shown to
it. Roughly four fifths of the causal information is discarded.

The reference (stable_worldmodel/data/buffer.py, `_gather_clip`) does this:

    action_idx = base + arange(history_len * frameskip)   # DENSE, every action
    clip['action'] = clip['action'].reshape(history_len, -1)
                                    # -> (history_len, frameskip * action_dim)

which is exactly why their released config has `Embedder input_dim: 10` --
frameskip 5 x action_dim 2. Their ten-wide action is the five raw actions of
each block, concatenated.

WHAT THIS PATCH DOES
--------------------
Reads `action` (and any other per-step key that is not `pixels`) densely across
the clip span, then reshapes to (num_steps, frameskip * action_dim). Pixels
keep the strided read. The behaviour is switched by a module-level flag so the
old convention stays reproducible:

    DENSE_ACTIONS = True     # reference behaviour (new default)
    DENSE_ACTIONS = False    # the original subsampled behaviour

Every number already committed was produced under False, so set it to False if
you ever need to reproduce them exactly.

AFTER APPLYING, TWO THINGS FOLLOW
---------------------------------
1. `action_dim` in the training config goes from 2 to 10. Our ToyJEPA already
   takes it as a parameter (`ActionEmbedder(action_dim, 10, embed_dim)`), so
   this is a config change, not a code change.
2. Run verify_dense_actions.py before training anything. The environment is
   deterministic, so the new action block must reconstruct the recorded
   displacement exactly -- that is an executable check, not an argument.

Same construction as the other patches (marker check, exact-once assertions,
backup, diff, byte-compile).

Usage (from the tinylab folder):
    python3 patch_tworoom_data_dense_actions.py
"""
from __future__ import annotations

import difflib
import py_compile
import sys
from pathlib import Path

TARGET = Path("tworoom_data.py")
MARKER = "DENSE_ACTIONS"

OLD_READ = '''        for k in self.keys:
            if k == "pixels":
                continue
            out[k] = np.asarray(f[k][idx], dtype=np.float32)'''

NEW_READ = '''        for k in self.keys:
            if k == "pixels":
                continue
            if DENSE_ACTIONS:
                # Keep per-step keys DENSE across the clip span and reshape to
                # (num_steps, frameskip * dim), as the reference does. The
                # strided read used before threw away four of every five
                # actions and asked the predictor to explain a five-step
                # displacement from a one-step action.
                fs = self.index.spec.frameskip
                n = self.index.spec.num_steps
                # Row t holds the fs raw actions taken between clip frame t
                # and clip frame t+1. A clip of n frames has only n-1 such
                # transitions, and reading n blocks would run past the clip
                # span, so the final row is zero-filled. The predictor pairs
                # frame t with frame t+1, so row n-1 has no target and is
                # never scored.
                dense_idx = int(idx[0]) + np.arange((n - 1) * fs)
                dense = np.asarray(f[k][dense_idx],
                                   dtype=np.float32).reshape(n - 1, -1)
                out[k] = np.concatenate(
                    [dense, np.zeros((1, dense.shape[1]), dtype=np.float32)],
                    axis=0)
            else:
                out[k] = np.asarray(f[k][idx], dtype=np.float32)'''

OLD_HEAD = '''class TwoRoomClips:'''

NEW_HEAD = '''# Reference behaviour: actions are gathered at full rate and reshaped to
# (num_steps, frameskip * action_dim). Set to False to reproduce every result
# committed before 2026-07-26, which used the strided (subsampled) read.
DENSE_ACTIONS = True


class TwoRoomClips:'''


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()

    if MARKER in src:
        print(f"'{MARKER}' already present in {TARGET}. No change made.")
        return

    out = src
    for name, old, new in (("module flag", OLD_HEAD, NEW_HEAD),
                           ("clip read", OLD_READ, NEW_READ)):
        cnt = out.count(old)
        if cnt != 1:
            sys.exit(f"ABORT: target for '{name}' appears {cnt} times "
                     f"(need exactly 1). Nothing written.")
        out = out.replace(old, new)

    TARGET.with_suffix(".py.bak").write_text(src)
    TARGET.write_text(out)
    print("--- diff applied ---")
    for line in difflib.unified_diff(src.splitlines(), out.splitlines(),
                                     "before", "after", lineterm="", n=1):
        print(line)
    py_compile.compile(str(TARGET), doraise=True)
    print(f"\nOK: patched and byte-compiles. Backup at {TARGET}.bak")
    print("\nNEXT, IN ORDER:")
    print("  1. python3 verify_dense_actions.py      <- executable gate")
    print("  2. set action_dim: 10 in the training config "
          "(frameskip 5 x action_dim 2)")
    print("  3. only then consider spending the run")


if __name__ == "__main__":
    main()
