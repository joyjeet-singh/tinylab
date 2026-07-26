"""
patch_tworoom_data_normalization.py -- add the two preprocessing steps the
reference applies and we never did.

WHAT THE REFERENCE DOES (traced to source, not inferred)
--------------------------------------------------------
le-wm/train.py:59  transforms = [get_img_preprocessor(...)]
le-wm/utils.py:6   get_img_preprocessor = ToImage(**dataset_stats.ImageNet)
                                          then Resize(img_size)
    -> pixels are ImageNet-normalised. Ours divides by 255 and stops.

le-wm/train.py:65  for every non-pixel column: get_column_normalizer(...)
le-wm/utils.py:25  computes per-dimension mean/std over the dataset column and
                   applies ZScoreNormalizer
    -> actions are z-scored by dataset statistics. Ours are raw.

Both are invisible in every config file, which is exactly why they survived
four fidelity passes.

WHAT THIS PATCH ADDS
--------------------
Two module flags, both defaulting to the reference behaviour, both switchable
so previously committed results stay reproducible:

    IMAGENET_PIXELS = True     # (px/255 - mean) / std, ImageNet constants
    ZSCORE_ACTIONS  = True     # (a - mean) / std from the dataset's own actions

Action statistics are computed once from the raw action column and cached on
the dataset object.

ONE INTERPRETATION, FLAGGED AS SUCH
-----------------------------------
The reference computes action statistics on the raw column, whose width is
action_dim (2), and applies them to the clip's action array, whose width is
frameskip x action_dim (10). The exact broadcast it relies on is NOT verified
from source. We tile the 2-wide statistics across the 5 concatenated actions,
which is the semantically correct reading -- every sub-pair of the 10 is one
raw action and should be normalised by raw-action statistics. This is OUR
INTERPRETATION and belongs in the deviations table as such until someone reads
`get_col_data` and confirms it.

Usage (from the tinylab folder, AFTER the dense-actions patch):
    python3 patch_tworoom_data_normalization.py
    python3 preflight_local.py --config configs/<config>.yaml
"""
from __future__ import annotations

import difflib
import py_compile
import sys
from pathlib import Path

TARGET = Path("tworoom_data.py")
MARKER = "IMAGENET_PIXELS"

OLD_FLAG = '''# Reference behaviour: actions are gathered at full rate and reshaped to'''

NEW_FLAG = '''# Reference pixel preprocessing: le-wm/utils.py:6 applies ImageNet mean/std
# (ToImage(**dataset_stats.ImageNet)) before resizing. Ours used to stop at
# dividing by 255. Set False to reproduce results committed before 2026-07-26.
IMAGENET_PIXELS = True
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Reference action preprocessing: le-wm/train.py:65 z-scores every non-pixel
# column using dataset statistics (utils.py:25). Ours used raw actions.
ZSCORE_ACTIONS = True


# Reference behaviour: actions are gathered at full rate and reshaped to'''

OLD_PX = '''            px = px.astype(np.float32) / 255.0           # -> 0..1 decimals'''

NEW_PX = '''            px = px.astype(np.float32) / 255.0           # -> 0..1 decimals
            if IMAGENET_PIXELS:
                # match the reference: ImageNet mean/std, channels last here
                px = (px - np.asarray(IMAGENET_MEAN, dtype=np.float32)) \\
                    / np.asarray(IMAGENET_STD, dtype=np.float32)'''

OLD_ACT = '''                out[k] = np.concatenate(
                    [dense, np.zeros((1, dense.shape[1]), dtype=np.float32)],
                    axis=0)'''

# Normalise the REAL rows only, then append the zero row. Normalising after the
# concatenation would turn the zero sentinel into -mean/std, destroying the
# marker the verifier checks and putting a spurious action in a slot that is
# meant to be empty.
NEW_ACT = '''                if ZSCORE_ACTIONS and k == "action":
                    mu, sd = self._action_stats(f)
                    reps = dense.shape[1] // mu.shape[0]
                    dense = (dense - np.tile(mu, reps)) / np.tile(sd, reps)
                out[k] = np.concatenate(
                    [dense, np.zeros((1, dense.shape[1]), dtype=np.float32)],
                    axis=0)'''

OLD_TAIL = '''        out["_start"] = idx[0]
        return out'''

NEW_TAIL = '''        out["_start"] = idx[0]
        return out

    def _action_stats(self, f):
        """Per-dimension mean/std of the RAW action column, computed once.

        The reference computes these on the raw column (width action_dim) and
        applies them to the clip's action array (width frameskip*action_dim).
        We tile across the concatenated actions, which is the semantically
        correct reading; the reference's exact broadcast is unverified, so
        this is listed as OUR INTERPRETATION in the deviations table.
        """
        if getattr(self, "_act_mu", None) is None:
            a = np.asarray(f["action"][:], dtype=np.float64)
            self._act_mu = a.mean(0).astype(np.float32)
            self._act_sd = a.std(0).astype(np.float32)
            self._act_sd[self._act_sd < 1e-8] = 1.0
        return self._act_mu, self._act_sd'''


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()

    if "DENSE_ACTIONS" not in src:
        sys.exit("run patch_tworoom_data_dense_actions.py first; this builds "
                 "on it.")
    if MARKER in src:
        print(f"'{MARKER}' already present in {TARGET}. No change made.")
        return

    out = src
    for name, old, new in (("flags", OLD_FLAG, NEW_FLAG),
                           ("pixels", OLD_PX, NEW_PX),
                           ("actions", OLD_ACT, NEW_ACT),
                           ("stats helper", OLD_TAIL, NEW_TAIL)):
        cnt = out.count(old)
        if cnt != 1:
            sys.exit(f"ABORT: target for '{name}' appears {cnt} times "
                     f"(need exactly 1). Nothing written.")
        out = out.replace(old, new)

    TARGET.with_suffix(".py.bak2").write_text(src)
    TARGET.write_text(out)
    print("--- diff applied ---")
    for line in difflib.unified_diff(src.splitlines(), out.splitlines(),
                                     "before", "after", lineterm="", n=1):
        print(line)
    py_compile.compile(str(TARGET), doraise=True)
    print(f"\nOK: patched and byte-compiles. Backup at {TARGET}.bak2")
    print("\nBoth flags default True (reference behaviour). Set either to False")
    print("to reproduce earlier results. Run preflight_local.py next.")


if __name__ == "__main__":
    main()
