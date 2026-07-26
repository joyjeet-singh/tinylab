"""
patch_action_stats_nan.py -- fix the NaN bug the local gate just caught.

WHAT HAPPENED
-------------
`verify_dense_actions.py` reported z-scored actions with mean nan, std nan.
The cause is mine: the reference's `get_column_normalizer` has a line I read
and then did not implement --

    le-wm/utils.py:29   data = data[~torch.isnan(data).any(dim=1)]

It drops NaN rows before computing statistics, because **the dataset contains
NaN actions**. Our `_action_stats` averaged over the raw column including them,
so mean and std came back NaN, and every action in every clip would have been
NaN.

This is worse than a statistics bug. The old subsampled loader read one action
in five and may have stepped over the NaNs by luck; the dense loader reads all
of them, so any clip whose block contains a NaN now carries NaN straight into
the loss. On a paid run that is a silent, total loss of the training signal --
NaN gradients from the first batch that touches one.

So this patch does two things:

1. Drops NaN rows before computing the mean and standard deviation, exactly as
   the reference does.
2. Adds `nan_audit()` to the dataset class: counts NaN actions in the file,
   reports where they sit relative to episode boundaries, and counts how many
   sampled clips contain one. The verifier calls it, and the gate fails if any
   clip is affected -- because a clip with a NaN action must be excluded or the
   run is wasted.

Usage (after patch_tworoom_data_normalization.py):
    python3 patch_action_stats_nan.py
    python3 verify_dense_actions.py
"""
from __future__ import annotations

import difflib
import py_compile
import sys
from pathlib import Path

TARGET = Path("tworoom_data.py")
MARKER = "nan_audit"

OLD = '''        if getattr(self, "_act_mu", None) is None:
            a = np.asarray(f["action"][:], dtype=np.float64)
            self._act_mu = a.mean(0).astype(np.float32)
            self._act_sd = a.std(0).astype(np.float32)
            self._act_sd[self._act_sd < 1e-8] = 1.0
        return self._act_mu, self._act_sd'''

NEW = '''        if getattr(self, "_act_mu", None) is None:
            a = np.asarray(f["action"][:], dtype=np.float64)
            # The reference drops NaN rows before computing statistics
            # (le-wm/utils.py:29). The dataset contains them; averaging over
            # them gives NaN mean/std and poisons every clip.
            keep = ~np.isnan(a).any(axis=1)
            self._n_nan_actions = int((~keep).sum())
            a = a[keep]
            self._act_mu = a.mean(0).astype(np.float32)
            self._act_sd = a.std(0).astype(np.float32)
            self._act_sd[self._act_sd < 1e-8] = 1.0
        return self._act_mu, self._act_sd

    def nan_audit(self, n_clips: int = 500, seed: int = 0):
        """Where are the NaN actions, and do any clips contain one?

        A clip whose action block holds a NaN produces a NaN loss and NaN
        gradients from the first batch that touches it. Better to find that
        here than on a rented machine.
        """
        import h5py
        with h5py.File(self.h5_path, "r") as f:
            a = np.asarray(f["action"][:], dtype=np.float64)
            bad = np.where(np.isnan(a).any(axis=1))[0]
            off = np.asarray(f["ep_offset"][:])
            ln = np.asarray(f["ep_len"][:])
        report = {"total_actions": len(a), "nan_actions": int(len(bad))}
        if len(bad):
            ends = set((off + ln - 1).tolist())
            starts = set(off.tolist())
            report["at_episode_end"] = int(sum(1 for i in bad if i in ends))
            report["at_episode_start"] = int(sum(1 for i in bad if i in starts))
            report["elsewhere"] = int(len(bad) - report["at_episode_end"]
                                      - report["at_episode_start"])
            report["first_few"] = bad[:5].tolist()
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(self), size=min(n_clips, len(self)),
                           replace=False)
        hit = 0
        for i in picks:
            if np.isnan(np.asarray(self[int(i)]["action"], dtype=np.float64)).any():
                hit += 1
        report["clips_sampled"] = len(picks)
        report["clips_with_nan"] = hit
        return report'''


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()
    if MARKER in src:
        print(f"'{MARKER}' already present in {TARGET}. No change made.")
        return
    if "_action_stats" not in src:
        sys.exit("run patch_tworoom_data_normalization.py first.")
    cnt = src.count(OLD)
    if cnt != 1:
        sys.exit(f"ABORT: target appears {cnt} times (need exactly 1). "
                 f"Nothing written.")
    out = src.replace(OLD, NEW)
    TARGET.with_suffix(".py.bak3").write_text(src)
    TARGET.write_text(out)
    print("--- diff applied ---")
    for line in difflib.unified_diff(src.splitlines(), out.splitlines(),
                                     "before", "after", lineterm="", n=1):
        print(line)
    py_compile.compile(str(TARGET), doraise=True)
    print(f"\nOK: patched and byte-compiles. Backup at {TARGET}.bak3")
    print("Run verify_dense_actions.py -- it now audits NaNs and fails if any")
    print("sampled clip contains one.")


if __name__ == "__main__":
    main()
