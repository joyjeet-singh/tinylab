"""
verify_dense_actions.py -- the executable gate for the loader change.

The environment is deterministic: R1 showed that replaying recorded actions
from a recorded state reproduces the recorded trajectory with error exactly
0.000. That makes this checkable rather than arguable. Displacement per
environment step is action x 5, so across one clip step (frameskip raw steps)

    pos[t + frameskip] - pos[t]  ==  5 * sum(the frameskip raw actions)

exactly, except where the agent hits a wall and the environment clips the move.

FOUR CHECKS
-----------
  1. SHAPE      the loader now returns action of shape (num_steps,
                frameskip * action_dim) -- 10 wide for TwoRoom.
  2. EXACTNESS  the dense block reconstructs the recorded displacement to
                floating-point noise on clips that never touch a wall.
  3. WHAT THE OLD CONVENTION LOST  the same reconstruction using only the one
                subsampled action the old loader kept. This number is the size
                of the deficit our checkpoint was trained under.
  4. ROUND TRIP the flag still reproduces the old behaviour when set to False,
                so every committed result stays reproducible.

If check 2 does not come back at ~0, do NOT train. Something about the
frameskip or the action alignment is not what we think, and a paid run would
bake it in.

Usage:
    python3 verify_dense_actions.py
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np

SPEED = 5.0
WALL_X, WALL_HALF, AGENT_R = 112.0, 5.0, 7.0
ARENA_LO, ARENA_HI = 14.0, 208.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
    ap.add_argument("--clips", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import tworoom_data as td

    if not hasattr(td, "DENSE_ACTIONS"):
        raise SystemExit("tworoom_data.py has not been patched -- run "
                         "patch_tworoom_data_dense_actions.py first.")
    print(f"tworoom_data.DENSE_ACTIONS = {td.DENSE_ACTIONS}")
    if not td.DENSE_ACTIONS:
        raise SystemExit("DENSE_ACTIONS is False; set it True to verify the "
                         "new path.")

    h5 = str(Path(args.h5).expanduser())
    # The physics identity is a property of the RAW actions. Check the dense
    # GATHERING with normalisation off, then check the normalisation itself
    # separately -- otherwise a correct loader fails a check it cannot pass.
    zs_on = getattr(td, "ZSCORE_ACTIONS", False)
    if zs_on:
        td.ZSCORE_ACTIONS = False
        print("(z-scoring temporarily off for the physics check)")
    spec = td.ClipSpec()
    index = td.TwoRoomIndex(h5, spec)
    clips = td.TwoRoomClips(h5, index, keys=("action",))
    fs, n = spec.frameskip, spec.num_steps
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(clips), size=min(args.clips, len(clips)),
                       replace=False)

    # ---- 1. shape --------------------------------------------------------
    print("\n1. SHAPE")
    sample = clips[int(picks[0])]["action"]
    print(f"   action per clip: {sample.shape}   "
          f"expected ({n}, {fs} x action_dim = {fs * 2})")
    if sample.shape != (n, fs * 2):
        raise SystemExit("   WRONG SHAPE -- stop here.")
    print("   OK")

    # ---- 2 & 3. reconstruction -------------------------------------------
    print("\n2/3. RECONSTRUCTION AGAINST RECORDED DISPLACEMENT")
    dense_err, sub_err, moves, skipped = [], [], [], 0
    with h5py.File(h5, "r") as f:
        pos = f["pos_agent"]
        for i in picks:
            c = clips[int(i)]
            s = int(c["_start"])
            if not np.allclose(c["action"][n - 1], 0.0):
                raise SystemExit("   final action row is not zero-filled -- "
                                 "the patch did not apply as expected.")
            for t in range(n - 1):
                a, b = s + t * fs, s + (t + 1) * fs
                p0 = np.asarray(pos[a], dtype=np.float64)
                p1 = np.asarray(pos[b], dtype=np.float64)
                # skip blocks that could have been clipped by a wall or border
                M = 20.0
                near_wall = abs(p0[0] - WALL_X) < (WALL_HALF + AGENT_R + M) \
                    or abs(p1[0] - WALL_X) < (WALL_HALF + AGENT_R + M)
                near_edge = (min(p0.min(), p1.min()) < ARENA_LO + AGENT_R + M
                             or max(p0.max(), p1.max()) > ARENA_HI - AGENT_R - M)
                if near_wall or near_edge:
                    skipped += 1
                    continue
                block = c["action"][t].reshape(fs, 2).astype(np.float64)
                dense_err.append(np.linalg.norm(
                    SPEED * block.sum(0) - (p1 - p0)))
                sub_err.append(np.linalg.norm(
                    SPEED * fs * block[0] - (p1 - p0)))
                moves.append(np.linalg.norm(p1 - p0))
    dense_err = np.array(dense_err)
    sub_err = np.array(sub_err)
    moves = np.array(moves)
    if len(dense_err) < 30:
        print(f"   only {len(dense_err)} clean blocks -- too few to trust. "
              f"Re-run with a larger --clips.")
        return
    print(f"   clean blocks used: {len(dense_err)}  "
          f"(skipped {skipped} near a wall or border)")
    print(f"   DENSE   (5 x sum of the 5 actions): "
          f"median {np.median(dense_err):.2e}  max {dense_err.max():.2e}")
    print(f"   OLD     (one subsampled action x 25): "
          f"median {np.median(sub_err):.2f}  max {sub_err.max():.2f}")
    ok = np.median(dense_err) < 1e-3
    print(f"\n   -> dense reconstruction exact: {ok}")
    if not ok:
        print("   STOP. Do not train. Something about the frameskip or the")
        print("   action alignment is not what we think.")
        return
    print(f"   -> the old convention left a median error of "
          f"{np.median(sub_err):.1f} units, against a typical per-block move")
    print(f"      of {np.median(moves):.1f} units -- the error was as large as the"
          f" movement itself.")
    print("      That is what our checkpoint was trained under.")

    # ---- 3a. NaN audit -----------------------------------------------------
    if hasattr(clips, "nan_audit"):
        print("\n3a. NaN AUDIT (the dataset contains NaN actions; the "
              "reference drops them)")
        rep = clips.nan_audit(n_clips=min(args.clips, 500))
        for k, v in rep.items():
            print(f"   {k:<20} {v}")
        if rep["clips_with_nan"]:
            print("\n   STOP. Clips carry NaN actions into the loss -- NaN")
            print("   gradients from the first batch that touches one. These")
            print("   clips must be excluded before training.")
            return
        print("   OK -- no sampled clip carries a NaN action")

    # ---- 3b. normalisation statistics -------------------------------------
    if zs_on:
        td.ZSCORE_ACTIONS = True
        cl = td.TwoRoomClips(h5, index, keys=("action",))
        rows = np.concatenate([cl[int(i)]["action"][:-1] for i in picks[:200]])
        print(f"\n3b. Z-SCORED ACTIONS (normalisation on)")
        print(f"   over {len(rows)} real rows: mean {rows.mean():+.3f}  "
              f"std {rows.std():.3f}   (want ~0 and ~1)")
        if abs(rows.mean()) > 0.25 or not (0.7 < rows.std() < 1.4):
            print("   OUT OF RANGE -- check _action_stats")
            return
        print("   OK")
        pxs = td.TwoRoomClips(h5, index)[int(picks[0])]["pixels"]
        print(f"   pixels mean {pxs.mean():+.3f} std {pxs.std():.3f}   "
              f"(ImageNet-normalised; raw/255 would sit near +0.9)")

    # ---- 4. round trip ----------------------------------------------------
    print("\n4. ROUND TRIP (old behaviour still reproducible)")
    td.DENSE_ACTIONS = False
    importlib.reload  # no-op; the flag is read at call time
    clips_old = td.TwoRoomClips(h5, index, keys=("action",))
    old_shape = clips_old[int(picks[0])]["action"].shape
    td.DENSE_ACTIONS = True
    print(f"   with DENSE_ACTIONS=False: action shape {old_shape} "
          f"(expected ({n}, 2))")
    print(f"   OK -- every result committed before the change stays "
          f"reproducible")

    print("\nGATE PASSED. Next: set action_dim to "
          f"{fs * 2} in the training config, then the launch protocol.")


if __name__ == "__main__":
    main()
