"""
diagnose_loader.py -- find out exactly HOW the parallel loader's data differs.

The equivalence proof failed on the real file: helpers returned something
different from the single-process read. Before fixing anything, localize it.
Four different failure modes leave four different fingerprints:

  1. OUT OF ORDER  -- every clip's content is fine, but clips arrive in a
                      different sequence. (A delivery-order bug.)
  2. CORRUPTED     -- some clips' content is actually wrong. (A read bug.)
  3. UNSTABLE      -- reading the same clips twice, single-process, gives
                      different bytes. (A file/driver problem -- would also
                      poison everything we did before, so we test it first.)
  4. SIZE-DEPENDENT-- small hand-offs between processes are fine but the
                      benchmark's ~77 MB batches are not. (A transport issue.)

This script fingerprints every clip (a short digest of its bytes) and compares
sets and sequences, so it can tell reordering from corruption. It also tries
both ways of starting helper processes -- the "fresh start" mode macOS uses by
default, and the "copy" mode Linux uses (which is what the rented machine will
use) -- because a fault in one and not the other localizes the bug AND tells us
whether the rental is even affected.

Run AS A FILE (never pasted into a live session):
    python diagnose_loader.py /Users/joyjeetsingh/Downloads/tworoom.h5
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import sys

import numpy as np

from parallel_data import make_loader
from tworoom_data import ClipSpec, TwoRoomIndex


def digest(t) -> str:
    """A 12-character fingerprint of one clip's exact bytes."""
    return hashlib.sha1(np.ascontiguousarray(t.numpy()).tobytes()).hexdigest()[:12]


def take(h5, idx, order, bs, n_batches, nw, ctx=None):
    """Read n_batches and return one (pixels, action) fingerprint per clip."""
    sub = order[:n_batches * bs]
    loader = make_loader(h5, idx, sub, bs, num_workers=nw)
    if ctx and nw > 0:
        loader.multiprocessing_context = ctx
    out = []
    for b in loader:
        for k in range(b["pixels"].shape[0]):
            out.append((digest(b["pixels"][k]), digest(b["action"][k])))
    return out


def report(name, ref, par):
    if ref == par:
        print(f"  {name:44s} IDENTICAL (content and order)")
        return "ok"
    if sorted(ref) == sorted(par):
        moved = sum(1 for a, b in zip(ref, par) if a != b)
        print(f"  {name:44s} SAME CONTENT, DIFFERENT ORDER "
              f"({moved}/{len(ref)} clips moved)")
        return "reordered"
    ref_px = {r[0] for r in ref}
    par_px = {p[0] for p in par}
    print(f"  {name:44s} CONTENT DIFFERS "
          f"({len(ref_px - par_px)} clips corrupted or replaced)")
    return "corrupted"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python diagnose_loader.py /path/to/tworoom.h5")
    h5 = sys.argv[1]

    import h5py
    import hdf5plugin
    import torch
    print(f"torch {torch.__version__} | h5py {h5py.__version__} | "
          f"hdf5plugin {hdf5plugin.version} | "
          f"helper start mode on this OS: {mp.get_start_method()}")
    print()

    spec = ClipSpec(history=3, frameskip=5)
    idx = TwoRoomIndex(h5, spec)
    order = np.random.default_rng(0).permutation(len(idx.starts))

    BS, NB = 4, 4          # small clips-per-hand-off for stages 1-4
    print(f"stages 1-4 use batch {BS} x {NB} batches ({BS*NB} clips, "
          "small hand-offs)")
    print()

    print("stage 1 -- is single-process reading even stable? (read twice)")
    ref = take(h5, idx, order, BS, NB, nw=0)
    ref2 = take(h5, idx, order, BS, NB, nw=0)
    s1 = report("workers=0 vs workers=0 (same clips twice):", ref, ref2)
    if s1 != "ok":
        print()
        print("  STOP HERE. The file itself does not read back the same bytes")
        print("  twice in one process. Nothing downstream can be trusted until")
        print("  this is understood -- and it would affect the rental equally.")
        return

    print()
    print("stage 2 -- one helper (no helper-to-helper interleaving)")
    w1 = take(h5, idx, order, BS, NB, nw=1)
    report("workers=1 (this OS's default start mode):", ref, w1)

    print()
    print("stage 3 -- two helpers, this OS's default start mode")
    w2 = take(h5, idx, order, BS, NB, nw=2)
    report("workers=2 (this OS's default start mode):", ref, w2)

    print()
    print("stage 4 -- two helpers, 'copy' start mode (what the rented Linux")
    print("           machine uses by default)")
    try:
        w2f = take(h5, idx, order, BS, NB, nw=2, ctx="fork")
        report("workers=2 (copy/fork mode):", ref, w2f)
    except Exception as e:                                    # noqa: BLE001
        print(f"  copy/fork mode unavailable or crashed here: {e}")
        print("  (possible on macOS; the rental runs Linux where it is native)")

    print()
    print("stage 5 -- the benchmark's actual hand-off size (batch 32, ~77 MB")
    print("           per hand-off on the real file)")
    big_ref = take(h5, idx, order, 32, 2, nw=0)
    big_w2 = take(h5, idx, order, 32, 2, nw=2)
    report("workers=2 at batch 32:", big_ref, big_w2)

    print()
    print("how to read this:")
    print("  * stage 3 fine but stage 5 not -> the fault appears only with")
    print("    LARGE hand-offs between processes (a transport-size issue).")
    print("  * 'SAME CONTENT, DIFFERENT ORDER' anywhere -> delivery-order bug;")
    print("    the data is intact, the sequencing is not.")
    print("  * 'CONTENT DIFFERS' -> real read corruption; note WHICH stages.")
    print("  * default-mode fails but copy/fork mode passes -> the fault is in")
    print("    macOS's fresh-start helper mode; the Linux rental is likely")
    print("    unaffected -- but we confirm that on the rented box, never assume.")


if __name__ == "__main__":
    main()
