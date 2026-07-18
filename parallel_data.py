"""
parallel_data.py -- reading clips with several helpers at once.

The problem, in plain terms
---------------------------
Reading one clip from the real file costs decompressing a ~15 MB block, and
that work happens on a single processor core. Our training loop reads clips one
at a time, so on a rented 32-core machine, 31 cores would sit idle while one
does all the unpacking. The fix: several helper processes, each decompressing
different clips at the same time, handing finished batches to the trainer.

Two safety rules make this correct rather than merely fast:

  1. EACH HELPER OPENS ITS OWN COPY OF THE FILE. A file handle shared across
     processes gets corrupted reads. Our reader already opens the file lazily;
     here we add a guard that notices when a helper inherited the parent's
     handle (which happens on Linux) and quietly opens a fresh one instead.

  2. THE ORDER OF CLIPS IS FIXED BEFORE ANY HELPER STARTS. Helpers change how
     fast batches arrive, never which clips are in them or in what order. The
     epoch's order is computed up front (from the seed, as before) and handed
     to the loader as a fixed list of batches. PyTorch's loader returns batches
     in that exact order no matter which helper finished first. This is what
     keeps the bit-identical resume guarantee intact.

macOS note: helper processes there start "fresh" (spawn) rather than as copies
(fork), which means the dataset object is shipped to them in serialized form.
An open file handle cannot be serialized, so we drop it before shipping
(__getstate__) and each helper opens its own on first read.

Run the benchmark on the REAL file to measure scaling:
    python parallel_data.py --h5 /path/to/tworoom.h5 --workers 0 1 2
It first PROVES helpers return identical data to the single-process path, then
times each worker count and projects training hours.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from tworoom_data import ClipSpec, TwoRoomClips, TwoRoomIndex


class WorkerSafeClips(TwoRoomClips):
    """
    TwoRoomClips, hardened for helper processes.

    - If this process inherited an already-open handle from its parent (Linux
      fork), abandon it and open a fresh one. Sharing the handle corrupts
      reads; abandoning the inherited copy is safe (the parent's is untouched).
    - When shipped to a fresh helper (macOS spawn), drop the handle first --
      open handles cannot be serialized.
    """

    def __init__(self, h5_path, index, keys=("pixels", "action")):
        super().__init__(h5_path, index, keys)
        self._pid = None

    def _file(self):
        me = os.getpid()
        if self._f is not None and self._pid != me:
            self._f = None                  # inherited from parent: abandon, reopen
        if self._f is None:
            import h5py
            import hdf5plugin  # noqa: F401  (each helper must register Blosc)
            self._f = h5py.File(self.h5_path, "r")
            self._pid = me
        return self._f

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_f"] = None                      # never ship an open handle
        d["_pid"] = None
        return d


class ClipDataset(torch.utils.data.Dataset):
    """One item = one clip, addressed by its position in index.starts."""

    def __init__(self, h5_path: str, index: TwoRoomIndex):
        self.clips = WorkerSafeClips(h5_path, index)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, pos):
        item = self.clips[int(pos)]
        return item["pixels"], item["action"]


def collate(batch):
    """Stack clips into the dict shape the training loop already expects."""
    return {
        "pixels": torch.from_numpy(np.stack([b[0] for b in batch])),
        "action": torch.from_numpy(np.stack([b[1] for b in batch])),
    }


def make_loader(h5_path: str, index: TwoRoomIndex, order: np.ndarray,
                batch_size: int, num_workers: int = 0, begin: int = 0,
                pin_memory: bool = False):
    """
    Batches of clips in EXACTLY the order given, read by `num_workers` helpers.

    `order` is the epoch's precomputed clip order (positions into
    index.starts); `begin` supports mid-epoch resume. num_workers=0 reads in
    the main process -- the old behaviour, byte for byte.
    """
    ds = ClipDataset(h5_path, index)
    batches = [order[s:s + batch_size]
               for s in range(begin, len(order) - batch_size + 1, batch_size)]
    kwargs = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4       # keep helpers a few batches ahead
    # A private random source. PyTorch's loader draws one number at creation
    # (to seed helpers); without this it would draw from the GLOBAL stream --
    # and a resumed run creates one extra loader, shifting every dropout mask
    # after the resume point. That would silently break the bit-identical
    # resume guarantee. Our reads use no randomness, so the seed value itself
    # is irrelevant; what matters is that the global stream is never touched.
    kwargs["generator"] = torch.Generator().manual_seed(0)
    return torch.utils.data.DataLoader(
        ds, batch_sampler=batches, num_workers=num_workers,
        collate_fn=collate, pin_memory=pin_memory,
        persistent_workers=False, **kwargs)


# ---------------------------------------------------------------------------
# proof + benchmark
# ---------------------------------------------------------------------------
def check_equivalence(h5_path, index, order, batch_size, num_workers,
                      n_batches=4, context=None) -> bool:
    """Helpers must change SPEED only, never CONTENT. Verified, not assumed."""
    def take(nw):
        # Build the loader over EXACTLY the clips we need, so iteration
        # completes naturally. Abandoning a loader mid-iteration leaves
        # helpers mid-prefetch, and their forced shutdown is noisy/crashy.
        sub = order[:n_batches * batch_size]
        loader = make_loader(h5_path, index, sub, batch_size, num_workers=nw)
        if context and nw > 0:
            loader.multiprocessing_context = context
        return [(b["pixels"].clone(), b["action"].clone()) for b in loader]

    ref, par = take(0), take(num_workers)

    def same(a, b):
        # Exact BYTES, not torch.equal. The real file's action array contains
        # NaN sentinel values (see the benchmark's scan), and by floating-point
        # rules NaN never equals anything -- not even itself -- so torch.equal
        # reports identical NaN-carrying tensors as different. Bitwise equality
        # is both NaN-proof and the strictest possible test.
        return (a.shape == b.shape and a.dtype == b.dtype
                and a.numpy().tobytes() == b.numpy().tobytes())

    for (p0, a0), (p1, a1) in zip(ref, par):
        if not (same(p0, p1) and same(a0, a1)):
            return False
    return True


def benchmark(h5_path, worker_counts, batch_size=32, n_batches=8):
    spec = ClipSpec(history=3, frameskip=5)
    index = TwoRoomIndex(h5_path, spec)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(index.starts))

    print(f"file: {h5_path}")
    print(f"legal clips: {len(index.starts):,}   batch {batch_size}, "
          f"{n_batches} batches per timing")

    # Scan the action array for NaN sentinel rows. These are real values in
    # the dataset (typically marking episode-final steps where no action
    # follows); training neutralises them (nan_to_num, mirroring the
    # reference), and the equivalence check below compares exact bytes so
    # identical NaNs count as identical. Reported here so they are visible
    # information instead of a silent trap.
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(h5_path, "r") as f:
        nan_rows = np.where(np.isnan(np.asarray(f["action"])).any(axis=1))[0]
        ep_off = np.asarray(f["ep_offset"][:])
        ep_len_arr = np.asarray(f["ep_len"][:])
    finals = set((ep_off + ep_len_arr - 1).tolist())
    at_final = sum(1 for r in nan_rows.tolist() if r in finals)
    if len(nan_rows):
        print(f"NaN action rows: {len(nan_rows):,} "
              f"({at_final:,} at episode-final steps)"
              + ("" if at_final == len(nan_rows) else
                 f"  <- {len(nan_rows) - at_final:,} NOT at episode ends: inspect"))
    else:
        print("NaN action rows: 0")
    print()

    max_w = max(w for w in worker_counts)
    if max_w > 0:
        print("proof first -- helpers must return IDENTICAL data:")
        ok = check_equivalence(h5_path, index, order, batch_size, max_w)
        print(f"  workers=0 vs workers={max_w}: "
              f"{'IDENTICAL -- PASS' if ok else 'MISMATCH -- FAIL, do not use'}")
        if not ok:
            return
        print()

    results = {}
    for nw in worker_counts:
        # loader sized to warmup + timed batches, so it completes naturally
        sub = order[:(n_batches + 1) * batch_size]
        loader = make_loader(h5_path, index, sub, batch_size, num_workers=nw)
        it = iter(loader)
        next(it)                            # warmup: helper startup + first block
        t0 = time.time()
        n = 0
        for b in it:
            n += b["pixels"].shape[0]
        dt = time.time() - t0
        results[nw] = n / dt
        print(f"  workers={nw}: {results[nw]:8.1f} clips/sec")

    base = results.get(0)
    print()
    if base:
        for nw, cps in results.items():
            if nw > 0:
                print(f"  workers={nw} speedup over single-process: {cps/base:.2f}x")
    print()
    print("projected DATA LOADING for the real run "
          "(693k train clips x 10 epochs):")
    for nw, cps in results.items():
        print(f"  workers={nw}: {693_000 * 10 / cps / 3600:7.1f} h")
    print()
    print("NOTE: run this ON the machine you care about. Clips/sec is a")
    print("property of that machine's cores and disk, not of the code.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batches", type=int, default=8)
    a = p.parse_args()
    benchmark(a.h5, a.workers, a.batch_size, a.batches)
