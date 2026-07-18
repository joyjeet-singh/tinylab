"""
tworoom_data.py -- reading the Two-Room dataset, and proving we read it right.

Plain-language summary
----------------------
The dataset is one long reel of 920,809 frames, with 10,000 separate episodes
laid end to end inside it. Two little tables tell us where each episode starts
(`ep_offset`) and how long it is (`ep_len`).

Our job is to hand the model short clips: a few frames in a row, plus the
actions that happened between them. The one rule that matters: a clip must never
cross an episode boundary. If it does, the model sees the agent teleport from
the end of one episode to the start of another, and quietly learns nonsense.

The file stores frames as whole numbers 0-255 with colour last (H, W, C).
PyTorch wants decimals 0-1 with colour first (C, H, W). We convert on the way
out, and only for the frames we actually use -- the full reel would be ~138 GB
uncompressed, so we never load it all.

The alignment checks are the point of this file
-----------------------------------------------
The most common silent killer of a reproduction is frames and actions being
off by one step. Training still runs. The loss still falls. The result is junk.

We check two ways:
  1. NUMERIC (the hard check): the file gives us `pos_agent`, the agent's true
     position at each step. If the action at step t is what moves the agent,
     then (pos_agent[t+1] - pos_agent[t]) should point the same way as
     action[t]. We measure the agreement across thousands of steps. This is a
     number, not an opinion.
  2. VISUAL (the sanity check): plot frames in order with their actions printed
     underneath, and look at whether the dot moves the way the actions say.

Run `python tworoom_data.py --h5 /path/to/tworoom.h5` for both checks.
No GPU needed. Everything here runs on a laptop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import h5py
import hdf5plugin  # noqa: F401  -- registers Blosc (filter 32001); MUST precede pixel reads
import numpy as np


# ---------------------------------------------------------------------------
# episode bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class ClipSpec:
    """How to cut clips out of the reel. Mirrors the reference config."""
    history: int = 3        # frames of context the predictor sees
    num_preds: int = 1      # frames to predict
    frameskip: int = 5      # take every 5th frame (reference config)

    @property
    def num_steps(self) -> int:
        """Frames per clip. Reference: num_steps = num_preds + history_size."""
        return self.num_preds + self.history

    @property
    def span(self) -> int:
        """How many raw frames a clip covers once frameskip is applied."""
        return (self.num_steps - 1) * self.frameskip + 1


class TwoRoomIndex:
    """
    Works out every legal clip start, without reading a single frame.

    A clip starting at raw index `s` inside an episode is legal only if the
    whole clip -- all `span` frames of it -- stays inside that same episode.
    """

    def __init__(self, h5_path: str, spec: ClipSpec):
        self.h5_path = h5_path
        self.spec = spec
        with h5py.File(h5_path, "r") as f:
            self.ep_offset = np.asarray(f["ep_offset"][:])   # (10000,) start index
            self.ep_len = np.asarray(f["ep_len"][:])         # (10000,) length
            self.n_frames = f["pixels"].shape[0]
            self.action_dim = f["action"].shape[1]
            self.img_shape = f["pixels"].shape[1:]
        self.starts = self._legal_starts()

    def _legal_starts(self) -> np.ndarray:
        """All raw indices where a full clip fits inside one episode."""
        span = self.spec.span
        out = []
        for off, ln in zip(self.ep_offset, self.ep_len):
            last = ln - span            # last in-episode start, relative
            if last < 0:
                continue                # episode too short for even one clip
            out.append(np.arange(off, off + last + 1, dtype=np.int64))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

    def clip_indices(self, start: int) -> np.ndarray:
        """The raw frame indices a clip starting at `start` uses."""
        return start + np.arange(self.spec.num_steps) * self.spec.frameskip

    def summary(self) -> str:
        eps_used = int((self.ep_len >= self.spec.span).sum())
        return (
            f"frames in file      : {self.n_frames:,}\n"
            f"episodes            : {len(self.ep_len):,} "
            f"(mean length {self.ep_len.mean():.1f})\n"
            f"clip shape          : {self.spec.num_steps} frames, "
            f"frameskip {self.spec.frameskip} -> spans {self.spec.span} raw frames\n"
            f"episodes long enough: {eps_used:,}\n"
            f"legal clip starts   : {len(self.starts):,}"
        )


# ---------------------------------------------------------------------------
# reading clips
# ---------------------------------------------------------------------------
class TwoRoomClips:
    """
    Reads clips on demand. Opens the file lazily so this object can be handed
    to worker processes safely (an open HDF5 handle cannot be pickled).
    """

    def __init__(self, h5_path: str, index: TwoRoomIndex, keys=("pixels", "action")):
        self.h5_path = h5_path
        self.index = index
        self.keys = keys
        self._f = None

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")
        return self._f

    def __len__(self):
        return len(self.index.starts)

    def __getitem__(self, i: int) -> dict:
        f = self._file()
        idx = self.index.clip_indices(int(self.index.starts[i]))
        out = {}
        # h5py needs sorted, increasing indices for fancy indexing -- ours are.
        if "pixels" in self.keys:
            px = f["pixels"][idx]                        # (T,224,224,3) uint8
            px = px.astype(np.float32) / 255.0           # -> 0..1 decimals
            out["pixels"] = np.transpose(px, (0, 3, 1, 2))  # -> (T,3,224,224)
        for k in self.keys:
            if k == "pixels":
                continue
            out[k] = np.asarray(f[k][idx], dtype=np.float32)
        out["_start"] = idx[0]
        return out

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


# ---------------------------------------------------------------------------
# CHECK 1 -- numeric alignment (the hard one)
# ---------------------------------------------------------------------------
def check_alignment_numeric(h5_path: str, n_samples: int = 4000, seed: int = 0) -> dict:
    """
    Does action[t] explain the move from pos_agent[t] to pos_agent[t+1]?

    We compare, for each of three candidate offsets, how well the action's
    direction agrees with the actual movement direction. The offset with the
    best agreement tells us the true convention:

        offset  0 : action[t] causes pos[t] -> pos[t+1]   (what we assume)
        offset -1 : action[t] causes pos[t-1] -> pos[t]   (off by one, early)
        offset +1 : action[t] causes pos[t+1] -> pos[t+2] (off by one, late)

    Agreement is measured with cosine similarity: +1 means the action and the
    movement point the same way, 0 means unrelated, -1 means opposite.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        ep_offset = np.asarray(f["ep_offset"][:])
        ep_len = np.asarray(f["ep_len"][:])

        # sample steps that are safely inside an episode (away from both ends)
        picks = []
        eps = rng.choice(len(ep_len), size=min(n_samples, len(ep_len)), replace=True)
        for e in eps:
            off, ln = int(ep_offset[e]), int(ep_len[e])
            if ln < 6:
                continue
            t = int(rng.integers(off + 2, off + ln - 3))
            picks.append(t)
        picks = np.array(sorted(set(picks)), dtype=np.int64)

        # read the small slices we need
        lo, hi = picks.min() - 2, picks.max() + 3
        pos = np.asarray(f["pos_agent"][lo:hi], dtype=np.float64)
        act = np.asarray(f["action"][lo:hi], dtype=np.float64)
        rel = picks - lo

    def cos_agreement(offset: int) -> float:
        a = act[rel]                          # the action at t
        p0 = pos[rel + offset]                # position before
        p1 = pos[rel + offset + 1]            # position after
        move = p1 - p0
        na = np.linalg.norm(a, axis=1)
        nm = np.linalg.norm(move, axis=1)
        keep = (na > 1e-8) & (nm > 1e-8)      # skip no-op steps
        if keep.sum() == 0:
            return float("nan")
        cos = (a[keep] * move[keep]).sum(1) / (na[keep] * nm[keep])
        return float(np.mean(cos))

    results = {off: cos_agreement(off) for off in (-1, 0, 1)}
    best = max(results, key=lambda k: (results[k] if not np.isnan(results[k]) else -9))
    return {"agreement": results, "best_offset": best, "n_used": len(picks)}


# ---------------------------------------------------------------------------
# CHECK 2 -- visual alignment
# ---------------------------------------------------------------------------
def check_alignment_visual(h5_path: str, episode: int = 0, n_show: int = 8,
                           out_png: str = "alignment_check.png") -> str:
    """Plot consecutive frames with their actions, so you can look with your eyes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(h5_path, "r") as f:
        off = int(f["ep_offset"][episode])
        ln = int(f["ep_len"][episode])
        n = min(n_show, ln)
        px = np.asarray(f["pixels"][off:off + n])        # (n,224,224,3) uint8
        ac = np.asarray(f["action"][off:off + n])
        pa = np.asarray(f["pos_agent"][off:off + n])

    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.9))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(px[i])
        moved = "" if i == 0 else f"\nmoved {np.round(pa[i]-pa[i-1], 2)}"
        ax.set_title(f"t={i}\nact {np.round(ac[i], 2)}{moved}", fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()
    return out_png


# ---------------------------------------------------------------------------
# CHECK 3 -- clips never cross episode boundaries
# ---------------------------------------------------------------------------
def check_no_boundary_crossing(h5_path: str, index: TwoRoomIndex,
                               n_samples: int = 5000, seed: int = 0) -> dict:
    """Every clip's frames must carry the same ep_idx. Verified, not assumed."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(index.starts), size=min(n_samples, len(index.starts)),
                       replace=False)
    bad = 0
    with h5py.File(h5_path, "r") as f:
        ep_idx = f["ep_idx"]
        for i in picks:
            idx = index.clip_indices(int(index.starts[i]))
            eps = ep_idx[idx]
            if len(np.unique(eps)) != 1:
                bad += 1
    return {"checked": len(picks), "crossing": bad}


# ---------------------------------------------------------------------------
# CHECK 4 -- read speed (random vs block sampling)
# ---------------------------------------------------------------------------
def _chunk_of(start: int, chunk_rows: int) -> int:
    """Which compressed block a raw frame index falls in."""
    return start // chunk_rows


def block_sampler(index: TwoRoomIndex, batch_size: int, chunk_rows: int,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Pick a batch of clips that mostly live in the SAME compressed block.

    Why: frames are stored 100-to-a-block, compressed together. Reading any one
    frame costs decompressing all 100 (~15 MB). If a whole batch comes from one
    block, we pay that cost once instead of `batch_size` times.

    The trade-off is real and must not be hidden: clips in a batch are now
    neighbours in time rather than independent draws from the whole dataset.
    That is a DEVIATION from shuffled training, not a free optimisation.
    """
    by_chunk = {}
    for pos, s in enumerate(index.starts):
        by_chunk.setdefault(_chunk_of(int(s), chunk_rows), []).append(pos)
    chunks = [c for c, v in by_chunk.items() if len(v) >= 1]

    picks = []
    while len(picks) < batch_size:
        c = int(rng.choice(chunks))
        pool = by_chunk[c]
        take = min(batch_size - len(picks), len(pool))
        picks.extend(rng.choice(pool, size=take, replace=False).tolist())
    return np.array(picks[:batch_size], dtype=np.int64)


def measure_read_speed(h5_path: str, index: TwoRoomIndex, batch_size: int = 32,
                       n_batches: int = 12, chunk_rows: int = 100,
                       seed: int = 0) -> dict:
    """
    Time both sampling strategies. Reports clips/second -- the number that says
    whether the data path or the GPU is the bottleneck.
    """
    import time

    results = {}
    for mode in ("random", "block"):
        rng = np.random.default_rng(seed)
        ds = TwoRoomClips(h5_path, index)
        # warmup: first read pays for opening the file + OS cache misses
        _ = ds[int(rng.integers(len(index.starts)))]

        t0 = time.time()
        n_clips = 0
        for _ in range(n_batches):
            if mode == "random":
                picks = rng.choice(len(index.starts), size=batch_size, replace=False)
            else:
                picks = block_sampler(index, batch_size, chunk_rows, rng)
            for p in picks:
                _ = ds[int(p)]
                n_clips += 1
        elapsed = time.time() - t0
        ds.close()

        results[mode] = {
            "clips_per_sec": n_clips / elapsed,
            "sec_per_batch": elapsed / n_batches,
            "n_clips": n_clips,
        }

    r, b = results["random"]["clips_per_sec"], results["block"]["clips_per_sec"]
    results["speedup"] = b / r if r > 0 else float("nan")
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True, help="path to tworoom.h5")
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-speed", action="store_true",
                   help="skip the read-speed benchmark")
    p.add_argument("--batch-size", type=int, default=32,
                   help="clips per batch for the speed test")
    p.add_argument("--speed-batches", type=int, default=12,
                   help="how many batches to time (keep small: each is real I/O)")
    p.add_argument("--chunk-rows", type=int, default=100,
                   help="frames per compressed block (from d.chunks[0])")
    args = p.parse_args()

    spec = ClipSpec(history=args.history, frameskip=args.frameskip)

    print("=" * 64)
    print("INDEX")
    print("=" * 64)
    index = TwoRoomIndex(args.h5, spec)
    print(index.summary())

    print()
    print("=" * 64)
    print("CHECK 1 -- numeric alignment (does action[t] explain the movement?)")
    print("=" * 64)
    res = check_alignment_numeric(args.h5)
    for off, v in sorted(res["agreement"].items()):
        label = {0: "assumed", -1: "off-by-one (early)", 1: "off-by-one (late)"}[off]
        print(f"  offset {off:+d} ({label:18s}): agreement {v:+.4f}")
    print(f"  steps used: {res['n_used']:,}")
    print()
    best = res["best_offset"]
    if best == 0 and res["agreement"][0] > 0.5:
        print("  PASS -- action[t] explains pos[t] -> pos[t+1]. Our assumption holds.")
    elif best == 0:
        print("  WEAK -- offset 0 is best but agreement is low. Actions may be")
        print("          accelerations/velocities rather than displacements.")
        print("          Inspect before trusting the loader.")
    else:
        print(f"  FAIL -- offset {best:+d} agrees better than 0. The frames and")
        print("          actions are misaligned relative to our assumption.")
        print("          FIX THIS BEFORE TRAINING ANYTHING.")

    print()
    print("=" * 64)
    print("CHECK 2 -- clips stay inside one episode")
    print("=" * 64)
    bc = check_no_boundary_crossing(args.h5, index)
    print(f"  clips checked: {bc['checked']:,}   crossing a boundary: {bc['crossing']}")
    print("  PASS" if bc["crossing"] == 0 else "  FAIL -- indexing bug")

    print()
    print("=" * 64)
    print("CHECK 3 -- one clip, end to end")
    print("=" * 64)
    ds = TwoRoomClips(args.h5, index)
    item = ds[0]
    for k, v in item.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:10s} shape={str(v.shape):24s} dtype={v.dtype} "
                  f"range=[{v.min():.3f}, {v.max():.3f}]")
    px = item["pixels"]
    ok = (px.ndim == 4 and px.shape[1] == 3 and 0.0 <= px.min() and px.max() <= 1.0)
    print("  PASS -- (T,3,224,224) decimals in 0..1" if ok else "  FAIL -- wrong shape/range")
    ds.close()

    if not args.no_plot:
        print()
        print("=" * 64)
        print("VISUAL CHECK")
        print("=" * 64)
        out = check_alignment_visual(args.h5)
        print(f"  wrote {out}")
        print("  LOOK AT IT: does the dot move the way each action says,")
        print("  in the frame AFTER the action? The printed 'moved' values")
        print("  should match the action's direction.")
        print()
        print("  NOTE ON DIRECTION: image rows count DOWNWARD from the top, while")
        print("  position coordinates count UPWARD. So a negative y-action renders")
        print("  as the dot moving UP. This is the usual image-vs-world flip. It")
        print("  does NOT affect training (the model only sees pixels + numbers),")
        print("  but it matters when interpreting plans. Do not 'fix' it.")

    if not args.no_speed:
        print()
        print("=" * 64)
        print("CHECK 4 -- read speed (the likely bottleneck)")
        print("=" * 64)
        print(f"  frames are stored {args.chunk_rows} to a compressed block")
        print(f"  (~{args.chunk_rows * 224 * 224 * 3 / 1e6:.0f} MB uncompressed each).")
        print("  Reading ONE frame costs decompressing the whole block.")
        print()
        print(f"  timing {args.speed_batches} batches of {args.batch_size} clips, "
              "both ways ...")
        sp = measure_read_speed(args.h5, index, batch_size=args.batch_size,
                                n_batches=args.speed_batches,
                                chunk_rows=args.chunk_rows)
        for mode in ("random", "block"):
            m = sp[mode]
            print(f"    {mode:7s}: {m['clips_per_sec']:8.1f} clips/sec   "
                  f"({m['sec_per_batch']:.2f} s per batch of {args.batch_size})")
        print(f"    block sampling is {sp['speedup']:.1f}x "
              f"{'faster' if sp['speedup'] > 1 else 'SLOWER'}")

        # --- what this means in hours -------------------------------------
        print()
        print("  PROJECTED TRAINING TIME (data loading only, single process):")
        train_clips = int(len(index.starts) * 0.9)     # train_split: 0.9
        for epochs, label in [(10, "paper App. E"), (100, "repo config")]:
            for mode in ("random", "block"):
                cps = sp[mode]["clips_per_sec"]
                hours = train_clips * epochs / cps / 3600
                print(f"    {label:13s} {epochs:3d} epochs, {mode:6s}: "
                      f"{hours:7.1f} h")
        print()
        print("  READ THIS:")
        print("  * This is DATA LOADING ALONE on this machine's CPU. It is a")
        print("    FLOOR: real training also runs the model.")
        print("  * If these hours already exceed the GPU-side estimate, the data")
        print("    path is the bottleneck and a faster GPU will NOT help.")
        print("  * Colab gives ~2 CPU cores; multiple loader workers help, but")
        print("    not without limit. Divide by ~2, not by 8.")
        print("  * BLOCK SAMPLING IS A DEVIATION, not a free win: clips in a")
        print("    batch become time-neighbours instead of independent draws.")
        print("    SIGReg is computed across the batch, so this changes what it")
        print("    measures. Record it as a decision if used.")


if __name__ == "__main__":
    main()
