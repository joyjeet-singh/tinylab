# Amendment 2 — 2026-07-15: LeWM Two-Room reproduction is not feasible on free compute

Recorded before any training run. Appended, never edited into the body.

This amendment covers the LeWM Two-Room reproduction (the direction adopted
after the KV-cache probe was abandoned under A1.5's stopping rule). It records
a feasibility finding that fired before any model was written.

## A2.1 — What was verified (all four checks pass on the real 12 GB file)

Dataset: `quentinll/lewm-tworooms`, `tworoom.h5`, 12 GB extracted from a 3.2 GB
zstd archive. Frames are Blosc-compressed inside HDF5 (filter 32001), requiring
`hdf5plugin` to read — this is a hard dependency and must be in every
requirements file.

Structure (measured, not assumed): flat storage, one long reel.

| dataset | shape | dtype |
|---|---|---|
| pixels | (920809, 224, 224, 3) | uint8 |
| action | (920809, 2) | float32 |
| pos_agent / pos_target / proprio | (920809, 2) | float32 |
| ep_offset / ep_len | (10000,) | int64 / int32 |
| ep_idx / step_idx | (920809,) | int32 / int64 |

Episodes are carved out via `ep_offset` + `ep_len`. Mean episode length 92.1,
matching paper Appendix E (~92). 770,809 legal clip starts at history=3,
frameskip=5 (clip spans 16 raw frames).

**Check 1 — frame/action alignment (numeric).** Using `pos_agent` as ground
truth, cosine agreement between action[t] and the movement it should explain:

| offset | meaning | agreement |
|---|---|---|
| -1 | off-by-one (early) | +0.0542 |
| **0** | **assumed convention** | **+0.9772** |
| +1 | off-by-one (late) | +0.0440 |

PASS, decisively, over 3,994 sampled steps. The check was validated against a
synthetic file with a *deliberately planted* off-by-one and correctly reported
FAIL — so it discriminates rather than always passing.

**Check 2 — no episode boundary crossing.** 5,000 clips checked, 0 crossing.

**Check 3 — one clip end to end.** pixels (4,3,224,224) float32 in [0,1];
action (4,2) float32 in [-1, 0.778].

**Incidental findings worth keeping:**
- Actions are *displacements scaled by ~5* (magnitude-1 action → 5-pixel move),
  not velocities or accelerations. Agreement is 0.977 rather than 1.000 because
  actions clamp at ±1 and the agent hits walls.
- Image rows count downward; position coordinates count upward. A negative
  y-action renders as the dot moving *up*. Harmless for training (the model
  sees only pixels and numbers) but matters when interpreting plans. Do not
  "fix" it.

## A2.2 — The finding: the data path, not the GPU, is the bottleneck

`pixels` is chunked at **(100, 224, 224, 3)** — 100 frames per compressed block,
~15 MB uncompressed each. Reading any single frame costs decompressing its
entire block. A 4-frame clip uses ~600 KB of a ~15 MB decompression:
**~25x read amplification**, structural to how the file was written.

Measured on the 2017 Intel MacBook (single process, 12 batches of 32 clips):

| strategy | clips/sec | sec/batch (32) |
|---|---|---|
| random | 9.5 | 3.37 |
| block | 9.8 | 3.26 |

**Projected DATA LOADING ALONE** (770,809 clips x 0.9 train split):

| epochs | source | random | block |
|---|---|---|---|
| 10 | paper App. E | **203.0 h** | 196.4 h |
| 100 | repo config | 2029.7 h | 1964.0 h |

Prior GPU-side estimate for the same run: 20–60 h on a T4. **The data path is
therefore 4–10x slower than the compute it feeds.** A faster GPU does not help;
it would idle waiting on Blosc decompression.

**Block sampling failed as a mitigation: 1.0x — no speedup.** The mechanism
worked (verified: a block-sampled batch of 32 touches 1 compressed block vs 21
for random) but bought nothing, because the cost is dominated by decompressing
*at all*, not by how many distinct blocks are touched.

This is a *useful* negative: block sampling would have been a real deviation
(clips in a batch become time-neighbours rather than independent draws, and
SIGReg is computed across the batch — so it changes what the anti-collapse term
measures). Since it buys nothing, there is no tradeoff to weigh. Ruled out on
evidence, not preference.

## A2.3 — Verdict

**Free-tier Colab reproduction of LeWM Two-Room is not feasible.** Not "tight" —
I/O alone exceeds what free sessions can deliver (~100 h optimistically on
Colab's ~2 cores, against ~12 h session limits), and the bottleneck is
structural rather than fixable with better code.

Cost of learning this: ~2 days, no model written, no training loop, no week lost
to timeouts. A verified loader is retained regardless of what happens next.

## A2.4 — Options, with what each costs

| option | cost | keeps it a reproduction? |
|---|---|---|
| Rent a box (many cores, fast local disk, GPU) | ~$10–20 | **Yes** |
| Decompress once → ~138 GB raw | needs big disk | Yes, if disk exists |
| Subset (e.g. 1,000 of 10,000 episodes) | free | **No** — not the published 87% |
| Downsample + re-store smaller frames | free-ish | **No** — large deviation |

**Recommendation: rent.** ~$10–20 preserves the one property that makes this
work worth doing — that the number we get is comparable to the number they
published. Every free option breaks that comparability, and a reproduction that
isn't comparable isn't a reproduction.

**If rented, record in every manifest:** machine type, core count, disk type,
the `hdf5plugin` version, and measured clips/sec on *that* box (the 9.5 figure
is this MacBook's, not a universal constant).

## A2.5 — Still open, carried forward

- **The epochs conflict is unresolved and now expensive.** Repo config says
  `max_epochs: 100`; paper Appendix E says 10 for Two-Room. At measured speeds
  that is 203 h vs 2,030 h of loading. Decide and record *before* renting; the
  repo is the reference, the paper is the description, but a 10x compute
  difference deserves an explicit decision rather than a default.
- **The history_size conflict.** Paper says history length 1 for Two-Room; repo
  config sets `history_size: 3` globally. Unresolved. The loader currently
  follows the repo (history=3).
- **Target remains 87%** on Two-Room (LeWM), against PLDM/DINO-WM at 97–100.
  Reproducing means landing near 87, NOT near 100. A higher number means
  something different was built, not something better.
