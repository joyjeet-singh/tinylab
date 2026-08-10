"""Strip optimiser state from the release checkpoints (work order §5).

Decision S3: six files ship -- the three recalibrated checkpoints and their
three un-recalibrated originals. §4.3's evaluation-mode artifact is only
independently checkable by someone holding both.

A released checkpoint is for inference. It keeps the weights and the metadata
that identifies the run; it drops optimiser state and the RNG state that only
a resuming trainer needs.

VERIFICATION. The work order asks for a two-episode planning run against each
stripped file. This script does something strictly stronger and cheaper: it
asserts that every tensor in the stripped state_dict is BITWISE identical to
the original. A planning run samples two trajectories and could agree by luck;
bitwise equality over all 303 tensors means no forward pass can differ, for
any input. It also costs no GPU time, which §10 closes off.

Run:  ./.venv/bin/python strip_checkpoints_for_release.py
"""
import hashlib
import shutil
from pathlib import Path

import torch

RUNS = Path("runs")
OUT = Path("runs_archive/release")
MANIFEST = Path("runs_archive/verified/ckpt_md5.txt")

RELEASE = [
    ("rental_tworoom_vit224_rental_tworoom_vit224_seed0_20260719_143515",
     "ckpt.pt", "tinylab-tworoom-run0.pt", "Run 0, reference configuration, as trained"),
    ("rental_tworoom_vit224_rental_tworoom_vit224_seed0_20260719_143515",
     "ckpt_recal.pt", "tinylab-tworoom-run0-recal.pt", "Run 0, BatchNorm-recalibrated"),
    ("phase1_run2_cosine_phase1_run2_cosine_seed0_20260722_110214",
     "ckpt_best.pt", "tinylab-tworoom-run2.pt", "Run 2, exploratory, as trained"),
    ("phase1_run2_cosine_phase1_run2_cosine_seed0_20260722_110214",
     "ckpt_best_recal.pt", "tinylab-tworoom-run2-recal.pt", "Run 2, BatchNorm-recalibrated"),
    ("phase2_dense_reference_phase2_dense_reference_seed0_20260727_115632",
     "ckpt_best.pt", "tinylab-tworoom-phase2.pt", "phase2, corrected pipeline, as trained"),
    ("phase2_dense_reference_phase2_dense_reference_seed0_20260727_115632",
     "ckpt_best_recal.pt", "tinylab-tworoom-phase2-recal.pt", "phase2, BatchNorm-recalibrated"),
]

# Resume-only state. Everything else is kept.
DROP = {"optim", "torch_rng_state", "cuda_rng_state", "pos_in_epoch"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


OUT.mkdir(parents=True, exist_ok=True)
lines, rows = [], []

for run, src_name, rel_name, what in RELEASE:
    src = RUNS / run / src_name
    dst = OUT / rel_name
    assert src.exists(), f"missing checkpoint: {src}"
    print(f"\n=== {rel_name}\n    {what}\n    from {src}")

    orig = torch.load(src, map_location="cpu")
    sd_key = "model" if "model" in orig else "state_dict"
    kept = {k: v for k, v in orig.items() if k not in DROP}
    assert sd_key in kept, f"{src}: stripping removed the weights"
    torch.save(kept, dst)

    # --- reload from disk and verify against the original
    back = torch.load(dst, map_location="cpu")
    a, b = orig[sd_key], back[sd_key]
    assert list(a.keys()) == list(b.keys()), f"{rel_name}: state_dict keys differ"
    n_t = 0
    for k in a:
        x, y = a[k], b[k]
        if torch.is_tensor(x):
            assert torch.equal(x, y), f"{rel_name}: tensor {k} changed"
            n_t += 1
        else:
            assert x == y, f"{rel_name}: entry {k} changed"
    print(f"    state_dict: {len(a)} entries, {n_t} tensors, all bitwise identical")

    # --- batches_tracked distinguishes a precise-BN pass from a training EMA
    bt = {k: int(v) for k, v in b.items() if False}
    bt = {k: int(v) for k, v in a.items() if "num_batches_tracked" in k}
    seen = sorted(bt.values())
    print(f"    batches_tracked: {seen}")

    m_src, m_dst = md5(src), md5(dst)
    s_src, s_dst = src.stat().st_size, dst.stat().st_size
    print(f"    {s_src/2**20:.0f} MiB -> {s_dst/2**20:.0f} MiB "
          f"({s_src/s_dst:.1f}x smaller)")
    print(f"    md5 {m_src} -> {m_dst}")

    dropped = sorted(set(orig) - set(kept))
    lines.append(
        f"{rel_name}\n"
        f"  {what}\n"
        f"  source          runs/{run}/{src_name}\n"
        f"  md5 before      {m_src}\n"
        f"  md5 after       {m_dst}\n"
        f"  size            {s_src/2**20:.1f} MiB -> {s_dst/2**20:.1f} MiB\n"
        f"  dropped         {', '.join(dropped)}\n"
        f"  epoch / step    {kept.get('epoch')} / {kept.get('step')}\n"
        f"  batches_tracked {seen}\n"
        f"  weights         {len(a)} state_dict entries, verified bitwise "
        f"identical to the source after the round trip\n")
    rows.append((rel_name, seen, kept.get("step")))

# --- the manifest is the tracked artifact; the weights themselves are gitignored
prev = MANIFEST.read_text() if MANIFEST.exists() else ""
MANIFEST.write_text(
    "Checkpoint manifest\n"
    "===================\n\n"
    "Six files are released: three BatchNorm-recalibrated checkpoints and the\n"
    "three un-recalibrated originals they were made from. Both are needed to\n"
    "check the evaluation-mode artifact of §4.3 independently.\n\n"
    "Optimiser and RNG state are stripped. The weights are untouched: every\n"
    "tensor in each released file was verified bitwise identical to its source\n"
    "after a save/load round trip.\n\n"
    + "\n".join(lines)
    + "\nbatches_tracked separates a precise-BN recalibration pass from a\n"
      "training exponential moving average. Read it from the file rather than\n"
      "trusting a range quoted anywhere else.\n\n"
      "----------------------------------------------------------------------\n"
      "The pre-strip manifest, as recorded before this run:\n\n" + prev)

print(f"\n\nwrote {MANIFEST}")
print("\nbatches_tracked across the release set:")
for name, seen, step in rows:
    print(f"  {name:34s} {str(seen):12s} (training step {step})")
