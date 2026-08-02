"""
recalibrate_authors_ckpt.py -- close the last inconsistency in Table 3.

WHY
---
Both of our checkpoints in the paper's planning table are BatchNorm-recalibrated
(§4.3). The authors' released checkpoint is not, because it loads through a
different path (`authors_adapter`) and the recalibration tool builds our model
class directly. A table in which two of three rows received a treatment the
third did not is a reviewer's first question.

The indirect evidence says their statistics are fine: the driving-spec
measurement gave their checkpoint 0.410 in evaluation mode, whereas our
corrected checkpoint read 2.632 in evaluation mode and 0.116 after
recalibration — a 22x inflation. Theirs does not look inflated. But that is
inference, and the whole point of §4.3 is that this particular inference is
exactly the one that misled us for a week.

WHAT THIS DOES
--------------
  1. builds the authors' checkpoint through the same adapter the planner uses
  2. reports whether it contains BatchNorm at all, and its running statistics
  3. measures the evaluation/training gap on held-out clips, driven with the
     encoding the driving spec settled on
  4. if the gap is material, recalibrates (precise BN, weights untouched) and
     re-measures; if it is not, says so and writes nothing

THE MEASUREMENT
---------------
We cannot use their training objective — we do not have it — so the gap is
measured on one-step latent prediction error, the same quantity
`verify_phase2_driving.py` reports. That is sufficient: the question is whether
evaluation mode and training mode disagree on the same inputs, and any forward
quantity answers it.

A gap near 1.0 means their released statistics are calibrated and Table 3 needs
only a footnote recording that we checked. A large gap means their 84.0% and
12.0% were measured through the same artifact we found in our own runs, and both
must be re-measured after recalibration.

Usage:
    python3 recalibrate_authors_ckpt.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="authors_driving_spec.json")
    ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
    ap.add_argument("--lewm", default=str(Path.home() / "le-wm"))
    ap.add_argument("--samples", type=int, default=256,
                    help="clips for the before/after measurement")
    ap.add_argument("--batches", type=int, default=100,
                    help="batches to accumulate statistics over")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import h5py
    import hdf5plugin  # noqa: F401
    from dense_action_adapter import action_stats  # noqa: F401  (parity check)
    from authors_adapter import load_authors_model

    print("=" * 68)
    print("THE AUTHORS' RELEASED CHECKPOINT — is its evaluation mode faithful?")
    print("=" * 68)
    model = load_authors_model(args.spec, lewm=args.lewm, verbose=True)
    inner = model.inner
    HS, A, fs = model.history_size, model.action_width, args.frameskip

    bns = [(n, m) for n, m in inner.named_modules()
           if isinstance(m, torch.nn.BatchNorm1d)]
    print(f"\nBatchNorm1d layers: {len(bns)}")
    if not bns:
        print("  NONE. Their released architecture exposes no BatchNorm on this")
        print("  path, so the artifact of §4.3 cannot apply to it and no")
        print("  recalibration is possible or needed. Record that in the table")
        print("  footnote and stop here.")
        return
    for n, m in bns:
        print(f"  {n:<24} running mean |{m.running_mean.abs().mean():.4f}|  "
              f"var {m.running_var.mean():.6f} "
              f"(min {m.running_var.min():.6f})  "
              f"batches {int(m.num_batches_tracked)}")

    # ---- real clips, driven as the planner drives them -------------------
    rng = np.random.default_rng(args.seed)
    h5 = str(Path(args.h5).expanduser())
    need = fs * HS + fs
    with h5py.File(h5, "r") as f:
        off = np.asarray(f["ep_offset"][:])
        ln = np.asarray(f["ep_len"][:])
        ok = np.where(ln > need + 2)[0]
        eps = rng.choice(ok, size=args.samples + args.batches * args.batch,
                         replace=True)
        frames, means = [], []
        for ep in eps:
            s = int(off[ep]) + int(rng.integers(0, ln[ep] - need - 1))
            frames.append(np.stack([f["pixels"][s + fs * i]
                                    for i in range(HS + 1)]))
            means.append(np.stack([
                np.asarray(f["action"][s + fs * i: s + fs * (i + 1)],
                           dtype=np.float32).mean(0) for i in range(HS)]))
    frames = np.stack(frames).astype(np.float32) / 255.0
    means = np.stack(means)
    keep = ~np.isnan(means).any(axis=(1, 2))
    frames, means = frames[keep], means[keep]
    px_all = torch.from_numpy(frames).permute(0, 1, 4, 2, 3)
    a_all = torch.from_numpy(means)
    n_meas = min(args.samples, len(px_all))
    print(f"\n{len(px_all)} clean clips; {n_meas} held back for measurement")

    @torch.no_grad()
    def one_step_ratio(mode):
        """err / frozen-world baseline, on the measurement clips."""
        inner.train() if mode == "train" else inner.eval()
        errs, stats = [], []
        for i in range(0, n_meas, args.batch):
            px, a = px_all[i:i + args.batch], a_all[i:i + args.batch]
            if len(px) < 2:
                break
            emb = model.encode({"pixels": px})["emb"]
            ctx, tgt = emb[:, :HS], emb[:, HS]
            pred = model.predict(ctx, model.action_encoder(a))[:, -1]
            errs.append(torch.norm(pred - tgt, dim=-1))
            stats.append(torch.norm(ctx[:, -1] - tgt, dim=-1))
        e = torch.cat(errs).mean()
        s = torch.cat(stats).mean()
        return float(e), float(s), float(e / s)

    def report(tag):
        ev = one_step_ratio("eval")
        tr = one_step_ratio("train")
        print(f"  {tag:<8} eval  err {ev[0]:8.3f}  ratio {ev[2]:7.3f}")
        print(f"  {'':<8} train err {tr[0]:8.3f}  ratio {tr[2]:7.3f}"
              f"   GAP {ev[0] / max(tr[0], 1e-9):.2f}x")
        return ev, tr

    print("\nBEFORE")
    ev0, tr0 = report("before")
    gap0 = ev0[0] / max(tr0[0], 1e-9)

    if gap0 < 1.5:
        print("\n" + "=" * 68)
        print("VERDICT")
        print("=" * 68)
        print(f"  Gap {gap0:.2f}x — their released statistics are calibrated.")
        print("  The 84.0% and 12.0% figures were NOT measured through the")
        print("  artifact of §4.3, and Table 3 is sound as it stands.")
        print("  Add a footnote: 'the authors' checkpoint was checked and")
        print(f"  required no recalibration (evaluation/training gap "
              f"{gap0:.2f}x).'")
        print("  Nothing written.")
        return

    print(f"\nGap is {gap0:.2f}x — material. Recalibrating over "
          f"{args.batches} batches (weights untouched).")
    saved = []
    for _, m in bns:
        saved.append(m.momentum)
        m.reset_running_stats()
        m.momentum = None
    inner.train()
    with torch.no_grad():
        start = n_meas
        for b in range(args.batches):
            i = start + b * args.batch
            px, a = px_all[i:i + args.batch], a_all[i:i + args.batch]
            if len(px) < 2:
                print(f"  ran out of clips at batch {b}; using what we have")
                break
            out = model.encode({"pixels": px})
            model.predict(out["emb"][:, :HS], model.action_encoder(a))
    for (_, m), mom in zip(bns, saved):
        m.momentum = mom

    print("\nAFTER")
    ev1, tr1 = report("after")
    gap1 = ev1[0] / max(tr1[0], 1e-9)
    for n, m in bns:
        print(f"  {n:<24} running mean |{m.running_mean.abs().mean():.4f}|  "
              f"var {m.running_var.mean():.6f}")

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    print(f"  gap {gap0:.2f}x -> {gap1:.2f}x   "
          f"eval ratio {ev0[2]:.3f} -> {ev1[2]:.3f}")
    print(f"  train-mode err {tr0[0]:.3f} -> {tr1[0]:.3f}  "
          f"(should be ~unchanged)")
    if abs(tr1[0] - tr0[0]) > 0.15 * max(tr0[0], 1e-9):
        print("\n  WARNING: training-mode error moved by more than 15%. Nothing")
        print("  should have changed it. Do not use this; send me the output.")
        return
    if gap1 < 1.5:
        out = Path(args.out or "authors_ckpt_recal.pt")
        torch.save({"model": inner.state_dict(),
                    "recalibrated": {"batches": args.batches,
                                     "gap_before": gap0, "gap_after": gap1}},
                   out)
        print(f"\n  GAP CLOSED. Wrote {out}.")
        print("  Their 84.0% and 12.0% were measured through the artifact and")
        print("  MUST be re-measured on the recalibrated weights before Table 3")
        print("  is final. That is ~45 min at offset 25 and ~54 min at 100.")
    else:
        print(f"\n  NOT CLOSED ({gap1:.2f}x). Stale statistics are not the whole")
        print("  story for their checkpoint. Nothing written; send me this.")


if __name__ == "__main__":
    main()
