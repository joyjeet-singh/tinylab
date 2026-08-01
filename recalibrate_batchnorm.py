"""
recalibrate_batchnorm.py -- repair a checkpoint's BatchNorm running statistics
so that evaluation mode measures the model instead of measuring stale
normalisation.

THE PROBLEM
-----------
Both projector and pred_proj carry a BatchNorm1d whose running variance sits at
~1e-4. In evaluation mode the layer divides by sqrt(running_var) ~ 0.01, so any
drift between the stored mean and the current activations is amplified about a
hundredfold, and squared error inflates that by another two orders of magnitude.
Measured on our checkpoints, evaluation-mode loss exceeds training-mode loss by
5x (Run 0) and 312x (phase2) on the SAME held-out clips, with no generalisation
gap at all.

THE FIX
-------
"Precise BN": reset the running statistics, then push many batches through the
model in TRAINING mode with momentum=None, which makes PyTorch accumulate a
cumulative average rather than an exponentially-weighted one. Weights are never
touched -- no optimiser, no gradients. The result is a model whose evaluation
mode reflects the same normalisation its training mode used.

WHY NOT JUST PLAN IN TRAINING MODE
-----------------------------------
Training-mode BatchNorm normalises by BATCH statistics, so a model's prediction
for one CEM candidate would depend on which other candidates happened to share
its batch. That is not a world model; it is a batch-dependent function.
Recalibration is the correct repair.

WHAT IT DOES
------------
  1. measures the eval/train gap on held-out clips BEFORE
  2. resets and recalibrates over --batches training batches
  3. measures the gap AFTER, and reports the running statistics either side
  4. saves <ckpt stem>_recal.pt if the gap closed, leaving the original intact

Then re-run the existing tools against the recalibrated checkpoint:
    python3 verify_phase2_driving.py --run <run> --ckpt ckpt_best_recal.pt
    python3 realenv_r2_planner_eval.py --run <run> ...

Usage:
    python3 recalibrate_batchnorm.py --run runs/<run dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="defaults to ckpt_best.pt, else ckpt.pt")
    ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
    ap.add_argument("--batches", type=int, default=200,
                    help="training batches to accumulate statistics over")
    ap.add_argument("--measure-batches", type=int, default=8,
                    help="held-out batches for the before/after measurement")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    import tworoom_data as td
    from toy_model import ToyJEPA
    from toy_sigreg import SIGReg, lewm_loss

    run_dir = Path(args.run)
    mf = json.loads((run_dir / "manifest.json").read_text())
    cfg = mf["config"]
    m, d, t = cfg["model"], cfg["data"], cfg["training"]

    conv = mf.get("loader_convention") or {}
    for name in ("dense_actions", "imagenet_pixels", "zscore_actions"):
        if hasattr(td, name.upper()):
            setattr(td, name.upper(), bool(conv.get(name, False)))
    print(f"loader convention: "
          f"{ {k: bool(conv.get(k, False)) for k in ('dense_actions','imagenet_pixels','zscore_actions')} }")

    model = ToyJEPA(embed_dim=m["embed_dim"], action_dim=m["action_dim"],
                    history_size=m["history_size"], depth=m["depth"],
                    heads=m["heads"], dim_head=m["dim_head"],
                    mlp_dim=m["mlp_dim"], proj_hidden=m["proj_hidden"],
                    dropout=m["dropout"], enc_width=m["enc_width"],
                    encoder=m.get("encoder", "cnn"),
                    img_size=m.get("img_size", 32),
                    patch_size=m.get("patch_size", 4),
                    enc_depth=m.get("enc_depth", 12),
                    enc_heads=m.get("enc_heads", 3))
    ck_name = args.ckpt or ("ckpt_best.pt"
                            if (run_dir / "ckpt_best.pt").exists() else "ckpt.pt")
    ck = torch.load(run_dir / ck_name, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.requires_grad_(False)
    print(f"checkpoint {ck_name} (epoch field {ck.get('epoch')})")

    bns = [(n, mod) for n, mod in model.named_modules()
           if isinstance(mod, torch.nn.BatchNorm1d)]
    if not bns:
        raise SystemExit("no BatchNorm1d in this model -- nothing to "
                         "recalibrate")
    print(f"{len(bns)} BatchNorm1d layers: {', '.join(n for n, _ in bns)}")

    # ---- data -----------------------------------------------------------
    h5 = str(Path(args.h5).expanduser())
    spec = td.ClipSpec(history=m["history_size"], frameskip=d["frameskip"])
    index = td.TwoRoomIndex(h5, spec)
    ds = td.TwoRoomClips(h5, index)
    n = len(getattr(index, "starts", []))
    if n == 0:
        raise SystemExit("could not read the clip index size")
    rng = np.random.default_rng(d.get("data_seed", 42))
    perm = rng.permutation(n)
    cut = int(n * d["train_split"])
    train_idx, val_idx = perm[:cut], perm[cut:]
    bs = t["batch_size"]
    sig = SIGReg(**cfg["loss"]["sigreg_kwargs"])
    print(f"{len(train_idx):,} training clips, {len(val_idx):,} held out, "
          f"batch {bs}")

    def make_batch(picks):
        items = [ds[int(p)] for p in picks]
        return {"pixels": torch.tensor(np.stack([i["pixels"] for i in items])),
                "action": torch.tensor(np.stack([i["action"] for i in items]))}

    @torch.no_grad()
    def measure(mode):
        model.train() if mode == "train" else model.eval()
        tot, k = 0.0, 0
        for s in range(0, args.measure_batches * bs, bs):
            picks = val_idx[s:s + bs]
            if len(picks) < 2:
                break
            out = lewm_loss(model, model.encode(make_batch(picks)), sig,
                            ctx_len=m["history_size"], n_preds=t["num_preds"],
                            lambd=cfg["loss"]["sigreg_weight"])
            tot += float(out["pred_loss"])
            k += 1
        return tot / max(k, 1)

    def stats_line(tag):
        for name, mod in bns:
            print(f"    {tag:<7} {name:<20} mean |{mod.running_mean.abs().mean():.4f}|"
                  f"  var {mod.running_var.mean():.6f}"
                  f"  (min {mod.running_var.min():.6f})"
                  f"  batches {int(mod.num_batches_tracked)}")

    # ---- before ---------------------------------------------------------
    print("\nBEFORE")
    ev0, tr0 = measure("eval"), measure("train")
    print(f"  held-out pred_loss:  eval {ev0:.4f}   train {tr0:.4f}   "
          f"gap {ev0 / max(tr0, 1e-12):.1f}x")
    stats_line("before")

    # ---- recalibrate ----------------------------------------------------
    print(f"\nRECALIBRATING over {args.batches} training batches "
          f"({args.batches * bs:,} clips)")
    print("  momentum=None -> cumulative average, i.e. the exact mean and")
    print("  variance over the batches seen. Weights are not touched.")
    saved_mom = []
    for _, mod in bns:
        saved_mom.append(mod.momentum)
        mod.reset_running_stats()
        mod.momentum = None
    model.train()
    picks_all = rng.permutation(train_idx)[:args.batches * bs]
    with torch.no_grad():
        for i in range(0, len(picks_all), bs):
            picks = picks_all[i:i + bs]
            if len(picks) < 2:
                break
            b = make_batch(picks)
            out = model.encode(b)
            # predict() drives pred_proj, which encode() alone does not touch
            model.predict(out["emb"][:, :m["history_size"]],
                          out["act_emb"][:, :m["history_size"]])
            done = i // bs + 1
            if done % max(1, args.batches // 10) == 0:
                print(f"    {done}/{args.batches} batches", flush=True)
    for (_, mod), mom in zip(bns, saved_mom):
        mod.momentum = mom

    # ---- after ----------------------------------------------------------
    print("\nAFTER")
    ev1, tr1 = measure("eval"), measure("train")
    print(f"  held-out pred_loss:  eval {ev1:.4f}   train {tr1:.4f}   "
          f"gap {ev1 / max(tr1, 1e-12):.1f}x")
    stats_line("after")

    # ---- verdict --------------------------------------------------------
    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    print(f"  eval-mode loss  {ev0:.4f} -> {ev1:.4f}   "
          f"({(1 - ev1 / max(ev0, 1e-12)) * 100:+.1f}%)")
    print(f"  gap             {ev0 / max(tr0, 1e-12):.1f}x -> "
          f"{ev1 / max(tr1, 1e-12):.1f}x")
    print(f"  train-mode loss {tr0:.4f} -> {tr1:.4f}   "
          f"(should be ~unchanged; weights were not touched)")

    if abs(tr1 - tr0) > 0.15 * max(tr0, 1e-12):
        print("\n  WARNING: the training-mode loss moved by more than 15%.")
        print("  Nothing should have changed it. Do not use this checkpoint;")
        print("  send me this output.")
        return

    gap_after = ev1 / max(tr1, 1e-12)
    if gap_after < 1.5:
        out = Path(args.out or (run_dir / (Path(ck_name).stem + "_recal.pt")))
        ck["model"] = model.state_dict()
        ck["recalibrated"] = {"batches": args.batches, "batch_size": bs,
                              "gap_before": ev0 / max(tr0, 1e-12),
                              "gap_after": gap_after,
                              "eval_before": ev0, "eval_after": ev1}
        torch.save(ck, out)
        print(f"\n  GAP CLOSED. Wrote {out}")
        print("  The original checkpoint is untouched. Next:")
        print(f"    python3 verify_phase2_driving.py --run {run_dir} "
              f"--ckpt {out.name}")
        print("  and if that says GO, the planning evaluation.")
    elif gap_after < 0.5 * (ev0 / max(tr0, 1e-12)):
        print(f"\n  PARTIAL: the gap fell substantially but is still "
              f"{gap_after:.1f}x.")
        print("  Try --batches 1000. If it plateaus above ~2x, stale running")
        print("  statistics are not the whole story and we should look again.")
    else:
        print(f"\n  NOT CLOSED ({gap_after:.1f}x). Stale statistics are not the")
        print("  explanation, or the recalibration did not reach these layers.")
        print("  Nothing was saved. Send me this output.")


if __name__ == "__main__":
    main()
