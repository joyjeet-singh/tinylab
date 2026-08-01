"""
check_train_eval_gap.py -- is the "oscillating loss" a property of training, or
of how we measured it?

WHAT PROMPTED THIS
------------------
Run 0's metrics.jsonl carries 5,419 per-step training records alongside the ten
per-epoch eval records. They disagree completely:

    training pred_loss, median by epoch:  0.292 0.300 0.300 0.302 0.303
                                          0.303 0.304 0.303 0.302 0.304
    eval pred_loss, by epoch:             5.481 0.971 1.295 0.809 2.070
                                          1.426 0.490 1.384 0.700 1.458

The training loss is flat to within 1% of its mean across ten epochs. The eval
loss swings by 107% of its mean. Our paper currently describes the second series
as evidence that the predictor does not converge. That description may be wrong,
and it needs settling before §4.3 is written.

TWO CANDIDATE EXPLANATIONS
--------------------------
1. BATCHNORM. `evaluate()` calls model.eval(), so the projector's BatchNorm1d
   switches from batch statistics to running statistics and dropout turns off.
   If the running statistics lag a drifting embedding distribution, eval-mode
   loss is both higher and unstable while train-mode loss is neither. Note the
   released config specifies BatchNorm1d, so this would be a property of the
   reference architecture, not of our reimplementation.
2. GENUINE GENERALISATION GAP. Train and validation clips come from a random
   0.9/0.1 split of the same 10,000 episodes, so they are near-identically
   distributed and a 3-18x gap would be surprising -- but it must be ruled out
   rather than dismissed.

WHAT THIS MEASURES
------------------
On ONE fixed set of held-out clips, the same checkpoint scored four ways:

    eval mode,  held-out    what the training loop reported
    train mode, held-out    isolates the mode, holding the data fixed
    eval mode,  training clips
    train mode, training clips   comparable to the per-step training log

If train-mode held-out loss lands near the training log's ~0.30 while eval mode
lands near the reported ~1.4, the mode is the explanation and the oscillation is
a measurement property. If both modes are high on held-out clips and low on
training clips, it is a generalisation gap. If neither, something else is going
on and no claim about convergence should be written until it is understood.

It also reports the BatchNorm running statistics themselves, since a drifting or
mis-estimated running mean is the proposed mechanism.

Usage:
    python3 check_train_eval_gap.py --run runs/<run dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--ckpt", default=None,
                    help="defaults to ckpt_best.pt, else ckpt.pt")
    args = ap.parse_args()

    import torch
    import tworoom_data as td
    from toy_model import ToyJEPA
    from toy_sigreg import SIGReg, lewm_loss

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
    m, d, t = cfg["model"], cfg["data"], cfg["training"]

    # honour the run's own loader convention, if it recorded one
    conv = (json.loads((run_dir / "manifest.json").read_text())
            .get("loader_convention") or {})
    for name in ("dense_actions", "imagenet_pixels", "zscore_actions"):
        if hasattr(td, name.upper()):
            setattr(td, name.upper(), bool(conv.get(name, False)))
    print(f"loader convention applied: "
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
    print(f"checkpoint {ck_name}, epoch field {ck.get('epoch')}")

    h5 = str(Path(args.h5).expanduser())
    spec = td.ClipSpec(history=m["history_size"], frameskip=d["frameskip"])
    index = td.TwoRoomIndex(h5, spec)
    ds = td.TwoRoomClips(h5, index)
    n = len(getattr(index, 'starts', []))
    if n == 0:
        raise SystemExit('could not read the clip index size; TwoRoomIndex exposes .starts')
    rng = np.random.default_rng(d.get("data_seed", 42))
    perm = rng.permutation(n)
    cut = int(n * d["train_split"])
    train_idx, val_idx = perm[:cut], perm[cut:]
    bs = t["batch_size"]
    sig = SIGReg(**cfg["loss"]["sigreg_kwargs"])

    def batches(idx):
        for s in range(0, min(len(idx), args.batches * bs), bs):
            picks = idx[s:s + bs]
            if len(picks) < 2:
                break
            items = [ds[int(p)] for p in picks]
            yield {"pixels": torch.tensor(np.stack([i["pixels"] for i in items])),
                   "action": torch.tensor(np.stack([i["action"] for i in items]))}

    @torch.no_grad()
    def score(idx, mode):
        model.train() if mode == "train" else model.eval()
        tot, k = {"pred_loss": 0.0, "sigreg_loss": 0.0}, 0
        for b in batches(idx):
            out = lewm_loss(model, model.encode(b), sig,
                            ctx_len=m["history_size"],
                            n_preds=t["num_preds"],
                            lambd=cfg["loss"]["sigreg_weight"])
            tot["pred_loss"] += float(out["pred_loss"])
            tot["sigreg_loss"] += float(out["sigreg_loss"])
            k += 1
        return {a: b / max(k, 1) for a, b in tot.items()}, k

    print(f"\n{'':<16}{'held-out':>22}{'training clips':>22}")
    print(f"  {'':<14}{'pred':>10}{'sigreg':>12}{'pred':>10}{'sigreg':>12}")
    print("  " + "-" * 58)
    res = {}
    for mode in ("eval", "train"):
        v, nv = score(val_idx, mode)
        tr, nt = score(train_idx, mode)
        res[mode] = (v, tr)
        print(f"  {mode + ' mode':<14}{v['pred_loss']:>10.4f}"
              f"{v['sigreg_loss']:>12.2f}{tr['pred_loss']:>10.4f}"
              f"{tr['sigreg_loss']:>12.2f}")
    print("  " + "-" * 58)
    print(f"  ({nv} held-out batches, {nt} training batches, "
          f"{bs} clips each)")

    # ---- BatchNorm running statistics -----------------------------------
    print("\nBATCHNORM RUNNING STATISTICS (the proposed mechanism)")
    found = False
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            found = True
            rm, rv = mod.running_mean, mod.running_var
            print(f"  {name:<28} mean |{rm.abs().mean():.4f}| "
                  f"max |{rm.abs().max():.4f}|   var {rv.mean():.4f} "
                  f"(min {rv.min():.4f}, max {rv.max():.4f})   "
                  f"batches seen {int(mod.num_batches_tracked)}")
    if not found:
        print("  none found -- BatchNorm is not the explanation; "
              "look at dropout instead")

    # ---- verdict ---------------------------------------------------------
    ev_v, ev_t = res["eval"]
    tr_v, tr_t = res["train"]
    print("\n" + "=" * 62)
    print("VERDICT")
    print("=" * 62)
    mode_effect = ev_v["pred_loss"] - tr_v["pred_loss"]
    gen_gap = tr_v["pred_loss"] - tr_t["pred_loss"]
    print(f"  effect of eval MODE, holding data fixed : {mode_effect:+.4f}")
    print(f"  effect of held-out DATA, in train mode  : {gen_gap:+.4f}")
    if abs(mode_effect) > 3 * max(abs(gen_gap), 1e-6):
        print("\n  THE MODE DOMINATES. The reported oscillation is a property")
        print("  of eval-mode measurement, not of training. §4.3 must not")
        print("  describe the eval series as training instability; the training")
        print("  loss is flat. Report both series and explain the gap.")
    elif abs(gen_gap) > 3 * max(abs(mode_effect), 1e-6):
        print("\n  THE DATA DOMINATES: a genuine generalisation gap, which is")
        print("  surprising for a random split of the same episodes and worth")
        print("  a sentence of its own.")
    else:
        print("\n  NEITHER dominates. Both contribute, or something else does.")
        print("  Do not write a convergence claim until this is understood.")
    print("\n  Either way: the training loss being flat at ~0.30 across ten")
    print("  epochs is a PLATEAU, not an oscillation. That is a different")
    print("  claim from the one currently drafted, and a cleaner one.")


if __name__ == "__main__":
    main()
