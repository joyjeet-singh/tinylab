"""
eval_rollout_horizon.py -- does the predictor stay good over the horizon that
planning actually needs? In-domain, on real validation clips, with the true
recorded actions.

WHY
---
Check A validated ONE-step prediction in-domain (0.83x a real step for
ckpt_best). But CEM imagines 5 steps ahead, feeding its own predictions back
in. This measures exactly that: encode a real clip, take the first
history_size latents, roll the predictor forward `horizon` steps
autoregressively under the clip's own recorded actions, and compare each
imagined latent to the encoder's true latent for that frame.

The imagination mechanics mirror toy_plan/dump_latents (`_imagine_rollout`):
predictions are appended and the context window slides, so errors compound
the same way they would inside the planner -- but the inputs stay in the
training domain, so this isolates dynamics quality from the eval-world gap.

A "static" baseline is reported alongside: the error you'd get by predicting
that nothing moves (last seen latent). Imagination is only useful to a
planner where it beats that.

Notes on the clip set: clips here span history+horizon frames, so the index
is larger-span than training's and is re-split with the same split function
and data seed. This is an evaluation of dynamics quality, not a training-set
holdout guarantee.

Run from the tinylab folder (venv active):
    python3 eval_rollout_horizon.py --run runs/<run_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tworoom_data import ClipSpec, TwoRoomClips, TwoRoomIndex
from toy_model import ToyJEPA
from train_toy_lewm import make_batch, split_indices


def _build_model(m: dict) -> ToyJEPA:
    return ToyJEPA(embed_dim=m["embed_dim"], action_dim=m["action_dim"],
                   history_size=m["history_size"], depth=m["depth"],
                   heads=m["heads"], dim_head=m["dim_head"], mlp_dim=m["mlp_dim"],
                   proj_hidden=m["proj_hidden"], dropout=m["dropout"],
                   enc_width=m["enc_width"],
                   encoder=m.get("encoder", "cnn"),
                   img_size=m.get("img_size", 32),
                   patch_size=m.get("patch_size", 4),
                   enc_depth=m.get("enc_depth", 12),
                   enc_heads=m.get("enc_heads", 3))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--h5", default="~/Downloads/tworoom.h5")
    p.add_argument("--horizon", type=int, default=5,
                   help="model steps to imagine; 5 = the CEM planning horizon")
    p.add_argument("--max-batches", type=int, default=4)
    p.add_argument("--ckpt", default="ckpt_best.pt")
    args = p.parse_args()

    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
    d, t, m = cfg["data"], cfg["training"], cfg["model"]
    HS, K = m["history_size"], args.horizon
    h5_path = str(Path(args.h5).expanduser())

    spec = ClipSpec(history=HS, num_preds=K, frameskip=d["frameskip"])
    index = TwoRoomIndex(h5_path, spec)
    ds = TwoRoomClips(h5_path, index)
    _, val_idx = split_indices(len(index.starts), d["train_split"], d["data_seed"])

    ckpt = run_dir / args.ckpt
    if not ckpt.exists():
        ckpt = run_dir / "ckpt.pt"
    model = _build_model(m)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    print("=" * 70)
    print(f"IN-DOMAIN ROLLOUT, HORIZON {K}  --  {run_dir.name}  ({ckpt.name})")
    print("=" * 70)
    bs = t["batch_size"]
    n_used = min(len(val_idx), args.max_batches * bs)
    print(f"real validation clips: first {n_used} "
          f"({args.max_batches} batches of {bs}); actions = the recorded ones")

    err = np.zeros(K); step = np.zeros(K); static = np.zeros(K); nb = 0
    for s in range(0, n_used, bs):
        picks = val_idx[s:s + bs]
        if len(picks) < 2:
            break
        batch = make_batch(ds, picks)
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        out = model.encode(batch)
        emb = out["emb"]                                   # (B, HS+K, D) truth
        if emb.size(1) < HS + K:
            continue
        act_emb = model.action_encoder(batch["action"])    # (B, HS+K, D)

        imag = emb[:, :HS].clone()
        for k in range(K):
            pos = HS + k
            pred = model.predict(imag[:, pos - HS:pos],
                                 act_emb[:, pos - HS:pos])[:, -1:]
            imag = torch.cat([imag, pred], dim=1)
            err[k] += (imag[:, pos] - emb[:, pos]).norm(dim=-1).mean().item()
            step[k] += (emb[:, pos] - emb[:, pos - 1]).norm(dim=-1).mean().item()
            static[k] += (emb[:, pos] - emb[:, HS - 1]).norm(dim=-1).mean().item()
        nb += 1
        print(f"  batch {nb} done", flush=True)

    if nb == 0:
        print("no full-length batches available; nothing measured.")
        return
    err, step, static = err / nb, step / nb, static / nb

    print(f"\n{'horizon':>8} {'imagined err':>13} {'real step':>10} "
          f"{'err/step':>9} {'static err':>11} {'err/static':>11}")
    for k in range(K):
        print(f"{k+1:>8} {err[k]:>13.3f} {step[k]:>10.3f} "
              f"{err[k]/step[k]:>9.3f} {static[k]:>11.3f} "
              f"{err[k]/static[k]:>11.3f}")

    print("\nHOW TO READ")
    print("  horizon 1 should match Check A / the train log (internal control).")
    print("  err/static < 1 : imagining beats assuming the world froze -- the")
    print("    prediction carries usable information at that horizon.")
    print("  err/static ~ 1 : by that horizon the rollout knows no more than")
    print("    'nothing moved'; a planner gains nothing from imagining that far.")
    print("  err/step is the same scale-free ratio Check A used, per horizon.")
    ds.close()


if __name__ == "__main__":
    main()
