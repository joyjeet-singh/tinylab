"""
probe_encoder_comparison.py -- is phase2's encoder WORSE, or DIFFERENT?

THE OBSERVATION
---------------
phase2's position probe averages R² 0.9436 across its ten epochs; Run 0's
averages 0.9949, and Run 0 is the one matching the paper's ~0.996. Both used
history_size 3 on byte-identical data, so the five-point gap comes from the
three data transforms.

THE HYPOTHESIS TO TEST
----------------------
In Run 0 the predictor was fed one subsampled action to explain a five-step
displacement, so ~80% of its target was unexplainable noise. It could not
usefully constrain the encoder, leaving the encoder free to become a nearly
pure position encoder -- which a linear probe reads off perfectly.

With dense actions the predictor has real signal, so the joint objective pulls
the encoder toward features that support dynamics. Those need not be maximally
*linearly* position-decodable. If that is right, phase2's encoder has not lost
information; it has spent some of its linear structure on something else.

FOUR MEASUREMENTS, BOTH ENCODERS, IDENTICAL FRAMES
--------------------------------------------------
  1. LINEAR position probe      the headline number, on a common footing
  2. NON-LINEAR position probe  if an MLP recovers position equally well for
                                both, the information is present in both and
                                only its LINEAR accessibility differs -- which
                                is the difference between "worse" and
                                "different"
  3. EFFECTIVE RANK             how many dimensions the embedding cloud
                                actually uses. A representation encoding more
                                than position should spread over more of them
  4. ACTION DECODING            from the pair (z_t, z_{t+frameskip}), linearly
                                predict the summed action of the block. This is
                                the decisive test: a dynamics-supporting
                                encoder should make the action MORE readable,
                                and that is what the hypothesis claims was
                                bought with the probe points

CRITICAL: each encoder is fed ITS OWN pixel convention, read from its
manifest's `loader_convention`. Feeding phase2's encoder raw [0,1] pixels would
measure a domain shift, not an encoder -- the same error that produced the 46%
artifact.

Usage:
    python3 probe_encoder_comparison.py --run-a runs/<run0 dir> --run-b runs/<phase2 dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ridge_r2(X, y, lam=1.0, seed=0):
    """Held-out R² of a ridge fit, averaged over the target's columns."""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    cut = int(n * 0.8)
    tr, te = idx[:cut], idx[cut:]
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ W
    ss_res = ((yte - pred) ** 2).sum(0)
    ss_tot = ((yte - yte.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / ss_tot))


def mlp_r2(X, y, seed=0, steps=3000, hidden=256):
    """Held-out R² of a small MLP -- how much information is present at all."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    mu, sd = X[idx[:cut]].mean(0), X[idx[:cut]].std(0) + 1e-8
    Xn = (X - mu) / sd
    Xtr = torch.tensor(Xn[idx[:cut]], dtype=torch.float32)
    ytr = torch.tensor(y[idx[:cut]], dtype=torch.float32)
    Xte = torch.tensor(Xn[idx[cut:]], dtype=torch.float32)
    yte = torch.tensor(y[idx[cut:]], dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, y.shape[1]))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        b = torch.randint(0, len(Xtr), (256,))
        loss = ((net(Xtr[b]) - ytr[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    with torch.no_grad():
        pred = net(Xte)
    ss_res = ((yte - pred) ** 2).sum(0)
    ss_tot = ((yte - yte.mean(0)) ** 2).sum(0)
    return float(torch.mean(1 - ss_res / ss_tot))


def effective_rank(Z):
    """Participation ratio of the covariance spectrum: how many dimensions
    the cloud actually uses, from 1 (a line) to D (isotropic)."""
    C = np.cov((Z - Z.mean(0)).T)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 0]
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def load_encoder(run_dir):
    """The model plus the pixel convention IT was trained with."""
    import torch
    from toy_model import ToyJEPA
    from dense_action_adapter import convention_from_manifest
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
    m = cfg["model"]
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
    ck = torch.load(run_dir / ("ckpt_best.pt"
                               if (run_dir / "ckpt_best.pt").exists()
                               else "ckpt.pt"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    model.requires_grad_(False)
    return model, convention_from_manifest(run_dir), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="e.g. the Run 0 directory")
    ap.add_argument("--run-b", required=True, help="e.g. the phase2 directory")
    ap.add_argument("--label-a", default="Run 0")
    ap.add_argument("--label-b", default="phase2")
    ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
    ap.add_argument("--frames", type=int, default=4000)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--mlp-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import h5py
    import hdf5plugin  # noqa: F401
    IMEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    ISTD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    # ---- one set of frames, used for both encoders ----------------------
    rng = np.random.default_rng(args.seed)
    fs = args.frameskip
    h5 = str(Path(args.h5).expanduser())
    with h5py.File(h5, "r") as f:
        off = np.asarray(f["ep_offset"][:])
        ln = np.asarray(f["ep_len"][:])
        ok = np.where(ln > fs + 2)[0]
        eps = rng.choice(ok, size=args.frames, replace=True)
        starts = np.array([int(off[e]) + int(rng.integers(0, ln[e] - fs - 1))
                           for e in eps])
        px0 = np.stack([f["pixels"][s] for s in starts])
        px1 = np.stack([f["pixels"][s + fs] for s in starts])
        pos = np.stack([np.asarray(f["pos_agent"][s]) for s in starts])
        act = np.stack([np.asarray(f["action"][s:s + fs],
                                   dtype=np.float64).sum(0) for s in starts])
    N = len(px0)
    print("=" * 72)
    print(f"ENCODER COMPARISON on {N} identical frames")
    print("=" * 72)
    if np.isnan(act).any():
        keep = ~np.isnan(act).any(axis=1)
        px0, px1, pos, act = px0[keep], px1[keep], pos[keep], act[keep]
        print(f"  dropped {int((~keep).sum())} windows containing a NaN action")

    results = {}
    for label, run in ((args.label_a, args.run_a), (args.label_b, args.run_b)):
        model, conv, m = load_encoder(run)
        img_norm = bool(conv.get("imagenet_pixels"))
        print(f"\n--- {label} ---")
        print(f"  action_dim {m['action_dim']}, history_size "
              f"{m['history_size']}, pixels "
              f"{'ImageNet-normalised' if img_norm else 'raw [0,1]'} "
              f"({conv['_source']})")

        def embed(px_uint8):
            out = []
            for i in range(0, len(px_uint8), 32):
                x = torch.from_numpy(
                    px_uint8[i:i + 32].astype(np.float32) / 255.0
                ).permute(0, 3, 1, 2)
                if img_norm:
                    x = (x - IMEAN) / ISTD
                with torch.no_grad():
                    out.append(model.encode({"pixels": x.unsqueeze(1)})
                               ["emb"][:, 0].numpy())
            return np.concatenate(out, 0)

        Z0, Z1 = embed(px0), embed(px1)
        lin = ridge_r2(Z0, pos)
        nonlin = mlp_r2(Z0, pos, steps=args.mlp_steps)
        er = effective_rank(Z0)
        act_lin = ridge_r2(np.hstack([Z0, Z1]), act)
        act_delta = ridge_r2(Z1 - Z0, act)
        results[label] = dict(lin=lin, nonlin=nonlin, er=er, D=Z0.shape[1],
                              act=act_lin, act_d=act_delta,
                              spread=float(Z0.std(0).mean()))
        print(f"  1. position, LINEAR probe      R² {lin:.4f}")
        print(f"  2. position, MLP probe         R² {nonlin:.4f}"
              + ("   <-- BELOW the linear probe: the MLP is UNDERTRAINED, so "
                 "test 2 is unreliable. Raise --mlp-steps." if nonlin < lin - 0.005
                 else ""))
        print(f"  3. effective rank              {er:.1f} of {Z0.shape[1]} dims")
        print(f"  4. action from (z_t, z_t+1)    R² {act_lin:.4f}")
        print(f"     action from (z_t+1 − z_t)   R² {act_delta:.4f}")
        print(f"     embedding spread            {results[label]['spread']:.4f}")

    a, b = results[args.label_a], results[args.label_b]
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  {'':<28}{args.label_a:>12}{args.label_b:>12}{'Δ':>10}")
    for key, name in (("lin", "position, linear"), ("nonlin", "position, MLP"),
                      ("act", "action, linear"), ("act_d", "action, from Δz"),
                      ("er", "effective rank")):
        print(f"  {name:<28}{a[key]:>12.4f}{b[key]:>12.4f}"
              f"{b[key] - a[key]:>+10.4f}")

    lin_gap = a["lin"] - b["lin"]
    nl_gap = a["nonlin"] - b["nonlin"]
    act_gain = b["act"] - a["act"]
    print()
    if lin_gap > 0.01 and nl_gap < lin_gap * 0.5:
        print(f"  The linear gap ({lin_gap:+.4f}) largely DISAPPEARS under a")
        print(f"  non-linear probe ({nl_gap:+.4f}). The position information is")
        print(f"  present in both encoders; what differs is how linearly")
        print(f"  accessible it is. That is 'different', not 'worse'.")
    elif lin_gap > 0.01:
        print(f"  The gap survives the non-linear probe ({nl_gap:+.4f} of "
              f"{lin_gap:+.4f}).")
        print(f"  {args.label_b}'s encoder carries genuinely less position")
        print(f"  information. Report it as a cost, not a trade.")
    else:
        print("  No meaningful linear gap on this sample -- the training-log")
        print("  difference may be a probe-protocol artifact. Worth chasing.")
    print()
    if act_gain > 0.02:
        print(f"  And {args.label_b} makes the action MORE decodable "
              f"({act_gain:+.4f}).")
        print("  That is the trade the hypothesis predicted: linear structure")
        print("  spent on dynamics rather than on position.")
    elif act_gain < -0.02:
        print(f"  But {args.label_b} makes the action LESS decodable "
              f"({act_gain:+.4f}),")
        print("  which contradicts the hypothesis. Report the observation and")
        print("  drop the explanation.")
    else:
        print(f"  Action decodability is unchanged ({act_gain:+.4f}), so the")
        print("  'spent on dynamics' story is NOT supported. State the")
        print("  trade-off as an unexplained observation.")


if __name__ == "__main__":
    main()
