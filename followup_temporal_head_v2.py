"""Temporal head trained on the distribution the planner actually scores.

v1 trained on pairs of ENCODED REAL frames. At planning time CEM scores
(IMAGINED embedding, encoded goal) -- and imagined embeddings only resemble
encoded ones to the extent the predictor is accurate.

On our phase2 checkpoint that holds: one-step error is 0.116 of a frozen-world
baseline, imagination is close to the real manifold, and the v1 head reaches
98.0% at goal offset 100. On the authors' released checkpoint one-step error
is 0.410, imagination drifts further, and the v1 head underperforms even a
linear position probe -- consistent with an MLP extrapolating badly off the
manifold it was fit on.

This trains on both kinds of pair:

  real  x real       as before
  imagined x real    context encoded from real frames, rolled forward under
                     the recorded block-mean actions exactly as the planner
                     drives it, paired against the encoded goal frame

If the mechanism is right, the mixed head should recover the authors' case.

    ./.venv/bin/python followup_temporal_head_v2.py --authors-spec <spec>
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=None)
ap.add_argument("--authors-spec", default=None)
ap.add_argument("--ckpt", default="ckpt_best_recal.pt")
ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
ap.add_argument("--episodes", type=int, default=120)
ap.add_argument("--frameskip", type=int, default=5)
ap.add_argument("--rollout", type=int, default=8, help="planner steps imagined")
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--patience", type=int, default=30)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--compare-head", default=None,
                help="a v1 head to evaluate on the same pairs")
ap.add_argument("--save", default="followup/temporal_head_v2.pt")
ap.add_argument("--out", default="followup/temporal_head_v2.txt")
args = ap.parse_args()

import h5py
import hdf5plugin  # noqa: F401
import stable_worldmodel  # noqa: F401
import torch
import torch.nn as nn
from toy_model import ToyJEPA
from toy_plan import frame_to_tensor
from dense_action_adapter import wrap_if_needed

torch.manual_seed(args.seed)

if args.authors_spec:
    from authors_adapter import load_authors_model
    model = load_authors_model(args.authors_spec)
    model.eval()
    run_dir = Path(args.authors_spec)
else:
    run_dir = Path(args.run)
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
    ck = torch.load(run_dir / args.ckpt, map_location="cpu")
    model.load_state_dict(ck.get("model") or ck["state_dict"])
    model.eval()
    model = wrap_if_needed(model, run_dir, str(Path(args.h5).expanduser()),
                           frameskip=cfg["data"]["frameskip"])

HS = model.history_size
FS = args.frameskip
rng = np.random.default_rng(args.seed)
h5 = str(Path(args.h5).expanduser())

ZA, ZB, DELTA, KIND, EPI = [], [], [], [], []

with h5py.File(h5, "r") as f:
    ln = np.asarray(f["ep_len"][:]); off = np.asarray(f["ep_offset"][:])
    eps = rng.choice(np.where(ln > 100)[0], size=args.episodes, replace=False)
    for n, e in enumerate(eps):
        o, L = int(off[e]), int(ln[e])
        span = min(L, HS * FS + args.rollout * FS + 1)
        px = np.asarray(f["pixels"][o: o + span])
        ac = np.asarray(f["action"][o: o + span], dtype=np.float32)

        with torch.no_grad():
            # context: the first HS frames, spaced by frameskip as the
            # evaluator does
            ctx_idx = [i * FS for i in range(HS)]
            ctx = torch.cat([frame_to_tensor(px[i]) for i in ctx_idx], 0).unsqueeze(0)
            emb = model.encode({"pixels": ctx})["emb"]            # (1,HS,D)
            # the actions the planner would have emitted: the block mean
            acts = []
            for p in range(HS):
                lo = p * FS
                acts.append(ac[lo: lo + FS].mean(0))
            act = torch.from_numpy(np.stack(acts)).unsqueeze(0)   # (1,HS,2)

            # every frame we will pair against, encoded for real
            goal_idx = list(range(0, span, FS))
            gz = torch.cat([model.encode(
                {"pixels": frame_to_tensor(px[i]).unsqueeze(0)})["emb"][:, 0]
                for i in goal_idx], 0)                            # (G,D)

            # roll imagination forward exactly as CEM does
            imagined, imag_frame = [], []
            for step in range(args.rollout):
                lo = (HS + step) * FS
                if lo + FS >= span:
                    break
                a = torch.from_numpy(ac[lo: lo + FS].mean(0)).view(1, 1, -1)
                act = torch.cat([act, a], dim=1)
                ae = model.action_encoder(act)
                pred = model.predict(emb[:, -HS:], ae[:, -HS:])[:, -1:]
                emb = torch.cat([emb, pred], dim=1)
                imagined.append(pred[:, 0])
                imag_frame.append(lo + FS)

        # real x real
        for a_ in range(len(goal_idx)):
            for b_ in range(a_ + 1, len(goal_idx)):
                ZA.append(gz[a_]); ZB.append(gz[b_])
                DELTA.append(abs(goal_idx[b_] - goal_idx[a_]))
                KIND.append(0); EPI.append(int(e))
        # imagined x real -- the pair CEM actually scores
        for k, zi in enumerate(imagined):
            for b_ in range(len(goal_idx)):
                ZA.append(zi[0]); ZB.append(gz[b_])
                DELTA.append(abs(goal_idx[b_] - imag_frame[k]))
                KIND.append(1); EPI.append(int(e))
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(eps)} episodes, {len(ZA):,} pairs")

ZA = torch.stack(ZA); ZB = torch.stack(ZB)
y = torch.tensor(DELTA, dtype=torch.float32)
KIND = np.asarray(KIND); EPI = np.asarray(EPI)
print(f"\n{len(y):,} pairs: {(KIND==0).sum():,} real x real, "
      f"{(KIND==1).sum():,} imagined x real")

ue = np.unique(EPI)
tr_eps = set(ue[: int(0.7 * len(ue))].tolist())
tr = np.array([x in tr_eps for x in EPI]); te = ~tr
print(f"train {tr.sum():,} | held out {te.sum():,} (disjoint episodes)")


class TemporalHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 1), nn.Softplus())

    def forward(self, za, zb):
        return 0.5 * (self.net(torch.cat([za, zb], -1))
                      + self.net(torch.cat([zb, za], -1))).squeeze(-1)


head = TemporalHead(ZA.shape[1])
opt = torch.optim.Adam(head.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=12)
Za_tr, Zb_tr, y_tr = ZA[tr], ZB[tr], y[tr]
ntr = len(y_tr)
best, best_state, patience = float("inf"), None, 0
for ep_i in range(args.epochs):
    perm = torch.randperm(ntr)
    tot = 0.0
    for k in range(0, ntr, args.batch):
        b = perm[k:k + args.batch]
        opt.zero_grad()
        loss = nn.functional.smooth_l1_loss(head(Za_tr[b], Zb_tr[b]), y_tr[b])
        loss.backward(); opt.step()
        tot += loss.item() * len(b)
    with torch.no_grad():
        v = nn.functional.l1_loss(head(ZA[te], ZB[te]), y[te]).item()
    sched.step(v)
    if v < best - 1e-3:
        best, best_state, patience = v, {k: t.clone() for k, t in
                                         head.state_dict().items()}, 0
    else:
        patience += 1
    if (ep_i + 1) % 20 == 0 or patience >= args.patience:
        print(f"  epoch {ep_i+1:>4} train {tot/ntr:7.3f} held-out {v:6.2f} best {best:6.2f}")
    if patience >= args.patience:
        print(f"  early stop at {ep_i+1}")
        break
if best_state:
    head.load_state_dict(best_state)
head.eval()
torch.save({"state_dict": head.state_dict(), "dim": ZA.shape[1]}, args.save)

lines = []
def say(s=""):
    print(s); lines.append(s)


say("=" * 70)
say(f"TEMPORAL HEAD v2 (real + imagined pairs)  --  {run_dir.name}")
say("=" * 70)
say(f"  {(KIND==0).sum():,} real x real, {(KIND==1).sum():,} imagined x real")
say(f"  held-out MAE overall            : {best:.2f} frames")
with torch.no_grad():
    for kind, label in ((0, "real x real"), (1, "imagined x real")):
        msk = te & (KIND == kind)
        if msk.sum():
            v = nn.functional.l1_loss(head(ZA[msk], ZB[msk]), y[msk]).item()
            say(f"  held-out MAE, {label:<17}: {v:.2f} frames  ({msk.sum():,} pairs)")
say("\n  The second number is the one that matters: it is the distribution CEM")
say("  scores at planning time.")

# Does a head trained ONLY on real pairs degrade on imagined ones? That is the
# mechanism this file exists to test.
if args.compare_head and Path(args.compare_head).exists():
    blob = torch.load(args.compare_head, map_location="cpu")
    v1 = TemporalHead(blob["dim"]); v1.load_state_dict(blob["state_dict"]); v1.eval()
    say(f"\n  v1 head ({args.compare_head}), trained on real pairs only,")
    say("  evaluated on these same held-out pairs:")
    with torch.no_grad():
        for kind, label in ((0, "real x real"), (1, "imagined x real")):
            msk = te & (KIND == kind)
            if msk.sum():
                v = nn.functional.l1_loss(v1(ZA[msk], ZB[msk]), y[msk]).item()
                say(f"    v1 MAE, {label:<17}: {v:.2f} frames")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text("\n".join(lines) + "\n")
print(f"\nwrote {args.out} and {args.save}")
