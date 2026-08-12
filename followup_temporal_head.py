"""A learned temporal-distance head, trained without any position supervision.

The decoded-position cost repaired long-horizon planning (26.0% -> 88.0%), but
it leans on something TwoRoom happens to provide: a two-dimensional state that
a linear probe can read out of the embedding. That is not general.

This asks whether the same repair survives when the cost is learned from the
only signal a deployed system always has -- how many steps apart two observed
frames were. Training pairs are real recorded frames from the same episode,
supervised by their frame separation. `pos_agent` is used ONLY to evaluate the
resulting metric, never to train it.

If predicted steps-to-reach orders true distance where latent L2 does not, the
claim stops being "TwoRoom has a readable state" and becomes "the objective,
not the predictor, is the bottleneck".

    ./.venv/bin/python followup_temporal_head.py --run <dir> --ckpt <name>
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
ap.add_argument("--episodes", type=int, default=60)
ap.add_argument("--per-episode", type=int, default=16)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--patience", type=int, default=30)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--save", default="followup/temporal_head.pt")
ap.add_argument("--out", default="followup/temporal_head.txt")
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
    cfg = None
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

rng = np.random.default_rng(args.seed)
h5 = str(Path(args.h5).expanduser())

# ---- encode real recorded frames -------------------------------------
with h5py.File(h5, "r") as f:
    ln = np.asarray(f["ep_len"][:])
    off = np.asarray(f["ep_offset"][:])
    eligible = np.where(ln > 100)[0]
    eps = rng.choice(eligible, size=args.episodes, replace=False)

    Z, EP, FR, POS = [], [], [], []
    for e in eps:
        o, L = int(off[e]), int(ln[e])
        idx = np.sort(rng.choice(L, size=min(args.per_episode, L), replace=False))
        px = np.asarray(f["pixels"][o + idx[0]: o + idx[-1] + 1])
        pos = np.asarray(f["pos_agent"][o + idx[0]: o + idx[-1] + 1])
        for j in idx - idx[0]:
            with torch.no_grad():
                t = frame_to_tensor(px[j]).unsqueeze(0)
                Z.append(model.encode({"pixels": t})["emb"][:, 0])
            POS.append(pos[j])
        EP += [int(e)] * len(idx)
        FR += list(idx)
    print(f"encoded {len(Z)} real frames from {len(eps)} episodes")

Z = torch.cat(Z, 0)
EP = np.asarray(EP); FR = np.asarray(FR); POS = np.asarray(POS, dtype=np.float32)

# ---- pairs within an episode, supervised ONLY by frame separation ----
ii, jj = [], []
for e in np.unique(EP):
    w = np.where(EP == e)[0]
    for a in range(len(w)):
        for b in range(a + 1, len(w)):
            ii.append(w[a]); jj.append(w[b])
ii = np.asarray(ii); jj = np.asarray(jj)
delta = np.abs(FR[jj] - FR[ii]).astype(np.float32)      # the only supervision
print(f"{len(ii):,} within-episode pairs, separation {delta.min():.0f}-{delta.max():.0f} frames")

# held-out split BY EPISODE, so the head is never tested on a trajectory it saw
tr_eps = set(np.unique(EP)[: int(0.7 * len(np.unique(EP)))].tolist())
tr = np.array([EP[a] in tr_eps and EP[b] in tr_eps for a, b in zip(ii, jj)])
te = ~tr
print(f"train pairs {tr.sum():,} | held-out pairs {te.sum():,} (disjoint episodes)")


class TemporalHead(nn.Module):
    """(z_t, z_goal) -> steps to reach. Symmetric by construction."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Softplus())

    def forward(self, za, zb):
        # average both orderings so the metric is symmetric
        return 0.5 * (self.net(torch.cat([za, zb], -1))
                      + self.net(torch.cat([zb, za], -1))).squeeze(-1)


head = TemporalHead(Z.shape[1])
opt = torch.optim.Adam(head.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=12)
Za, Zb = Z[ii], Z[jj]
y = torch.from_numpy(delta)
Za_tr, Zb_tr, y_tr = Za[tr], Zb[tr], y[tr]
ntr = len(y_tr)

# Minibatch SGD with early stopping on the held-out episodes. Full-batch
# descent for a fixed step count leaves the head under-fit on larger samples
# and gives no signal about when to stop.
best, best_state, patience = float("inf"), None, 0
for ep_i in range(args.epochs):
    perm = torch.randperm(ntr)
    tot = 0.0
    for k in range(0, ntr, args.batch):
        b = perm[k:k + args.batch]
        opt.zero_grad()
        loss = nn.functional.smooth_l1_loss(head(Za_tr[b], Zb_tr[b]), y_tr[b])
        loss.backward()
        opt.step()
        tot += loss.item() * len(b)
    with torch.no_grad():
        v = nn.functional.l1_loss(head(Za[te], Zb[te]), y[te]).item()
    sched.step(v)
    if v < best - 1e-3:
        best, best_state, patience = v, {k: t.clone() for k, t in
                                         head.state_dict().items()}, 0
    else:
        patience += 1
    if (ep_i + 1) % 10 == 0 or patience >= args.patience:
        print(f"  epoch {ep_i+1:>4}  train {tot/ntr:7.3f}  "
              f"held-out MAE {v:6.2f}  best {best:6.2f}")
    if patience >= args.patience:
        print(f"  early stop at epoch {ep_i+1}; best held-out MAE {best:.2f}")
        break

if best_state is not None:
    head.load_state_dict(best_state)
head.eval()
torch.save({"state_dict": head.state_dict(), "dim": Z.shape[1]}, args.save)

# ---- evaluate the resulting METRIC against true spatial distance ------
# pos_agent appears here and nowhere in training.
with torch.no_grad():
    pred = head(Za[te], Zb[te]).numpy()
lat = torch.norm(Za[te] - Zb[te], dim=-1).numpy()
true_d = np.linalg.norm(POS[ii[te]] - POS[jj[te]], axis=-1)

lines = []
def say(s=""):
    print(s); lines.append(s)

say("=" * 70)
say(f"LEARNED TEMPORAL-DISTANCE HEAD  --  {run_dir.name}")
say("=" * 70)
say("trained on within-episode frame separation only; no position supervision\n")
say(f"  held-out pairs (disjoint episodes) : {te.sum():,}")
say(f"  held-out MAE                        : "
    f"{np.abs(pred - delta[te]).mean():.2f} frames\n")
say(f"  Pearson r vs TRUE SPATIAL distance")
say(f"    latent L2 (the current objective) : {np.corrcoef(true_d, lat)[0,1]:.4f}")
say(f"    learned temporal head             : {np.corrcoef(true_d, pred)[0,1]:.4f}")

say(f"\n  {'true dist':>13}  {'pairs':>6}  {'latent L2':>10}  {'temporal head':>14}")
edges = [0, 20, 40, 60, 80, 100, 120, 150, 300]
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    msk = (true_d >= lo) & (true_d < hi)
    if msk.sum() < 10:
        continue
    rows.append((lo, hi, lat[msk].mean(), pred[msk].mean()))
    say(f"  {lo:>4}-{hi:<7}  {msk.sum():>6}  {rows[-1][2]:>10.2f}  {rows[-1][3]:>14.2f}")

lat_mono = all(b[2] > a[2] for a, b in zip(rows, rows[1:]))
tmp_mono = all(b[3] > a[3] for a, b in zip(rows, rows[1:]))
say("\n  VERDICT")
say(f"  latent L2 monotone in true distance   : {lat_mono}")
say(f"  temporal head monotone                : {tmp_mono}")
far = [r for r in rows if r[0] >= 80]
if len(far) >= 2:
    say(f"  spread across {far[0][0]}-{far[-1][1]} units:")
    say(f"    latent L2      {(far[-1][2]-far[0][2])/far[0][2]*100:+7.1f}%")
    say(f"    temporal head  {(far[-1][3]-far[0][3])/far[0][3]*100:+7.1f}%")
if tmp_mono and not lat_mono:
    say("\n  The repair does not depend on TwoRoom's readable state. A cost")
    say("  learned from temporal separation alone orders distance where the")
    say("  embedding metric does not.")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text("\n".join(lines) + "\n")
print(f"\nwrote {args.out} and {args.save}")
