"""Why does the temporal head beat the position probe at planning?

The temporal head predicts SPATIAL distance worse than the position probe
(r 0.819 against 0.990) yet plans better (98.0% against 88.0% at offset 100).
The obvious explanation is that spatial distance is the wrong target: TwoRoom
has a dividing wall, so two states can be close in space and far in
reachability, and only the temporal head can know that.

This tests it. Pairs are matched on true spatial distance and split by whether
they lie in the same room or on opposite sides of the wall. If the hypothesis
holds, the temporal head assigns a larger cost to cross-wall pairs at the same
spatial separation, and the position probe cannot -- by construction it sees
only Euclidean distance.

    ./.venv/bin/python followup_wall_reachability.py --run <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="ckpt_best_recal.pt")
ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
ap.add_argument("--head", default="followup/temporal_head_phase2.pt")
ap.add_argument("--episodes", type=int, default=60)
ap.add_argument("--per-episode", type=int, default=16)
ap.add_argument("--wall-x", type=float, default=111.2,
                help="wall position, detected from the data in the "
                     "reproduction's own wall analysis")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default="followup/wall_reachability.txt")
args = ap.parse_args()

import h5py
import hdf5plugin  # noqa: F401
import stable_worldmodel  # noqa: F401
import torch
import torch.nn as nn
from toy_model import ToyJEPA
from toy_plan import frame_to_tensor
from dense_action_adapter import wrap_if_needed

run_dir = Path(args.run)
cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
m = cfg["model"]
model = ToyJEPA(embed_dim=m["embed_dim"], action_dim=m["action_dim"],
                history_size=m["history_size"], depth=m["depth"],
                heads=m["heads"], dim_head=m["dim_head"], mlp_dim=m["mlp_dim"],
                proj_hidden=m["proj_hidden"], dropout=m["dropout"],
                enc_width=m["enc_width"], encoder=m.get("encoder", "cnn"),
                img_size=m.get("img_size", 32), patch_size=m.get("patch_size", 4),
                enc_depth=m.get("enc_depth", 12), enc_heads=m.get("enc_heads", 3))
ck = torch.load(run_dir / args.ckpt, map_location="cpu")
model.load_state_dict(ck.get("model") or ck["state_dict"])
model.eval()
model = wrap_if_needed(model, run_dir, str(Path(args.h5).expanduser()),
                       frameskip=cfg["data"]["frameskip"])


class TemporalHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 1), nn.Softplus())

    def forward(self, za, zb):
        return 0.5 * (self.net(torch.cat([za, zb], -1))
                      + self.net(torch.cat([zb, za], -1))).squeeze(-1)


blob = torch.load(args.head, map_location="cpu")
head = TemporalHead(blob["dim"])
head.load_state_dict(blob["state_dict"])
head.eval()

rng = np.random.default_rng(args.seed)
with h5py.File(str(Path(args.h5).expanduser()), "r") as f:
    ln = np.asarray(f["ep_len"][:]); off = np.asarray(f["ep_offset"][:])
    eps = rng.choice(np.where(ln > 100)[0], size=args.episodes, replace=False)
    Z, POS = [], []
    for e in eps:
        o, L = int(off[e]), int(ln[e])
        idx = np.sort(rng.choice(L, size=min(args.per_episode, L), replace=False))
        px = np.asarray(f["pixels"][o + idx[0]: o + idx[-1] + 1])
        pos = np.asarray(f["pos_agent"][o + idx[0]: o + idx[-1] + 1])
        for j in idx - idx[0]:
            with torch.no_grad():
                Z.append(model.encode(
                    {"pixels": frame_to_tensor(px[j]).unsqueeze(0)})["emb"][:, 0])
            POS.append(pos[j])
Z = torch.cat(Z, 0)
POS = np.asarray(POS, dtype=np.float32)
print(f"encoded {len(Z)} frames")

# all pairs, across episodes as well -- this is about geometry, not trajectories
iu = np.triu_indices(len(Z), k=1)
a, b = iu
true_d = np.linalg.norm(POS[a] - POS[b], axis=-1)
side_a = POS[a][:, 0] > args.wall_x
side_b = POS[b][:, 0] > args.wall_x
cross = side_a != side_b

with torch.no_grad():
    tmp = head(Z[a], Z[b]).numpy()
lat = torch.norm(Z[a] - Z[b], dim=-1).numpy()

lines = []
def say(s=""):
    print(s); lines.append(s)

say("=" * 72)
say("REACHABILITY vs PROXIMITY -- does the temporal head see the wall?")
say("=" * 72)
say(f"wall at x = {args.wall_x}; {len(true_d):,} pairs, "
    f"{cross.sum():,} cross-wall\n")
say("Pairs matched on TRUE SPATIAL distance, split by whether they cross the")
say("wall. A cost that measures reachability should charge more to cross the")
say("wall at the same spatial separation; Euclidean position distance cannot.\n")
say(f"  {'true dist':>12} {'pairs s/c':>12} | {'temporal head':>22} | {'latent L2':>18}")
say(f"  {'':>12} {'':>12} | {'same':>9} {'cross':>7} {'ratio':>4} | "
    f"{'same':>7} {'cross':>6} {'ratio':>3}")
rows = []
for lo, hi in ((20, 40), (40, 60), (60, 80), (80, 100)):
    msk = (true_d >= lo) & (true_d < hi)
    s_, c_ = msk & ~cross, msk & cross
    if s_.sum() < 20 or c_.sum() < 20:
        continue
    ts, tc = tmp[s_].mean(), tmp[c_].mean()
    ls, lc = lat[s_].mean(), lat[c_].mean()
    rows.append((lo, hi, ts, tc, ls, lc))
    say(f"  {lo:>5}-{hi:<6} {s_.sum():>6}/{c_.sum():<5} | "
        f"{ts:>9.1f} {tc:>7.1f} {tc/ts:>4.2f} | {ls:>7.2f} {lc:>6.2f} {lc/ls:>3.2f}")

say("\n  VERDICT")
if rows:
    tr = np.mean([r[3] / r[2] for r in rows])
    lr = np.mean([r[5] / r[4] for r in rows])
    say(f"  mean cross/same ratio, temporal head : {tr:.2f}")
    say(f"  mean cross/same ratio, latent L2     : {lr:.2f}")
    if tr > 1.15:
        say(f"\n  At matched spatial distance the temporal head charges "
            f"{100*(tr-1):.0f}% more to")
        say("  cross the wall. It has learned reachability, not proximity --")
        say("  which is why it outplans a cost built from true position, and")
        say("  why predicting spatial distance WORSE makes it a better")
        say("  planning objective.")
    else:
        say("\n  No meaningful wall penalty; the planning advantage over the")
        say("  position probe needs another explanation.")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text("\n".join(lines) + "\n")
print(f"\nwrote {args.out}")
