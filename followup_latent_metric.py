"""Is latent distance usable as a planning cost at long range?

CEM's objective is the squared L2 distance between the imagined final
embedding and the goal embedding. That is only a sensible objective if latent
distance increases with true distance across the range the planner has to
cover. The offset-100 failures overshoot -- 26 of 37 end farther from the goal
than they began, at a median 1.41x the original separation -- which is what a
saturating or non-monotone cost would produce.

This measures the relationship directly, with no planning involved: render
real recorded positions, encode them, and compare pairwise latent distance
against pairwise true distance.

    ./.venv/bin/python followup_latent_metric.py --run <dir> --ckpt <name>
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="ckpt_best_recal.pt")
ap.add_argument("--h5", default="~/Downloads/tworoom.h5")
ap.add_argument("--n", type=int, default=120, help="positions to sample")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default="followup/latent_metric.txt")
args = ap.parse_args()

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401
import stable_worldmodel  # noqa: F401
import torch
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

env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
u = env.unwrapped
env.reset(seed=args.seed)

rng = np.random.default_rng(args.seed)
with h5py.File(str(Path(args.h5).expanduser()), "r") as f:
    pos = np.asarray(f["pos_agent"][:200000], dtype=np.float32)
idx = rng.choice(len(pos), size=args.n, replace=False)
P = pos[idx]

@torch.no_grad()
def embed(p):
    img = u._render_frame(agent_pos=torch.tensor(p, dtype=torch.float32))
    img = img.cpu().numpy().transpose(1, 2, 0)
    return model.encode({"pixels": frame_to_tensor(img).unsqueeze(0)})["emb"][:, 0]

Z = torch.cat([embed(p) for p in P], 0).numpy()
print(f"encoded {len(P)} positions, embedding dim {Z.shape[1]}")

iu = np.triu_indices(len(P), k=1)
true_d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)[iu]
lat_d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)[iu]

lines = []
def say(s=""):
    print(s)
    lines.append(s)

say("=" * 68)
say(f"LATENT DISTANCE vs TRUE DISTANCE  --  {run_dir.name} ({args.ckpt})")
say("=" * 68)
say(f"{len(true_d):,} pairs from {len(P)} sampled real positions\n")
say(f"  Pearson r over all pairs        : {np.corrcoef(true_d, lat_d)[0,1]:.4f}")
say(f"  Spearman (rank) over all pairs  : "
    f"{np.corrcoef(np.argsort(np.argsort(true_d)), np.argsort(np.argsort(lat_d)))[0,1]:.4f}")

say("\n  Binned by TRUE distance -- if the planner can still tell distances")
say("  apart in a band, latent distance must keep rising across it:")
say(f"\n  {'true dist':>14}  {'pairs':>6}  {'mean latent':>11}  {'r within band':>13}")
edges = [0, 20, 40, 60, 80, 100, 120, 150, 300]
prev = None
for lo, hi in zip(edges[:-1], edges[1:]):
    msk = (true_d >= lo) & (true_d < hi)
    if msk.sum() < 10:
        continue
    mu = lat_d[msk].mean()
    r = np.corrcoef(true_d[msk], lat_d[msk])[0, 1]
    arrow = "" if prev is None else ("  rising" if mu > prev else "  FALLING")
    say(f"  {lo:>5}-{hi:<7}  {msk.sum():>6}  {mu:>11.3f}  {r:>13.3f}{arrow}")
    prev = mu

# The planning-relevant question: from a state D away, does the cost still
# point at the goal? Check that mean latent distance is monotone in true.
say("\n  VERDICT")
means = []
for lo, hi in zip(edges[:-1], edges[1:]):
    msk = (true_d >= lo) & (true_d < hi)
    if msk.sum() >= 10:
        means.append((lo, hi, lat_d[msk].mean()))
mono = all(b[2] > a[2] for a, b in zip(means, means[1:]))
say(f"  mean latent distance is {'MONOTONE' if mono else 'NOT monotone'} "
    f"in true distance across the sampled range")
far = [m for m in means if m[0] >= 80]
if len(far) >= 2:
    spread = (far[-1][2] - far[0][2]) / far[0][2] * 100
    say(f"  across the far bands ({far[0][0]}-{far[-1][1]} units) mean latent "
        f"distance changes by {spread:+.1f}%")
    say("  -- a planner minimising latent distance can only distinguish states"
        " in that\n     range to the extent this number is large.")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text("\n".join(lines) + "\n")
print(f"\nwrote {args.out}")
