"""Can a decoded-position cost replace latent L2 as the planning objective?

followup_latent_metric.py showed the planner's objective -- squared L2 between
latent embeddings -- saturates by ~80 units and inverts past ~120, which is
where offset-100 planning overshoots.

The information is not missing: position decodes from a single embedding at
R^2 0.9971. So the fix is a metric, not a better model. This fits a ridge
probe on frozen embeddings and asks whether distance between DECODED positions
stays monotone in true distance where latent L2 does not.

If it does, CEM can be re-pointed at that cost with no retraining of the world
model, and long-horizon planning should recover.

    ./.venv/bin/python followup_probe_metric.py --run <dir> --ckpt <name>
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
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default="followup/probe_metric.txt")
args = ap.parse_args()

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401
import stable_worldmodel  # noqa: F401
import torch
from toy_model import ToyJEPA
from toy_plan import frame_to_tensor
from dense_action_adapter import wrap_if_needed

if args.authors_spec:
    from authors_adapter import load_authors_model
    model = load_authors_model(args.authors_spec)
    model.eval()
    run_dir = Path(args.authors_spec)
    cfg = None
else:
    run_dir = Path(args.run)
    cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
if cfg is not None:
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
    pos = np.asarray(f["pos_agent"][:400000], dtype=np.float32)
P = pos[rng.choice(len(pos), size=args.n, replace=False)]


@torch.no_grad()
def embed(p):
    img = u._render_frame(agent_pos=torch.tensor(p, dtype=torch.float32))
    img = img.cpu().numpy().transpose(1, 2, 0)
    return model.encode({"pixels": frame_to_tensor(img).unsqueeze(0)})["emb"][:, 0]


Z = torch.cat([embed(p) for p in P], 0).numpy()
print(f"encoded {len(P)} positions")

# ---- ridge probe, fit on a train split only ---------------------------
ntr = int(0.7 * len(P))
Ztr, Ptr, Zte, Pte = Z[:ntr], P[:ntr], Z[ntr:], P[ntr:]
A = np.hstack([Ztr, np.ones((len(Ztr), 1))])
W = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ Ptr)
pred = np.hstack([Zte, np.ones((len(Zte), 1))]) @ W
ss_res = ((Pte - pred) ** 2).sum()
ss_tot = ((Pte - Pte.mean(0)) ** 2).sum()
r2 = 1 - ss_res / ss_tot
mae = np.abs(Pte - pred).mean()

lines = []
def say(s=""):
    print(s)
    lines.append(s)

say("=" * 68)
say(f"DECODED-POSITION COST vs LATENT L2  --  {run_dir.name} ({args.ckpt})")
say("=" * 68)
say(f"{len(P)} positions, ridge probe fit on {ntr}, evaluated on {len(Pte)}\n")
say(f"  probe position R^2 (held out) : {r2:.4f}")
say(f"  mean absolute error           : {mae:.2f} arena units\n")

iu = np.triu_indices(len(Pte), k=1)
true_d = np.linalg.norm(Pte[:, None] - Pte[None], axis=-1)[iu]
lat_d = np.linalg.norm(Zte[:, None] - Zte[None], axis=-1)[iu]
dec_d = np.linalg.norm(pred[:, None] - pred[None], axis=-1)[iu]

say(f"  Pearson r vs true distance, latent L2        : "
    f"{np.corrcoef(true_d, lat_d)[0,1]:.4f}")
say(f"  Pearson r vs true distance, decoded position : "
    f"{np.corrcoef(true_d, dec_d)[0,1]:.4f}")

say(f"\n  {'true dist':>13}  {'pairs':>6}  {'latent L2':>10}  {'decoded':>9}")
edges = [0, 20, 40, 60, 80, 100, 120, 150, 300]
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    msk = (true_d >= lo) & (true_d < hi)
    if msk.sum() < 10:
        continue
    rows.append((lo, hi, lat_d[msk].mean(), dec_d[msk].mean()))
    say(f"  {lo:>4}-{hi:<7}  {msk.sum():>6}  {rows[-1][2]:>10.2f}  {rows[-1][3]:>9.2f}")

lat_mono = all(b[2] > a[2] for a, b in zip(rows, rows[1:]))
dec_mono = all(b[3] > a[3] for a, b in zip(rows, rows[1:]))
say("\n  VERDICT")
say(f"  latent L2 monotone in true distance          : {lat_mono}")
say(f"  decoded-position distance monotone           : {dec_mono}")
far = [r for r in rows if r[0] >= 80]
if len(far) >= 2:
    say(f"  spread across the far bands ({far[0][0]}-{far[-1][1]} units):")
    say(f"    latent L2        {(far[-1][2]-far[0][2])/far[0][2]*100:+7.1f}%")
    say(f"    decoded position {(far[-1][3]-far[0][3])/far[0][3]*100:+7.1f}%")
if dec_mono and not lat_mono:
    say("\n  The world model already encodes what the planner needs; the "
        "objective\n  is what fails. Re-pointing CEM at the decoded-position "
        "cost requires no\n  retraining.")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text("\n".join(lines) + "\n")
print(f"\nwrote {args.out}")
