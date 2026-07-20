"""
dump_latents.py -- capture the states behind the planning result, for the
visual mechanism study (Views 1-3).

WHY THIS EXISTS
---------------
The planner's success rate is a number. To explain WHY it failed we need to
look at the latent space the planner actually searched through -- the exact
states it visited, colored by true position, plus where its imagination
diverged from reality. This script re-runs the SAME episodes toy_plan.py ran
(same seed, bit-identical model) with logging switched on, and saves the raw
ingredients the three views read.

It saves RAW latents only. Projection to 2D is a downstream PLOTTING decision
(made in the view code, per the pre-registered basis rule in prereg_views.md) --
never baked in here, so it stays reversible and honest.

Grounded in the real toy_model.py:
  encode(info)["emb"]  : (B,T,D)  -- projector(encoder(pixels)); THE latent
  predict(emb, act_emb): (B,T,D)  -- same D-space as emb; frame t -> t+1
  history_size, embed_dim on the model.

Run (on the Mac, venv active):
  python dump_latents.py --run runs/<dir> --num-eval 50 --seed 0
  python dump_latents.py --run runs/<dir> --num-eval 50 --seed 0 --random
Outputs (next to the run):
  runs/<dir>/latents_<tag>.npz      -- the arrays below
  runs/<dir>/latents_<tag>.manifest.json  -- git commit, ckpt md5, seed, config echo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from make_toy_tworoom import ToyTwoRoom
from toy_model import ToyJEPA
from toy_plan import CEMPlanner, frame_to_tensor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_model(run_dir: Path):
    cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
    m = cfg["model"]
    model = ToyJEPA(embed_dim=m["embed_dim"], action_dim=m["action_dim"],
                    history_size=m["history_size"], depth=m["depth"],
                    heads=m["heads"], dim_head=m["dim_head"], mlp_dim=m["mlp_dim"],
                    proj_hidden=m["proj_hidden"], dropout=m["dropout"],
                    enc_width=m["enc_width"],
                    encoder=m.get("encoder", "cnn"),
                    img_size=m.get("img_size", 32),
                    patch_size=m.get("patch_size", 4),
                    enc_depth=m.get("enc_depth", 12),
                    enc_heads=m.get("enc_heads", 3))
    ck = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, m


@torch.no_grad()
def _emb_of(model, img):
    """single frame -> its latent emb (D,)."""
    px = frame_to_tensor(img).unsqueeze(0)          # (1,1,3,H,W)
    return model.encode({"pixels": px})["emb"][0, 0].cpu().numpy()


# ---------------------------------------------------------------------------
# instrumented episode: same control flow as toy_plan.run_episode, logging on
# ---------------------------------------------------------------------------
@torch.no_grad()
def dump_episode(model, planner, env, img_size, start_pos, goal_pos,
                 budget, frameskip, rng, ep_id, use_random,
                 success_radius=3.0):
    """
    Mirrors toy_plan.run_episode's stepping EXACTLY, but records at each visited
    state:
      latent (D)         : emb of the frame the agent currently sees
      pos    (2)         : true (x,y) at that frame
      pred_err (float)   : || predict(...)[-1] - emb(next real frame) ||, or nan
                           for random (no plan) / final step (no next frame)
      is_doorway (bool)  : whether the agent is in the doorway band (View 2)
    Plus, for the FIRST planned step of each episode (View 3):
      imagined (H,2)     : the planner's scored rollout, decoded to (x,y) via emb->?
                           NOTE: we store imagined LATENTS (H,D); decoding to (x,y)
                           happens downstream with the position probe, per prereg.
    Returns a dict of per-step lists + episode meta.
    """
    HS = model.history_size
    rec = {"ep_id": ep_id, "latent": [], "pos": [], "pred_err": [],
           "is_doorway": [], "step_idx": [], "imagined_latents": None,
           "start": start_pos.tolist(), "goal": goal_pos.tolist()}

    # doorway band of the toy env (wall_x +/- a small margin); used only for View 2 tagging
    wall_x = env.wall_x
    def in_doorway(p):
        return abs(float(p[0]) - wall_x) < 4.0

    goal_img = env.render(goal_pos, img_size)
    goal_emb = model.encode(
        {"pixels": frame_to_tensor(goal_img).unsqueeze(0)})["emb"][:, 0]

    pos = start_pos.copy()
    hist_imgs = [env.render(pos, img_size) for _ in range(HS)]
    hist_acts = [np.zeros(2, np.float32) for _ in range(HS)]

    steps = 0
    reached = False
    first_plan_done = False
    while steps < budget:
        cur_img = env.render(pos, img_size)
        cur_emb = _emb_of(model, cur_img)

        if use_random:
            a = rng.uniform(-1, 1, 2).astype(np.float32)
            pred_err = np.nan
        else:
            px = torch.cat([frame_to_tensor(i) for i in hist_imgs[-HS:]], 0).unsqueeze(0)
            ac = torch.from_numpy(np.stack(hist_acts[-HS:])).float().unsqueeze(0)
            out = model.encode({"pixels": px, "action": ac})
            ctx_emb, ctx_act = out["emb"], ac
            seq = planner.plan(ctx_emb, ctx_act, goal_emb, rng).cpu().numpy()
            a = seq[0]

            # one-step prediction error: what the model THINKS the next frame's
            # emb will be under action a, vs the real next frame's emb.
            act_emb = model.action_encoder(
                torch.cat([ctx_act, torch.from_numpy(a[None, None]).float()], dim=1))
            pred_next = model.predict(
                torch.cat([ctx_emb, ctx_emb[:, -1:]], dim=1)[:, -HS:],
                act_emb[:, -HS:])[:, -1]                     # (1,D)
            # (computed AFTER we know the real next frame, below)

            # capture the imagined rollout ONCE per episode, on the first plan
            if not first_plan_done:
                rec["imagined_latents"] = _imagine_rollout(
                    model, planner, ctx_emb, ctx_act, seq)
                first_plan_done = True

        rec["latent"].append(cur_emb)
        rec["pos"].append(pos.copy())
        rec["is_doorway"].append(in_doorway(pos))
        rec["step_idx"].append(steps)

        # execute frameskip real steps
        moved_to = pos.copy()
        for _ in range(frameskip):
            moved_to = env.step(moved_to, a)
            steps += 1
            if np.linalg.norm(moved_to - goal_pos) < success_radius:
                reached = True
                break
            if steps >= budget:
                break

        # now compute pred_err against the REAL next frame we just moved to
        if not use_random:
            next_emb = torch.from_numpy(
                _emb_of(model, env.render(moved_to, img_size)))[None]
            rec["pred_err"].append(float((pred_next.cpu() - next_emb).norm()))
        else:
            rec["pred_err"].append(np.nan)

        pos = moved_to
        hist_imgs.append(env.render(pos, img_size))
        hist_acts.append(a.astype(np.float32))
        if reached:
            break

    rec["reached"] = bool(reached)
    rec["steps"] = int(steps)
    rec["final_dist"] = float(np.linalg.norm(pos - goal_pos))
    return rec


@torch.no_grad()
def _imagine_rollout(model, planner, ctx_emb, ctx_act, seq):
    """
    Roll the model forward under the chosen plan `seq` (H,A), returning the
    imagined LATENT sequence (H, D). Decoding to (x,y) is done downstream with
    the position probe (per prereg_views.md), NOT here.
    """
    HS = model.history_size
    emb = ctx_emb.clone()
    act = ctx_act.clone()
    out = []
    for t in range(seq.shape[0]):
        act = torch.cat([act, torch.from_numpy(seq[t][None, None]).float()], dim=1)
        act_emb = model.action_encoder(act)
        pred = model.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
        out.append(pred[0, 0].cpu().numpy())
        emb = torch.cat([emb, pred], dim=1)
    return np.stack(out)                                  # (H, D)


# ---------------------------------------------------------------------------
# driver -- mirrors toy_plan.evaluate's episode sampling EXACTLY (same seed path)
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--random", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = Path(args.run)
    model, cfg, m = _load_model(run_dir)
    env = ToyTwoRoom()
    img_size = m.get("img_size", 32)
    frameskip = cfg["data"]["frameskip"]
    planner = CEMPlanner(model, horizon=5, num_samples=300, n_steps=30, topk=30,
                         var_scale=1.0, action_dim=m["action_dim"])

    # IMPORTANT: same rng construction + same sample() logic as toy_plan.evaluate,
    # so the episodes are identical to the ones behind the reported numbers.
    rng = np.random.default_rng(args.seed)
    lo, hi = env.margin, env.size - env.margin

    def sample(on_left):
        x = (rng.uniform(lo, env.wall_x - 4) if on_left
             else rng.uniform(env.wall_x + 4, hi))
        return np.array([x, rng.uniform(lo, hi)], dtype=np.float32)

    episodes = []
    for i in range(args.num_eval):
        left = rng.random() < 0.5
        start, goal = sample(left), sample(not left)
        rec = dump_episode(model, planner, env, img_size, start, goal,
                           budget=50, frameskip=frameskip, rng=rng,
                           ep_id=i, use_random=args.random)
        episodes.append(rec)
        print(f"  ep {i+1:2d}/{args.num_eval}: "
              f"{'REACHED' if rec['reached'] else 'missed ':7s} "
              f"dist {rec['final_dist']:5.1f}  states {len(rec['latent'])}")

    # flatten per-state arrays (View 1 & 2), keep per-episode imagined rollouts (View 3)
    lat = np.concatenate([np.array(e["latent"]) for e in episodes], 0)
    pos = np.concatenate([np.array(e["pos"]) for e in episodes], 0)
    perr = np.concatenate([np.array(e["pred_err"]) for e in episodes], 0)
    door = np.concatenate([np.array(e["is_doorway"]) for e in episodes], 0)
    epid = np.concatenate([np.full(len(e["latent"]), e["ep_id"]) for e in episodes], 0)
    reached = np.array([e["reached"] for e in episodes])

    tag = "random" if args.random else "cem"
    npz = run_dir / f"latents_{tag}.npz"
    np.savez_compressed(
        npz,
        latent=lat, pos=pos, pred_err=perr, is_doorway=door, ep_id=epid,
        ep_reached=reached,
        # per-episode imagined rollouts, object array (ragged is fine; all H here)
        imagined=np.array([e["imagined_latents"] for e in episodes], dtype=object),
        ep_start=np.array([e["start"] for e in episodes]),
        ep_goal=np.array([e["goal"] for e in episodes]),
    )

    manifest = {
        "git_commit": _git_commit(),
        "ckpt_md5": _md5(run_dir / "ckpt.pt"),
        "run_dir": str(run_dir),
        "seed": args.seed,
        "num_eval": args.num_eval,
        "random": args.random,
        "embed_dim": m["embed_dim"],
        "history_size": m["history_size"],
        "n_states": int(lat.shape[0]),
        "note": "raw latents; project downstream per prereg_views.md",
    }
    (run_dir / f"latents_{tag}.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  wrote {npz}")
    print(f"  wrote {run_dir / f'latents_{tag}.manifest.json'}")
    print(f"  {lat.shape[0]} states, latent dim {lat.shape[1]}")


if __name__ == "__main__":
    main()
