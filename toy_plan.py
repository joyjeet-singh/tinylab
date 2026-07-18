"""
toy_plan.py -- turning the world model into a success rate.

The idea, in plain terms
-----------------------
The trained model can imagine: give it some frames and a made-up action, it
predicts what the next scene-SUMMARY would be. So we can ask "of many possible
action sequences, which ends up nearest the goal?" without ever touching the
real world.

CEM (Cross-Entropy Method) is the search. Guess-and-refine:
    1. GUESS a few hundred random action sequences.
    2. IMAGINE each -- roll the model forward to see where you would end up.
    3. SCORE each by how far the predicted final summary is from the goal
       picture's summary.
    4. KEEP the best handful, note their average and spread.
    5. Draw the next round of guesses from around those, and repeat.
The guesses tighten toward good plans. No reward function, no training -- the
planner is pure search, using the model as an imagination engine.

Then EXECUTE: take the first action for real, watch the world respond, re-plan
from the new picture. Repeat until the goal is reached or the budget runs out.
Success = the fraction of attempts that get there.

Reference settings (le-wm/config/eval/solver/cem.yaml + tworoom.yaml):
    num_samples 300, n_steps 30, topk 30, var_scale 1.0
    horizon 5, receding_horizon 5, action_block 5 (= frameskip)
    goal_offset_steps 25, eval_budget 50, num_eval 50
Note goal_offset_steps is 25 -- NOT the 100 the paper's TwoRoom section implies.
Another paper-vs-repo conflict, recorded rather than resolved.

THE HONEST CAVEAT
-----------------
The scoring above -- straight distance between summaries -- is exactly what the
"Beyond Euclidean Proximity" paper argues makes LeWM fail on TwoRoom: two points
either side of a wall look CLOSE in summary-space but are far apart in reality.
That is the published explanation for the 87%. We are not testing that claim; we
are reproducing the setup faithfully. It is simply why 87 rather than 97 is the
honest target.

Run: python toy_plan.py --run runs/<dir>          (uses that run's checkpoint)
     python toy_plan.py --run runs/<dir> --random (random-action control)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from make_toy_tworoom import ACTION_CLIP, ToyTwoRoom
from toy_model import ToyJEPA


# ---------------------------------------------------------------------------
# CEM: guess, imagine, score, refine
# ---------------------------------------------------------------------------
class CEMPlanner:
    """
    Faithful in structure to the reference CEMSolver, sized for the toy.

    horizon      : how many actions ahead to plan
    num_samples  : how many candidate sequences per round
    n_steps      : how many refine rounds
    topk         : how many of the best to learn from each round
    var_scale    : how wide the first round's guesses are
    """

    def __init__(self, model, horizon=5, num_samples=300, n_steps=30, topk=30,
                 var_scale=1.0, action_dim=2, device="cpu"):
        self.model = model
        self.horizon = horizon
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.var_scale = var_scale
        self.action_dim = action_dim
        self.device = device

    @torch.no_grad()
    def _imagine_cost(self, ctx_emb, ctx_act, cand, goal_emb):
        """
        Roll the model forward under each candidate sequence; return the distance
        from each predicted ending to the goal.

        ctx_emb : (1, H, D)      summaries of the frames we have actually seen
        ctx_act : (1, H, A)      the actions that produced them
        cand    : (S, horizon, A) candidate future action sequences
        goal_emb: (1, D)         summary of the goal picture
        """
        S = cand.size(0)
        emb = ctx_emb.expand(S, -1, -1).clone()      # (S, H, D)
        act = ctx_act.expand(S, -1, -1).clone()      # (S, H, A)

        for t in range(self.horizon):
            act = torch.cat([act, cand[:, t:t + 1]], dim=1)
            act_emb = self.model.action_encoder(act)
            # keep only the last `history_size` frames -- the predictor's window
            HS = self.model.history_size
            pred = self.model.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)

        final = emb[:, -1]                            # (S, D)
        return (final - goal_emb).pow(2).sum(-1)      # (S,)

    @torch.no_grad()
    def plan(self, ctx_emb, ctx_act, goal_emb, rng):
        """Return the best action sequence found: (horizon, action_dim)."""
        H, A = self.horizon, self.action_dim
        mean = torch.zeros(H, A, device=self.device)
        std = torch.full((H, A), self.var_scale, device=self.device)

        for _ in range(self.n_steps):
            noise = torch.from_numpy(
                rng.standard_normal((self.num_samples, H, A))).float().to(self.device)
            cand = (mean.unsqueeze(0) + std.unsqueeze(0) * noise)
            cand = cand.clamp(-ACTION_CLIP, ACTION_CLIP)

            cost = self._imagine_cost(ctx_emb, ctx_act, cand, goal_emb)
            elite = cand[cost.topk(self.topk, largest=False).indices]   # best ones

            mean = elite.mean(0)
            std = elite.std(0).clamp(min=1e-3)        # never collapse to a point

        return mean.clamp(-ACTION_CLIP, ACTION_CLIP)


# ---------------------------------------------------------------------------
# acting in the world
# ---------------------------------------------------------------------------
def frame_to_tensor(img: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 -> (1,3,H,W) float in 0..1, matching the loader."""
    x = img.astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def run_episode(model, planner, env, img_size, start_pos, goal_pos,
                budget, frameskip, rng, success_radius=3.0):
    """
    One attempt. Plan, execute the first block of actions for real, re-plan from
    what actually happened. Repeat until the goal is reached or budget runs out.

    Returns (reached, steps_used, final_distance).
    """
    HS = model.history_size

    # the goal picture -> its summary (this is what we steer toward)
    goal_img = env.render(goal_pos, img_size)
    goal_emb = model.encode({"pixels": frame_to_tensor(goal_img).unsqueeze(0)})["emb"][:, 0]

    # seed the history by standing still, so we have H frames to start from
    pos = start_pos.copy()
    hist_imgs = [env.render(pos, img_size) for _ in range(HS)]
    hist_acts = [np.zeros(2, np.float32) for _ in range(HS)]

    steps = 0
    while steps < budget:
        px = torch.cat([frame_to_tensor(i) for i in hist_imgs[-HS:]], 0).unsqueeze(0)
        ac = torch.from_numpy(np.stack(hist_acts[-HS:])).float().unsqueeze(0)
        out = model.encode({"pixels": px, "action": ac})
        ctx_emb, ctx_act = out["emb"], ac

        seq = planner.plan(ctx_emb, ctx_act, goal_emb, rng).cpu().numpy()

        # receding horizon: execute the plan's first action, then re-plan.
        # action_block = frameskip: one planned action = `frameskip` real steps,
        # because the model was trained on frames `frameskip` apart.
        a = seq[0]
        for _ in range(frameskip):
            pos = env.step(pos, a)
            steps += 1
            if np.linalg.norm(pos - goal_pos) < success_radius:
                return True, steps, float(np.linalg.norm(pos - goal_pos))
            if steps >= budget:
                break
        hist_imgs.append(env.render(pos, img_size))
        hist_acts.append(a.astype(np.float32))

    return False, steps, float(np.linalg.norm(pos - goal_pos))


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def probe_position(model, env, img_size=32, n=400, seed=0):
    """
    Can we read the dot's position back out of a summary? (A linear probe.)

    This is the same check the paper runs in Appendix F.2, and it is the FIRST
    thing to look at before trusting any planning number, because it separates
    two very different failures:

      low R^2  -> the encoder has not learned to see yet. The planner has no
                  signal to follow. Train longer; planning results are meaningless.
      high R^2 -> the encoder sees fine. If planning still fails, the problem is
                  the SCORING -- straight-line distance between summaries is a
                  poor measure of how far apart two states really are. That is
                  exactly what the paper's own probing table shows for the real
                  TwoRoom (LeWM: R^2 0.996, yet planning underperforms), and what
                  "Beyond Euclidean Proximity" argues is the cause.

    Reference LeWM reaches R^2 ~0.996 on real TwoRoom after full training.
    """
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(seed)
    lo, hi = env.margin, env.size - env.margin
    pos, emb = [], []
    with torch.no_grad():
        for _ in range(n):
            p = np.array([rng.uniform(lo, hi), rng.uniform(lo, hi)], np.float32)
            e = model.encode(
                {"pixels": frame_to_tensor(env.render(p, img_size)).unsqueeze(0)}
            )["emb"][:, 0]
            pos.append(p)
            emb.append(e.squeeze(0).numpy())
    pos, emb = np.array(pos), np.array(emb)
    cut = int(n * 0.75)
    r = Ridge(alpha=1.0).fit(emb[:cut], pos[:cut])
    return float(r.score(emb[cut:], pos[cut:]))


def evaluate(run_dir: Path, num_eval: int, use_random: bool, seed: int = 42):
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

    env = ToyTwoRoom()
    img_size = m.get("img_size", 32)
    frameskip = cfg["data"]["frameskip"]

    planner = CEMPlanner(model, horizon=5, num_samples=300, n_steps=30, topk=30,
                         var_scale=1.0, action_dim=m["action_dim"])

    rng = np.random.default_rng(seed)
    lo, hi = env.margin, env.size - env.margin

    def sample(on_left):
        x = (rng.uniform(lo, env.wall_x - 4) if on_left
             else rng.uniform(env.wall_x + 4, hi))
        return np.array([x, rng.uniform(lo, hi)], dtype=np.float32)

    results = []
    for i in range(num_eval):
        left = rng.random() < 0.5
        start, goal = sample(left), sample(not left)   # goal in the OTHER room
        if use_random:
            pos, steps, reached = start.copy(), 0, False
            while steps < 50:
                pos = env.step(pos, rng.uniform(-1, 1, 2).astype(np.float32))
                steps += 1
                if np.linalg.norm(pos - goal) < 3.0:
                    reached = True
                    break
            dist = float(np.linalg.norm(pos - goal))
        else:
            reached, steps, dist = run_episode(
                model, planner, env, img_size, start, goal,
                budget=50, frameskip=frameskip, rng=rng)
        results.append({"reached": reached, "steps": steps, "final_dist": dist})
        print(f"  episode {i+1:2d}/{num_eval}: "
              f"{'REACHED' if reached else 'missed ':7s} in {steps:2d} steps, "
              f"final distance {dist:5.1f}")

    n = len(results)
    success = sum(r["reached"] for r in results) / n
    mean_dist = sum(r["final_dist"] for r in results) / n
    out = {"success_rate": success, "mean_final_distance": mean_dist,
           "n_eval": n, "results": results}
    if not use_random:
        out["probe_r2"] = probe_position(model, env, img_size)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run dir with ckpt.pt + manifest.json")
    p.add_argument("--num-eval", type=int, default=20,
                   help="reference uses 50; fewer is faster for debugging")
    p.add_argument("--random", action="store_true",
                   help="random-action control instead of the planner")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_dir = Path(args.run)
    label = "RANDOM ACTIONS (control)" if args.random else "CEM PLANNER"
    print("=" * 64)
    print(f"{label}  --  {run_dir.name}")
    print("=" * 64)

    out = evaluate(run_dir, args.num_eval, args.random, args.seed)

    print()
    print(f"  success rate        : {out['success_rate']*100:.1f}%  "
          f"({sum(r['reached'] for r in out['results'])}/{out['n_eval']})")
    print(f"  mean final distance : {out['mean_final_distance']:.2f}")

    if not args.random:
        print()
        print("  DIAGNOSTIC -- run this before trusting any planning number:")
        r2 = out["probe_r2"]
        print(f"    linear probe, position from summary: R^2 = {r2:.4f}")
        print("    (reference LeWM reaches ~0.996 on real TwoRoom after full training)")
        if r2 < 0.8:
            print("    -> LOW. The encoder has not learned to see yet, so the planner")
            print("       has no signal to follow. This success rate says nothing")
            print("       about the planner. Train longer.")
        else:
            print("    -> HIGH. The encoder sees fine. If planning still fails, the")
            print("       problem is the SCORING: straight-line distance between")
            print("       summaries is a poor measure of real distance. That is the")
            print("       published explanation for LeWM's 87% on TwoRoom.")

    print()
    print("  Reference LeWM scores 87% on the REAL TwoRoom (PLDM/DINO-WM: 97-100).")
    print("  This is the TOY world with a CNN encoder -- the number here is NOT")
    print("  comparable to 87 and is not a result. It only tells us the planning")
    print("  loop works end to end before we pay for compute.")

    tag = "random" if args.random else "cem"
    (run_dir / f"plan_eval_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"  wrote {run_dir / f'plan_eval_{tag}.json'}")


if __name__ == "__main__":
    main()
