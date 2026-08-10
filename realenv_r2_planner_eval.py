"""
realenv_r2_planner_eval.py -- Stage R2: the planner evaluation, in the real
environment, under the reference protocol. The first planning number of this
project that is measured in the world the model was trained on.

WHAT R0/R1 ESTABLISHED (why this is allowed to run)
---------------------------------------------------
R1 proved the installed swm/TwoRoom-v1 IS the dataset's generator: frame
reconstruction at recorded positions gave pixel MAE 0.00 and dot offset 0.00;
replaying recorded actions gave one-step and 40-step errors of exactly 0.000;
the encoder's paired latent distance between real frames and env renders at
the same position is 0.01 (real cloud spacing ~2.6; the toy fixture scored
61). The eval world and the training world are the same world.

PROTOCOL (reference: le-wm config/eval/tworoom.yaml + solver/cem.yaml)
----------------------------------------------------------------------
- num_eval 50 episodes; eval_budget 50 env steps; img 224.
- Start = a recorded dataset state; goal = the recorded state
  goal_offset_steps=25 raw steps later on the same trajectory
  (LOCAL goals -- replayed expert trajectories, not cross-room).
- Goal image = the environment rendered with the agent at the goal
  (the package's own _set_goal_state convention).
- Success = the environment's own terminated flag (distance to target < 16).
- CEM: num_samples 300, n_steps 30, topk 30, var_scale 1.0, horizon 5,
  receding_horizon 5, action_block 5 (each planned action is executed for
  5 env steps, matching the training frameskip). We use toy_plan.CEMPlanner,
  which already carries exactly these values.
- Random control: same starts/goals/budget, a fresh uniform action in
  [-1,1]^2 every env step.

DEVIATIONS from the reference, stated plainly (also printed at the end):
- Episode/start selection: 50 random episodes (seed 42), start = the
  episode's first frame, goal = frame 25 of that episode. The reference's
  exact trajectory selection is not documented in the config.
- receding_horizon 5 interpreted as: execute 5 planned actions, then
  re-plan (2 planning rounds per 50-step budget).
- The model is OUR reproduction checkpoint, not the authors' released
  weights. The published 87% belongs to their weights; this measures ours.

THE GUARD RUNS FIRST. If env renders are not latent-indistinguishable from
the training frames (paired median < 1.0), this script refuses to produce a
planning number.

Run from the tinylab folder (venv active):
    python3 realenv_r2_planner_eval.py --run runs/<run_dir>
    python3 realenv_r2_planner_eval.py --run runs/<run_dir> --random
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

REPORT = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    REPORT.append(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--h5", default="~/Downloads/tworoom.h5")
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--receding", type=int, default=5)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--random", action="store_true",
                   help="random-action control instead of the planner")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", default=None,
                   help="JSON file from make_balanced_episode_set.py: use a "
                        "committed, auditable episode list instead of random "
                        "sampling")
    p.add_argument("--authors-spec", default=None,
                   help="authors_driving_spec.json: evaluate the AUTHORS' "
                        "released checkpoint instead of ours, with every "
                        "other protocol element unchanged")
    p.add_argument("--cem-steps", type=int, default=30,
               help="CEM iterations; App. D specifies 10 for TwoRoom")
    p.add_argument("--random-start", action="store_true",
                   help="draw the start state uniformly from within the "
                        "trajectory rather than using its first frame, as "
                        "App. F.1 of the reference describes")
    p.add_argument("--goal-target", action="store_true",
                   help="use the episode's recorded target position as the "
                        "goal, rather than a state --goal-offset frames later; "
                        "this is the environment's own task")
    p.add_argument("--successful-only", action="store_true",
                   help="restrict to episodes in which the data policy reached "
                        "the target (terminated rather than timed out)")
    p.add_argument("--ckpt", default=None,
                   help="checkpoint filename inside the run dir; defaults to "
                        "ckpt_best.pt, else ckpt.pt")
    p.add_argument("--action-scale", type=float, default=1.0,
                   help="multiply the action before it reaches the model "
                        "(NOT the environment). Our loader subsampled one "
                        "action per clip step, so the model under-predicts "
                        "displacement by the frameskip factor; 5.0 corrects "
                        "it. Default 1.0 reproduces every committed result.")
    p.add_argument("--unsafe-skip-guard", action="store_true",
                   help="mechanics testing ONLY; recorded loudly in outputs")
    args = p.parse_args()

    import gymnasium as gym
    import torch
    import h5py
    import hdf5plugin  # noqa: F401
    import stable_worldmodel  # noqa: F401
    from toy_model import ToyJEPA
    from toy_plan import CEMPlanner, frame_to_tensor

    run_dir = Path(args.run)
    h5_path = str(Path(args.h5).expanduser())
    tag = ("authors_" if args.authors_spec else "") + \
          ("random" if args.random else "cem")

    say("=" * 70)
    say(f"REAL-ENVIRONMENT PLANNER EVAL ({tag.upper()})  --  {run_dir.name}")
    say("=" * 70)

    # ---- model ------------------------------------------------------------
    if args.authors_spec:
        # their weights carry their own architecture; the run dir is unused
        m, model = {"action_dim": 2}, None
    else:
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
    if args.authors_spec:
        from authors_adapter import load_authors_model
        say("MODEL: the AUTHORS' released checkpoint (calibration run)")
        model = load_authors_model(args.authors_spec)
        ckpt = Path(args.authors_spec)
        ck = {"epoch": -1}
        say("  every other protocol element is unchanged from our own run")
    else:
        ckpt = run_dir / (args.ckpt or
                          ("ckpt_best.pt"
                           if (run_dir / "ckpt_best.pt").exists()
                           else "ckpt.pt"))
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        # A checkpoint trained with the corrected pipeline expects
        # ImageNet-normalised pixels and dense z-scored actions. The manifest
        # records which; wrap_if_needed() returns the model untouched when the
        # run predates the change, so older results are unaffected.
        from dense_action_adapter import wrap_if_needed
        _wrapped = wrap_if_needed(model, run_dir, args.h5,
                                  frameskip=cfg["data"]["frameskip"])
        if _wrapped is not model:
            if args.action_scale != 1.0:
                raise SystemExit(
                    "--action-scale cannot be combined with a corrected-"
                    "pipeline checkpoint: the adapter already applies the "
                    "dataset z-score, and the two rescalings would compose.")
            model = _wrapped
            # The planner must search the ENVIRONMENT's action space, which is
            # 2-D. The manifest's action_dim of 10 is the width of the model's
            # DENSE input (frameskip x 2), and the adapter produces that from a
            # 2-D action. Leaving it at 10 makes CEM sample candidates the
            # environment cannot execute.
            m = {**m, "action_dim": 2}
            say("  planner action space pinned to 2-D (the environment's); "
                "the adapter widens each action to the model's 10-D input")
        say(f"model: {m.get('encoder', 'cnn')} at {m.get('img_size', 32)}px "
            f"({ckpt.name}, epoch field {ck.get('epoch')})")

    if args.action_scale != 1.0:
        if args.authors_spec:
            raise SystemExit(
                "--action-scale and --authors-spec both rescale actions; the "
                "authors' encoding comes from the driving spec. Pass only one.")
        class _ScaledActionEncoder(torch.nn.Module):
            """Wraps the real encoder; must be a Module to be assignable."""

            def __init__(self, inner, k):
                super().__init__()
                self.inner, self.k = inner, float(k)

            def forward(self, a):
                return self.inner(a * self.k)

        _k = float(args.action_scale)
        model.action_encoder = _ScaledActionEncoder(model.action_encoder, _k)
        say(f"ACTION SCALE {_k:g} applied to the model input only "
            f"(the environment still receives the unscaled action)")

    @torch.no_grad()
    def emb_of(img_hwc_uint8):
        px = frame_to_tensor(np.asarray(img_hwc_uint8)).unsqueeze(0)  # (1,1,3,H,W)
        return model.encode({"pixels": px})["emb"]                    # (1,1,D)

    # ---- environment ------------------------------------------------------
    env = gym.make("swm/TwoRoom-v1", disable_env_checker=True)
    u = env.unwrapped
    env.reset(seed=args.seed)

    # ---- THE GUARD (G3): refuse to score off-manifold inputs --------------
    say("\ndomain guard (precondition):")
    guard = {"status": "not run"}
    try:
        rng_g = np.random.default_rng(0)
        with h5py.File(h5_path, "r") as f:
            picks = np.sort(rng_g.choice(f["pixels"].shape[0], 10, replace=False))
            real = f["pixels"][picks]
            pos = np.asarray(f["pos_agent"][picks], dtype=np.float32)
        if real[0].shape != np.asarray(u.render()).shape:
            raise RuntimeError(f"frame shape mismatch: data {real[0].shape} "
                               f"vs env {np.asarray(u.render()).shape}")
        ds = []
        for i in range(len(picks)):
            u._set_state(pos[i])
            with torch.no_grad():
                ze = emb_of(np.asarray(u.render()))[0, 0]
                zr = emb_of(real[i])[0, 0]
            ds.append(float(torch.norm(ze - zr)))
        med = float(np.median(ds))
        guard = {"status": "pass" if med < 1.0 else "FAIL",
                 "paired_median": med}
        say(f"  paired ||z_real - z_env|| median over 10 frames: {med:.3f} "
            f"(threshold 1.0) -> {guard['status']}")
        if guard["status"] == "FAIL" and not args.unsafe_skip_guard:
            say("  REFUSING to produce a planning number on off-manifold "
                "inputs. (This is the gate that would have saved three runs.)")
            Path(f"realenv_plan_{tag}_report.txt").write_text("\n".join(REPORT))
            return
    except Exception:
        say("  guard errored:")
        say(traceback.format_exc())
        if not args.unsafe_skip_guard:
            say("  REFUSING to run without a passing guard.")
            Path(f"realenv_plan_{tag}_report.txt").write_text("\n".join(REPORT))
            return
    if args.unsafe_skip_guard:
        guard["status"] = "SKIPPED (unsafe flag)"
        say("  !! GUARD SKIPPED VIA --unsafe-skip-guard -- mechanics test "
            "only; numbers below are NOT results !!")

    # ---- episode sampling: replayed trajectories --------------------------
    rng = np.random.default_rng(args.seed)
    meta = None
    if args.episodes:
        spec = json.loads(Path(args.episodes).read_text())
        file_off = int(spec.get("goal_offset", args.goal_offset))
        if file_off != args.goal_offset:
            raise SystemExit(
                f"episode file was built for goal_offset {file_off} but "
                f"--goal-offset is {args.goal_offset}; refusing to mix them")
        eps = np.array([int(r["episode"]) for r in spec["episodes"]])
        meta = [{"geometry": r.get("geometry"), "pair_id": r.get("pair_id")}
                for r in spec["episodes"]]
        say(f"episode set: {args.episodes} "
            f"({spec.get('n_pairs')} matched pairs, "
            f"selection: {spec.get('selection', 'n/a')})")
    if args.random_start and getattr(args, "episodes", None):
        raise SystemExit(
            "--random-start cannot be combined with --episodes: a committed "
            "episode set fixes its starts as part of the pre-registration, and "
            "redrawing them would invalidate the matching the design rests on.")
    with h5py.File(h5_path, "r") as f:
        off = np.asarray(f["ep_offset"][:])
        ln = np.asarray(f["ep_len"][:])
        if meta is None:
            if args.goal_target:
                ok = np.arange(len(ln))     # every episode has a target
            else:
                ok = np.where(ln > args.goal_offset)[0]
            if args.successful_only:
                term = np.asarray(f["terminated"][:])
                reached = np.array([bool(term[int(o) + int(l) - 1])
                                    for o, l in zip(off, ln)])
                ok = np.array([e for e in ok if reached[e]])
                if len(ok) == 0:
                    raise SystemExit("no episodes satisfy --successful-only")
            eps = rng.choice(ok, size=min(args.num_eval, len(ok)),
                             replace=False)
            meta = [{} for _ in eps]
        starts, goals, start_shifts = [], [], []
        for e in eps:
            # App. F.1: the initial state is sampled from within the
            # trajectory, not fixed at its first frame. For the offset-100
            # population -- only episodes longer than 100 steps -- frame 0 is
            # the hardest available segment, so this is not a neutral choice.
            s0 = int(off[e])
            if args.random_start:
                span = int(ln[e]) - args.goal_offset - 1
                if span > 0:
                    s0 += int(rng.integers(0, span))
            start_shifts.append(s0 - int(off[e]))
            starts.append(np.asarray(f["pos_agent"][s0], dtype=np.float32))
            goals.append(np.asarray(
                f["pos_target"][s0] if args.goal_target
                else f["pos_agent"][s0 + args.goal_offset],
                dtype=np.float32))
    PROTO_FILTER = ' [policy-successful only]' if args.successful_only else ''
    PROTO_START = ('a uniformly random frame' if args.random_start
                   else 'episode frame 0')
    if args.random_start:
        _moved = sum(1 for sh in start_shifts if sh > 0)
        PROTO_START += (f" (realised: {_moved}/{len(eps)} starts actually moved"
                        f"; max shift {max(start_shifts) if start_shifts else 0}"
                        f" frames)")
    PROTO_GOAL = ("the episode's recorded TARGET position" if args.goal_target
                  else f"{args.goal_offset} frames later in the same episode")
    PROTO_OFFSET = ("goal = recorded target" if args.goal_target
                    else f"goal_offset {args.goal_offset}")
    say(f"\nepisodes: {len(eps)} sampled (seed {args.seed}){PROTO_FILTER}"
        f"; start = {PROTO_START}; goal = {PROTO_GOAL}")

    planner = None
    if not args.random:
        planner = CEMPlanner(model, horizon=5, num_samples=300,
                             n_steps=args.cem_steps,
                             topk=30, var_scale=1.0,
                             action_dim=m["action_dim"])

    # ---- the evaluation ----------------------------------------------------
    results = []
    t0 = time.time()
    for i, e in enumerate(eps):
        start, goal = starts[i], goals[i]
        env.reset(seed=int(args.seed * 1000 + i))
        u._set_state(start)
        u._set_goal_state(goal)
        sg = float(np.linalg.norm(start - goal))
        trivial = sg < 16.0

        with torch.no_grad():
            goal_img = u._render_frame(
                agent_pos=torch.tensor(goal, dtype=torch.float32))
            goal_img = goal_img.cpu().numpy().transpose(1, 2, 0)
            goal_emb = emb_of(goal_img)[:, 0]                     # (1, D)

        HS = model.history_size
        steps, success, plans = 0, False, 0
        dist = sg
        hist_imgs = [np.asarray(u.render()) for _ in range(HS)]
        hist_acts = [np.zeros(2, dtype=np.float32) for _ in range(HS)]
        while steps < args.budget and not success:
            if args.random:
                a = rng.uniform(-1, 1, 2).astype(np.float32)
                _, _, term, _, info = u.step(a)
                steps += 1
                dist = float(info["distance_to_target"])
                success = bool(term)
                continue
            with torch.no_grad():
                px = torch.cat([frame_to_tensor(f) for f in hist_imgs[-HS:]],
                               0).unsqueeze(0)                    # (1,HS,3,H,W)
                ctx_emb = model.encode({"pixels": px})["emb"]     # (1,HS,D)
                ctx_act = torch.from_numpy(
                    np.stack(hist_acts[-HS:])).float().unsqueeze(0)
                seq = planner.plan(ctx_emb, ctx_act, goal_emb,
                                   rng).cpu().numpy()
            plans += 1
            for k in range(min(args.receding, seq.shape[0])):
                a = seq[k].astype(np.float32)
                for _ in range(args.frameskip):
                    _, _, term, _, info = u.step(a)
                    steps += 1
                    dist = float(info["distance_to_target"])
                    if term:
                        success = True
                        break
                    if steps >= args.budget:
                        break
                hist_imgs.append(np.asarray(u.render()))
                hist_acts.append(a)
                if success or steps >= args.budget:
                    break

        results.append({"episode": int(e), "success": bool(success),
                        "steps": int(steps), "final_dist": round(dist, 2),
                        "start_goal_dist": round(sg, 2),
                        "trivial_start": bool(trivial),
                        "plans": int(plans), **meta[i]})
        say(f"  ep {i+1:2d}/{len(eps)} (#{int(e):5d}): "
            f"{'REACHED' if success else 'missed '} in {steps:2d} steps, "
            f"final dist {dist:6.1f}, start-goal {sg:6.1f}"
            f"{'  [trivial start]' if trivial else ''}")

    # ---- summary -----------------------------------------------------------
    n = len(results)
    succ = sum(r["success"] for r in results)
    n_triv = sum(r["trivial_start"] for r in results)
    nt = [r for r in results if not r["trivial_start"]]
    say("\n" + "=" * 70)
    say(f"SUMMARY ({tag})")
    say("=" * 70)
    say(f"  success rate           : {succ}/{n} = {succ/n*100:.1f}%")
    say(f"  mean final distance    : "
        f"{np.mean([r['final_dist'] for r in results]):.2f}")
    say(f"  trivial starts (<16)   : {n_triv}/{n}")
    if nt:
        st = sum(r["success"] for r in nt)
        say(f"  non-trivial success    : {st}/{len(nt)} = "
            f"{st/len(nt)*100:.1f}%")
    say(f"  wall-clock             : {(time.time()-t0)/60:.1f} min")
    say("\nDEVIATIONS from the reference protocol (state wherever quoted):")
    say("  - episode/start selection ours ("
        + (f"committed set {Path(args.episodes).name}"
           if getattr(args, "episodes", None)
           else f"random {args.num_eval} eps seed {args.seed}{PROTO_FILTER}")
        + ", start =")
    say(f"    {PROTO_START}, goal = {PROTO_GOAL}); reference's "
        f"exact selection unknown")
    say(f"  - receding_horizon {args.receding} read as: execute "
        f"{args.receding} planned actions, replan")
    say("  - the AUTHORS' released weights, driven through our harness"
        if getattr(args, "authors_spec", None)
        else "  - OUR reproduction checkpoint, not the authors' released weights")
    if getattr(args, "action_scale", 1.0) != 1.0:
        say(f"  - action scaled by {args.action_scale:g} before reaching the "
            f"model (the environment receives the unscaled action)")
    say("  - success = the registered env's terminated rule (distance < 16)")
    say("  matched: env, CEM 300/30/30/1.0, horizon 5, action_block 5,")
    say(f"  budget {args.budget}, {PROTO_OFFSET}, "
        f"img 224, goal-image")
    say("  convention.")

    out = {"tag": tag, "guard": guard, "results": results,
           "success_rate": succ / n, "n": n, "trivial_starts": n_triv,
           "action_scale": args.action_scale,
           "goal_construction": PROTO_GOAL, "budget": args.budget,
           "goal_offset": args.goal_offset,
           "goal_target": bool(args.goal_target),
           "successful_only": bool(args.successful_only),
           "random_start": bool(args.random_start),
           "authors_spec": getattr(args, "authors_spec", None),
           "protocol": {"num_eval": args.num_eval, "budget": args.budget,
                        "goal_offset": args.goal_offset,
                        "receding": args.receding,
                        "frameskip": args.frameskip, "seed": args.seed,
                        "success_rule": "env terminated (dist<16)",
                        "cem": None if args.random else
                        {"num_samples": 300, "n_steps": 30, "topk": 30,
                         "var_scale": 1.0, "horizon": 5}},
           "ckpt": str(ckpt), "epoch_field": int(ck.get("epoch", -1))}
    Path(f"realenv_plan_{tag}.json").write_text(json.dumps(out, indent=2))
    Path(f"realenv_plan_{tag}_report.txt").write_text("\n".join(REPORT))
    say(f"\nwrote realenv_plan_{tag}.json and realenv_plan_{tag}_report.txt")


if __name__ == "__main__":
    main()
