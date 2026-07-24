"""
realenv_r1_reconstruction.py -- Stage R1 of the real-environment planner eval:
prove the installed environment reproduces the dataset, frame by frame and
step by step, before any planning is wired.

WHAT R0 ESTABLISHED (and what this settles)
-------------------------------------------
R0's domain guard put default-reset env renders at median 5.81 from the
nearest training latent (real cloud spacing 2.57; the toy fixture scored
61.03). Close -- but that number mixes true rendering differences with
position-coverage mismatch, because env resets and dataset frames sample
different positions. R1 removes the mixing by comparing at the SAME
positions, and adds the dynamics test:

  R1.1  constants + conventions of the registered env (introspected, on
        your machine, not assumed from mine).
  R1.2  FRAME RECONSTRUCTION: put the env's agent at recorded pos_agent
        values and pixel-compare the render against the actual dataset
        frame at that index. Reports pixel MAE, dot-center offset, and
        latent distance per frame; writes a real|env|difference PNG.
  R1.3  DYNAMICS REPLAY: (a) one-step -- set the recorded state, apply the
        recorded action, compare to the recorded next position, ~200
        transitions; (b) compounding -- replay 40 recorded actions from an
        episode start and track drift. Container-verified arithmetic:
        displacement = action x speed(5.0), deterministic.
  R1.4  MATCHED-POSITION DOMAIN GUARD: paired latent distance
        ||z(real frame i) - z(env render at pos_i)|| -- the true rendering
        residual, directly comparable to R0's 5.81 and the 2.57 spacing.

All sections isolated; report to realenv_r1_report.txt.

Run from the tinylab folder (venv active):
    python3 realenv_r1_reconstruction.py --run runs/<run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

REPORT = []


def say(*parts):
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    REPORT.append(line)


def section(title):
    say("\n" + "=" * 70)
    say(title)
    say("=" * 70)


def dot_center(frame):
    """(x, y) of the reddest pixel -- the agent's rendered center."""
    m = frame[:, :, 0].astype(int) - (frame[:, :, 1].astype(int)
                                      + frame[:, :, 2].astype(int)) // 2
    y, x = np.unravel_index(np.argmax(m), m.shape)
    return int(x), int(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--h5", default="~/Downloads/tworoom.h5")
    p.add_argument("--recon-frames", type=int, default=12)
    p.add_argument("--paired-frames", type=int, default=100)
    p.add_argument("--one-step", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = Path(args.run)
    h5_path = str(Path(args.h5).expanduser())

    import gymnasium as gym
    import torch
    import stable_worldmodel  # noqa: F401

    env = gym.make("swm/TwoRoom-v1", disable_env_checker=True)
    u = env.unwrapped
    env.reset(seed=args.seed)
    u._set_goal_state(np.array([200.0, 200.0], dtype=np.float32))  # far corner

    # ---- R1.1 ---------------------------------------------------------------
    section("R1.1  registered env constants + conventions")
    try:
        for attr in ("IMG_SIZE", "BORDER_SIZE", "DOT_STD", "WALL_CENTER",
                     "WALL_WIDTH_DEFAULT", "MAX_SPEED"):
            say(f"  {attr} = {getattr(type(u), attr, '?')}")
        vs = u.variation_space
        say("  door position/size/number:",
            np.asarray(vs["door"]["position"].value).tolist(),
            np.asarray(vs["door"]["size"].value).tolist(),
            "(size = half-extent in pixels)",
            int(np.asarray(vs["door"]["number"].value)))
        say("  agent radius/speed:",
            float(np.asarray(vs["agent"]["radius"].value)[0]),
            float(np.asarray(vs["agent"]["speed"].value)[0]))
        say("  success rule in step(): terminated = distance_to_target < 16.0")
        say("  conventions verified in container: pos = (x=column, y=row),")
        say("  no flip; displacement per step = action x speed, exact;")
        say("  target NOT rendered (render_target False).")
    except Exception:
        say(traceback.format_exc())

    # ---- data handles -------------------------------------------------------
    import h5py
    import hdf5plugin  # noqa: F401
    rng = np.random.default_rng(args.seed)

    # ---- R1.2 FRAME RECONSTRUCTION -----------------------------------------
    section("R1.2  frame reconstruction at recorded positions")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with h5py.File(h5_path, "r") as f:
            picks = np.sort(rng.choice(f["pixels"].shape[0],
                                       args.recon_frames, replace=False))
            real = f["pixels"][picks]
            pos = np.asarray(f["pos_agent"][picks], dtype=np.float32)

        maes, dots, recon = [], [], []
        for i in range(len(picks)):
            u._set_state(pos[i])
            fr = np.asarray(env.render())
            recon.append(fr)
            if fr.shape == real[i].shape:
                maes.append(float(np.mean(np.abs(fr.astype(int)
                                                 - real[i].astype(int)))))
                ex, ey = dot_center(fr)
                rx_, ry_ = dot_center(real[i])
                dots.append(float(np.hypot(ex - rx_, ey - ry_)))
            else:
                say(f"  frame {int(picks[i])}: shape mismatch "
                    f"{fr.shape} vs {real[i].shape} -- pixel compare skipped")
        if maes:
            say(f"  pixel MAE (0-255 scale): median {np.median(maes):.2f}  "
                f"max {max(maes):.2f}   over {len(maes)} frames")
            say(f"  dot-center offset (px) : median {np.median(dots):.2f}  "
                f"max {max(dots):.2f}")
            say("  READ: MAE near 0 and dot offset ~0-1 px = the env IS the")
            say("  dataset's renderer. A constant small MAE with 0 dot offset")
            say("  = micro style difference (e.g. dot softness) -- quantified")
            say("  in latent terms by R1.4.")
        fig, ax = plt.subplots(3, 3, figsize=(9, 9.6))
        for j in range(3):
            ax[0, j].imshow(real[j]); ax[0, j].set_title(f"real {int(picks[j])}")
            ax[1, j].imshow(recon[j]); ax[1, j].set_title("env at same pos")
            if recon[j].shape == real[j].shape:
                d = np.abs(recon[j].astype(int) - real[j].astype(int)).sum(-1)
                ax[2, j].imshow(np.minimum(d * 5, 255), cmap="magma")
                ax[2, j].set_title("abs diff x5")
        for a in ax.flat:
            a.axis("off")
        fig.tight_layout()
        fig.savefig("realenv_r1_reconstruction.png", dpi=120)
        say("  wrote realenv_r1_reconstruction.png")
    except Exception:
        say(traceback.format_exc())

    # ---- R1.3 DYNAMICS REPLAY ----------------------------------------------
    section("R1.3  dynamics replay against recorded trajectories")
    try:
        with h5py.File(h5_path, "r") as f:
            off = np.asarray(f["ep_offset"][:400])
            ln = np.asarray(f["ep_len"][:400])
            errs = []
            tried = 0
            while len(errs) < args.one_step and tried < args.one_step * 3:
                tried += 1
                e = int(rng.integers(0, len(off)))
                if ln[e] < 3:
                    continue
                t = int(rng.integers(0, ln[e] - 1))
                i0 = int(off[e]) + t
                p0 = np.asarray(f["pos_agent"][i0], dtype=np.float32)
                p1 = np.asarray(f["pos_agent"][i0 + 1], dtype=np.float32)
                a = np.asarray(f["action"][i0], dtype=np.float32)
                u._set_state(p0)
                _, _, _, _, info = u.step(a)
                q1 = np.asarray(info["proprio"], dtype=np.float32)
                errs.append(float(np.linalg.norm(q1 - p1)))
            errs = np.array(errs)
            say(f"  one-step replay ({len(errs)} transitions): "
                f"median {np.median(errs):.3f}  p90 {np.percentile(errs, 90):.3f}  "
                f"max {errs.max():.3f}  (file units; median real move is ~6.2)")

            say("  compounding replay (5 episodes, 40 steps from the start):")
            for k in range(5):
                e = int(rng.integers(0, len(off)))
                n = int(min(40, ln[e] - 1))
                P = np.asarray(f["pos_agent"][off[e]:off[e] + n + 1],
                               dtype=np.float32)
                A = np.asarray(f["action"][off[e]:off[e] + n],
                               dtype=np.float32)
                u._set_state(P[0])
                drift = []
                for t in range(n):
                    _, _, _, _, info = u.step(A[t])
                    drift.append(float(np.linalg.norm(
                        np.asarray(info["proprio"]) - P[t + 1])))
                say(f"    ep {e:4d}: drift at step 1/10/20/40 = "
                    f"{drift[0]:.2f} / {drift[min(9, n-1)]:.2f} / "
                    f"{drift[min(19, n-1)]:.2f} / {drift[n-1]:.2f}")
            say("  READ: one-step median ~0 = same dynamics. Compounding drift")
            say("  growing only at wall contacts = collision-detail difference;")
            say("  growing everywhere = a speed/convention mismatch (STOP).")
    except Exception:
        say(traceback.format_exc())

    # ---- R1.4 MATCHED-POSITION DOMAIN GUARD --------------------------------
    section("R1.4  matched-position domain guard (the true rendering residual)")
    try:
        import torch
        from toy_model import ToyJEPA
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
        ckpt = run_dir / ("ckpt_best.pt" if (run_dir / "ckpt_best.pt").exists()
                          else "ckpt.pt")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()

        @torch.no_grad()
        def encode(frames):
            out = []
            for s in range(0, len(frames), 32):
                chunk = np.asarray(frames[s:s + 32], dtype=np.float32) / 255.0
                x = torch.from_numpy(chunk).permute(0, 3, 1, 2).unsqueeze(1)
                out.append(model.encode({"pixels": x})["emb"][:, 0].cpu().numpy())
            return np.concatenate(out, 0)

        with h5py.File(h5_path, "r") as f:
            picks = np.sort(rng.choice(f["pixels"].shape[0],
                                       args.paired_frames, replace=False))
            real = f["pixels"][picks]
            pos = np.asarray(f["pos_agent"][picks], dtype=np.float32)
        envf = []
        for q in pos:
            u._set_state(q)
            envf.append(np.asarray(env.render()))
        z_real = encode(real)
        z_env = encode(envf)
        paired = np.linalg.norm(z_real - z_env, axis=-1)
        shuf = rng.permutation(len(z_real))
        spacing = np.linalg.norm(z_real - z_real[shuf], axis=-1)
        say(f"  paired ||z_real - z_env|| at same position: "
            f"median {np.median(paired):.2f}  mean {paired.mean():.2f}  "
            f"max {paired.max():.2f}")
        say(f"  scale references: real cloud spacing (NN, from R0) ~2.6; "
            f"random-pair distance within real: {np.median(spacing):.2f}; "
            f"one-step prediction error: ~5.7")
        say("  READ: paired median <= ~2.6 -> env frames are the training")
        say("  frames for the encoder's purposes; R0's 5.81 was coverage, and")
        say("  R2 (the planner run) is fully green-lit. Paired ~5-8 -> a real")
        say("  but one-step-sized rendering residual; R2 still proceeds, with")
        say("  this number stated as a known offset. Paired >> 10 -> stop and")
        say("  find the rendering difference in the R1.2 diff panels first.")
    except Exception:
        say(traceback.format_exc())

    Path("realenv_r1_report.txt").write_text("\n".join(REPORT))
    say("\nwrote realenv_r1_report.txt")


if __name__ == "__main__":
    main()
