"""
realenv_r0_recon.py -- Stage R0 of the real-environment planner evaluation:
reconnaissance and the domain guard, before a single planning number exists.

WHY THIS SCRIPT EXISTS
----------------------
The reference eval runs in stable_worldmodel's swm/TwoRoom-v1 -- the
environment family that generated tworoom.h5. Container recon confirmed:
plain `pip install stable-worldmodel` suffices (the heavy [env] extra that
needs box2d is NOT required for TwoRoom), the env renders 224x224 uint8
frames in the training style, and the reference CEM settings match our
planner's. But three things must be settled EMPIRICALLY before any eval is
wired, because each one recreates a domain gap if wrong:

  1. Which environment class TwoRoom-v1 actually registers (the package
     ships env.py AND legacy_env.py) and its constants (arena size, wall
     position, success rule).
  2. What the variation space does across resets (door position/size/count,
     agent radius are SAMPLED -- the eval must pin the dataset's layout).
  3. Whether default-reset renders land on the training manifold at all
     (the domain guard -- the same measurement that exposed the toy fixture
     at 61 units off; the toy's number is the yardstick for "failed").

This script answers all three and writes realenv_r0_report.txt. It makes NO
planning claims. Each section is isolated: a failure prints its error and
the rest still run.

Run from the tinylab folder (venv active), after:
    pip install stable-worldmodel
    python3 realenv_r0_recon.py --run runs/<run_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # headless-safe rendering

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run dir with manifest.json + ckpt_best.pt")
    p.add_argument("--h5", default="~/Downloads/tworoom.h5")
    p.add_argument("--frames", type=int, default=100,
                   help="frames per side for the domain guard")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = Path(args.run)
    h5_path = str(Path(args.h5).expanduser())

    # ---- R0.1: install + registration ------------------------------------
    section("R0.1  install + which environment TwoRoom-v1 actually is")
    env = None
    try:
        import gymnasium as gym
        import stable_worldmodel
        say("stable_worldmodel version:",
            getattr(stable_worldmodel, "__version__", "unknown"))
        env = gym.make("swm/TwoRoom-v1", disable_env_checker=True)
        u = env.unwrapped
        import inspect
        src = inspect.getfile(type(u))
        say("registered class:", type(u).__name__, "from", src)
        for attr in ("window_size", "border_size", "size", "render_size",
                     "wall_pos", "max_step_norm", "control_hz", "dt",
                     "energy_bound"):
            if hasattr(u, attr):
                say(f"  {attr} = {getattr(u, attr)}")
        say("action space:", env.action_space)
    except Exception:
        say("R0.1 FAILED -- nothing below can run without the env:")
        say(traceback.format_exc())
        Path("realenv_r0_report.txt").write_text("\n".join(REPORT))
        return

    # ---- R0.2: variation space across resets ------------------------------
    section("R0.2  what the variation space does across resets")
    try:
        keys = ("number", "size", "position")
        say("door settings after reset(seed=k), plus agent radius/speed:")
        for s in range(6):
            env.reset(seed=s)
            vs = env.unwrapped.variation_space
            door = {k: np.asarray(vs["door"][k].value).tolist()
                    for k in keys if k in vs["door"].spaces}
            ag = {k: np.asarray(vs["agent"][k].value).tolist()
                  for k in ("radius", "speed") if k in vs["agent"].spaces}
            say(f"  seed {s}: door={door}  agent={ag}")
        say("READ: if door values CHANGE across seeds, the eval must pin the")
        say("dataset's layout before frames are on-manifold. Compare door")
        say("position/size against the values detected FROM THE DATA by")
        say("eval_wall_scoring (door y 38.4-61.9 at 224 scale, wall x 111.2).")
    except Exception:
        say("R0.2 failed:")
        say(traceback.format_exc())

    # ---- R0.3: render grid for the eyeball check ---------------------------
    section("R0.3  render grid (env, several seeds) vs real frames")
    try:
        import h5py
        import hdf5plugin  # noqa: F401
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(args.seed)
        env_frames = []
        for s in range(3):
            env.reset(seed=s)
            env_frames.append(np.asarray(env.render()))
        say("env render shape/dtype:", env_frames[0].shape, env_frames[0].dtype)
        with h5py.File(h5_path, "r") as f:
            picks = np.sort(rng.choice(f["pixels"].shape[0], 3, replace=False))
            real = f["pixels"][picks]
        fig, ax = plt.subplots(2, 3, figsize=(9, 6.4))
        for i in range(3):
            ax[0, i].imshow(real[i]); ax[0, i].set_title(f"real frame {int(picks[i])}")
            ax[1, i].imshow(env_frames[i]); ax[1, i].set_title(f"swm reset seed {i}")
        for a in ax.flat:
            a.axis("off")
        fig.suptitle("training frames (top) vs swm/TwoRoom-v1 renders (bottom)")
        fig.tight_layout()
        fig.savefig("realenv_r0_eyeball.png", dpi=120)
        say("wrote realenv_r0_eyeball.png -- LOOK AT IT: border ticks, door,")
        say("dot size/softness, wall thickness.")
    except Exception:
        say("R0.3 failed:")
        say(traceback.format_exc())

    # ---- R0.4: THE DOMAIN GUARD -------------------------------------------
    section("R0.4  domain guard: env renders vs the training latent cloud")
    try:
        import h5py
        import hdf5plugin  # noqa: F401
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

        rng = np.random.default_rng(args.seed)
        with h5py.File(h5_path, "r") as f:
            picks = np.sort(rng.choice(f["pixels"].shape[0],
                                       min(args.frames, f["pixels"].shape[0]),
                                       replace=False))
            real = f["pixels"][picks]
        envf = []
        for s in range(args.frames):
            env.reset(seed=1000 + s)
            envf.append(np.asarray(env.render()))
        z_real = encode(real)
        z_env = encode(envf)

        def nn(a, b, ex=False):
            d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
            if ex:
                np.fill_diagonal(d, np.inf)
            return d.min(1)

        e2r = nn(z_env, z_real)
        r2r = nn(z_real, z_real, ex=True)
        say(f"env render -> nearest real latent : median {np.median(e2r):.2f}  "
            f"mean {e2r.mean():.2f}")
        say(f"real -> nearest other real        : median {np.median(r2r):.2f}")
        say(f"centroid distance                 : "
            f"{np.linalg.norm(z_real.mean(0) - z_env.mean(0)):.2f}")
        say("")
        say("YARDSTICKS: the toy fixture scored 61.03 against real spacing 2.43")
        say("(25x off-manifold) and every planner number it produced was an")
        say("artifact. READ:")
        say("  env->real ~ real spacing  : env frames are ON the training")
        say("    manifold even with default variations -> full green light.")
        say("  env->real elevated but << 61 : close; likely the sampled door /")
        say("    agent-radius mismatch -> pin the dataset's layout (R1) and")
        say("    re-run this guard; expect it to drop to ~spacing.")
        say("  env->real ~ 61 : the installed env version does not produce")
        say("    the dataset's visual style -- STOP; a version/config hunt is")
        say("    needed before any planning number can mean anything.")
    except Exception:
        say("R0.4 failed:")
        say(traceback.format_exc())

    Path("realenv_r0_report.txt").write_text("\n".join(REPORT))
    say("\nwrote realenv_r0_report.txt")


if __name__ == "__main__":
    main()
