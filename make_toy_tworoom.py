"""
make_toy_tworoom.py -- a small, learnable stand-in for the real Two-Room data.

Why this exists
---------------
The real dataset is 12 GB and takes ~200 hours just to READ on a laptop. We
cannot debug a model against it. But we also cannot debug against random noise:
a model fed noise learns nothing, so it would tell us nothing about whether our
anti-collapse machinery works -- which is the single most important thing to get
right.

So we make a toy world with three properties:

  1. THE SAME FILE FORMAT as the real one. Flat arrays, ep_offset/ep_len,
     actions as displacements scaled by 5, pixels stored (H, W, C) as whole
     numbers 0-255. This means `tworoom_data.py` reads it UNCHANGED -- the
     verified loader stays verified. We are not touching it.

  2. GENUINELY LEARNABLE. A red dot in two rooms with a wall and a door,
     moving exactly as the actions say. A working model should learn to predict
     where the dot goes. A broken one should visibly fail.

  3. SMALL AND UNCOMPRESSED. 32x32 pixels, a few hundred episodes, no Blosc.
     Loads instantly. Debug for free.

What it is NOT
--------------
This is a debugging fixture, not data. Nothing measured on it is a result, and
nothing about it carries into the reproduction. The moment we rent a machine,
this file is deleted and the real data takes its place. Its only job is to let
us find bugs without a clock running.

The world
---------
A square room split by a vertical wall with a door gap in it. The agent (a red
dot) starts somewhere random on one side and must reach a target on the other
side, which means going through the door. The behaviour policy copies what the
real dataset's does (paper Appendix E): head for the door in a straight line,
then head for the target once through -- with noise added, so trajectories vary.

Run:
    python make_toy_tworoom.py --out toy_tworoom.h5
    python tworoom_data.py --h5 toy_tworoom.h5      # the SAME checks should pass
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# world constants -- deliberately mirrors the real world's conventions
# ---------------------------------------------------------------------------
ACTION_SCALE = 5.0      # a magnitude-1 action moves the dot 5 units (as measured
                        # in the real data: actions clamp at +/-1, moves hit +/-5)
ACTION_CLIP = 1.0       # actions are clamped to [-1, 1] (as in the real data)


class ToyTwoRoom:
    """
    A square arena, size x size units, split by a vertical wall at the middle.
    The wall has a door: a gap the agent must pass through.

    Coordinates are in WORLD units, y counting UP. Rendering flips y, because
    image rows count DOWN -- exactly the convention we measured in the real
    dataset. We reproduce the flip on purpose so the toy behaves like the real
    thing.
    """

    def __init__(self, size: float = 40.0, door_half: float = 5.0,
                 margin: float = 4.0):
        self.size = size
        self.wall_x = size / 2.0            # wall runs vertically at the middle
        self.door_y = size / 2.0            # door centred vertically
        self.door_half = door_half          # half-height of the door gap
        self.margin = margin                # keep the dot away from the edges

    # -- geometry ----------------------------------------------------------
    def in_door(self, y: float) -> bool:
        return abs(y - self.door_y) <= self.door_half

    def blocked(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        """Would moving p0 -> p1 cross the wall somewhere other than the door?"""
        x0, x1 = p0[0], p1[0]
        if (x0 - self.wall_x) * (x1 - self.wall_x) > 0:
            return False                    # both on the same side: fine
        if x1 == x0:
            return False
        # where does the path cross the wall's x?
        t = (self.wall_x - x0) / (x1 - x0)
        y_cross = p0[1] + t * (p1[1] - p0[1])
        return not self.in_door(y_cross)

    def step(self, pos: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Apply one action. Clamp, scale, block at the wall, clip to arena."""
        a = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
        nxt = pos + a * ACTION_SCALE
        nxt = np.clip(nxt, self.margin, self.size - self.margin)
        if self.blocked(pos, nxt):
            return pos.copy()               # bumped the wall: stay put
        return nxt

    # -- rendering ---------------------------------------------------------
    def render(self, pos: np.ndarray, img_size: int) -> np.ndarray:
        """Draw the scene as (H, W, 3) whole numbers 0-255, like the real data."""
        img = np.full((img_size, img_size, 3), 255, dtype=np.uint8)
        s = img_size / self.size            # world units -> pixels

        def to_px(p):
            # NOTE the y flip: image rows count DOWN, world y counts UP.
            return int(p[0] * s), int((self.size - p[1]) * s)

        # -- wall (black), with the door left white
        wx = int(self.wall_x * s)
        w_half = max(1, img_size // 64)
        for row in range(img_size):
            y_world = self.size - (row / s)
            if not self.in_door(y_world):
                img[row, max(0, wx - w_half):min(img_size, wx + w_half + 1)] = 0

        # -- border
        b = max(1, img_size // 64)
        img[:b, :] = 0; img[-b:, :] = 0; img[:, :b] = 0; img[:, -b:] = 0

        # -- agent (a soft red dot, so sub-pixel movement is visible)
        cx, cy = to_px(pos)
        r = max(1.5, img_size / 16.0)
        ys, xs = np.mgrid[0:img_size, 0:img_size]
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        mask = np.exp(-d2 / (2 * (r / 2) ** 2))
        img[..., 1] = np.minimum(img[..., 1], (255 * (1 - mask)).astype(np.uint8))
        img[..., 2] = np.minimum(img[..., 2], (255 * (1 - mask)).astype(np.uint8))
        return img


# ---------------------------------------------------------------------------
# the behaviour policy -- copies the real dataset's (paper Appendix E)
# ---------------------------------------------------------------------------
def heuristic_action(env: ToyTwoRoom, pos: np.ndarray, target: np.ndarray,
                     rng: np.random.Generator, noise: float = 0.25) -> np.ndarray:
    """
    "First head for the door in a straight line, then head for the target
    once through." Plus noise, so trajectories vary and the data isn't trivial.
    """
    same_side = (pos[0] - env.wall_x) * (target[0] - env.wall_x) > 0
    if same_side or env.in_door(pos[1]) and abs(pos[0] - env.wall_x) < 8:
        aim = target
    else:
        aim = np.array([env.wall_x, env.door_y])   # go to the door first

    d = aim - pos
    n = np.linalg.norm(d)
    step = d / n if n > 1e-6 else np.zeros(2)
    step = step + rng.normal(0, noise, size=2)
    return np.clip(step, -ACTION_CLIP, ACTION_CLIP).astype(np.float32)


def rollout(env: ToyTwoRoom, rng: np.random.Generator, max_len: int = 92):
    """
    One episode. Random start on one side, random target on the other.

    Episodes run a FIXED length (~92 steps), matching the real dataset's mean of
    92.1 -- the real behaviour policy does not stop on arrival. Once the agent
    reaches the target we re-target it somewhere new, so it keeps moving and the
    later frames are not a frozen dot (which would teach the model that "nothing
    happens", and make collapse look like a good solution).
    """
    left = rng.random() < 0.5
    lo, hi = env.margin, env.size - env.margin

    def sample(on_left):
        x = (rng.uniform(lo, env.wall_x - 4) if on_left
             else rng.uniform(env.wall_x + 4, hi))
        return np.array([x, rng.uniform(lo, hi)], dtype=np.float32)

    pos = sample(left)
    target = sample(not left)
    target_side_left = not left

    positions, actions, targets = [], [], []
    for _ in range(max_len):
        a = heuristic_action(env, pos, target, rng)
        positions.append(pos.copy())
        actions.append(a)
        targets.append(target.copy())
        pos = env.step(pos, a)
        if np.linalg.norm(pos - target) < 3.0:      # arrived -> pick a new goal
            target_side_left = not target_side_left
            target = sample(target_side_left)
    return (np.array(positions, np.float32), np.array(actions, np.float32),
            np.array(targets, np.float32))


# ---------------------------------------------------------------------------
# writing the file -- SAME layout as the real tworoom.h5
# ---------------------------------------------------------------------------
def build(out: str, n_episodes: int, img_size: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    env = ToyTwoRoom()

    all_pos, all_act, all_tgt, lens = [], [], [], []
    for _ in range(n_episodes):
        pos, act, tgt = rollout(env, rng)
        all_pos.append(pos)
        all_act.append(act)
        all_tgt.append(tgt)
        lens.append(len(pos))

    pos = np.concatenate(all_pos).astype(np.float32)
    act = np.concatenate(all_act).astype(np.float32)
    tgt = np.concatenate(all_tgt).astype(np.float32)
    lens = np.array(lens, dtype=np.int32)
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    total = int(lens.sum())

    ep_idx = np.zeros(total, np.int32)
    step_idx = np.zeros(total, np.int64)
    for e, (o, L) in enumerate(zip(offs, lens)):
        ep_idx[o:o + L] = e
        step_idx[o:o + L] = np.arange(L)

    px = np.zeros((total, img_size, img_size, 3), np.uint8)
    for i in range(total):
        px[i] = env.render(pos[i], img_size)

    with h5py.File(out, "w") as f:
        # exact key names + dtypes from the real file, so the loader is unchanged
        f.create_dataset("pixels", data=px)                       # uint8 (N,H,W,3)
        f.create_dataset("action", data=act)                      # float32 (N,2)
        f.create_dataset("pos_agent", data=pos)                   # float32 (N,2)
        f.create_dataset("pos_target", data=tgt)                  # float32 (N,2)
        f.create_dataset("proprio", data=pos.copy())              # float32 (N,2)
        f.create_dataset("ep_offset", data=offs)                  # int64 (E,)
        f.create_dataset("ep_len", data=lens)                     # int32 (E,)
        f.create_dataset("ep_idx", data=ep_idx)                   # int32 (N,)
        f.create_dataset("step_idx", data=step_idx)               # int64 (N,)

    return {"frames": total, "episodes": n_episodes,
            "mean_len": float(lens.mean()), "img_size": img_size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="toy_tworoom.h5")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--preview", action="store_true", help="save a picture strip")
    args = p.parse_args()

    info = build(args.out, args.episodes, args.img_size, args.seed)
    import os
    print(f"wrote {args.out}")
    print(f"  frames    : {info['frames']:,}")
    print(f"  episodes  : {info['episodes']:,} (mean length {info['mean_len']:.1f})")
    print(f"  image size: {info['img_size']}x{info['img_size']}")
    print(f"  file size : {os.path.getsize(args.out)/1e6:.1f} MB")
    print()
    print("Now run the SAME verified checks against it:")
    print(f"  python tworoom_data.py --h5 {args.out}")

    if args.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with h5py.File(args.out, "r") as f:
            n = 8
            px = np.asarray(f["pixels"][:n])
            ac = np.asarray(f["action"][:n])
            pa = np.asarray(f["pos_agent"][:n])
        fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.6))
        for i, ax in enumerate(axes):
            ax.imshow(px[i])
            mv = "" if i == 0 else f"\nmoved {np.round(pa[i]-pa[i-1],1)}"
            ax.set_title(f"t={i}\nact {np.round(ac[i],2)}{mv}", fontsize=7)
            ax.axis("off")
        plt.tight_layout(); plt.savefig("toy_preview.png", dpi=110); plt.close()
        print("  wrote toy_preview.png")


if __name__ == "__main__":
    main()
