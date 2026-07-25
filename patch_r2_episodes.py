"""
patch_r2_episodes.py -- add an --episodes flag to realenv_r2_planner_eval.py
so the evaluation can consume a pre-selected, committed episode list instead
of sampling at random.

WHY A PATCH, AND WHY THIS ONE IS SAFE
--------------------------------------
The Run-2 learning-rate incident happened because two patch wordings each
passed their own uniqueness assertion and both applied, silently doubling a
scheduler step. This patch is built so that cannot happen:

  - it FIRST checks whether the change is already present and exits without
    touching the file if so (idempotent);
  - every target string must appear EXACTLY ONCE or the patch aborts;
  - it writes a .bak, prints the diff it made, and byte-compiles the result.

Three edits: the argparse flag, the episode-selection block, and carrying the
geometry/pair labels through into the results so downstream analysis needs no
re-derivation.

Usage (from the tinylab folder):
    python3 patch_r2_episodes.py
"""
from __future__ import annotations

import difflib
import py_compile
import sys
from pathlib import Path

TARGET = Path("realenv_r2_planner_eval.py")

MARKER = '--episodes'

EDITS = [
    # (name, old, new)
    ("argparse flag",
     '''    p.add_argument("--unsafe-skip-guard", action="store_true",''',
     '''    p.add_argument("--episodes", default=None,
                   help="JSON file from make_balanced_episode_set.py: use a "
                        "committed, auditable episode list instead of random "
                        "sampling")
    p.add_argument("--unsafe-skip-guard", action="store_true",'''),

    ("episode selection",
     '''    rng = np.random.default_rng(args.seed)
    with h5py.File(h5_path, "r") as f:
        off = np.asarray(f["ep_offset"][:])
        ln = np.asarray(f["ep_len"][:])
        ok = np.where(ln > args.goal_offset)[0]
        eps = rng.choice(ok, size=min(args.num_eval, len(ok)), replace=False)
        starts, goals = [], []''',
     '''    rng = np.random.default_rng(args.seed)
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
    with h5py.File(h5_path, "r") as f:
        off = np.asarray(f["ep_offset"][:])
        ln = np.asarray(f["ep_len"][:])
        if meta is None:
            ok = np.where(ln > args.goal_offset)[0]
            eps = rng.choice(ok, size=min(args.num_eval, len(ok)),
                             replace=False)
            meta = [{} for _ in eps]
        starts, goals = [], []'''),

    ("carry labels into results",
     '''        results.append({"episode": int(e), "success": bool(success),
                        "steps": int(steps), "final_dist": round(dist, 2),
                        "start_goal_dist": round(sg, 2),
                        "trivial_start": bool(trivial),
                        "plans": int(plans)})''',
     '''        results.append({"episode": int(e), "success": bool(success),
                        "steps": int(steps), "final_dist": round(dist, 2),
                        "start_goal_dist": round(sg, 2),
                        "trivial_start": bool(trivial),
                        "plans": int(plans), **meta[i]})'''),
]


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()

    if MARKER in src:
        print(f"'{MARKER}' is already present in {TARGET} -- patch was "
              f"applied before. No change made.")
        return

    out = src
    for name, old, new in EDITS:
        cnt = out.count(old)
        if cnt != 1:
            sys.exit(f"ABORT: target for '{name}' appears {cnt} times "
                     f"(need exactly 1). Nothing written.")
        out = out.replace(old, new)

    if out.count(MARKER) != 1:
        sys.exit(f"ABORT: post-patch marker count is {out.count(MARKER)}, "
                 f"expected 1. Nothing written.")

    TARGET.with_suffix(".py.bak").write_text(src)
    TARGET.write_text(out)
    print("--- diff applied ---")
    for line in difflib.unified_diff(src.splitlines(), out.splitlines(),
                                     "before", "after", lineterm="", n=1):
        print(line)
    py_compile.compile(str(TARGET), doraise=True)
    print(f"\nOK: {TARGET} patched and byte-compiles. Backup at "
          f"{TARGET}.bak")
    print("Re-run this script to confirm it now reports 'already present'.")


if __name__ == "__main__":
    main()
