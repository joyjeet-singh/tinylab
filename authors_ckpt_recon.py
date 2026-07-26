"""
authors_ckpt_recon.py (v2) -- discover, load and interface-check the authors'
released TwoRoom checkpoint.

WHY V2
------
v1 assumed two API details that I had read in the reference repo's eval.py but
had NOT executed: `get_cache_dir(path)` accepting a positional argument, and
`swm.wm.utils` being reachable as an attribute. Both failed on your machine.
The cause is version drift -- the sandbox I test in has stable-worldmodel 0.1.1,
where `get_cache_dir` accepts positionals and `swm.wm.utils` is auto-imported;
your install has neither. So v2 assumes nothing: it probes every entry point at
runtime and reports which ones exist.

*** DO NOT `pip install -U stable-worldmodel` TO FIX THIS. ***
The whole result chain -- R1's three zeros, the domain guard at 0.013, the
72/18 and 48/0 planner numbers, the balanced wall experiment -- was measured
against the environment as currently installed. Upgrading in place could change
the environment and silently invalidate all of it. If a newer version turns out
to be necessary, install it in a SEPARATE throwaway virtualenv used only to
convert the checkpoint to a plain state_dict, and leave this venv pinned.

WHAT THIS DOES
--------------
0. reports installed versions, so the drift is on the record
1. finds the cache directory, trying each calling convention in turn
2. enumerates every available route to a checkpoint loader
3. tries each (loader, name) combination until one works
4. reports the loaded object -- class, parameter count, mode
5. checks its interface against the four things our CEMPlanner calls
6. runs one real forward pass on an environment frame and prints shapes

It plans nothing and produces no success rates. If nothing loads it prints one
bounded manual route and stops; no paper claim depends on this experiment.
"""
from __future__ import annotations

import argparse
import importlib
import os
import traceback
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

NAMES = [
    "quentinll/lewm-tworooms",   # HF model repo named in the reference README
    "two-room/lewm",             # Drive layout + the README's policy= convention
    "tworooms/lewm",
    "tworoom/lewm",
    "tworooms",
]


def section(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def try_cache_dir(override, sub=None):
    """get_cache_dir's signature differs between versions -- try them all."""
    get_cache_dir = None
    for path in ("stable_worldmodel.data.utils", "stable_worldmodel.utils"):
        try:
            get_cache_dir = getattr(importlib.import_module(path),
                                    "get_cache_dir", None)
            if get_cache_dir:
                break
        except Exception:
            continue
    if get_cache_dir is None:
        return None, "get_cache_dir not importable"

    attempts = [
        ("positional", lambda: get_cache_dir(override, sub) if sub is not None
         else get_cache_dir(override)),
        ("keyword", lambda: get_cache_dir(override_root=override,
                                          sub_folder=sub)),
        ("no-args", lambda: get_cache_dir()),
    ]
    last = "no calling convention accepted"
    for how, fn in attempts:
        try:
            return Path(fn()), how
        except TypeError as ex:
            last = f"TypeError: {ex}"
            continue
        except Exception as ex:
            return None, f"{type(ex).__name__}: {ex}"
    return None, last


def find_loaders():
    """Every route to a checkpoint loader that actually exists here."""
    out = []
    for path, attr, label in [
        ("stable_worldmodel.wm", "load_pretrained", "swm.wm.load_pretrained"),
        ("stable_worldmodel.wm.utils", "load_pretrained",
         "stable_worldmodel.wm.utils.load_pretrained"),
        ("stable_worldmodel.utils", "load_pretrained",
         "stable_worldmodel.utils.load_pretrained"),
        ("stable_worldmodel.policy", "AutoCostModel",
         "swm.policy.AutoCostModel"),
    ]:
        try:
            fn = getattr(importlib.import_module(path), attr, None)
            if callable(fn):
                out.append((label, fn))
        except Exception:
            pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default=None,
                   help="single checkpoint name to try instead of the list")
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()

    import torch

    # ---- 0 ---------------------------------------------------------------
    section("0. installed versions (record these)")
    try:
        from importlib.metadata import version
        print(f"  stable-worldmodel : {version('stable-worldmodel')}   "
              f"(my sandbox has 0.1.1)")
    except Exception:
        print("  stable-worldmodel : version unavailable")
    print(f"  torch             : {torch.__version__}")
    try:
        import hydra  # noqa: F401
        print("  hydra             : present")
        have_hydra = True
    except Exception:
        print("  hydra             : MISSING")
        have_hydra = False
    print("  NOTE: do not upgrade this package in place -- every result in")
    print("  this project was measured against the environment as installed.")
    if not have_hydra:
        print("\n  STOP: load_pretrained builds the model via hydra.instantiate,")
        print("  and hydra is not a stable-worldmodel dependency. Install it:")
        print("      pip install hydra-core")
        print("  That is an ADDITIVE install of a new package, not an upgrade")
        print("  of anything this project's results depend on -- safe. Then")
        print("  re-run this script.")
        return

    # ---- 1 ---------------------------------------------------------------
    section("1. cache locations")
    root, how = try_cache_dir(args.cache_dir)
    ck, how2 = try_cache_dir(args.cache_dir, "checkpoints")
    print(f"  STABLEWM_HOME env : {os.environ.get('STABLEWM_HOME', '(unset)')}")
    print(f"  cache root        : {root}  (via {how})")
    print(f"  checkpoint root   : {ck}  (via {how2})")
    seen = set()
    for d in (root, ck, Path.home() / ".stable-wm",
              Path.home() / ".stable_worldmodel"):
        if d and Path(d).exists() and str(d) not in seen:
            seen.add(str(d))
            items = sorted(x.name for x in Path(d).iterdir())[:20]
            print(f"    {d}: {items if items else '(empty)'}")

    if ck and Path(ck).exists():
        stale = [d for d in Path(ck).glob("models--*")
                 if d.is_dir() and not list(d.rglob("*.pt"))]
        if stale:
            print("\n  PARTIAL DOWNLOAD DETECTED. A previous attempt created")
            print("  these cache folders without weights in them; the loader")
            print("  will now 'load from local cache' and fail confusingly.")
            for d in stale:
                print(f"    rm -rf '{d}'")
            print("  Run those, then re-run this script.")

    # ---- 2 ---------------------------------------------------------------
    section("2. available loader routes")
    loaders = find_loaders()
    if not loaders:
        print("  NONE FOUND. Nothing below can work -- send me this output.")
        return
    for label, _ in loaders:
        print(f"  available: {label}")

    # ---- 3 ---------------------------------------------------------------
    section("3. load attempts (the first may download)")
    names = [args.name] if args.name else NAMES
    model, used = None, None
    for label, fn in loaders:
        for nm in names:
            try:
                print(f"  {label}({nm!r}) ...", flush=True)
                try:
                    m = fn(nm, cache_dir=args.cache_dir) if args.cache_dir \
                        else fn(nm)
                except TypeError:
                    m = fn(nm)
                model, used = m, f"{label}({nm!r})"
                print(f"  SUCCESS: {used}")
                break
            except Exception as ex:
                first = (str(ex).splitlines() or [""])[0]
                print(f"    {type(ex).__name__}: {first[:150]}")
        if model is not None:
            break

    if model is None:
        section("NOTHING LOADED -- the one bounded manual route")
        print("  1. Open https://huggingface.co/quentinll/lewm-tworooms and")
        print("     download the weights file plus config.json.")
        print("  2. Put both in one folder under the checkpoint root above,")
        print(f"     e.g. {ck or '~/.stable-wm/checkpoints'}/tworooms/")
        print("  3. Re-run: python3 authors_ckpt_recon.py --name tworooms/")
        print("  If that fails too, STOP and send me the output. This")
        print("  experiment is optional; no paper claim depends on it.")
        return

    # ---- 4 ---------------------------------------------------------------
    section("4. the loaded object")
    print(f"  loaded via   : {used}")
    print(f"  class        : {type(model).__name__} ({type(model).__module__})")
    try:
        n = sum(q.numel() for q in model.parameters())
        print(f"  parameters   : {n/1e6:.2f} M   (paper says ~15M)")
        print(f"  training mode: {model.training}  (want False)")
    except Exception:
        print("  (not an nn.Module -- report what it is)")

    # ---- 5 ---------------------------------------------------------------
    section("5. interface check against our CEMPlanner")
    for attr, why in (("encode", "model.encode({'pixels': ...})['emb']"),
                      ("predict", "model.predict(emb, act_emb)"),
                      ("action_encoder", "model.action_encoder(actions)"),
                      ("history_size", "frames of context the planner feeds")):
        have = hasattr(model, attr)
        extra = (f" = {getattr(model, attr)}"
                 if have and attr == "history_size" else "")
        print(f"  {attr:<15} {'present' if have else 'MISSING':<8}{extra}"
              f"   ({why})")
    if not hasattr(model, "history_size"):
        print("\n  history_size missing is expected: their training config sets")
        print("  3 and their rollout() defaults to 3, so the adapter sets it to")
        print("  3 and that goes in the deviations table.")

    # ---- 6 ---------------------------------------------------------------
    section("6. forward pass on a real environment frame (shapes only)")
    try:
        import numpy as np
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        env = gym.make("swm/TwoRoom-v1", disable_env_checker=True)
        u = env.unwrapped
        env.reset(seed=0)
        u._set_state(np.array([60.0, 120.0], dtype=np.float32))
        frame = np.asarray(u.render(), dtype=np.float32) / 255.0
        px = torch.from_numpy(frame).permute(2, 0, 1)[None, None]
        with torch.no_grad():
            emb = model.encode({"pixels": px})["emb"]
            print(f"  encode         -> emb {tuple(emb.shape)}  "
                  f"norm {float(emb.norm()):.3f}")
            ae = model.action_encoder(torch.zeros(1, emb.shape[1], 2))
            print(f"  action_encoder -> {tuple(ae.shape)}")
            pred = model.predict(emb, ae)
            print(f"  predict        -> {tuple(pred.shape)}")
        ok = pred.shape[-1] == emb.shape[-1]
        print(f"\n  prediction lives in the embedding space: {ok}")
        if ok:
            print("  => their checkpoint drops into our planner with an adapter")
            print("     that only sets history_size. Step 2 is one run of the")
            print("     existing evaluation with their weights and nothing else")
            print("     changed.")
    except Exception:
        print("  forward pass failed:")
        print(traceback.format_exc())
        print("  Send me this; the shape fix is small but only worth making if")
        print("  everything above looked healthy.")


if __name__ == "__main__":
    main()
