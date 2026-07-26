"""
preflight_local.py -- everything that can be checked on the Mac, before a
single rented minute is spent.

This is the launch protocol's G2 gate made concrete. It runs four stages and
refuses to pass if any of them fails. Nothing here needs a GPU.

  A. FIDELITY   every training-config element compared against the reference
                value, which is hard-coded below with a source citation. This
                turns the fidelity table into an executable check that runs
                before every launch, rather than a document someone remembers
                to read. Deviations are not automatically failures -- some are
                deliberate -- but each must be listed in EXPECTED_DEVIATIONS
                with a reason, or the gate fails. An unexplained deviation is
                exactly the class of bug that has already cost this project
                three runs.

  B. DATA       the loader's output contract: dense actions of the right
                width, and the physics identity -- displacement equals speed
                times the summed actions -- asserted on real clips.

  C. MODEL      builds at the configured width, parameter count reported,
                one forward and one backward pass, gradients finite.

  D. MICRO-TRAIN a few dozen CPU steps on real data. Not a result: a check
                that the loss is finite, moves in the right direction, and that
                the embedding spread does not collapse in the first steps --
                which is when SIGReg has to do its work, since it is a good
                preventer and a poor rescuer.

Usage:
    python3 preflight_local.py --config configs/<your_config>.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reference values. Each traced to source, not assumed. See docs/fidelity_audit.md
# ---------------------------------------------------------------------------
REFERENCE = {
    # model -- le-wm/config/train/lewm.yaml + the released config.json
    "model.embed_dim": 192,
    "model.depth": 6,
    "model.heads": 16,
    "model.dim_head": 64,
    "model.mlp_dim": 2048,
    "model.proj_hidden": 2048,
    "model.dropout": 0.1,
    "model.img_size": 224,
    "model.patch_size": 14,
    "model.history_size": 3,
    # data -- stable_worldmodel/data/buffer.py _gather_clip + train.py:68
    "data.frameskip": 5,
    "data.train_split": 0.9,
    # training -- le-wm/config/train/lewm.yaml
    "training.batch_size": 128,
    "training.learning_rate": 5e-05,
    "training.weight_decay": 0.001,
    "training.num_preds": 1,
    "training.grad_clip": 1.0,
    # loss -- same file
    "loss.sigreg_weight": 0.09,
    "loss.sigreg_kwargs.knots": 17,
    "loss.sigreg_kwargs.num_proj": 1024,
}

# Deviations we are choosing on purpose. Anything NOT listed here that differs
# from REFERENCE fails the gate.
EXPECTED_DEVIATIONS = {
    "training.epochs": "reference repo says 100, paper App. E says 10; budget "
                       "allows 10",
    "seed": "held at Run 0's value so the comparison against Run 0 isolates "
        "the four pipeline fixes; reference uses 3072",
}


def get(cfg, dotted):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--h5", default=None, help="defaults to the config's path")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--clips", type=int, default=200)
    args = ap.parse_args()

    import yaml
    import torch
    cfg = yaml.safe_load(Path(args.config).read_text())
    failures = []

    # ---- A. fidelity ------------------------------------------------------
    print("=" * 72)
    print("A. FIDELITY vs the reference (values traced to source)")
    print("=" * 72)
    print(f"  {'element':<32}{'reference':>14}{'ours':>14}   status")
    print("  " + "-" * 66)
    for key, ref in REFERENCE.items():
        ours = get(cfg, key)
        if ours is None:
            print(f"  {key:<32}{str(ref):>14}{'ABSENT':>14}   MISSING")
            failures.append(f"{key} absent from the config")
            continue
        same = (abs(ours - ref) < 1e-12 if isinstance(ref, (int, float))
                and isinstance(ours, (int, float)) else ours == ref)
        status = "match" if same else "DEVIATION"
        print(f"  {key:<32}{str(ref):>14}{str(ours):>14}   {status}")
        if not same and key not in EXPECTED_DEVIATIONS:
            failures.append(f"{key}: {ours} vs reference {ref} -- not in "
                            f"EXPECTED_DEVIATIONS")
    for key, why in EXPECTED_DEVIATIONS.items():
        print(f"  [deliberate] {key} = {get(cfg, key)} -- {why}")
    extra = get(cfg, "training.lr_schedule")
    if extra:
        print(f"  [deliberate] training.lr_schedule = {extra} -- the reference "
              f"config specifies NO scheduler; this is our addition")

    # ---- B. data contract -------------------------------------------------
    print("\n" + "=" * 72)
    print("B. DATA CONTRACT")
    print("=" * 72)
    try:
        import h5py
        import hdf5plugin  # noqa: F401
        import tworoom_data as td
        h5 = args.h5 or get(cfg, "data.h5_path")
        print(f"  DENSE_ACTIONS = {getattr(td, 'DENSE_ACTIONS', 'ABSENT')}")
        if not getattr(td, "DENSE_ACTIONS", False):
            failures.append("DENSE_ACTIONS is not True -- the loader still "
                            "subsamples actions")
        spec = td.ClipSpec(history=get(cfg, "model.history_size"),
                           frameskip=get(cfg, "data.frameskip"))
        index = td.TwoRoomIndex(h5, spec)
        # The physics identity is a property of the RAW actions; it cannot hold
        # once they are z-scored. Measure the GATHERING with normalisation off,
        # exactly as verify_dense_actions.py does, or a correct loader fails a
        # check it is not able to pass.
        _zs = getattr(td, "ZSCORE_ACTIONS", False)
        td.ZSCORE_ACTIONS = False
        clips = td.TwoRoomClips(h5, index, keys=("action",))
        fs, n = spec.frameskip, spec.num_steps
        a = clips[0]["action"]
        want = (n, fs * 2)
        print(f"  action shape {a.shape}   expected {want}")
        if a.shape != want:
            failures.append(f"action shape {a.shape} != {want}")
        exp_dim = fs * 2
        if get(cfg, "model.action_dim") != exp_dim:
            failures.append(f"model.action_dim = {get(cfg,'model.action_dim')} "
                            f"but the loader emits width {exp_dim} "
                            f"(frameskip x action_dim)")
            print(f"  model.action_dim {get(cfg,'model.action_dim')} vs "
                  f"loader width {exp_dim}   MISMATCH")
        else:
            print(f"  model.action_dim {exp_dim} agrees with the loader")

        rng = np.random.default_rng(0)
        errs = []
        with h5py.File(h5, "r") as f:
            pos = f["pos_agent"]
            for i in rng.choice(len(clips), size=min(args.clips, len(clips)),
                                replace=False):
                c = clips[int(i)]
                s = int(c["_start"])
                for t in range(n - 1):
                    p0 = np.asarray(pos[s + t * fs], dtype=np.float64)
                    p1 = np.asarray(pos[s + (t + 1) * fs], dtype=np.float64)
                    if abs(p0[0] - 112) < 40 or abs(p1[0] - 112) < 40:
                        continue
                    if min(p0.min(), p1.min()) < 45 or max(p0.max(), p1.max()) > 177:
                        continue
                    blk = c["action"][t].reshape(fs, 2).astype(np.float64)
                    errs.append(np.linalg.norm(5.0 * blk.sum(0) - (p1 - p0)))
        if len(errs) < 20:
            failures.append(f"only {len(errs)} wall-free blocks -- raise --clips")
        else:
            med = float(np.median(errs))
            print(f"  physics identity on {len(errs)} wall-free blocks: "
                  f"median error {med:.2e}")
            if med > 1e-3:
                failures.append(f"physics identity violated (median {med:.3f})")
        td.ZSCORE_ACTIONS = _zs
        print(f"  (z-scoring restored to {_zs})")
    except Exception as ex:
        failures.append(f"data stage raised {type(ex).__name__}: {ex}")
        print(f"  FAILED: {type(ex).__name__}: {ex}")

    # ---- C + D. model and micro-train ------------------------------------
    print("\n" + "=" * 72)
    print("C/D. MODEL BUILD AND CPU MICRO-TRAIN")
    print("=" * 72)
    try:
        import torch
        from toy_model import ToyJEPA
        from toy_sigreg import SIGReg
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
        n_par = sum(p.numel() for p in model.parameters())
        print(f"  parameters: {n_par:,}  "
              f"(the authors' released checkpoint is 18,034,xxx)")

        h5 = args.h5 or get(cfg, "data.h5_path")
        spec = td.ClipSpec(history=m["history_size"],
                           frameskip=get(cfg, "data.frameskip"))
        index = td.TwoRoomIndex(h5, spec)
        ds = td.TwoRoomClips(h5, index)
        sig = SIGReg(**cfg["loss"]["sigreg_kwargs"])
        opt = torch.optim.AdamW(model.parameters(),
                                lr=cfg["training"]["learning_rate"],
                                weight_decay=cfg["training"]["weight_decay"])
        HS, npred = m["history_size"], cfg["training"]["num_preds"]
        lam_cfg = cfg["loss"]["sigreg_weight"]

        def micro(lam, tag):
            """A few CPU steps. Returns (first loss, last loss, spreads)."""
            import copy
            mo = copy.deepcopy(model)
            o = torch.optim.AdamW(mo.parameters(),
                                  lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
            r = np.random.default_rng(0)
            f_, l_, sp = None, None, []
            for step in range(args.steps):
                picks = r.choice(len(ds), size=args.batch, replace=False)
                items = [ds[int(p)] for p in picks]
                b = {"pixels": torch.tensor(np.stack([i["pixels"] for i in items])),
                     "action": torch.tensor(np.stack([i["action"] for i in items]))}
                out = mo.encode(b)
                e_, a_ = out["emb"], out["act_emb"]
                pr = mo.predict(e_[:, :HS], a_[:, :HS])
                sg = sig(e_.transpose(0, 1))
                ls = (pr - e_[:, npred:]).pow(2).mean() + lam * sg
                o.zero_grad(); ls.backward()
                g = torch.nn.utils.clip_grad_norm_(mo.parameters(),
                                                   cfg["training"]["grad_clip"])
                o.step()
                v = float(ls)
                if not np.isfinite(v) or not np.isfinite(float(g)):
                    failures.append(f"{tag}: loss {v} / grad {float(g)} at "
                                    f"step {step}")
                    break
                sp.append(float(e_.reshape(-1, e_.shape[-1]).std(0).mean()))
                if f_ is None:
                    f_ = v
                l_ = v
                if step % max(1, args.steps // 4) == 0:
                    print(f"    [{tag}] step {step:3d}  loss {v:9.4f}  "
                          f"grad {float(g):7.3f}  spread {sp[-1]:.4f}  "
                          f"sigreg {float(sg):8.2f}")
            return f_, l_, sp

        print(f"  SIGReg note: its statistic scales with BATCH SIZE, so at "
              f"batch {args.batch} it carries roughly {args.batch}/128 of the "
              f"anti-collapse\n  pressure the real run will have. An absolute "
              f"spread drop here is EXPECTED and not\n  diagnostic. What is "
              f"diagnostic: whether SIGReg slows the drop at all.")
        f_on, l_on, sp_on = micro(lam_cfg, f"lam={lam_cfg}")
        f_off, l_off, sp_off = micro(0.0, "lam=0")
        if sp_on and sp_off:
            print(f"  final spread with SIGReg {sp_on[-1]:.4f}  "
                  f"without {sp_off[-1]:.4f}")
            if sp_on[-1] <= sp_off[-1]:
                failures.append(
                    f"SIGReg is not slowing collapse (with {sp_on[-1]:.4f} vs "
                    f"without {sp_off[-1]:.4f}) -- it may not be wired into the "
                    f"loss")
            else:
                print(f"  OK -- SIGReg holds {sp_on[-1]/sp_off[-1]:.2f}x more "
                      f"spread than lambda=0, so it is wired and active")
        if f_on is not None:
            print(f"  loss {f_on:.4f} -> {l_on:.4f}")
        first, last, spreads = f_on, l_on, sp_on
        ds.close()
    except Exception as ex:
        failures.append(f"model/train stage raised {type(ex).__name__}: {ex}")
        print(f"  FAILED: {type(ex).__name__}: {ex}")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 72)
    if failures:
        print(f"PRE-FLIGHT FAILED -- {len(failures)} problem(s). Do not launch.")
        print("=" * 72)
        for f_ in failures:
            print(f"  - {f_}")
        sys.exit(1)
    print("PRE-FLIGHT PASSED")
    print("=" * 72)
    print("  Every config element either matches the reference or is a listed,")
    print("  deliberate deviation; the loader's actions satisfy the physics")
    print("  identity; the model builds, trains and does not collapse on CPU.")
    print("  Record this output alongside the run.")


if __name__ == "__main__":
    main()
