"""
train_toy_lewm.py -- the training loop, wired to the tinylab scaffold.

Usage:
    python train_toy_lewm.py --config configs/toy_lewm.yaml --seed 0
    python train_toy_lewm.py --config configs/toy_lewm.yaml --seed 0 --resume runs/<dir>

What this does, in plain terms
------------------------------
Repeats one small step many times: grab a batch of clips, ask the model to
predict each frame's summary from the frames before it, measure two things --
how wrong the prediction was, and how far the summaries have drifted from a
bell-curve shape -- add them up, and nudge the model to do better.

The second measurement is the anti-collapse term. Without it the model cheats by
making every summary identical (we reproduced that: prediction loss fell 245x
while the summaries became 45x less varied). It must be present from step one:
SIGReg's pull is WEAKEST exactly where collapse is worst, so it prevents well
and rescues poorly.

House rules, inherited from the scaffold
----------------------------------------
- Seed everything through seed.set_seed, which also refuses any operation with
  no deterministic version. A loud crash now beats a silently different number
  later.
- Write the manifest BEFORE any work happens. Conditions only, never results.
- One measurement per line, on disk the moment it is measured.
- Data seed and run seed are separate: the same data seen by every arm, with
  only the run's randomness differing.
- Fingerprint the data so a scorekeeper can prove every arm saw the same clips.

Checkpoint and resume
---------------------
Colab sessions die. A run must be splittable across sessions WITHOUT changing
the result. That needs three things saved, not one:
  1. the model's weights
  2. the optimizer's state (AdamW carries running averages per weight; drop
     these and every restart jolts the model)
  3. where we were in the data, and the random state
`check_resume_equivalence.py` proves a split run matches an unbroken one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import lablog
from parallel_data import make_loader
from seed import set_seed
from toy_model import ToyJEPA
from toy_sigreg import SIGReg
from tworoom_data import ClipSpec, TwoRoomClips, TwoRoomIndex


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def data_fingerprint(index: TwoRoomIndex, h5_path: str) -> str:
    """Content hash of the clip set: same clips <-> same hash, always."""
    h = hashlib.sha256()
    h.update(Path(h5_path).name.encode())
    h.update(str(index.spec).encode())
    h.update(index.starts.tobytes())
    return h.hexdigest()


def split_indices(n: int, train_split: float, data_seed: int):
    """Fixed train/val split, driven by the DATA seed (never the run seed)."""
    rng = np.random.default_rng(data_seed)
    perm = rng.permutation(n)
    cut = int(n * train_split)
    return perm[:cut], perm[cut:]


def make_batch(ds: TwoRoomClips, picks: np.ndarray) -> dict:
    items = [ds[int(p)] for p in picks]
    return {
        "pixels": torch.from_numpy(np.stack([i["pixels"] for i in items])),
        "action": torch.from_numpy(np.stack([i["action"] for i in items])),
    }


def epoch_order(train_idx: np.ndarray, epoch: int, run_seed: int) -> np.ndarray:
    """
    The clip order for a given epoch -- derived ONLY from (run_seed, epoch).

    Why not a running generator: if you carry one generator across epochs and
    resume mid-epoch, the generator has already advanced past that epoch's
    shuffle. Reshuffling with it produces a DIFFERENT order, so the second half
    of the epoch sees different clips than an unbroken run would have. Same
    seed, different data -- which is exactly the bug the resume-equivalence gate
    caught. Deriving each epoch's order from (seed, epoch) makes it reproducible
    from scratch at any point, so resume lands on the identical order.
    """
    g = torch.Generator().manual_seed(run_seed * 100_003 + epoch)
    return train_idx[torch.randperm(len(train_idx), generator=g).numpy()]


# ---------------------------------------------------------------------------
# the loss -- faithful to reference train.py lejepa_forward
# ---------------------------------------------------------------------------
def lewm_step(model, sigreg, batch, ctx_len, n_preds, lambd):
    """
    One forward pass. Returns the losses and a couple of diagnostics.

    Note: the target is NOT detached. The reference computes
    (pred_emb - tgt_emb).pow(2).mean(), so gradient flows into both sides --
    the encoder is actively pulled toward making summaries easy to predict.
    That makes collapse MORE tempting, which is exactly why SIGReg matters.
    """
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    out = model.encode(batch)
    emb, act_emb = out["emb"], out["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]                       # label, NOT detached
    pred_emb = model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))        # (T,B,D): per timestep
    loss = pred_loss + lambd * sigreg_loss

    with torch.no_grad():
        spread = emb.reshape(-1, emb.size(-1)).std(0).mean().item()
    return loss, {"pred_loss": pred_loss.item(),
                  "sigreg_loss": sigreg_loss.item(),
                  "loss": loss.item(),
                  "spread": spread}


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------
def save_ckpt(path: Path, model, opt, step: int, epoch: int,
              pos_in_epoch: int) -> None:
    """
    Everything needed to carry on as if nothing happened.

    Note what is saved and why:
      - model weights          : obvious
      - optimizer state        : AdamW carries running averages per weight. Drop
                                 these and every restart jolts the model.
      - epoch + pos_in_epoch   : WHERE we were. Each epoch's clip order is
                                 derived from (seed, epoch), so we do not need
                                 to save a generator for shuffling -- we can
                                 regenerate the exact order and skip ahead.
      - global RNG state       : dropout draws from torch's GLOBAL generator.
                                 On resume, set_seed() resets that generator to
                                 the beginning, so the resumed steps would see
                                 the SAME dropout masks the first steps saw,
                                 while an unbroken run sees fresh ones. Every
                                 weight then diverges. Saving and restoring the
                                 state makes dropout continue rather than
                                 restart. (Found by the resume-equivalence gate.)
    """
    torch.save({
        "model": model.state_dict(),
        "optim": opt.state_dict(),
        "step": step,
        "epoch": epoch,
        "pos_in_epoch": pos_in_epoch,
        "torch_rng_state": torch.get_rng_state(),
    }, path)


def load_ckpt(path: Path, model, opt):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optim"])
    return ck


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, sigreg, ds, val_idx, cfg, max_batches=8):
    model.eval()
    bs = cfg["training"]["batch_size"]
    tots = {"pred_loss": 0.0, "sigreg_loss": 0.0, "spread": 0.0}
    n = 0
    for s in range(0, min(len(val_idx), max_batches * bs), bs):
        picks = val_idx[s:s + bs]
        if len(picks) < 2:                       # SIGReg needs a real batch
            break
        _, m = lewm_step(model, sigreg, make_batch(ds, picks),
                         cfg["model"]["history_size"], cfg["training"]["num_preds"],
                         cfg["loss"]["sigreg_weight"])
        for k in tots:
            tots[k] += m[k]
        n += 1
    model.train()
    return {k: round(v / max(n, 1), 6) for k, v in tots.items()}


@torch.no_grad()
def probe_r2(model, h5_path: str, n: int = 300, seed: int = 0):
    """
    The R^2 gate, logged every epoch so the learning curve is visible.

    Question: can the dot's true position be read back out of the model's
    summary with a simple linear read-out? Low R^2 = the encoder cannot see
    yet, and no planning number means anything. High R^2 = the encoder sees.

    Grounded in the FILE this run trains on: sample frames, read the recorded
    true position (`pos_agent`, present in both the toy and the real dataset),
    encode, fit. This works unchanged on the rented real-data run -- unlike a
    probe that re-renders a toy world.

    Returns None (and says so) rather than killing a long run if anything is
    missing, e.g. scikit-learn.
    """
    try:
        import h5py
        import hdf5plugin  # noqa: F401
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(seed)
        with h5py.File(h5_path, "r") as f:
            N = f["pixels"].shape[0]
            picks = np.sort(rng.choice(N, size=min(n, N), replace=False))
            px = f["pixels"][picks].astype(np.float32) / 255.0
            pos = np.asarray(f["pos_agent"][picks], dtype=np.float32)

        x = torch.from_numpy(px).permute(0, 3, 1, 2)      # (n,3,H,W)
        was_training = model.training
        model.eval()
        embs = []
        for s in range(0, len(picks), 64):
            out = model.encode({"pixels": x[s:s + 64].unsqueeze(1)})
            embs.append(out["emb"][:, 0].cpu().numpy())
        if was_training:
            model.train()
        emb = np.concatenate(embs)

        cut = int(len(picks) * 0.75)
        r = Ridge(alpha=1.0).fit(emb[:cut], pos[:cut])
        return round(float(r.score(emb[cut:], pos[cut:])), 6)
    except Exception as e:                     # noqa: BLE001 -- never kill a run
        print(f"  (probe_r2 skipped: {e})")
        return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--resume", default=None,
                   help="run dir to continue from (uses its last checkpoint)")
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop early (used by the resume-equivalence check)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = args.seed
    set_seed(cfg["seed"])        # also bans non-deterministic ops: loud > silent

    d, t, m = cfg["data"], cfg["training"], cfg["model"]

    # -- data ------------------------------------------------------------
    spec = ClipSpec(history=m["history_size"], num_preds=t["num_preds"],
                    frameskip=d["frameskip"])
    index = TwoRoomIndex(d["h5_path"], spec)
    ds = TwoRoomClips(d["h5_path"], index)
    train_idx, val_idx = split_indices(len(index.starts), d["train_split"],
                                       d["data_seed"])

    # -- model -----------------------------------------------------------
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
    sigreg = SIGReg(**cfg["loss"]["sigreg_kwargs"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"],
                            weight_decay=t["weight_decay"])

    # -- resume, or start fresh ------------------------------------------
    start_step, start_epoch, start_pos = 0, 0, 0

    if args.resume:
        run_dir = Path(args.resume)
        ck = load_ckpt(run_dir / "ckpt.pt", model, opt)
        start_step = ck["step"]
        start_epoch = ck["epoch"]
        start_pos = ck["pos_in_epoch"]           # where in the epoch we stopped
        # Restore the global RNG so dropout CONTINUES rather than restarting.
        # set_seed() above reset it to the beginning; without this, resumed
        # steps replay the dropout masks the first steps already saw.
        torch.set_rng_state(ck["torch_rng_state"])
        log_f = open(run_dir / "metrics.jsonl", "a")

        def log(rec):
            import time
            rec["t"] = time.time()
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

        def close():
            log_f.close()
        print(f"resumed: step {start_step}, epoch {start_epoch}, "
              f"position {start_pos} within the epoch")
    else:
        run_dir, log, close = lablog.start_run(
            cfg, tag=m["name"],
            extra={"data_sha256": data_fingerprint(index, d["h5_path"]),
                   "n_train_clips": int(len(train_idx)),
                   "n_val_clips": int(len(val_idx)),
                   "n_params": sum(p.numel() for p in model.parameters()),
                   "requirements_file": cfg.get("requirements_file", "requirements.txt")})
        log({"kind": "start", "n_params": sum(p.numel() for p in model.parameters())})

    # -- train -----------------------------------------------------------
    bs = t["batch_size"]
    step = start_step
    for epoch in range(start_epoch, t["epochs"]):
        # Order derived from (seed, epoch) alone -- reproducible from scratch,
        # so resuming mid-epoch lands on the identical order. See epoch_order().
        order = epoch_order(train_idx, epoch, cfg["seed"])
        begin = start_pos if epoch == start_epoch else 0
        start_pos = 0                             # only applies to the first epoch

        # Helpers change how fast batches arrive, never which clips are in
        # them: the order above is fixed before any helper starts, and the
        # loader returns batches in exactly that order. num_workers: 0 keeps
        # single-process reading (fine for the toy); set it to ~cores-4 on a
        # rented many-core box, where it is the difference between hours and
        # days of Blosc decompression.
        loader = make_loader(d["h5_path"], index, order, bs,
                             num_workers=t.get("num_workers", 0), begin=begin)
        for i, batch in enumerate(loader):
            s = begin + i * bs
            loss, mets = lewm_step(model, sigreg, batch,
                                   m["history_size"], t["num_preds"],
                                   cfg["loss"]["sigreg_weight"])
            opt.zero_grad()
            loss.backward()
            if t.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            step += 1

            if step % t["log_every"] == 0:
                log({"kind": "train", "epoch": epoch, "step": step, **mets})

            if args.max_steps and step >= args.max_steps:
                save_ckpt(run_dir / "ckpt.pt", model, opt, step, epoch, s + bs)
                log({"kind": "stopped_early", "step": step})
                close(); ds.close()
                print(f"stopped at step {step} (max-steps); checkpoint written")
                return

        ev = evaluate(model, sigreg, ds, val_idx, cfg)
        ev["probe_r2"] = probe_r2(model, d["h5_path"])
        ev.update({"kind": "eval", "epoch": epoch, "step": step})
        log(ev)
        save_ckpt(run_dir / "ckpt.pt", model, opt, step, epoch + 1, 0)
        r2s = f"{ev['probe_r2']:.4f}" if ev["probe_r2"] is not None else "n/a"
        print(f"epoch {epoch}: pred {ev['pred_loss']:.5f}  "
              f"bell {ev['sigreg_loss']:.3f}  spread {ev['spread']:.5f}  "
              f"R2 {r2s}")

    log({"kind": "done", "total_steps": step})
    close(); ds.close()
    print("wrote", run_dir)


if __name__ == "__main__":
    main()
