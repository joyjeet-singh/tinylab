"""
patch_run2_schedule_and_ckpt.py -- three anchored edits to train_toy_lewm.py for
Phase 1 Run 2 (the decider). Each edit refuses loudly if its anchor is not unique,
so nothing is ever half-applied.

  1. COSINE LR SCHEDULE (opt-in via training.lr_schedule: cosine).
     The script has NO scheduler today. Run 1 (constant LR 1e-5) descended
     43.6 -> 2.59 by epoch 7 then DESTABILIZED to 13.26 by epoch 9: an LR that can
     REACH the minimum but not HOLD it. Cosine keeps the early step size that
     produced the descent and shrinks it late, exactly where Run 1 fell apart.

  2. BEST-LOSS CHECKPOINT (ckpt_best.pt).
     Today only ckpt.pt exists and is overwritten every epoch, so Run 1's saved
     model was its DESTABILIZED epoch-9 state (pred 13.26) and its best epoch-7
     model (2.59) was lost -- we evaluated the wrong frame. Run 2 is scored on
     ckpt_best.pt (ckpt.pt reported alongside). Resume path unchanged.

  3. PER-EPOCH STEP-0 LATENT ERROR (the metric that actually matters).
     Runs 0 and 1 showed training loss and one-step latent prediction quality can
     DIVERGE (Run 1's loss reached 2.59 while its step-0 error got WORSE: 73.9 ->
     88.3). So training loss is an unreliable proxy. This logs the real primary
     metric every epoch -- mean ||predicted next latent - true next latent|| and the
     mean real per-step latent movement (the scale reference), computed on held-out
     validation clips. It answers directly, per epoch: did the predictor EVER get
     good, and when? No more inferring from a proxy.

Run from the tinylab folder:  python3 patch_run2_schedule_and_ckpt.py
Verify:  grep -n "lr_schedule\|ckpt_best\|step0_err\|sched.step" train_toy_lewm.py
"""
from pathlib import Path

p = Path("train_toy_lewm.py")
src = p.read_text()

# ---- edit 1: cosine scheduler, right after the optimizer --------------------
old1 = """    opt = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"],
                            weight_decay=t["weight_decay"])
"""
new1 = """    opt = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"],
                            weight_decay=t["weight_decay"])
    # optional LR schedule (Run 2): cosine decay over the planned epochs.
    # Run 1 reached its minimum then destabilized under a constant LR; decaying the
    # step size late holds the minimum without slowing the early descent.
    sched = None
    if t.get("lr_schedule") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=t["epochs"], eta_min=t.get("lr_min", 0.0))
        print(f"lr_schedule: cosine  base={t['learning_rate']} "
              f"eta_min={t.get('lr_min', 0.0)} T_max={t['epochs']}", flush=True)
"""
assert src.count(old1) == 1, "STOP: optimizer anchor not unique; file unchanged"
src = src.replace(old1, new1)

# ---- edit 2: the per-epoch step-0 latent error probe ------------------------
# inserted just before probe_r2's decorator so it sits with the other diagnostics
old2 = """@torch.no_grad()
def probe_r2(model, h5_path: str, n: int = 300, seed: int = 0):
"""
new2 = '''@torch.no_grad()
def step0_latent_error(model, ds, val_idx, cfg, max_batches: int = 4):
    """
    THE primary metric, measured every epoch instead of inferred from the loss.

    Runs 0 and 1 proved training loss is an unreliable proxy: Run 1's pred_loss fell
    to 2.59 while its one-step latent error got WORSE (73.9 -> 88.3). So we measure
    the real thing here.

    For held-out validation clips: encode the context, predict ONE step forward, and
    compare that predicted latent to the TRUE next latent from the encoder.
    Also returns the mean real per-step latent movement, so the error can be read as
    a multiple of how far the world actually moves in one step (scale-free WITHIN a
    run; note the latent magnitude changes BETWEEN runs, so never compare raw ratios
    across runs).

    Returns {"step0_err": float, "real_step": float, "err_over_step": float}.
    """
    model.eval()
    bs = cfg["training"]["batch_size"]
    HS = cfg["model"]["history_size"]
    errs, moves = [], []
    for s in range(0, min(len(val_idx), max_batches * bs), bs):
        picks = val_idx[s:s + bs]
        if len(picks) < 2:
            break
        batch = make_batch(ds, picks)
        dev = next(model.parameters()).device
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        out = model.encode(batch)
        emb = out["emb"]                                  # (B, T, D) true latents
        if emb.size(1) < HS + 1:
            continue
        act_emb = model.action_encoder(batch["action"])
        pred = model.predict(emb[:, :HS], act_emb[:, :HS])[:, -1]   # predicted next
        true_next = emb[:, HS]                                     # actual next
        errs.append((pred - true_next).norm(dim=-1).mean().item())
        moves.append((emb[:, HS] - emb[:, HS - 1]).norm(dim=-1).mean().item())
    model.train()
    if not errs:
        return {"step0_err": None, "real_step": None, "err_over_step": None}
    e = sum(errs) / len(errs)
    m = sum(moves) / len(moves)
    return {"step0_err": round(e, 4), "real_step": round(m, 4),
            "err_over_step": round(e / m, 3) if m > 1e-9 else None}


@torch.no_grad()
def probe_r2(model, h5_path: str, n: int = 300, seed: int = 0):
'''
assert src.count(old2) == 1, "STOP: probe_r2 anchor not unique; file unchanged"
src = src.replace(old2, new2)

# ---- edit 3: log step-0 error each epoch + best ckpt + scheduler step -------
old3 = """        save_ckpt(run_dir / "ckpt.pt", model, opt, step, epoch + 1, 0)
"""
new3 = """        # the primary metric, measured (not inferred) every epoch
        s0 = step0_latent_error(model, ds, val_idx, cfg)
        ev.update(s0)
        log({"kind": "step0", "epoch": epoch, **s0})
        if s0["step0_err"] is not None:
            print(f"  step0_err {s0['step0_err']:.3f}  real_step {s0['real_step']:.3f}"
                  f"  ratio {s0['err_over_step']}", flush=True)

        save_ckpt(run_dir / "ckpt.pt", model, opt, step, epoch + 1, 0)
        # keep the BEST model too: Run 1's final checkpoint was its worst
        # (pred 13.26) while its best (2.59) was overwritten and lost.
        _pl = ev.get("pred_loss")
        if _pl is not None and _pl < globals().get("_best_pred", float("inf")):
            globals()["_best_pred"] = _pl
            save_ckpt(run_dir / "ckpt_best.pt", model, opt, step, epoch + 1, 0)
            log({"kind": "best_ckpt", "epoch": epoch, "pred_loss": _pl,
                 "step0_err": s0["step0_err"]})
            print(f"  (new best pred {_pl:.5f} -> ckpt_best.pt)", flush=True)
        if sched is not None:
            sched.step()
"""
assert src.count(old3) == 1, "STOP: end-of-epoch save anchor not unique; file unchanged"
src = src.replace(old3, new3)

p.write_text(src)
print("patched train_toy_lewm.py:")
print("  1. cosine LR schedule (opt-in: training.lr_schedule: cosine)")
print("  2. ckpt_best.pt on best epoch pred_loss")
print("  3. per-epoch step0_latent_error logged to metrics.jsonl + stdout")
print()
print("verify with:")
print("  grep -n 'lr_schedule\\|ckpt_best\\|step0_err\\|sched.step' train_toy_lewm.py")
