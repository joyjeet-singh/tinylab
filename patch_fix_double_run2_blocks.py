"""
patch_fix_double_run2_blocks.py -- remove the DOUBLE-APPLIED Run-2 blocks from
train_toy_lewm.py. Two anchored deletions; each refuses loudly if its anchor is
not found exactly once, so nothing is ever half-applied. Idempotent: run twice
and the second run just says "already applied".

WHAT HAPPENED
-------------
patch_run2_schedule_and_ckpt.py was applied twice (an earlier wording of it and
the current one -- the two insertions differ only in comment line-wrapping, which
is why the second application's uniqueness asserts still passed). Result:

  1. The cosine scheduler is CONSTRUCTED twice.  Harmless -- the second
     construction just replaces the first with an identical object.

  2. The best-checkpoint block appears twice.  Harmless -- the second copy's
     "is this a new best?" test can never pass, because the first copy already
     recorded the value.

  3. sched.step() is called TWICE PER EPOCH.  NOT harmless. CosineAnnealingLR
     follows a cosine curve of period 2*T_max in scheduler steps. Stepping twice
     per epoch with T_max=10 walks the FULL cycle in 10 epochs: the learning rate
     fell to the floor (1e-7) by epoch 5 and then CLIMBED BACK toward the base
     (9.05e-6 by epoch 9). Run 2's LR per epoch was actually:

        epoch:  0        1        2        3        4        5
        LR   :  1.0e-5   9.0e-6   6.6e-6   3.5e-6   1.0e-6   1.0e-7
        epoch:  6        7        8        9
        LR   :  1.0e-6   3.5e-6   6.6e-6   9.0e-6      <- climbing back up

     which is exactly when the per-epoch step0 error came apart again
     (5.67 at epoch 5 -> 10.3, 31.2, 81.6, 31.1 as the LR rose). The
     "cosine didn't hold" mystery is this bug, not the schedule.

This patch deletes the SECOND copy of each duplicated region (keeping the copy
whose best-ckpt log line also records step0_err). After it, cosine is a true
one-way decay: reach the floor at the last epoch and stay there.

Run from the tinylab folder:  python3 patch_fix_double_run2_blocks.py
Verify:  grep -c "sched.step()" train_toy_lewm.py     (must print 1)
"""
from pathlib import Path

p = Path("train_toy_lewm.py")
src = p.read_text()

# ---- deletion 1: the duplicate cosine-scheduler block -----------------------
# (distinguishable from the kept copy by its comment line-wrap: "decaying\n# the")
dup_cosine = '''    # optional LR schedule (Run 2): cosine decay over the planned epochs.
    # Run 1 reached its minimum then destabilized under a constant LR; decaying
    # the step size late holds the minimum without slowing the early descent.
    sched = None
    if t.get("lr_schedule") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=t["epochs"], eta_min=t.get("lr_min", 0.0))
        print(f"lr_schedule: cosine  base={t['learning_rate']} "
              f"eta_min={t.get('lr_min', 0.0)} T_max={t['epochs']}", flush=True)
'''

# ---- deletion 2: the duplicate best-ckpt block + its EXTRA sched.step() -----
# (distinguishable from the kept copy by "BEST-loss" in the comment and by its
#  log line NOT recording step0_err)
dup_best = '''        # keep the BEST-loss model too: Run 1's final checkpoint was its worst
        # (pred 13.26) while its best (2.59) was overwritten and lost.
        _pl = ev.get("pred_loss")
        if _pl is not None and _pl < globals().get("_best_pred", float("inf")):
            globals()["_best_pred"] = _pl
            save_ckpt(run_dir / "ckpt_best.pt", model, opt, step, epoch + 1, 0)
            log({"kind": "best_ckpt", "epoch": epoch, "pred_loss": _pl})
            print(f"  (new best pred {_pl:.5f} -> ckpt_best.pt)", flush=True)
        if sched is not None:
            sched.step()
'''

already = (src.count(dup_cosine) == 0 and src.count(dup_best) == 0
           and src.count("sched.step()") == 1)
if already:
    print("already applied: one cosine block, one best-ckpt block, one sched.step()")
    raise SystemExit(0)

assert src.count(dup_cosine) == 1, (
    "STOP: duplicate cosine block not found exactly once; file unchanged")
assert src.count(dup_best) == 1, (
    "STOP: duplicate best-ckpt block not found exactly once; file unchanged")

src = src.replace(dup_cosine, "")
src = src.replace(dup_best, "")

assert src.count("sched.step()") == 1, (
    "STOP: expected exactly one sched.step() after dedup; file unchanged")
assert src.count("sched = None") == 1, (
    "STOP: expected exactly one scheduler construction after dedup; file unchanged")

p.write_text(src)
print("applied: removed duplicate cosine block and duplicate best-ckpt block")
print("now: one scheduler construction, one best-ckpt save, ONE sched.step() per epoch")
print("verify: grep -c \"sched.step()\" train_toy_lewm.py   (must print 1)")
