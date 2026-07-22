# Pre-registration amendment — Phase 1, Run 2 (the decider)

Amends `prereg_phase1_convergence.md`. Committed BEFORE Run 2. Records a deviation
and a metric-handling correction, both driven by Run 1's outcome.

## What Run 1 showed (the justification for amending)

Run 1 (bundle: LR 1e-5, SIGReg 0.045, history_size 1, epochs 10) produced:
- pred trajectory 43.6, 28.8, 20.7, 20.7, 33.2, 23.2, **3.97, 2.59**, 6.14, **13.26**
  — a clean descent to a minimum at epoch 7, then DESTABILIZATION back up.
- Therefore **NOT SCOREABLE** under the convergence gate (loss non-stationary, and
  rising, at the final epoch). Neither STRONG nor NULL was recorded.
- Evaluated anyway (free, on Mac): step-0 latent error **88.30**, planning 3/50 (6%,
  matching the random baseline; all three "reached" episodes were trivially-close
  flukes at 1-2 states).

Critically, the evaluated checkpoint was the **destabilized epoch-9 state (pred
13.26)**, not the epoch-7 minimum (pred 2.59) — `ckpt.pt` is overwritten each epoch,
so the best model was lost. The 88.30 therefore measures the run's worst late state,
not what this configuration is capable of.

## Amendment 1 — cosine LR schedule (ONE change)

Run 2 = Run 1 config **plus a cosine LR decay**, base LR unchanged at 1e-5,
`lr_min` 1e-7. Justification: a constant LR that REACHES a minimum but cannot HOLD it
is the standard case for decay — preserve the early step size that produced
43.6 -> 2.59, shrink it late where Run 1 fell apart.

Deliberately NOT also lowering the base LR to 5e-6: decaying from an already-halved
base risks not reaching the minimum inside the 10-epoch budget (Run 1 needed until
epoch 7 to get there). One variable. If Run 2 still destabilizes, base 5e-6 + cosine
is the pre-registered Run 3, not a bundled change now.

Requires a code patch (the script had NO scheduler): `patch_run2_schedule_and_ckpt.py`,
adding an opt-in `training.lr_schedule: cosine`. Default behaviour unchanged when the
key is absent.

## Amendment 2 — best-loss checkpointing, and WHICH checkpoint is scored

The same patch writes `ckpt_best.pt` whenever an epoch's `pred_loss` is the lowest so
far. **Run 2's primary metric is computed on `ckpt_best.pt`.** `ckpt.pt` (final epoch)
is evaluated and reported alongside, so the gap between best and final is visible
rather than hidden.

This is a metric-handling correction, not a goalpost move: the pre-registered
threshold values are UNCHANGED (STRONG <15, PARTIAL 15-40, NULL >55). What changes is
that we score the model the configuration actually produced at its best, instead of
whatever state it happened to end in. Recorded here, before the run, precisely so it
cannot be mistaken for post-hoc selection.

## Amendment 3 — cross-run ratio comparisons are INVALID (correction)

The latent space rescaled between runs: mean real per-step latent movement was
**5.07** in the original reproduction and **17.25** in Run 1 (history_size 1 changes
the latent magnitude). Consequently the printed "prediction error / real step size"
ratio (14.6x original, 5.1x Run 1) is **not comparable across runs** — both numerator
and denominator moved. Run 1's apparently-better 5.1x is a scale artifact, not an
improvement.

Committed handling: report step-0 error against the SAME run's real-step scale, and
compare runs on (a) the qualitative shape of the divergence curve (flat-high vs
descending) and (b) planning-vs-random on identical episodes. Do not compare ratios
across runs anywhere in the paper.

## Amendment 4 — Views 1 and 2 run on every checkpoint

Run 1 was evaluated with View 3 only. Amended: Views 1 (latent geometry, shared
basis RULE = per-model probe-coefficient axes) and 2 (prediction-error map,
basis-independent, doorway/interior ratio vs the 0.93 baseline) are computed for
every scored checkpoint, including retroactively for Run 1. This gives the paper the
three-way before/after panel (original -> bundle -> stabilized) the prereg promised,
and lets us see whether the failure MODE changed even when the headline metric did not.

## Unchanged

Thresholds (STRONG <15 / PARTIAL 15-40 / NULL >55), the convergence gate (no verdict
on a non-stationary run), the secondary metric (planner beats random on the same 50
episodes/seed), the outcome table, the "bundled success is not a reproduction claim
until ablated" rule, and the <=4-run budget cap (Run 2 is run 2 of 4).

## Standing interpretation, stated in advance

If Run 2 converges (loss settles) and step-0 error on `ckpt_best.pt` still does not
fall below 55, that is a clean, defensible NULL: *even with a stabilized predictor
reaching substantially lower training loss, one-step latent prediction and planning
did not recover.* If step-0 error falls under 15 and planning beats random, H1 is
supported and the ablations follow. Either result is reportable; that is the point.
