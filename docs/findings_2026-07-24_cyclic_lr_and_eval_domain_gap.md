# Findings, July 24 2026 — the cyclic-LR incident and the eval-world domain gap

Two root causes found by code reading, then confirmed by measurement. Both
change how earlier numbers must be read. Everything below ran free on the Mac;
every number is from a script in this repo.

## Summary

1. The Run-2 patch was applied twice. The duplicate `sched.step()` turned the
   one-way cosine decay into a full down-and-up cycle: the learning rate hit
   its floor at epoch 5 and climbed back to ~9e-6 by epoch 9. The post-epoch-5
   destabilization is fully explained; the intended schedule has never
   actually been run.
2. Every planner evaluation (toy_plan.py, dump_latents.py, hence Views 1-3 and
   all success rates, in all three runs) ran in the synthetic ToyTwoRoom
   debugging fixture, not on the real data domain the models were trained on.
   The fixture's frames land far outside the latent region the encoder
   learned, and that offset alone reproduces the planner-side step-0 error.
   The planner harness measured the domain gap, not the model.

## Incident 1 — cyclic learning rate (double-applied patch)

Two wordings of `patch_run2_schedule_and_ckpt.py` both ran; the second's
uniqueness asserts passed because its insertions differ from the first's only
in comment line-wrapping. Result: duplicate scheduler construction (harmless),
duplicate best-checkpoint block (harmless), and `sched.step()` twice per epoch
(not harmless). Effective LR by epoch, against the per-epoch step-0 error:

    epoch      0       1       2       3       4       5       6       7       8       9
    LR         1.0e-5  9.0e-6  6.6e-6  3.5e-6  1.0e-6  1.0e-7  1.0e-6  3.5e-6  6.6e-6  9.0e-6
    step0_err  91.3    73.0    73.4    46.6    8.79    5.67    10.29   31.2    81.6    31.1

Near dose-response: error is small only while LR <= ~1e-6. This strengthens
the optimization hypothesis (the model holds the minimum at small step sizes
and loses it at large ones) and removes the "cosine didn't hold" mystery --
the schedule the config asked for was never what ran.

Fix: `patch_fix_double_run2_blocks.py` (applied; anchored, idempotent).
Consequence: the intended one-way schedule is untested. See "Open decisions".

## Incident 2 — the eval world is the debugging fixture

`toy_plan.evaluate()` does `env = ToyTwoRoom()`; `dump_latents.py` mirrors it.
`make_toy_tworoom.py`'s own docstring: "a debugging fixture, not data ...
nothing measured on it is a result." Run 2 trained on the real tworoom.h5
(ViT at 224px, per manifest). Measurement chain (`bridge_step0_check.py` v2,
run on the Mac against the training data file, fingerprint-matched):

- CHECK A (instrument + checkpoint validation): the training script's own
  `step0_latent_error()` on the same validation clips the train loop used.
  ckpt_best: 5.668 (log: 5.671). ckpt_final: 31.207 (log: 31.1). Both
  reproduce to float noise. The 5.67-vs-56.55 dissociation is therefore real
  and about WHAT was measured.
- CHECK M (motion): real data moves a median 6.17 file units/step across a
  ~192-unit arena = 3.2% of the arena per step. The toy world: 15.6% per
  step. The fixture runs ~4.5-4.9x faster relative to its arena. (The noise
  structure was mirrored correctly: per-5-frame/per-step ratio 2.18 real vs
  2.21 toy. Only the arena scale is off.)
- CHECK B (static rendering gap, convention-free): toy renders land a median
  61.03 from the NEAREST real-data latent, while real latents sit 2.43 from
  each other -- 25x the cloud's own spacing off-manifold. 61 ~ the mystery
  56.55.
- CHECK C / C2 (attribution): the verbatim training metric on toy-rendered
  dataset-style clips gives 57.54 at toy speed and 57.41 with the dot slowed
  to real relative speed. Slowing the dot moved the error by 0.13: rendering
  style ALONE reproduces the planner-side error; the speed mismatch is real
  but secondary.
- EYEBALL (`save_eyeball_frames.py`, `eyeball_real_vs_toy.png`): the two
  worlds are nearly the same art style. Visible systematic differences: the
  real frames' border is inset with corner-overshoot ticks (toy border is
  flush with the edge); the real door sits in the upper quarter of the wall
  (toy door is centered); wall proportions differ accordingly. A stand-in a
  human would call "basically identical" is 25x off-manifold for the ViT --
  and the position probe (R^2 0.9916) was blind to it.

## Corrections to previously recorded claims

- "Cosine didn't fully hold" -> the schedule was accidentally cyclic.
- "One-step dynamics globally wrong (14.6x / 9.8x a real step)" -> toy-domain
  measurements; in-domain, the Run-2 best checkpoint predicts one step at
  0.83x a real step.
- "Planner performs at/below random" -> a harness artifact for real-trained
  models; the planner steered by imagination computed on off-manifold inputs.
- toy_plan's DIAGNOSTIC else-branch ("probe high -> the problem is the
  SCORING") -> demonstrably invalid under domain shift: the probe stayed at
  0.9916 across a 25x off-manifold gap. (Banner fixed by
  `patch_toy_plan_banner.py`; the DIAGNOSTIC text still needs the same
  treatment.)
- The pre-registered primary, AS INSTRUMENTED (toy-world), remains NULL. The
  validated in-domain 5.668 is a separate measurement that reframes the
  interpretation; it is not a re-read of the registered bands.

## Artifacts

- Fixed: `train_toy_lewm.py` (deduped), `toy_plan.py` (truthful banner).
- New diagnostics: `bridge_step0_check.py`, `save_eyeball_frames.py`,
  `patch_fix_double_run2_blocks.py`, `patch_toy_plan_banner.py`.
- Figure: `eyeball_real_vs_toy.png`.
- Checkpoint identities (unchanged): ckpt.pt == ckpt_best.pt
  md5 e90b5c0496073ad692357ac23d1e5b91 (epoch-5 best);
  ckpt_final.pt md5 783c17bef057603e71783c6a7590251e (epoch-9).

## Open decisions

1. Free in-domain evals (next): multi-step rollout error on real clips
   (`eval_rollout_horizon.py`) and the wall-geometry scoring test on real
   data (`eval_wall_scoring.py`).
2. Scope: a legitimate planner eval needs the real environment; the reference
   stack ships it (stable-worldmodel). Decide whether porting it is in scope.
3. The last rented run: seed-1 with the FIXED one-way cosine. Rationale: the
   paper's positive claim (the recipe converges to a good in-domain
   predictor) currently rests on one seed whose good checkpoint was produced
   by an accidental schedule. A clean-schedule run tests the intended recipe
   for the first time and replicates convergence across seeds -- and is
   decision-relevant in both directions (if it fails to converge, that must
   be known before publishing the claim).
