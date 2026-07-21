# Pre-registration — Phase 1: predictor-convergence experiment (bundle-first)

Committed BEFORE running anything. Purpose: settle the confound the mechanism
section names — is the below-random planning result caused by predictor
UNDER-TRAINING (our side, fixable) or a FUNDAMENTAL limitation of the method on
this data (their side)? Design principle: **fail cheaply.** Run the strongest
plausible configuration FIRST as a screening run; only spend further runs if it
succeeds. This maximizes the evidential value of a failure and defers attribution
cost to the case where attribution is actually worth paying for.

Baseline being tested against (current reproduction, seed 0, 10 epochs):
- position probe R^2 = 0.99 (encoder fine)
- planner success 1/50 (2.0%); random control 3/50 (6.0%) — planner BELOW random
- one-step latent prediction error at rollout step 0 = **73.9** (std 3.65)
- mean real per-step latent movement = **5.07**  ->  error is **14.6x a real step**
- predictor loss non-stationary at final epoch (oscillated 0.49-2.07, never settled)

The below-random result has been localized (Views 1-3) to the predictor's one-step
dynamics: encoder good (R^2 0.99), error uniform not localized (doorway/interior
ratio 0.93), error immediate not accumulating (flat ~74->82 across the rollout).
So the lever is predictor CONVERGENCE, not geometry/horizon/scoring.

---

## Design: bundle-first, then ablate only on success

The asymmetry that drives this design:
- A **bundled failure** is a STRONGER negative result than a single-change failure
  ("even our best-possible config could not converge the predictor" beats "lowering
  the LR alone didn't help"). Bundling maximizes the value of a NULL.
- A **bundled success** is NOT attributable — four simultaneous changes cannot tell
  us WHICH one mattered, and a reproduction paper's contribution IS the attribution.
  So a bundled success must be followed by ablations before any mechanism claim.

Therefore:

### RUN 1 — the screening (reconnaissance) run: change everything at once
Apply, together, every plausible convergence lever, from rental_tworoom_v2.yaml:
- **learning_rate:** 5e-5 -> **[LOWER — set exact value in the run card before launch; candidate 1e-5]**
- **SIGReg weight:** **[set — candidate: reduce so the predictor gets stronger accuracy gradient; exact value fixed before launch]**
- **history_size:** 3 -> **1** (the paper's TwoRoom value; also an easier prediction problem)
- **epochs:** 10 -> **[EXTEND — set exact value before launch; candidate 20-30, long enough to converge]**
- everything else IDENTICAL (img 224, patch 14, batch 128, seed 0). Same data, same
  pipeline, same gates, same verified apparatus (GPU device patch, bit-identical
  resume gate, HF pull, retrieval-first).

This run's role is pre-registered as SCREENING: it answers "is convergence
achievable at all with a favorable config?" It does NOT, by itself, constitute a
reproduction finding. That is stated here so a bundled success cannot later be
reported as "we reproduced LeWM" without the ablations below.

### RUNS 2-3 — targeted ablations: ONLY IF Run 1 succeeds
If Run 1 converges the predictor and planning recovers, spend the remaining budget
turning ONE lever back to its baseline value at a time, to find the culprit:
- Run 2: Run-1 config but **history_size back to 3** (isolates the history change).
- Run 3: Run-1 config but **learning_rate back to 5e-5** (isolates the LR change).
- (SIGReg/epochs ablations as budget allows, logged as amendments.)
Each ablation that RE-BREAKS convergence identifies a necessary lever; each that
doesn't rules one out. This is what lets the paper say WHICH of the paper's settings
was the problem — the actual reproduction contribution.

If Run 1 FAILS (NULL) -> STOP. No ablations needed (nothing to attribute). The
strong fundamental-limitation result is already in hand at one run's cost.

---

## Primary metric (committed): step-0 one-step latent prediction error

WHY not planning success: success is a noisy binary at tiny counts (1/50 vs 3/50 —
a 1-2 episode swing is within noise). Step-0 latent error is continuous, low-variance
(std ~3.7 on n=50), and is the quantity the intervention directly targets. Measured
identically to the current run: dump_latents.py on the new checkpoint -> View-3
latent-divergence value at step 0.

**Committed thresholds (write the call, THEN run) — applied to RUN 1:**
- **STRONG support (H1: under-training):** step-0 error < **15** (< ~3x a real step;
  a >4x reduction from 73.9). => convergence is achievable; proceed to ablations.
- **PARTIAL:** step-0 error **15-40**. Real improvement, not converged. Treat as
  inconclusive — extend training within the same run's budget or as one amendment
  run; do NOT proceed to ablations on a partial.
- **NULL (H1 falsified):** step-0 error stays **> 55** (< 25% reduction) despite a
  SETTLED training loss. => the method cannot converge the predictor on this data
  under a favorable config. STOP; report the fundamental-limitation result.

**Convergence gate (committed):** do not score H1 on a run whose predictor loss is
still non-stationary at the final epoch (variance over last 3 epochs not << the
0.49-2.07 baseline range). Such a run is under-trained by definition — extend, don't
verdict. A NULL is only valid if the loss actually SETTLED and error still > 55.

## Secondary metric (committed): planning success vs random
Same eval: toy_plan --num-eval 50 --seed 0, planner AND --random, success_radius=3.0,
goal_offset_steps=25. **Recovery** = planner beats random on the same 50 eps/seed
(the below-vs-above-random line is the bar, not an absolute number).

## Outcome table (every cell pre-assigned)
| Run-1 step-0 error | Run-1 planning | reading & next action |
|---|---|---|
| STRONG (<15) | beats random | convergence achievable -> RUN ABLATIONS to attribute -> headline reproduction-with-diagnosis |
| STRONG (<15) | still <= random | dynamics fixed but planning still fails -> the SCORING is the wall (Euclidean latent distance, paper's App. F.2). A different clean finding; ablate to confirm the dynamics fix, then pivot the paper to the scoring result. |
| NULL (>55), loss settled | still <= random | fundamental limitation on this data under a favorable config. STOP. Strong negative result. |
| PARTIAL (15-40) | either | inconclusive; extend training (one amendment run) before any verdict. |

---

## Decision rules & budget (committed)
1. **Bundle first, ablate only on success.** Run 1 = all levers together. Ablations
   (Runs 2-3) happen IFF Run 1 hits STRONG.
2. **Budget cap: <= 4 rented runs total (~$24)** — 1 screening + up to 3 ablations.
   If Run 1 is NULL, actual spend is 1 run (~$6). A bundled failure ends the
   experiment cheaply; that is the point of this design.
3. **No metric swapping.** step-0 latent error is primary; planning success is
   confirmation. Report the table cell we land in; do not promote whichever metric
   looks better after the fact.
4. **Before/after figure** reuses the SAME probe-axis basis rule as the current views
   (share the rule, not the matrix). Views 1/2/3 rerun on the converged checkpoint.
5. **A bundled success is not a reproduction claim until ablated.** Written here so it
   cannot be walked back.

## What each outcome means for the paper (none is a "failure")
- **STRONG + beats random + ablated:** strongest result — reproduction recovered by
  correcting an optimization issue, WITH attribution of which setting caused the
  original discrepancy. The full reproduction-with-diagnosis paper.
- **STRONG + still <= random:** reproduces LeWM's own Euclidean-scoring caveat with
  direct evidence. Clean contribution.
- **NULL:** honest negative — "the method as specified does not converge the
  predictor on this data within a reasonable budget, even under a favorable config."
  Publishable at MLRC, which explicitly welcomes failures to reproduce.

The point of this prereg: commit to reporting whichever cell the data gives, at
thresholds fixed before we look, and spend budget only where attribution is worth it.
