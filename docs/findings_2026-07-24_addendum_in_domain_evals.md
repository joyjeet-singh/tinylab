# Addendum, July 24 2026 (late) — in-domain evaluations: both planning prerequisites pass

Follow-up to `findings_2026-07-24_cyclic_lr_and_eval_domain_gap.md`. Two free
in-domain evaluations of Run 2's ckpt_best, both on real validation data.
Scripts: `eval_rollout_horizon.py`, `eval_wall_scoring.py` (harness validated
against known toy ground truth before use: the geometry detector recovered the
toy's wall x=20.0 and door 20.0±5 exactly).

## 1. Multi-step rollout (does imagination survive the planning horizon?)

Real validation clips, the recorded actions, autoregressive rollout mirroring
the planner's imagination mechanics:

    horizon   imagined err   real step   err/step   static err   err/static
       1         5.222         6.647       0.786       6.647       0.786
       2         6.899         6.721       1.027       9.746       0.708
       3         8.112         6.907       1.175      11.947       0.679
       4         9.108         6.714       1.357      13.788       0.661
       5         9.887         6.948       1.423      15.201       0.650

Reading: error growth DECELERATES (increments 1.68, 1.21, 1.00, 0.78). At the
CEM horizon (5), the imagined state is off by 1.42x a single real step while
the true state has moved 15.2 latent units from the start -- imagination beats
the "world froze" baseline at every horizon, and by more (relatively) the
further out it goes (err/static falls 0.79 -> 0.65). The predictor's rollouts
carry usable signal across the full planning horizon, in-domain.

Note on the horizon-1 control: 5.222 here vs 5.668 in Check A. Not a
discrepancy -- the horizon-5 clip set spans 26 raw frames (vs training's 6),
so the clip index has fewer legal starts and the same split seed selects a
different first-512 sample. Same measurement, different clips; the err/step
ratios agree (0.786 vs 0.829).

## 2. Wall-geometry scoring test (does latent distance see the wall?)

Geometry detected from the data itself (639 side-change crossings):
wall x ~ 111.2 (= the arena midpoint, 111.1 -- symmetric, as expected);
door y in [38.4, 61.9], median 53.9. First localization of the real door:
~19% from the low end of the y range, consistent with the off-center door
visible in `eyeball_real_vs_toy.png`. The real door is also proportionally
NARROWER than the toy's (~12% of span vs the toy's 31%) -- one more geometry
difference for the fixture record.

Position-matched pairs (distance 9.7-29.1 file units, 5-15% of span):

    same-room pairs            : 20,966   latent distance median 10.26
    cross-wall, far-from-door  :    210   latent distance median 18.33
    cross/same ratio           : 1.79
    overall latent-vs-position correlation: 0.832

Reading: the encoder SEPARATES the rooms. A state just across the wall scores
~1.8x farther in latent space than an equally-distant state in the same room.
The strong form of the published failure mechanism ("Beyond Euclidean
Proximity": the wall is invisible to straight-line latent distance) is NOT
supported in-domain for our encoder. Whether a 1.79x penalty is
quantitatively sufficient to steer CEM correctly is a planning question this
test alone cannot settle; what it rules out is the metric being blind to the
wall.

## What this changes

Yesterday the model looked unplannable. Today, both measurable prerequisites
for planning pass in the training domain: rollouts stay informative to
horizon 5, and the scoring metric sees the wall. Caveats stated plainly:
rollouts used the recorded (in-distribution) action sequences, not CEM's
explored ones; the wall test used our reproduction's encoder, not the
authors' released weights; and neither result is a planning success rate.
But the value of a legitimate real-environment planner evaluation (the
reference stack's stable-worldmodel layer) just went up substantially: it is
now the single missing experiment between this checkpoint and a statement
about whether the reproduction actually plans.
