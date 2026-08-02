# Drafts — §4.5 and §5.3

Written as paper prose. Every number traces to a committed report. `[REF:n]`
markers need filling from the original paper.

---

## 4.5 Planning at the reference goal offset

*(~470 words. Home of the planning reproduction claim.)*

The original reports approximately 87% of goals reached on TwoRoom under
cross-entropy-method planning over the learned model [REF:C2]. Under our
protocol at the repository's evaluation goal offset of 25 steps, our corrected
reproduction reaches **47 of 50 = 94.0%** (non-trivial 45 of 48 = 93.8%). The
95% Wilson interval is [83.8%, 97.9%] and contains the reported figure; a
one-sample test against 0.87 gives p = 0.203.

Two comparisons place that number, and both use the identical fifty episodes.

The authors' own released checkpoint, driven through our harness with only the
weights changed, reaches **42 of 50 = 84.0%**. Our checkpoint is higher, but the
matched-pair test gives 5 improvements against 0 reversals, **p = 0.0625** — a
difference not established at this sample size. What the comparison does
establish is that our evaluation protocol is faithful: it recovers the reported
result from the reported weights (§4.2).

Our own earlier checkpoint, trained before the pipeline corrections of §3.2,
reaches **39 of 50 = 78.0%**. Against it, the corrected checkpoint improves 11
episodes and loses 3, **p = 0.0574** — again higher but not established at
n = 50. We note that both checkpoints were normalisation-recalibrated before
this comparison (§4.4); against the same checkpoint *without* recalibration the
figure is 72.0% and the test reaches p = 0.0074, but that comparison confounds
the pipeline correction with the recalibration and we do not rely on it.

One structural detail supports the reading that these are genuine model
differences rather than measurement noise. Our corrected checkpoint fails on
three episodes, and **all three are among the eight on which the authors'
checkpoint also fails**. The failures nest rather than scatter, and all three
are among the longest goals in the set (56.9, 72.1 and 96.8 units against a
median of roughly 48).

Two qualifications belong with the headline figure. First, the 87% is measured
under the authors' episode selection, which is not published; ours is a fixed
random draw at a stated seed, and remains a listed deviation (Table 1). The
like-for-like comparison is therefore 94.0% against 84.0% on identical
episodes, not against 87%. Second, all three of our figures come from a single
seed, and we make no claim about seed variance.

Finally, the number is specific to the goal offset. The repository's evaluation
configuration uses 25 steps while the paper's description implies 100 (Table 1),
and the choice is consequential: at offset 100 the same checkpoint reaches
20.0%. Section 5.3 takes that up, because the effect is not a simple
degradation with distance.

---

## 5.3 One-step accuracy does not predict long-horizon planning

*(~620 words. The paper's principal finding beyond the original.)*

A world model is trained to predict, and used to plan. It is natural to treat
prediction error as a proxy for planning competence. Across three checkpoints
spanning a sevenfold range in one-step prediction error, we find that the proxy
holds at short horizons and fails at long ones — and that the two most accurate
models plan **worse than a random-action control** at the longer horizon.

Table 3 gives the comparison. All three checkpoints are evaluated under one
protocol on identical episodes; one-step error is reported relative to a
frozen-world baseline, so a value below 1 means the model predicts better than
assuming nothing moves.

At goal offset 25 the ordering is monotone: as one-step error falls from 0.830
to 0.410 to 0.116, success rises from 78.0% to 84.0% to 94.0%. Prediction
accuracy behaves exactly as the proxy assumption expects.

At goal offset 100 the ordering does not hold at all. Success runs 54.0%, 12.0%
and 20.0% over the same three checkpoints. The **least** accurate model is by a
wide margin the best long-horizon planner: against it, the authors' checkpoint
loses 23 episodes and gains 2 (p = 1.9×10⁻⁵), and our corrected checkpoint
loses 18 and gains 1 (p = 7.6×10⁻⁵). The two more accurate checkpoints are not
distinguishable from each other (p = 0.29).

The failure mode is overshoot rather than stalling, and it is visible without
any modelling. The random-action control finishes a mean of 111.1 units from the
goal. The two accurate checkpoints finish at 116.6 and 122.5 units — **farther
away than random**, with individual final distances of 140 to 193 units in an
arena roughly 192 units across. The least accurate checkpoint finishes at 40.5
units. Neither accurate model is incapable of long goals: our corrected
checkpoint reached a 173-unit goal in 45 of its 50 allotted steps. They
systematically travel too far.

That the pattern holds for the **authors' own released weights**, and most
strongly there, matters for how it should be read. It is not an artifact of our
reimplementation, our pipeline corrections, or our recalibration procedure. It
is a property of this task, this planner and this class of model.

We are careful about mechanism. A plausible account is that a more accurate
model produces a sharper cost landscape, so the optimiser commits to
near-maximal actions, which a terminal-cost objective does not penalise until
the horizon ends; a weaker model yields a flatter landscape and more moderate
actions. The mean-final-distance column is consistent with this, but we have not
tested it, and we do not claim it.

A confound must also be stated plainly. The two overshooting checkpoints both
use a three-frame context; the cautious one uses a single frame. Context length
is therefore an alternative explanation to prediction accuracy, and three
checkpoints cannot separate the two. Distinguishing them would require training
matched checkpoints that vary one factor at a time, which our compute budget did
not allow.

The practical implication stands regardless of mechanism. **Selecting a world
model by held-out one-step prediction error is not a reliable way to select a
world model for long-horizon planning**, and on this task at this horizon it
would have selected the worst of three available options. Reporting a single
planning number without its horizon is correspondingly misleading: across these
three checkpoints, the goal offset alone moves success between 12% and 54%.

---

## Table 3 — caption and content

> **Table 3: One-step prediction error against planning success at two goal
> horizons.** All figures from 50 episodes per cell, identical across
> checkpoints, under one protocol. One-step error is relative to a frozen-world
> baseline (below 1 = better than assuming no motion). Mean final distance is
> at offset 100; the random-action control finishes at 111.1 units.

| checkpoint | one-step error | offset 25 | offset 100 | mean final dist. @100 | context frames |
|---|---|---|---|---|---|
| our pre-correction checkpoint | 0.830 | 78.0% | **54.0%** | **40.5** | 1 |
| authors' released | 0.410 | 84.0% | **12.0%** | 122.5 | 3 |
| our corrected checkpoint | 0.116 | **94.0%** | 20.0% | 116.6 | 3 |
| random-action control | — | 18.0% | 0.0% | 111.1 | — |

## Figure 3 — caption

> **Figure 3: Prediction accuracy orders short-horizon planning success and
> fails to order long-horizon planning success.** **(a)** Goals reached against
> one-step prediction error, with the axis running from worse to better
> prediction. At goal offset 25 the relationship is monotone; at offset 100 it
> is not, and the least accurate checkpoint is the strongest planner. **(b)**
> Mean final distance at offset 100. Bars above the dashed line finish farther
> from the goal than a random-action policy does; the two most accurate
> checkpoints both do.

Generate with:

```
python3 figure_horizon_dissociation.py \
  --entry "our pre-correction"=0.83=<r2 offset25 report>=<r2 offset100 report> \
  --entry "authors' released"=0.410=<authors offset25 report>=<authors offset100 report> \
  --entry "our corrected"=0.116=<phase2 offset25 report>=<phase2 offset100 report> \
  --random-final 111.07
```

---

## Drafting notes

- §4.5 must not restate the harness validation (§4.2) or the deviations (§3.2);
  reference them.
- **Do not quote 94% without the goal offset anywhere in the paper.** The last
  paragraph of §4.5 exists to make that impossible to do accidentally.
- §5.3 should not be softened. The p-values on the long-horizon reversals are
  10⁻⁵; the claim that is *not* established is our own 94-versus-84 and
  94-versus-78, and §4.5 says so.
- The mechanism paragraph in §5.3 is explicitly marked as untested. Keep it that
  way; a reviewer will otherwise read it as a claim.
