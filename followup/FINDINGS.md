# Follow-up experiments — CPU only, no GPU training

Everything here post-dates arXiv:2608.10145 and is **separate from the paper's
evidence base**. Outputs are named `followup_plan_*` rather than
`realenv_plan_*` so that `extract_all_results.py` — which globs
`**/realenv_plan_*report*.txt` from the repository root — cannot see them and
`docs/paper/results_from_disk.csv` cannot silently change. The published
paper's numbers are untouched.

Goal: find capability improvements reachable without retraining.

---

## 1. Imagination is not the long-horizon bottleneck

`followup/rollout_h15_phase2_recal.txt` — the released flagship
(`ckpt_best_recal.pt`, phase2) rolled out autoregressively on real validation
clips under the recorded actions:

| horizon (planner steps) | env steps | err/static |
|---|---|---|
| 1 | 5 | 0.066 |
| 5 | 25 | 0.090 |
| 10 | 50 | 0.152 |
| 15 | 75 | 0.189 |

At 15 steps out the imagined state is still only 19% as wrong as assuming the
world froze. Horizon 20 could not be measured at all: the clip would need 102
raw frames and episodes cap at 101 — the same ceiling that degenerates the
offset-100 protocol in §4.2 of the paper.

**The predictor can see 75 environment steps ahead. CEM only ever asks it for
25** (`horizon=5` planner actions x `frameskip=5`), toward a goal 100 steps
away. That is a planner limitation, not a model limitation.

## 2. The planning objective saturates, then inverts

CEM minimises squared L2 between the imagined final embedding and the goal
embedding. `followup/latent_metric_phase2_recal.txt` measures whether that
quantity actually tracks distance, with no planning involved — render real
recorded positions, encode, compare pairwise latent distance to pairwise true
distance:

| true distance | mean latent dist | r within band |
|---|---|---|
| 0–20 | 15.92 | 0.668 |
| 20–40 | 18.52 | 0.182 |
| 40–60 | 19.68 | 0.115 |
| 60–80 | 20.04 | 0.104 |
| 80–100 | 20.61 | 0.122 |
| 100–120 | 20.68 | −0.029 |
| 120–150 | 20.60 | falling |
| 150–300 | 19.93 | falling |

Overall Pearson r = 0.426. Across 80–300 units the objective varies by −3.3%,
and past ~120 units it **decreases** with true distance: moving farther from
the goal can lower the planner's cost.

This accounts for what the paper reports without explaining: 94.0% at offset
25 (short range, where r = 0.668), collapse to 26.0% at offset 100, and the
overshoot — 26 of 37 offset-100 failures end farther from the goal than they
began, at a median 1.41x the original separation. Binned by difficulty,
baseline success is 100% under 60 units and 6% in the 100–140 band.

It is also, measured on our own checkpoint, the failure mode of the Li et al.
(2026) critique the paper cites.

## 3. A mechanism for the accuracy/planning dissociation

§5.3 reports that one-step accuracy does not order long-horizon planning, and
§1 lists "any mechanism for the overshoot" as **not claimed**. Measuring the
same metric geometry on all three released checkpoints supplies one:

| checkpoint | one-step err | offset 100, budget 150 | Spearman | monotone | 80–300 spread |
|---|---|---|---|---|---|
| Run 2 recal | 0.829 | **40/50 = 80.0%** | 0.872 | yes | +38.6% |
| Run 0 recal | — | **33/50 = 66.0%** | 0.846 | yes | +34.9% |
| phase2 recal | 0.116 | **13/50 = 26.0%** | 0.445 | **no** | −3.3% |

Long-horizon planning success rank-orders **exactly** with latent-metric
quality, and inversely with prediction accuracy. The most accurate predictor
has the least usable planning geometry. That is a candidate mechanism for the
paper's central dissociation, and it is testable rather than speculative.

Caveat: three checkpoints, one seed each. This is a correspondence across the
three models we have, not an established law.

## 3b. The pathology is the method's, not the reproduction's

The authors' own released checkpoint, measured the same way
(`followup/latent_metric_authors.txt`):

| | Pearson | Spearman | monotone | 80-300 spread |
|---|---|---|---|---|
| authors' released | 0.388 | 0.423 | **no** | **-2.2%** |

That is phase2's pathology almost exactly, on weights we did not train. Adding
it to the table gives four checkpoints and a perfect rank correspondence
between metric quality and long-horizon planning:

| checkpoint | offset 100 / budget 150 | Spearman | monotone | 80-300 spread |
|---|---|---|---|---|
| Run 2 recal | 80.0% | 0.872 | yes | +38.6% |
| Run 0 recal | 66.0% | 0.846 | yes | +34.9% |
| phase2 recal | 26.0% | 0.445 | no | -3.3% |
| authors' released | 14.0% | 0.423 | no | -2.2% |

The split is not arbitrary. The two checkpoints with usable geometry (Run 0,
Run 2) were trained with `history_size` 1 and `action_dim` 2. Both pathological
ones -- phase2 and the authors' own -- use the reference pipeline, with
`history_size` 3 and dense 10-wide actions. The conventions that fix one-step
prediction appear to cost the metric its long-range ordering. That is a
hypothesis from four checkpoints, not a demonstrated cause.

## 3c. A partial account of the extra rank dimensions

§1 also lists "any account of what the extra rank dimensions encode" as **not
claimed**. Two already-committed numbers now line up with the metric quality
measured here:

| checkpoint | effective rank (of 192) | Spearman | monotone |
|---|---|---|---|
| Run 0 recal | 18.6 | 0.846 | yes |
| phase2 recal | 67.8 | 0.445 | no |

`runs_archive/verified/encoder_probe_both_recal.txt` records the ranks; the
Spearman values are from this work. The corrected pipeline spreads the
representation over roughly 3.6x as many effective dimensions, and the
distance ordering degrades over the same interval.

This does not say what those dimensions encode. It says that whatever they
encode, it is not distance-relevant, and it dilutes an L2 metric computed over
all 192 of them — which is consistent with the probe result, since a two-
dimensional linear read-out recovers position at R² 0.99 from the same vector
that L2 cannot order. Two checkpoints; suggestive, not established.

## 4. The information is present; only the metric fails

`followup/probe_metric_phase2_recal.txt` — a ridge probe fit on frozen
embeddings of 350 rendered positions, evaluated on 150 held out:

- probe position R² **0.9922** held out, mean absolute error **1.72** arena units
- Pearson r vs true distance — latent L2 **0.437**, decoded position **0.9897**
- monotone in true distance — latent L2 **no**, decoded position **yes**
- spread across 80–300 units — latent L2 **+1.2%**, decoded position **+73.4%**

The encoder already represents position almost perfectly. Nothing about the
world model needs to change; the objective does.

The same holds on the authors' checkpoint
(`followup/probe_metric_authors.txt`): probe R^2 **0.9627** held out, 4.13
units mean error, decoded-position distance monotone at r = **0.9573**
(+57.5% across the far bands) where their latent L2 is neither
(r = 0.4236, +1.2%). Whatever this fix is worth, it is available to the
original work too.

## 5. Intervention under test

`--cost probe` re-points CEM at the distance between **decoded positions**
rather than embeddings. The probe is fit at startup on rendered positions
(R² 0.9993 in-sample, 0.72 units mean error); the encoder and predictor are
frozen and unmodified.

Running at `--num-eval 50 --seed 42 --budget 150 --goal-offset 100`, which
reproduces `exp_ref_p2`'s exact 50-episode draw (verified), so the result is
directly paired against the published **13/50 = 26.0%**.

Early two-episode signal at a 60-step budget — a third of the baseline's —
reached a 150.7-unit goal in 25 steps.

## 6. Result: the fix works, and costs nothing at the short horizon

Every episode is matched -- same checkpoint, same protocol, same episode draw
-- so McNemar applies to the discordant pairs.

### Goal offset 25, budget 50 -- COMPLETE, n = 50

Paired against `exp_phase2_recal_25`. The harness validates itself here: the
baseline column reproduces **47/50 = 94.0%**, the published figure exactly.

| | success |
|---|---|
| baseline, latent L2 cost | 47/50 = 94.0% |
| follow-up, decoded-position cost | 46/50 = 92.0% |

discordant 3-4, exact McNemar **p = 1**. No detectable cost where the latent
metric already works. Episode 1650, which the baseline missed 79.2 units out,
the fix reached in 15 steps.

### Goal offset 100, budget 150 -- COMPLETE, n = 50

Paired against the published `exp_ref_p2`. The baseline column reproduces
**13/50 = 26.0%**, the published figure exactly, as it did at offset 25.

| | success |
|---|---|
| baseline, latent L2 cost | 13/50 = 26.0% |
| follow-up, decoded-position cost | **44/50 = 88.0%** |

discordant **31-0** in favour of the fix, exact McNemar **p = 9.3e-10**. It
wins 31 episodes the baseline lost and loses none that the baseline won.

The shape of the change matters as much as the rate. Baseline failures
exhaust the 150-step budget and finish 115-193 units out having started
71-170 out -- the overshoot. The fix typically arrives in 24-49 steps, a
third of the budget.

Two comparisons worth stating plainly:

- 88.0% beats **Run 2's 80.0%**, the best long-horizon planner in the paper.
  Under a sound objective the most accurate predictor becomes the best
  long-horizon planner as well, so §5.3's dissociation does not merely get
  explained -- it dissolves.
- 88.0% at offset 100 approaches the paper's **94.0% at offset 25**. The
  long-horizon deficit was mostly the objective, not the horizon.

(An earlier window of 12 episodes read 10/12 against 0/12. That window was
exactly the baseline's failures -- its 13 successes all fall at draw
positions >= 12 -- so the percentages there flattered the fix. Superseded by
the complete run above.)

### The authors' own released weights, goal offset 100 -- COMPLETE, n = 50

Paired against the published `exp_ref_protocol`, whose baseline column
reproduces **7/50 = 14.0%** exactly. These are the original authors' weights,
not ours; only the planner's objective differs.

| | success |
|---|---|
| baseline, latent L2 cost | 7/50 = 14.0% |
| follow-up, decoded-position cost | **35/50 = 70.0%** |

discordant **28-0**, exact McNemar **p = 7.5e-09**.

So this is not a repair for a defect the reproduction introduced. The metric
pathology is in the released method (§3b), and re-pointing the objective
repairs planning on the authors' own checkpoint.

### Summary

Three paired comparisons, 50 episodes each, every baseline column reproducing
its published figure exactly -- which is what makes the harness trustworthy
rather than merely self-consistent.

| checkpoint | protocol | baseline | decoded-position cost | McNemar |
|---|---|---|---|---|
| ours, phase2 recal | offset 25, budget 50 | 47/50 = 94.0% | 46/50 = 92.0% | p = 1 |
| ours, phase2 recal | offset 100, budget 150 | 13/50 = 26.0% | **44/50 = 88.0%** | p = 9.3e-10 |
| **authors' released** | offset 100, budget 150 | 7/50 = 14.0% | **35/50 = 70.0%** | p = 7.5e-09 |

In neither long-horizon comparison does the fix lose a single episode the
baseline won: 31-0 and 28-0 on the discordant pairs.

No retraining, no GPU, one flag. The encoder and predictor are the released
weights, unmodified; only the planner's objective changed.

## 7. Reachability, not proximity, is what the objective must measure

The learned temporal head (§5) predicts spatial distance WORSE than the
position probe -- r = 0.819 against 0.9897 -- and plans BETTER:

| objective | r vs spatial distance | cross/same wall ratio | offset 100 |
|---|---|---|---|
| squared latent distance | 0.437 | **0.96** | 13/50 = 26.0% |
| decoded position | 0.9897 | 1.00 by construction | 44/50 = 88.0% |
| learned temporal head | 0.819 | **1.24** | **49/50 = 98.0%** |

`followup/wall_reachability.txt`, 460,320 pairs matched on true spatial
distance and split by whether they cross TwoRoom's dividing wall: at the same
spatial separation the temporal head charges 24% more to cross the wall, while
squared latent distance charges 4% LESS -- not merely blind to the wall but
biased the wrong way.

Planning success orders by the wall ratio, not by accuracy against spatial
distance. The decoded-position cost cannot represent the wall at all --
Euclidean distance between positions is identical whether or not a barrier
lies between them -- and it recovers most of the gap because proximity
approximates reachability in open space. The temporal head recovers the rest
because it is not an approximation.

98.0% at offset 100 also exceeds the 94.0% the same checkpoint reaches at
offset 25 under the published objective. The long-horizon deficit was not a
horizon problem.

## 8. The learned cost must be trained on what the planner scores

The v1 head is trained on pairs of ENCODED REAL frames. At planning time CEM
scores (IMAGINED embedding, encoded goal). Those coincide only to the extent
the predictor is accurate, and that is checkable.

Evaluated on identical held-out pairs, split by which kind:

| checkpoint | one-step error | v1 head, real x real | v1 head, imagined x real | degradation |
|---|---|---|---|---|
| ours, phase2 recal | 0.116 | 12.47 | 12.86 | **+3%** |
| authors' released | 0.410 | 10.38 | **17.55** | **+69%** |

Our predictor is accurate enough that imagined embeddings stay near the
manifold the head was fit on, so a head trained on real pairs transfers.
The authors' predictor drifts further, the head extrapolates, and the cost
becomes unreliable exactly where planning needs it. That is why the same
recipe gave 98.0% on our weights and underperformed a linear position probe
on theirs -- a linear map degrades off-manifold far more gracefully than an
MLP.

`followup_temporal_head_v2.py` trains on both kinds of pair, rolling the
predictor forward under the recorded block-mean actions exactly as the planner
drives it. On the authors' checkpoint that removes the gap: 10.85 real against
11.17 imagined, a +3% degradation instead of +69%.

**This is a real condition on the method, not a detail.** A learned planning
cost has to be trained on the distribution the planner will evaluate it on. We
found it only because the repair failed on a second checkpoint; on ours the
two distributions happened to coincide and the requirement was invisible.

## 9. Efficiency: the same gain under a third of the budget

Goal offset 100 under the repository's own 50-step budget, paired against the
published `exp_phase2_recal`, whose baseline column reproduces 10/50 = 20.0%
exactly:

| | success |
|---|---|
| baseline, latent L2 cost | 10/50 = 20.0% |
| temporal cost | **46/50 = 92.0%** |

discordant 37-1, exact McNemar **p = 2.8e-10**.

92% of hundred-step goals are reached inside fifty environment steps. The
repair is not buying success with time.

## 10. Where the learned cost does NOT win

On the authors' released checkpoint the learned cost underperforms the linear
position probe, and the v2 fix of §8 does not close it:

| objective on the authors' weights | offset 100, budget 150 |
|---|---|
| published latent L2 | 7/50 = 14.0% |
| decoded position (linear probe) | **35/50 = 70.0%** |
| learned temporal head v1 | far worse; stopped early |
| learned temporal head v2 | 6/20 = 30.0% (partial) |

v2 removed the distribution mismatch -- +11% degradation on imagined pairs
instead of +74% -- and planning improved over v1, but not to the probe's
level.

The reading we take from this: **a learned cost is the better objective only
where the predictor is good enough to support it.** Our checkpoint's one-step
error is 0.116 and the learned cost wins decisively. The authors' is 0.410;
imagination drifts, and a two-parameter-per-output linear map is more robust
to that drift than an MLP, however well the MLP is trained. Fixing the
training distribution helps and does not rescue it.

This is a limitation of the repair, not of the diagnosis. The diagnosis --
that the published objective saturates and inverts, and is the binding
constraint -- holds on both checkpoints. What differs is which replacement
objective is safe to use.

## What was tried and set aside

- **Planning horizon 15** (`--plan-horizon 15`): motivated by §1, stopped
  after ~35 minutes without completing an episode. A longer horizon lets the
  planner reach farther under an objective that is blind past 80 units, so §2
  predicts it cannot help much on its own. Worth rerunning against the probe
  cost, where the objective is sound.
- **Oracle subgoal decomposition** (`--subgoals 3`): 2/3 in a smoke test at
  offset 100. Paused at 3/50 to give the probe-cost run the machine. It routes
  around the broken metric by keeping every hop inside the short range where
  it works — but subgoals come from the recorded trajectory, which is
  privileged information a deployed planner would not have. The probe cost
  needs no such information, which is why it took priority.
