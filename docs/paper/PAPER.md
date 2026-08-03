<!-- ASSEMBLED by assemble_paper.py. Do not edit the drafts after this point; edit this file. -->

# Abstract

*(~250 words)*

LeWorldModel trains a latent world model with a prediction loss and a single
anti-collapse regulariser, and reports approximately 87% of goals reached on
TwoRoom — its simplest diagnostic environment, where comparable methods report
97–100%. We reproduce that result by independent reimplementation, on four
rented GPU-runs costing about six dollars each, with all evaluation on one
laptop CPU.

We reach **94.0%** at the repository's evaluation goal offset, against **84.0%**
for the authors' own released checkpoint measured under our protocol on
identical episodes, and we reproduce the reported representation result directly
(position probe R² 0.9977). Reaching that point required correcting **four
conventions that determine the outcome and appear in no released configuration
file**: dense action gathering across a frameskip block, a programmatically-set
action-encoder width, ImageNet pixel normalisation, and action z-scoring. A
reproducer following the released configurations alone obtains a model whose
predictor cannot converge.

Two findings generalise beyond this reproduction. First, **one-step prediction
accuracy does not predict long-horizon planning success**: across three
checkpoints spanning a sevenfold range in prediction error — including the
authors' own — accuracy orders short-horizon success monotonically and fails to
order long-horizon success at all, where the two most accurate checkpoints
finish farther from the goal than a random-action policy. Second, **a batch
normalisation layer inflated our reported validation loss by up to three orders
of magnitude**, concealing for three training runs a training loss that was
descending monotonically; we give the two conditions under which this occurs and
a cheap check for it.

We also report a pre-registered mechanism-level result that did not survive a
change of checkpoint, and what we take from that.

---

# 1 Introduction

LeWorldModel [REF:paper] proposes a latent world model trained with a
prediction loss and a single anti-collapse regulariser, deliberately without the
exponential moving averages, frozen encoders and auxiliary objectives that
comparable methods require. Among the environments it evaluates, TwoRoom is the
simplest: a point agent in a two-room arena joined by a single door, with an
action space of two dimensions and deterministic dynamics. It is also where the
method looks weakest. The paper reports approximately 87% of goals reached
there, against 97–100% for the baselines it compares against on the same task
[REF:baselines]. A method's behaviour on its easiest diagnostic is informative,
and an anomaly on that diagnostic is worth resolving before drawing conclusions
from harder ones.

We set out to reproduce three claims on TwoRoom: that the learned encoder
recovers agent position under a linear probe at approximately R² 0.996, that
planning over the learned model reaches approximately 87% of goals, and that the
released configuration produces such a model within the stated training budget.
We reimplemented the method from the released code and paper rather than
rerunning the released training script, trained four models on rented GPUs at
roughly six dollars each, and ran every evaluation on a single laptop CPU.

The reproduction succeeded, but not on the first attempt, and the failures along
the way turned out to be more useful than the success. Three of our four
training runs appeared not to converge. Every planning number we measured for
three of those runs was worthless. A carefully pre-registered mechanism-level
result did not survive a change of checkpoint. In each case the cause lay in our
instrumentation or in an undocumented convention rather than in the method under
study, and in each case we reported a confident and wrong conclusion before
finding it. This paper reports the reproduction and the failures together,
because the failures are what a reproducer attempting the same work would most
benefit from knowing.

### Contributions

**We reproduce the reported result and exceed it.** With four undocumented
pipeline conventions corrected and a normalisation artifact repaired, our
checkpoint reaches **94.0%** of goals at the repository's evaluation goal
offset (§4.5), against a reported ~87% and against **84.0%** for the authors'
own released checkpoint measured under our protocol on identical episodes. The
representation result reproduces directly: R² 0.9977 under a linear probe
(§4.1).

**We identify four conventions that determine the outcome and appear in no
configuration file** (§3.2): actions must be gathered densely across a frameskip
block rather than sub-sampled, the action-encoder width is set programmatically
from that block, pixels are ImageNet-normalised, and actions are z-scored by
dataset statistics with NaN rows removed. A reproducer following the released
configurations alone obtains a model whose predictor cannot converge, and we
quantify why: the sub-sampled action implies a displacement wrong by roughly
twice the movement it is required to explain.

**We show that one-step prediction accuracy does not predict long-horizon
planning success** (§5.3). Across three checkpoints spanning a sevenfold range
in one-step prediction error — including the authors' own — accuracy orders
short-horizon planning success monotonically and fails to order long-horizon
success at all. At the longer horizon the two most accurate checkpoints finish
*farther* from the goal than a random-action policy does, and the least accurate
is the strongest planner by a wide margin.

**We document two measurement failures that a probe would not catch** (§4.3,
§5.1). A 32-pixel debugging fixture, visually near-indistinguishable from the
real environment, sat twenty-five times farther from the training distribution
than training frames sit from each other, and produced below-random planning
results for three paid runs while a position probe read R² 0.99 throughout. And
a batch-normalisation layer specified by the released configuration inflated our
reported validation loss by up to three orders of magnitude, concealing a
training loss that was descending monotonically the whole time. We give the
conditions under which the second occurs — a normalisation layer whose running
variance is far below its activation scale, combined with weights still moving
— and show that the authors' released checkpoint satisfies only the second and
is therefore unaffected.

**We report a pre-registered result that did not survive** (§5.2). A same-room
planning advantage of +39.1 points at p = 3.4 × 10⁻⁸, distance-matched
pair-by-pair with reachability verified at 100% by an oracle and the episode
list committed before evaluation, falls to +12.7 points under a change of action
scaling and to −6.4 points on a different checkpoint. We report all three arms,
and take from it that effect sizes measured on a single reproduction checkpoint
should not be read as properties of the method.

### Scope

All results concern the TwoRoom diagnostic. We make no claim about the original
paper's embodied or zero-shot results, about its other environments, or about
the method at scales other than the 18.03M-parameter configuration studied here.
Every figure in this paper comes from a single seed. Our full deviation set is
given in Table 1, our limitations in §7, and all code, configurations,
evaluation reports and gate outputs are available at [REF:repo].

---

# 2 Scope of reproducibility

We test three claims the original makes about TwoRoom. Each is stated below in
the form the original makes it, followed by our verdict and a pointer to the
evidence. We also state two claims we deliberately do not test.

### Claim 1 — the encoder recovers agent position

> The original reports that a linear probe recovers the agent's position from
> the learned embedding at approximately R² 0.996 [REF:§x].

**Reproduced.** On 4,000 held-out frames, a ridge probe fitted on 80% recovers
position at **R² 0.9977**, and a two-layer network on the same split reaches
0.9995 (§4.1). The result appears within a single training epoch and is robust
across every pipeline configuration we trained, including those whose predictor
does not converge.

We report one methodological caveat that bears on any comparison of probe
values. Our own per-epoch training logs report probe scores between 0.9305 and
0.9974 for the same encoders, a spread that vanishes to 0.003 under a single
protocol on identical frames. Probe values are protocol-dependent; we state ours
(4,000 held-out frames, ridge, 80/20 split) wherever we report one.

### Claim 2 — planning over the learned model reaches approximately 87%

> The original reports approximately 87% of goals reached under cross-entropy
> method planning [REF:§y].

**The protocol reproduces; our checkpoint exceeds the figure; the comparison is
qualified.** Two measurements bear on this.

First, the authors' own released checkpoint, driven through our evaluation
harness with only the weights changed, reaches **42/50 = 84.0%** (§4.2). A
one-sample test against 0.87 gives p = 0.53. Our harness therefore recovers the
reported result from the reported weights, which validates the protocol
independently of anything we trained.

Second, our own corrected checkpoint reaches **47/50 = 94.0%** at the
repository's evaluation goal offset of 25 steps, with a 95% interval of
[83.8%, 97.9%] that contains the reported figure (§4.5). Against the authors'
checkpoint on identical episodes the difference is not established at our sample
size (p = 0.0625).

This claim carries a conflict we cannot resolve from the released material. The
repository's evaluation configuration uses a goal offset of 25 steps while the
paper's description implies 100 (Table 1), and the choice is consequential: at
offset 100 the same checkpoint reaches 20.0%, and across the three checkpoints
we evaluate the figure ranges from 12.0% to 54.0% (§5.3). We report both
offsets throughout and quote no planning number without one.

### Claim 3 — the released configuration produces such a model in the stated budget

> The paper's appendix states ten training epochs [REF:appx]; the released
> repository configuration specifies one hundred [REF:cfg].

**Not reproduced as released; reproduced once four undocumented conventions are
corrected.** Our reimplementation of the released configuration plateaus: its
training loss reaches ~0.30 within the first epoch and moves less than 4% over
the following nine (§4.4). With dense action gathering, the programmatic action
encoder width, ImageNet pixel normalisation and action z-scoring applied — none
of which appears in any released configuration file — the training loss descends
monotonically to a held-out value 36 times lower, within the same ten epochs
(§3.2, §4.4).

We are explicit that this is a statement about our reimplementation. We did not
rerun the authors' training script, and we make no claim that their training
procedure fails. Our verdict is that the released *configuration*, as
implementable from the released *configuration files*, is insufficient to
specify a converging run.

### Not tested

**The original's other environments and its embodied and zero-shot results.**
Our budget covered four training runs on one task. We make no claim about any of
these.

**Seed variance.** Every figure in this paper comes from a single seed. Our
evaluations are deterministic — re-running one from the committed commit
reproduces its per-episode outcomes exactly — so the variance we have not
measured is between-seed, not within-run. Where we report a difference between
checkpoints, we report its test and, where it is not established at our sample
size, say so.

---

## 3.1 Model and objective

*(~280 words)*

We reimplement the architecture the released configuration specifies. The
encoder is a ViT-Tiny at 224 pixels with patch size 14, twelve layers and three
heads, producing a 192-dimensional embedding. A projector maps that embedding
through a two-layer network with a batch-normalisation layer; the predictor is a
six-layer transformer with sixteen heads, head dimension 64 and MLP width 2048,
consuming a context of three frames; a second projection of the same shape as
the first is applied to the predictor's output. An action embedder maps each
step's action into the predictor's dimension. Our implementation totals
**18,034,670 parameters** against **18,034,590** for the reference checkpoint
reconstructed from its released configuration; the difference of 80 is exactly
the width of the action encoder's first layer, which the reference sets
programmatically (§3.2).

The objective is the sum of a prediction term and a regularisation term. The
prediction term is the mean squared error between the predicted next embedding
and the encoded next frame, **with no stop-gradient on the target**: gradient
flows into both sides, which makes representational collapse an available
solution and is why the second term carries real weight. The regulariser is a
sketched isotropy test — the cloud of embeddings is projected onto many random
directions and each one-dimensional shadow is tested against a standard normal
using an Epps–Pulley statistic — computed per timestep across the batch and
averaged. We follow the released weighting of 0.09 with 17 quadrature knots and
1024 projections.

We reimplemented rather than reran. That choice is what surfaced the four
undocumented conventions of §3.2: a rerun would have inherited them silently,
and a reproduction that inherits an undocumented convention has not tested
whether the release specifies it.

---

## 3.2 Fidelity of the reimplementation

*(~380 words as drafted)*

A reproduction is only as trustworthy as its account of where it differs from
the original. We therefore audited our reimplementation element by element
against the reference **source**, not against its configuration files, and
recorded each element as matching, deviating, or unverified. The distinction
turned out to be decisive: of the four deviations that mattered most, none
appears in any configuration file, and all four sit in code that a reader
following the released configs would never open.

Twenty of twenty-five audited elements match exactly, including the encoder
architecture (ViT-Tiny/14 at 224 pixels), the predictor geometry, batch size,
weight decay, gradient clipping, and the SIGReg parameters. Our parameter count
of 18,034,670 differs from the reference checkpoint's only by the width of the
action encoder.

The four deviations were these. First, the reference gathers actions at full
rate and reshapes them to `(history_len, frameskip × action_dim)` [REF:1],
whereas we sub-sampled one action per clip step; the released `config.json`
records an action-encoder width of ten, which is exactly frameskip five times
action dimension two. Second, and consequently, the action-encoder width is set
programmatically at training time [REF:2] rather than in the config. Third,
pixels are ImageNet-normalised before resizing [REF:3]; we divided by 255 and
stopped. Fourth, non-pixel columns are z-scored using dataset statistics with
NaN rows dropped [REF:4]; we used raw actions, and the dataset contains exactly
one NaN action per episode, at the final step.

Two of these are independently corroborated by the released artifact rather than
only by the code: the checkpoint's action encoder has ten input channels, and
the same checkpoint attains a one-step prediction error of 0.410 relative to a
frozen-world baseline under ImageNet normalisation against 5.1 under raw
`[0,1]` inputs — a twelvefold difference that makes the convention effectively
mandatory.

The action deviation is the most consequential and the easiest to quantify.
Because the environment is deterministic, the displacement across a clip step
equals the speed times the summed actions of that block. Measured on the
released dataset, the sub-sampled convention leaves a median error of 25.59
units against a typical per-block displacement of 13.3 — the action supplied to
the predictor was wrong by roughly twice the movement it was required to
explain. The remaining deviations, all deliberate, are listed in Table 1.

> **Table 1 note.** Generate from `docs/fidelity_audit.md` plus
> `close_debts.py` section C. Include the environment row: Python 3.12.3 for
> phase2 against 3.11.15 for Runs 0–2, torch 2.2.2+cu121 throughout. Add a
> footnote that `data_sha256` in our manifests is a fingerprint over (clip
> index, file) and not a file hash, so runs differing in `history_size`
> legitimately differ in that field.

---

## 3.3 Environment verification

*(~230 words)*

Every planning number in this paper depends on the evaluation environment being
the same environment that generated the training data. We establish that
directly rather than assuming it.

Placing the agent at each recorded position and rendering gives a **pixel mean
absolute error of 0.00** against the corresponding recorded frame. Replaying the
recorded action sequence from a recorded state reproduces the recorded
trajectory with an error of **0.000** at one step and at forty. Encoding paired
real and re-rendered frames gives a median latent distance of **0.01**, against
a nearest-neighbour spacing within the real data of 2.43.

This last figure is the instrument we use as a precondition throughout. Every
evaluation in this paper computes it before planning and refuses to report a
success rate if it exceeds a threshold of 1.0. The measured values across all
runs reported here lie between 0.009 and 0.014.

The check earns its place. An earlier phase of this work evaluated in a
32-pixel fixture built for cheap iteration on a laptop, where the same
instrument reads **61.03** — twenty-five times the real data's own
nearest-neighbour spacing. Three training runs' worth of planning results were
produced there and are worthless. We describe that episode in §5.1; here we note
only that the precondition exists because it was needed.

---

## 3.4 Experimental protocol and gates

*(~380 words)*

Rented compute forces a discipline that is worth stating, because it shaped what
we were able to conclude. Our budget allowed four training runs. A run that
fails for a preventable reason is not recoverable, so we adopted a rule that a
run counts against the budget only after a set of executable checks passes.

Four gates ran before each launch. **G1** clones the repository at the committed
head into a temporary directory, resolves every local module the training entry
point imports transitively, and verifies that all of them, and the configuration,
are present in the clone and compile there. This catches the classic failure of
a run that works locally because it imports a file that was never committed.
**G2** compares every element of the training configuration against
source-derived reference values, asserts the loader's data contract — including
the physics identity of §3.2 — builds the model, and runs a short CPU training
loop checking for finite losses and for the regulariser being wired into the
objective. A deviation does not fail this gate, but an *unexplained* deviation
does: each must be listed with a reason. **G3** is the domain precondition of
§3.3, embedded in every evaluation rather than run separately. **G4** is a
written statement, committed before launch, of what each possible outcome will
and will not license.

Two properties of this arrangement mattered more than we expected. Gates that
fail loudly are worth more than gates that are correct: of the gate failures we
investigated, more were caused by defects in the gate than in the run (§6.2),
and each of those defects was itself a finding about what we had assumed. And
committing G4 before launch prevented at least one post-hoc reinterpretation: our
recorded prediction for the run of §4.4 was wrong, and having written it down
made that unambiguous.

We also pre-registered one experiment in full — the design, the decision rule for
every outcome, and the exact episode list — before evaluating any of it (§5.2).
Its outcome is reported in §5.2, including the fact that the registered effect
did not survive subsequent analysis on other checkpoints.

All gate outputs, the pre-registration, and the expected-outcome statements are
in the repository.

---

## 3.5 Computational requirements

*(~260 words)*

Four training runs on a single rented GPU, at approximately six US dollars each,
totalling about twenty-four dollars of compute. Each run is ten epochs over
roughly 780,000 clips at 224-pixel resolution and completes in a few hours.

Everything else ran on one laptop CPU with 8 GB of memory: all evaluation, all
planning, every probe, every gate, the environment verification, the fidelity
audit, and the normalisation recalibration of §4.3. A planning evaluation of
fifty episodes takes fifteen to sixty minutes depending on the goal offset and
the model; the 220-episode matched-pair experiment of §5.2 takes approximately
four hours. The recalibration procedure takes a few minutes.

Two consequences of this budget bear on our conclusions. First, we train for ten
epochs, following the paper's appendix, where the released repository
configuration specifies one hundred (Table 1); a hundred-epoch run was outside
our means, so our convergence result is a statement about the paper's stated
budget and not about the asymptote. Second, we have one seed per configuration,
and the differences we report between checkpoints are correspondingly qualified
(§7).

We note the ratio deliberately. Twenty-four dollars of GPU time produced four
checkpoints; several hundred hours of CPU time produced everything that made
those checkpoints interpretable, including all four of the findings we consider
most transferable. A reproduction of this kind is not principally a compute
problem.

---

## 4.1 The representation reproduces, and is not the bottleneck

The original reports that a linear probe recovers agent position from the
learned embedding at approximately R² 0.996 [REF:C1]. We reproduce this. On
4,000 held-out frames drawn uniformly from the released dataset, a ridge probe
fitted on 80% and evaluated on the remaining 20% recovers position at **R²
0.9977**; a two-layer MLP probe on the same split reaches 0.9995, confirming the
linear probe is not limited by its own capacity.

The protocol matters more here than the number. Our own training logs report
per-epoch probe values ranging from 0.9922 to 0.9974 for the reference-faithful
run and from 0.9305 to 0.9525 for the corrected-pipeline run — an apparent
five-point difference between the two pipelines. Measured under a single
protocol on identical frames, that difference is **0.003** (0.9977 against
0.9946) and vanishes entirely under the non-linear probe (0.9995 against
0.9996). The in-training probe fits far fewer samples, where ridge
regularisation dominates at 192 dimensions. We report the common-protocol
numbers throughout and recommend that probe protocols be stated wherever probe
values are compared, including within a single paper's own logs.

A second measurement matters more for what follows. From a pair of consecutive
embeddings (z_t, z_{t+k}), the summed action executed between them is linearly
decodable at **R² 0.9207**; from their difference alone, at 0.8925. Together
with the position result this characterises the latent space precisely: it is a
near-linear encoding of agent position, and transitions within it carry the
action that produced them in linearly accessible form.

This has a direct consequence for the training results in §4.3 and §4.4. The
environment's dynamics are deterministic and, in position space, affine in the
action: displacement equals a fixed speed times the summed actions of the block
(§3.3). Because the embedding is a near-linear encoding of position, the
forward map a predictor must learn in latent space is approximately as simple as
the true dynamics, and every quantity it requires is present and linearly
accessible in its inputs. **The failure to converge documented below is
therefore not an information-theoretic limitation of the representation. It is a
property of the predictor and its optimisation.**

Finally, the regulariser does its job. Mean embedding spread remained within
0.830–0.960 across the reference-faithful run and 0.797–1.039 across the
corrected-pipeline run, with no monotone decline in either (Figure 1c). We
observed no representation collapse under any configuration we trained,
including at the reference learning rate where the prediction loss does not
settle. This supports the original's central architectural claim — that the
two-term objective is sufficient to prevent collapse without an exponential
moving average, a frozen encoder, or auxiliary supervision — independently of
whether the predictor converges.

---

> **Table 2: Encoder comparison under a single probe protocol.** 4,000 held-out
> frames, identical for both models; each encoder receives the pixel convention
> it was trained with. Ridge probes, 80/20 split.

| measurement | reference-faithful (Run 0) | corrected pipeline (phase2) |
|---|---|---|
| position, linear probe | **0.9977** | 0.9946 |
| position, MLP probe | 0.9995 | 0.9996 |
| summed action from (z_t, z_{t+k}) | **0.9207** | 0.8733 |
| summed action from z_{t+k} − z_t | 0.8925 | 0.8502 |
| effective rank (of 192) | 11.9 | 16.5 |
| mean embedding spread | 1.000 | 0.930 |

*Effective rank is discussed in §5.4; it is listed here so the comparison is
presented once.*

---

## 4.2 The evaluation protocol reproduces

A reproduction that reports a planning number is reporting the product of two
things: a model, and a protocol for evaluating it. If the protocol is wrong,
every number it produces is wrong in a way no amount of internal consistency
checking will reveal. Before reporting any figure from our own checkpoints, we
therefore ran the **authors' released checkpoint** through our evaluation
harness, changing nothing but the weights.

It reaches **42 of 50 goals = 84.0%**, against the reported approximately 87%.
A one-sample test against 0.87 gives p = 0.53, and the 95% Wilson interval,
[71.5%, 91.7%], contains the reported figure. Our episode selection, goal
construction, success criterion, step budget, planner settings and action
convention therefore recover the reported result from the reported weights.

Reaching that point required settling two things the released material does not
state, and both were settled by measurement rather than assumption.

**The input convention.** The released configuration specifies the architecture
but not the preprocessing. Encoding frames under raw `[0,1]` pixel values, their
checkpoint predicts the next latent state with an error 5.1 times a
frozen-world baseline — five times *worse* than assuming nothing moves. Under
ImageNet normalisation the same weights score 0.410. A twelvefold difference
makes the convention effectively mandatory, and it is discoverable only by
reading the reference's preprocessing chain or by measuring, as we did.

**The action convention.** Their action encoder accepts ten inputs while the
environment's action space has two dimensions. Reading the reference's clip
loader resolves this — actions are gathered densely across a frameskip block and
concatenated, so ten is frameskip five times action dimension two (§3.2). A
planner emitting one action held across a block corresponds to a block whose
mean is that action, and we supply it accordingly.

We record one earlier failure here because it bears on how such numbers should
be read. Our first attempt at this calibration supplied the action at the wrong
scale, and produced **46.0%** — a number that arrived with every surrounding
check passing: the domain guard passed, the random-action control replicated
exactly, and the run completed without error. What identified it as an artifact
was not any check but the *shape* of the failures: successes came unusually fast
(median 5 steps against our checkpoint's 18) while misses were extreme
overshoots, with 22 of 27 finishing farther from the goal than they started.
That is the signature of a planner whose model understates how far an action
moves the agent. Measuring the action scale directly then confirmed it, and the
corrected figure is the 84.0% above. We report the episode as evidence that a
completed run with passing checks is not the same as a correct measurement, and
that a wrong answer of this kind is more readily caught by inspecting the
distribution of failures than by any single summary statistic.

Two properties of this validation are worth stating for what follows. First, it
is independent of everything we trained: no checkpoint of ours enters it, so
§4.5's and §5.3's protocol is validated regardless of how our own training
turned out. Second, it is not a validation of our *model* in any respect, and we
do not use it as one. Finally, we verified separately that the authors'
checkpoint requires no normalisation recalibration — its evaluation-to-training
gap is 1.09× (§4.3) — so the figures we report for it here and in §5.3 are not
affected by the artifact described in that section.

---

## 4.3 An evaluation-mode artifact concealed the training result

*(~700 words)*

For three of our four training runs we recorded a per-epoch validation
prediction loss that oscillated by more than 100% of its mean and showed no
trend across ten epochs. Read on its own, that series says the predictor does
not converge, and we reported it that way for a week. The per-step **training**
loss, written to the same log file, was flat or monotonically descending in
every one of those runs. We had not opened it.

The two series cannot both be describing the model's progress. To find out which
was misleading, we scored a single fixed checkpoint four ways: in evaluation
mode and in training mode, on held-out clips and on training clips. If the gap
were a generalisation gap, it would follow the data; if it were an artifact of
the evaluation procedure, it would follow the mode.

| checkpoint | eval mode, held-out | train mode, held-out | eval mode, train clips | train mode, train clips |
|---|---|---|---|---|
| released configuration | 1.4585 | **0.3079** | 1.4791 | 0.2973 |
| corrected pipeline | 4.6034 | **0.0149** | 4.5525 | 0.0153 |

The mode effect is +1.15 and +4.59. The data effect is +0.011 and **−0.0004**.
There is no generalisation gap in either run — on the corrected checkpoint the
held-out loss is fractionally *lower* than the training loss — and the
training-mode held-out values match the training logs to within 2%.

The mechanism is in the checkpoints, and it has two factors rather than one.
The projector's `BatchNorm1d`, specified by the released configuration
[REF:cfg], carries a running variance of order 10⁻⁴ in all three of our
checkpoints. In evaluation mode the layer divides by the square root of that
quantity, so any drift between the stored statistics and the current activations
is amplified by a factor of 72 to 141; a squared error inflates that by two
further orders of magnitude. But amplification alone is not sufficient. The
checkpoint saved at the minimum of our accidental learning-rate cycle has an
amplification of 78 and a gap of exactly 1.00×, because at a learning rate of
10⁻⁷ the weights had stopped moving and the running statistics had caught up.
The gap requires both a large amplification and weights that are still moving.

That this is a property of particular checkpoints rather than of the
architecture is established directly by the authors' released weights, which use
the same two `BatchNorm1d` layers. We measured their evaluation-to-training gap
on the same held-out clips and found **1.09×** — calibrated. Their projector's
running variance is 0.0172, **89 times larger than our corrected checkpoint's
0.00019**, so their evaluation mode divides by 0.131 where ours divides by
0.014, an amplification of 7.6 rather than 72. For contrast the second
normalisation layer, `pred_proj`, is near-identical across all four checkpoints
(1.163 against 1.157). It is the projector alone that is near-degenerate in
ours.

Why our projector output is so much narrower than theirs we do not establish.
Training length is the obvious candidate — we train for the ten epochs of the
paper's appendix against the repository's hundred (Table 1) — but we have not
tested it, and we report it as open. The practical consequence does not depend
on the cause: their released checkpoint required no recalibration and the
figures we report for it in §4.5 and §5.3 are unaffected, while all three of
ours did.

Two further observations support this account.

First, the size of the gap tracks the learning rate. One of our runs applied a
cyclic schedule by accident (§3.4), sweeping the rate from 1×10⁻⁵ down to
1×10⁻⁷ and back. The ratio of evaluation to training loss follows it with
r = +0.899 on a log-log scale, reaching exactly **1.00×** at the minimum, where
the weights stop moving and the running statistics catch up, and reopening to
186× as the rate climbs again.

Second, the artifact is repairable without touching a weight. Resetting the
running statistics and accumulating a cumulative average over 100–200 training
batches in training mode — a standard "precise BN" recalibration — restores
agreement:

| checkpoint | eval before | eval after | train mode | gap |
|---|---|---|---|---|
| released configuration | 1.4585 | **0.3061** | 0.3076 (unchanged) | 4.7× → 1.0× |
| corrected pipeline | 4.6034 | **0.0085** | 0.0151 (unchanged) | 302.7× → 0.6× |
| already-calibrated control | 0.1846 | 0.1845 | 0.1811 (unchanged) | 1.0× → 1.0× |

The first row lands on the training-mode value we had measured independently
beforehand, which validates the procedure on a case whose answer was known. The
third row is the control: on a checkpoint whose statistics were already correct,
the loss does not move, so recalibration is a repair and not a general
performance improvement.

The consequence for checkpoint selection is worth stating separately, because it
is easy to reproduce elsewhere. Our training loop saved a "best" checkpoint by
validation loss. Under this artifact, that criterion does not select the best
model: it selects the epoch whose normalisation statistics happen to be best
calibrated. In the cyclic run, the saved checkpoint is precisely the epoch at
which the gap reaches 1.00×.

We emphasise the scope. This is a property of our measurement of our
reimplementation, using an architecture the released configuration specifies. We
make no claim about the original authors' training procedure, which may
recalibrate, evaluate differently, or never encounter drift of this size.

---

## 4.4 Training under the released and the corrected configuration

*(~550 words)*

With the measurement repaired, the training results can be read directly. Both
runs below use the same data, clip index, learning rate, regularisation weight,
context length, schedule (none), epoch count and seed; they differ only in the
four pipeline corrections of §3.2.

**The released configuration, as reimplemented, plateaus.** Its training loss
reaches approximately 0.30 within the first epoch and stays there — the median
per-epoch value moves from 0.292 to 0.304 over ten epochs, a drift of under 4%,
with a fitted slope of +0.0004 per epoch. Recalibrated held-out loss: **0.3061**.
This is not instability; it is a floor.

The floor has a straightforward cause, and it is the deviation quantified in
§3.2. Under our original clip loader the predictor received a single
sub-sampled action and was asked to explain a five-step displacement, with the
remaining four actions unobserved. Measured on the released dataset, the action
supplied implies a displacement wrong by a median of 25.59 units against a
typical per-block movement of 13.3. From the predictor's perspective the
majority of its target's variance was unexplainable from its input, and a
conditional mean is the best available fit.

**The corrected configuration converges.** With actions gathered densely and the
input normalisations applied, the training loss descends monotonically at every
epoch — 0.0412, 0.0268, 0.0226, 0.0205, 0.0186, 0.0176, 0.0166, 0.0159, 0.0148,
0.0146 — a 65% reduction with no oscillation, and the lowest value at the final
epoch. Recalibrated held-out loss: **0.0085**, against 0.3061 for the released
configuration. The corrected pipeline is **36 times better** on the same
held-out clips.

The resulting model predicts substantially better than a frozen-world baseline.
Given the dense action sequence it was trained on, its one-step error is
**0.068** relative to that baseline; given the displacement-matched
constant-action encoding that a planner is able to emit, **0.116**. The gap
between those two figures — a factor of 1.7 — is what a planner gives up by
being unable to vary its action within a frameskip block, and we report it as a
stated limitation rather than a surprise. Removing the action normalisation
alone moves the second figure to 0.337, confirming empirically a deviation we
found only by reading the reference source.

Two remarks on scope. Our runs use ten epochs, following the paper's appendix,
while the repository configuration specifies one hundred (Table 1); the
convergence reported here is therefore convergence within the paper's stated
budget, not a claim about the asymptote. And all figures are from a single seed.

Finally, the regulariser behaves as the original describes throughout. Mean
embedding spread remained within 0.830–0.960 for the released configuration and
0.797–1.039 for the corrected one, with no monotone decline in either and no
collapse under any configuration we trained — including at the learning rate
where the validation loss appeared not to settle. The two-term objective was
sufficient to prevent collapse without an exponential moving average, a frozen
encoder, or auxiliary supervision.

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

## 5.1 A silent evaluation-domain gap

*(~400 words as drafted)*

Our most transferable finding concerns not the method but the way it was
evaluated. For three paid training runs we measured planning success in a
32×32-pixel fixture built to iterate cheaply on a CPU laptop, and every one of
those measurements was worthless. The fixture reproduces the environment's
layout faithfully enough that the two are difficult to tell apart by eye: same
two rooms, same dividing wall, same door, same red agent. It differs in an
inset border with corner ticks, and in a door that is narrower and higher.

Those differences are invisible to a human and decisive for a Vision
Transformer. Encoding fixture frames and measuring the distance to the nearest
real-data latent gives a median of 61.03, against a nearest-neighbour spacing
within the real data of 2.43 — the evaluation frames sat twenty-five times
further from the training distribution than training frames sit from each
other. Every planning number produced there was an extrapolation, and all of
them came back below a random-action control.

Three properties of this failure are worth stating, because each defeated a
check we thought sufficient.

It is **invisible to a representation probe.** A linear probe recovering agent
position from the encoder's output scored R² 0.9916 throughout, and the probe
itself is evaluated on real frames, so it never registered the shift.

It is **fully explained by rendering style alone.** Holding the environment
dynamics fixed and changing only the renderer reproduces the anomalous result to
within noise (57.5 against the observed 56.6), so no property of the dynamics,
the planner, or the checkpoint is required to explain three runs of apparent
failure.

It is **cheap to detect once measured rather than argued.** The same
instrument — median latent distance from evaluation frames to their nearest
training-set neighbour — reads 61.03 on the fixture and 0.01 on the real
environment. We now run it as a precondition inside every evaluation, which
refuses to emit a success rate when the check fails.

fidelity…"*

The general lesson is that visual fidelity is not distributional fidelity, and
that a probe demonstrating a representation is good does not demonstrate that
the inputs being fed to it are in-distribution. A debugging fixture that looks
right is exactly the kind of artifact that survives review by inspection.

That lesson recurs in a different form in §4.3, where a normalisation layer's
stored statistics — not the model, and not the data — determined a reported
loss for three training runs. In both cases a quantity we were measuring
routinely and reporting confidently was a property of the instrument rather than
of the system. In both cases the check that would have caught it was cheap, and
in neither case had we thought to run it. We return to this in §6.3.

---

# Questions for the original authors

Three of the four are answerable from released artifacts. Say so when you ask —
it turns each question into a request for confirmation rather than for
information, and shows you have done the work.

| # | question | answerable from artifacts? |
|---|---|---|
| 1 | `history_size` for TwoRoom: Appendix E says 1, the repo config says 3 | **Yes** — the released checkpoint's `predictor.pos_embedding` is (1, 3, 192), so 3 is operative |
| 2 | Epochs: Appendix E says 10, the repo config says `max_epochs: 100` | **No.** This one genuinely matters — our reproduction does not converge in 10 |
| 3 | `goal_offset_steps`: the eval config uses 25, the paper implies 100 | **No.** We measure this as worth 24 points (72% vs 48%) |
| 4 | Are ImageNet normalisation and dense action gathering intended, given neither appears in any config? | **Yes** for both — but worth reporting, because a reproducer following configs alone gets a silently broken model |

**One more worth adding, and possibly the most valuable to us:** the learning
rate and schedule that produced the released checkpoint. Our reimplementation
produces a predictor beating a frozen-world baseline only when the learning rate
falls to ~1e-6 or below, whereas the released config specifies a constant 5e-5.
If they used a schedule that is not in the config, that single answer explains
our central negative result.

## 5.2 A pre-registered effect that did not survive a change of checkpoint

A published critique of latent world models argues that scoring a plan by
Euclidean distance between latent states conflates latent proximity with
reachability, and predicts that a planner will fail disproportionately when a
goal requires moving *away* from it before approaching [REF:critique]. TwoRoom
provides a clean test: goals in the opposite room require routing through a
single door, and goals in the same room do not. We designed and pre-registered
an experiment to measure it, obtained a large and highly significant effect, and
then found that the effect does not survive either a change in how the model is
driven or a change of checkpoint. We report all three arms.

### The design

The obvious confound is distance: cross-wall goals are farther on average. We
therefore built a matched-pair design. From the 6,056 episodes long enough to
supply a goal at our longer offset, we performed one-to-one caliper matching on
start-to-goal distance, obtaining **110 matched pairs**. Matching quality far
exceeded the caliper: the median within-pair distance difference was 0.02 units
against a caliper of 6, the worst was 0.35, and the two arms' median distances
were identical at 124.8. A simulated power analysis on the matched-pair test
gave 82% power for the 20-point difference we assumed. The design, the decision
rule for every outcome, and the exact list of 220 episodes were committed to the
repository before any of them was evaluated.

Two further explanations were eliminated after the fact. First, reachability: a
door-routing oracle with access to true positions, planning no further ahead
than the real dynamics allow, reached the goal in **110 of 110 episodes in both
arms** within the step budget, so the achievable ceiling is 100% for both
geometries. Second, direction: same-room goals in this set are predominantly
vertical and cross-wall goals predominantly horizontal, but the oracle clears
both at 100%, so orientation cannot account for a difference. A residual
path-length effect was bounded analytically: deleting the quarter of cross-wall
episodes with the longest geometric requirement and counting every one of them
as a length-caused failure still leaves a 27-point difference.

### The three arms

| driving regime | same-room | cross-wall | difference | exact p |
|---|---|---|---|---|
| pre-correction checkpoint, as measured | 79.1% | 40.0% | **+39.1** [+27.2, +51.0] | 3.4 × 10⁻⁸ |
| pre-correction checkpoint, action-scale corrected | 74.5% | 61.8% | +12.7 [+0.5, +24.9] | 0.054 |
| corrected checkpoint | 14.5% | 20.9% | −6.4 [−16.4, +3.7] | 0.248 |

The first row is the pre-registered primary test and stands as registered: on
that checkpoint, with distance matched pair-by-pair and reachability verified at
100% for both geometries, the planner reached 79.1% of same-room goals and 40.0%
of cross-wall goals, a difference of 39.1 points at p = 3.4 × 10⁻⁸.

The second row applies a correction to the action scale supplied to the same
checkpoint (§5.4). The difference falls to +12.7 points and the test becomes
marginal. The third row uses the corrected checkpoint of §4.4. The point estimate
is now negative, and the test is not significant.

The estimate therefore falls monotonically across the three arms and crosses
zero. We are careful about what the third arm establishes. It **rules out** an
effect as large as the +20 points the study was designed to detect: the
confidence interval excludes it. It does **not** establish a reversal — at
p = 0.248 the honest reading is no detectable difference. And a confound must be
named: overall success across the three arms is 61.6%, 68.2% and 17.7%, so they
are not compared at matched performance, and the low rate in the third arm could
in principle mask a real difference. The random-action control reached 0 of 110
in both geometries in every arm; being at the floor, it is uninformative about
whether the episodes differ in intrinsic difficulty, and we do not use it as
evidence that they do not.

### What we take from it

We do not claim there is no geometric effect on this task, and we do not claim
the published critique is wrong. What our three arms support is narrower and, we
think, more useful: **a large, well-powered, pre-registered, confound-controlled
effect measured on one reproduction checkpoint survived neither a change in how
that checkpoint was driven nor a change of checkpoint.** Everything a careful
reader would ask of the first row — matching, power, a committed episode list, an
oracle ceiling, an analytic bound on the residual confound — was in place, and
none of it was sufficient to make the number a property of the method rather
than of the checkpoint.

The implication for reproduction work is direct. A reproduction that establishes
a mechanism-level effect on its own trained checkpoint has established it for
that checkpoint. Whether it generalises is a separate question requiring
separate checkpoints, and in our case the answer was no. We report the
pre-registered result because we pre-registered it, and we report the other two
arms because they are what the result turned out to mean.

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

## 5.4 What the corrected pipeline does to the representation

*(~380 words)*

The pipeline corrections of §3.2 change the predictor's task substantially, and
it is natural to ask what they do to the representation the encoder learns. We
compared the two encoders — one trained under the released configuration, one
under the corrected pipeline — on 4,000 identical held-out frames, with each
encoder receiving the pixel convention it was trained with.

Three of the four measurements are unchanged. Position is decodable at R² 0.9977
against 0.9971 by a linear probe, and at 0.9995 against 0.9994 by a two-layer
network. The summed action executed between two frames is decodable from the
pair of embeddings at 0.9207 against 0.9132. On the information a planner needs,
the two encoders are equivalent.

The fourth measurement is not. The **effective rank** of the embedding cloud —
the participation ratio of its covariance spectrum, which counts how many
dimensions the representation actually occupies — rises from **11.9 to 67.8 of
192**. The corrected pipeline produces a representation spread over roughly
five and a half times more dimensions while carrying the same position and
action information.

We tested one explanation and it did not survive. Our hypothesis was that the
extra dimensions carry dynamics-relevant structure, purchased at the cost of
some of the linear structure that a position probe reads: under the released
configuration the predictor received one sub-sampled action to explain a
five-step displacement (§3.2), so it could not usefully constrain the encoder,
leaving it free to become a nearly pure position code. That hypothesis predicts
that the corrected encoder should make actions *more* linearly decodable. It
does not — action decodability is unchanged to within 0.008. We therefore report
the rank difference as an observation and offer no account of what the
additional dimensions encode.

We record one methodological point, because it changed our own answer. Both
measurements above were first taken before the normalisation repair of §4.3, and
both were wrong: the corrected encoder then appeared to have an effective rank of
16.5 rather than 67.8, and appeared to lose action decodability (0.8733 rather
than 0.9132). The position figures were unaffected, because a ridge probe
standardises its inputs and is scale-invariant. Any measurement on a latent space
whose scale a normalisation layer controls should be taken after verifying that
layer, and probe-style measurements are precisely the ones that will not warn
you.

---

## 5.5 Scoring geometry within the training distribution

*(~220 words)*

A published critique of latent world models argues that scoring plans by
Euclidean distance in latent space conflates latent proximity with reachability
[REF:critique]. §5.2 tests the behavioural prediction that follows from it.
Here we report the representation-level measurement, which is narrower and
points the other way.

Sampling pairs of real frames at matched physical distance and comparing their
latent separation, embeddings of positions in *different* rooms are separated by
**1.79 times** as much as embeddings of positions in the same room at the same
physical distance. The wall is represented: at equal Euclidean distance in the
arena, the latent space places cross-wall pairs farther apart, which is the
direction the scoring function would need in order to prefer routing. The strong
form of the critique — that the latent geometry is blind to the obstacle — is
therefore not supported for this encoder.

Two qualifications. This measurement is from a single checkpoint, taken before
the pipeline corrections, and was not repeated afterwards. And given §5.2, where
the behavioural effect this measurement was originally offered to explain did
not survive a change of checkpoint, we do not present the 1.79× figure as
support for any behavioural claim. It is a property of one encoder's latent
geometry, reported as such.

---

# 6 Discussion

*(~1,150 words)*

## 6.1 What was easy

The released environment installs from PyPI and runs without modification. The
released checkpoint downloads from the model hub and, once the architecture is
reconstructed from its configuration, loads with strict key matching. The
representation result is easy to obtain and robust: a position probe reaches
R² 0.99 within a single epoch under every pipeline configuration we tried, and
never degraded thereafter. The anti-collapse regulariser behaves exactly as
described, in every run, without tuning.

## 6.2 What was difficult

**The four deviations that mattered were invisible in configuration files.**
Dense action gathering, the programmatic action-encoder width, ImageNet pixel
normalisation and action z-scoring are all determined in code, and a reproducer
following the released configuration alone obtains a silently broken model.
Three of the four we found only by reading the reference's data loader and
training script line by line, after a measurement told us something was wrong;
the fourth we found only because their released weights refused to behave.

**The evaluation-domain gap survived a 0.99 probe for three paid runs (§5.1),
and a normalisation artifact concealed the training result for as long
(§4.3).** Both were failures of instrumentation rather than of the method under
study, and in both cases we reported the wrong conclusion confidently before
finding the cause.

**Ten thousand NaN actions sit at the end of each episode in the released
dataset.** Our original loader read one action in five and stepped over them by
luck; the corrected loader reads all of them, and would have produced NaN
gradients from the first affected batch. The reference drops NaN rows before
computing normalisation statistics — one line we read and did not implement.

**Our own checks failed more often than the runs did.** Of the gate failures we
investigated, more were caused by defects in the gate than by defects in the
run: a physics assertion measured on normalised rather than raw actions, a
collapse test invalid at the batch size it ran at, a log reader that silently
took the wrong file, and a driving-spec diagnostic that scored a working model
as unusable because it supplied displacement-mismatched actions. We report this
because a reproduction paper that documents only the subject's failures is not
being straight about where the effort goes.

## 6.3 Recommendations

The following are addressed to authors releasing work and to reproducers
attempting it. Each is drawn from a specific failure above.

**1. Publish the data convention, not only the configuration.** Action
aggregation and input normalisation determined every result in this paper, and
neither appears in any released configuration file. A short section of the
README stating how actions are aggregated across a frameskip block, and what
normalisation is applied to inputs, would have saved us several days and one
paid training run. This is the single highest-value change available to authors
of the work we reproduced.

**2. Check fidelity against source, not configuration.** Our audit compared 25
pipeline elements against the reference implementation's source with
file-and-line citations, and marked each as matching, deviating, or unverified
(Table 1, Appendix A). Every one of the four expensive deviations lived in code
that no configuration file mentions. An audit against configurations would have
found none of them, and we had performed exactly such an audit twice before,
concluding both times that we matched.

**3. Assert data contracts against physics, not shapes.** The environment we
studied is deterministic, so displacement across a block equals speed times the
summed actions of that block. That identity is one line of code, and it
detects the action-aggregation deviation immediately and unambiguously. A
shape check does not: the incorrect array had the correct rank, the correct
dtype, and plausible magnitudes. Where a dataset admits an exact invariant,
assert the invariant.

**4. Differential-test against a released artifact.** Running the authors'
released checkpoint through our own evaluation harness (§4.2) established that
our protocol reproduces the reported result, which no amount of internal
consistency checking could have established. It also caught a convention error
on our side, because their weights refused to behave when driven incorrectly. If
a reproduction target releases weights, running them through your harness before
trusting your own numbers is the highest-information check available.

**5. Verify that evaluation mode measures the model, not the normalisation.**
This is the recommendation we most wish we had received. Where a network
contains batch normalisation, compare its layers' running variance against the
scale of the activations reaching them. In our checkpoints the projector's
running variance was of order 10⁻⁴, so evaluation mode divided by 0.014 and
amplified any staleness in the stored statistics by a factor of 72 or more; in
the authors' released checkpoint the same layer holds 0.0172 and amplifies by
7.6, and its evaluation mode is faithful. Two symptoms are worth watching for:
a validation loss that oscillates while the training loss does not, and a gap
between the two that shrinks as the learning rate falls. Recalibrating the
running statistics — a few hundred forward passes in training mode with no
gradient updates — is a cheap repair, and on a checkpoint whose statistics are
already correct it is a verified no-op (§4.3).

A corollary concerns checkpoint selection. Under this artifact, saving the
checkpoint with the best validation loss does not select the best model: it
selects the epoch whose normalisation statistics happen to be best calibrated.
In our cyclic run, the saved checkpoint is precisely the epoch at which the gap
reaches 1.00×, and we spent some time believing that epoch's weights were
special.

## 6.4 Communication with the original authors

We wrote to the corresponding author on 31 July 2026, reporting the two
undocumented pipeline steps of §3.2 and asking four questions: which of the two
published `history_size` values is operative for TwoRoom; whether the dense
action gathering and ImageNet normalisation are intended as we describe them;
which goal offset the reported figure uses; and what learning rate and schedule
produced the released checkpoint. As of submission we have received no
response, and the questions above remain open. We note that three of the four
are answerable from the released artifacts, and we have answered them that way
in Table 1; the fourth is not, and it bears on §4.4.

---

# 7 Limitations

We list the constraints on our results in roughly descending order of how much
they should change a reader's confidence.

**A single seed throughout.** Every training run, every planning evaluation and
every probe in this paper uses one seed. We make no estimate of seed variance,
and none of our differences should be read as robust to reinitialisation. This
matters most for the planning figures in §4.5, where the differences we report
between checkpoints (94.0% against 84.0%, p = 0.0625; 94.0% against 78.0%,
p = 0.0574) are already not established at our sample size of fifty episodes.
Our evaluations are, however, deterministic: re-running a planning evaluation
from the committed commit reproduces its per-episode outcomes exactly, so the
variance we have not measured is between-seed and not within-run.

**Ten epochs, not one hundred.** We follow the paper's appendix, which states
ten epochs; the released repository configuration specifies `max_epochs: 100`
(Table 1). Our convergence result (§4.4) is therefore convergence within the
budget the paper states, and says nothing about the asymptote. At four rented
GPU-runs of roughly six dollars each, a hundred-epoch run was outside our
budget, and we note that this constraint is itself a reproducibility finding: a
reader with our resources cannot test the repository's own configuration.

**Our results characterise our checkpoints, not the method.** This is the
strongest lesson of §5.2. A same-room planning advantage that was pre-registered,
distance-matched, verified reachable by an oracle, and significant at
p = 3.4 × 10⁻⁸ on one checkpoint fell to +12.7 points under a change of action
scaling and to −6.4 points on a different checkpoint. Effect sizes measured on a
single reproduction checkpoint should not be read as properties of the method,
and we no longer read our own that way. Claims 23 and 24 in particular
(§5.3, §5.5) were measured on one checkpoint only and were not re-tested after
the pipeline correction.

**The mechanism of the long-horizon reversal is untested.** Our principal
finding (§5.3) is that one-step prediction accuracy orders short-horizon
planning success and fails to order long-horizon planning success, with the two
most accurate checkpoints finishing farther from the goal than a random-action
control. We offer a candidate explanation — that a sharper cost landscape leads
the optimiser to commit to near-maximal actions that a terminal-cost objective
does not penalise until the horizon ends — but we have not tested it, and a
confound remains: the two overshooting checkpoints use a three-frame context and
the cautious one uses a single frame. Three checkpoints cannot separate
prediction accuracy from context length. Distinguishing them requires training
matched checkpoints varying one factor at a time.

**An independent reimplementation, with a documented but non-empty deviation
set.** We did not rerun the authors' code; we reimplemented from the released
code and paper, which is what surfaced the four undocumented pipeline
differences of §3.2. Our audit matched twenty of twenty-five elements against
reference source, and the remainder are listed in Table 1. Two deviations
deserve individual mention. Our runs use full 32-bit precision where the
reference specifies bfloat16; the direction favours numerical accuracy and
cannot explain a failure to converge, but it is a difference. And the reference
computes action normalisation statistics on a two-wide raw column and applies
them to a ten-wide clip array; the broadcast it relies on is not visible in the
source we read, so we tile the two-wide statistics across the five concatenated
actions. That is our interpretation, and if it is wrong, §4.4's figures move.

**Episode selection differs from the original's, which is not published.** Our
episodes are a fixed random draw at a stated seed, with the start at the first
frame of an episode and the goal a fixed number of steps later. The authors'
selection is not described in enough detail to reproduce. Consequently the
like-for-like comparison in §4.5 is our checkpoint against the authors' released
checkpoint on identical episodes (94.0% against 84.0%), not against the reported
87%, which was measured under a selection we cannot replicate.

**Recalibration data overlaps the evaluation episodes.** The BatchNorm
recalibration of §4.3 accumulates statistics over training clips drawn from the
same dataset as the planning episodes. Precise-BN sets normalisation statistics
only and updates no weight, so it cannot memorise episode content, and our
control run shows it is a no-op on a checkpoint whose statistics are already
correct. We nonetheless record the overlap rather than leave a reader to find
it.

**A known defect in the committed evaluation reports.** Until we corrected it,
the deviations block printed at the end of each planner report used hardcoded
default values rather than the run's actual parameters, so reports from runs at
a non-default goal offset, from the authors' checkpoint, or over the committed
episode set misdescribe themselves in that block. The measurements in those
reports are unaffected — the header of each report states the parameters
correctly — and the repository records which reports predate the fix.

**Scope.** All results concern the TwoRoom diagnostic environment. We make no
claim about the original's embodied or zero-shot results, about its other
environments, or about the method's behaviour at scales other than the
18.03M-parameter configuration studied here.

---

# 8 Conclusion

*(~330 words)*

The three claims we set out to test (§2) resolve as follows. The representation
claim reproduces directly and easily. The planning claim reproduces, at 94.0%
against a reported ~87% and against 84.0% for the authors' own weights under our
protocol, once four undocumented conventions are corrected. The training claim
does not reproduce from the released configuration files alone, and does
reproduce once those conventions are supplied — which we take to be a
documentation gap rather than a defect in the method.

What we did not anticipate is how much of the work would consist of establishing
that our own measurements meant what we thought they meant. Three training runs
appeared not to converge because a normalisation layer's stored statistics, not
the model, determined the loss we were reporting. Three runs' worth of planning
results were produced in a debugging fixture that looks correct and sits
twenty-five times outside the training distribution. A carefully pre-registered
effect of +39.1 points at p = 3.4 × 10⁻⁸ fell to −6.4 points on a different
checkpoint. In each case a probe or a summary statistic read exactly as it
should have while the underlying quantity was wrong, and in each case the check
that would have caught it was cheap.

The finding we expect to be most useful outside this reproduction is the
negative one. A world model that predicts one step ahead seven times more
accurately than another was not detectably better at short-horizon planning and
was decisively worse at long-horizon planning — and this holds for the authors'
released checkpoint as well as for ours. Selecting a world model by held-out
prediction error is not a reliable way to select a world model for planning, and
on this task at the longer horizon it would have selected the worst of the three
available.

We release the reimplementation, all four checkpoints, every evaluation report,
the fidelity audit against reference source, the pre-registration, and the gate
outputs, at [REF:repo].

---
