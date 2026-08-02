# Draft — §1 Introduction

*(~750 words. Written after §6 so the contributions list and the
recommendations agree.)*

---

## 1 Introduction

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

## Drafting notes

- The five contribution bullets map one-to-one onto §4.5, §3.2, §5.3, §4.3+§5.1
  and §5.2. Check they still match after any section is revised.
- **Never state 94.0% without the goal offset.** The bullet includes it; keep it
  there.
- The third paragraph ("The reproduction succeeded, but not on the first
  attempt…") is the paper's thesis. If a reviewer reads only one paragraph,
  this is the one that should land.
- `[REF:baselines]` needs the PLDM and DINO-WM numbers as the original reports
  them, cited to the original's table rather than to those papers.
- Do not describe the original as flawed anywhere in §1. The undocumented
  conventions are a documentation gap, the method reproduces, and §6.1 credits
  the release.
