# Draft — §7 Limitations

*(~900 words. Written from the claims ledger; no new measurement required.)*

---

## 7 Limitations

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

## Drafting notes

- Do not soften the first item. One seed is the single largest constraint and a
  reviewer will find it immediately; stating it first buys credibility for the
  rest.
- The "our results characterise our checkpoints" paragraph is doing double duty
  as a limitation and as the honest framing of §5.2. Cross-reference rather than
  repeat.
- The reports-defect paragraph is optional if every cited report is regenerated
  after the fix. Keep it if any are not.
- Add a line on compute totals (four GPU-runs at ~$6; all evaluation on one
  CPU laptop) if §3.5 does not already carry it.
