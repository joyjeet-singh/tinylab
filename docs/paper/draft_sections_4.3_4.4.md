# Drafts — §4.3 and §4.4

**Note on ordering.** The skeleton had §4.3 as "training does not reproduce" and
§4.4 as "convergence is learning-rate-governed". Both were written when we
believed training had failed. They are replaced and **their order is swapped**:
the measurement artifact has to be established before any training result can be
read, because it is the reason the training results looked as they did.

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

The mechanism is in the checkpoints. The projector's `BatchNorm1d`, specified by
the released configuration [REF:cfg], carries a running variance of order 10⁻⁴.
In evaluation mode the layer divides by √(running variance) ≈ 0.01, so any drift
between the stored statistics and the current activations is amplified roughly a
hundredfold, and a squared error inflates that by two further orders of
magnitude. A position probe on the same embeddings was unaffected throughout,
because a ridge probe standardises its inputs and is scale-invariant where mean
squared error is not — which is why the encoder appeared healthy while the
predictor appeared broken.

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

## Drafting notes

- §4.3 **must** precede §4.4. Every number in §4.4 is a recalibrated one and
  would be unreadable otherwise.
- Do not describe the validation series as "training instability" anywhere. It
  is a measurement artifact and the paper's own §4.3 says so.
- The "36 times better" figure compares recalibrated held-out losses. Say so;
  the un-recalibrated comparison is meaningless.
- §4.3's final paragraph — the scope disclaimer about the original authors — is
  not optional. Without it the section reads as an accusation.
- The checkpoint-selection paragraph is the most portable thing in §4.3. Consider
  promoting it to §6.3 (Recommendations) as well, by reference.
