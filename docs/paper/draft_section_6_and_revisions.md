# Draft — §6 Discussion, with revisions to §4.3 and §5.1

The two-factor result from the authors'-checkpoint check changes §4.3's
mechanism paragraph and adds a recommendation to §6.3. Both revisions are given
in full below, as drop-in replacements.

---

# REVISION — §4.3, mechanism paragraphs

*Replaces the single paragraph beginning "The mechanism is in the checkpoints."*

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

*A position probe on the same embeddings was unaffected throughout, because a
ridge probe standardises its inputs and is scale-invariant where mean squared
error is not — which is why the encoder appeared healthy while the predictor
appeared broken.* [keep as the following paragraph]

---

# REVISION — §5.1, closing paragraph

*Replaces "The general lesson is that visual fidelity is not distributional
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

# §6 Discussion

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

## Drafting notes

- §6.2's fourth paragraph (our own checks failing) is the one an editor will
  suggest cutting. Keep it. It is the most credible thing in the section.
- Recommendation 5 is the paper's most transferable contribution and should be
  cross-referenced from the abstract.
- Recommendation 1 is addressed to the authors specifically; keep it in the
  register of a suggestion, not a complaint. They released code, data and
  weights, which is why this reproduction was possible at all — §6.1 should say
  so and does.
- Update §6.4's final sentence if a reply arrives before submission.
