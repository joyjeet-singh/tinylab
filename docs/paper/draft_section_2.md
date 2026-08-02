# Draft — §2 Scope of reproducibility

*(~600 words. Must agree with §1's contributions and with the claims ledger.)*

---

## 2 Scope of reproducibility

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

## Drafting notes

- The three block quotes need exact wording and section numbers from the
  original. Do not paraphrase a claim you are about to grade.
- Claim 2's verdict is deliberately three-part. Resist compressing it to
  "reproduced"; the protocol reproducing and our checkpoint exceeding are
  different facts with different evidence.
- Claim 3's final paragraph is the one a reviewer will check most carefully.
  "The released configuration is insufficient to specify a converging run" is
  the defensible claim; "the method does not converge" is not, and must not
  appear anywhere in the paper.
- The probe-protocol caveat in Claim 1 is small but earns trust early: it is us
  reporting a disagreement between our own numbers before anyone asks.
