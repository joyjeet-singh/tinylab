# Draft — §3.1, §3.3, §3.4, §3.5

§3.2 (the fidelity audit) is drafted separately. These are the surrounding
subsections. Total ~1,150 words.

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

## Drafting notes

- §3.1's final paragraph (reimplemented rather than reran) justifies the whole
  methodology and should not be cut for length.
- §3.3 is deliberately short. It is a precondition, not a result; §5.1 carries
  the finding.
- §3.4's "gates that fail loudly" paragraph and §6.2 make the same point.
  Cross-reference; do not repeat the four examples in both.
- §3.5's closing ratio is a claim about reproduction work in general. Keep it to
  three sentences or cut it entirely — it should not read as a lament.
- Numbers to verify against the manifests before submission: clip count (780k is
  approximate; Run 0 recorded 693,728 train + 77,081 val at history_size 3, and
  Runs 1–2 recorded 783,728 + 87,081 at history_size 1), and per-run wall-clock.
