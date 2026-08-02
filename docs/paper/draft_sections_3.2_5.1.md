# Drafts — sections 3.2 and 5.1

Written as paper prose, not outline. Numbers are load-bearing; every one traces
to a committed artifact. Citations marked `[REF:n]` need filling from the
reference repository at the commit you audited.

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

The general lesson is that visual fidelity is not distributional fidelity, and
that a probe demonstrating a representation is good does not demonstrate that
the inputs being fed to it are in-distribution. A debugging fixture that looks
right is exactly the kind of artifact that survives review by inspection.

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
