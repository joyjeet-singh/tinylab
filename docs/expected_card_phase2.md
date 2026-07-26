# Expected card — phase2_dense_reference (written BEFORE launch)

Required by the standing rule on multi-change runs: a run that changes more
than one thing must state, before launch, what each outcome will license, what
it will not, and that attribution is being traded away on purpose.

**Commit this file before the run starts.**

## The question

Does the released LeWM configuration converge its predictor in 10 epochs once
four verified deviations in our pipeline are corrected?

Run 0 answered "no" with one deviation live (`action_dim` 2 instead of 10) and
three more that no config revealed (subsampled actions, unnormalised pixels,
unnormalised actions). This run answers the same question with all four closed.

## What is being changed, and why attribution is not needed for it

Four changes, all moving **toward** the reference, each independently justified
by a source citation:

| change | reference source |
|---|---|
| dense actions, reshaped to (T, frameskip × action_dim) | `stable_worldmodel/data/buffer.py` `_gather_clip` |
| `action_dim` 2 → 10 | `le-wm/train.py:68` sets it to `frameskip × action_dim` |
| ImageNet pixel normalisation | `le-wm/utils.py:6` `get_img_preprocessor` |
| z-scored actions | `le-wm/train.py:65` + `utils.py:25` |

None is a tuning knob. Each is a correction that should be made whether or not
it changes the outcome, which is what distinguishes this bundle from the
July-21 one — that bundle moved hyperparameters *away* from the reference to
values we chose, and its improvement was consequently uninterpretable.

**Traded away deliberately:** if this run converges, we will not know whether
the dense actions, the pixels, or the action normalisation was responsible.
Separating them needs three more runs we do not have. Accepted.

## Baselines to beat

Run 0, same learning rate, same SIGReg weight, same history size, same absence
of a schedule, 10 epochs:

- prediction loss: 5.48 at epoch 0, then oscillating **0.490–2.070** for nine
  epochs, linear slope −0.0044/epoch, consecutive swings up to 107% of the mean
- SIGReg: 32.2–188.8, also unsettled
- embedding spread: 0.830–0.960 (no collapse)
- probe R²: 0.9951 at epoch 0, drifting **down** to 0.9922 by epoch 9

## Outcomes, and what each licenses

**A — converges.** Prediction loss falls monotonically or near-monotonically
and settles, with epoch-to-epoch swings well under Run 0's ~100% of the mean.

*Licenses:* "The released configuration does converge once four
implementation deviations are corrected. Our earlier non-convergence was ours,
and here is exactly how each deviation was found." That is a more useful result
for a reproducer than the original claim.
*Does not license:* attributing convergence to any single one of the four. Nor
does it retroactively validate `ckpt_best`, which was trained under all four
deviations plus three hyperparameter departures.

**B — does not converge**, oscillating like Run 0.

*Licenses:* "The released configuration does not converge its predictor in 10
epochs, with the four known implementation deviations eliminated." The founding
claim survives with its biggest confound removed — stronger than it is today.
*Does not license:* claiming the method cannot converge. The repo config
specifies 100 epochs and we run 10; the reference's own trained checkpoint
plainly exists and works. The honest form is a statement about 10 epochs and
this reimplementation.

**C — trains but the representation degrades**, e.g. probe R² materially below
Run 0's 0.99, or the spread collapsing.

*Licenses:* nothing about convergence. It means one of the four changes broke
something, most likely the pixel normalisation, since it moves the input
distribution from std ~0.2 to ~1.4. Diagnose locally before drawing any
conclusion.

**D — NaN or crash.** The pre-flight and the NaN audit exist to make this
impossible; if it happens anyway, the gates have a hole and finding it takes
priority over the result.

## Pre-launch checklist

- [ ] `python3 verify_dense_actions.py` passes, including the NaN audit
- [ ] `python3 preflight_local.py --config configs/phase2_dense_reference.yaml`
      passes with every deviation listed and explained
- [ ] the loader's module flags agree with the config's `data:` block
- [ ] this card and the config are committed **before** the run starts
- [ ] retrieval rehearsed: prove the checkpoint pull works before destroying
      the instance

## One prediction, recorded so it can be wrong

I expect **A**, mainly because the old convention's action was wrong by
roughly twice the actual per-block movement (median error 25.6 units against a
13.3-unit typical move), which is a large enough noise floor to explain an
oscillating predictor loss on its own. Confidence: moderate, not high — the
authors' own checkpoint achieves only 0.448 relative one-step error with dense
actions, so dense actions alone do not guarantee a well-behaved predictor.
