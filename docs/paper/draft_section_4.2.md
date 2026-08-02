# Draft — §4.2 The evaluation protocol reproduces

*(~600 words. Referenced from §2, §4.5, §5.3 and §6.3, so it must be
self-contained but must not restate them.)*

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

## Drafting notes

- Do not restate the four deviations (§3.2), the horizon result (§5.3) or the
  BatchNorm mechanism (§4.3). Reference them.
- The 46% paragraph is the section's most valuable content and the most likely
  to be cut for length. It is the concrete instance behind recommendation 4 in
  §6.3; keep both or neither.
- "It is not a validation of our model in any respect" prevents a reader from
  carrying the 84% forward as evidence about our checkpoint. Keep it.
- The 5.1-vs-0.410 figures come from the driving-spec measurement; cite the
  committed report rather than restating the method.
