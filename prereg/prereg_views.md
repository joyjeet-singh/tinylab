# Pre-registration — Latent-Space Visual Study (Views 1-3)

Committed BEFORE looking at any current-weights plot. Purpose: stop View 1 from
becoming a Rorschach test, and make the eventual before/after (current vs Phase-1)
comparison honest. This file is written first; the plots come after.

Run: rental_tworoom_vit224_..._143515 (the 10-epoch reference run; planner 2.0%,
mean final distance 26.44; probe R² 0.9937).

---

## The basis rule (ironclad, shared as a RULE not a matrix)

Both panels (current model, and later the Phase-1 model) are drawn in axes defined
the SAME way, even though the projection matrix is fit per-model:

> **Axes = the position probe's two coefficient directions.** Fit the linear probe
> (Ridge, as in toy_plan.probe_position) from latents -> (x, y) ON EACH MODEL, and
> use its two weight vectors (the direction that best recovers x, and the one that
> best recovers y) as the 2D plot axes for that model.

Why this rule and not a fixed PCA basis:
- A fixed PCA basis fit on the current (broken) model would view the Phase-1 model
  through the wrong axes — its variance structure may differ, so the "after" panel
  could look degenerate for a basis reason, not a model reason. That would confound
  the comparison we care about.
- The probe-direction rule anchors axes to REAL POSITION in both models by
  construction: "horizontal ≈ latent direction encoding true x." A change between
  panels is then unambiguously a change in how the model organizes position — which
  IS the folding hypothesis — and cannot be an artifact of basis choice.
- Sharing the RULE (project onto position-directions), not the MATRIX, is what makes
  the two models comparable despite arbitrarily different internals.

Secondary cross-check (report alongside, not as headline): raw PCA-to-2D of the same
latents. If PCA and probe-axis views tell the same story, good; if they disagree,
say so.

---

## Pre-registered predictions (write the call, THEN plot)

### View 1 — latent geometry, colored by true (x,y)
Points = per-state latents (latents_cem.npz `latent`), projected to the probe axes,
colored by true position (`pos`).

- **SMOOTH (planning-should-work shape):** the true-position color gradient is
  MONOTONE across the axes — nearby-in-color stays nearby-in-plot, the two rooms are
  contiguous regions, the doorway band is a continuous bridge between them. Straight
  lines in this space ≈ straight paths in the room.
- **FOLDED (planning-fails shape):** the gradient REVERSES or TEARS — two far-apart
  true positions land adjacent in latent space, OR the doorway band is discontinuous
  / one room folds back over the other. Straight lines cut across folds → CEM steers
  confidently wrong.
- **My prediction (committed):** given planner performs BELOW random (2% vs 6%) at
  R² 0.99, I expect FOLDED — specifically a tear or fold at/near the doorway, since
  that is where cross-room planning must pass and where the below-random behavior
  implies the model's geometry misleads. If instead it looks SMOOTH, the folding
  hypothesis is FALSIFIED for this model and the failure is elsewhere (predictor
  dynamics / scoring threshold) — an honest redirect, recorded as such.

### View 2 — one-step prediction error painted on the room floor plan
Heatmap of `pred_err` over true (x,y); `is_doorway` flag available for a
doorway-vs-interior split.

- **My prediction (committed):** error is NOT uniform — it concentrates at the
  doorway band (cross-room transitions are the hardest to predict), and/or grows with
  distance from the states the model saw most. A UNIFORM error map would be evidence
  the predictor is globally weak rather than geometry-specific — different story,
  recorded as such.

### View 3 — imagined vs real, one missed episode
For a chosen missed episode, overlay: the agent's REAL path (`pos` along that ep_id)
vs the planner's IMAGINED rollout (`imagined` for that episode), each decoded to
(x,y) via the SAME position probe.

- **My prediction (committed):** the imagined rollout peels away from reality in a
  CONSISTENT direction (systematic bias — the model imagines it is approaching the
  goal while physically drifting), NOT random scatter. This is what "below random"
  predicts: a confidently-wrong plan, not a noisy one. If the imagined path instead
  drifts randomly, the failure is noise/imprecision, not systematic misdirection —
  recorded as such.

---

## Discipline

- All arrays come from latents_cem.npz + latents_random.npz (raw latents; projection
  is done here, downstream, reversibly).
- The manifest (git commit + ckpt md5 + seed) ties every figure to exact inputs.
- Phase-1 model, when it exists, is dumped the SAME way and plotted with the SAME
  rule (its own probe axes), for the before/after figure.
- Falsifiers above are real: a SMOOTH View 1, a UNIFORM View 2, or a RANDOM-scatter
  View 3 each redirect the diagnosis rather than confirm the story, and get reported
  honestly if seen.
