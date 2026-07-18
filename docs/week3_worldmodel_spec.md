# Week 3 spec — a tiny world model from scratch (and watching it collapse)

Goal: build the smallest honest version of the LeWM pipeline — pixel
observations -> encoder -> latent -> action-conditioned predictor -> CEM
planning — and personally witness representation collapse, fix it with a
variance-style regularizer, and probe what the latent learned. ~500 lines
total across small files. This week is the direct on-ramp to Phase 1.

## Plain-language map of the ideas (read first)

A world model here is two machines: an ENCODER that compresses each image
into a short list of numbers (the latent — the model's private summary of
the scene), and a PREDICTOR that answers "given this summary and this
action, what will the next summary be?" Training signal: make the predicted
next-summary match the actual next-summary (MSE).

The trap — REPRESENTATION COLLAPSE: the easiest way to make predictions
perfectly accurate is for the encoder to output the SAME summary for every
image. Then prediction is trivial and the loss hits zero — and the summary
contains nothing. The loss is satisfied; the model is useless. You will
build this failure ON PURPOSE in Stage 1 and watch it happen in the logs.

The fix family: force the summaries to stay spread out. LeWM's SIGReg pushes
the whole cloud of latents toward a standard bell-curve shape along many
random directions. We build a simpler cousin first (a variance floor per
dimension, VICReg-style), then a "SIGReg-lite," so the paper's Appendix A
reads as familiar machinery later.

Planning: once the predictor is decent, you can plan WITHOUT touching the
environment — imagine futures in latent space. CEM (cross-entropy method)
is guess-and-check made respectable: sample many random action sequences,
score each by how close its imagined final summary lands to the goal's
summary, keep the best few, resample around them, repeat.

## Stage 0 — the environment: dotworld (envs/dotworld.py)

Build it ourselves; zero heavy dependencies; and it doubles as Phase 2
apparatus (the "dimensionality dial").
- 32x32x3 numpy image; background dark; agent = 3px bright dot.
- True state: (x, y) floats in [0, 32). Action: (dx, dy) in [-2, 2]^2,
  clipped at walls. Deterministic dynamics.
- `reset(seed)`, `step(a)`, `render() -> uint8 array`, and — for probing
  only — `true_state()`. The model NEVER sees true_state during training.
- Config flag `n_blocks: 0` now; a pushable block comes in Phase 2.
- Data collection script `collect.py`: random policy, 2000 episodes x 30
  steps -> one .npz of (obs uint8, actions, episode ids) + a manifest with
  the collection seed and a sha256 of the file. Data is a by-product:
  data/ stays gitignored; the manifest is the reproducibility anchor.

## Stage 1 — build the collapse (wm/train_wm.py, wm/models.py)

- Encoder: small CNN 32x32x3 -> conv(16) -> conv(32) -> flatten -> Linear
  -> latent d=16. Predictor: MLP on concat(z, a) -> z_next_hat. ~200k knobs
  total. Loss: MSE(z_next_hat, z_next). No stop-gradients, nothing else.
- Log per 50 steps to metrics.jsonl: pred_loss, AND the collapse
  instruments: mean per-dimension std of z across the batch (`z_std`), and
  effective spread `z_std_min` (the quietest dimension).
- PREDICTION TO WRITE FIRST: pred_loss will fall toward ~0 AND z_std will
  fall with it. Success this stage = loss near zero + z_std near zero +
  the linear probe (below) failing. A perfect loss and a dead model.

## Stage 2 — the fix (config-switchable, same script)

- Variance floor (VICReg-style): loss += lam_v * mean(relu(1 - std_per_dim)).
  Plain words: any dimension whose spread drops below 1 gets pushed back up.
- Then SIGReg-lite: sample M=64 random unit directions each step, project
  the batch of latents onto each, and penalize each projection's
  (mean^2 + (std - 1)^2). This is moment-matching toward the standard
  bell curve along random directions — the same GEOMETRY as SIGReg, with a
  cruder statistical test (the paper uses Epps-Pulley; note the difference
  honestly in comments). One knob: lam.
- Rerun Stage 1's exact experiment with the regularizer on. Watch z_std
  stabilize near 1 while pred_loss falls (more slowly — it must now work
  for its living).

## Stage 3 — probing what the latent knows (wm/probe.py)

All probes read a frozen encoder; nothing feeds back (the paper's
"decoder as window, not steering wheel").
- Linear probe: ridge/least-squares from z to true (x, y). Report R^2 on
  held-out episodes. Collapsed model: R^2 ~ 0. Regularized: expect > 0.95.
- Decoder probe (optional but recommended): tiny deconv trained to redraw
  the image from z, encoder DETACHED. Save a 6-image grid (real vs redrawn)
  into the run folder. Seeing the dot re-appear from 16 numbers is the
  week's reward.
- Straightness metric (from the LeWM read): mean cosine similarity between
  consecutive latent velocity vectors, logged per epoch. Just watch it —
  does it drift up during training here too?

## Stage 4 — planning with CEM (wm/plan.py)

- Task: given current obs and a goal obs (a state ~10-15 steps away,
  sampled from a held-out trajectory), act for a budget of 20 steps.
- CEM: horizon H=8, N=64 sampled action sequences per iteration, keep
  K=8 elites, 4 iterations, refit a diagonal Gaussian to elites; execute
  the first action, then replan (MPC). Cost = ||z_H_hat - z_goal||^2.
- Success = final TRUE distance to goal position < 2.0 (true state used
  for scoring only). Baselines to run under the identical protocol:
  (a) random actions, (b) greedy oracle on true state (upper bound).
- 3 seeds x 50 episodes each, all through lablog; analyzer reports
  success-rate mean +/- std per policy.
- PREDICTIONS FIRST: random ~5-15%, CEM-on-regularized-model well above,
  CEM-on-collapsed-model ~= random (the punchline that ties the week).

## Stage 5 — the teleport test (wm/surprise.py)

Violation-of-expectation, LeWM Sec 5.2 in miniature: roll a normal episode,
but at step 15 teleport the dot to a random cell. Feed the frames through
encoder+predictor and plot per-step prediction error. A healthy model shows
a spike exactly at the teleport. One figure into the run folder. This is
also your Phase 1 evaluation-literacy warm-up.

## Compute notes
Stage 1-3 are laptop-feasible (small CNN, 32x32; expect minutes-to-an-hour
scale — do the steps-x-cost arithmetic and write the estimate down first).
Stage 4's planning loop is cheap. If the laptop drags, this is the designed
moment for the first cloud run: push repo to GitHub, open Colab/Kaggle,
clone, pip install -r requirements.txt, run collect + train, download the
runs/ folder back, analyze LOCALLY. The scaffold makes this a ritual, not a
project.

## Failure signatures
- z_std ~1 but probe R^2 low: regularizer too strong (lam) — spread without
  meaning; halve lam.
- pred_loss identical whether actions are shuffled or not: predictor is
  ignoring actions (check the concat; check actions aren't constant in the
  dataset).
- CEM no better than random despite good probes: horizon too long for the
  predictor's error accumulation — drop H to 4; or cost using un-normalized
  latents from different checkpoints.
- Everything works too easily: good — turn the dial (add the block) and
  watch which part breaks first. That observation IS Phase 2 groundwork.

## Paper pairing
This is the week to re-read LeWM properly (it is the active read), plus
Appendix A (SIGReg) and B (CEM) now that you've built both cousins. Log a
per-claim note: what experiment verifies it, at what compute.
