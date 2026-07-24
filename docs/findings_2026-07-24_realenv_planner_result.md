# Findings, July 24 2026 (III) — the real-environment planner result

The planner plans. Same checkpoint, same CEM settings, same episode budget as
every earlier evaluation — but measured, for the first time, in the world the
model was trained on:

    CEM     : 36/50 = 72.0%   (non-trivial: 34/48 = 70.8%)   mean final dist 26.5
    random  :  9/50 = 18.0%   (non-trivial:  7/48 = 14.6%)   mean final dist 47.0

Paired on the 50 shared episodes: CEM succeeded alone on 32, random alone on
5, both on 4, neither on 9. Exact McNemar two-sided p = 7.4e-6. After three
runs of "worse than random" in the toy fixture, the identical checkpoint is
decisively above random in the real environment. Reference LeWM reports 87%
with their fully-trained weights; our 72% comes from an epoch-5 checkpoint
produced by an accidentally cyclic schedule, under the deviations below.

## How the harness earned the right to produce this number

The R-series, all free on the Mac, each gated before the next:

- R0 (recon + domain guard): env renders landed median 5.81 from the nearest
  training latent (cloud spacing 2.57) at default variations — 10x closer
  than the toy fixture's 61.03, with the residual suspected to be position
  coverage, not rendering.
- R1 (reconstruction + replay): pixel MAE 0.00 (median AND max) placing the
  agent at recorded positions; one-step and 40-step action replay error
  0.000; matched-position paired latent distance 0.01. The installed
  swm/TwoRoom-v1 IS the dataset's generator — renderer and dynamics,
  bit-level. R0's residual confirmed as coverage.
- R2 (this result): the domain guard runs as a precondition inside the eval
  (paired median 0.013, threshold 1.0 — it refuses to score otherwise), then
  50 CEM episodes and 50 random-control episodes under the reference
  protocol.

One table justifies the whole gate: the same guard instrument gave the toy
fixture 61.03 and the real environment 0.01.

## Protocol (reference: le-wm config/eval/tworoom.yaml + solver/cem.yaml)

Starts = recorded episode states (frame 0); goal = the recorded state 25 raw
steps later on the same trajectory; goal image = the environment rendered
with the agent at the goal (the package's own convention); success = the
registered environment's termination rule (distance to target < 16); budget
50 env steps; CEM 300 samples / 30 rounds / top-30 / var 1.0, horizon 5,
action_block 5; 5 planned actions executed per replan.

Deviations, stated wherever these numbers are quoted: episode selection ours
(50 random episodes, seed 42, start = frame 0, goal = frame 25 — the
reference's exact trajectory selection is not documented in the config);
receding_horizon 5 read as "execute 5 planned actions, then replan"; OUR
reproduction checkpoint, not the authors' released weights. The recorded
config conflict also applies: goal_offset 25 is the repo's eval value; the
paper implies 100.

## The residual failure mode

CEM's 14 misses are not diffuse: 10 of them ended FARTHER from the goal than
they started — confident travel in a wrong direction, while random at least
diffuses (its misses average nearer). Combined with the in-domain wall test
(the encoder separates the rooms by only 1.79x in latent distance), the
leading suspect is that straight-line latent scoring steers correctly most
of the time but not reliably through the door — which would be the honest,
quantified form of the published scoring critique, now measurable:
`analyze_realenv_by_wall.py` breaks both arms down by same-room vs
cross-wall geometry and draws every episode's start->goal arrow colored by
outcome.

## What this changes

The paper's arc is now complete and true end to end: the released config
as-specified does not converge (real-data evidence, run 0); convergence is
reachable and learning-rate-governed (runs 1-2, dose-response); the toy
evaluation harness manufactured a below-random phantom for three runs and
survived a 0.9916 probe (measured anatomy); and the validated checkpoint,
evaluated in the true environment under the reference protocol, plans at
72% vs 18% random (p = 7e-6), approaching but not matching the published
87%. Gap-attribution candidates, in testable order: checkpoint quality (ours
held its minimum for ~2 epochs by accident — the seed-1 fixed-schedule run
addresses exactly this), episode selection, and the receding-horizon
interpretation. The pre-registered primary as instrumented remains NULL and
is reported as such, with this section explaining what the instrument
actually measured.
