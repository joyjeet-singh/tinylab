# Phase 1 (Aug–Sept) — reproduce LeWM, TwoRoom first

Vehicle candidate: MLRC 2026 with LeWM as target (confirm registration
details and deadlines online at Week 4; also re-check the NeurIPS 2026
competition list — a fitting competition REPLACES MLRC, never adds).
Check for a v3 of the paper before locking anything (v2 has a duplicated
paragraph in Sec 4.2 — it is a live preprint).

## Order of attack and why
1. **TwoRoom first.** Smallest dataset (10k episodes, ~92 steps), simplest
   predictor setting (history length 1), and it is the environment where
   LeWM LOSES (87% vs 97–100 for PLDM/DINO-WM/GC baselines). Reproducing
   the anomaly is worth more than reproducing the victory: it is the
   doorway to Phase 2, and anomalies are where reproductions earn trust.
2. **PushT second.** The headline: 96 ± 2.83% success, 3 training seeds,
   50 eval trajectories (their Table 5). Bigger dataset (20k episodes).

## Reproduce-the-ruler rule
Match their evaluation protocol EXACTLY before comparing numbers:
TwoRoom — eval budget 150 steps, goal sampled 100 steps ahead;
PushT — budget 50, goal 25 ahead; CEM 300 samples / 30 elites /
30 iters (PushT) or 10 iters (others), horizon 5 with frame-skip 5,
MPC executing the full plan before replanning. A reproduction with a
different ruler proves nothing.

## Compute arithmetic (do this before spending a single GPU-hour)
Paper's setup: 224x224 frames, ViT-Tiny encoder + ViT-S predictor (~15M
params), batch 128, 10 epochs, single L40S, "a few hours" per environment.
Free-tier GPUs (Kaggle P100/T4) are roughly 3–5x slower than an L40S for
this workload. Estimate: TwoRoom training ≈ 8–20 GPU-hours + evaluation
(50 planning episodes) ≈ 1–3 hours. That fits inside one Kaggle week
(~30 h) with margin — but only barely inside one SESSION limit, so plan
checkpoint-and-resume from the start (their framework supports it; verify
early with a 10-minute run).

Contingency if it does not fit: 112x112 resolution halves token count —
but that is a DEVIATION from the paper; if used, label every resulting
number as such, and report both when possible.

## Execution ladder (each rung is a committed checkpoint)
1. Clone their stable-worldmodel framework; run its smallest example
   end-to-end on Colab CPU/GPU to prove the plumbing.
2. Generate the TwoRoom dataset with their heuristic policy; verify
   episode count and an image looks right.
3. Train 1 epoch only; confirm loss curves resemble Fig. 18's early shape
   (pred loss falling, SIGReg dropping sharply then plateauing).
4. Full 10-epoch train, 1 seed; run their eval; compare to 87%.
5. If within a few points: 2 more seeds. If not: bisect (data? train?
   eval protocol?) — the Week 3 build gives you the instincts for which.
6. Write the reproduction note as you go, not after: per-claim, what you
   ran, what it cost, what matched.

Budget guardrail: keep the paid tier (Vast.ai ~$10) untouched until a
specific run is blocked on Kaggle/Colab limits, then price that one run.

# Phase 2 (Oct–Dec) — the research question, v0 (provisional)

Status: DRAFT. Revisit after Week 3 with hands-on collapse intuition;
re-derive rather than obey. One question only; the backup exists so the
primary can be dropped without drama, not so both get worked.

## Primary: the Two-Room anomaly
Paper's own guess (their Sec 4.2 + limitations): SIGReg forces latents
toward a full isotropic Gaussian in d≈192 dims, while TwoRoom's true
state is ~2-dimensional; matching that prior may SCRAMBLE a
low-intrinsic-dimension world. They state this as a possible explanation
and never test it. Gift: falsifiable, pre-endorsed by the authors,
cheapest environment in the paper, and structurally the "structured vs
generic prior" playbook already run once in the k-hop project.

Hypothesis H: LeWM's planning deficit grows as (latent dim d) /
(environment intrinsic dim k) grows, holding all else fixed; matching the
regularization target's active dimensionality to k removes the deficit.

Apparatus: dotworld's dimensionality dial — k=2 (one dot), k≈4–5 (dot +
pushable block, + block angle), k≈7+ (two blocks). Same pixels, same
model, same protocol; only k moves.

Interventions, in increasing strength:
(a) sweep d in {4, 16, 64, 192} at each k;
(b) sweep lam (their only hyperparameter) — is the anomaly just a
    miscalibrated lam at low k?
(c) anisotropic target: regularize only m < d dimensions toward N(0,1),
    leave the rest unconstrained; sweep m across k;
(d) report effective rank of the latent cloud + probe R^2 + planning
    success as the three-way outcome measure.

Falsification: if the deficit persists when d≈k or under (c) with m≈k —
H is wrong, and the anomaly lives elsewhere (data diversity, predictor
capacity, or the planner). That result is still publishable as a careful
negative; the PHIL-DEQ experience applies.

Deliverable shape: workshop-length paper or strong blog + code. Compute
class: entirely 2D toy worlds — Colab-sized by design.

## Backup: the rotation blind spot
Every method in their Table 4 fails to encode block orientation (yaw,
quaternion), and rollouts visibly forget end-effector angle. Physics
instinct: angles live on a circle; everything in the pipeline assumes
flat Euclidean, Gaussian-shaped statistics. Question: does giving a small
slice of the latent circular structure (e.g., regularize a 2-dim block
toward the unit circle / wrapped distribution) recover rotation probes
and rotation-dependent planning without hurting the rest? Needs a
rotating-shape dotworld variant (still 2D-cheap). Park unless the primary
dies.

## Explicitly shelved (do not drift back)
Hierarchical long-horizon planning (a research program, not a question);
video pretraining (a compute program); temporal-straightening-for-planning
(already claimed by the authors' circle — arXiv 2603.12231; the metric
stays useful as a diagnostic only).
