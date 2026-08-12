# Follow-up experiments — CPU only, no GPU training

Everything here post-dates arXiv:2608.10145 and is **separate from the paper's
evidence base**. Outputs are named `followup_plan_*` rather than
`realenv_plan_*` so that `extract_all_results.py` — which globs
`**/realenv_plan_*report*.txt` from the repository root — cannot see them and
`docs/paper/results_from_disk.csv` cannot silently change. The published
paper's numbers are untouched.

Goal: find capability improvements reachable without retraining.

---

## 1. Imagination is not the long-horizon bottleneck

`followup/rollout_h15_phase2_recal.txt` — the released flagship
(`ckpt_best_recal.pt`, phase2) rolled out autoregressively on real validation
clips under the recorded actions:

| horizon (planner steps) | env steps | err/static |
|---|---|---|
| 1 | 5 | 0.066 |
| 5 | 25 | 0.090 |
| 10 | 50 | 0.152 |
| 15 | 75 | 0.189 |

At 15 steps out the imagined state is still only 19% as wrong as assuming the
world froze. Horizon 20 could not be measured at all: the clip would need 102
raw frames and episodes cap at 101 — the same ceiling that degenerates the
offset-100 protocol in §4.2 of the paper.

**The predictor can see 75 environment steps ahead. CEM only ever asks it for
25** (`horizon=5` planner actions x `frameskip=5`), toward a goal 100 steps
away. That is a planner limitation, not a model limitation.

## 2. The planning objective saturates, then inverts

CEM minimises squared L2 between the imagined final embedding and the goal
embedding. `followup/latent_metric_phase2_recal.txt` measures whether that
quantity actually tracks distance, with no planning involved — render real
recorded positions, encode, compare pairwise latent distance to pairwise true
distance:

| true distance | mean latent dist | r within band |
|---|---|---|
| 0–20 | 15.92 | 0.668 |
| 20–40 | 18.52 | 0.182 |
| 40–60 | 19.68 | 0.115 |
| 60–80 | 20.04 | 0.104 |
| 80–100 | 20.61 | 0.122 |
| 100–120 | 20.68 | −0.029 |
| 120–150 | 20.60 | falling |
| 150–300 | 19.93 | falling |

Overall Pearson r = 0.426. Across 80–300 units the objective varies by −3.3%,
and past ~120 units it **decreases** with true distance: moving farther from
the goal can lower the planner's cost.

This accounts for what the paper reports without explaining: 94.0% at offset
25 (short range, where r = 0.668), collapse to 26.0% at offset 100, and the
overshoot — 26 of 37 offset-100 failures end farther from the goal than they
began, at a median 1.41x the original separation. Binned by difficulty,
baseline success is 100% under 60 units and 6% in the 100–140 band.

It is also, measured on our own checkpoint, the failure mode of the Li et al.
(2026) critique the paper cites.

## 3. A mechanism for the accuracy/planning dissociation

§5.3 reports that one-step accuracy does not order long-horizon planning, and
§1 lists "any mechanism for the overshoot" as **not claimed**. Measuring the
same metric geometry on all three released checkpoints supplies one:

| checkpoint | one-step err | offset 100, budget 150 | Spearman | monotone | 80–300 spread |
|---|---|---|---|---|---|
| Run 2 recal | 0.829 | **40/50 = 80.0%** | 0.872 | yes | +38.6% |
| Run 0 recal | — | **33/50 = 66.0%** | 0.846 | yes | +34.9% |
| phase2 recal | 0.116 | **13/50 = 26.0%** | 0.445 | **no** | −3.3% |

Long-horizon planning success rank-orders **exactly** with latent-metric
quality, and inversely with prediction accuracy. The most accurate predictor
has the least usable planning geometry. That is a candidate mechanism for the
paper's central dissociation, and it is testable rather than speculative.

Caveat: three checkpoints, one seed each. This is a correspondence across the
three models we have, not an established law.

## 4. The information is present; only the metric fails

`followup/probe_metric_phase2_recal.txt` — a ridge probe fit on frozen
embeddings of 350 rendered positions, evaluated on 150 held out:

- probe position R² **0.9922** held out, mean absolute error **1.72** arena units
- Pearson r vs true distance — latent L2 **0.437**, decoded position **0.9897**
- monotone in true distance — latent L2 **no**, decoded position **yes**
- spread across 80–300 units — latent L2 **+1.2%**, decoded position **+73.4%**

The encoder already represents position almost perfectly. Nothing about the
world model needs to change; the objective does.

## 5. Intervention under test

`--cost probe` re-points CEM at the distance between **decoded positions**
rather than embeddings. The probe is fit at startup on rendered positions
(R² 0.9993 in-sample, 0.72 units mean error); the encoder and predictor are
frozen and unmodified.

Running at `--num-eval 50 --seed 42 --budget 150 --goal-offset 100`, which
reproduces `exp_ref_p2`'s exact 50-episode draw (verified), so the result is
directly paired against the published **13/50 = 26.0%**.

Early two-episode signal at a 60-step budget — a third of the baseline's —
reached a 150.7-unit goal in 25 steps.

## What was tried and set aside

- **Planning horizon 15** (`--plan-horizon 15`): motivated by §1, stopped
  after ~35 minutes without completing an episode. A longer horizon lets the
  planner reach farther under an objective that is blind past 80 units, so §2
  predicts it cannot help much on its own. Worth rerunning against the probe
  cost, where the objective is sound.
- **Oracle subgoal decomposition** (`--subgoals 3`): 2/3 in a smoke test at
  offset 100. Paused at 3/50 to give the probe-cost run the machine. It routes
  around the broken metric by keeping every hop inside the short range where
  it works — but subgoals come from the recorded trajectory, which is
  privileged information a deployed planner would not have. The probe cost
  needs no such information, which is why it took priority.
