# tinylab — LeWorldModel reproduced on TwoRoom

Six checkpoints from an independent reimplementation of **LeWorldModel** (Maes
et al., 2026, arXiv:2603.19312) on the TwoRoom environment.

**These are not the authors' weights.** They were trained from scratch, from
our own code, to test whether the published result reproduces. The authors'
released weights are at
[`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms).

## What is here

| file | what it is | size | `batches_tracked` | md5 |
|---|---|---|---|---|
| `tinylab-tworoom-phase2-recal.pt` | recalibrated, epoch 9, step 48771 | 69 MiB | [208] | `f1127023d44d…` |
| `tinylab-tworoom-phase2.pt` | as trained, epoch 9, step 48771 | 69 MiB | [48771] | `eb2c9c1bac83…` |
| `tinylab-tworoom-run0-recal.pt` | recalibrated, epoch 10, step 54190 | 69 MiB | [108] | `ad46d13559c3…` |
| `tinylab-tworoom-run0.pt` | as trained, epoch 10, step 54190 | 69 MiB | [54190] | `1a5fc805a713…` |
| `tinylab-tworoom-run2-recal.pt` | recalibrated, epoch 6, step 36732 | 69 MiB | [208] | `953aeb6ac1fb…` |
| `tinylab-tworoom-run2.pt` | as trained, epoch 6, step 36732 | 69 MiB | [36732] | `985b1d779a66…` |

Optimiser and RNG state are stripped. Every weight tensor was verified bitwise
identical to its source after the round trip; the full manifest, with md5s
before and after stripping, is `runs_archive/verified/ckpt_md5.txt`.

Three runs, each in two versions:

- **run0** — the released configuration followed as closely as we could read it.
- **run2** — an exploratory run (cosine schedule, different λ and learning rate).
- **phase2** — the corrected pipeline, after the four conventions below. This is
  the checkpoint the paper's headline planning number comes from.

`-recal` files are BatchNorm-recalibrated (below). **The paper's planning
numbers are measured on the recalibrated files.** The un-recalibrated originals
are included so that the evaluation-mode artifact can be checked independently,
not because they should be used for planning.

## Four conventions the released material does not state

A model trained from the released configuration files alone does not converge
its predictor. Four conventions are visible only in the reference *source*:

| convention | where it is visible |
|---|---|
| actions gathered **densely**, reshaped to `(T, frameskip × action_dim)` | `stable_worldmodel/data/buffer.py`, `_gather_clip` |
| action-encoder input width **10**, not the configured 2 | `le-wm/train.py:68`, set programmatically |
| pixels normalised with **ImageNet** mean/std, not scaled to [0,1] | `le-wm/utils.py:6` |
| actions **z-scored** per dimension | `le-wm/train.py:65`, `utils.py:25` |

Line numbers refer to `le-wm` commit `8edfeb336732…`; see
`docs/lewm_audit_commit.txt`.

## BatchNorm recalibration — read this before quoting a loss

The projector specified by the released configuration ends in a BatchNorm
layer. Its running statistics, accumulated as a training exponential moving
average, do not describe the distribution the model is evaluated on. In
evaluation mode the reported prediction loss is inflated — by up to roughly a
factor of 300 relative to the same checkpoint's training-mode loss — while the
training loss is flat. It contaminates four separate quantities: prediction
loss, the SIGReg term, effective rank, and planning outcomes.

The repair is a precise-BN pass: recompute the statistics over training clips,
updating no weight. Held-out prediction loss in evaluation mode, before and
after:

| checkpoint | eval mode, before | eval mode, after | train mode, same scoring run |
|---|---|---|---|
| Run 0 | 1.4585 | **0.3064** | 0.3077 |
| Run 2 | 0.1846 | **0.1811** | 0.1812 |
| phase2 | 4.6034 | **0.0086** | 0.0151 |

`batches_tracked` in the table above tells the two apart: a training average
carries tens of thousands, a precise-BN pass carries hundreds. Read it from the
file rather than trusting any range quoted elsewhere, including here.

## Planning results, with the protocol attached

**A success rate without its protocol is meaningless here.** The released
material publishes two evaluation protocols that disagree, and on the authors'
own weights they give 84.0% and 14.0%. Goal offset 25 with a
50-step budget is the released repository's evaluation configuration; offset
100 with a 150-step budget is what Appendix F.1 describes.

Every figure below is a file in the repository.

**`tinylab-tworoom-phase2-recal.pt`** — the corrected pipeline

| goal offset | budget | goals reached | guard | report |
|---|---|---|---|---|
| 25 | 50 | 47/50 = **94.0%** | 0.009 | `exp_phase2_recal_25/realenv_plan_cem_report.txt` |
| 100 | 50 | 10/50 = **20.0%** | 0.009 | `exp_phase2_recal/realenv_plan_cem_report.txt` |
| 100 | 150 | 13/50 = **26.0%** | 0.009 | `exp_ref_p2/realenv_plan_cem_report.txt` |

**`tinylab-tworoom-run2-recal.pt`** — exploratory run

| goal offset | budget | goals reached | guard | report |
|---|---|---|---|---|
| 25 | 50 | 39/50 = **78.0%** | 0.014 | `exp_run2_recal_25/realenv_plan_cem_report.txt` |
| 100 | 50 | 27/50 = **54.0%** | 0.014 | `exp_run2_recal/realenv_plan_cem_report.txt` |
| 100 | 150 | 40/50 = **80.0%** | 0.014 | `exp_ref_r2/realenv_plan_cem_report.txt` |

**`tinylab-tworoom-run0-recal.pt`** — released configuration

| goal offset | budget | goals reached | guard | report |
|---|---|---|---|---|
| 25 | 50 | 31/50 = **62.0%** | 0.004 | `exp_r0_short/realenv_plan_cem_report.txt` |
| 100 | 150 | 33/50 = **66.0%** | 0.004 | `exp_ref_r0/realenv_plan_cem_report.txt` |

For reference, the authors' released checkpoint measured under the same harness
on the same episodes reaches 42/50 = 84.0% at goal
offset 25 (`exp_authors/realenv_plan_authors_cem_report.txt`).

## The domain guard

The evaluation refuses to report a success rate when the frames it is scoring
sit outside the distribution the encoder was trained on. It measures the paired
distance between the embedding of a rendered frame and of the same state
reached in the environment, and compares the median against a threshold of 1.0.
Across every run reported here it lies between 0.004 and 0.014,
against a nearest-neighbour spacing within the real data of 2.43 — so the
margin is roughly two orders of magnitude.

The guard value also fingerprints which checkpoint produced a report — the
`guard` column above. **Anyone evaluating these weights in their own renderer
needs this check.** A silently out-of-distribution renderer produces numbers
that look ordinary and mean nothing: an earlier phase of this work evaluated in
a 32-pixel fixture where the same instrument reads 61.03, and three runs' worth
of planning results produced there are worthless.

## Representation

Ridge probes on 4,000 held-out frames, both checkpoints recalibrated
(`runs_archive/verified/encoder_probe_both_recal.txt`):

| | run0 | phase2 |
|---|---|---|
| position, linear probe | 0.9977 | 0.9971 |
| summed action from (z_t, z_t+1) | 0.9290 | 0.9132 |

## Architecture and training

ViT-Tiny encoder, patch 14, 224px, embedding
dimension 192, history 3 frames; predictor depth
6, 16 heads, MLP dimension 2048;
18,034,206 parameters. Trained 10 epochs, batch
128, learning rate 5e-05, weight decay
0.001, SIGReg weight 0.09, seed
0, fp32. Full configuration:
`configs/phase2_dense_reference.yaml`.

## Dataset

Trained on the authors' TwoRoom dataset: 10,000 episodes, mean length
92.1 frames. **We do not redistribute it.** Get it from the authors at
[`quentinll/lewm-tworooms`](https://huggingface.co/datasets/quentinll/lewm-tworooms).

## Limitations

- **One seed.** Every number here is a single run. We make no estimate of seed
  variance.
- **10 epochs, against the repository configuration's 100.** This is
  the paper's appendix value, chosen because the full budget was beyond ours.
  Nothing here is a claim about the asymptote.
- **TwoRoom only.** No claim is made about the original's other environments,
  its embodied or zero-shot results, or scales other than this one.
- **Long-horizon planning is poor, and one-step accuracy does not predict it.**
  The most accurate predictor here is not the best long-horizon planner. Do not
  select a checkpoint on prediction loss.
- The checkpoint-versus-checkpoint differences we report are not established at
  n = 50; see the paper.

## Licence

MIT, matching the reference implementation
([`le-wm`](https://github.com/lucas-maes/le-wm), MIT, © 2026 Lucas Maes).

This is a reproduction. The original work is the authors'.

## Citation

```bibtex
@article{singh2026tinylab,
  title  = {The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWorldModel on TwoRoom},
  author = {Singh, Joyjeet},
  year   = {2026},
  eprint = {<ARXIV_ID>},
  note   = {Independent reproduction of arXiv:2603.19312}
}
```

Code, every evaluation report, the fidelity audit and the pre-registration:
[github.com/joyjeet-singh/tinylab](https://github.com/joyjeet-singh/tinylab).
