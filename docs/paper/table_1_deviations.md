# Table 1 — Fidelity of the reimplementation

Two parts. **1a** compares our reimplementation against the reference
implementation element by element. **1b** records what differed between our four
training runs, since they are not identical to each other.

Every reference value in 1a is cited to **source**, not to a configuration file.
That distinction is the subject of §3.2: all four of the deviations that
determined our results live in code that no configuration file mentions.

Source keys: **B** = `stable_worldmodel/data/buffer.py`; **T** =
`le-wm/train.py`; **U** = `le-wm/utils.py`; **Y** =
`le-wm/config/train/lewm.yaml`; **C** = the released `config.json` accompanying
the published checkpoint.

---

## Table 1a — Element-by-element fidelity

> **Table 1: Fidelity of our reimplementation against the reference.** Reference
> values are cited to implementation source rather than to configuration files;
> the four rows marked **undocumented** are determined in code and appear in no
> released configuration. "Corrected" indicates a deviation present in our
> earlier runs and fixed in the final configuration (§3.2).

### Data pipeline

| element | reference | source | ours | status |
|---|---|---|---|---|
| clip frame indices | `base + arange(history_len) · frameskip` | B `_gather_clip` | identical | match |
| **action gathering** | dense: `base + arange(history_len · frameskip)`, reshaped to (T, frameskip × action_dim) | B `_gather_clip` | sub-sampled one action per clip step | **undocumented — corrected** |
| **action-encoder width** | set programmatically to frameskip × action_dim = 10 | T:68, C | 2 | **undocumented — corrected** |
| **pixel preprocessing** | ImageNet mean/std normalisation, then resize | U:6 | divide by 255 only | **undocumented — corrected** |
| **action normalisation** | per-dimension z-score from dataset statistics, NaN rows dropped | T:65, U:25–32 | raw actions | **undocumented — corrected** |
| frameskip | 5 | Y | 5 | match |
| train/validation split | 0.9 / 0.1 | T:74 | 0.9 / 0.1, separate data seed | match (mechanism differs) |

### Architecture

| element | reference | source | ours | status |
|---|---|---|---|---|
| encoder | ViT-Tiny, patch 14, 224 px | Y, C | identical | match |
| encoder depth / heads | 12 / 3 | C | 12 / 3 | match |
| embedding dimension | 192 | Y, C | 192 | match |
| predictor depth / heads | 6 / 16 | C | 6 / 16 | match |
| predictor head dim / MLP | 64 / 2048 | C | 64 / 2048 | match |
| projector hidden width | 2048, BatchNorm1d | C | 2048, BatchNorm1d | match |
| dropout | 0.1 | C | 0.1 | match |
| context frames (`history_size`) | 3 | Y, C | 3 (Runs 0 and 4); **1** (Runs 1–2) | deviation, Runs 1–2 |
| parameter count | 18,034,590 (rebuilt from C) | C | 18,034,670 | +80 = action encoder width |

### Objective

| element | reference | source | ours | status |
|---|---|---|---|---|
| prediction loss | MSE, target **not** detached | T:39 | identical | match |
| total loss | prediction + λ · regulariser | T:41 | identical | match |
| regulariser weight λ | 0.09 | Y | 0.09 (Runs 0, 4); **0.045** (Runs 1–2) | deviation, Runs 1–2 |
| regulariser knots / projections | 17 / 1024 | Y | 17 / 1024 | match |
| regulariser axis | per timestep, across batch | T | identical | match |

### Optimisation

| element | reference | source | ours | status |
|---|---|---|---|---|
| optimiser | AdamW | Y | AdamW | match |
| learning rate | 5 × 10⁻⁵ | Y | 5 × 10⁻⁵ (Runs 0, 4); **1 × 10⁻⁵** (Runs 1–2) | deviation, Runs 1–2 |
| weight decay | 1 × 10⁻³ | Y | 1 × 10⁻³ | match |
| batch size | 128 | Y | 128 | match |
| gradient clipping | 1.0 | Y | 1.0 | match |
| prediction steps | 1 | Y | 1 | match |
| learning-rate schedule | **none specified** | Y | none (Runs 0, 1, 4); **cosine** (Run 2) | deviation, Run 2 |
| epochs | 100 | Y | 10 | deviation — paper's appendix states 10 |
| precision | bfloat16 | Y | float32 | deviation — benign |
| seed | 3072 | Y | 0 | deviation — benign |

### Evaluation

| element | reference | source | ours | status |
|---|---|---|---|---|
| environment | `swm/TwoRoom-v1` | package | identical, verified bit-level (§3.3) | match |
| success criterion | registered env rule, distance < 16 | package | identical | match |
| CEM settings | 300 samples / 30 elite / 30 iterations / variance 1.0 | Y | identical | match |
| horizon, action block | 5, 5 | Y | 5, 5 | match |
| step budget | 50 | Y | 50 | match |
| episodes evaluated | 50 | Y | 50 (220 for §5.2) | match |
| goal offset | **25 in the evaluation config; the paper implies 100** | Y | both reported (§4.5, §5.3) | **conflict — unresolved** |
| episode selection | not published | — | fixed random draw, seed 42; start at episode frame 0 | deviation — unavoidable |
| receding horizon | 5 | Y | read as: execute 5 planned actions, then replan | interpretation, tested (§5.3) |

### Environment

| element | ours | note |
|---|---|---|
| PyTorch | 2.2.2+cu121 | identical across all four runs |
| Python | 3.11.15 (Runs 0–2); **3.12.3** (Run 4) | deviation — benign, recorded |

---

## Table 1b — Configuration of the four training runs

> **Table 1b: Our four training runs.** Runs 1 and 2 were exploratory and vary
> three and four hyperparameters from the reference respectively; their results
> are reported as ablations. Run 0 and Run 4 differ **only** in the four
> pipeline corrections, and form the controlled pair of §4.4.

| | Run 0 | Run 1 | Run 2 | Run 4 (`phase2`) |
|---|---|---|---|---|
| purpose | reference config, as reimplemented | exploratory bundle | Run 1 + schedule | corrected pipeline |
| learning rate | 5e-5 | 1e-5 | 1e-5 | 5e-5 |
| schedule | none | none | cosine¹ | none |
| regulariser weight | 0.09 | 0.045 | 0.045 | 0.09 |
| context frames | 3 | 1 | 1 | 3 |
| action width | 2 | 2 | 2 | **10** |
| dense actions | no | no | no | **yes** |
| ImageNet pixels | no | no | no | **yes** |
| z-scored actions | no | no | no | **yes** |
| epochs | 10 | 10 | 10 | 10 |
| deviations from reference | 4 | 7 | 8 | **3** |
| used for | §4.4, §4.1 | ablation | §4.5, §5.2, §5.3 planning | §4.4, §4.5, §5.2, §5.3 |

¹ A patch applied twice caused the scheduler to step twice per epoch, turning
the intended one-way cosine decay into a full cycle from 1 × 10⁻⁵ down to
1 × 10⁻⁷ and back. The incident is disclosed in §3.4; the resulting learning-rate
sweep is what identified the normalisation artifact of §4.3, and we report it
as an accident rather than a design.

---

## Notes for assembly

- **Run numbering.** The runs are chronologically 0, 1, 2 and then the corrected
  one. Calling the last "Run 4" when there is no Run 3 will confuse a reader;
  either renumber to 0–3 throughout, or name them by role ("released-config
  run", "corrected-pipeline run") and drop numbers entirely. The role-based
  naming reads better in §4.4 and §5.3, where the contrast matters more than the
  chronology.
- **The four undocumented rows are the paper's contribution and should be
  visually distinct** — bold, shaded, or moved to their own sub-table directly
  after 1a. A reader skimming Table 1 should see them without reading it.
- Deviation counts in 1b: Run 0 = action width, epochs, precision, seed. Run 4 =
  epochs, precision, seed. Verify these against the audit before submission; the
  count is the kind of number that drifts during editing.
- Source line numbers (T:68, U:6, U:25–32, T:39, T:65, T:74) should be checked
  against the reference at the commit you audited and that commit cited in the
  caption.
