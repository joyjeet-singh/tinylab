# Fidelity audit — our reimplementation against the LeWM reference

Every row cites reference **source**, not config, because the two bugs that cost
us most (action subsampling, and the ones below) live in code that no YAML
mentions. A row that cannot be filled from source is an unread file, and that
is itself the finding.

Reference sources: `lucas-maes/le-wm` (train.py, utils.py, jepa.py, module.py)
and the installed `stable_worldmodel` package.

Status: **MATCH** = verified identical · **DEVIATION** = verified different ·
**UNVERIFIED** = not yet traced to source, do not assume.

---

## Data pipeline

| # | element | reference (source) | ours | status |
|---|---|---|---|---|
| 1 | clip frame indices | `data/buffer.py` `_gather_clip`: `base + arange(history_len) * frameskip` | `tworoom_data.py` `clip_indices`: `start + arange(num_steps) * frameskip` | **MATCH** |
| 2 | action gathering | same fn: `base + arange(history_len * frameskip)` — **dense**, then `reshape(history_len, -1)` → `(T, frameskip × action_dim)` | was one subsampled action per step, `(T, 2)`; **fixed 2026-07-26** to dense `(T, 10)` | **DEVIATION — FIXED** |
| 3 | action encoder width | `train.py:68` sets it programmatically: `cfg.model.action_encoder.input_dim = frameskip * dataset.get_dim("action")` = 10 | `action_dim: 2` in config → `ActionEmbedder(2, 10, embed_dim)` | **DEVIATION — fix pending** (set `action_dim: 10`) |
| 4 | **pixel preprocessing** | `utils.py:6` `get_img_preprocessor`: `ToImage(**dataset_stats.ImageNet)` then `Resize(img_size)` — **ImageNet mean/std normalisation** | `tworoom_data.py:146` `px.astype(float32) / 255.0` — **[0,1] only** | **DEVIATION — OPEN** |
| 5 | **action normalisation** | `train.py:65` applies `get_column_normalizer` to every non-pixel column; `utils.py:25-32` computes per-dimension **z-score** (dataset mean/std) | raw actions, no normalisation | **DEVIATION — OPEN** |
| 6 | train/val split | `train.py:74` `spt.data.random_split(lengths=[0.9, 0.1], generator=seed)` | to check | **UNVERIFIED** |

Rows 4 and 5 corroborate the empirical driving-spec result: their released
checkpoint scored 0.410 with ImageNet-normalised pixels and 5.1 with raw
`[0,1]` — a fivefold difference that now has a source citation behind it.

## Model

| # | element | reference | ours | status |
|---|---|---|---|---|
| 7 | encoder | `vit_hf(size, patch_size, image_size, pretrained=False, use_mask_token=False)`; released config: tiny / 14 / 224 | ViT-Tiny, patch 14, 224 | **MATCH** (verified: identical state-dict keys, 5.501M params) |
| 8 | embed_dim | `config/train/lewm.yaml`: 192 | to confirm against our manifest | **UNVERIFIED** |
| 9 | history_size | `lewm.yaml`: **3**; released checkpoint `predictor.pos_embedding` is `(1,3,192)` → **3** | **1** (the paper's TwoRoom value) | **DEVIATION — deliberate, but against our own standing rule** (repo is the reference; the paper is the description) |
| 10 | predictor | released config: `num_frames 3, dims 192/192/192, depth 6, heads 16, mlp_dim 2048, dim_head 64, dropout 0.1` | to compare row by row against our manifest | **UNVERIFIED** |
| 11 | projector / pred_proj | released config: `MLP 192→192, hidden 2048, norm_fn BatchNorm1d` | `MLP(embed_dim, proj_hidden, embed_dim)` — hidden width to check | **UNVERIFIED** |

## Objective and optimisation

| # | element | reference | ours | status |
|---|---|---|---|---|
| 12 | prediction loss | `train.py:39` `(pred_emb - tgt_emb).pow(2).mean()` — plain MSE | to confirm | **UNVERIFIED** |
| 13 | total loss | `train.py:41` `pred_loss + lambda * sigreg_loss` | same form | **UNVERIFIED** |
| 14 | SIGReg weight | `lewm.yaml`: **0.09**, `knots 17`, `num_proj 1024` | 0.09 used in run 1-2; knots/num_proj to check | **PARTIAL** |
| 15 | optimizer | `lewm.yaml`: **AdamW, lr 5e-5, weight_decay 1e-3** | runs 0-2 used lr 1e-5 (run 0) then 1e-5 + cosine | **DEVIATION — OPEN and important** |
| 16 | epochs | `lewm.yaml`: `max_epochs 100`; paper App. E says 10 | 10 | **DEVIATION — recorded conflict** |
| 17 | batch size | `lewm.yaml`: 128 | 128 | **MATCH** |
| 18 | precision / grad clip | `bf16`, `gradient_clip_val 1.0` | to check | **UNVERIFIED** |
| 19 | seed | `lewm.yaml`: 3072 | 0 | **DEVIATION — benign, but record it** |

Row 15 deserves attention. The reference's learning rate is **5e-5**, five
times what run 0 used. Our founding claim is that the released configuration
does not converge — but we did not run the released learning rate.

## Evaluation (already verified end to end)

| # | element | status |
|---|---|---|
| 20 | environment | **MATCH** — bit-identical to the data generator (pixel MAE 0.00, replay 0.000) |
| 21 | goal construction | **MATCH** — recorded state `goal_offset` steps ahead, goal image rendered with the agent at the goal |
| 22 | success rule | **MATCH** — the registered env's `distance_to_target < 16` |
| 23 | CEM settings | **MATCH** — 300 / 30 / 30 / var 1.0, horizon 5, action_block 5 |
| 24 | goal_offset | **CONFLICT** — repo 25, paper implies 100; quantified at 24 points |
| 25 | whole harness | **VALIDATED** — the authors' released weights score 84.0% through it against a published ~87% |

---

## What this changes

Three deviations are open and were unknown yesterday: **pixel normalisation**
(row 4), **action normalisation** (row 5), and the **learning rate** (row 15).
Rows 4 and 5 mean our encoder saw a different input distribution than the
reference's throughout training. Row 15 means the claim "the released config
does not converge" was tested at one fifth of the released learning rate.

Six rows remain **UNVERIFIED** and must be closed before the paid run, because
every one of them could be another silent divergence of exactly the kind that
has already cost this project three runs and two withdrawn claims.

## Rule going forward

No row may be marked MATCH from a config file alone. Configs describe
intentions; the two most expensive bugs here lived entirely in code that no
config mentions.
