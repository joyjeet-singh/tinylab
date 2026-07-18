# Amendment 3 — 2026-07-16: from-scratch LeWM build complete; debugged free, ready to rent

Recorded before any paid compute. Appended, never edited into the body.

Covers the from-scratch LeWM reproduction adopted after Amendment 2 found the
real dataset unusable on free compute (203 h of data loading alone). Strategy:
**debug free, verify paid** — build and test every component against a small
synthetic stand-in, rent only when the sole remaining unknown is the number.

## A3.1 — What was built and verified (all on a 2017 MacBook, zero cost)

| file | what it is | verified by |
|---|---|---|
| `tworoom_data.py` | dataset reader (Amendment 2) | 4 checks incl. alignment vs planted fault |
| `make_toy_tworoom.py` | synthetic Two-Room, real file's exact format | verified loader reads it **unmodified** |
| `toy_model.py` | encoder, action encoder, predictor, 2 projectors | shapes, gradients, causality |
| `toy_sigreg.py` | SIGReg anti-collapse | collapse reproduced on demand, then prevented |
| `train_toy_lewm.py` | training loop on `lablog` scaffold | resume-equivalence gate |
| `check_resume_equivalence.py` | proof a split run == an unbroken run | **0.000e+00 across 93 params** |
| `toy_plan.py` | CEM planner + evaluation + position probe | runs end to end; self-diagnoses |

The toy is a debugging fixture, **not data**. Nothing measured on it is a result.
It is deleted when the real data takes over. Its only job was finding bugs
without a clock running. It found four.

## A3.2 — Bugs caught by the gates (each would have corrupted a paid run)

1. **Resume reshuffled the data mid-epoch.** A single generator carried across
   epochs had already advanced past that epoch's shuffle; resuming reshuffled
   with it, so the second half of the epoch saw *different clips* than an
   unbroken run. Same seed, different data. Fix: derive each epoch's order from
   `(seed, epoch)` alone — regenerable from scratch at any point.

2. **Resume replayed dropout.** `set_seed()` on resume reset torch's global RNG
   to the beginning, so resumed steps saw the *same dropout masks* the first
   steps saw while an unbroken run saw fresh ones. Every weight diverged. Fix:
   save and restore `torch.get_rng_state()`.

   Both were invisible without bit-for-bit comparison — training ran, loss fell,
   nothing looked wrong. **This is the entire argument for the gate.**

3. **My description of the projector was wrong.** I had said it gives SIGReg a
   separate view so the summary stays useful for prediction. Reading `jepa.py`:
   the projector is applied inline in `encode()` and its output **is** `emb` —
   the same tensor the predictor consumes and SIGReg acts on. There is no
   separate view. There is also a **second** projector (`pred_proj`) on the
   predictor's output that I had not mentioned. Both now implemented.

4. **The prediction target is NOT detached.** Reference `train.py`:
   `(pred_emb - tgt_emb).pow(2).mean()` — gradient flows into both sides, so the
   encoder is actively pulled toward making summaries *easy to predict*. That
   makes collapse **more** tempting, not less. My first demo detached it and was
   therefore not faithful.

## A3.3 — Properties of SIGReg found by testing (keep these)

- **Collapse is reproducible on demand.** Prediction alone: loss 0.199 → 0.00004
  (245x "better") while summary spread fell 0.249 → 0.0055. Perfect loss,
  useless model. With SIGReg at the config's λ=0.09: spread held at 0.746 —
  **135x** the collapsed run — while the bell score fell 20.9 → 3.4.
- **SIGReg is a good preventer, a poor rescuer.** Its gradient is *weakest
  exactly where collapse is worst*: 2.0e-04 at spread 0.01 vs 4.0e-03 at spread
  0.5. It must be present from step one.
- **Perfect collapse is a true fixed point.** With all points identical the
  gradient is identical for every point (variation exactly 0.0), so it can only
  shift the cloud, never pull it apart. Unreachable in practice — a real encoder
  always gives slightly different outputs for different pictures.
- Plain SGD struggles here (steps sized by a tiny gradient); Adam/AdamW escapes.
  The reference uses AdamW, so this is fine.

## A3.4 — The planning result, stated honestly

Toy world, 3 epochs (~1000 steps), 20 evaluation episodes:

| | success | mean final distance |
|---|---|---|
| CEM planner | **15%** (3/20) | 22.23 |
| random actions (control) | **20%** (4/20) | 17.66 |

**The planner loses to random.** The diagnostic explains why:

- Linear probe, position recovered from summary: **R² = 0.617**
  (reference LeWM on real TwoRoom after full training: **R² ≈ 0.996**)
- Summary-distance does not track real distance: 20 units away scored 8.1 while
  5 units away scored 64.3 — the distant picture looked *closer*.

**Diagnosis: the encoder has not learned to see. The planner has no signal to
follow.** ~1000 steps against the reference's ~14,000+. This is undertraining,
not a broken planner, and the toy cannot fix it — validating the planner needs a
properly trained encoder, which needs real compute. **This is the boundary of
what the free phase can establish, and it was reached deliberately.**

## A3.5 — The R² gate (new, adopted now)

`toy_plan.py` reports the position probe alongside every success rate, and this
becomes a **pre-condition for interpreting any planning number**:

- **R² < 0.8** → the encoder cannot see. The planning number says nothing about
  the planner. Train longer. Do not report it.
- **R² ≥ 0.8 and planning still fails** → the encoder sees fine; the *scoring* is
  the problem. Straight-line distance between summaries is a poor measure of real
  distance. This is the paper's own Appendix F.2 finding (LeWM: R² 0.996 on
  TwoRoom, yet planning underperforms; the authors attribute the gap to "the
  dynamics model or the planning procedure itself", not the representation), and
  what "Beyond Euclidean Proximity" (arXiv 2605.22164) argues is the cause.

We are **not** testing that claim. We reproduce the setup faithfully. It is
simply why **87** rather than 97 is the honest target.

## A3.6 — Deviations, all toy-only, all reverting when we rent

| item | toy | reference |
|---|---|---|
| encoder | small CNN | ViT-Tiny, patch 14, 224px |
| image size | 32x32 | 224x224 |
| embed_dim | 64 | 192 |
| predictor depth / heads | 4 / 4 | 6 / 16 |
| mlp_dim, proj_hidden | 256 | 2048 |
| batch size | 64 | 128 |
| num_proj (SIGReg) | 512 | 1024 |
| epochs | 3 | 10 (paper) / 100 (repo config) |
| total params | 0.553M | ~15-18M |

Faithful and unchanged: λ=0.09, lr 5e-5, weight decay 1e-3, frameskip 5,
train_split 0.9, history_size 3, num_preds 1, SIGReg knots 17, AdaLN-zero
conditioning, causal attention, undetached target, per-timestep SIGReg.

## A3.7 — Paper-vs-repo conflicts (now four, all unresolved, all recorded)

1. **history_size**: paper says 1 for TwoRoom; repo config says 3 globally.
2. **epochs**: paper App. E says 10; repo config says `max_epochs: 100`. At
   measured speeds that is 203 h vs 2,030 h of loading alone.
3. **goal_offset_steps**: repo `config/eval/tworoom.yaml` says **25**; the
   paper's TwoRoom section implies 100.
4. **projector role**: paper's description vs what `encode()` actually does.

Standing rule: **the repo is the reference, the paper is the description.** Where
they conflict we follow the repo and record it. None of these is resolved by
picking one — each is a live question for the reproduction to answer.

## A3.8 — Renting: what to buy and what to check first

Amendment 2 established free-tier is impossible (I/O alone 4-10x the compute it
feeds; block sampling bought **1.0x** — nothing). Revised guidance:

- **Filter for CPU cores and NVMe, not GPU model.** The bottleneck is Blosc
  decompression (~25x read amplification, structural to the file). A 4090 with
  32 cores beats an A100 with 8.
- Budget **$30-50**, not the $10 originally planned — you are paying for cores
  and disk, not just the GPU.
- **First hour is a measurement, not a run**: re-measure clips/sec on that box
  (9.5 was this MacBook's, not a universal constant) and confirm the projected
  hours before committing to a full run.
- Record in every manifest: machine type, core count, disk type, `hdf5plugin`
  version, measured clips/sec, and which requirements file was used.

**Order of operations when renting:**
1. Swap `h5_path` to the real `tworoom.h5`; swap the CNN encoder for ViT-Tiny at
   224px; restore reference sizes per A3.6.
2. Re-run `check_resume_equivalence.py` **on that box** before training. It
   caught two bugs on the laptop; a new environment is a new chance to break it.
3. Train. Checkpoint often — the gate proves resume is safe.
4. **Check R² before looking at the success rate** (A3.5). If R² < 0.8 the model
   is undertrained and the planning number is meaningless.
5. Only then compare to 87%. **Reproducing means landing near 87, not near 100.**
   A higher number means something different was built, not something better.
