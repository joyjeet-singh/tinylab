# Rental-Day Runbook — LeWM TwoRoom (real data)

**One job:** when the meter starts, no decisions remain. Only commands, gates, and
pre-written abort criteria. If you find yourself *thinking* on a paid clock, stop —
the thinking should already be on this page.

Reconstructed from project memory (state: *free phase closed, rent next*, 2026-07-18).
Depends on two sibling files: `real_data_recipe.yaml` and `rental_requirements.md`.

Repro target: **land near 0.87, not near 1.00.** A higher number means a *different*
model was built, not a better one.

---

## Phase −1 — Pre-boot (do this OFF the clock, before you rent)

Everything below must already be green on the Mac / in the repo, or you are paying to
discover something you could have found for free:

- [ ] tinylab clean, pushed, at the commit you intend to run (`manifest.json` will
      record this hash — verify it's the right one).
- [ ] `parallel_data.py` equivalence **passes bitwise** on the Mac against the real
      12 GB file (workers=0 vs 2 → IDENTICAL). ✔ already done 2026-07-18.
- [ ] NaN scan on the real file returns **10,000 rows, all at episode-final steps.**
      ✔ already confirmed.
- [ ] **Gap 1 decided:** batch size = **16** or **128**? (see recipe card + notes below).
- [ ] **Gap 2 acknowledged:** you are booting into a *measurement* phase, not the full
      run. Viability is still a prediction until Gate C lands on the box.
- [ ] Abort criteria (bottom of this file) read and accepted.

---

## Phase 0 — Provision (meter starts here)

Rent per `rental_requirements.md`: Vast.ai, **on-demand (not interruptible)**, 3090
preferred, **CPU cores ≥ 32**, RAM ≥ 64 GB, disk ≥ 60 GB, reliability ≥ 99%, sorted by
price.

```bash
# on the box
git clone https://github.com/joyjeet-singh/tinylab && cd tinylab
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt         # pinned: torch 2.2.2, transformers 4.38.2, numpy 1.26.4
pip install hdf5plugin                  # REQUIRED — real file uses Blosc filter 32001
# get the 12 GB file onto the box (scp / rclone / provider upload)
```

---

## Phase 1 — First hour = gates, not training

Run these **in order**. Each has a pre-written verdict. Do not start the full run until
all four say GO.

### Gate A — Right file, right version (NaN fingerprint)
```bash
python parallel_data.py --h5 data/tworoom.h5 --scan-nan   # or your benchmark's scan flag
```
- **GO:** exactly **10,000** NaN rows, **all** at episode-final steps.
- **ABORT:** any other count (esp. 0) → wrong file or wrong version. Do not train.

### Gate B — Resume is bitwise-safe on THIS box
```bash
python parallel_data.py --h5 data/tworoom.h5 --workers 0 2   # bitwise equivalence
python check_resume_equivalence.py                           # 0.000e+00 across all params
```
- **GO:** IDENTICAL / `0.000e+00`.
- **ABORT:** any mismatch. The Mac spawn-path NaN red herring is already resolved; a new
      mismatch here is a *new* bug — do not spend the run on it.

### Gate C — Viability (the number the whole rental hinges on)
```bash
python parallel_data.py --h5 data/tworoom.h5 --workers 0 1 2 4 8 16 28   # clips/sec sweep
```
- Record best clips/sec → set `num_workers` in the recipe (target ≈ cores − 4).
- Project data-path hours; with prefetch, wall ≈ **max(data path, GPU path)**.
- **GO:** projected cost (hours × $/hr) is **under the $50 ceiling** with margin.
- **ABORT / reconsider:** projection blows the envelope. (Mac was 166.5 h data-limited;
      32 cores should be dramatically better — but confirm, don't assume.)

### Gate D — Step timing + launch config
```bash
# short timed burst at the chosen batch + num_workers
python train_toy_lewm.py --config real_data_recipe.yaml --max-steps 50 --time-only
```
- Measure sec/step → project full-run wall time and $ against the budget one more time.
- **GO:** numbers hold → launch. **NO-GO:** revisit batch / workers.

---

## Phase 2 — Full run

```bash
nohup python -u train_toy_lewm.py --config real_data_recipe.yaml > runs/live.log 2>&1 &
# -u AND stdout to a file — empty nohup logs were stdout buffering; this is the fix
```

**Watch these; trust only these** (see Gap 1):
- `train_pred` and `spread` — should stay healthy (pred low, spread NOT collapsing).
- **Ignore eval-mode `pred`.** It swings ~100× (BatchNorm × batch-16 artifact), *not* a
  health signal. If you're on batch 128 and it settles, good — but still don't gate on it.
- Collapse watch: if `spread → ~0` while `pred → ~0`, SIGReg has lost the fight. Abort.

**Mid-run probe (don't skip):**
```bash
python toy_plan.py --checkpoint runs/<ts>/ckpt_latest.pt --probe-only   # R² from checkpoint
```
- R² climbing toward ~0.9 = encoder is learning. R² stuck low = train longer or something's wrong.

Checkpoint often enough that a crash costs < ~15 min. Resume is proven safe — but only
because Gate B passed on *this* box.

---

## Phase 3 — Eval, then teardown

```bash
python toy_plan.py --checkpoint runs/<ts>/ckpt_best.pt        # CEM planner vs random control
python analyze.py runs/<ts>/                                  # ALL numbers from the logs
```

**Enforced gate order — do not skip:**
1. `probe_r2 ≥ 0.8`?
   - **No** → encoder can't see yet. The planning number is meaningless. Train longer.
     Do not interpret it, do not report it.
   - **Yes** → proceed to (2).
2. Planning success:
   - Near **0.87** → reproduction landed. Stop.
   - Near **1.00** → suspicious. A different model got built. Investigate config, don't celebrate.
   - Fails **despite R² ≥ 0.8** → this is the **pre-registered fork**: the *scoring* is the
     problem (paper App. F.2; "Beyond Euclidean Proximity" arXiv 2605.22164), not the training.

**Pull everything OFF the box before you kill it:**
```bash
# manifest.json, metrics.jsonl, all checkpoints, the R² log, live.log
scp -r box:tinylab/runs/<ts>/ ./runs/
```

**DESTROY the instance — do not stop it.** Stopped instances bill for storage.

---

## Consolidated abort criteria (pre-written — no judgment calls on the clock)

| Gate | Abort when | Why |
|---|---|---|
| A | NaN rows ≠ 10,000 (esp. 0) | wrong file / version |
| B | resume equivalence ≠ IDENTICAL | new determinism bug — don't debug on the clock |
| C | projected cost > $50 ceiling | viability failed; the run doesn't fit the envelope |
| D | sec/step projection blows budget | config wrong before you've spent the run |
| live | `spread → 0` with `pred → 0` | SIGReg lost; representation collapsed |
| eval | R² < 0.8 | any planning number is meaningless — never report it |

When in doubt: **destroy and re-cost.** The box is cheap; a wrong run interpreted as
right is not.
