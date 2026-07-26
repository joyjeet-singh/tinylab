# Launch runbook — phase2_dense_reference

Everything from here to a retrieved, analysed run. Written before launch so the
abort criteria are pre-registered rather than invented while the meter runs.

---

## 0. Repository hygiene — do this first

The provenance commit swept in **seven `.bak` files**:

```
realenv_r2_planner_eval.py.bak2  .bak3  .bak4
train_toy_lewm.py.bak
tworoom_data.py.bak  .bak2  .bak3
```

These are pre-patch snapshots. A stale `tworoom_data.py.bak` sitting beside the
live `tworoom_data.py` is precisely the "which version produced this?" hazard
that cost this project three days — and a reproducer cloning the repo has no way
to know which is authoritative. Git history already holds every prior version.

```
git rm --cached realenv_r2_planner_eval.py.bak2 realenv_r2_planner_eval.py.bak3 \
  realenv_r2_planner_eval.py.bak4 train_toy_lewm.py.bak tworoom_data.py.bak \
  tworoom_data.py.bak2 tworoom_data.py.bak3
printf '*.bak\n*.bak[0-9]\n' >> .gitignore
git add .gitignore && git commit -m "Untrack patch backups; git history is the backup"
```

The files stay on disk — only the tracking goes.

**Also fix the seed comment** in `configs/phase2_dense_reference.yaml`. Replace
"arm identity; reference uses 3072" with:

> held at Run 0's value so the comparison against Run 0 isolates the four
> pipeline fixes; the reference uses 3072, which buys no comparability and
> would add a fifth difference

Then re-run G1 (the working-tree check will fail on uncommitted edits, correctly).

---

## 1. Local gates — all four must pass, in this order

```
python3 verify_dense_actions.py
python3 preflight_local.py --config configs/phase2_dense_reference.yaml \
    --h5 ~/Downloads/tworoom.h5 --steps 12 --batch 4
python3 gate_g1_fresh_clone.py --config configs/phase2_dense_reference.yaml
```

G3 (the domain guard) is embedded in every evaluation and needs no separate run.
G4 is `docs/expected_card_phase2.md`, already committed.

**Save all three outputs.** They are the evidence that the run was gated, and
they belong in the paper's appendix.

---

## 2. Staging — before the meter starts

1. Rent the box, clone the repo at the committed HEAD (G1 proved this suffices).
2. Install from `requirements.txt` — pinned, per the rental law.
3. Stage `tworoom.h5` to `/dev/shm/data/`. Verify `data_sha256` matches
   `ce1d3ebb...` from the Run 1/2 manifests **before** training starts.
4. **Rehearse retrieval now, not later.** Create a dummy file under `runs/`,
   `rsync` it back to the Mac, confirm it arrives. If retrieval is broken you
   want to know at minute two, not after ten epochs.

```
ssh <box> 'mkdir -p ~/tinylab/runs/_rehearsal && echo probe > ~/tinylab/runs/_rehearsal/probe.txt'
rsync -avP <box>:~/tinylab/runs/ ./runs_pulled/
cat runs_pulled/_rehearsal/probe.txt        # must print: probe
```

---

## 3. Launch, and the first thirty minutes

```
nohup python3 train_toy_lewm.py --config configs/phase2_dense_reference.yaml \
    --seed 0 > live.log 2>&1 &
tail -f live.log
```

### Abort criteria, pre-registered

Abort and diagnose locally rather than paying to watch it fail:

| check | expected | abort if |
|---|---|---|
| loader convention line | `{'dense_actions': True, 'imagenet_pixels': True, 'zscore_actions': True}` | any flag False — the config did not take effect |
| `manifest.json` | contains `loader_convention` and `n_params` **18,034,670** | different parameter count — the wrong architecture is training |
| `data_sha256` | `ce1d3ebb...` | mismatch — wrong data file staged |
| first loss values | finite | any NaN or inf — the NaN audit missed something and the gates have a hole |
| epoch 0 probe R² | Run 0 reached **0.9951** at epoch 0 | materially below ~0.95 — this is outcome **C** in the expected card: the pixel normalisation has broken the encoder. Abort; it is diagnosable locally for free |
| embedding spread after epoch 0 | Run 0 held 0.83–0.96 | below ~0.3 — collapse at full batch, which SIGReg should prevent |

Absolute prediction-loss values are **not** comparable to Run 0's 5.48: the
action encoding and input normalisation both changed, so the latent scale is
different. Judge the *trajectory*, not the level.

---

## 4. Retrieval — the law

Before destroying anything:

```
rsync -avP <box>:~/tinylab/runs/ ./runs_pulled/
md5 runs_pulled/<run>/ckpt_best.pt          # on the Mac
ssh <box> 'md5sum ~/tinylab/runs/<run>/ckpt_best.pt'
```

Both digests must match. Only then destroy the instance. Also pull `live.log`
and `manifest.json` — the log is the evidence, not a convenience.

---

## 5. After the run — and the free upgrade

Read the outcome against the expected card and record it whatever it is.

**If it converged**, the largest remaining improvement to the paper is free:
re-run every planning evaluation on the new checkpoint. That moves the entire
Tier-3 half of the ledger from a seven-deviation checkpoint to a faithful one.

```
export NEW=runs/<phase2 run dir>
python3 realenv_r2_planner_eval.py --run $NEW
python3 realenv_r2_planner_eval.py --run $NEW --random
python3 realenv_r2_planner_eval.py --run $NEW --episodes balanced_episodes.json --goal-offset 100
python3 realenv_r2_planner_eval.py --run $NEW --episodes balanced_episodes.json --goal-offset 100 --random
python3 analyze_balanced_wall.py --episodes balanced_episodes.json
```

Two things to settle first, though, because the new checkpoint has
`history_size` 3 where `ckpt_best` had 1, and dense 10-wide actions where
`ckpt_best` had 2:

- the evaluator's history buffer already reads `model.history_size`, so 3 is
  handled — but confirm it on a two-episode run before committing to 50
- **the planner emits 2-wide actions and the new model expects 10.** The
  natural encoding is the planner's action repeated `frameskip` times, exactly
  as `authors_adapter.py` does for the authors' checkpoint. That needs a small
  adapter and its own measurement — do not assume it, measure it the way
  `action_scale_check.py` did.

**If it did not converge**, the founding claim survives with its largest
confound eliminated, which is a stronger paper than today's. Record the log,
update the ledger, and move to writing.
