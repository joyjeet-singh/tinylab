# Rental Runbook v2 — TwoRoom real-data run (session 2)

Session 1 (2026-07-18) reached: box validated, env pinned + verified on GPU,
real data staged server-side from HF (md5 120e4327f4f2cf64b74928ca5bfca719),
Gates A/B/C all PASS (10,000 NaN @ episode-final; workers 0 vs 64 IDENTICAL;
knee at 16 workers ≈ 1000 clips/s → ~1.9 h loading). Blocked at the resume
gate + Gate D by three off-clock bugs. Fix them free, rehearse free, then
re-rent. Box destroyed between sessions by design (replay ≈ 20 min, ~$0.15).

---

## Phase −1 — Mac-side fixes (tonight, free)

Do these in the Mac tinylab folder, in order:

### 1. Complete the pins (the ViT import failure)
Add to requirements.txt:
    transformers==4.38.2
(It was installed ad hoc on the Mac and never pinned — the box exposed it.)

### 2. Untrack the data file (the corrupt-h5-via-git failure)
    git rm --cached toy_tworoom.h5
    printf "*.h5\n" >> .gitignore
Rule restated: code travels by git; data is regenerated (make_toy_tworoom.py)
or downloaded (HF). An 84 MB binary in git nearly hit GitHub's 100 MB hard
limit and arrived on the box corrupt.
Optional forensics (only if curious why): compare
    git show HEAD:toy_tworoom.h5 | md5sum     vs     md5sum toy_tworoom.h5

### 3. Parametrize the resume gate (the hardcoded-config bug)
In check_resume_equivalence.py, replace line 22
    CFG = "configs/toy_lewm.yaml"
with:
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--config", default="configs/toy_lewm.yaml")
    CFG = _p.parse_args().config
Nothing else changes (lines 52/60/64 keep using CFG). Default behavior is
byte-identical; on the box it will run with --config configs/rental_tworoom.yaml.

### 4. Make sure the rental card is in the repo
configs/rental_tworoom.yaml (batch 128, num_workers 16, h5_path
/dev/shm/data/tworoom.h5, epochs 10 — the 10-vs-8 call is still open;
flip to 8 before committing if you choose continuity with the Mac run).

### 5. Commit and push
    git add -A
    git commit -m "session2 prep: pin transformers, untrack h5, --config for resume gate, rental card"
    git push
    git status        # must be clean

---

## Phase −0.5 — THE REHEARSAL (Mac, free) — this is the repo verification

A clean clone from GitHub is the only honest test that "everything's there."

    cd /tmp && rm -rf tinylab-rehearsal
    git clone https://github.com/joyjeet-singh/tinylab tinylab-rehearsal
    cd tinylab-rehearsal
    python3.11 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt && pip install hdf5plugin
    python make_toy_tworoom.py                  # regenerate toy data (use your usual flags)
    python check_resume_equivalence.py          # default config → expect 0.000e+00, as before
    time python train_toy_lewm.py --config configs/toy_lewm_vit.yaml --seed 0 --max-steps 10

PASS = the clone alone, plus pins alone, reproduces your known results.
Then and only then is the GitHub repo proven complete, and the box replay
is deterministic. Any failure here is a failure you just avoided paying for.

---

## Phase 0 — Provision + replay (next box, meter on)

Same shopping filter: cores ≥ 32, reliability ≥ 99%, RAM ≥ 64 GB, on-demand,
cheapest passer. NVIDIA CUDA template. Then:

    tmux new -s run
    apt-get update && apt-get install -y zstd tmux
    cd /workspace && git clone https://github.com/joyjeet-singh/tinylab && cd tinylab
    uv venv --python 3.11 .venv && source .venv/bin/activate     # fallback: python3.11 -m venv
    uv pip install -r requirements.txt && uv pip install hdf5plugin "huggingface_hub[hf_transfer]"
    python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
        # expect: 2.2.2+cu121 True <GPU>   ← env gate

    mkdir -p /dev/shm/data
    hf download quentinll/lewm-tworooms --repo-type dataset --local-dir /dev/shm/data/hf
    cd /dev/shm/data && tar --zstd -xvf hf/*.tar.zst
    md5sum tworoom.h5                            # expect 120e4327f4f2cf64b74928ca5bfca719
    rm -f hf/*.tar.zst

## Phase 1 — Gates (first minutes)

    cd /workspace/tinylab
    python parallel_data.py --h5 /dev/shm/data/tworoom.h5 --workers 0 2 16
        # Gate A: 10,000 NaN rows, all episode-final   → else ABORT
        # Gate B: IDENTICAL                            → else ABORT
        # Gate C: ~1000 clips/s at 16 on a 40-core box (scale expectation to cores)

    python check_resume_equivalence.py --config configs/rental_tworoom.yaml
        # validates resume AT BATCH 128 on THIS box → expect 0.000e+00   → else ABORT

    time python train_toy_lewm.py --config configs/rental_tworoom.yaml --seed 0 --max-steps 50
        # Gate D: sec/step from settled steps; total steps ≈ 693k/128 × epochs (≈54k at 10)
        # project: wall_h ≈ sec/step × 54,000 / 3600 ; cost ≈ wall_h × $/hr
        # GO if single-digit dollars; STOP and re-cost past ~$15–20
        # glance nvidia-smi: batch-128 VRAM comfortably under card limit

## Phase 2 — Launch (inside tmux)

    nohup python -u train_toy_lewm.py --config configs/rental_tworoom.yaml --seed 0 \
        > runs/live.log 2>&1 &
    tail -f runs/live.log
        # trust: train pred + spread (spread NOT → 0). distrust: eval-mode pred.
        # mid-run: probe R² from a checkpoint — climbing toward ~0.9 = learning.

## Phase 3 — Eval, pull, destroy

    # gate order enforced: probe R² ≥ 0.8 FIRST, only then interpret planning success
    # target ≈ 0.87 ; ≈ 1.00 is suspicious ; fail-with-good-R² → scoring fork (App. F.2)
    scp/rsync runs/ off the box (manifest, metrics.jsonl, checkpoints, live.log)
    DESTROY the instance (never stop).
