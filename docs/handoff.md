# Handoff — lab conventions and session context

Two purposes: (1) the standing rules any assistant helping in this repo
must follow; (2) a paste-ready context block for starting new sessions.

## Lab conventions (non-negotiable)
1. Every reported number is computed by an analyzer script reading
   runs/*/metrics.jsonl and manifest.json. No means, gaps, or claims from
   memory, screenshots, or chat — including the assistant's.
2. Every experiment is fully described by a YAML recipe card in configs/.
   New experiment = new card (or CLI override recorded in the manifest),
   never a code edit.
3. Every run writes its own timestamped folder under runs/ with a
   manifest (full config copy, run seed, git commit, data hashes, library
   versions) written BEFORE training, and one JSONL line per measurement,
   flushed as it happens.
4. Determinism is a test, not a vibe: identical command twice must give
   byte-identical metrics. data_seed (which data) is fixed forever at 42;
   the run seed varies 0/1/2. Multi-seed (n=3) for any compared claim.
5. A one-line PREDICTION is written before every main run and graded
   after. Intuition is good at signs, bad at sizes.
6. Pin every dependency the moment it proves itself (pip freeze | grep).
   Environment = recorded choice, not weather.
7. Files are delivered one per heredoc (or downloaded), then
   `python -m py_compile` before first run. git status before every
   commit. Commit per checkpoint, not per week.
8. Explanations: plain language first, then the term of art. New concepts
   get a mechanical picture before math.
9. One lane (latent world models), one vehicle, one question at a time.
   New ideas get one line in a parking-lot list, not a branch.
10. Reading: one deep paper per week, paired with that week's build,
    logged per-claim ("what experiment verifies this, at what cost").

## Session-start context block (paste into a new chat)

CONTEXT: tinylab learning program (Phase 0), July 2026.
I am Joyjeet — independent ML researcher, Chandigarh; 2017 Intel MacBook
8GB (Python 3.11 venv, torch pinned 2.2.2 — Intel-Mac ceiling), free
Kaggle (~30 GPU-h/wk) + Colab, ~$10 Vast.ai reserve. Learning style:
build-first, plain language before jargon, minimal formatting.
Repo ~/tinylab: config-driven runs (configs/*.yaml), seeded determinism
(data_seed 42 fixed; run seeds 0/1/2), lablog manifest+JSONL per run,
analyze.py computes ALL reported numbers from logs. Read docs/handoff.md
conventions and follow them strictly.
State: Week 1 done — MLP vs CNN, CIFAR-10 5k subset, 3 seeds:
cnn 0.5240±0.0263 vs mlp 0.4357±0.0100 (final-epoch metric); reruns
bit-identical. Direction: Lane A latent world models (LeWM lineage);
thesis "structure and search in latent space buy back what small models
lack in scale."
Now executing: docs/week2_rl_spec.md (then week3_worldmodel_spec.md,
then phase1_and_phase2_plan.md). Help me execute the current spec
step-by-step; do not redesign it unless I ask; enforce the conventions
(especially: predictions before runs, numbers from the analyzer only).

## Mechanical to-dos (any session can clear these)
- GitHub: create private repo `tinylab`; git remote add origin ...;
  git push -u origin main. Then the Colab/Kaggle ritual is: clone,
  pip install -r requirements.txt, run scripts, download runs/, analyze
  locally.
- Accounts: Kaggle (verify phone for GPU), Colab sign-in, Vast.ai with
  ~$10 (do not spend until a run is blocked on free tiers).
- Optional: `pip freeze > requirements.lock.txt` for a full snapshot
  alongside the curated requirements.txt.
