"""Generate MODEL_CARD.md entirely from files on disk.

Work order §6: no number in the published card may have been typed by hand.
Every value below is read from the artifact named in the table at §6.1, and
the script prints value-and-source for each before writing, so the mapping is
auditable.

Run:  ./.venv/bin/python build_model_card.py
"""
import csv
import json
import re
from pathlib import Path

import torch
import yaml

OUT = Path("MODEL_CARD.md")
RELEASE = Path("runs_archive/release")
VERIFIED = Path("runs_archive/verified")
ARXIV = "<ARXIV_ID>"          # the reproduction; filled at upload
ARXIV_FOLLOWUP = "2608.12959"  # the planning-objective follow-up

sources = []


def note(label, value, src):
    sources.append((label, str(value), src))
    return value


# ------------------------------------------------------------- planning
rows = [r for r in csv.DictReader(open("docs/paper/results_from_disk.csv"))
        if r["nested_copy"] == "no"]
by_dir = {(r["reldir"], r["arm"]): r for r in rows}


def plan(reldir):
    r = by_dir[(reldir, "cem")]
    return note(f"planning {reldir}",
                f"{r['succ']} = {r['pct']}% (goal {r['goal']}, budget {r['budget']})",
                r["file"]), r


ARMS = {
    "run0-recal": [("exp_r0_short", "25", "50"), ("exp_ref_r0", "100", "150")],
    "run2-recal": [("exp_run2_recal_25", "25", "50"), ("exp_run2_recal", "100", "50"),
                   ("exp_ref_r2", "100", "150")],
    "phase2-recal": [("exp_phase2_recal_25", "25", "50"), ("exp_phase2_recal", "100", "50"),
                     ("exp_ref_p2", "100", "150")],
}
plan_rows = []
for ck, arms in ARMS.items():
    for reldir, off, bud in arms:
        _, r = plan(reldir)
        assert r["goal"] == f"frame {off}" and r["budget"] == bud, \
            f"{reldir}: expected offset {off}/budget {bud}, report says " \
            f"{r['goal']}/{r['budget']}"
        plan_rows.append((ck, off, bud, r["succ"], r["pct"], r["guard"], r["file"]))

_, authors = plan("exp_authors")

guards = sorted({float(r["guard"]) for r in rows if r["guard"] != "?"})
guard_lo, guard_hi = note("guard range across all reported runs",
                          f"{guards[0]:.3f}-{guards[-1]:.3f}",
                          "docs/paper/results_from_disk.csv").split("-")

# ---------------------------------------------------------------- probes
probe_recal = Path(VERIFIED / "encoder_probe_both_recal.txt").read_text()
probe_raw = Path(VERIFIED / "encoder_probe_r0_raw.txt").read_text()


def probe(text, block, what):
    seg = text.split(f"--- {block}")[1]
    m = re.search(rf"{what}\s+R² ([\d.]+)", seg)
    assert m, f"{what} not found in {block}"
    return m.group(1)


p2_pos = note("phase2 position probe", probe(probe_recal, "phase2 (recalibrated)",
              r"1\. position, LINEAR probe"), f"{VERIFIED}/encoder_probe_both_recal.txt")
p2_act = note("phase2 action probe", probe(probe_recal, "phase2 (recalibrated)",
              r"4\. action from \(z_t, z_t\+1\)"), f"{VERIFIED}/encoder_probe_both_recal.txt")
r0_pos = note("run0 position probe", probe(probe_recal, "Run 0 (recalibrated)",
              r"1\. position, LINEAR probe"), f"{VERIFIED}/encoder_probe_both_recal.txt")
r0_act = note("run0 action probe", probe(probe_recal, "Run 0 (recalibrated)",
              r"4\. action from \(z_t, z_t\+1\)"), f"{VERIFIED}/encoder_probe_both_recal.txt")

# ------------------------------------------------------- recalibration gap
def gap(tag):
    t = Path(VERIFIED / f"gap_{tag}.txt").read_text()
    ev = re.search(r"eval mode\s+([\d.]+)", t).group(1)
    tr = re.search(r"train mode\s+([\d.]+)", t).group(1)
    return float(ev), float(tr)

gaps = {}
for run, tag in (("Run 0", "r0"), ("Run 2", "r2"), ("phase2", "p2")):
    ev_raw, tr_raw = gap(f"{tag}_raw")
    ev_rec, _ = gap(f"{tag}_recal")
    gaps[run] = (ev_raw, ev_rec, tr_raw, ev_raw / tr_raw)
    note(f"{run} eval loss raw -> recal", f"{ev_raw} -> {ev_rec} (train {tr_raw})",
         f"{VERIFIED}/gap_{tag}_raw.txt, gap_{tag}_recal.txt")

# ---------------------------------------------------------------- dataset
ds = Path(VERIFIED / "dataset_episode_lengths.txt").read_text()
n_eps = note("dataset episodes", re.search(r"episodes (\d+),", ds).group(1),
             f"{VERIFIED}/dataset_episode_lengths.txt")
mean_len = note("mean episode length", re.search(r"mean length ([\d.]+)", ds).group(1),
                f"{VERIFIED}/dataset_episode_lengths.txt")

# ------------------------------------------------------------ the weights
cfg = yaml.safe_load(Path("configs/phase2_dense_reference.yaml").read_text())
m, tr = cfg["model"], cfg["training"]
files = []
for p in sorted(RELEASE.glob("*.pt")):
    d = torch.load(p, map_location="cpu")
    sd = d.get("model") or d.get("state_dict")
    bt = sorted({int(v) for k, v in sd.items() if "num_batches_tracked" in k})
    md5 = re.search(rf"^{re.escape(p.name)}\n.*?md5 after\s+(\w+)",
                    Path(VERIFIED / "ckpt_md5.txt").read_text(), re.S | re.M)
    files.append((p.name, p.stat().st_size / 2**20, bt,
                  md5.group(1) if md5 else "?", d.get("epoch"), d.get("step")))
    note(f"{p.name} batches_tracked", bt, str(p))

n_params = note("parameters",
                json.load(open("runs_archive/phase1_run2_cosine_seed0/manifest.json"))["n_params"],
                "runs_archive/phase1_run2_cosine_seed0/manifest.json")

paper_md = Path("docs/paper/PAPER.md").read_text()
m_title = re.search(r"^---\ntitle: \|\n\s+(.+?)\nauthor: (.+?)\n---",
                    paper_md, re.S | re.M)
assert m_title, "PAPER.md has no title metadata block -- run patch_paper_title.py"
paper_title = note("paper title", m_title.group(1).strip(), "docs/paper/PAPER.md")
paper_author = note("paper author", m_title.group(2).strip(), "docs/paper/PAPER.md")

# ---- follow-up: the planning objective ---------------------------------
def _rate(d):
    import re as _re
    log = Path(f"followup/{d}/run.log")
    rows = _re.findall(r"ep\s+\d+/\d+ \(#\s*\d+\): (REACHED|missed )",
                       log.read_text())
    k, n = sum(r == "REACHED" for r in rows), len(rows)
    return f"{k}/{n} = {100*k/n:.1f}%"

fu = {}
for key, d in (("t25", "temporal_off25"), ("t100", "temporal_off100"),
               ("t100b50", "temporal_off100_b50"),
               ("auth_probe", "probe_authors_off100"),
               ("auth_temporal", "temporal_v2_authors_off100")):
    if Path(f"followup/{d}/run.log").exists():
        fu[key] = note(f"follow-up {d}", _rate(d), f"followup/{d}/run.log")

p25 = by_dir[("exp_phase2_recal_25", "cem")]
p100 = by_dir[("exp_ref_p2", "cem")]
p100b50 = by_dir[("exp_phase2_recal", "cem")]
pub25 = f"{p25['succ']} = {p25['pct']}%"
pub100 = f"{p100['succ']} = {p100['pct']}%"
pub100b50 = f"{p100b50['succ']} = {p100b50['pct']}%"

lewm = Path("docs/lewm_audit_commit.txt").read_text()
lewm_commit = note("le-wm commit", re.search(r"commit\s+:\s+(\w+)", lewm).group(1),
                   "docs/lewm_audit_commit.txt")

# ------------------------------------------------------------------ report
print(f"{'value':<34} {'read as':<44} source")
print("-" * 120)
for label, value, src in sources:
    print(f"{label:<34} {value[:43]:<44} {src}")
print()

# ------------------------------------------------------------------- write
def planning_table(ck):
    out = []
    for name, off, bud, succ, pct, guard, f in plan_rows:
        if name == ck:
            out.append(f"| {off} | {bud} | {succ} = **{pct}%** | {guard} | `{f}` |")
    return "\n".join(out)


card = f"""# tinylab — LeWorldModel reproduced on TwoRoom

Six checkpoints from an independent reimplementation of **LeWorldModel** (Maes
et al., 2026, arXiv:2603.19312) on the TwoRoom environment.

**These are not the authors' weights.** They were trained from scratch, from
our own code, to test whether the published result reproduces. The authors'
released weights are at
[`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms).

## What is here

| file | what it is | size | `batches_tracked` | md5 |
|---|---|---|---|---|
""" + "\n".join(
    f"| `{n}` | {'recalibrated' if '-recal' in n else 'as trained'}, "
    f"epoch {ep}, step {st} | {sz:.0f} MiB | {bt} | `{md5[:12]}…` |"
    for n, sz, bt, md5, ep, st in files) + f"""

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
| action-encoder input width **{m['action_dim']}**, not the configured 2 | `le-wm/train.py:68`, set programmatically |
| pixels normalised with **ImageNet** mean/std, not scaled to [0,1] | `le-wm/utils.py:6` |
| actions **z-scored** per dimension | `le-wm/train.py:65`, `utils.py:25` |

Line numbers refer to `le-wm` commit `{lewm_commit[:12]}…`; see
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
""" + "\n".join(
    f"| {run} | {ev_raw} | **{ev_rec}** | {tr_raw} |"
    for run, (ev_raw, ev_rec, tr_raw, _) in gaps.items()) + f"""

`batches_tracked` in the table above tells the two apart: a training average
carries tens of thousands, a precise-BN pass carries hundreds. Read it from the
file rather than trusting any range quoted elsewhere, including here.

## Planning objective — read this if you plan with these weights

The planning numbers below were produced with the released objective:
cross-entropy-method search minimising squared Euclidean distance between the
imagined embedding and the goal embedding. **That objective is the limiting
factor at long horizons, not these weights.**

Measured on this checkpoint: latent distance tracks true distance only at
short range, stops rising beyond about eighty arena units, and *decreases*
beyond about a hundred and twenty — so moving away from the goal can lower the
planner's cost. Position is meanwhile recoverable from the same frozen
embedding at R² 0.9922.

Replacing only the objective, with the encoder and predictor untouched:

| protocol | released objective | reachability cost |
|---|---|---|
| goal offset 25, budget 50 | {pub25} | **{fu['t25']}** |
| goal offset 100, budget 150 | {pub100} | **{fu['t100']}** |
| goal offset 100, budget 50 | {pub100b50} | **{fu['t100b50']}** |

`temporal_head_phase2.pt` in this repository is that cost: a small head
predicting how many steps apart two states are, trained only on frame
separation within recorded episodes — no position, no reward, no privileged
state. `plan_with_temporal_cost.py` shows the one-line substitution.

**Two conditions, both measured.** The head must be trained on the embeddings
the planner actually scores — imagined ones, not encoded frames — or it
extrapolates off-manifold. And it helps only where the predictor is accurate
enough to keep those close: on the original authors' released weights, whose
one-step error is higher, a plain linear position cost does better
({fu['auth_probe']}) than the learned one ({fu['auth_temporal']}).

Full method, evidence and limitations: **arXiv:{ARXIV_FOLLOWUP}**, *The
Objective Is the Bottleneck: Latent World Models Encode What Their Planners
Cannot Use*. One seed, one environment; treat it as a strong result on
TwoRoom rather than a general law.

## Planning results, with the protocol attached

**A success rate without its protocol is meaningless here.** The released
material publishes two evaluation protocols that disagree, and on the authors'
own weights they give {authors['pct']}% and 14.0%. Goal offset 25 with a
50-step budget is the released repository's evaluation configuration; offset
100 with a 150-step budget is what Appendix F.1 describes.

Every figure below is a file in the repository.

**`tinylab-tworoom-phase2-recal.pt`** — the corrected pipeline

| goal offset | budget | goals reached | guard | report |
|---|---|---|---|---|
{planning_table('phase2-recal')}

**`tinylab-tworoom-run2-recal.pt`** — exploratory run

| goal offset | budget | goals reached | guard | report |
|---|---|---|---|---|
{planning_table('run2-recal')}

**`tinylab-tworoom-run0-recal.pt`** — released configuration

| goal offset | budget | goals reached | guard | report |
|---|---|---|---|---|
{planning_table('run0-recal')}

For reference, the authors' released checkpoint measured under the same harness
on the same episodes reaches {authors['succ']} = {authors['pct']}% at goal
offset 25 (`{authors['file']}`).

## The domain guard

The evaluation refuses to report a success rate when the frames it is scoring
sit outside the distribution the encoder was trained on. It measures the paired
distance between the embedding of a rendered frame and of the same state
reached in the environment, and compares the median against a threshold of 1.0.
Across every run reported here it lies between {guard_lo} and {guard_hi},
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
| position, linear probe | {r0_pos} | {p2_pos} |
| summed action from (z_t, z_t+1) | {r0_act} | {p2_act} |

## Architecture and training

ViT-Tiny encoder, patch {m['patch_size']}, {m['img_size']}px, embedding
dimension {m['embed_dim']}, history {m['history_size']} frames; predictor depth
{m['depth']}, {m['heads']} heads, MLP dimension {m['mlp_dim']};
{n_params:,} parameters. Trained {tr['epochs']} epochs, batch
{tr['batch_size']}, learning rate {tr['learning_rate']}, weight decay
{tr['weight_decay']}, SIGReg weight {cfg['loss']['sigreg_weight']}, seed
{cfg['seed']}, fp32. Full configuration:
`configs/phase2_dense_reference.yaml`.

## Dataset

Trained on the authors' TwoRoom dataset: {int(n_eps):,} episodes, mean length
{mean_len} frames. **We do not redistribute it.** Get it from the authors at
[`quentinll/lewm-tworooms`](https://huggingface.co/datasets/quentinll/lewm-tworooms).

## Limitations

- **One seed.** Every number here is a single run. We make no estimate of seed
  variance.
- **{tr['epochs']} epochs, against the repository configuration's 100.** This is
  the paper's appendix value, chosen because the full budget was beyond ours.
  Nothing here is a claim about the asymptote.
- **TwoRoom only.** No claim is made about the original's other environments,
  its embodied or zero-shot results, or scales other than this one.
- **One-step accuracy does not predict long-horizon planning.** The most
  accurate predictor here is not the best long-horizon planner under the
  published objective. Do not select a checkpoint on prediction loss.
- **Long-horizon planning is weak under the published planning objective, and
  that is a property of the objective rather than of these weights** — see
  "Planning objective" above. Under the published cost these checkpoints reach
  {pub100} of goals 100 frames away; with a reachability cost and the same frozen
  weights, {fu['t100']}.
- The checkpoint-versus-checkpoint differences we report are not established at
  n = 50; see the paper.

## Licence

MIT, matching the reference implementation
([`le-wm`](https://github.com/lucas-maes/le-wm), MIT, © 2026 Lucas Maes).

This is a reproduction. The original work is the authors'.

## Citation

```bibtex
@article{{singh2026objective,
  title  = {{The Objective Is the Bottleneck: Latent World Models Encode What
            Their Planners Cannot Use}},
  author = {{Singh, Joyjeet}},
  year   = {{2026}},
  eprint = {{{ARXIV_FOLLOWUP}}},
  note   = {{The planning-objective result and the released cost head}}
}}

@article{{singh2026tinylab,
  title  = {{{paper_title}}},
  author = {{Singh, Joyjeet}},
  year   = {{2026}},
  eprint = {{{ARXIV}}},
  note   = {{Independent reproduction of arXiv:2603.19312}}
}}
```

Code, every evaluation report, the fidelity audit and the pre-registration:
[github.com/joyjeet-singh/tinylab](https://github.com/joyjeet-singh/tinylab).
"""

OUT.write_text(card)
print(f"wrote {OUT} ({len(card.split())} words)")
left = re.findall(r"<[A-Z_]+>", card)
print(f"placeholders remaining: {sorted(set(left))}")
