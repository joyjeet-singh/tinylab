---
title: "Reproducibility Study: LeWorldModel on TwoRoom"
target: TMLR (then MLRC 2026 self-nomination if eligible)
status: skeleton — 31 July 2026
---

# TMLR paper skeleton

## What TMLR actually asks

Two criteria, and only two:

1. Are the claims made in the submission **supported by accurate, convincing and
   clear evidence**?
2. Would **some individuals** in TMLR's audience be interested in the findings?

Novelty and significance are explicitly not required. A careful reproduction
reporting a mixed result is a natural fit, and the entire game is evidence
quality. That is what this project has optimised for, so the writing job is to
present what exists — not to argue for importance.

**No page limit.** But TMLR reviewers read closely and reward concision. The
target is *thorough*, not *long*.

---

## The anti-repetition rule

Every claim gets **exactly one home section** where it is stated, evidenced and
qualified in full. Everywhere else it is *referenced*, not restated:
"…as established in §4.3…". If you find yourself re-explaining the domain gap in
the Discussion, stop — cite §5.1 instead.

The table below is the contract. Draft against it, and check it before
submission.

| claim | home | referenced from |
|---|---|---|
| 1 Harness reproduces 84.0% vs ~87% | §4.2 | Abstract, §3.4, §6.3 |
| 2 Environment bit-identical to generator | §3.3 | §4.2 |
| 3 Evaluation-domain gap (61.03 vs 0.01) | §5.1 | Abstract, §6.2, §6.3 |
| 4 Four undocumented deviations | §3.2 | Abstract, §4.3, §6.2, §6.3 |
| 5 Action deviation cost (25.59 vs 13.3) | §3.2 | §4.3 |
| 6 Executable-gate methodology | §3.4 | §6.3 |
| 7 Does not converge in 10 epochs | §4.3 | Abstract, §6.1 |
| 8 Dense actions change trend, not outcome | §4.3 | §5.4 |
| 9 Epochs conflict, empirically live | §4.3 | §6.1, §7 |
| 10 Convergence is LR-governed | §4.4 | Abstract, §6.1, §7 |
| 11 phase2 predictor never beats frozen world | §4.4 | §5.4 |
| 12 Representation reproduces; not the bottleneck | §4.1 | Abstract, §4.4 |
| 13 Higher-dimensional, less efficient code | §5.4 | — |
| 14 SIGReg prevents collapse | §4.1 | — |
| 15 Planning 72%/18% and 48%/0% | §4.5 | Abstract |
| 16 Wall effect, regime-dependent | §5.2 | §7 |
| 17 Receding horizon indistinguishable from random | §5.3 | — |
| 18 One-step accuracy ≠ planning success | §5.3 | §7 |
| 19 Scoring geometry 1.79× | §5.5 | — |

---

# Structure

## Abstract (~250 words)

Four sentences of setup, four of result, one of scope. Name the numbers: 84.0%
against ~87% for the protocol, non-convergence at ten epochs across two
configurations, four deviations invisible in configuration files, a
twenty-five-fold evaluation-domain gap. End with the scope limit: TwoRoom only.

## 1. Introduction (~600 words)

- The anomaly: LeWM reports ~87% on TwoRoom against baselines' 97–100%. The
  simplest task in the suite is where the method looks weakest, which makes it
  the informative one to reproduce.
- What this study contributes beyond pass/fail: deviations found by measurement
  rather than assumed, and a set of failure modes invisible in configuration
  files.
- **Contributions list** — five bullets, each pointing to its home section.
- **Scope paragraph.** TwoRoom diagnostic only; no claim about the paper's
  embodied or zero-shot results; one seed per configuration.

## 2. Scope of reproducibility (~400 words)

The three tested claims, quoted from the original with section citations:

- **C1** the encoder recovers agent position (~R² 0.996) — **reproduced**
- **C2** CEM planning over the learned model reaches ~87% — **protocol
  reproduced; our checkpoint does not reach it**
- **C3** the released configuration produces such a model in the stated budget
  — **not reproduced**

State plainly which are supported, which are not, and which are partial. TMLR
reviewers look for this up front.

## 3. Methodology (~1,800 words)

### 3.1 Model and objective (~300)
Architecture from the released config; MSE-plus-SIGReg with an undetached
target; 18,034,670 parameters against the reference checkpoint's 18,034,590.

### 3.2 Fidelity audit — **home of claims 4, 5** (~600)
*Draft exists: `docs/draft_sections_3.2_5.1.md`.* Source-level, not
config-level. 20 of 25 elements match. The four deviations, each with a
file-and-line citation, plus the two artifact-level corroborations. Table 1 is
the full deviation set.

### 3.3 Environment verification — **home of claim 2** (~250)
Pixel MAE 0.00, replay 0.000 at one and forty steps, paired latent 0.01. Short;
it is a precondition for everything in §4 and §5, not a result.

### 3.4 Experimental protocol and gates — **home of claim 6** (~450)
The four gates and the rule that runs count against budget only after they
pass. Include the pre-registration of the wall experiment. This section is what
§6.3 pays off.

### 3.5 Computational requirements (~200)
Four GPU runs at ~$6 each; ten epochs against the repository's 100; every
evaluation, gate and analysis on one 8 GB CPU laptop. State this plainly — a
reproduction achievable on that budget is itself useful information.

## 4. Results: reproducing the original (~2,200 words)

### 4.1 The representation reproduces — **home of claims 12, 14** (~450)
R² 0.9977 linear and 0.9995 by MLP on 4,000 held-out frames against ~0.996.
Then the sharper point: the summed action is linearly decodable from a latent
pair at R² 0.9207, so the information a predictor needs is present and
accessible — and the predictor still does not converge. **Lead with this
framing; it localises the failure and sets up §4.3–4.4.**

### 4.2 The evaluation protocol reproduces — **home of claim 1** (~400)
The released checkpoint through our harness: 42/50 = 84.0% against ~87%
(one-sample p = 0.53, CI [71.5, 91.7]). Every protocol element we report is
thereby validated. Note this is what licenses the numbers in §4.5.

### 4.3 Training does not reproduce — **home of claims 7, 8, 9** (~700)
Run 0 swings to 107% of its mean; phase2, with all four deviations eliminated on
byte-identical data, to 139%. Neither converges. phase2's loss falls 57% between
halves against Run 0's 15% — a descending oscillation rather than a flat one.
The epochs conflict (10 vs 100) becomes empirical here.

### 4.4 Convergence is learning-rate-governed — **home of claims 10, 11** (~450)
Table 2: run × learning rate × volatility × trend × best one-step ratio. The
only predictor beating a frozen-world baseline is Run 2's at 0.83, whose rate
fell to 1e-7. phase2's never does, under any input encoding — the true dense
actions score 2.266 against zeros at 2.044.

### 4.5 Planning, with provenance stated — **home of claim 15** (~250)
72.0% vs 18.0% at goal offset 25 (p = 7.4×10⁻⁶); 48.0% vs 0.0% at offset 100
(p = 1.2×10⁻⁷). **The checkpoint carries seven deviations from the reference;
say so here, in the body.**

## 5. Results: beyond the original (~1,800 words)

*No page limit means all five subsections survive. Keep each tight.*

### 5.1 A silent evaluation-domain gap — **home of claim 3** (~500)
*Draft exists.* The strongest transferable finding.

### 5.2 The wall costs the planner — **home of claim 16** (~450)
Pre-registered, distance-matched, ceiling verified at 100% by a door-routing
oracle. 79.1% vs 40.0%, +39.1 points, p = 3.4×10⁻⁸ — **and +12.7 points,
p = 0.054, at a corrected action scale, with non-overlapping intervals.**
Direction consistent, magnitude regime-dependent. Never quote +39.1 alone.

### 5.3 Two planner properties — **home of claims 17, 18** (~400)
Re-planning every block makes the planner indistinguishable from random
(p = 0.48). And a correction that improved one-step prediction cut success from
72% to 48% — accuracy and planning success are not monotonically related.

### 5.4 What the corrected pipeline does to the representation — **home of
claim 13** (~300)
Effective rank 11.9 → 16.5 of 192, position information unchanged, action
decodability lower. **Report the observation; state that the explanation we
tested was refuted.** Reporting a failed hypothesis is exactly the kind of
honesty TMLR rewards.

### 5.5 Scoring geometry — **home of claim 19** (~150)
Rooms separated by 1.79× at matched distance; the strong form of the Euclidean
critique is unsupported at the representation level.

## 6. Discussion (~1,200 words)

### 6.1 What was easy (~200)
Environment installation; the released checkpoint downloads and loads; the
encoder result appears within one epoch and is robust across pipelines.

### 6.2 What was difficult (~400)
The four deviations were invisible in configuration files. The domain gap
survived a 0.99 probe for three runs. Ten thousand NaN actions sat where a
subsampling loader stepped over them and a dense loader would not.

**Include the corollary: four of our gate failures were the gate's fault, not
the run's.** A reproduction that reports only the subject's failures is not
being straight, and TMLR reviewers notice.

### 6.3 Recommendations (~450)
Four, addressed to authors and reproducers: check fidelity against source, not
config; assert data contracts against physics, not shapes; differential-test
against a released artifact; publish the data convention alongside the config.

### 6.4 Communication with the original authors (~150)
Date contacted, questions asked, response received or not. **No response is a
complete and reportable answer.**

## 7. Limitations (~450 words)

TMLR expects this explicitly, and it is where honesty is cheapest and most
valuable.

- One seed per configuration; no seed-variance estimate.
- Ten epochs against the repository's 100 — the central negative result is
  bounded by budget, not by the method (§4.3).
- The wall effect's magnitude is regime-dependent (§5.2).
- Independent reimplementation, not a rerun of the authors' code: our
  deviations are documented but not zero.
- Planning results come from a checkpoint carrying seven deviations (§4.5).
- The action z-score broadcast is our interpretation of the reference; the
  exact broadcast is unverified.

## 8. Conclusion (~250 words)

No new claims. What reproduced, what did not, and the one sentence worth
carrying away: *the failure is in the predictor and its optimisation, not in the
representation.*

---

## Appendices — no page limit, use it

- **A. Full fidelity audit** — all 25 elements with source citations
  (`docs/fidelity_audit.md`).
- **B. Gate outputs** — G1 and pre-flight transcripts as run
  (`runs_archive/phase2_gates/`).
- **C. Per-run manifests and environment table** — `close_debts.py` section C.
- **D. Pre-registration** — the wall experiment's design and decision rule,
  committed before the episodes were evaluated.
- **E. Anatomised incidents** — the double-applied LR patch, the toy fixture,
  the action subsampling, the NaN trap. Short narratives; these are the
  reproducibility contribution and they do not belong in the body.
- **F. Encoder probe protocol** — 4,000 frames, ridge, 80/20 held out, MLP
  control (`runs_archive/phase2_analyses/encoder_probe.txt`).

---

## Before submission

- [ ] Every claim appears in its home section **once**; elsewhere referenced.
- [ ] Every number traces to a committed artifact.
- [ ] Scope paragraph in §1 and Limitations in §7 both present.
- [ ] Wall effect never quoted at +39.1 without the +12.7.
- [ ] Tier-3 provenance stated in §4.5 body, not a footnote.
- [ ] Author contact recorded in §6.4 whatever the outcome.
- [ ] Repository at a tagged commit; fresh-clone reproduction passes.

## Timeline

Writing 10–14 days → **submit to TMLR by ~8 August** → decision by ~30 September
is possible on TMLR's two-month target but not guaranteed. If it slips, the
paper is still a TMLR publication and MLRC 2027 is open.

**Check first:** MLRC requires the reproduced paper to be published in a listed
conference or journal. LeWM is currently an arXiv preprint. One email to
`reproducibility-chairs@neurips.cc` settles whether MLRC is available at all —
it does not affect TMLR.
