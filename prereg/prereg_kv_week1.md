# Pre-registration — Week 1: KV-cache compression, random-baseline feasibility probe

**Written:** before any run. Nothing in this file may be edited after the first
run starts. Corrections go in a dated amendment section at the bottom, never by
rewriting what's above.

**Author:** solo, independent.
**Status:** feasibility probe. This is NOT the study. It decides whether the
study is worth doing.

---

## 1. What this week is for

This week answers one question, and it is a question about *us*, not about the
field:

> Can we stand up NVIDIA's KVPress, reproduce a published number from its
> leaderboard within a stated tolerance, and measure a random-eviction baseline
> on the same footing — on the hardware we actually have?

That's it. It's a plumbing test. If the plumbing doesn't hold, no claim about
the literature is worth making, because every downstream number would inherit
the leak.

**What this week cannot do, and will not claim:**

- It cannot tell us whether KV-cache compression methods "beat random" in
  general. One model, one benchmark, one context length, a handful of methods.
- It cannot tell us anything about the ~30 methods we don't run.
- It cannot support any claim about the field's rigor, incentives, or whether
  published numbers are inflated.
- It cannot establish that a gap we measure is *caused* by anything. We are
  measuring, not explaining.

Any sentence in the write-up that starts "this shows the field..." is out of
scope by construction. If the probe succeeds, those questions become *askable*
in a later, larger study — which will need its own pre-registration.

## 2. Background — stated at the strength the evidence actually supports

Three things are documented in the literature. Stated carefully:

1. **Random eviction is already used as a baseline by some method papers.**
   Compactor benchmarks against random eviction on RULER alongside SnapKV, H2O,
   and PyramidKV. "Learning to Evict from Key-Value Cache" (Feb 2026) includes a
   random baseline in its attention-free category and puts all baselines on a
   common ranking framework at a uniform per-head, per-layer budget.
   **So: random-as-baseline is NOT novel. We are not the first to think of it.**

2. **One paper reports a learned scoring method failing to beat simple
   heuristics, with random often comparable** ("On the Limits of Learned
   Importance Scoring for KV Cache Compression", Jan 2026). The authors
   explicitly scope this to non-query-aware learned token-level importance
   scoring under fixed-budget compression, and caution it is not evidence about
   other compression paradigms. **We take that scoping at face value. It is a
   result about one family, not about the field.**

3. **At least two papers import competitors' numbers rather than re-running
   them** (one states results are "directly taken from" another paper;
   ForesightKV states the same for G-KV). **This is a documented practice in two
   papers. It is NOT evidence that the practice is widespread, and we will not
   say that it is.**

What is genuinely *not* established, as far as our searching found: a sweep of
random eviction against a large number of methods under one harness at matched
budgets, run by someone with no method of their own to promote. That is the
study this probe is a gate for. Our searching is not exhaustive and may have
missed prior work; if we find it later, we say so in an amendment and adjust.

**Framing discipline.** Prior framings in this project ran ahead of the
evidence and had to be narrowed on contact with a search. The claim above is
deliberately the smallest one the sources support. Before the larger study, we
re-run the novelty search and record the result, whatever it is.

## 3. Falsifiable predictions — written before looking

Recorded now so we can be scored later. Confidence is deliberate; being wrong
in writing is the point.

**P1 — Reproduction.** Running a KVPress-supported method on a KVPress-supported
benchmark at a stated compression ratio will land within **±2 accuracy points**
of the corresponding published leaderboard entry, using the same model and
dataset.
*Confidence: 60%.*
*Wrong if:* the gap exceeds 2 points. Then the harness (or our use of it) is not
faithful, and no downstream number is trustworthy until that's resolved.

**P2 — Random is runnable and lands below the methods.** Random eviction at a
matched budget will be implementable in the same harness and will score **below**
the best method we run at the same compression ratio.
*Confidence: 70%.*
*Wrong if:* random matches or beats the methods. That's the interesting
outcome — and, being one model on one benchmark, still only grounds "worth a
larger study", not "the methods don't work".

**P3 — The gap widens with compression.** The margin between the best method and
random will be **larger at aggressive compression (≈10–20% retention) than at
mild compression (≈50% retention)**.
*Confidence: 65%.*
*Wrong if:* the gap is flat or inverts. Would suggest the budget, not the
scoring rule, drives most of the behaviour at the ratios we test.

**P4 — Feasibility.** The whole probe fits in the compute available (free-tier
GPU, small model) without hitting memory or time limits.
*Confidence: 55%.*
*Wrong if:* it doesn't. Then the larger study needs re-scoping to smaller models
or shorter contexts before anything else.

*(A confidence is a prediction, not a defence. When one is wrong, it gets
recorded as wrong in the results table, not explained away.)*

## 4. Method

**Fixed before running — no changes once the first run starts:**

- **Harness:** NVIDIA KVPress, unmodified, at a pinned commit recorded in the
  manifest. We do not patch it. If we must, that's an amendment.
- **Model:** the smallest KVPress-supported model that also appears on the
  public leaderboard, so P1 has a published number to check against.
- **Benchmark:** one KVPress-supported long-context benchmark, matching a
  published leaderboard configuration.
- **Compression ratios:** three points — mild (~50% retention), moderate
  (~30%), aggressive (~10–20%). Exact values fixed at first run and recorded.
- **Methods:** random eviction + a small number of KVPress methods spanning
  distinct families, chosen for leaderboard coverage, not for expected outcome.
- **Seeds:** random eviction gets **5 seeds**; a single random draw is not a
  baseline. Deterministic methods get 1 run. Everything reports mean ± spread.
- **Budget rule:** uniform budget across heads and layers for every method
  including random, following the matched-comparison approach used by prior
  work, so the comparison is between scoring rules and not between budget
  allocation schemes.

**Rigor scaffold (reused, not reinvented):**

- One recipe card per configuration; nothing configured at the command line.
- Manifest per run before the run starts: harness commit, model ID, dataset ID
  and hash, ratio, seed, library versions. Conditions only — never results.
- One measurement per line, appended as it happens, never edited afterwards.
- A single scorekeeper program computes every reported number from the logs.
  Numbers are never typed by hand into the write-up.
- Predictions above are logged next to each run's outcome and scored honestly.

**Stopping rule.** The probe is done when P1–P4 have been scored, whatever they
say. We do not add methods, ratios, or benchmarks to chase a nicer result. If
something looks interesting mid-probe, it goes in the notes as a candidate for
the next pre-registration — not into this one.

## 5. What each outcome licenses

Written now, so the conclusion isn't chosen after seeing the data.

| Outcome | What we may conclude | What we may NOT conclude |
|---|---|---|
| P1 holds, P2 holds | Harness is faithful; the larger sweep is feasible. Proceed to pre-register it. | That methods beat random in general. One model, one benchmark. |
| P1 holds, P2 fails (random competitive) | We can reproduce, and on this one setting random is competitive with the methods we ran. Worth a larger, pre-registered sweep. | That KV-cache compression doesn't work. That the literature is wrong. Neither is supported by one setting. |
| P1 fails | Our setup does not reproduce a published number. Diagnose before anything else. | That the published number is wrong. The likeliest explanation is our own error, and we investigate ours first. |
| P4 fails | The study needs re-scoping to fit the hardware. | Anything about the methods. |

**Publication stance.** This probe is not a paper and will not be written as
one. It is a gate. A negative or boring result here is a successful week — it
saves months. If we later publish, the eventual claim gets its own
pre-registration and its own novelty search, and it will be scoped to exactly
what was run: named models, named benchmarks, named ratios, named methods.

## 6. Known limits of this probe — stated up front, not in a footnote later

- One model and one benchmark is not a field. Everything here is a single
  setting.
- Reproducing one leaderboard number does not validate the harness in general,
  only for that configuration.
- Random eviction is one baseline. Comparable-to-random does not mean a method
  is worthless; it means the scoring rule didn't help *at that budget, on that
  task*.
- We are not measuring throughput, latency, or memory. Accuracy at a stated
  budget only.
- Our novelty search is best-effort and may have missed prior work.

## 7. Amendments

*(Dated entries only. Never edit above this line once the first run has started.)*

---

*Nothing in this document should be read as a claim about the KV-cache
compression literature. It is a plan to find out whether we can measure
anything trustworthy at all.*
