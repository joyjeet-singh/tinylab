# Pre-registration — geometry-balanced wall experiment (July 25 2026)

**This document is committed BEFORE the evaluation runs.** So is the episode
list it describes. Everything below — the design, the sample size, the single
primary test, and what each possible outcome will be taken to mean — is fixed
in advance, so that no result can be reached by choosing an analysis after
seeing the data.

## The question

Does the dividing wall cost the planner anything **beyond** the fact that
goals in the other room are usually farther away?

## Why the existing evidence cannot answer it

| run | same-room | cross-wall | Fisher exact |
|---|---|---|---|
| goal offset 25 | 29/43 = 67.4% | 5/5 = 100.0% | p = 0.303 |
| goal offset 100 | 7/11 = 63.6% | 17/39 = 43.6% | p = 0.314 |

Neither is significant; the point estimates contradict each other; each has
one cell too small to carry a conclusion. A distance-stratified analysis over
both runs (`analyze_wall_controlled.py`) gave a Mantel-Haenszel odds ratio of
1.26 with a within-band permutation p of 0.558.

Crucially, that analysis also showed the obstacle is **not** confounding. At
goal offset 100 the two geometries already have near-identical goal-distance
distributions (rank z = +0.11; medians 113.8 same-room vs 108.3 cross-wall).
The obstacle is **sample size**: eleven same-room episodes cannot resolve a
20-point difference. An earlier claim in
`docs/findings_2026-07-24_realenv_planner_result.md` that the wall hypothesis
was falsified rested on the 5/5 cell and has been withdrawn.

## Design

- **Setting:** goal offset 100, budget 50 env steps, receding 5, CEM
  300/30/30/1.0 horizon 5 action-block 5 — identical to the completed
  offset-100 run in every respect except episode selection. Offset 100 is
  chosen because it is where the 20-point estimate came from and where both
  geometries are well populated. A replication at offset 25 is optional
  follow-up, not part of this registration.
- **Selection:** every episode longer than 100 steps is labelled same-room or
  cross-wall from its recorded start and goal positions, then cross-wall
  episodes are matched one-to-one with same-room episodes whose goal distance
  is within a 6-unit caliper. Distance is therefore held fixed **by
  construction**, not by statistical adjustment. Selection touches no model
  and no outcome.
- **Size:** 110 matched pairs (220 episodes, roughly 77 minutes of CEM plus
  seconds for the control). Simulation puts power at about 82% to detect the
  observed 20-point difference at alpha 0.05.
- **Gate:** `make_balanced_episode_set.py` prints GO or NO-GO from the power
  simulation for the number of pairs it actually achieved. If it says NO-GO,
  the experiment does not run at that size. Running underpowered would
  produce a third inconclusive result and spend the question.
- **Control:** the random-action arm runs on the identical episode set.
- **Guard:** the domain guard runs as a precondition inside the evaluation,
  as in every other evaluation in this project.

## Primary test (one, named in advance)

Matched-pair exact binomial test on the pairs where the two episodes
disagree, two-sided, alpha 0.05, implemented in `analyze_balanced_wall.py`.

## Secondary, reported but not decisive

Unpaired success rates with the difference and its 95% confidence interval;
the same split for the random control.

## Decision rule, fixed in advance

- **p < 0.05, same-room better:** the wall costs the planner something beyond
  distance. This becomes a stated claim, scoped to long-range goals at offset
  100.
- **p < 0.05, cross-wall better:** report it as a surprise and look for what
  else distinguishes cross-wall episodes.
- **p ≥ 0.05:** no effect of the size this study was built to detect. The
  paper will state that an effect as large as 20 points is ruled out and that
  **smaller effects are not**. The wall question stays open at that finer
  scale, and no claim is made in either direction.
- **Control shows the same gap:** the gap belongs to the task rather than to
  planning, and the two effect sizes are compared before anything is
  attributed to the planner.
- **Integrity checks fail** (evaluated episodes differ from the committed
  list, or the within-pair distance caliper was exceeded): no test is
  reported; the run is repaired and repeated.

## Known limits, stated now

- Episodes must exceed 100 steps to qualify, so the sample is length-biased
  relative to the dataset (mean episode length 92.1). This is the same
  deviation already recorded for the offset-100 run.
- The conclusion applies to the goal-distance range that matching actually
  covers. If the availability table shows a shortfall at the extremes, the
  claim narrows to the covered range and that range is quoted.
- One checkpoint, one seed. This measures our reproduction's planner, not
  the method in general.

## Order of operations

1. Commit this document and the three scripts.
2. Run `make_balanced_episode_set.py`; read the GO/NO-GO verdict.
3. If GO, commit `balanced_episodes.json` — the record that selection
   preceded results.
4. Apply `patch_r2_episodes.py`, run both arms against the committed list.
5. Run `analyze_balanced_wall.py`, record the outcome whatever it is.
