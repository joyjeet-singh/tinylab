# Findings, July 25 2026 — two free experiments: what the planner needs, and what the config conflict costs

Two follow-ups to the real-environment planner result (72% vs 18% random,
exact McNemar p = 7.4e-6). Both were free, both changed the paper.

## Experiment 1 — re-planning every step makes the planner WORSE, by a lot

Hypothesis going in: the 14 misses were caused by *open-loop commitment*.
Under the reference reading of `receding_horizon 5`, the planner commits five
planned actions — 25 of the 50-step budget — before it can correct. Ten of
the fourteen misses ended farther from the goal than they started, which
looks exactly like one bad plan being executed to completion. Prediction:
re-planning after every action block (`--receding 1`, ten re-plans instead of
two) should recover most of those episodes.

Result, same 50 episodes, same checkpoint, same everything else:

    receding 5 (baseline) : 36/50 = 72.0%   non-trivial 34/48 = 70.8%
    receding 1 (this run) : 13/50 = 26.0%   non-trivial 11/48 = 22.9%

Paired: both succeeded on 11, receding-5 alone on 25, receding-1 alone on 2,
neither on 12. Exact McNemar two-sided p = 5.7e-6. The hypothesis is
falsified, and not marginally — it is falsified in the opposite direction.
Committing to the whole plan is not this planner's failure mode; it is
load-bearing.

The failure signature also inverted. Under receding 5 the misses were few and
extreme (median final distance 64.1, range 23.9-80.7 — wrong-direction
travel). Under receding 1 the misses are many and uniform (median 43.8, range
24.4-56.9 — no catastrophic runaways, but almost no arrivals either). The
planner stopped going the wrong way fast and started going nowhere.

### Leading explanation (hypothesis, with the test that would confirm it)

The planner's cost is evaluated only at the END of the imagined horizon:
CEM scores a five-action sequence by how close the final imagined latent sits
to the goal latent. In this environment the dynamics are additive —
displacement is the action times a constant, so the endpoint depends
essentially on the SUM of the five actions, not on how that sum is
distributed across them. Every sequence with the right total is equally
optimal. That means the optimizer identifies the total, and leaves the
individual actions — including the first one — largely unconstrained.

Executing the whole sequence realizes the quantity that was actually
optimized. Executing only its first action realizes a quantity that was
never optimized, so each step is a noisy draw with only a fraction of the
intended drift; repeated ten times it becomes a weakly-directed random walk.
That predicts exactly what was measured: no runaways (no long committed
mistake) and few arrivals (no reliable progress).

Cheap confirmation, not yet run: dump the elite action sequences and compare
the goal-direction alignment of the first action against that of the summed
sequence, plus the elite-set variance per action slot. If the sum is aligned
and the first action is not, the explanation is confirmed.

### What this changes

The reference config sets `receding_horizon`, `action_block`, and `horizon`
all to 5, which is consistent with "plan five, execute five". Our reading was
therefore correct, and this experiment shows it is also the reading under
which the method works at all. The receding interpretation is REMOVED as a
candidate explanation for our gap to the published 87%; if anything it is the
favourable branch. What replaces it is a methodological finding worth
stating: terminal-cost planners on additive dynamics must execute the
sequence they optimized.

## Experiment 2 — the goal-offset conflict costs 24 points

The recorded config conflict: the repository's eval uses
`goal_offset_steps = 25`; the paper implies 100. Until now this was a
documented conflict of unknown consequence. Both arms re-run at offset 100
(independent episode sample — only episodes longer than 100 steps qualify, so
these are NOT the offset-25 episodes and the two settings cannot be compared
pairwise):

    goal offset 25  : CEM 72.0%   random 18.0%
    goal offset 100 : CEM 48.0%   random  0.0%

At offset 100 the goals sit a mean of ~118 units away (range 50-173) instead
of ~48, and the 50-step budget makes many of them barely reachable. Random
action selection succeeds zero times out of fifty; CEM succeeds 24 times.
Discordant 24 vs 0, exact McNemar two-sided p = 1.2e-7 — the cleanest
demonstration of planning signal in the project, since at this difficulty the
control has no accidental successes at all. Several individual episodes are
striking: a 169.8-unit goal reached in 31 steps, a 173.0-unit goal in 30.

The consequence for the paper is that the comparison to the published 87%
now depends on a setting the sources disagree about, and the dependence is
large: 72% under the repository's value, 48% under the paper's. Both rows go
in the deviations table, and no single-number comparison to 87% is quoted
without stating which offset it assumes.

New deviation to record: the offset-100 sample is drawn only from episodes
longer than 100 steps (dataset mean length 92.1), so it is a length-biased
subset, and its episodes differ from the offset-25 run's.

## Standing after both experiments

Solid, unchanged: the representation reproduces; the released config as
reimplemented does not converge; convergence is learning-rate-governed; the
checkpoint is a validated world model; the harness domain-gap anatomy; the
in-domain scoring geometry; the executable-gates process.

Solid, strengthened: real-environment planning, now at two difficulty
settings with controls (72/18 and 48/0), plus a mechanism result about how
the planner must be run.

Still needed: a second seed under the fixed one-way schedule to support
"this recipe converges" (currently one accident-assisted seed); and the
bounded calibration attempt — the authors' released TwoRoom checkpoint
through our harness — which would attribute the residual gap to checkpoint
quality if their weights score near 87 under this protocol.
