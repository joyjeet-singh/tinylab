# Findings, July 25 2026 (III) — the wall costs the planner 39 points, and nothing else explains it

## The pre-registered result

Design, sample size, single primary test and decision rule were committed
(`docs/prereg_2026-07-25_wall_balanced_eval.md`, commit 1d095ef) and the
episode list was committed before the planner ran (`balanced_episodes.json`,
commit 41e526a). 110 matched pairs, each pair one same-room and one cross-wall
episode at the same goal distance, goal offset 100, budget 50, receding 5.

    same-room  87/110 = 79.1%
    cross-wall 44/110 = 40.0%
    discordant pairs: same-only 53, cross-only 10
    exact matched-pair two-sided p = 3.38e-08
    difference +39.1 points [95% CI +27.2, +51.0]

The effect is nearly double the 20-point difference the study was powered to
detect; the confidence interval's lower bound alone exceeds the design target.

Integrity: 220 episodes committed, 220 evaluated, exact match; domain guard
passed; worst within-pair goal-distance gap 0.35 units against a 6-unit
caliper, so distance was controlled about seventeen times more tightly than
the design required.

## Everything that could explain it away, and why none of it does

**Distance.** Controlled by construction, not by adjustment. Both arms have a
median goal distance of 124.8.

**Reachability.** The wall blocks completely — pushing straight into it from
x=100 at y=150 leaves the agent stuck at x=99.5 permanently — so cross-wall
goals require routing through a door that sits high in the arena while the
arena spans y 14 to 208. Since the goal is where the data-collection policy
stood 100 steps later and our budget is 50, some cross-wall goals could have
been geometrically impossible. They are not. Minimum steps required:

    same-room  median 22.2, 90th pct 29.9, max 32.8   reachable 110/110
    cross-wall median 26.4, 90th pct 33.6, max 44.7   reachable 110/110

**The ceiling, measured rather than computed.** A door-routing oracle using
true positions and no world model at all reaches **110/110 in both arms** under
the real dynamics and collision handling. So the planner's 79.1% and 40.0% are
both measured against a ceiling of 100%. The restricted analysis — matched
pairs where both sides are reachable — is the entire sample, and the difference
is unchanged at +39.1 points.

**Movement direction.** Because the half-arena is 97 wide but 196 tall, long
same-room goals are mostly vertical (94 of 110) while cross-wall goals are
mostly horizontal (73 of 110). The oracle clears both at 100%, so no left/right
versus up/down asymmetry is masquerading as a wall effect.

**Residual path length.** Cross-wall routes need about four more steps at the
median and have a tail that same-room never reaches. Worst-case bound: delete
every cross-wall episode above same-room's maximum requirement of 32.8 steps
and count all of them as failures caused by length alone. At 11 deletions the
gap is +34.7 points; at 20, +30.2; at 25 — a quarter of the arm — +27.3. The
effect cannot be produced by path length.

## What the result says

Same-room and cross-wall episodes in this experiment are equally far, equally
reachable, and equally solvable by a trivial waypoint controller. The only
difference is that a cross-wall goal requires a path that first moves *away*
from the goal — up to the door — before approaching it. Planning by scoring an
imagined terminal latent against the goal latent solves that 40% of the time,
against 79% in open space.

This is the behavioural counterpart to the in-domain measurement recorded on
July 24, where the encoder separated same-room from cross-wall latents at
matched distance by a factor of only 1.79. That measurement showed the wall is
represented weakly; this one shows what the weakness costs. It is the
"Beyond Euclidean Proximity" critique (arXiv 2605.22164) in quantified,
behavioural form — but stated as a property of *our* reproduction's
checkpoint, not of the method in general.

Scope, stated plainly: one checkpoint, one seed, goal offset 100, goal
distances up to about 203 units (beyond that the arena admits no same-room
counterpart, excluding 22 of 4,373 cross-wall candidates). The oracle
comparison establishes a ceiling; it is not a fair competitor, since it has
true positions and the planner has only images.

## Claim for the ledger

> With goal distance matched pair-by-pair and reachability verified at 100%
> for both geometries by a door-routing oracle, the planner reaches 79.1% of
> same-room goals and 40.0% of cross-wall goals (110 matched pairs, exact
> matched-pair p = 3.4e-8, difference +39.1 points, 95% CI +27.2 to +51.0).
> The dividing wall, not distance or reachability, accounts for the
> difference.

The July-24 claim that the wall hypothesis had been falsified stays withdrawn.
It was based on five episodes and pointed the wrong way.
