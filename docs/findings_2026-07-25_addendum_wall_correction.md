# Findings addendum, July 25 2026 — correction: the wall claim was over-read

## The correction

The July 24 findings document states that the wall hypothesis was
"falsified" because cross-wall episodes succeeded 5/5 while same-room
episodes succeeded 29/43. **That conclusion was not supported by the
evidence and is withdrawn.** It rested on a cross-wall cell of five
episodes. Run through an exact Fisher test, 5/5 against 29/43 gives
p = 0.30 — entirely consistent with no difference, or with a difference in
either direction.

The goal-offset-100 run, whose episodes are 78% cross-wall instead of 10%,
points the other way:

    goal offset  25 : same-room 29/43 = 67.4%   cross-wall  5/5  = 100.0%   Fisher p = 0.30
    goal offset 100 : same-room  7/11 = 63.6%   cross-wall 17/39 =  43.6%   Fisher p = 0.31

Neither run is significant, each has one small cell, and their point
estimates contradict each other. Pooling them is not legitimate either,
since they are different tasks over different episode samples. **The honest
status of the wall question is OPEN.** No claim about the wall — in either
direction — goes in the paper on this evidence.

A second reason for caution: goals in the other room are usually farther
away, so a "wall effect" and a "distance effect" are confounded by
construction. At offset 100 the success rate falls with distance (62% / 56% /
28% across distance thirds), which is the competing explanation for the
cross-wall deficit, and these samples cannot separate the two.
`analyze_wall_controlled.py` runs the separation properly — geometry compared
within distance bands, pooled by Mantel-Haenszel, with a permutation test
that shuffles geometry labels inside each band so distance cannot leak into
the result. If it comes back inconclusive, a geometry-balanced evaluation
(equal numbers of same-room and cross-wall episodes, matched on goal
distance) is the experiment that would settle it, and it is free.

## What survives, and what is new

**Unaffected.** The headline planning results do not depend on the geometry
split at all: 72.0% vs 18.0% random at offset 25 (exact McNemar p = 7.4e-6)
and 48.0% vs 0.0% at offset 100 (p = 1.2e-7) are both intact.

**New and solid — the receding-1 planner is indistinguishable from random.**
The paired comparison against the random control on the same episodes gives
13/50 vs 9/50, discordant 11 vs 7, exact McNemar p = 0.48. Broken down,
same-room success under receding-1 is 7/43 = 16.3% and the random control's
is 7/43 = 16.3% — the same rate. So executing only the first action of each
plan does not merely degrade the planner; it removes the planning signal
entirely on the majority task. This is the strongest available support for
the terminal-cost explanation: if only the SUM of the action sequence is
identified by the optimizer, then executing one unidentified component of it
should perform like noise, and it does.

(The receding-1 cross-wall cell reads 4/5 vs 0/5, Fisher p = 0.048. Five
episodes. Given the correction above, this is recorded and not interpreted.)

**Refinement of the offset finding.** The drop from 70.8% (offset 25) to
48.0% (offset 100) is not a task-mix artifact. Holding the within-group
success rates at their offset-25 values and changing only the mix predicts
92.8%, not 48% — the mix shift alone would have raised the number. The drop
therefore lives in the within-group rates, and almost all of it is the
cross-wall rate falling from the unreliable 5/5 estimate to 17/39. Which is
one more reason to treat 5/5 as noise rather than signal.

## Standing claim, revised

Claim 6 in the ledger read: "real-environment planning 72% vs 18%, p=7.4e-6,
env proven bit-identical, deviations tabled; cross-wall 5/5, misses same-room
active runaways." The clause after the semicolon is withdrawn. The claim now
reads:

> Real-environment planning: 72.0% vs 18.0% random at goal offset 25
> (exact McNemar p = 7.4e-6) and 48.0% vs 0.0% at goal offset 100
> (p = 1.2e-7), in an environment proven bit-identical to the data
> generator, with all protocol deviations tabled. Whether the dividing wall
> costs the planner anything beyond goal distance is not resolved by these
> samples.
