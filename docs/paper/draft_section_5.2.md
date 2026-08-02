# Draft — §5.2 A pre-registered effect that did not survive a change of checkpoint

*(~780 words. Home of the wall claim.)*

---

## 5.2 A pre-registered effect that did not survive a change of checkpoint

A published critique of latent world models argues that scoring a plan by
Euclidean distance between latent states conflates latent proximity with
reachability, and predicts that a planner will fail disproportionately when a
goal requires moving *away* from it before approaching [REF:critique]. TwoRoom
provides a clean test: goals in the opposite room require routing through a
single door, and goals in the same room do not. We designed and pre-registered
an experiment to measure it, obtained a large and highly significant effect, and
then found that the effect does not survive either a change in how the model is
driven or a change of checkpoint. We report all three arms.

### The design

The obvious confound is distance: cross-wall goals are farther on average. We
therefore built a matched-pair design. From the 6,056 episodes long enough to
supply a goal at our longer offset, we performed one-to-one caliper matching on
start-to-goal distance, obtaining **110 matched pairs**. Matching quality far
exceeded the caliper: the median within-pair distance difference was 0.02 units
against a caliper of 6, the worst was 0.35, and the two arms' median distances
were identical at 124.8. A simulated power analysis on the matched-pair test
gave 82% power for the 20-point difference we assumed. The design, the decision
rule for every outcome, and the exact list of 220 episodes were committed to the
repository before any of them was evaluated.

Two further explanations were eliminated after the fact. First, reachability: a
door-routing oracle with access to true positions, planning no further ahead
than the real dynamics allow, reached the goal in **110 of 110 episodes in both
arms** within the step budget, so the achievable ceiling is 100% for both
geometries. Second, direction: same-room goals in this set are predominantly
vertical and cross-wall goals predominantly horizontal, but the oracle clears
both at 100%, so orientation cannot account for a difference. A residual
path-length effect was bounded analytically: deleting the quarter of cross-wall
episodes with the longest geometric requirement and counting every one of them
as a length-caused failure still leaves a 27-point difference.

### The three arms

| driving regime | same-room | cross-wall | difference | exact p |
|---|---|---|---|---|
| pre-correction checkpoint, as measured | 79.1% | 40.0% | **+39.1** [+27.2, +51.0] | 3.4 × 10⁻⁸ |
| pre-correction checkpoint, action-scale corrected | 74.5% | 61.8% | +12.7 [+0.5, +24.9] | 0.054 |
| corrected checkpoint | 14.5% | 20.9% | −6.4 [−16.4, +3.7] | 0.248 |

The first row is the pre-registered primary test and stands as registered: on
that checkpoint, with distance matched pair-by-pair and reachability verified at
100% for both geometries, the planner reached 79.1% of same-room goals and 40.0%
of cross-wall goals, a difference of 39.1 points at p = 3.4 × 10⁻⁸.

The second row applies a correction to the action scale supplied to the same
checkpoint (§5.4). The difference falls to +12.7 points and the test becomes
marginal. The third row uses the corrected checkpoint of §4.4. The point estimate
is now negative, and the test is not significant.

The estimate therefore falls monotonically across the three arms and crosses
zero. We are careful about what the third arm establishes. It **rules out** an
effect as large as the +20 points the study was designed to detect: the
confidence interval excludes it. It does **not** establish a reversal — at
p = 0.248 the honest reading is no detectable difference. And a confound must be
named: overall success across the three arms is 61.6%, 68.2% and 17.7%, so they
are not compared at matched performance, and the low rate in the third arm could
in principle mask a real difference. The random-action control reached 0 of 110
in both geometries in every arm; being at the floor, it is uninformative about
whether the episodes differ in intrinsic difficulty, and we do not use it as
evidence that they do not.

### What we take from it

We do not claim there is no geometric effect on this task, and we do not claim
the published critique is wrong. What our three arms support is narrower and, we
think, more useful: **a large, well-powered, pre-registered, confound-controlled
effect measured on one reproduction checkpoint survived neither a change in how
that checkpoint was driven nor a change of checkpoint.** Everything a careful
reader would ask of the first row — matching, power, a committed episode list, an
oracle ceiling, an analytic bound on the residual confound — was in place, and
none of it was sufficient to make the number a property of the method rather
than of the checkpoint.

The implication for reproduction work is direct. A reproduction that establishes
a mechanism-level effect on its own trained checkpoint has established it for
that checkpoint. Whether it generalises is a separate question requiring
separate checkpoints, and in our case the answer was no. We report the
pre-registered result because we pre-registered it, and we report the other two
arms because they are what the result turned out to mean.

---

## Drafting notes

- **Never quote +39.1 without the other two arms**, anywhere in the paper,
  including the abstract and the discussion.
- The design paragraph is not padding: the lesson lands only if the reader
  believes the first row was done properly. Keep the matching quality, the power
  gate, the pre-commitment and the oracle.
- The "does not establish a reversal" sentence must survive editing. p = 0.248.
- The floored random control is stated as a limitation of our own analysis
  script's original verdict text, which was too strong. Do not reinstate it.
- `[REF:critique]` is arXiv 2605.22164.
