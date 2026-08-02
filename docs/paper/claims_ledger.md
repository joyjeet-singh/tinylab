# Claims ledger — 1 August 2026, measurement complete

**All four paid runs are spent. All planned measurements are done.**

Status: **SOLID** = measured, no known confound. **QUALIFIED** = measured, with
a stated limitation. **WITHDRAWN** = claimed at some point, did not survive.

---

# Tier 1 — independent of anything we trained (6 claims, all SOLID)

**1. Our harness reproduces the published number.** The authors' released
checkpoint through our harness, weights the only change: **42/50 = 84.0%**
against ~87% (p = 0.53, CI [71.5, 91.7]).

**2. The evaluation environment is the data generator, bit-level.** Pixel MAE
0.00, action replay error 0.000 at one and forty steps, paired latent 0.01.

**3. The evaluation-domain gap.** One instrument reads **61.03** on a visually
near-identical 32-pixel fixture and **0.01** on the real environment. Rendering
style alone reproduces a below-random result; a probe at R² 0.99 was blind to it
for three paid runs.

**4. Four undocumented pipeline deviations**, traced to reference source, none
present in any config file: dense action gathering reshaped to
(T, frameskip × action_dim); the action-encoder width set programmatically to
10; ImageNet pixel normalisation; z-scored actions with NaN rows dropped.
Corroborated by the released `config.json` (width 10) and the released weights
(0.410 with ImageNet normalisation against 5.1 without).

**5. The action deviation's cost on real data.** Median 25.59 units against a
13.3-unit typical per-block move — the action supplied to the predictor was
wrong by roughly twice the movement it had to explain.

**6. The process contribution.** Eight anatomised incidents, including a
validation metric that disagreed with the training metric by a factor of 600 in
data we had recorded and never opened, and a diagnostic of our own that scored a
working model at 1.159 because it supplied displacement-mismatched actions.

---

# Tier 2 — training and representation (13 claims)

**7. An evaluation-mode normalisation artifact can hide convergence entirely.
SOLID. The paper's most transferable finding.** Scoring one checkpoint four ways
(eval/train mode × held-out/training clips) isolates it: the mode effect is
+1.15 and +4.59; the data effect is +0.011 and **−0.0004**. No generalisation
gap exists in either run.

**8. The artifact tracks the learning rate. SOLID.** r = +0.899 between log(LR)
and log(gap); the gap closes to exactly **1.00×** at 1e-7 where the weights stop
moving, and reopens as the rate climbs.

**9. Recalibration repairs it, and is a repair rather than a booster. SOLID.**

| | eval before | eval after | train mode | gap |
|---|---|---|---|---|
| Run 0 | 1.4585 | **0.3061** | 0.3076 (unchanged) | 4.7× → 1.0× |
| phase2 | 4.6034 | **0.0085** | 0.0151 (unchanged) | 302.7× → 0.6× |
| Run 2 (control, gap already 1.0×) | 0.1846 | 0.1845 | 0.1811 | 1.0× → 1.0× |

Run 0's recalibrated value lands on its independently-measured train-mode loss
of 0.3079. On the already-calibrated control the loss does not move.

**10. Planning outcomes are sensitive to running statistics even when the loss
is not. QUALIFIED.** On the control checkpoint, a running-variance change of
6×10⁻⁷ left the loss unchanged to four decimals but reshuffled 7 of 50 planning
episodes (72.0% → 78.0%, p = 0.453 — noise, but not identity).

**11. Validation-loss checkpoint selection selected for normalisation
calibration, not model quality. SOLID.** Run 2's `ckpt_best` is epoch 5 — the
single epoch where the gap is 1.00×.

**12. The corrected pipeline converges. SOLID.** phase2's training loss descends
monotonically 0.0412 → 0.0146 with no oscillation; recalibrated held-out loss
**0.0085** against Run 0's **0.3061** — 36× better.

**13. The action-aggregation deviation caused the apparent non-convergence.
SOLID.** Run 0's training loss sits flat at ~0.30, the noise floor from
supplying one sub-sampled action to explain a five-step displacement.

**14. The corrected model is a strong one-step world model. SOLID.** Against a
frozen-world baseline: **0.068** with true dense actions, **0.116** with the
displacement-matched constant-action encoding a planner can emit.

**15. The constant-action penalty is real and small. SOLID.** 0.068 → 0.116.

**16. Action z-scoring is worth 2.9×. SOLID.** 0.116 with, 0.337 without.

**17. The representation reproduces and is not the bottleneck. SOLID.**
Position R² **0.9977** linear, 0.9995 by MLP on 4,000 held-out frames; the
summed action decodable from a latent pair at **0.9207**. Scale-invariant, so
unaffected by the normalisation artifact — which is why the encoder looked
healthy while the predictor looked broken.

**18. The corrected pipeline yields a far higher-dimensional code. SOLID as an
observation, unexplained.** Effective rank **11.9 → 67.8 of 192**, while
position (0.9977 / 0.9971) and action (0.9207 / 0.9132) decoding are equal.

**19. SIGReg prevents collapse. SOLID.** Spread 0.797–1.039 and 0.830–0.960
across all configurations.

---

# Tier 3 — planning (5 claims)

**20. The reproduction reaches 94.0% at the reference goal offset. QUALIFIED.**
47/50, CI [83.8%, 97.9%], containing the published ~87% (p = 0.203). Against the
authors' checkpoint on identical episodes: 84.0%, 5 vs 0, **p = 0.0625**.
Against our pre-correction checkpoint (also recalibrated): 78.0%, 11 vs 3,
**p = 0.0574**. *Both higher, neither established at n = 50.* Our three failures
are a strict subset of the authors' eight.
*Qualifications: one seed; the 87% uses the authors' unpublished episode
selection; recalibration used training clips overlapping the planning episodes.*

**21. One-step accuracy orders short-horizon planning success and fails to
order long-horizon planning success. SOLID — the principal finding.**

| checkpoint | one-step err | @25 | @100 | mean final @100 | context |
|---|---|---|---|---|---|
| our pre-correction | 0.830 | 78.0% | **54.0%** | **40.5** | 1 |
| authors' released | 0.410 | 84.0% | **12.0%** | 122.5 | 3 |
| our corrected | 0.116 | **94.0%** | 20.0% | 116.6 | 3 |
| random control | — | 18.0% | 0.0% | 111.1 | — |

Monotone at offset 25 (78 → 84 → 94); not at offset 100 (54 → 12 → 20). Paired:
authors vs pre-correction p = 1.9×10⁻⁵; corrected vs pre-correction
p = 7.6×10⁻⁵; the two accurate checkpoints indistinguishable (p = 0.29).
**The two most accurate models finish farther from the goal than a random-action
control.** Holds for the authors' own weights, so it is not an artifact of our
reimplementation.
*Confound to state: the two overshooting checkpoints use 3 context frames, the
cautious one uses 1. Three points cannot separate accuracy from context length.*

**22. The pre-registered wall effect does not survive. QUALIFIED.**

| regime | same | cross | gap | p |
|---|---|---|---|---|
| pre-correction, as measured | 79.1% | 40.0% | **+39.1** [+27.2, +51.0] | 3.4×10⁻⁸ |
| pre-correction, action-scale 5 | 74.5% | 61.8% | +12.7 [+0.5, +24.9] | 0.054 |
| corrected checkpoint | 14.5% | 20.9% | −6.4 [−16.4, +3.7] | 0.248 |

The estimate falls monotonically and crosses zero. The third arm rules out an
effect as large as the pre-registered +20; it does not establish a reversal.
*Confound: overall success is 61.6%, 68.2% and 17.7%, so the arms are not
compared at matched performance.*
**The usable claim:** a large, highly significant, pre-registered effect on one
checkpoint survived neither a change in action scaling nor a change of
checkpoint.

**23. Terminal-cost planners must execute the sequence they optimise.
QUALIFIED.** Re-planning every action block makes the planner indistinguishable
from random (13/50 vs 9/50, p = 0.48). *Measured on the pre-correction
checkpoint only.*

**24. In-domain scoring geometry. QUALIFIED.** Rooms separate by 1.79× at
matched distance. *Pre-correction checkpoint only; given claim 22, do not
present as support for a behavioural wall effect.*

---

# Withdrawn

- "The released configuration does not converge in ten epochs." It does.
- "phase2's predictor never beats a frozen-world baseline." It beats it
  eightfold.
- "The model is not reading our actions." Our test supplied
  displacement-mismatched ones.
- "Position information is worse under the corrected pipeline." 0.003 under a
  common protocol.
- "Action decodability fell under the corrected pipeline." Equal after
  recalibration.
- "The wall effect's direction is consistent; only the magnitude varies." The
  direction does not survive.
- "94% is significantly better than our earlier checkpoint (p = 0.0074)." That
  comparison was confounded by recalibration; the controlled figure is
  p = 0.0574.

# Explicitly not claimed

- The pre-registered primary, as instrumented, is **NULL**.
- No claim the published 87% is wrong — our harness reproduces it.
- No claim the original authors' training suffered the normalisation artifact.
- No claim about the mechanism of the overshoot in claim 21.
- No embodied or zero-shot claims.

---

# What remains — writing only

**Drafted:** §3.2 (fidelity audit), §4.1 (representation), §4.5 (planning),
§5.1 (domain gap), §5.3 (horizon dissociation).

**To draft:** §1, §2, §3.1, §3.3–3.5, §4.2, §4.3, §4.4, §5.2 (wall), §5.4
(effective rank), §5.5, §6, §7 (limitations), §8, abstract.

§4.3 and §4.4 must be rewritten around the inverted result — they were drafted
when we believed training had failed.

**Figures:** Figure 1 (representation) rendered; Figure 3 (horizon dissociation)
scripted and tested. Domain-gap and wall figures exist as committed PNGs.

**Debts:** strip optimiser state before release (216 MB → ~72 MB); the
`ckpt_best.pt` epoch field reports n+1 for log-epoch n; improve the
recalibration script's reporting to show per-feature change in normalised units.

**Not sent:** the author email. Longest lead time of anything remaining, and now
more interesting to them — the artifact concerns an architecture their config
specifies, and their own checkpoint shows claim 21.
