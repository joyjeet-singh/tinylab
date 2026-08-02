# Drafts — §5.4 and §5.5

Both are short. §5.4 reports an observation whose explanation we tested and
refuted; §5.5 reports a measurement that bears on a published critique and is
qualified by §5.2. Total ~600 words.

---

## 5.4 What the corrected pipeline does to the representation

*(~380 words)*

The pipeline corrections of §3.2 change the predictor's task substantially, and
it is natural to ask what they do to the representation the encoder learns. We
compared the two encoders — one trained under the released configuration, one
under the corrected pipeline — on 4,000 identical held-out frames, with each
encoder receiving the pixel convention it was trained with.

Three of the four measurements are unchanged. Position is decodable at R² 0.9977
against 0.9971 by a linear probe, and at 0.9995 against 0.9994 by a two-layer
network. The summed action executed between two frames is decodable from the
pair of embeddings at 0.9207 against 0.9132. On the information a planner needs,
the two encoders are equivalent.

The fourth measurement is not. The **effective rank** of the embedding cloud —
the participation ratio of its covariance spectrum, which counts how many
dimensions the representation actually occupies — rises from **11.9 to 67.8 of
192**. The corrected pipeline produces a representation spread over roughly
five and a half times more dimensions while carrying the same position and
action information.

We tested one explanation and it did not survive. Our hypothesis was that the
extra dimensions carry dynamics-relevant structure, purchased at the cost of
some of the linear structure that a position probe reads: under the released
configuration the predictor received one sub-sampled action to explain a
five-step displacement (§3.2), so it could not usefully constrain the encoder,
leaving it free to become a nearly pure position code. That hypothesis predicts
that the corrected encoder should make actions *more* linearly decodable. It
does not — action decodability is unchanged to within 0.008. We therefore report
the rank difference as an observation and offer no account of what the
additional dimensions encode.

We record one methodological point, because it changed our own answer. Both
measurements above were first taken before the normalisation repair of §4.3, and
both were wrong: the corrected encoder then appeared to have an effective rank of
16.5 rather than 67.8, and appeared to lose action decodability (0.8733 rather
than 0.9132). The position figures were unaffected, because a ridge probe
standardises its inputs and is scale-invariant. Any measurement on a latent space
whose scale a normalisation layer controls should be taken after verifying that
layer, and probe-style measurements are precisely the ones that will not warn
you.

---

## 5.5 Scoring geometry within the training distribution

*(~220 words)*

A published critique of latent world models argues that scoring plans by
Euclidean distance in latent space conflates latent proximity with reachability
[REF:critique]. §5.2 tests the behavioural prediction that follows from it.
Here we report the representation-level measurement, which is narrower and
points the other way.

Sampling pairs of real frames at matched physical distance and comparing their
latent separation, embeddings of positions in *different* rooms are separated by
**1.79 times** as much as embeddings of positions in the same room at the same
physical distance. The wall is represented: at equal Euclidean distance in the
arena, the latent space places cross-wall pairs farther apart, which is the
direction the scoring function would need in order to prefer routing. The strong
form of the critique — that the latent geometry is blind to the obstacle — is
therefore not supported for this encoder.

Two qualifications. This measurement is from a single checkpoint, taken before
the pipeline corrections, and was not repeated afterwards. And given §5.2, where
the behavioural effect this measurement was originally offered to explain did
not survive a change of checkpoint, we do not present the 1.79× figure as
support for any behavioural claim. It is a property of one encoder's latent
geometry, reported as such.

---

## Drafting notes

- §5.4's final paragraph is the transferable part; the rank number itself is
  minor. If the section must be cut, keep that paragraph and fold the numbers
  into a sentence.
- Do not let §5.4 imply the corrected encoder is *better*. On everything we can
  measure that a planner uses, the two are equivalent.
- §5.5's second qualification is load-bearing. Earlier drafts of this work used
  the 1.79× figure as the representational counterpart of the wall result; after
  §5.2 that framing is unavailable and must not survive into the final text.
- `[REF:critique]` is arXiv 2605.22164.
- If space is tight, §5.5 is the strongest candidate for the appendix — it is
  the only subsection of §5 that supports no claim in the abstract.
