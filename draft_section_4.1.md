# Draft — §4.1 The representation reproduces, and is not the bottleneck

*Home of claims 12 and 14. ~520 words as drafted. Figure 1 and Table 2 specified
below; generate with `figure_representation.py`.*

---

## 4.1 The representation reproduces, and is not the bottleneck

The original reports that a linear probe recovers agent position from the
learned embedding at approximately R² 0.996 [REF:C1]. We reproduce this. On
4,000 held-out frames drawn uniformly from the released dataset, a ridge probe
fitted on 80% and evaluated on the remaining 20% recovers position at **R²
0.9977**; a two-layer MLP probe on the same split reaches 0.9995, confirming the
linear probe is not limited by its own capacity.

The protocol matters more here than the number. Our own training logs report
per-epoch probe values ranging from 0.9922 to 0.9974 for the reference-faithful
run and from 0.9305 to 0.9525 for the corrected-pipeline run — an apparent
five-point difference between the two pipelines. Measured under a single
protocol on identical frames, that difference is **0.003** (0.9977 against
0.9946) and vanishes entirely under the non-linear probe (0.9995 against
0.9996). The in-training probe fits far fewer samples, where ridge
regularisation dominates at 192 dimensions. We report the common-protocol
numbers throughout and recommend that probe protocols be stated wherever probe
values are compared, including within a single paper's own logs.

A second measurement matters more for what follows. From a pair of consecutive
embeddings (z_t, z_{t+k}), the summed action executed between them is linearly
decodable at **R² 0.9207**; from their difference alone, at 0.8925. Together
with the position result this characterises the latent space precisely: it is a
near-linear encoding of agent position, and transitions within it carry the
action that produced them in linearly accessible form.

This has a direct consequence for the training results in §4.3 and §4.4. The
environment's dynamics are deterministic and, in position space, affine in the
action: displacement equals a fixed speed times the summed actions of the block
(§3.3). Because the embedding is a near-linear encoding of position, the
forward map a predictor must learn in latent space is approximately as simple as
the true dynamics, and every quantity it requires is present and linearly
accessible in its inputs. **The failure to converge documented below is
therefore not an information-theoretic limitation of the representation. It is a
property of the predictor and its optimisation.**

Finally, the regulariser does its job. Mean embedding spread remained within
0.830–0.960 across the reference-faithful run and 0.797–1.039 across the
corrected-pipeline run, with no monotone decline in either (Figure 1c). We
observed no representation collapse under any configuration we trained,
including at the reference learning rate where the prediction loss does not
settle. This supports the original's central architectural claim — that the
two-term objective is sufficient to prevent collapse without an exponential
moving average, a frozen encoder, or auxiliary supervision — independently of
whether the predictor converges.

---

## Figure 1 — caption

> **Figure 1: The representation carries what the predictor needs.**
> **(a)** Position decoded by a ridge probe against true position, in arena
> coordinates, for 800 held-out frames; the dividing wall and door are drawn for
> reference. Held-out R² = 0.9977. **(b)** Summed action decoded from a pair of
> consecutive embeddings against the true summed action; held-out R² = 0.9207.
> **(c)** Mean embedding spread per epoch for both training configurations; no
> monotone decline appears in either, and no run collapsed. Panels (a) and (b)
> use the reference-faithful checkpoint; the corrected-pipeline checkpoint gives
> 0.9946 and 0.8733 respectively (Table 2).

## Table 2 — caption and content

> **Table 2: Encoder comparison under a single probe protocol.** 4,000 held-out
> frames, identical for both models; each encoder receives the pixel convention
> it was trained with. Ridge probes, 80/20 split.

| measurement | reference-faithful (Run 0) | corrected pipeline (phase2) |
|---|---|---|
| position, linear probe | **0.9977** | 0.9946 |
| position, MLP probe | 0.9995 | 0.9996 |
| summed action from (z_t, z_{t+k}) | **0.9207** | 0.8733 |
| summed action from z_{t+k} − z_t | 0.8925 | 0.8502 |
| effective rank (of 192) | 11.9 | 16.5 |
| mean embedding spread | 1.000 | 0.930 |

*Effective rank is discussed in §5.4; it is listed here so the comparison is
presented once.*

---

## Drafting notes

- **Do not** restate the domain gap, the deviations, or the convergence result
  here. §4.1 establishes only that the representation is good and that the
  bottleneck lies elsewhere; §4.3 and §4.4 carry the failure.
- The recommendation about stating probe protocols is a small genuine
  contribution — our own logs disagreed with our own measurement by five points.
  Keep it to one sentence; expand it in §6.3 only if it earns the space.
- `[REF:C1]` needs the section number from the original paper where the probing
  result is reported.
- Figure 1c needs Run 0 and phase2 spread series from their logs; both are in
  `runs_archive/`.
