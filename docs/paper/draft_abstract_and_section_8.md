# Drafts — Abstract and §8 Conclusion

Written last, from the finished body. Both are assembled from claims already
evidenced elsewhere; neither introduces a number that does not appear in a
section above.

---

# Abstract

*(~250 words)*

LeWorldModel trains a latent world model with a prediction loss and a single
anti-collapse regulariser, and reports approximately 87% of goals reached on
TwoRoom — its simplest diagnostic environment, where comparable methods report
97–100%. We reproduce that result by independent reimplementation, on four
rented GPU-runs costing about six dollars each, with all evaluation on one
laptop CPU.

We reach **94.0%** at the repository's evaluation goal offset, against **84.0%**
for the authors' own released checkpoint measured under our protocol on
identical episodes, and we reproduce the reported representation result directly
(position probe R² 0.9977). Reaching that point required correcting **four
conventions that determine the outcome and appear in no released configuration
file**: dense action gathering across a frameskip block, a programmatically-set
action-encoder width, ImageNet pixel normalisation, and action z-scoring. A
reproducer following the released configurations alone obtains a model whose
predictor cannot converge.

Two findings generalise beyond this reproduction. First, **one-step prediction
accuracy does not predict long-horizon planning success**: across three
checkpoints spanning a sevenfold range in prediction error — including the
authors' own — accuracy orders short-horizon success monotonically and fails to
order long-horizon success at all, where the two most accurate checkpoints
finish farther from the goal than a random-action policy. Second, **a batch
normalisation layer inflated our reported validation loss by up to three orders
of magnitude**, concealing for three training runs a training loss that was
descending monotonically; we give the two conditions under which this occurs and
a cheap check for it.

We also report a pre-registered mechanism-level result that did not survive a
change of checkpoint, and what we take from that.

---

# §8 Conclusion

*(~330 words)*

The three claims we set out to test (§2) resolve as follows. The representation
claim reproduces directly and easily. The planning claim reproduces, at 94.0%
against a reported ~87% and against 84.0% for the authors' own weights under our
protocol, once four undocumented conventions are corrected. The training claim
does not reproduce from the released configuration files alone, and does
reproduce once those conventions are supplied — which we take to be a
documentation gap rather than a defect in the method.

What we did not anticipate is how much of the work would consist of establishing
that our own measurements meant what we thought they meant. Three training runs
appeared not to converge because a normalisation layer's stored statistics, not
the model, determined the loss we were reporting. Three runs' worth of planning
results were produced in a debugging fixture that looks correct and sits
twenty-five times outside the training distribution. A carefully pre-registered
effect of +39.1 points at p = 3.4 × 10⁻⁸ fell to −6.4 points on a different
checkpoint. In each case a probe or a summary statistic read exactly as it
should have while the underlying quantity was wrong, and in each case the check
that would have caught it was cheap.

The finding we expect to be most useful outside this reproduction is the
negative one. A world model that predicts one step ahead seven times more
accurately than another was not detectably better at short-horizon planning and
was decisively worse at long-horizon planning — and this holds for the authors'
released checkpoint as well as for ours. Selecting a world model by held-out
prediction error is not a reliable way to select a world model for planning, and
on this task at the longer horizon it would have selected the worst of the three
available.

We release the reimplementation, all four checkpoints, every evaluation report,
the fidelity audit against reference source, the pre-registration, and the gate
outputs, at [REF:repo].

---

## Drafting notes

- The abstract's final short paragraph exists so §5.2 is not buried. If a
  reviewer reads only the abstract, they should know the pre-registered result
  did not survive.
- **94.0% appears with its goal offset in the abstract.** This is the one place
  the omission would do most damage.
- §8 introduces nothing new. Verify that before submission: every number in it
  appears in §4 or §5.
- The second paragraph of §8 is the paper's real subject. It is also the part
  most likely to be trimmed as "not results"; it should survive.
- Do not end on the release list. If the venue's template allows, move
  `[REF:repo]` into a footnote on the first page and end §8 on the paragraph
  above it.

---

# Assembly checklist

All sections are now drafted. Before assembly:

| section | draft file |
|---|---|
| Abstract, §8 | this file |
| §1 Introduction | `draft_section_1.md` |
| §2 Scope | `draft_section_2.md` |
| §3.1, §3.3–3.5 | `draft_sections_3.1_3.3_3.4_3.5.md` |
| §3.2 Fidelity audit | `draft_sections_3.2_5.1.md` |
| §4.1 Representation | `draft_section_4.1.md` |
| §4.2 Protocol validation | `draft_section_4.2.md` |
| §4.3, §4.4 | `draft_sections_4.3_4.4.md` (+ revision in `draft_section_6_and_revisions.md`) |
| §4.5, §5.3 | `draft_sections_4.5_5.3.md` |
| §5.1 Domain gap | `draft_sections_3.2_5.1.md` (+ revision in `draft_section_6_and_revisions.md`) |
| §5.2 Wall | `draft_section_5.2.md` |
| §5.4, §5.5 | `draft_sections_5.4_5.5.md` |
| §6 Discussion | `draft_section_6_and_revisions.md` |
| §7 Limitations | `draft_section_7_limitations.md` |

**Consistency passes to run on the assembled document:**

1. **Every claim appears in its home section once**, and is referenced
   elsewhere rather than restated (the mapping table in
   `paper_skeleton_tmlr.md`).
2. **94.0% never appears without its goal offset**, anywhere.
3. **+39.1 never appears without +12.7 and −6.4**, anywhere.
4. Tier-3 provenance — that the planning checkpoints are ours and carry the
   deviations of Table 1 — is stated in §4.5's body, not a footnote.
5. Every number traces to a committed report or archived output.
6. §1's five contribution bullets still match §4.5, §3.2, §5.3, §4.3+§5.1, §5.2.
7. Apply the §4.3 and §5.1 revisions from
   `draft_section_6_and_revisions.md` — the originals predate the two-factor
   result.

**Still to write:** Table 1 (the deviation set, from `docs/fidelity_audit.md`
plus `close_debts.py` section C), Table 2 (encoder comparison, in §4.1), Table 3
(horizon dissociation, in §5.3), and the figure captions, which exist in their
respective draft files.

**Still open:** the reply from the original authors (§6.4), and the three
`[REF:...]` markers for the original's own section numbers.
