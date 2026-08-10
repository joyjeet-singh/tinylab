"""P9 -- give Table 3's one-step ratios their provenance, and correct 0.830.

Until now the three one-step prediction errors in Table 3 and §5.3 were the
only load-bearing figures in the paper with no committed measurement behind
them. All three have now been re-derived on CPU from the checkpoints and the
released dataset, and the outputs are committed under runs_archive/verified/.

Two of the three reproduced the paper exactly. The third, the pre-correction
checkpoint, measures 0.829 where the paper prints 0.830.

Every number written below is read from the artifact in this script.
"""
import re
import shutil
import subprocess
from pathlib import Path

PAPER = Path("docs/paper/PAPER.md")
V = Path("runs_archive/verified")

SRC = {
    "pre-correction": (V / "check_a_step0_run2.txt",
                       r"step0_err\s+[\d.]+\s+real_step\s+[\d.]+\s+ratio\s+([\d.]+)"),
    "authors": (V / "repeat_encoding_authors.txt",
                r"repeat \(constant-action planner\)\s+[\d.]+\s+([\d.]+)"),
    "corrected": (V / "driving_spec_phase2_recal.txt",
                  r"repeat of MEAN \(displacement-matched\)\s+[\d.]+\s+([\d.]+)"),
}


def next_backup(path):
    n = 0
    while True:
        cand = Path(str(path) + (".bak" if n == 0 else f".bak{n}"))
        if not cand.exists():
            return cand
        n += 1


def patch(old, new, path=PAPER):
    text = path.read_text()
    pat = re.compile(r'[\s>]+'.join(re.escape(w) for w in old.split()))
    matches = list(pat.finditer(text))
    assert len(matches) == 1, (
        f"expected exactly 1 match, found {len(matches)} for: {old[:70]!r}")
    bak = next_backup(path)
    shutil.copy2(path, bak)
    start, end = matches[0].span()
    path.write_text(text[:start] + new + text[end:])
    subprocess.run(["diff", "-u", str(bak), str(path)])
    print(f"  patched; backup at {bak}\n")


vals = {}
print("one-step ratios, read from the committed measurements:")
for name, (p, rx) in SRC.items():
    assert p.exists(), f"missing evidence file: {p}"
    m = re.search(rx, p.read_text())
    assert m, f"could not read the ratio from {p}"
    vals[name] = m.group(1)
    print(f"  {name:<16} {vals[name]}   {p}")

pre, auth, corr = vals["pre-correction"], vals["authors"], vals["corrected"]
old_pre = "0.830"
assert pre != old_pre, "the paper's value already matches; nothing to correct"
print(f"\npaper prints {old_pre} for the pre-correction checkpoint; "
      f"the measurement says {pre}\n")

# --- the three places the pre-correction ratio appears as a one-step error.
# Line 600's "0.830-0.960" is an embedding-spread range and must not match:
# each pattern below is anchored on its surrounding one-step-error prose.
print("correcting the pre-correction ratio, 3 occurrences")
patch(
    old=f"as one-step error falls from {old_pre} to {auth} to {corr}",
    new=f"as one-step error falls from {pre} to {auth} to {corr}",
)
patch(
    old=f"one-step errors of {old_pre}, {auth} and",
    new=f"one-step errors of {pre}, {auth} and",
)
patch(
    old=f"| our pre-correction checkpoint | {old_pre} |",
    new=f"| our pre-correction checkpoint | {pre} |",
)

print("adding the provenance sentence to §5.3")
patch(
    old=("Table 3 gives the comparison. All three checkpoints are evaluated under one "
         "protocol on identical episodes; one-step error is reported relative to a "
         "frozen-world baseline, so a value below 1 means the model predicts better than "
         "assuming nothing moves."),
    new=("Table 3 gives the comparison. All three checkpoints are evaluated under one "
         "planning protocol on identical episodes; one-step error is reported relative "
         "to a frozen-world baseline, so a value below 1 means the model predicts better "
         "than assuming nothing moves. The one-step figures are separate in-domain "
         "measurements on real validation clips, each taken under the constant-action "
         "encoding a planner actually emits, and each committed: "
         f"`{SRC['corrected'][0]}` for the corrected checkpoint, "
         f"`{SRC['authors'][0]}` for the authors' released weights, and "
         f"`{SRC['pre-correction'][0]}` for the pre-correction checkpoint. They come "
         "from three different instruments at different sample draws, so the third "
         "decimal is not comparable across rows; the ordering, which is all the "
         "argument uses, is unaffected."),
)

s = PAPER.read_text()
print("verification:")
print(f"  '{old_pre}' remaining (should be 1, the embedding-spread range):",
      len(re.findall(re.escape(old_pre), s)))
print(f"  '{pre}' occurrences (should be 3):", len(re.findall(re.escape(pre), s)))
assert len(re.findall(re.escape(pre), s)) == 3
assert "0.830–0.960" in s, "the embedding-spread range was damaged"
print("  embedding-spread range 0.830-0.960 intact")
