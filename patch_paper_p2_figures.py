"""P2 -- figure numbering, image files and body-text references.

Decision S2: use the figure images already on disk rather than regenerating.

Three defects:
  - the paper has a Figure 1 and a Figure 3 and no Figure 2;
  - Figure 3 is never referenced from the body text, only from its own caption;
  - neither caption is attached to an image file, so nothing resolves.

This renumbers Figure 3 to Figure 2, attaches each caption to the PNG on disk,
and adds the missing body-text reference. It asserts both image files exist
before writing anything -- a caption pointing at a missing file is the arXiv
failure the work order warns about (§8).

The image paths are written relative to docs/paper/, so they resolve when
PAPER.md is rendered from its own directory.
"""
import re
import shutil
import subprocess
from pathlib import Path

PAPER = Path("docs/paper/PAPER.md")

FIGURES = {
    "fig1_representation.png": "Figure 1",
    "fig_horizon_dissociation.png": "Figure 2 (renumbered from 3)",
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


# --------------------------------------------------- the files must exist
for fname, which in FIGURES.items():
    p = Path(fname)
    assert p.exists(), f"{which} has no image file at {p} -- refusing to write"
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)],
                             capture_output=True).returncode == 0
    assert tracked, f"{p} is not tracked; a released paper cannot cite it"
    print(f"  {which:32s} {p}  ({p.stat().st_size:,} bytes, tracked)")
print()

# ------------------------------------------------------------------- P2a
print("P2a -- attach Figure 1 to its image file")
patch(
    old="> **Figure 1: The representation carries what the predictor needs.**",
    new=("![Figure 1](../../fig1_representation.png)\n\n"
         "> **Figure 1: The representation carries what the predictor needs.**"),
)

# ------------------------------------------------------------------- P2b
print("P2b -- renumber Figure 3 to Figure 2 and attach its image file")
patch(
    old="> **Figure 3: Prediction accuracy orders short-horizon planning success and fails to order long-horizon planning success.**",
    new=("![Figure 2](../../fig_horizon_dissociation.png)\n\n"
         "> **Figure 2: Prediction accuracy orders short-horizon planning "
         "success and fails to order long-horizon planning success.**"),
)

# ------------------------------------------------------------------- P2c
print("P2c -- give Figure 2 the body-text reference it never had")
patch(
    old="At goal offset 25 the ordering is monotone: as one-step error falls from 0.830 to 0.410 to 0.116, success rises from 78.0% to 84.0% to 94.0%. Prediction accuracy behaves exactly as the proxy assumption expects.",
    new=("At goal offset 25 the ordering is monotone: as one-step error falls "
         "from 0.830 to 0.410 to 0.116, success rises from 78.0% to 84.0% to "
         "94.0% (Figure 2a, upper line). Prediction accuracy behaves exactly as "
         "the proxy assumption expects."),
)

# ------------------------------------------------------------------- P2d
print("P2d -- reference Figure 2b where the overshoot is described")
patch(
    old="mean of 111.1 units from the goal, while the two accurate checkpoints finish at 116.6 and 122.5 units — **farther away than random** —",
    new=("mean of 111.1 units from the goal, while the two accurate checkpoints "
         "finish at 116.6 and 122.5 units — **farther away than random** "
         "(Figure 2b) —"),
)

# ------------------------------------------------------- nothing dangles
text = PAPER.read_text()
refs = set(re.findall(r"Figure (\d)", text))
print(f"figure numbers now referenced or captioned: {sorted(refs)}")
assert refs == {"1", "2"}, f"unexpected figure numbering: {sorted(refs)}"
for n in ("1", "2"):
    assert len(re.findall(rf"\*\*Figure {n}:", text)) == 1, \
        f"Figure {n} does not have exactly one caption"
    body = len(re.findall(rf"\(Figure {n}[abc]?", text))
    assert body >= 1, f"Figure {n} is never referenced from the body text"
    print(f"  Figure {n}: 1 caption, {body} body reference(s)")
print("\nP2 complete.")
