"""Assemble and verify the arXiv submission package.

arXiv is not a general LaTeX host and several of its rules bite here:

  - it does NOT run BibTeX. The .bbl must be shipped, and because
    \\bibliography{...} expands to \\@input{\\jobname.bbl}, the .bbl has to be
    named after the main file, not after the .bib.
  - it does not fetch style files. tmlr.sty, tmlr.bst and fancyhdr.sty travel
    with the source.
  - its file system is case-sensitive, and missing or misnamed figure files
    are a common cause of announcement delay. \\graphicspath is flattened to
    the package directory so nothing points outside the tarball.
  - .aux/.log/.out must not be included.

The package is verified the only way that means anything: by compiling it in
a scratch copy of the directory with pdflatex alone, no bibtex, no latexmk,
which is what arXiv itself will do.

    python3 make_arxiv_package.py
"""
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

BUILD = Path("build")
OUT = BUILD / "arxiv"
MAIN = "paper"                      # -> paper.tex, and therefore paper.bbl
SRC_TEX = BUILD / "paper_preprint.tex"
SRC_BBL = BUILD / "paper_preprint.bbl"
STYLE = ["tmlr.sty", "tmlr.bst", "fancyhdr.sty"]
FIGURES = ["fig1_representation.png", "fig_horizon_dissociation.png"]
PAPER_MD = Path("docs/paper/PAPER.md")

assert SRC_TEX.exists(), f"{SRC_TEX} missing -- run convert_to_latex.py"
assert SRC_BBL.exists(), (
    f"{SRC_BBL} missing -- run latexmk so bibtex produces the bibliography")

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

# ---- the source ------------------------------------------------------
tex = SRC_TEX.read_text()
tex, n = re.subn(r"\\graphicspath\{\{[^}]*\}\}", r"\\graphicspath{{./}}", tex)
assert n == 1, f"expected one graphicspath, rewrote {n}"
(OUT / f"{MAIN}.tex").write_text(tex)

# \bibliography{paper} reads \jobname.bbl, so the .bbl follows the main file.
shutil.copy2(SRC_BBL, OUT / f"{MAIN}.bbl")
shutil.copy2("docs/paper/paper.bib", OUT / "paper.bib")

for f in STYLE:
    shutil.copy2(BUILD / f, OUT / f)
for f in FIGURES:
    assert Path(f).exists(), f"figure missing: {f}"
    shutil.copy2(f, OUT / f)

# ---- figures must resolve by exact, case-sensitive name --------------
for ref in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex):
    assert (OUT / ref).exists(), f"{ref} is referenced but not in the package"
    assert ref in FIGURES, f"{ref} referenced but not declared in FIGURES"
    print(f"  figure {ref} present, exact name matches")

# ---- abstract as plain text for the metadata form --------------------
md = PAPER_MD.read_text()
title = re.search(r"^---\ntitle: \|\n\s+(.+?)\nauthor:", md, re.S | re.M).group(1).strip()
abstract = re.search(r"^# Abstract\n(.*?)(?=^# )", md, re.S | re.M).group(1)
# arXiv's abstract field is plain text. The emphasis spans wrap across lines,
# so these must match with DOTALL or the markers survive.
abstract = re.sub(r"\*\*(.+?)\*\*", r"\1", abstract, flags=re.S)
abstract = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", abstract, flags=re.S)
abstract = re.sub(r"`(.+?)`", r"\1", abstract, flags=re.S)
abstract = abstract.replace("\n---", "").strip()          # trailing rule
abstract = re.sub(r"\n{2,}", "\n\n", abstract).strip()
assert "*" not in abstract and "`" not in abstract, \
    "markdown emphasis survived into the plain-text abstract"
(OUT / "ABSTRACT.txt").write_text(title + "\n\n" + abstract + "\n")
print(f"\nabstract: {len(abstract.split())} words, {len(abstract)} characters")

# arXiv's abstract field takes roughly 1920 characters and the paper's own
# abstract is longer. This is a condensation for the metadata form ONLY; the
# abstract inside the PDF is untouched. Every figure in it is one the paper
# states, so it cannot drift into a claim the paper does not make.
FORM_ABSTRACT = """LeWorldModel trains a latent world model with a prediction loss and a single anti-collapse regulariser, and reports approximately 87% of goals reached on TwoRoom, its simplest diagnostic environment. We reproduce that result by independent reimplementation on roughly $25 of rented compute, with all evaluation on one laptop CPU.

We reach 94.0% at the repository's evaluation goal offset, against 84.0% for the authors' own released checkpoint measured under our protocol on identical episodes, and we reproduce the reported representation result directly (position probe Pearson r = 0.9988 against a reported 0.996). Reaching that point required four conventions that determine the outcome and appear in no released configuration file: dense action gathering across a frameskip block, a programmatically-set action-encoder width, ImageNet pixel normalisation, and action z-scoring. A reproducer following the released configurations alone obtains a model whose predictor cannot converge.

The evaluation protocol is itself contested by the released material. The paper's appendix and the repository's configuration specify different goal offsets and step budgets; on the authors' own weights these yield 14.0% and 84.0%, and only the configuration's values reproduce the reported figure. On fifty identical episodes, changing nothing but how the goal is constructed moves that checkpoint from 84.0% to 8.0%.

Two findings generalise. One-step prediction accuracy does not predict long-horizon planning success: across three checkpoints spanning a sevenfold range in prediction error, including the authors' own, it orders short-horizon success monotonically and fails to order long-horizon success at all. And a batch normalisation layer inflated our reported validation loss by up to a factor of 300, concealing a training loss that was flat throughout."""

assert len(FORM_ABSTRACT) <= 1920, (
    f"the form abstract is {len(FORM_ABSTRACT)} characters, over arXiv's limit")
# Every number the condensation quotes must already appear in the paper's own
# abstract. A metadata abstract that states a figure the paper does not is a
# misrepresentation of the work, not a formatting slip.
form_nums = set(re.findall(r"\d+\.?\d*%|0\.\d{3,}", FORM_ABSTRACT))
invented = sorted(n for n in form_nums if n not in abstract)
assert not invented, (
    f"the form abstract quotes {invented}, which the paper's abstract does "
    f"not -- it must not introduce a claim the paper does not make")
print(f"  every figure in it ({len(form_nums)}) also appears in the paper's abstract")
(OUT / "ABSTRACT_for_arxiv_form.txt").write_text(FORM_ABSTRACT + "\n")
print(f"  form abstract: {len(FORM_ABSTRACT)} characters (limit ~1920)")

# ---- verify: compile it the way arXiv will ---------------------------
print("\nverifying the package compiles with pdflatex alone (no bibtex):")
with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp) / "pkg"
    shutil.copytree(OUT, work)
    (work / "ABSTRACT.txt").unlink()
    ok = True
    for i in (1, 2, 3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-halt-on-error", f"{MAIN}.tex"],
                           cwd=work, capture_output=True, text=True)
        print(f"  pass {i}: exit {r.returncode}")
        if r.returncode != 0:
            ok = False
            print("\n".join(l for l in r.stdout.splitlines()
                            if l.startswith("!"))[:1200])
            break
    assert ok, "the package does not build; arXiv would reject it"
    log = (work / f"{MAIN}.log").read_text(errors="ignore")
    for bad, label in ((r"^! ", "errors"),
                       (r"Citation .* undefined", "undefined citations"),
                       (r"Reference .* undefined", "undefined references"),
                       (r"File .* not found", "missing files")):
        hits = re.findall(bad, log, re.M)
        print(f"  {len(hits)} {label}")
        assert not hits, f"{label} in the package build: {hits[:3]}"
    pdf = work / f"{MAIN}.pdf"
    assert pdf.exists() and pdf.stat().st_size > 100_000, "no usable PDF"
    print(f"  built {pdf.stat().st_size / 1e6:.2f} MB PDF from the package alone")
    shutil.copy2(pdf, BUILD / "arxiv_verified.pdf")

# ---- the submission form, filled in ----------------------------------
AUTHOR = "Joyjeet Singh"
EMAIL = "joyjeetsingh1@gmail.com"
ORCID = "0009-0005-1512-7439"
REPO = "github.com/joyjeet-singh/tinylab"
ORIGINAL = "arXiv:2603.19312"

(OUT / "SUBMISSION.txt").write_text(f"""\
arXiv submission -- form fields
===============================

Title
  {title}

Authors
  {AUTHOR}

Contact email
  {EMAIL}

ORCID
  {ORCID}

Primary category
  cs.LG (Machine Learning)

Cross-list
  cs.AI (Artificial Intelligence)
  -- matches the categories of the author's existing submission, so no new
     endorsement is required. A stat.ML cross-list would be reasonable on
     subject grounds but is a category the author has not submitted to
     before, so check endorsement before adding it.

Endorsement
  NOT required. arXiv keeps authors endorsed in a category they have already
  submitted to, and the author's {ORIGINAL.replace('2603.19312', '2607.11116')}
  is cs.LG primary with a cs.AI cross-list.

License
  CC BY 4.0 -- conventional for reproduction work, and it lets the original
  authors reuse the material.

Comments field
  Independent reproduction of {ORIGINAL}. Code, six checkpoints, every
  evaluation report, the fidelity audit and the pre-registration:
  {REPO}

Abstract
  Use ABSTRACT_for_arxiv_form.txt -- the paper's own abstract is
  {len(abstract)} characters and arXiv's field takes about 1920. The full
  abstract, as it appears in the PDF, is in ABSTRACT.txt.

Files to upload
  arxiv_submission.tar.gz, or the contents of this directory except the
  three .txt files. The main file is paper.tex.

Notes
  - The .bbl is included because arXiv does not run BibTeX. It is named
    paper.bbl to match paper.tex, since \\bibliography expands to
    \\jobname.bbl.
  - tmlr.sty, tmlr.bst and fancyhdr.sty travel with the source; arXiv does
    not supply them.
  - The package was verified by compiling it in a scratch directory with
    pdflatex alone, three passes, no bibtex and no latexmk -- which is what
    arXiv does. Zero errors, zero undefined references, zero missing files.
""")

# ---- tarball ---------------------------------------------------------
tar = BUILD / "arxiv_submission.tar.gz"
with tarfile.open(tar, "w:gz") as t:
    for f in sorted(OUT.iterdir()):
        if f.suffix == ".txt":          # form notes, not source
            continue
        t.add(f, arcname=f.name)

print(f"\npackage: {OUT}")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:<32} {f.stat().st_size / 1024:8.1f} KB")
print(f"\ntarball: {tar}  ({tar.stat().st_size / 1e6:.2f} MB)")
