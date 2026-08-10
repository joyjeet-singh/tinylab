"""Give PAPER.md a title.

The paper had none -- it opened at "# Abstract". arXiv requires a title, the
model card's BibTeX entry needs it to match exactly, and any LaTeX conversion
needs \\title{}.

The title is added as a pandoc YAML metadata block rather than another `# `
heading, so it does not compete with the numbered sections in the document
outline and converts straight to \\title{} / \\author{}.

The title states the paper's central finding rather than naming the method:
the same released weights score 84.0% or 14.0% depending only on which of two
published evaluation protocols is followed, and the representation result
reproduces directly. "A reproduction of LeWorldModel" describes the activity;
it does not describe the result.
"""
import re
import shutil
import subprocess
from pathlib import Path

PAPER = Path("docs/paper/PAPER.md")

TITLE = ("The Evaluation Protocol Determines the Result: "
         "An Independent Reproduction of LeWorldModel on TwoRoom")
AUTHOR = "Joyjeet Singh"
ORCID = "0009-0005-1512-7439"

text = PAPER.read_text()
assert not text.startswith("---"), "PAPER.md already carries a metadata block"
assert text.startswith("# Abstract"), \
    f"expected the file to open at '# Abstract', got: {text[:40]!r}"

bak = Path(str(PAPER) + ".bak_title")
n = 0
while bak.exists():
    n += 1
    bak = Path(f"{PAPER}.bak_title{n}")
shutil.copy2(PAPER, bak)

block = (f"---\n"
         f"title: |\n"
         f"  {TITLE}\n"
         f"author: {AUTHOR}\n"
         f"---\n\n")
PAPER.write_text(block + text)

subprocess.run(["diff", "-u", str(bak), str(PAPER)])
print(f"patched; backup at {bak}\n")

# the title must not disturb the section outline or introduce a setext hazard
s = PAPER.read_text()
body = s.split("---\n\n", 1)[1]
print("title :", TITLE)
print("author:", AUTHOR, f"(ORCID {ORCID})")
print("words :", len(TITLE.split()))
print("top-level headings:", [l for l in body.split("\n") if re.match(r"^# ", l)][:3], "...")
print("setext hazards in body:", len(re.findall(r"(?<!\n)\n---\n", body)))
assert body.startswith("# Abstract"), "the body no longer opens at the abstract"
