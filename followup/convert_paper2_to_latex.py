"""Convert followup/PAPER2.md to TMLR LaTeX, and build the arXiv package.

Reuses the decisions that worked for the reproduction paper, which are
documented in convert_to_latex.py at the repository root:

  - headings keep their manual numbers, secnumdepth -2. The paper refers to
    its own sections as plain text ("§7.4"); letting LaTeX renumber would
    break every reference silently.
  - table captions stay as bold paragraphs rather than floats, so nothing
    renumbers.
  - pandoc's captionless-longtable idiom needs the caption package and a
    counter called "none".
  - every non-ASCII character is converted rather than passed through, and
    the conversion asserts that none survives.

PAPER2 differs in being simpler: no figures, and a References section that
stays prose because the paper cites in running text ("arXiv:2608.10145")
rather than with \\cite.

    ./.venv/bin/python followup/convert_paper2_to_latex.py
"""
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

PAPER = Path("followup/PAPER2.md")
OUT = Path("build/paper2")
MAIN = "paper2"
PANDOC = os.environ.get("PANDOC_BIN") or shutil.which("pandoc") or "pandoc"
STYLE_SRC = Path("build")
STYLE = ["tmlr.sty", "tmlr.bst", "fancyhdr.sty"]

EMAIL = "joyjeetsingh1@gmail.com"
AFFIL = "Independent researcher \\\\ ORCID 0009-0005-1512-7439"

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[preprint]{tmlr}

\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{calc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{textcomp}
\usepackage{etoolbox}
\usepackage{caption}
% pandoc emits \def\LTcaptype{none} for captionless longtables; LaTeX and
% hyperref then look for a counter of that name. Our tables carry their
% captions as adjacent paragraphs, so nothing is numbered.
\newcounter{none}
\providecommand{\theHnone}{none.\arabic{none}}
\makeatletter
\patchcmd\longtable{\par}{\if@noskipsec\mbox{}\fi\par}{}{}
\makeatother

% The paper numbers its own sections and refers to them as plain text.
\setcounter{secnumdepth}{-2}

% Several tables are seven columns wide against a single-column measure.
\AtBeginEnvironment{longtable}{\footnotesize}
\setlength{\tabcolsep}{4pt}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}

\title{%TITLE%}

\author{\name %AUTHOR% \email %EMAIL% \\
      \addr %AFFIL%}

\begin{document}
\maketitle

\begin{abstract}
%ABSTRACT%
\end{abstract}

%BODY%

\end{document}
"""

SUPERSCRIPT = {"⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
               "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
UNICODE = [("χ²", r"$\chi^{2}$"), ("R²", r"$R^{2}$"), ("—", "---"), ("–", "--"),
           ("§", r"\S{}"), ("×", r"$\times$"), ("−", r"$-$"), ("·", r"$\cdot$"),
           ("→", r"$\rightarrow$"), ("λ", r"$\lambda$"), ("χ", r"$\chi$")]


def latexify(tex: str) -> str:
    for a, b in UNICODE:
        tex = tex.replace(a, b)
    tex = re.sub("[" + "".join(SUPERSCRIPT) + "]+",
                 lambda m: r"\textsuperscript{"
                           + "".join(SUPERSCRIPT[c] for c in m.group(0)) + "}",
                 tex)
    left = sorted({c for c in tex if ord(c) > 127})
    assert not left, ("characters pdflatex cannot set survive: "
                      + " ".join(f"U+{ord(c):04X} {c!r}" for c in left))
    return tex


def pandoc(md: str) -> str:
    p = subprocess.run([PANDOC, "--from=markdown+pipe_tables+tex_math_dollars+raw_tex",
                        "--to=latex", "--wrap=preserve"],
                       input=md, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"pandoc failed:\n{p.stderr}")
    return p.stdout


def main():
    if subprocess.run([PANDOC, "--version"], capture_output=True).returncode != 0:
        sys.exit(f"pandoc not usable at {PANDOC!r}; set PANDOC_BIN")
    text = PAPER.read_text()

    m = re.match(r"^---\ntitle: \|\n\s+(.+?)\nauthor: (.+?)\n---\n", text, re.S)
    assert m, "PAPER2.md has no title metadata block"
    title, author = m.group(1).strip(), m.group(2).strip()
    body_md = text[m.end():]

    am = re.search(r"^# Abstract\n(.*?)(?=^# )", body_md, re.S | re.M)
    assert am, "no Abstract section"
    abstract_md = am.group(1).strip()
    body_md = body_md[:am.start()] + body_md[am.end():]
    body_md = re.sub(r"^---\s*$", "", body_md, flags=re.M)

    print(f"title    : {title}")
    print(f"abstract : {len(abstract_md.split())} words")
    print(f"body     : {len(body_md.split())} words")

    abstract_tex = latexify(pandoc(abstract_md).strip())
    body_tex = latexify(pandoc(body_md).strip())
    body_tex = body_tex.replace("\\begin{quote}", "\\medskip\\noindent")
    body_tex = body_tex.replace("\\end{quote}", "\\medskip")
    assert "\\includegraphics" not in body_tex, "PAPER2 is expected to have no figures"
    print(f"tables   : {body_tex.count('begin{longtable}')}")
    print("unicode  : all converted")

    OUT.mkdir(parents=True, exist_ok=True)
    tex = (PREAMBLE.replace("%TITLE%", title).replace("%AUTHOR%", author)
           .replace("%EMAIL%", EMAIL).replace("%AFFIL%", AFFIL)
           .replace("%ABSTRACT%", abstract_tex).replace("%BODY%", body_tex))
    (OUT / f"{MAIN}.tex").write_text(tex)
    for f in STYLE:
        shutil.copy2(STYLE_SRC / f, OUT / f)
    print(f"wrote {OUT / MAIN}.tex plus style files")

    # ---- verify by compiling the package the way arXiv will --------------
    print("\nverifying with pdflatex alone, three passes, no bibtex:")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "pkg"
        shutil.copytree(OUT, work)
        for f in work.glob("*.txt"):
            f.unlink()
        for i in (1, 2, 3):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                                "-halt-on-error", f"{MAIN}.tex"],
                               cwd=work, capture_output=True, text=True)
            print(f"  pass {i}: exit {r.returncode}")
            if r.returncode != 0:
                print("\n".join(l for l in r.stdout.splitlines()
                                if l.startswith("!"))[:1500])
                sys.exit("the package does not build")
        log = (work / f"{MAIN}.log").read_text(errors="ignore")
        for pat, label in ((r"^! ", "errors"),
                           (r"Reference .* undefined", "undefined references"),
                           (r"File .* not found", "missing files")):
            hits = re.findall(pat, log, re.M)
            print(f"  {len(hits)} {label}")
            assert not hits, f"{label}: {hits[:3]}"
        pdf = work / f"{MAIN}.pdf"
        assert pdf.exists() and pdf.stat().st_size > 50_000
        pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
        print(f"  built {pdf.stat().st_size/1e6:.2f} MB PDF")
        shutil.copy2(pdf, Path("build") / "paper2_verified.pdf")

    tar = Path("build") / "arxiv_paper2.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        for f in sorted(OUT.iterdir()):
            if f.suffix in (".txt", ".pdf"):
                continue
            t.add(f, arcname=f.name)
    print(f"\npackage: {OUT}")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:<22} {f.stat().st_size/1024:8.1f} KB")
    print(f"tarball: {tar} ({tar.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
