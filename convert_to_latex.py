"""Convert docs/paper/PAPER.md to TMLR LaTeX.

Two builds from one source, as the work order asks:

    build/paper_preprint.tex   \\usepackage[preprint]{tmlr}  -- de-anonymised,
                               no TMLR branding; this is the arXiv version
    build/paper_submission.tex \\usepackage{tmlr}            -- anonymous, for
                               an OpenReview submission later

DECISIONS, and why
------------------
Headings keep their manual numbers ("## 4.2 Two published ...") and LaTeX is
told not to number sections (secnumdepth -2). The paper cross-references
sections as plain text -- "§4.2" appears throughout -- and Table 1b is not a
number LaTeX can produce. Letting LaTeX renumber would silently break every
one of those references. Keeping the author's numbers as literal text cannot.

Table and figure captions stay as bold paragraphs rather than becoming
floats, for the same reason: floats renumber, and "Table 1b" would become
"Table 2", breaking the nine in-text references to Table 1. This is the
work order's sanctioned alternative -- an unnumbered float with a descriptive
caption -- taken to its simplest form.

Tables are set \\footnotesize and allowed to break across pages; several are
seven columns wide and would otherwise overflow TMLR's single-column measure.

Citations are converted to natbib here rather than in PAPER.md. The Markdown
cites inline in plain prose -- "(Maes et al., 2026a, Fig. 6)" -- which keeps
the source readable and keeps check_paper_numbers.py working on it. CITATIONS
below maps each such string to a \\citep against docs/paper/paper.bib, every
pattern is asserted to occur exactly as many times as expected, and the prose
References section is replaced by a real bibliography. Nothing about the
Markdown changes.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path("docs/paper/PAPER.md")
OUT = Path("build")
# Set PANDOC_BIN to use a pandoc that is not on PATH -- e.g. the official
# prebuilt macOS binary, which takes seconds to fetch where building it from
# source drags GHC in behind it.
PANDOC = os.environ.get("PANDOC_BIN") or shutil.which("pandoc") or "pandoc"

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[%OPTION%]{tmlr}

\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{calc}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{textcomp}
\usepackage{etoolbox}
% pandoc emits \def\LTcaptype{none} for its captionless longtables, which is
% the caption package's idiom; without it the build dies on "No counter
% 'none' defined".
\usepackage{caption}
% ...and \LTcaptype{none} then makes LaTeX and hyperref look for a counter
% called "none". Our tables carry their captions as adjacent paragraphs, so
% nothing is numbered; defining the counter satisfies both harmlessly.
\newcounter{none}
\providecommand{\theHnone}{none.\arabic{none}}
\makeatletter
% pandoc's own fix for longtable ordering after a \paragraph.
\patchcmd\longtable{\par}{\if@noskipsec\mbox{}\fi\par}{}{}
\makeatother

% The paper numbers its own sections and refers to them as plain text
% (section 4.2, "Table 1b"). Let LaTeX typeset those numbers rather than
% invent its own, which would silently break every cross-reference.
\setcounter{secnumdepth}{-2}

% Several tables are seven columns wide; TMLR is single-column at 10pt.
\AtBeginEnvironment{longtable}{\footnotesize}
\setlength{\tabcolsep}{4pt}

% pandoc emits these when it converts pipe tables and images.
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\graphicspath{{../}}

\title{%TITLE%}

\author{\name %AUTHOR% \email %EMAIL% \\
      \addr %AFFIL%}

\begin{document}
\maketitle

\begin{abstract}
%ABSTRACT%
\end{abstract}

%BODY%

% Every entry in paper.bib corresponds to one entry in the Markdown's
% References section; \nocite{*} keeps the ones the text does not cite
% inline (the code release, this repository, LeJEPA) in the list.
\nocite{*}
\bibliographystyle{tmlr}
\bibliography{paper}

\end{document}
"""

EMAIL = "joyjeetsingh1@gmail.com"
AFFIL = "Independent researcher \\\\ ORCID 0009-0005-1512-7439"

# Plain-text citation -> (natbib replacement, expected occurrences).
# Longest patterns first: "Maes et al., 2026a, Fig. 6" must be consumed before
# the bare "Maes et al., 2026a" can match its prefix.
CITATIONS = [
    ("(Maes et al., 2026a, Tab. 3, App. F.2)",
     r"\citep[Tab.~3, App.~F.2]{maes2026lewm}", 2),
    ("(Maes et al., 2026a, Tab. 3 caption)",
     r"\citep[Tab.~3 caption]{maes2026lewm}", 1),
    ("(Maes et al., 2026a, Fig. 6)", r"\citep[Fig.~6]{maes2026lewm}", 3),
    ("(Maes et al., 2026a, App. E)", r"\citep[App.~E]{maes2026lewm}", 1),
    ("(Maes et al., 2026a, §3.1)", r"\citep[\S3.1]{maes2026lewm}", 1),
    ("(Maes et al., 2026a, §4.2)", r"\citep[\S4.2]{maes2026lewm}", 1),
    ("(Maes et al., 2026b, `stable_worldmodel/data/buffer.py`)",
     r"\citep[\texttt{stable\_worldmodel/data/buffer.py}]{maes2026swm}", 1),
    ("(Maes et al., 2026a)", r"\citep{maes2026lewm}", 1),
    ("(Li et al., 2026)", r"\citep{li2026beyond}", 2),
]


def apply_citations(md: str) -> str:
    """Replace the prose citations with natbib, counting every one."""
    total = 0
    for plain, tex, expected in CITATIONS:
        found = md.count(plain)
        assert found == expected, (
            f"expected {expected} occurrence(s) of {plain!r}, found {found}. "
            f"The citation map is out of step with PAPER.md; fix it rather "
            f"than letting a citation silently stay as prose.")
        md = md.replace(plain, tex)
        total += found
        print(f"  {found} x  {plain[:52]:<54} -> {tex[:34]}")
    leftover = re.findall(r"\((?:Maes|Li|Balestriero)[^)]*\d{4}[^)]*\)", md)
    assert not leftover, f"citation-shaped text left unconverted: {leftover}"
    print(f"  {total} citations converted, none left as prose")
    return md


SUPERSCRIPT = {"⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
               "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}

# Applied in order, before the superscript runs are collapsed.
UNICODE = [
    ("χ²", r"$\chi^{2}$"),
    ("R²", r"$R^{2}$"),
    ("—", "---"),
    ("–", "--"),
    ("§", r"\S{}"),
    ("×", r"$\times$"),
    ("−", r"$-$"),
    ("·", r"$\cdot$"),
    ("→", r"$\rightarrow$"),
    ("λ", r"$\lambda$"),
    ("χ", r"$\chi$"),
]


def latexify_unicode(tex: str) -> str:
    """pdflatex cannot set these; convert rather than pass them through.

    Runs of superscript characters become one \\textsuperscript, so that
    "10⁻⁸" sets as 10^-8 and not as three separate raised glyphs.
    """
    for src, dst in UNICODE:
        tex = tex.replace(src, dst)
    tex = re.sub("[" + "".join(SUPERSCRIPT) + "]+",
                 lambda m: r"\textsuperscript{"
                           + "".join(SUPERSCRIPT[c] for c in m.group(0)) + "}",
                 tex)
    left = sorted({c for c in tex if ord(c) > 127})
    assert not left, ("characters pdflatex cannot set are still present: "
                      + " ".join(f"U+{ord(c):04X} {c!r}" for c in left))
    return tex


def pandoc(markdown: str, extra=()) -> str:
    """Markdown fragment -> LaTeX fragment (no preamble)."""
    p = subprocess.run(
        [PANDOC, "--from=markdown+pipe_tables+tex_math_dollars+raw_tex",
         "--to=latex", "--wrap=preserve", *extra],
        input=markdown, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"pandoc failed:\n{p.stderr}")
    return p.stdout


def main():
    if subprocess.run([PANDOC, "--version"], capture_output=True).returncode != 0:
        sys.exit(f"pandoc not usable at {PANDOC!r}; set PANDOC_BIN")
    print(f"pandoc   : {PANDOC}")
    OUT.mkdir(exist_ok=True)
    text = PAPER.read_text()

    # ---- metadata -----------------------------------------------------
    m = re.match(r"^---\ntitle: \|\n\s+(.+?)\nauthor: (.+?)\n---\n", text, re.S)
    assert m, "PAPER.md has no title metadata block"
    title, author = m.group(1).strip(), m.group(2).strip()
    body_md = text[m.end():]

    # ---- abstract -----------------------------------------------------
    am = re.search(r"^# Abstract\n(.*?)(?=^# )", body_md, re.S | re.M)
    assert am, "could not find the Abstract section"
    abstract_md = am.group(1).strip()
    body_md = body_md[:am.start()] + body_md[am.end():]

    # A horizontal rule right after the abstract is a Markdown separator, not
    # content; pandoc would turn it into a stray \rule across the page.
    body_md = re.sub(r"^---\s*$", "", body_md, flags=re.M)

    # ---- citations and bibliography -----------------------------------
    print("citations:")
    body_md = apply_citations(body_md)
    rm = re.search(r"^# References\n.*\Z", body_md, re.S | re.M)
    assert rm, "could not find the References section"
    refs_words = len(rm.group(0).split())
    body_md = body_md[:rm.start()]
    print(f"  prose References section ({refs_words} words) replaced by "
          f"\\bibliography{{paper}}")

    print(f"title    : {title}")
    print(f"author   : {author}")
    print(f"abstract : {len(abstract_md.split())} words")
    print(f"body     : {len(body_md.split())} words")

    abstract_tex = latexify_unicode(pandoc(abstract_md).strip())
    body_tex = latexify_unicode(pandoc(body_md).strip())
    print("unicode  : all non-ASCII converted for pdflatex")

    # pandoc wraps each image in a float with \pandocbounded and an
    # auto-caption taken from the alt text, which would render as
    # "Figure 1: Figure 1" and renumber independently of the real caption
    # paragraph that follows. Replace the whole block with a plain centred
    # graphic so the image stays next to its caption.
    def _figure(m):
        return ("\\begin{center}\n\\includegraphics[width=\\textwidth]{"
                + Path(m.group(1)).name + "}\n\\end{center}")

    body_tex, n_img = re.subn(
        r"\\begin\{figure\}\s*\n\\centering\s*\n"
        r"\\pandocbounded\{\\includegraphics\[[^\]]*\]\{([^}]+)\}\}\s*\n"
        r"\\caption\{[^}]*\}\s*\n\\end\{figure\}",
        _figure, body_tex)
    assert n_img == 2, f"expected to rewrite 2 figure blocks, rewrote {n_img}"
    assert "\\pandocbounded" not in body_tex, "a \\pandocbounded survived"
    for fname in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", body_tex):
        assert Path(fname).exists(), f"figure not found from repo root: {fname}"
        print(f"figure   : {fname} resolves from the repository root")

    # pandoc renders "> **Table 1: ...**" blockquotes as quote environments;
    # they are captions, so set them as their own paragraph instead.
    body_tex = body_tex.replace("\\begin{quote}", "\\medskip\\noindent")
    body_tex = body_tex.replace("\\end{quote}", "\\medskip")

    n_tab = body_tex.count("\\begin{longtable}")
    n_fig = body_tex.count("\\includegraphics")
    print(f"tables   : {n_tab}\nfigures  : {n_fig}")
    assert n_fig == 2, f"expected 2 figures, found {n_fig}"

    for name, option in (("paper_preprint", "preprint"),
                         ("paper_submission", "")):
        tex = (PREAMBLE
               .replace("%OPTION%", option)
               .replace("%TITLE%", title)
               .replace("%AUTHOR%", author)
               .replace("%EMAIL%", EMAIL)
               .replace("%AFFIL%", AFFIL)
               .replace("%ABSTRACT%", abstract_tex)
               .replace("%BODY%", body_tex))
        if not option:
            tex = tex.replace("\\usepackage[]{tmlr}", "\\usepackage{tmlr}")
        (OUT / f"{name}.tex").write_text(tex)
        print(f"wrote {OUT / name}.tex")


if __name__ == "__main__":
    main()
