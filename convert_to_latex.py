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

The References section is left as prose. The paper cites inline in plain text
("Maes et al., 2026a, Fig. 6"), not with \\cite, so paper.bib is NOT wired into
this build -- converting forty-odd inline citations to natbib is a separate
editing job on PAPER.md, not a conversion step.
"""
import re
import subprocess
import sys
from pathlib import Path

PAPER = Path("docs/paper/PAPER.md")
OUT = Path("build")
PANDOC = "pandoc"

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[%OPTION%]{tmlr}

\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{textcomp}
\usepackage{etoolbox}

% The paper numbers its own sections and refers to them as plain text
% ("§4.2", "Table 1b"). Let LaTeX typeset those numbers rather than invent
% its own, which would silently break every cross-reference.
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

\end{document}
"""

EMAIL = "joyjeetsingh1@gmail.com"
AFFIL = "Independent researcher \\\\ ORCID 0009-0005-1512-7439"


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
    if not subprocess.run(["which", PANDOC], capture_output=True).returncode == 0:
        sys.exit("pandoc is not installed")
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

    print(f"title    : {title}")
    print(f"author   : {author}")
    print(f"abstract : {len(abstract_md.split())} words")
    print(f"body     : {len(body_md.split())} words")

    abstract_tex = pandoc(abstract_md).strip()
    body_tex = pandoc(body_md).strip()

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
