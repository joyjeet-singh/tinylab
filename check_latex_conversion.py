"""Gate: nothing was lost converting PAPER.md to LaTeX.

The work order's §7 definition of done asks for the rendered text to be
compared against PAPER.md section by section. This does that mechanically,
which is stricter than reading it: every heading, every decimal number, every
table and both figures have to survive, and no character pdflatex cannot set
may remain.

Run after convert_to_latex.py:
    python3 check_latex_conversion.py
"""
import re
import sys
from pathlib import Path

MD = Path("docs/paper/PAPER.md")
TEX = Path("build/paper_preprint.tex")
LOG = Path("build/paper_preprint.log")

md = MD.read_text()
tex = TEX.read_text()
body = md.split("# Abstract", 1)[1].split("# References")[0]
fail = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        fail.append(label)


print("PAPER.md -> LaTeX conversion\n")

# --- headings ---------------------------------------------------------
md_heads = [h.strip() for h in re.findall(r"^#{1,4} (.+)$", md, re.M)]
md_heads = [h for h in md_heads if h not in ("Abstract", "References")]
tex_heads = re.findall(r"\\(?:sub){0,2}section\{(.+?)\}\\label", tex)
check("every heading survives", len(md_heads) == len(tex_heads),
      f"{len(md_heads)} in Markdown, {len(tex_heads)} in LaTeX")

# LaTeX escapes the text, so normalise before comparing the pairs.
def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

mismatch = [(a, b) for a, b in zip(md_heads, tex_heads) if norm(a) != norm(b)]
check("headings match in order and text", not mismatch,
      f"{len(mismatch)} differ" if mismatch else "")
for a, b in mismatch[:5]:
    print(f"        {a!r}\n     -> {b!r}")

# --- numbers ----------------------------------------------------------
nums = set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", body))
lost = sorted(n for n in nums if n not in tex)
check("every decimal number survives", not lost,
      f"{len(nums)} numbers, {len(lost)} lost" + (f" {lost[:8]}" if lost else ""))

# --- structure --------------------------------------------------------
md_tables = len(re.findall(r"\n\|[-: |]+\|\n", body))
check("every table survives", md_tables == tex.count("begin{longtable}"),
      f"{md_tables} in Markdown, {tex.count('begin{longtable}')} in LaTeX")
check("both figures present", tex.count("includegraphics") == 2,
      f"{tex.count('includegraphics')} found")
for f in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex):
    check(f"figure file {f} exists", Path(f).exists())

# --- citations --------------------------------------------------------
check("citations converted to natbib", tex.count("\\citep") == 13,
      f"{tex.count(chr(92) + 'citep')} \\citep commands")
check("no prose citation left", not re.search(r"\((?:Maes|Li) et al\., 20\d\d", tex))

# --- characters pdflatex cannot set -----------------------------------
non_ascii = sorted({c for c in tex if ord(c) > 127})
check("no unconvertible unicode", not non_ascii,
      " ".join(f"U+{ord(c):04X}" for c in non_ascii))

# --- the build itself -------------------------------------------------
if LOG.exists():
    log = LOG.read_text(errors="ignore")
    errs = re.findall(r"^! .*", log, re.M)
    check("LaTeX build has no errors", not errs, f"{len(errs)} found")
    undef = re.findall(r"Citation `([^']+)' .*undefined", log)
    check("no undefined citations", not undef, f"{len(set(undef))} undefined")
    check("no undefined references",
          "There were undefined references" not in log)
else:
    check("build log present", False, str(LOG))

# --- anonymity of the submission build -------------------------------
# TMLR rejects a non-anonymous submission without review, so this is a
# rejection risk rather than a cosmetic one.
IDENT = re.compile(r"joyjeet|tinylab|0009-0005-1512-7439|Singh", re.I)
print()
for art in ("build/paper_submission.tex", "build/paper_anon.bib"):
    p = Path(art)
    if not p.exists():
        check(f"{art} exists", False)
        continue
    hits = IDENT.findall(p.read_text())
    check(f"{art} is anonymous", not hits,
          f"{len(hits)} identifying fragment(s)" if hits else "")

pdf = Path("build/paper_submission.pdf")
if pdf.exists():
    raw = pdf.read_bytes()
    # the embedded text and the document metadata are both leak surfaces
    leaks = [w for w in (b"joyjeet", b"tinylab", b"0009-0005-1512-7439")
             if w in raw.lower()]
    check("submission PDF text carries no identity", not leaks,
          str([w.decode() for w in leaks]) if leaks else "")
    meta = re.search(rb"/Author\(([^)]*)\)", raw)
    check("submission PDF metadata has no author",
          not (meta and meta.group(1).strip()),
          meta.group(1).decode(errors="replace") if meta and meta.group(1).strip() else "")

print()
if fail:
    print(f"{len(fail)} check(s) failed:")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("conversion is faithful: nothing lost between PAPER.md and the PDF.")
