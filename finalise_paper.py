"""
finalise_paper.py -- carry out steps 4 and 5 on the assembled paper.

WHAT IT DOES
------------
  step 4  replaces the two paragraphs that predate the two-factor normalisation
          result, taking the replacements from the REVISION blocks in
          draft_section_6_and_revisions.md
  step 5  inserts Table 1 (end of §3.2), Table 2 (end of §4.1) and Table 3
          (in §5.3), taking each from its draft file
          and renumbers "Run 4" to "Run 3", since there is no Run 3

Everything is taken from the draft files rather than retyped, so the paper and
the drafts cannot diverge.

WHAT IT DELIBERATELY LEAVES TO YOU
----------------------------------
The role-based run renaming ("the released-config run" instead of "Run 0") is a
stylistic choice that needs different forms in prose and in table cells, and
correct articles and capitalisation. A script does it badly. This one performs
the minimal correction — Run 4 becomes Run 3, removing the implication of a
missing run — and then prints every remaining reference so you can do the
rename in an editor if you want it.

It is idempotent: applied twice, the second run reports that everything is
already in place and changes nothing. A backup is written before any change.

Usage:
    python3 finalise_paper.py --dir docs/paper
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


def extract(path: Path, start_pat: str, stop_pat: str, drop: list[str] | None = None):
    """Text between the line matching start_pat and the next line matching
    stop_pat, exclusive of both, with `drop` lines removed."""
    if not path.exists():
        return None
    lines = path.read_text(errors="ignore").splitlines()
    s = next((i for i, l in enumerate(lines) if re.search(start_pat, l)), None)
    if s is None:
        return None
    e = next((i for i in range(s + 1, len(lines))
              if re.search(stop_pat, lines[i])), len(lines))
    body = lines[s + 1:e]
    if drop:
        body = [l for l in body if not any(re.search(d, l) for d in drop)]
    return "\n".join(body).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/paper")
    ap.add_argument("--paper", default=None)
    args = ap.parse_args()

    d = Path(args.dir)
    paper = Path(args.paper or d / "PAPER.md")
    if not paper.exists():
        sys.exit(f"{paper} not found — run assemble_paper.py first")
    text = paper.read_text()
    original = text
    log = []

    # ---------------------------------------------------------------- step 4
    print("=" * 70)
    print("STEP 4 — the two revisions")
    print("=" * 70)

    rev_src = d / "draft_section_6_and_revisions.md"
    jobs = [
        ("4.3", r"^#\s*REVISION\s*—\s*§?4\.3",
         "The mechanism is in the checkpoints",
         r"^\*A position probe on the same embeddings"),
        ("5.1", r"^#\s*REVISION\s*—\s*§?5\.1",
         "The general lesson is that visual fidelity",
         r"^---\s*$"),
    ]
    for sec, start, anchor, stop in jobs:
        new = extract(rev_src, start, stop,
                      drop=[r"^\*Replaces", r"^\*\s*$"])
        if not new:
            print(f"  §{sec}: SKIP — revision block not found in "
                  f"{rev_src.name}")
            continue
        # Idempotency FIRST. The replacement text itself begins with the
        # anchor phrase, so a naive anchor search matches again on a second
        # run and would append the revision's later paragraphs a second time.
        # Compare on whitespace-normalised text so line wrapping cannot fool it.
        def flat(x):
            return " ".join(x.split())
        if flat(new) in flat(text):
            print(f"  §{sec}: already applied")
            continue

        # the paragraph to replace: from the anchor to the next blank line
        m = re.search(re.escape(anchor) + r".*?(?=\n\s*\n)", text, re.S)
        if not m:
            print(f"  §{sec}: SKIP — anchor \"{anchor[:38]}...\" not found "
                  f"in the paper")
            continue
        text = text[:m.start()] + new + text[m.end():]
        log.append(f"§{sec} revision applied ({len(new.split())} words)")
        print(f"  §{sec}: replaced {len(m.group(0).split())} words with "
              f"{len(new.split())}")

    # remove the assembler's markers
    n_marks = text.count("REVISION REQUIRED")
    text = re.sub(r"^<!--\s*=+\s*REVISION REQUIRED.*?-->\s*\n", "", text,
                  flags=re.M)
    if n_marks:
        print(f"  removed {n_marks} marker comment(s)")

    # ---------------------------------------------------------------- step 5
    print("\n" + "=" * 70)
    print("STEP 5 — the three tables")
    print("=" * 70)

    tables = [
        ("Table 1", d / "table_1_deviations.md",
         r"^##\s*Table 1a", r"^##\s*Notes for assembly",
         r"(?=^#{2,3}\s+3\.3\s)", "end of §3.2"),
        ("Table 2", d / "draft_section_4.1.md",
         r"^##\s*Table 2\s*—", r"^##\s+(?!Table 2)", r"(?=^#{2,3}\s+4\.2\s)",
         "end of §4.1"),
        ("Table 3", d / "draft_sections_4.5_5.3.md",
         r"^##\s*Table 3\s*—", r"^##\s+(?!Table 3)", r"(?=^#{2,3}\s+5\.4\s)",
         "end of §5.3"),
    ]
    for name, src, tstart, tstop, insert_before, where in tables:
        if f"**{name}:" in text or f"**{name} " in text:
            print(f"  {name}: already present")
            continue
        body = extract(src, tstart, tstop)
        if not body:
            print(f"  {name}: SKIP — not found in {src.name}")
            continue
        m = re.search(insert_before, text, re.M)
        if not m:
            print(f"  {name}: SKIP — could not locate the insertion point "
                  f"({where}); insert by hand")
            continue
        text = text[:m.start()] + body + "\n\n" + text[m.start():]
        log.append(f"{name} inserted at the {where}")
        print(f"  {name}: inserted at the {where} "
              f"({len(body.splitlines())} lines)")

    # ---- Run 4 -> Run 3 -------------------------------------------------
    n = len(re.findall(r"\bRun 4\b", text))
    if n:
        text = re.sub(r"\bRun 4\b", "Run 3", text)
        log.append(f"renumbered Run 4 -> Run 3 ({n} occurrences)")
        print(f"\n  renumbered Run 4 -> Run 3 in {n} place(s) — there is no "
              f"Run 3 otherwise")

    # ---------------------------------------------------------------- write
    if text == original:
        print("\nNo change — everything was already in place.")
        return
    bak = paper.with_suffix(".md.bak")
    shutil.copy(paper, bak)
    paper.write_text(text)
    print(f"\nwrote {paper}  (backup at {bak.name})")
    for item in log:
        print(f"  - {item}")

    # ---------------------------------------------------------------- verify
    print("\n" + "=" * 70)
    print("VERIFY")
    print("=" * 70)
    checks = [
        ("no revision markers remain", text.count("REVISION REQUIRED") == 0),
        ("§4.3 has the two-factor text",
         "two factors rather than one" in text),
        ("§5.1 has the extended closing", "We return to this in §6.3" in text),
        ("Table 1 present", "Table 1: Fidelity" in text),
        ("Table 2 present", "Table 2: Encoder comparison" in text),
        ("Table 3 present", "Table 3: One-step prediction error" in text),
        ("no Run 4 remains", not re.search(r"\bRun 4\b", text)),
    ]
    for label, ok in checks:
        print(f"  [{'x' if ok else ' '}] {label}")

    # ---------------------------------------------------------------- report
    runs = sorted({m.group(0) for m in re.finditer(r"\bRuns? [0-9](?:[–-][0-9])?",
                                                   text)})
    if runs:
        print("\n" + "=" * 70)
        print("OPTIONAL — role-based run naming")
        print("=" * 70)
        print(f"  {len(re.findall(r'Runs? [0-9]', text))} references remain, "
              f"as: {', '.join(runs)}")
        print("  If you want role names, the mapping is:")
        print("    Run 0 -> the released-config run     (tables: "
              "'released-config')")
        print("    Run 1 -> the first exploratory run   (tables: "
              "'exploratory 1')")
        print("    Run 2 -> the second exploratory run  (tables: "
              "'exploratory 2')")
        print("    Run 3 -> the corrected-pipeline run  (tables: 'corrected')")
        print("  Prose and table cells need different forms, and sentence-"
              "initial\n  uses need capitals, so this is an editor job rather "
              "than a script one.")

    print("\nNEXT:  python3 check_paper_consistency.py " + str(paper))


if __name__ == "__main__":
    main()
