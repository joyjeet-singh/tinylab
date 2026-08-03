"""
apply_wording_fixes.py -- the six edits the consistency checker identified as
real, applied by exact match.

THE SIX
-------
  1. §5.2  +39.1 asserted without the two arms that follow it
  2. §8    +39.1 and -6.4 given without the middle arm (+12.7)
  3. §7    94.0% vs 84.0% without the goal offset (single-seed paragraph)
  4. §7    94.0% vs 84.0% without the goal offset (episode-selection paragraph)
  5. §4.2  "released checkpoint" without saying whose
  6. --    the assembler's HTML comment at the top of the file

A seventh, §2's unqualified "does not converge", is **not** applied here: it
needs the surrounding sentence read before it can be corrected, and the right
qualifier depends on what that sentence is about. The script prints the line and
its context so you can judge.

Each edit matches an exact string. If the paper has been edited by hand since
the checker ran, a match may fail — the script reports which and changes nothing
else. It is idempotent and writes a backup.

Usage:
    python3 apply_wording_fixes.py --paper docs/paper/PAPER.md
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# (label, what to find, what to replace it with, why)
FIXES = [
    ("§5.2 — +39.1 needs its two arms",
     "of cross-wall goals, a difference of 39.1 points at p = 3.4 × 10⁻⁸.",
     "of cross-wall goals, a difference of 39.1 points at p = 3.4 × 10⁻⁸ — a "
     "figure that survives neither of the two arms below.",
     "the paragraph asserts the registered effect alone; the table above it "
     "shows all three arms but a reader may quote the paragraph"),

    ("§8 — the middle arm is missing",
     "fell to −6.4 points on a different checkpoint.",
     "fell to +12.7 points under a change of action scaling and to −6.4 points "
     "on a different checkpoint.",
     "the estimate falls monotonically across three arms; naming only the "
     "endpoints implies a single jump"),

    ("§7 — goal offset, single-seed paragraph",
     "(94.0% against 84.0%, p = 0.0625;",
     "(94.0% against 84.0% at goal offset 25, p = 0.0625;",
     "the same checkpoint reaches 20.0% at offset 100"),

    ("§7 — goal offset, episode-selection paragraph",
     "our checkpoint against the authors' released\ncheckpoint on identical "
     "episodes (94.0% against 84.0%), not against the reported\n87%",
     "our checkpoint against the authors' released\ncheckpoint on identical "
     "episodes at goal offset 25 (94.0% against 84.0%), not\nagainst the "
     "reported 87%",
     "the same checkpoint reaches 20.0% at offset 100"),

    ("§4.2 — say whose checkpoint",
     "we therefore ran the **authors' released checkpoint** through our "
     "evaluation\nharness",
     "we therefore ran the **authors' released checkpoint** — not ours — "
     "through our\nevaluation harness",
     "a reader landing on the 84.0% directly should not have to look up whose "
     "it is"),

    ("assembler comment",
     "<!-- ASSEMBLED by assemble_paper.py. Do not edit the drafts after this "
     "point; edit this file. -->\n\n",
     "",
     "scaffolding, not paper"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="docs/paper/PAPER.md")
    args = ap.parse_args()

    paper = Path(args.paper)
    if not paper.exists():
        sys.exit(f"{paper} not found")
    text = paper.read_text()
    original = text

    applied, already, failed = [], [], []
    for label, old, new, why in FIXES:
        if new and new in text:
            already.append(label)
            continue
        n = text.count(old)
        if n == 1:
            text = text.replace(old, new)
            applied.append(label)
        elif n == 0:
            failed.append((label, old, why, "no match"))
        else:
            failed.append((label, old, why, f"{n} matches — too ambiguous"))

    print("=" * 70)
    print("WORDING FIXES")
    print("=" * 70)
    for l in applied:
        print(f"  [x] {l}")
    for l in already:
        print(f"  [-] {l}  (already applied)")
    for l, old, why, reason in failed:
        print(f"\n  [ ] {l}  — {reason}")
        print(f"      looked for: {old[:72].replace(chr(10), ' / ')}")
        print(f"      why it matters: {why}")

    if text != original:
        shutil.copy(paper, paper.with_suffix(".md.bak3"))
        paper.write_text(text)
        print(f"\nwrote {paper}  (backup at PAPER.md.bak3)")
    else:
        print("\nNo change.")

    # ---- the seventh, which needs a human -------------------------------
    print("\n" + "=" * 70)
    print("SEVENTH — §2's unqualified \"does not converge\" (not automated)")
    print("=" * 70)
    lines = text.splitlines()
    hits = [i for i, l in enumerate(lines)
            if re.search(r"does not converge", l)]
    if not hits:
        print("  no occurrence found — already fixed?")
    for i in hits:
        lo, hi = max(0, i - 3), min(len(lines), i + 2)
        print(f"\n  line {i+1}, with context:")
        for j in range(lo, hi):
            mark = ">>" if j == i else "  "
            print(f"  {mark} {lines[j][:82]}")
        print("\n     If the sentence is about OUR reimplementation, qualify "
              "it:")
        print("       '... does not converge within the paper's stated "
              "ten-epoch budget.'")
        print("     If it is about the METHOD, rewrite it — that claim is not "
              "supported")
        print("     anywhere in this paper (§2's own verdict is 'insufficient "
              "to specify")
        print("     a converging run').")

    print("\nNEXT:")
    print("  python3 check_paper_consistency.py " + str(paper) +
          " 2>&1 | sed -n '/^1\\./,/^2\\./p'")
    print("  The +39.1 entries and the 94.0% entries at the §7 lines should "
          "be gone.")


if __name__ == "__main__":
    main()
