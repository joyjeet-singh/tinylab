"""
add_header_and_fix_s8.py -- two last edits.

1. **§8's missing middle arm.** Line 1392 reads "...+39.1 points at
   p = 3.4 × 10⁻⁸ fell to −6.4 points on a different checkpoint." The estimate
   actually falls across three arms — +39.1, then +12.7 under a change of action
   scaling, then −6.4 on a different checkpoint — and naming only the endpoints
   implies a single jump rather than a monotone decline.

   An earlier script reported this as "already applied". It was not: the
   idempotency test searched the whole document and matched §5.2's line 970,
   which contains the same replacement phrasing. Idempotency tests that search
   globally can be satisfied by a different section.

2. **A header recording the four accepted checker findings**, so that neither a
   reviewer nor a future editor re-litigates them. Each names the section, the
   reason, and the condition under which it should be re-checked.

Both are idempotent and back up before writing.

Usage:
    python3 add_header_and_fix_s8.py --paper docs/paper/PAPER.md
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

HEADER = """<!-- Consistency checker: four pass-1 findings are reviewed and accepted.
     L593, L617  §4.2 — the subject is "the authors' released checkpoint",
                 named in the sentence that introduces the measurement.
     L715        §4.3 — a cell in the recalibration table; the eval-before
                 and eval-after columns make the context unambiguous.
     L1025       §5.3 — the sentence opens "At goal offset 25" and Table 3
                 sits directly above it naming all three checkpoints.
     L1291       §7  — the paragraph says "between checkpoints" and now
                 carries the offset.
     Re-check these if the surrounding text is rewritten. -->

"""

S8_OLD = "fell to −6.4 points on a different"
S8_NEW = ("fell to +12.7 points under a change of action scaling and to −6.4\n"
          "points on a different")


def flex(s: str) -> str:
    return r"\s+".join(re.escape(w) for w in s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="docs/paper/PAPER.md")
    args = ap.parse_args()
    paper = Path(args.paper)
    if not paper.exists():
        sys.exit(f"{paper} not found")
    text = paper.read_text()
    original = text

    # ---- 1. §8's middle arm ---------------------------------------------
    print("1. §8 — the middle arm")
    if re.search(flex("fell to +12.7 points under a change of action scaling"),
                 text):
        print("   already present")
    else:
        hits = list(re.finditer(flex(S8_OLD), text))
        if len(hits) == 1:
            h = hits[0]
            text = text[:h.start()] + S8_NEW + text[h.end():]
            print("   inserted — §8 now names all three arms")
        elif not hits:
            print("   NO MATCH — search for '39.1' and edit line ~1392 by hand")
        else:
            print(f"   {len(hits)} matches — too ambiguous; edit by hand")

    # ---- 2. the header ---------------------------------------------------
    print("\n2. accepted-findings header")
    if "Consistency checker: four pass-1 findings" in text:
        print("   already present")
    else:
        text = HEADER + text.lstrip("\n")
        print(f"   added ({len(HEADER.splitlines())} lines) at the top")

    if text == original:
        print("\nNo change.")
        return
    shutil.copy(paper, paper.with_suffix(".md.bak5"))
    paper.write_text(text)
    print(f"\nwrote {paper}  (backup at PAPER.md.bak5)")

    # ---- verify -----------------------------------------------------------
    print("\n" + "=" * 62)
    print("VERIFY")
    print("=" * 62)
    checks = [
        ("§8 names all three arms",
         bool(re.search(flex("+12.7 points under a change of action scaling "
                             "and to −6.4"), text))),
        ("header present",
         text.lstrip().startswith("<!-- Consistency checker")),
        ("§5.2 still names all three arms",
         bool(re.search(flex("falls to +12.7 points"), text))),
        ("§4.5 carries the goal offset",
         bool(re.search(flex("94.0% against 84.0% at goal offset 25"), text))),
    ]
    for label, ok in checks:
        print(f"  [{'x' if ok else ' '}] {label}")

    print("\nNOTE: the header cites line numbers, which will shift as the paper")
    print("is edited. It is a record of what was reviewed, not an index — the")
    print("section numbers and reasons are what matter.")
    print("\nNEXT:")
    print(f"  python3 check_paper_consistency.py {paper} 2>&1 | "
          f"sed -n '/^1\\./,/^2\\./p'")
    print("  Expect four findings, all matching the header. Then step 8: the")
    print("  17 [REF:] markers.")


if __name__ == "__main__":
    main()
