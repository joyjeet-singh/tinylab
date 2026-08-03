"""
finish_wording_fixes.py -- the remaining edits, plus a fix to the checker.

THREE THINGS WENT WRONG LAST ROUND, WITH THREE DIFFERENT CAUSES
---------------------------------------------------------------
1. **Two fixes did not match.** My anchor strings were copied from the draft
   files and guessed where the ~78-character line wrap falls. Assembly preserved
   the drafts' wrapping, so any disagreement about a break position meant no
   match. This script matches on whitespace-flexible patterns instead: every run
   of spaces or newlines in the anchor matches any run of whitespace in the
   paper.

2. **§5.2's fix applied but the checker still flags it.** The replacement warns
   the reader ("a figure that survives neither of the two arms below") without
   naming +12.7 and -6.4, and the checker's rule requires the numbers. On the
   paper's most delicate claim, naming them is the better prose anyway, so the
   text is amended rather than the rule.

3. **The checker has a whitespace bug.** §1's contribution bullet reads "at the
   repository's evaluation goal\\noffset", and the rule looks for "goal offset".
   The phrase is split by a line wrap and never matches, so a correctly-written
   sentence is reported as a violation. Several pass-1 findings are this, not
   missing qualifiers. This script patches the checker to collapse whitespace
   in each paragraph before matching.

All edits are idempotent and back up before writing.

Usage:
    python3 finish_wording_fixes.py --paper docs/paper/PAPER.md
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


def flex(s: str) -> str:
    """A regex matching `s` with any whitespace where `s` has whitespace."""
    return r"\s+".join(re.escape(w) for w in s.split())


# (label, anchor, replacement, why)
FIXES = [
    ("§5.2 — name the two arms rather than gesturing at them",
     "a difference of 39.1 points at p = 3.4 × 10⁻⁸ — a figure that survives "
     "neither of the two arms below.",
     "a difference of 39.1 points at p = 3.4 × 10⁻⁸ — a figure that falls to "
     "+12.7 points under a change of action scaling and to −6.4 points on a "
     "different checkpoint, as the two rows below show.",
     "the paragraph is quotable on its own; the numbers must travel with it"),

    ("§8 — the middle arm",
     "fell to −6.4 points on a different checkpoint.",
     "fell to +12.7 points under a change of action scaling and to −6.4 points "
     "on a different checkpoint.",
     "naming only the endpoints implies a single jump rather than a monotone "
     "decline"),

    ("§8 — goal offset on the headline figure",
     "The planning claim reproduces, at 94.0% against a reported ~87%",
     "The planning claim reproduces, at 94.0% at goal offset 25 against a "
     "reported ~87%",
     "the same checkpoint reaches 20.0% at offset 100"),

    ("§4.2 — say whose checkpoint",
     "we therefore ran the **authors' released checkpoint** through our "
     "evaluation harness",
     "we therefore ran the **authors' released checkpoint** — not ours — "
     "through our evaluation harness",
     "a reader landing on the 84.0% should not have to look up whose it is"),

    ("§4.5 — goal offset in the qualifications paragraph",
     "the like-for-like comparison is 94.0% against 84.0% on identical "
     "episodes",
     "the like-for-like comparison is 94.0% against 84.0% at goal offset 25 "
     "on identical episodes",
     "the same checkpoint reaches 20.0% at offset 100"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="docs/paper/PAPER.md")
    ap.add_argument("--checker", default="check_paper_consistency.py")
    args = ap.parse_args()

    paper = Path(args.paper)
    if not paper.exists():
        sys.exit(f"{paper} not found")
    text = paper.read_text()
    original = text

    print("=" * 70)
    print("REMAINING WORDING FIXES (whitespace-tolerant)")
    print("=" * 70)
    for label, old, new, why in FIXES:
        if re.search(flex(new), text):
            print(f"  [-] {label}  (already applied)")
            continue
        pat = flex(old)
        hits = list(re.finditer(pat, text))
        if len(hits) == 1:
            text = text[:hits[0].start()] + new + text[hits[0].end():]
            print(f"  [x] {label}")
        elif not hits:
            print(f"\n  [ ] {label}  — no match")
            print(f"      looked for: {old[:70]}")
            print(f"      why: {why}")
            print(f"      -> search the paper for a distinctive fragment and "
                  f"edit by hand")
        else:
            print(f"\n  [ ] {label}  — {len(hits)} matches, too ambiguous")

    if text != original:
        shutil.copy(paper, paper.with_suffix(".md.bak4"))
        paper.write_text(text)
        print(f"\nwrote {paper}  (backup at PAPER.md.bak4)")
    else:
        print("\nNo change to the paper.")

    # ---- patch the checker's whitespace handling ------------------------
    print("\n" + "=" * 70)
    print("CHECKER — collapse whitespace before matching")
    print("=" * 70)
    ck = Path(args.checker)
    if not ck.exists():
        print(f"  {ck} not found — skipping")
    else:
        src = ck.read_text()
        if "flatten(para)" in src:
            print("  already patched")
        else:
            old_block = '''        for start, para in paragraphs(text):
            for rule in PAIRED:
                if not re.search(rule["trigger"], para, re.I):
                    continue
                if any(re.search(n, para, re.I) for n in rule["needs"]):
                    continue'''
            new_block = '''        for start, para in paragraphs(text):
            # Collapse whitespace first: a qualifier split across a line wrap
            # ("...evaluation goal\\noffset") would otherwise never match its
            # rule, and a correctly-written sentence would be reported as a
            # violation.
            flat = " ".join(para.split())
            for rule in PAIRED:
                if not re.search(rule["trigger"], flat, re.I):
                    continue
                if any(re.search(n, flat, re.I) for n in rule["needs"]):
                    continue'''
            if src.count(old_block) != 1:
                print("  could not locate the matching block — patch by hand:")
                print("    in pass 1, replace `para` with a whitespace-"
                      "collapsed copy")
            else:
                shutil.copy(ck, ck.with_suffix(".py.bak_ws"))
                ck.write_text(src.replace(old_block, new_block))
                print(f"  patched (backup at {ck.name}.bak_ws)")
                print("  pass 1 now matches across line wraps")

    print("\nNEXT:")
    print(f"  python3 {args.checker} {paper} 2>&1 | sed -n '/^1\\./,/^2\\./p'")
    print("\nWhat should remain in pass 1, and why each is acceptable:")
    print("  - the 84.0% flags in §4.2, §5.3, §7 — 'the authors'' appears in")
    print("    the preceding paragraph or a table column header")
    print("  - the 0.0085 flag — it is a cell in §4.3's recalibration table,")
    print("    where the before/after columns make the context unambiguous")
    print("Note each one in a comment when you finalise; do not chase them.")


if __name__ == "__main__":
    main()
