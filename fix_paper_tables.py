"""
fix_paper_tables.py -- repair four defects that came through assembly.

WHAT WENT WRONG
---------------
1. **Table 1 was never inserted.** `finalise_paper.py`'s "already present" test
   looked for the string "**Table 1 " and matched a *drafting instruction* in
   §3.2 — "**Table 1 note.** Generate from docs/fidelity_audit.md ..." — so it
   skipped the insertion. Its own verification caught this and printed
   `[ ] Table 1 present`; the two disagreed and the verification was right.

2. **That drafting instruction is in the paper.** It sits inside §3.2's body as
   a blockquote rather than under a "Drafting notes" heading, so the assembler
   had no way to recognise it. It must be removed.

3. **A stray heading sits inside §5.3.** The line
   "## Table 3 — caption and content" was neither a numbered section nor a
   recognised note heading, so the assembler appended it to §5.3's body along
   with the table it introduces. The table belongs; the heading does not.

4. **Both figure captions were dropped.** The assembler skips headings matching
   "Figure N", which correctly excluded the generation commands but also
   excluded the captions themselves. Figures 1 and 3 currently have no captions.

All four are repaired here, from the draft files rather than by retyping.
Idempotent; writes a backup.

Usage:
    python3 fix_paper_tables.py --dir docs/paper
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


def extract(path: Path, start_pat: str, stop_pat: str):
    if not path.exists():
        return None
    lines = path.read_text(errors="ignore").splitlines()
    s = next((i for i, l in enumerate(lines) if re.search(start_pat, l)), None)
    if s is None:
        return None
    e = next((i for i in range(s + 1, len(lines))
              if re.search(stop_pat, lines[i])), len(lines))
    return "\n".join(lines[s + 1:e]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/paper")
    args = ap.parse_args()
    d = Path(args.dir)
    paper = d / "PAPER.md"
    if not paper.exists():
        sys.exit(f"{paper} not found")
    text = paper.read_text()
    original = text

    # ---- 1. remove the drafting instruction from §3.2 -------------------
    print("1. drafting instruction in §3.2")
    note = re.search(r"\n>\s*\*\*Table 1 note\.\*\*.*?(?=\n\s*\n)", text, re.S)
    if note:
        text = text[:note.start()] + text[note.end():]
        print(f"   removed ({len(note.group(0).split())} words)")
    else:
        print("   not present — already removed or never assembled")

    # ---- 2. insert Table 1 at the end of §3.2 ---------------------------
    print("\n2. Table 1")
    if "Table 1: Fidelity" in text:
        print("   already present")
    else:
        body = extract(d / "table_1_deviations.md",
                       r"^##\s*Table 1a", r"^##\s*Notes for assembly")
        if not body:
            print("   FAILED — could not read table_1_deviations.md")
        else:
            m = re.search(r"(?=^#{2,3}\s+3\.3\s)", text, re.M)
            if not m:
                print("   FAILED — could not find §3.3 to insert before")
            else:
                text = text[:m.start()] + body + "\n\n" + text[m.start():]
                print(f"   inserted at the end of §3.2 "
                      f"({len(body.splitlines())} lines, "
                      f"{body.count('|')} table cells)")

    # ---- 3. remove the stray heading in §5.3 ----------------------------
    print("\n3. stray heading in §5.3")
    n = len(re.findall(r"^##\s*Table [23]\s*—\s*caption and content\s*$",
                       text, re.M))
    if n:
        text = re.sub(r"^##\s*Table [23]\s*—\s*caption and content\s*\n+", "",
                      text, flags=re.M)
        print(f"   removed {n} heading(s)")
    else:
        print("   none present")

    # ---- 4. figure captions ---------------------------------------------
    print("\n4. figure captions")
    figs = [
        ("Figure 1", d / "draft_section_4.1.md", r"^##\s*Figure 1\s*—",
         r"^##\s+(?!Figure 1)", r"(?=^#{2,3}\s+4\.2\s)", "end of §4.1"),
        ("Figure 3", d / "draft_sections_4.5_5.3.md", r"^##\s*Figure 3\s*—",
         r"^Generate with:", r"(?=^#{2,3}\s+5\.4\s)", "end of §5.3"),
    ]
    for name, src, fstart, fstop, before, where in figs:
        if f"**{name}:" in text:
            print(f"   {name}: already present")
            continue
        cap = extract(src, fstart, fstop)
        if not cap:
            print(f"   {name}: FAILED — not found in {src.name}")
            continue
        m = re.search(before, text, re.M)
        if not m:
            print(f"   {name}: FAILED — could not find the insertion point "
                  f"({where})")
            continue
        text = text[:m.start()] + cap + "\n\n" + text[m.start():]
        print(f"   {name}: inserted at the {where}")

    # ---- write and verify -----------------------------------------------
    if text == original:
        print("\nNo change — everything was already in place.")
        return
    shutil.copy(paper, paper.with_suffix(".md.bak2"))
    paper.write_text(text)
    print(f"\nwrote {paper}  (backup at PAPER.md.bak2)")

    print("\n" + "=" * 66)
    print("VERIFY")
    print("=" * 66)
    checks = [
        ("Table 1 present", "Table 1: Fidelity" in text),
        ("Table 1b present", "Table 1b:" in text),
        ("Table 2 present", "Table 2: Encoder comparison" in text),
        ("Table 3 present", "Table 3: One-step prediction error" in text),
        ("Figure 1 caption present", "**Figure 1:" in text),
        ("Figure 3 caption present", "**Figure 3:" in text),
        ("no drafting instruction", "Table 1 note" not in text),
        ("no stray caption heading",
         not re.search(r"^##\s*Table \d\s*—\s*caption", text, re.M)),
        ("no assembler markers", "REVISION REQUIRED" not in text),
    ]
    for label, ok in checks:
        print(f"  [{'x' if ok else ' '}] {label}")

    fails = [l for l, ok in checks if not ok]
    if fails:
        print(f"\n  {len(fails)} unresolved: {', '.join(fails)}")
        print("  Fix by hand before continuing.")
    else:
        print("\n  All clear. Next: re-run check_paper_consistency.py")

    # ---- residue scan ----------------------------------------------------
    print("\n" + "=" * 66)
    print("RESIDUE SCAN — drafting scaffolding that may have survived")
    print("=" * 66)
    pats = [(r"^\s*-\s*(?:Do not|Never|Keep|Cut|Verify|Check) ", "an imperative "
             "note to yourself"),
            (r"needs? the (?:exact |original|section numbers)", "a fill-in "
             "instruction"),
            (r"Generate with:", "a figure generation command"),
            (r"draft_\w+\.md|\.py\b", "a filename")]
    hits = 0
    for i, line in enumerate(text.splitlines(), 1):
        for p, why in pats:
            if re.search(p, line):
                hits += 1
                if hits <= 12:
                    print(f"  {i}: {line.strip()[:76]}")
                    print(f"      -> {why}")
                break
    if hits > 12:
        print(f"  ... and {hits - 12} more")
    if not hits:
        print("  none")
    else:
        print(f"\n  {hits} line(s) to review. Some are legitimate (a filename "
              f"in\n  §3.2's source citations); most are not.")


if __name__ == "__main__":
    main()
