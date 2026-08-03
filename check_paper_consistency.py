"""
check_paper_consistency.py -- run the mechanical consistency passes on the
assembled paper.

WHAT IT CHECKS
--------------
Four of the seven passes from the assembly checklist are mechanical and are
implemented here. The other three (that every number traces to a committed
artifact, that the contribution bullets still match their sections, and that the
[REF:] markers are filled) need a human and are only reported as reminders.

  1. PAIRED NUMBERS. Some figures are misleading alone and must appear with a
     companion in the same paragraph:
       - 94.0% must appear with a goal offset
       - +39.1 must appear with both +12.7 and -6.4
       - any planning percentage must appear with a goal offset
     This is the check that matters most: the whole point of §4.5 and §5.2 is
     that a single number misrepresents the result.

  2. CLAIM PLACEMENT. Each headline claim has one home section where it is
     stated in full. Elsewhere it should be referenced, not restated. The
     checker counts full statements per section and flags any claim stated in
     more than one.

  3. FORBIDDEN PHRASINGS. Formulations the evidence does not support, which
     earlier drafts of this work did contain:
       - "does not converge" without a qualifier
       - "the wall effect" asserted without its three arms
       - any claim that the original authors' training suffers the
         normalisation artifact
       - "reversed" applied to the §5.2 third arm (p = 0.248)

  4. UNRESOLVED MARKERS. [REF:...], TODO, XXX, placeholder text.

HOW TO READ THE OUTPUT
----------------------
Every finding prints its file, line and the offending text. A finding is not
automatically an error -- a paragraph may legitimately discuss 94.0% in a
context where the offset appears two sentences earlier, and the checker's
paragraph window may miss it. Read each one. The checker is a reminder system,
not a gate; it exits 0 always, and says so.

Usage:
    python3 check_paper_consistency.py paper.md
    python3 check_paper_consistency.py drafts/*.md          # before assembly
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. numbers that must not appear alone
# ---------------------------------------------------------------------------
PAIRED = [
    dict(name="94.0% without a goal offset",
         trigger=r"94\.0\s*%",
         needs=[r"offset\s*(?:of\s*)?25", r"goal offset", r"offset 25"],
         why="the same checkpoint reaches 20.0% at offset 100"),
    dict(name="+39.1 without the other two arms",
         trigger=r"\+?39\.1",
         needs=[r"12\.7"],
         why="the effect falls to +12.7 and then -6.4 (§5.2)"),
    dict(name="+39.1 without the third arm",
         trigger=r"\+?39\.1",
         needs=[r"-\s*6\.4|−\s*6\.4"],
         why="the third arm is negative and must accompany the first"),
    dict(name="84.0% without identifying whose checkpoint",
         trigger=r"84\.0\s*%",
         needs=[r"author", r"released checkpoint", r"their"],
         why="84.0% is the AUTHORS' checkpoint under our protocol"),
    dict(name="0.0085 without saying it is recalibrated",
         trigger=r"0\.0085",
         needs=[r"recalibrat", r"§?4\.3", r"repaired"],
         why="the un-recalibrated figure is 4.6034 and means nothing"),
]

# ---------------------------------------------------------------------------
# 2. one home section per claim
# ---------------------------------------------------------------------------
CLAIMS = [
    dict(name="harness validation (84.0%)", home="4.2", pat=r"42\s*/\s*50|84\.0\s*%"),
    dict(name="domain gap (61.03)", home="5.1", pat=r"61\.03"),
    dict(name="action deviation cost (25.59)", home="3.2", pat=r"25\.59"),
    dict(name="planning reproduction (94.0%)", home="4.5", pat=r"47\s*/\s*50|94\.0\s*%"),
    dict(name="horizon dissociation", home="5.3", pat=r"12\.0\s*%.*54\.0\s*%|54\.0\s*%.*12\.0\s*%"),
    dict(name="wall experiment", home="5.2", pat=r"79\.1\s*%|39\.1"),
    dict(name="normalisation artifact", home="4.3", pat=r"302\.7|1\.4585"),
    dict(name="representation (0.9977)", home="4.1", pat=r"0\.9977"),
    dict(name="effective rank", home="5.4", pat=r"67\.8"),
    dict(name="scoring geometry (1.79x)", home="5.5", pat=r"1\.79"),
]

# ---------------------------------------------------------------------------
# 3. phrasings the evidence does not support
# ---------------------------------------------------------------------------
FORBIDDEN = [
    (r"the method does not converge",
     "unsupported: our claim is about the released CONFIGURATION as "
     "implementable from released config files (§2)"),
    (r"does not converge(?!.{0,120}(configuration|reimplement|as released|"
     r"in ten epochs|within))",
     "qualify it: which thing, under what budget?"),
    (r"(?<!not )reversed(?!.{0,80}(not established|p\s*=\s*0\.248))",
     "§5.2's third arm is p = 0.248 — 'no detectable difference', not "
     "'reversed'"),
    (r"authors'?\s+(?:training|checkpoint).{0,60}artifact",
     "we make no claim their training suffers the artifact; theirs measured "
     "1.09x (§4.3)"),
    (r"proves?\s+that|demonstrates?\s+conclusively|clearly shows",
     "overclaiming; prefer 'shows', 'indicates', or state the test"),
    (r"significantly better(?!.{0,100}p\s*=)",
     "state the test — 94 vs 84 is p = 0.0625 and 94 vs 78 is p = 0.0574, "
     "neither established at n = 50"),
]

MARKERS = [(r"\[REF:[^\]]*\]", "unfilled reference"),
           (r"\bTODO\b|\bXXX\b|\bFIXME\b", "leftover marker"),
           (r"<[a-z][^>]{2,40}>", "placeholder in angle brackets")]

SECTION = re.compile(r"^#{1,4}\s*§?\s*(\d+(?:\.\d+)?)", re.M)


def paragraphs(text):
    """(start line, text) for each blank-line-separated block."""
    out, buf, start = [], [], 1
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip():
            if not buf:
                start = i
            buf.append(line)
        elif buf:
            out.append((start, "\n".join(buf)))
            buf = []
    if buf:
        out.append((start, "\n".join(buf)))
    return out


def section_of(text, line_no):
    """The most recent section heading at or before line_no."""
    cur = "?"
    for i, line in enumerate(text.splitlines(), 1):
        if i > line_no:
            break
        m = SECTION.match(line)
        if m:
            cur = m.group(1)
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--quiet-markers", action="store_true",
                    help="skip the [REF:] / TODO pass (noisy before assembly)")
    args = ap.parse_args()

    findings = 0

    # ---- pass 1: paired numbers --------------------------------------
    print("=" * 72)
    print("1. NUMBERS THAT MUST NOT APPEAR ALONE")
    print("=" * 72)
    for f in args.files:
        text = Path(f).read_text(errors="ignore")
        for start, para in paragraphs(text):
            # Collapse whitespace first: a qualifier split across a line wrap
            # ("...evaluation goal\noffset") would otherwise never match its
            # rule, and a correctly-written sentence would be reported as a
            # violation.
            flat = " ".join(para.split())
            for rule in PAIRED:
                if not re.search(rule["trigger"], flat, re.I):
                    continue
                if any(re.search(n, flat, re.I) for n in rule["needs"]):
                    continue
                findings += 1
                first = next(l for l in para.splitlines() if l.strip())
                print(f"\n  {Path(f).name}:{start}  {rule['name']}")
                print(f"    {first[:96]}")
                print(f"    -> {rule['why']}")
    if findings == 0:
        print("  none")

    # ---- pass 2: one home per claim ----------------------------------
    print("\n" + "=" * 72)
    print("2. CLAIMS STATED OUTSIDE THEIR HOME SECTION")
    print("=" * 72)
    hits = 0
    for f in args.files:
        text = Path(f).read_text(errors="ignore")
        for c in CLAIMS:
            secs = {}
            for i, line in enumerate(text.splitlines(), 1):
                if re.search(c["pat"], line, re.I):
                    secs.setdefault(section_of(text, i), []).append(i)
            elsewhere = {s: v for s, v in secs.items()
                         if s not in (c["home"], "?")}
            if elsewhere:
                hits += 1
                print(f"\n  {Path(f).name}  {c['name']}  (home §{c['home']})")
                for s, lines in sorted(elsewhere.items()):
                    print(f"    also in §{s} at line(s) "
                          f"{', '.join(map(str, lines[:6]))}")
                print("    -> reference it rather than restating, unless this "
                      "is a table or caption")
    if hits == 0:
        print("  none")
    findings += hits

    # ---- pass 3: forbidden phrasings ---------------------------------
    print("\n" + "=" * 72)
    print("3. PHRASINGS THE EVIDENCE DOES NOT SUPPORT")
    print("=" * 72)
    hits = 0
    for f in args.files:
        for i, line in enumerate(Path(f).read_text(errors="ignore").splitlines(), 1):
            if line.strip().startswith((">", "|", "- [")):
                continue          # quoted material, tables, checklists
            for pat, why in FORBIDDEN:
                m = re.search(pat, line, re.I)
                if m:
                    hits += 1
                    print(f"\n  {Path(f).name}:{i}  \"{m.group(0)[:60]}\"")
                    print(f"    {line.strip()[:96]}")
                    print(f"    -> {why}")
    if hits == 0:
        print("  none")
    findings += hits

    # ---- pass 4: unresolved markers ----------------------------------
    if not args.quiet_markers:
        print("\n" + "=" * 72)
        print("4. UNRESOLVED MARKERS")
        print("=" * 72)
        counts = {}
        for f in args.files:
            for i, line in enumerate(Path(f).read_text(errors="ignore").splitlines(), 1):
                for pat, why in MARKERS:
                    for m in re.finditer(pat, line):
                        counts.setdefault(why, []).append(
                            f"{Path(f).name}:{i} {m.group(0)[:40]}")
        for why, items in counts.items():
            print(f"\n  {why}: {len(items)}")
            for it in items[:8]:
                print(f"    {it}")
            if len(items) > 8:
                print(f"    ... and {len(items) - 8} more")
            findings += len(items)
        if not counts:
            print("  none")

    # ---- what this cannot check --------------------------------------
    print("\n" + "=" * 72)
    print("STILL NEEDS A HUMAN")
    print("=" * 72)
    print("  - every number traces to a committed report or archived output")
    print("  - §1's five contribution bullets still match §4.5, §3.2, §5.3,")
    print("    §4.3+§5.1 and §5.2 after any revision")
    print("  - the §4.3 and §5.1 revisions have been applied (the originals")
    print("    predate the two-factor normalisation result)")
    print("  - Tier-3 provenance is stated in §4.5's body, not a footnote")
    print(f"\n{findings} finding(s). None is automatically an error — read each "
          f"one.\nThis is a reminder system, not a gate; it always exits 0.")


if __name__ == "__main__":
    main()
