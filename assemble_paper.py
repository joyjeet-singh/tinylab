"""
assemble_paper.py -- concatenate the section drafts into one ordered document.

WHAT IT DOES
------------
Each draft file contains one or more paper sections under headings like
`## 4.1 The representation reproduces ...`, together with material that is NOT
part of the paper: a `# Draft — ...` title, `## Drafting notes`, `# REVISION`
blocks, and an assembly checklist. This extracts only the numbered sections,
orders them by section number, and writes a single document.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
Two paragraphs need replacing by hand, because they predate the two-factor
normalisation result and a script cannot safely swap a paragraph:

  * §4.3's mechanism paragraph  (beginning "The mechanism is in the
    checkpoints")
  * §5.1's closing paragraph    (beginning "The general lesson is that visual
    fidelity")

Both replacements are in `draft_section_6_and_revisions.md` under `# REVISION`.
The assembled document marks each site with a visible banner so neither can be
missed.

It also does not touch the abstract's position, table placement, figure
references, or the `[REF:...]` markers.

Usage:
    python3 assemble_paper.py --dir docs/paper --out docs/paper/PAPER.md
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# `## 4.1 Title`, `## 4 Title`, `# Abstract`
SEC = re.compile(r"^(#{1,3})\s+(?:§\s*)?(\d+(?:\.\d+)?)\s+(.+?)\s*$")
ABSTRACT = re.compile(r"^#{1,3}\s+Abstract\s*$", re.I)
# headings that introduce non-paper material
SKIP = re.compile(r"^#{1,3}\s+(Drafting notes|REVISION|Assembly|Notes for "
                  r"assembly|Table 1[ab]?\b|Figure \d|Drafts? —|Draft —)",
                  re.I)

REVISION_SITES = {
    "4.3": ("The mechanism is in the checkpoints",
            "replace this paragraph with the two-factor version in "
            "draft_section_6_and_revisions.md"),
    "5.1": ("The general lesson is that visual fidelity",
            "replace this paragraph with the extended version in "
            "draft_section_6_and_revisions.md"),
}


def blocks(path: Path):
    """Yield (kind, number, title, body). kind is 'abstract' or 'section'."""
    lines = path.read_text(errors="ignore").splitlines()
    cur = None
    buf = []
    for line in lines:
        if SKIP.match(line):
            if cur:
                yield (*cur, "\n".join(buf).strip())
                cur, buf = None, []
            continue
        m = SEC.match(line)
        a = ABSTRACT.match(line)
        if m or a:
            if cur:
                yield (*cur, "\n".join(buf).strip())
            cur = ("abstract", "0", "Abstract") if a else \
                  ("section", m.group(2), m.group(3))
            buf = []
            continue
        if cur:
            buf.append(line)
    if cur:
        yield (*cur, "\n".join(buf).strip())


def sort_key(num: str):
    parts = num.split(".")
    return (int(parts[0]), int(parts[1]) if len(parts) > 1 else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/paper")
    ap.add_argument("--out", default="docs/paper/PAPER.md")
    ap.add_argument("--exclude", nargs="*",
                    default=["claims_ledger.md", "paper_skeleton_tmlr.md",
                             "table_1_deviations.md"],
                    help="files that are not section drafts")
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(f for f in d.glob("*.md")
                   if f.name not in args.exclude and f.name != Path(args.out).name)
    print(f"reading {len(files)} draft files from {d}")

    found = {}
    for f in files:
        for kind, num, title, body in blocks(f):
            if not body:
                continue
            if num in found:
                print(f"  DUPLICATE §{num} in {f.name} and "
                      f"{found[num]['file']} — keeping the longer one")
                if len(body) <= len(found[num]["body"]):
                    continue
            found[num] = dict(kind=kind, title=title, body=body, file=f.name)
            print(f"  §{num:<5} {title[:52]:<54} ({f.name}, "
                  f"{len(body.split())} words)")

    if not found:
        raise SystemExit("no numbered sections found — check --dir")

    order = sorted(found, key=sort_key)
    out = ["<!-- ASSEMBLED by assemble_paper.py. Do not edit the drafts after "
           "this point; edit this file. -->", ""]
    total = 0
    for num in order:
        s = found[num]
        total += len(s["body"].split())
        head = "# Abstract" if s["kind"] == "abstract" else \
               f"{'#' * (1 + num.count('.'))} {num} {s['title']}"
        out += [head, ""]
        body = s["body"]
        if num in REVISION_SITES:
            probe, why = REVISION_SITES[num]
            if probe in body:
                body = body.replace(
                    probe,
                    f"<!-- ===== REVISION REQUIRED: {why} ===== -->\n{probe}",
                    1)
                print(f"\n  marked revision site in §{num}")
            else:
                print(f"\n  NOTE: §{num}'s revision anchor not found "
                      f"(\"{probe[:40]}...\") — apply it by hand")
        out += [body, ""]

    Path(args.out).write_text("\n".join(out))
    print(f"\nwrote {args.out}")
    print(f"  {len(order)} sections, ~{total:,} words of body")

    expected = {"0", "1", "2", "3.1", "3.2", "3.3", "3.4", "3.5", "4.1", "4.2",
                "4.3", "4.4", "4.5", "5.1", "5.2", "5.3", "5.4", "5.5", "6.1",
                "6.2", "6.3", "6.4", "7", "8"}
    missing = expected - set(found)
    extra = set(found) - expected
    if missing:
        print(f"\n  MISSING: {', '.join(sorted(missing, key=sort_key))}")
        print("  (§6.1-6.4 appear as subsections; if §6 came through whole "
              "that is fine)")
    if extra:
        print(f"  UNEXPECTED: {', '.join(sorted(extra, key=sort_key))}")

    print("\nNEXT, IN ORDER:")
    print("  1. apply the two revisions marked in the file")
    print("  2. insert Tables 1, 2, 3 and the figure references")
    print("  3. python3 check_paper_consistency.py "
          f"{args.out}")
    print("  4. fill the [REF:...] markers from the original paper")


if __name__ == "__main__":
    main()
