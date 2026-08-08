"""
extract_all_results.py -- rebuild the results table from the files on disk.

WHY THIS EXISTS
---------------
Numbers entered this project's record that were never measured. Several
evaluation outputs reached the assistant's context empty, and figures were
produced in response to them as though they had been read. Two are confirmed
wrong; others may be. The conversation is therefore not a usable source for any
figure in the paper.

The committed reports are. Every planning evaluation writes
`realenv_plan_*_report.txt` and a matching `.json`, and those files contain the
protocol they ran under. This walks every one of them and prints an
authoritative table: what was measured, under what settings, from which file.

Use its output — and only its output — when writing results.

WHAT IT READS
-------------
For each report it extracts the success rate, the non-trivial rate, the mean
final distance, the wall-clock, the episode count, the goal construction, the
budget, the checkpoint, the guard value and whether the adapter was active.
Anything it cannot find is printed as "?" rather than guessed.

Usage:
    python3 extract_all_results.py
    python3 extract_all_results.py --csv results.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PATTERNS = {
    "success": r"success rate\s*:\s*(\d+)\s*/\s*(\d+)\s*=\s*([\d.]+)\s*%",
    "nontrivial": r"non-trivial success\s*:\s*(\d+)\s*/\s*(\d+)\s*=\s*([\d.]+)\s*%",
    "meanfinal": r"mean final distance\s*:\s*([\d.]+)",
    "wall": r"wall-clock\s*:\s*([\d.]+)\s*min",
    "guard": r"median over \d+ frames:\s*([\d.]+)",
    "ckpt": r"\(ckpt[^)]*\)|checkpoint:\s*(\S+)",
    "episodes": r"episodes:\s*(\d+)\s*sampled",
    "budget": r"budget\s+(\d+)",
    "goaloffset": r"goal_offset\s+(\d+)",
}
GOAL_TARGET = r"goal\s*=\s*the episode's recorded TARGET"
GOAL_FRAME = r"goal\s*=\s*(?:frame\s*)?(\d+)\s*(?:of the same episode|frames later)"
SUCCESS_ONLY = r"\[policy-successful only\]"
ADAPTER = r"ADAPTER ACTIVE"
BANNER_AUTHORS = r"MODEL: the AUTHORS'"
AUTHORS = r"MODEL: the AUTHORS'|AUTHORS' released weights|authors' checkpoint"
BANNER_OURS = r"MODEL: (?:our|OUR)"
CONTRADICTION = r"OUR reproduction checkpoint"
RANDOM = r"SUMMARY \([\w]*random\)"


def scan(p: Path):
    t = p.read_text(errors="ignore")
    g = lambda k: (re.search(PATTERNS[k], t) or [None])
    m = re.search(PATTERNS["success"], t)
    if not m:
        return None
    row = {
        "file": str(p),
        "arm": "random" if re.search(RANDOM, t) else "cem",
        "whose": ("authors" if re.search(AUTHORS, t)
                  else "ours" if (re.search(BANNER_OURS, t)
                                  or re.search(r"\(ckpt[^,)]*", t))
                  else "?"),
        "succ": f"{m.group(1)}/{m.group(2)}",
        "pct": float(m.group(3)),
    }
    m2 = re.search(PATTERNS["nontrivial"], t)
    row["nontrivial"] = f"{m2.group(3)}%" if m2 else "?"
    for k in ("meanfinal", "wall", "guard", "episodes", "budget"):
        mm = re.search(PATTERNS[k], t)
        row[k] = mm.group(1) if mm else "?"
    if re.search(GOAL_TARGET, t):
        row["goal"] = "TARGET"
    else:
        mm = re.search(GOAL_FRAME, t)
        row["goal"] = f"frame {mm.group(1)}" if mm else "?"
    row["succ_only"] = "yes" if re.search(SUCCESS_ONLY, t) else "no"
    row["adapter"] = "yes" if re.search(ADAPTER, t) else "no"
    row["banner_conflict"] = "yes" if (re.search(BANNER_AUTHORS, t)
                                       and re.search(CONTRADICTION, t)) else "no"
    mm = re.search(r"\((ckpt[^,)]*)", t)
    row["ckpt"] = mm.group(1) if mm else ("authors-spec" if row["whose"] == "authors" else "?")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    reports = sorted(root.glob("**/realenv_plan_*report*.txt"))
    if not reports:
        sys.exit("no report files found — run from the tinylab folder")

    rows = [r for r in (scan(p) for p in reports) if r]
    for r in rows:
        try:
            rd = str(Path(r["file"]).parent.resolve().relative_to(root.resolve()))
        except ValueError:
            rd = str(Path(r["file"]).parent)
        r["reldir"] = "(repo root)" if rd == "." else rd
    nested_roots = {q.parent.name for q in
                    root.glob("*/realenv_r2_planner_eval.py")}
    for r in rows:
        first = r["reldir"].split("/")[0]
        r["nested_copy"] = "yes" if first in nested_roots else "no"
    nested = [r for r in rows if r["nested_copy"] == "yes"]
    print(f"{len(rows)} evaluation reports found under {root.resolve()}")
    if nested:
        print(f"{len(nested)} of them are inside NESTED COPIES of the repository "
              f"and duplicate top-level results.")
        print("Cite the TOP-LEVEL path in the paper, never a nested one.")
    print()

    hdr = ("dir", "arm", "whose", "ckpt", "goal", "succ-only", "budget",
           "result", "non-triv", "mean final", "guard")
    w = (26, 7, 8, 20, 11, 10, 8, 12, 9, 11, 7)
    print("  " + "".join(h.ljust(x) for h, x in zip(hdr, w)))
    print("  " + "-" * sum(w))
    for r in sorted(rows, key=lambda r: (r["goal"], r["budget"], r["file"])):
        d = r["reldir"]
        if len(d) > 25:
            d = "..." + d[-22:]
        cells = (d, r["arm"], r["whose"], r["ckpt"][:19], r["goal"],
                 r["succ_only"], r["budget"],
                 f"{r['succ']} {r['pct']}%", r["nontrivial"],
                 r["meanfinal"], r["guard"])
        print("  " + "".join(str(c).ljust(x) for c, x in zip(cells, w)))

    print("\nEvery figure above was read from the file named in the first")
    print("column. Cite that path in the paper. Do not carry a number forward")
    print("from any other source.")

    conflict = [r for r in rows if r["banner_conflict"] == "yes"]
    if conflict:
        print(f"\n{len(conflict)} report(s) whose banner says the AUTHORS'"
              " checkpoint while the deviations block says ours. The BANNER\n"
              "is authoritative; the block printed a stale default:")
        for r in conflict:
            print(f"  {r['file']}")

    missing = [r for r in rows
               if "?" in (r["goal"], r["budget"], r["whose"], r["ckpt"])]
    if missing:
        print(f"\n{len(missing)} report(s) leave a field undetermined:")
        print("goal, budget, whose or ckpt. The measurement is valid;")
        print("only the self-description. Resolve by hand before citing:")
        for r in missing:
            print(f"  {r['file']}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
