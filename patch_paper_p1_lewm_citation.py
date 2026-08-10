"""P1 -- resolve the `le-wm` code release citation.

The References section ends with a literal placeholder: <URL>, <DATE>, <HASH>.
Table 1's caption says "A number after a colon is a line number in the version
we audited, whose commit is recorded in the repository", so every file-and-line
citation in the fidelity table rests on this entry.

The three values are recovered from the audited clone itself. None is typed
into this file. If the clone is absent, or if it is not the pristine checkout
that was audited, this script refuses to write anything.

The clone is verified to be the audited version three ways:
  - its reflog holds exactly one entry, a clone, with no later fetch or
    checkout that could have moved HEAD;
  - its tracked tree is clean;
  - two of the line numbers PAPER.md cites still land on the lines it cites
    them for.

It also writes docs/lewm_audit_commit.txt, which is what makes Table 1's
"recorded in the repository" true.
"""
import re
import shutil
import subprocess
from pathlib import Path

PAPER = Path("docs/paper/PAPER.md")
LEWM = Path.home() / "le-wm"
PROVENANCE = Path("docs/lewm_audit_commit.txt")


def git(*args):
    return subprocess.run(["git", "-C", str(LEWM), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def next_backup(path):
    n = 0
    while True:
        cand = Path(str(path) + (".bak" if n == 0 else f".bak{n}"))
        if not cand.exists():
            return cand
        n += 1


def patch(old, new, path=PAPER):
    text = path.read_text()
    pat = re.compile(r'[\s>]+'.join(re.escape(w) for w in old.split()))
    matches = list(pat.finditer(text))
    assert len(matches) == 1, (
        f"expected exactly 1 match, found {len(matches)} for: {old[:60]!r}")
    bak = next_backup(path)
    shutil.copy2(path, bak)
    start, end = matches[0].span()
    path.write_text(text[:start] + new + text[end:])
    subprocess.run(["diff", "-u", str(bak), str(path)])
    print(f"patched; backup at {bak}")


# ---------------------------------------------------------------- recover
assert LEWM.exists(), f"audited clone not found at {LEWM} -- STOP (S4)"
assert (LEWM / ".git").exists(), f"{LEWM} is not a git checkout -- STOP (S4)"

url = git("remote", "get-url", "origin").removesuffix(".git")
commit = git("rev-parse", "HEAD")
commit_date = git("log", "-1", "--date=short", "--format=%cd")

# HEAD must not have moved since the clone, or the audited version is unknown.
reflog = [l for l in git("reflog", "--date=iso-strict").splitlines() if l.strip()]
assert len(reflog) == 1, (
    f"clone reflog has {len(reflog)} entries; HEAD may have moved since the "
    f"audit and the audited commit is therefore not established -- STOP (S4)\n"
    + "\n".join(reflog))
assert "clone:" in reflog[0], f"first reflog entry is not a clone: {reflog[0]}"
m = re.search(r"HEAD@\{(\d{4})-(\d{2})-(\d{2})", reflog[0])
assert m, f"cannot read the clone date from the reflog: {reflog[0]}"
MONTHS = ("January February March April May June July August September "
          "October November December").split()
accessed = f"{int(m.group(3))} {MONTHS[int(m.group(2)) - 1]} {m.group(1)}"

dirty = [l for l in git("status", "--porcelain").splitlines()
         if not l.startswith("??")]
assert not dirty, f"audited clone has modified tracked files: {dirty}"

# The paper cites file:line into this checkout. Confirm two of them still land.
LINE_CITATIONS = {
    ("utils.py", 29): "torch.isnan",
    ("train.py", 65): "get_column_normalizer",
}
for (fname, lineno), needle in LINE_CITATIONS.items():
    line = (LEWM / fname).read_text().splitlines()[lineno - 1]
    assert needle in line, (
        f"{fname}:{lineno} does not contain {needle!r} -- this checkout is not "
        f"the version PAPER.md audited. Got: {line.strip()!r} -- STOP (S4)")
    print(f"  verified {fname}:{lineno} -> {line.strip()[:60]}")

host = url.replace("https://", "").replace("http://", "")
print(f"\n  url      {host}\n  commit   {commit}\n  accessed {accessed}\n")

# ---------------------------------------------------------------- write
PROVENANCE.write_text(
    "The le-wm code release audited by this reproduction\n"
    "===================================================\n\n"
    f"repository : {url}\n"
    f"commit     : {commit}\n"
    f"commit date: {commit_date}\n"
    f"accessed   : {accessed} (clone; the checkout's reflog holds this single\n"
    "             entry, so HEAD did not move afterwards)\n\n"
    "Every `file.py:NN` citation in Table 1 of the paper is a line number in\n"
    "this commit. Two were re-checked mechanically when this file was written:\n"
    + "".join(f"  {f}:{n}  contains {s!r}\n"
             for (f, n), s in LINE_CITATIONS.items())
    + "\nThe clone itself is not vendored here: it is the authors' code, MIT\n"
      "licensed, and available at the URL above.\n")
print(f"wrote {PROVENANCE}")

patch(
    old="The `le-wm` code release accompanying Maes et al. (2026a). <URL>, accessed <DATE>, commit <HASH>.",
    new=(f"The `le-wm` code release accompanying Maes et al. (2026a).\n"
         f"{host}, accessed {accessed}, commit `{commit}`.\n"
         f"Line numbers cited in Table 1 refer to this commit; it is recorded in\n"
         f"our repository at `{PROVENANCE}`."),
)
