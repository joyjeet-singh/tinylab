"""P4 -- the checkpoint count in §8.

§8 promises "all four checkpoints". Four runs were trained, but one run's best
checkpoint was never saved, and decision S3 fixed the release set at six files:
three BatchNorm-recalibrated checkpoints and the three un-recalibrated
originals they were made from.

The count is derived from a listing of the files actually staged for upload,
not typed. If the release directory changes, rerunning this reports the
mismatch instead of silently agreeing with a stale sentence.
"""
import re
import shutil
import subprocess
from pathlib import Path

PAPER = Path("docs/paper/PAPER.md")
RELEASE = Path("runs_archive/release")
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine"}


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
        f"expected exactly 1 match, found {len(matches)} for: {old[:70]!r}")
    bak = next_backup(path)
    shutil.copy2(path, bak)
    start, end = matches[0].span()
    path.write_text(text[:start] + new + text[end:])
    subprocess.run(["diff", "-u", str(bak), str(path)])
    print(f"patched; backup at {bak}")


staged = sorted(p.name for p in RELEASE.glob("*.pt"))
assert staged, f"no checkpoints staged in {RELEASE} -- run the strip script first"
recal = [n for n in staged if "-recal" in n]
plain = [n for n in staged if "-recal" not in n]
assert len(recal) == len(plain), (
    f"the release set is not paired: {len(recal)} recalibrated against "
    f"{len(plain)} originals")

total, half = len(staged), len(recal)
print(f"staged for release ({total}):")
for n in staged:
    print(f"  {n}")

patch(
    old="We release the reimplementation, all four checkpoints, every evaluation report,",
    new=(f"We release the reimplementation, {WORDS[total]} checkpoints — "
         f"{WORDS[half]} BatchNorm-recalibrated and the {WORDS[half]} "
         f"un-recalibrated originals they were made from, so that the "
         f"evaluation-mode artifact of §4.3 can be checked independently — "
         f"every evaluation report,"),
)
