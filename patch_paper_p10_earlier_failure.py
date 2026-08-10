"""P10 -- give §4.2's "An earlier failure, recorded" its evidence.

That paragraph was the last place in the paper quoting numbers with no
committed run behind them: the superseded calibration's 46.0%, the median
steps to success, and the count of misses that finished farther from the goal
than they began. The run has now been reproduced on CPU -- the real driving
spec with exactly two fields changed, encoding back to slots and scale back
to 1.0 -- and it lands on the paper's figures exactly.

One number in the paragraph does not survive contact with the disk. It
compares the artifact's median steps against "our checkpoint's 18"; no
committed offset-25 run has a median of 18. The corrected checkpoint's is
read here and written in.

Every number below is read from a report or its episode data.
"""
import json
import re
import shutil
import statistics
import subprocess
from pathlib import Path

PAPER = Path("docs/paper/PAPER.md")
WRONG = Path("exp_wrongscale_25/realenv_plan_authors_cem.json")
OURS = Path("exp_phase2_recal_25/realenv_plan_cem.json")


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
    print(f"  patched; backup at {bak}\n")


def stats(p):
    res = json.load(open(p))["results"]
    ok = [e for e in res if e["success"]]
    miss = [e for e in res if not e["success"]]
    return {
        "n": len(res),
        "ok": len(ok),
        "pct": 100 * len(ok) / len(res),
        "median_steps": int(statistics.median(e["steps"] for e in ok)),
        "miss": len(miss),
        "farther": sum(e["final_dist"] > e["start_goal_dist"] for e in miss),
    }


w, o = stats(WRONG), stats(OURS)
print(f"superseded run ({WRONG.parent})")
print(f"  {w['ok']}/{w['n']} = {w['pct']:.1f}%, median {w['median_steps']} steps "
      f"to success, {w['farther']} of {w['miss']} misses finished farther")
print(f"corrected checkpoint ({OURS.parent})")
print(f"  {o['ok']}/{o['n']}, median {o['median_steps']} steps to success\n")

assert f"{w['pct']:.1f}" == "46.0", (
    f"the reproduction gives {w['pct']:.1f}%, not the 46.0% the paper records; "
    f"report that rather than patching around it")

patch(
    old=("successes came unusually fast (median 5 steps against our checkpoint's 18) "
         "while misses were extreme overshoots, with 22 of 27 finishing farther "
         "from the goal than they started."),
    new=(f"successes came unusually fast (median {w['median_steps']} steps "
         f"against our checkpoint's {o['median_steps']}) while misses were "
         f"extreme overshoots, with {w['farther']} of {w['miss']} finishing "
         f"farther from the goal than they started."),
)

patch(
    old=("Measuring the action scale directly then confirmed it, and the "
         "corrected figure is the 84.0% above."),
    new=(f"Measuring the action scale directly then confirmed it, and the "
         f"corrected figure is the 84.0% above. The superseded run is "
         f"committed rather than described: `{WRONG.parent}/`, driven by a "
         f"spec identical to the corrected one but for the two fields that "
         f"were wrong."),
)
