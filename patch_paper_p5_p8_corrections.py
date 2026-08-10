"""Paper corrections P5-P8.

P5  §7   the report-header claim is too strong (work order §3, P5).
P6  §4.2 the 84.0%/8.0% pair is described as differing only in goal
         construction; the two protocols also carry different step budgets.
P7  §4.5 the 72.0% figure has no citable report and must come out.
P8  §4.2 scope the "eligible episodes are the data policy's failures"
         argument to the exact-offset reading (handoff §3.3).

Every number written below is read from a file in this script. None is typed.
"""
import csv
import re
import shutil
import subprocess
from pathlib import Path

PAPER = Path("docs/paper/PAPER.md")


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


# ------------------------------------------------------------ read the disk
rows = {r["reldir"]: r for r in
        csv.DictReader(open("docs/paper/results_from_disk.csv"))
        if r["nested_copy"] == "no" and r["arm"] == "cem"}

budget_offset25 = rows["exp_authors"]["budget"]
budget_target = rows["exp_target_all"]["budget"]
assert budget_offset25 != budget_target, (
    "the two arms share a budget; P6's correction would be wrong")
ratio = int(budget_target) // int(budget_offset25)

ds = Path("runs_archive/verified/dataset_episode_lengths.txt").read_text()
total_eps = re.search(r"episodes (\d+),", ds).group(1)
eligible100 = re.search(r"eligible at goal_offset 100 \(ln>100\): (\d+)", ds).group(1)
assert int(total_eps) > int(eligible100), "dataset query contradicts the argument"
total_fmt = f"{int(total_eps):,}"
eligible_fmt = f"{int(eligible100):,}"

print(f"read from disk: budgets {budget_offset25} vs {budget_target} "
      f"({ratio}x); episodes {total_fmt} total, {eligible_fmt} eligible\n")

# ------------------------------------------------------------------- P6
print("P6 -- the budget difference in the goal-construction comparison")
patch(
    old="an identical planner and an identical driving convention. They differ only in how the goal is constructed.",
    new=("an identical planner and an identical driving convention. They differ "
         f"in how the goal is constructed and, because each protocol carries its "
         f"own step budget, in budget: {budget_offset25} steps for the first and "
         f"{budget_target} for the second. That difference runs against the "
         f"comparison rather than for it — the recorded-target arm was given "
         f"{ratio} times as many steps and still collapsed."),
)

# ------------------------------------------------------------------- P7
print("P7 -- remove the 72.0% figure, which has no citable report")
patch(
    old="We note that both checkpoints were normalisation-recalibrated before this comparison (§4.3); against the same checkpoint *without* recalibration the figure is 72.0% and the test reaches p = 0.0074, but that comparison confounds the pipeline correction with the recalibration and we do not rely on it.",
    new=("We note that both checkpoints were normalisation-recalibrated before "
         "this comparison (§4.3). The same comparison against the "
         "un-recalibrated checkpoint would confound the pipeline correction "
         "with the recalibration, and we do not report it."),
)

# ------------------------------------------------------------------- P8
print("P8 -- scope the eligibility argument to the exact-offset reading")
patch(
    old="Evaluating at an offset of 100 therefore asks the planner to reach the point at which a failing policy ran out of time.",
    new=("Evaluating at an offset of 100 therefore asks the planner to reach the "
         "point at which a failing policy ran out of time.\n\n"
         "This argument holds under the reading that the offset is exact. "
         "`stable-worldmodel` describes the offline protocol as constraining the "
         "*maximum* number of steps separating start and goal rather than fixing "
         f"it, and under that reading all {total_fmt} episodes are eligible "
         f"rather than {eligible_fmt}, so the eligible set is no longer the data "
         "policy's failures and the argument above does not apply to it. We "
         "report the exact-offset reading because it is the one the appendix's "
         "own wording — a goal sampled 100 timesteps in the future — most "
         "directly supports, and because it is the reading under which the "
         "protocol is reproducible at all. We did not evaluate the alternative."),
)

# ------------------------------------------------------------------- P5
print("P5 -- the report-header claim is too strong")
patch(
    old="The measurements in those reports are unaffected — the header of each report states the parameters correctly — and the repository records which reports predate the fix.",
    new=("The measurements in those reports are unaffected — the header of each "
         "report records the protocol that was *requested*, accurately — and the "
         "repository records which reports predate the fix. The header is not a "
         "complete description of a run. Where the data constrain a requested "
         "instruction, the protocol actually realised can be narrower than the "
         "one the header names: initial-state sampling at a goal offset of 100 "
         "is requested and recorded, and has no effect (§4.2). And where two "
         "runs differ only in a driving convention the header does not print, "
         "the `spec_as_used.json` committed beside the report is what "
         "distinguishes them."),
)

print("all four patches applied.")
