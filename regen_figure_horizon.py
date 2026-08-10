"""Regenerate fig_horizon_dissociation.png with every input read from disk.

The figure's x-axis labels carry the one-step ratios. Those ratios now have
committed measurements, and the pre-correction checkpoint's corrected from
0.830 to 0.829, so the figure has to be rebuilt or it will contradict Table 3.

figure_horizon_dissociation.py already parses the planning rates and mean final
distances out of the committed reports. The one-step ratios are the only values
it takes on the command line, and this supplies them from the artifacts rather
than by hand. Nothing here is typed.
"""
import csv
import re
import subprocess
from pathlib import Path

V = Path("runs_archive/verified")

RATIO = {
    "our pre-correction": (V / "check_a_step0_run2.txt",
                           r"step0_err\s+[\d.]+\s+real_step\s+[\d.]+\s+ratio\s+([\d.]+)"),
    "authors' released": (V / "repeat_encoding_authors.txt",
                          r"repeat \(constant-action planner\)\s+[\d.]+\s+([\d.]+)"),
    "our corrected": (V / "driving_spec_phase2_recal.txt",
                      r"repeat of MEAN \(displacement-matched\)\s+[\d.]+\s+([\d.]+)"),
}
REPORTS = {
    "our pre-correction": ("exp_run2_recal_25", "exp_run2_recal"),
    "authors' released": ("exp_authors", "exp_authors_100"),
    "our corrected": ("exp_phase2_recal_25", "exp_phase2_recal"),
}

rows = {(r["reldir"], r["arm"]): r for r in
        csv.DictReader(open("docs/paper/results_from_disk.csv"))
        if r["nested_copy"] == "no"}

cmd = ["./.venv/bin/python", "figure_horizon_dissociation.py"]
for label, (path, rx) in RATIO.items():
    assert path.exists(), f"missing evidence file: {path}"
    m = re.search(rx, path.read_text())
    assert m, f"could not read the one-step ratio from {path}"
    d25, d100 = REPORTS[label]
    r25, r100 = rows[(d25, "cem")], rows[(d100, "cem")]
    assert r25["goal"] == "frame 25" and r100["goal"] == "frame 100", \
        f"{label}: report offsets are not 25 and 100"
    print(f"  {label:<20} ratio {m.group(1)}  ({path.name})")
    cmd += ["--entry", f"{label}={m.group(1)}={r25['file']}={r100['file']}"]

# the random-action control line, read from its own report
rnd = rows[("exp_phase2_recal", "random")]
control = re.search(r"mean final distance\s*:\s*([\d.]+)",
                    Path(rnd["file"]).read_text()).group(1)
print(f"  {'random control':<20} mean final {control}  ({rnd['file']})")
cmd += ["--random-final", control, "--out", "fig_horizon_dissociation.png"]

print()
subprocess.run(cmd, check=True)
