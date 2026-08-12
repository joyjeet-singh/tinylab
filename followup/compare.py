"""Paired comparison of a follow-up run against its published baseline.

Both runs draw the same episodes (same seed, same num_eval, verified), so
every completed episode is a matched pair and McNemar applies.

    python3 followup/compare.py probe_off100 exp_ref_p2/realenv_plan_cem.json
    python3 followup/compare.py probe_off25  exp_phase2_recal_25/realenv_plan_cem.json
"""
import json
import re
import sys
from pathlib import Path

run = sys.argv[1] if len(sys.argv) > 1 else "probe_off100"
base_path = (sys.argv[2] if len(sys.argv) > 2
             else "exp_ref_p2/realenv_plan_cem.json")

base = {e["episode"]: e for e in json.load(open(base_path))["results"]}
log = Path(f"followup/{run}/run.log").read_text()
rows = re.findall(
    r"ep\s+\d+/\d+ \(#\s*(\d+)\): (REACHED|missed )\s+in\s+(\d+) steps, "
    r"final dist\s+([\d.]+), start-goal\s+([\d.]+)", log)
if not rows:
    sys.exit(f"{run}: no episodes finished yet")

print(f"{run}  vs  {base_path}\n")
print(f"{'ep':>6} {'start-goal':>10} | {'baseline':>20} | {'follow-up':>20}")
print("-" * 64)
nb = nf = both = neither = only_b = only_f = 0
for epid, res, steps, fin, sg in rows:
    b = base.get(int(epid))
    if b is None:
        continue
    f_ok, b_ok = res == "REACHED", bool(b["success"])
    nb += b_ok; nf += f_ok
    both += b_ok and f_ok
    neither += (not b_ok) and (not f_ok)
    only_b += b_ok and not f_ok
    only_f += f_ok and not b_ok
    print(f"{epid:>6} {float(sg):>10.1f} | "
          f"{'REACHED' if b_ok else 'missed':>8} {b['steps']:>3}st {b['final_dist']:>7.1f} | "
          f"{'REACHED' if f_ok else 'missed':>8} {steps:>3}st {float(fin):>7.1f}")

n = both + neither + only_b + only_f
print("-" * 64)
print(f"after {n} paired episodes")
print(f"  baseline  {nb}/{n} = {100*nb/n:5.1f}%")
print(f"  follow-up {nf}/{n} = {100*nf/n:5.1f}%")
print(f"\n  discordant pairs: follow-up only {only_f}, baseline only {only_b}")
if only_b + only_f:
    # exact McNemar (binomial on the discordant pairs)
    from math import comb
    k, m = min(only_b, only_f), only_b + only_f
    p = sum(comb(m, i) for i in range(k + 1)) / 2 ** m * 2
    print(f"  exact McNemar p = {min(p, 1.0):.4g}")
