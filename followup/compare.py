"""Paired comparison: probe cost vs the published latent-cost baseline."""
import json, re, sys
from pathlib import Path
base = {e["episode"]: e for e in
        json.load(open("exp_ref_p2/realenv_plan_cem.json"))["results"]}
log = Path("followup/probe_off100/run.log").read_text()
rows = re.findall(r"ep\s+(\d+)/50 \(#\s*(\d+)\): (REACHED|missed )\s+in\s+(\d+) steps, "
                  r"final dist\s+([\d.]+), start-goal\s+([\d.]+)", log)
if not rows:
    sys.exit("no episodes yet")
nb = np_ = 0
print(f"{'ep':>6} {'start-goal':>10} | {'baseline':>18} | {'probe cost':>18}")
print("-" * 62)
for _, epid, res, steps, fin, sg in rows:
    b = base[int(epid)]
    ok = res == "REACHED"
    nb += b["success"]; np_ += ok
    print(f"{epid:>6} {float(sg):>10.1f} | "
          f"{'REACHED' if b['success'] else 'missed ':>8} {b['steps']:>3}st {b['final_dist']:>6.1f} | "
          f"{'REACHED' if ok else 'missed ':>8} {steps:>3}st {float(fin):>6.1f}")
n = len(rows)
print("-" * 62)
print(f"after {n} paired episodes:  baseline {nb}/{n} = {100*nb/n:.1f}%   "
      f"probe cost {np_}/{n} = {100*np_/n:.1f}%")
