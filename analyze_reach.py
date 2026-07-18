"""
analyze_reach.py -- the scorekeeper for the reachability arms.

Same house rule as analyze.py: the only program allowed to compute a mean.
Reads runs/*/manifest.json + metrics.jsonl, computes every number from those
files alone, never trains, never imports torch. Looks only at runs whose
experiment name starts with 'reach'.

Usage: python analyze_reach.py            (reads ./runs)
       python analyze_reach.py some_dir
"""
import json
import statistics
import sys
from pathlib import Path


def load_run(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg = manifest["config"]
    if not str(cfg.get("experiment", "")).startswith("reach"):
        return None
    last_eval, done = None, False
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec["kind"] == "eval":
                last_eval = rec
            elif rec["kind"] == "done":
                done = True
    return {
        "name": manifest["run_name"],
        "arm": cfg["model"]["name"],
        "seed": cfg["seed"],
        "git": manifest["git_commit"][:7],
        "train_hash": manifest.get("train_data_sha256", "?")[:12],
        "test_hash": manifest.get("test_data_sha256", "?")[:12],
        "final": last_eval["test_accuracy"] if last_eval else None,
        "reach": last_eval.get("acc_reachable") if last_eval else None,
        "unreach": last_eval.get("acc_unreachable") if last_eval else None,
        "by_distance": last_eval.get("acc_by_distance", {}) if last_eval else {},
        "done": done,
    }


def check(label, values):
    ok = len(set(values)) == 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: "
          f"{values[0] if ok else ' vs '.join(sorted(set(map(str, values))))}")


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    runs = [r for d in sorted(runs_dir.iterdir()) if d.is_dir()
            for r in [load_run(d)] if r is not None]
    complete = [r for r in runs if r["done"] and r["final"] is not None]
    for r in runs:
        if r not in complete:
            print(f"WARNING: {r['name']} incomplete, excluded")
    if not complete:
        print("no complete reachability runs found"); return

    print(f"per-run results ({len(complete)} complete runs)")
    for r in sorted(complete, key=lambda r: (r["arm"], r["seed"])):
        print(f"  {r['arm']:<16} seed {r['seed']}  overall {r['final']:.4f}  "
              f"reach {r['reach']:.4f}  unreach {r['unreach']:.4f}  git {r['git']}")

    print("\nintegrity checks (must be identical across all arms compared)")
    check("train-data hash", [r["train_hash"] for r in complete])
    check("test-data hash", [r["test_hash"] for r in complete])
    check("git commit", [r["git"] for r in complete])

    by_arm = {}
    for r in complete:
        by_arm.setdefault(r["arm"], []).append(r)

    print("\nsummary: overall test accuracy, mean +/- std over seeds (primary metric)")
    arm_mean = {}
    for arm, rs in sorted(by_arm.items()):
        mo, so = mean_std([r["final"] for r in rs])
        mr, _ = mean_std([r["reach"] for r in rs])
        mu, _ = mean_std([r["unreach"] for r in rs])
        arm_mean[arm] = (mo, so)
        print(f"  {arm:<16}: {mo:.4f} +/- {so:.4f}   "
              f"(reachable {mr:.4f}, unreachable {mu:.4f}, n={len(rs)})")

    print("\nreachable accuracy by true distance (mean over seeds)")
    all_d = sorted({int(d) for rs in by_arm.values() for r in rs for d in r["by_distance"]})
    header = "  arm".ljust(18) + "".join(f" d{d:<4}" for d in all_d)
    print(header)
    for arm, rs in sorted(by_arm.items()):
        cells = []
        for d in all_d:
            m, _ = mean_std([r["by_distance"].get(str(d), r["by_distance"].get(d)) for r in rs])
            cells.append(f"{m*100:>4.0f} " if m is not None else "   . ")
        print(f"  {arm:<16}" + "".join(cells))

    if len(arm_mean) == 2:
        (a1, (v1, s1)), (a2, (v2, s2)) = sorted(arm_mean.items(),
                                                key=lambda kv: kv[1][0], reverse=True)
        print(f"\n  {a1} beats {a2} by {v1 - v2:.4f} overall "
              f"(spread: {s1:.4f} vs {s2:.4f})")


if __name__ == "__main__":
    main()
