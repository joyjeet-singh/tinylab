"""Scorekeeper for RL runs. Reads runs/*/metrics.jsonl episode records
and computes every reported number from those files alone. Never trains.

Usage: python analyze_rl.py            (reads ./runs)
       python analyze_rl.py some_dir
"""
import json
import statistics
import sys
from pathlib import Path


def moving_avg(xs, w):
    out, s = [], 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= w:
            s -= xs[i - w]
        out.append(s / min(i + 1, w))
    return out


def load_run(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    returns, entropies = [], []
    done = False
    t_first = t_last = None
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if t_first is None:
                t_first = rec["t"]
            t_last = rec["t"]
            if rec["kind"] == "episode":
                returns.append(rec["return"])
                entropies.append(rec["entropy"])
            elif rec["kind"] == "done":
                done = True
    cfg = manifest["config"]
    ma50 = moving_avg(returns, 50) if returns else []
    cross = next((i for i, v in enumerate(ma50) if v >= 400), None)
    return {
        "name": manifest["run_name"],
        "exp": cfg["experiment"],
        "env": cfg["env"],
        "seed": cfg["seed"],
        "git": manifest["git_commit"][:7],
        "episodes": len(returns),
        "final100": statistics.mean(returns[-100:]) if returns else None,
        "best100": max(moving_avg(returns, 100)) if returns else None,
        "cross400": cross,
        "final_entropy": statistics.mean(entropies[-100:]) if entropies else None,
        "minutes": (t_last - t_first) / 60 if t_first is not None else None,
        "done": done,
    }


def check(label, values):
    """All runs must agree on this value, or the comparison is not fair."""
    ok = len(set(values)) == 1
    tag = "PASS" if ok else "FAIL"
    detail = values[0] if ok else " vs ".join(sorted(set(map(str, values))))
    print(f"  [{tag}] {label}: {detail}")


def main():
    runs_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    runs = [load_run(d) for d in dirs if (d / "metrics.jsonl").exists()]
    complete = [r for r in runs if r["done"] and r["episodes"]]
    for r in runs:
        if r not in complete:
            print(f"WARNING: {r['name']} incomplete (no 'done' record), excluded")
    if not complete:
        print("no complete runs found")
        return

    print(f"per-run results ({len(complete)} complete runs)")
    for r in sorted(complete, key=lambda r: (r["exp"], r["seed"])):
        cross = f"ep {r['cross400']}" if r["cross400"] is not None else "never"
        print(f"  {r['exp']} seed {r['seed']}  eps {r['episodes']}  final100 {r['final100']:.1f}  "
              f"best100 {r['best100']:.1f}  ma50>=400 at {cross}  "
              f"entropy(end) {r['final_entropy']:.3f}  "
              f"{r['minutes']:.1f} min  git {r['git']}")

    print("\nintegrity checks (must be identical across all runs)")
    check("experiment", [r["exp"] for r in complete])
    check("env", [r["env"] for r in complete])
    check("episode count", [r["episodes"] for r in complete])
    check("git commit", [r["git"] for r in complete])

    finals = [r["final100"] for r in complete]
    if len(finals) > 1:
        print(f"\nsummary: final-100 return {statistics.mean(finals):.1f} "
              f"+/- {statistics.stdev(finals):.1f} over {len(finals)} runs "
              f"(range {min(finals):.1f} to {max(finals):.1f})")


if __name__ == "__main__":
    main()
