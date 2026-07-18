"""The scorekeeper. The only program in the lab allowed to compute a mean.

Reads runs/*/manifest.json and runs/*/metrics.jsonl and computes every
reported number from those files alone. Never trains, never touches the
dataset, does not even import torch.

Usage: python analyze.py            (reads ./runs)
       python analyze.py some_dir   (reads some_dir)
"""
import json
import statistics
import sys
from pathlib import Path


def load_run(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    evals = []
    done = False
    t_first = None
    t_last = None
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if t_first is None:
                t_first = rec["t"]
            t_last = rec["t"]
            if rec["kind"] == "eval":
                evals.append(rec["test_accuracy"])
            elif rec["kind"] == "done":
                done = True
    return {
        "name": manifest["run_name"],
        "model": manifest["config"]["model"]["name"],
        "seed": manifest["config"]["seed"],
        "git": manifest["git_commit"][:7],
        "train_hash": manifest["train_indices_sha256"][:12],
        "test_hash": manifest["test_indices_sha256"][:12],
        "evals": evals,
        "final": evals[-1] if evals else None,
        "best": max(evals) if evals else None,
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
    runs = [load_run(d) for d in dirs]

    complete = [r for r in runs if r["done"] and r["evals"]]
    for r in runs:
        if r not in complete:
            print(f"WARNING: {r['name']} incomplete (no 'done' record), excluded")
    if not complete:
        print("no complete runs found")
        return

    print(f"per-run results ({len(complete)} complete runs)")
    for r in sorted(complete, key=lambda r: (r["model"], r["seed"])):
        print(f"  {r['model']:>4} seed {r['seed']}  final {r['final']:.4f}  "
              f"best {r['best']:.4f}  {r['minutes']:.1f} min  git {r['git']}")

    print("\nintegrity checks (must be identical across all runs)")
    check("train-data hash", [r["train_hash"] for r in complete])
    check("test-data hash", [r["test_hash"] for r in complete])
    check("git commit", [r["git"] for r in complete])

    print("\nsummary: final-epoch accuracy, mean +/- std over seeds (primary metric)")
    by_model = {}
    for r in complete:
        by_model.setdefault(r["model"], []).append(r)
    means = {}
    for model, rs in sorted(by_model.items()):
        finals = [r["final"] for r in rs]
        bests = [r["best"] for r in rs]
        mins = [r["minutes"] for r in rs]
        mean = statistics.mean(finals)
        std = statistics.stdev(finals) if len(finals) > 1 else 0.0
        means[model] = mean
        print(f"  {model:>4}: {mean:.4f} +/- {std:.4f}   "
              f"(best-epoch mean {statistics.mean(bests):.4f}, "
              f"avg {statistics.mean(mins):.1f} min/run, n={len(finals)})")

    if len(means) == 2:
        (m1, v1), (m2, v2) = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
        print(f"\n  {m1} beats {m2} by {v1 - v2:.4f} ({(v1 - v2) * 100:.1f} points)")


if __name__ == "__main__":
    main()
