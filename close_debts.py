"""
close_debts.py -- settle the three record-keeping debts, and print the rows
they produce for the paper.

  A. COMMIT PROVENANCE   is the commit phase2 actually ran (4f2efbb) the gated
                         commit (d8308b9) plus the gate outputs, or something
                         else? "We gated X and ran Y" is not a sentence a
                         reproduction paper can leave unchecked.
  B. CHECKPOINT EPOCHS   phase2's directory holds ckpt.pt and ckpt_best.pt with
                         no ckpt_final.pt, and they differ by ~63 kB. Which
                         epoch is in each? The paper quotes epoch 8's numbers,
                         so this has to be confirmed rather than assumed.
  C. ENVIRONMENT TABLE   python 3.12.3 on phase2 against 3.11.15 on Runs 0-2,
                         plus torch and the data fingerprint, collected from
                         every manifest into one table for the deviations
                         section.

Nothing here changes any file. It reads and reports.

Usage (from the tinylab folder):
    python3 close_debts.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def sh(*cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", default="d8308b9")
    ap.add_argument("--ran", default="4f2efbb")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    # ---- A. commit provenance -------------------------------------------
    print("=" * 70)
    print(f"A. COMMIT PROVENANCE — is {args.ran} a descendant of {args.gated}?")
    print("=" * 70)
    rc, out, err = sh("git", "merge-base", "--is-ancestor", args.gated, args.ran)
    if rc == 0:
        print(f"  YES — {args.gated} is an ancestor of {args.ran}.")
        _, log, _ = sh("git", "log", "--oneline", f"{args.gated}..{args.ran}")
        n = len(log.splitlines()) if log else 0
        print(f"  {n} commit(s) between them:")
        for line in log.splitlines():
            print(f"    {line}")
        _, stat, _ = sh("git", "diff", "--stat", args.gated, args.ran)
        print(f"  files changed:")
        for line in stat.splitlines()[-12:]:
            print(f"    {line}")
        _, names, _ = sh("git", "diff", "--name-only", args.gated, args.ran)
        code = [f for f in names.splitlines()
                if f.endswith(".py") or f.endswith(".yaml")]
        if code:
            print(f"\n  *** CODE OR CONFIG CHANGED between gating and running: "
                  f"{code}")
            print(f"  *** The run did not execute the tree G1 checked. Say so "
                  f"in the paper.")
        else:
            print(f"\n  No .py or .yaml differs — the run executed the gated "
                  f"code. Only non-executable files (gate outputs, docs) were "
                  f"added.")
    else:
        print(f"  NO — {args.gated} is NOT an ancestor of {args.ran}"
              f"{': ' + err if err else ''}.")
        print(f"  The run did not descend from the gated commit. Investigate "
              f"before quoting either.")

    # ---- B. checkpoint epochs -------------------------------------------
    print("\n" + "=" * 70)
    print("B. WHICH EPOCH IS IN EACH CHECKPOINT?")
    print("=" * 70)
    import torch
    for run_dir in sorted(Path(args.runs).glob("*phase2*")):
        print(f"  {run_dir.name}")
        for name in ("ckpt.pt", "ckpt_best.pt", "ckpt_final.pt"):
            f = run_dir / name
            if not f.exists():
                print(f"    {name:<15} (absent)")
                continue
            ck = torch.load(f, map_location="cpu", weights_only=False)
            keys = [k for k in ck.keys() if k != "model"] if isinstance(ck, dict) else []
            print(f"    {name:<15} epoch {ck.get('epoch')}   "
                  f"size {f.stat().st_size/1e6:8.1f} MB   "
                  f"other keys {keys}")
        print("    (the paper quotes epoch 8: pred 4.603, step0 ratio 2.255 — "
              "confirm that is ckpt_best)")

    # ---- C. environment table -------------------------------------------
    print("\n" + "=" * 70)
    print("C. ENVIRONMENT ACROSS ALL RUNS (for the deviations section)")
    print("=" * 70)
    rows = []
    for mf in sorted(Path(args.runs).glob("*/manifest.json")):
        d = json.loads(mf.read_text())
        conv = d.get("loader_convention")
        rows.append((mf.parent.name.split("_seed")[0][:34],
                     d.get("python_version", "?"),
                     d.get("torch_version", "?"),
                     str(d.get("data_sha256", "?"))[:8],
                     d["config"]["training"].get("learning_rate"),
                     d["config"]["loss"].get("sigreg_weight"),
                     d["config"]["model"].get("history_size"),
                     d["config"]["model"].get("action_dim"),
                     "yes" if conv else "—"))
    hdr = ("run", "python", "torch", "data", "lr", "sigreg", "hs", "act", "conv")
    w = (36, 9, 14, 10, 9, 8, 4, 5, 5)
    print("  " + "".join(h.ljust(x) for h, x in zip(hdr, w)))
    print("  " + "-" * sum(w))
    for r in rows:
        print("  " + "".join(str(v).ljust(x) for v, x in zip(r, w)))
    pys = {r[1] for r in rows}
    if len(pys) > 1:
        print(f"\n  DEVIATION TO RECORD: python differs across runs {sorted(pys)}.")
        print(f"  Benign — but a reproduction paper states it rather than "
              f"leaving a reader to find it.")
    print("\n  Note: data_sha256 is a FINGERPRINT over (clip index, file), not a")
    print("  file hash. Runs sharing a history_size share a value; a difference")
    print("  between history_size 1 and 3 runs is expected, not a data change.")


if __name__ == "__main__":
    main()
