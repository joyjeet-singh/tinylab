"""
check_resume_equivalence.py -- proof that splitting a run does not change it.

Colab sessions die. We need to train across several sittings. That is only safe
if a run split in half lands on EXACTLY the same numbers as one unbroken run.

This runs the same config three ways:
    A) unbroken, N steps
    B) N/2 steps, stop, resume, N/2 more
and compares the model weights bit for bit.

If they differ, checkpointing is broken and every chunked result is untrustworthy
-- so this is a gate, not a nicety. Same spirit as check_repro.py.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import torch

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--config", default="configs/toy_lewm.yaml")
CFG = _p.parse_args().config
STEPS = 20
HALF = 10


def run(args):
    r = subprocess.run([sys.executable] + args, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit("run failed")
    return r.stdout


def latest_run_dir(before: set) -> Path:
    after = {p for p in Path("runs").iterdir() if p.is_dir()}
    new = after - before
    assert len(new) == 1, f"expected 1 new run dir, got {len(new)}"
    return new.pop()


def weights_of(run_dir: Path):
    ck = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    return ck["model"], ck["step"]


def main():
    Path("runs").mkdir(exist_ok=True)

    print(f"A) unbroken run: {STEPS} steps")
    before = {p for p in Path("runs").iterdir() if p.is_dir()}
    run(["train_toy_lewm.py", "--config", CFG, "--seed", "0",
         "--max-steps", str(STEPS)])
    dir_a = latest_run_dir(before)
    w_a, step_a = weights_of(dir_a)
    print(f"   -> {dir_a.name}  (stopped at step {step_a})")

    print(f"B) split run: {HALF} steps, stop, resume, {STEPS - HALF} more")
    before = {p for p in Path("runs").iterdir() if p.is_dir()}
    run(["train_toy_lewm.py", "--config", CFG, "--seed", "0",
         "--max-steps", str(HALF)])
    dir_b = latest_run_dir(before)
    print(f"   -> {dir_b.name}  (stopped at step {HALF})")
    run(["train_toy_lewm.py", "--config", CFG, "--seed", "0",
         "--resume", str(dir_b), "--max-steps", str(STEPS)])
    w_b, step_b = weights_of(dir_b)
    print(f"   -> resumed to step {step_b}")

    print()
    print("comparing every weight, bit for bit ...")
    assert step_a == step_b, f"different step counts: {step_a} vs {step_b}"
    worst, worst_name = 0.0, ""
    for k in w_a:
        d = (w_a[k].float() - w_b[k].float()).abs().max().item()
        if d > worst:
            worst, worst_name = d, k
    print(f"  parameters compared : {len(w_a)}")
    print(f"  largest difference  : {worst:.3e}  ({worst_name})")
    print()
    if worst == 0.0:
        print("  PASS -- a split run is bit-for-bit identical to an unbroken one.")
        print("  Chunked training across sessions is safe.")
    else:
        print("  FAIL -- resuming changes the result. Do NOT chunk training")
        print("  until this is fixed; every chunked number would be untrustworthy.")
        raise SystemExit(1)

    for d in (dir_a, dir_b):
        shutil.rmtree(d)
    print()
    print("  (cleaned up both test run folders)")


if __name__ == "__main__":
    main()
