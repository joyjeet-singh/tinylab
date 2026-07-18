"""Train one racer per the recipe card. Measure, write, exit.

Usage:  python train.py --config configs/mlp.yaml --seed 0
Writes: runs/<experiment>_<model>_seed<k>_<timestamp>/
          manifest.json   the jar label: exact conditions, no results
          metrics.jsonl   one line per measurement, written as it happens
"""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data import load_subsets
from models import build_model
from seed import set_seed


def git_commit_id():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256_of_ints(ints):
    """Content hash: same list of numbers <-> same code, always."""
    return hashlib.sha256(",".join(map(str, ints)).encode()).hexdigest()


def evaluate(model, loader):
    """Exam on held-out images. Observe only: no blame, no learning."""
    model.eval()                      # exam mode
    correct, total = 0, 0
    with torch.no_grad():             # no blame-tracing during exams
        for images, labels in loader:
            scores = model(images)
            correct += (scores.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    model.train()                     # back to learning mode
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = args.seed           # command line overrides the card's default
    set_seed(cfg["seed"])

    # --- the run's own folder ---
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg['experiment']}_{cfg['model']['name']}_seed{cfg['seed']}_{stamp}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True)

    # --- data, with a real fingerprint this time ---
    train_set, test_set = load_subsets(cfg)
    g = torch.Generator().manual_seed(cfg["seed"])   # run seed shuffles batches
    train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, generator=g)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # --- the jar label: conditions only, never results ---
    manifest = {
        "run_name": run_name,
        "config": cfg,                          # full copy: folder is self-contained
        "git_commit": git_commit_id(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "train_indices_sha256": sha256_of_ints(train_set.indices),
        "test_indices_sha256": sha256_of_ints(test_set.indices),
        "started_at": datetime.now().isoformat(),
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # --- model, wrongness-scorer, knob-nudger ---
    model = build_model(cfg)
    loss_fn = nn.CrossEntropyLoss()             # punishes confident wrong answers hardest
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg["training"]["learning_rate"])

    log_f = open(run_dir / "metrics.jsonl", "a")

    def log(record):
        """One measurement -> one line on disk, immediately."""
        record["t"] = time.time()
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()                           # even a crash leaves honest partials

    # --- the rhythm: blame and nudge ---
    step = 0
    for epoch in range(cfg["training"]["epochs"]):
        for images, labels in train_loader:
            scores = model(images)
            loss = loss_fn(scores, labels)      # how wrong, one number
            optimizer.zero_grad()               # clear old blame
            loss.backward()                     # trace each knob's share
            optimizer.step()                    # nudge every knob against its blame
            step += 1
            if step % 10 == 0:
                log({"kind": "train", "epoch": epoch, "step": step,
                     "loss": round(loss.item(), 6)})

        acc = evaluate(model, test_loader)      # end-of-epoch exam
        log({"kind": "eval", "epoch": epoch, "step": step,
             "test_accuracy": round(acc, 6)})
        print(f"epoch {epoch}: test accuracy {acc:.4f}")

    log({"kind": "done", "total_steps": step})
    log_f.close()


if __name__ == "__main__":
    main()
