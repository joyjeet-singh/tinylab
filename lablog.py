"""Shared run bookkeeping. Every experiment gets: its own folder under
runs/, a manifest written before any work happens, and a JSONL logger
that puts one measurement per line on disk the moment it is measured.

Training scripts import this; analyzers read what it writes.
"""
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch


def git_commit_id():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def start_run(cfg, tag, extra=None):
    """Create the run folder and manifest. Returns (run_dir, log, close).

    cfg    : the full recipe-card dict (must contain experiment + seed)
    tag    : short label folded into the run folder's name
    extra  : optional dict of additional manifest fields (versions, hashes)
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg['experiment']}_{tag}_seed{cfg['seed']}_{stamp}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True)

    manifest = {
        "run_name": run_name,
        "config": cfg,                      # full copy: folder self-contained
        "git_commit": git_commit_id(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "started_at": datetime.now().isoformat(),
    }
    manifest.update(extra or {})
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log_f = open(run_dir / "metrics.jsonl", "a")

    def log(record):
        """One measurement -> one line on disk, immediately."""
        record["t"] = time.time()
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

    def close():
        log_f.close()

    return run_dir, log, close
