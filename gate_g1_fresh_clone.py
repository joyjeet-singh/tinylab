"""
gate_g1_fresh_clone.py -- prove the run depends on nothing that lives only on
this machine, and that its outputs can be got back off the rented box.

WHAT G1 IS FOR
--------------
Every failure this project has had was a thing that looked fine locally. The
one class it has not yet been bitten by, only because of luck, is the classic:
a run that works on the author's machine and dies on a rented one because it
imports a file that was never committed, or reads a config that only exists in
a scratch directory. G1 makes that impossible to discover at $6 an hour.

It clones the repository into a temporary directory -- the same way the rented
box will -- and checks that everything the run needs is inside that clone.

FIVE CHECKS
-----------
  1. WORKING TREE   no uncommitted edits to tracked files the run imports, and
                    no untracked file that the run imports. Either means the
                    rented box would get a different program than you tested.
  2. FRESH CLONE    clone HEAD into a temp dir. Everything below runs there.
  3. IMPORT CLOSURE every local module the entry point imports, transitively,
                    exists in the clone. Missing one is the classic failure.
  4. COMPILES       every Python file in the clone byte-compiles, and the
                    entry point's --help runs inside the clone.
  5. ARTIFACT AUDIT what the run will write, whether the output directory is
                    gitignored (it should be -- checkpoints do not belong in
                    git), and a printed retrieval command to rehearse BEFORE
                    launching, per the retrieval-first rule.

Data files are deliberately NOT required in the clone: tworoom.h5 is 25 GB and
is staged separately. G1 checks that the config's data path is declared, not
that it resolves here.

Usage (from the tinylab folder):
    python3 gate_g1_fresh_clone.py --config configs/phase2_dense_reference.yaml
"""
from __future__ import annotations

import argparse
import ast
import py_compile
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          shell=isinstance(cmd, str))


def local_imports(path: Path, root: Path, seen=None):
    """Every local module reachable from `path`, transitively."""
    seen = seen if seen is not None else set()
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return seen
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    for n in sorted(names):
        cand = root / f"{n}.py"
        if cand.exists() and n not in seen:
            seen.add(n)
            local_imports(cand, root, seen)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--entry", default="train_toy_lewm.py")
    ap.add_argument("--remote", default=None,
                    help="clone from this instead of the local repo")
    ap.add_argument("--keep", action="store_true",
                    help="leave the clone in place for inspection")
    args = ap.parse_args()

    root = Path.cwd()
    failures = []
    print("=" * 70)
    print("G1 — FRESH CLONE DRY RUN AND ARTIFACT AUDIT")
    print("=" * 70)

    # ---- 1. working tree --------------------------------------------------
    print("\n1. WORKING TREE")
    entry = Path(args.entry)
    if not entry.exists():
        sys.exit(f"entry point {entry} not found -- run from the repo root")
    needed = local_imports(entry, root) | {entry.stem}
    print(f"   the run imports {len(needed)} local modules: "
          f"{', '.join(sorted(needed))}")

    st = run(["git", "status", "--porcelain"])
    dirty, untracked = [], []
    for line in st.stdout.splitlines():
        code, name = line[:2], line[3:].strip()
        (untracked if code == "??" else dirty).append(name)
    watch = {f"{n}.py" for n in needed} | {args.config}
    dirty_relevant = [f for f in dirty if f in watch]
    untracked_relevant = [f for f in untracked if f in watch]
    print(f"   uncommitted changes overall: {len(dirty)}, untracked: "
          f"{len(untracked)}")
    if dirty_relevant:
        failures.append(f"uncommitted changes to files the run needs: "
                        f"{dirty_relevant}")
        print(f"   FAIL — modified but not committed: {dirty_relevant}")
    if untracked_relevant:
        failures.append(f"the run needs untracked files: {untracked_relevant}")
        print(f"   FAIL — needed but not in git: {untracked_relevant}")
    if not dirty_relevant and not untracked_relevant:
        print("   OK — every file the run needs is committed as tested")

    # ---- 2. fresh clone ---------------------------------------------------
    print("\n2. FRESH CLONE")
    tmp = Path(tempfile.mkdtemp(prefix="g1_"))
    src = args.remote or str(root)
    r = run(["git", "clone", "--depth", "1", src, str(tmp / "repo")])
    if r.returncode != 0:
        print(f"   FAIL — clone failed: {r.stderr.strip()[:200]}")
        failures.append("git clone failed")
        clone = None
    else:
        clone = tmp / "repo"
        head = run(["git", "rev-parse", "--short", "HEAD"], cwd=clone).stdout.strip()
        n = len(list(clone.rglob("*.py")))
        print(f"   cloned {src} at {head} -> {clone} ({n} python files)")

    if clone:
        # ---- 3. import closure -------------------------------------------
        print("\n3. IMPORT CLOSURE")
        missing = [f"{n}.py" for n in sorted(needed)
                   if not (clone / f"{n}.py").exists()]
        if missing:
            failures.append(f"missing from the clone: {missing}")
            print(f"   FAIL — the run imports these but they are not in git: "
                  f"{missing}")
        else:
            print(f"   OK — all {len(needed)} local modules present in the clone")
        cfg_in_clone = clone / args.config
        if not cfg_in_clone.exists():
            failures.append(f"{args.config} is not in the clone")
            print(f"   FAIL — {args.config} not committed")
        else:
            print(f"   OK — {args.config} is committed")

        # ---- 4. compiles --------------------------------------------------
        print("\n4. COMPILES INSIDE THE CLONE")
        bad = []
        for f in sorted(clone.rglob("*.py")):
            try:
                py_compile.compile(str(f), doraise=True, quiet=1)
            except Exception as ex:
                bad.append(f"{f.relative_to(clone)}: {type(ex).__name__}")
        if bad:
            failures.append(f"{len(bad)} file(s) fail to compile in the clone")
            print(f"   FAIL — {bad[:4]}")
        else:
            print(f"   OK — every python file compiles")
        h = run([sys.executable, args.entry, "--help"], cwd=clone)
        if h.returncode != 0:
            failures.append("the entry point's --help fails inside the clone")
            print(f"   FAIL — --help: "
                  f"{(h.stderr or h.stdout).strip().splitlines()[-1][:160]}")
        else:
            print("   OK — the entry point runs --help inside the clone")

    # ---- 5. artifact audit -------------------------------------------------
    print("\n5. ARTIFACT AUDIT")
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    name = cfg.get("model", {}).get("name", cfg.get("experiment", "run"))
    print(f"   run will write:  runs/<{name}_...>/")
    print("     manifest.json, log.jsonl, ckpt.pt, ckpt_best.pt, ckpt_final.pt")
    # check-ignore needs the trailing slash to match a "runs/" rule when the
    # directory does not exist yet, which it will not before the first run
    ignored = any(run(["git", "check-ignore", p_]).returncode == 0
                  for p_ in ("runs/", "runs", "runs/x"))
    print(f"   runs/ gitignored: {'yes' if ignored else 'NO'}")
    if not ignored:
        failures.append("runs/ is not gitignored -- checkpoints would be "
                        "committed")
    dpath = cfg.get("data", {}).get("h5_path", "(unset)")
    print(f"   data path in the config: {dpath}")
    print("     (not required in the clone; stage it on the box separately)")
    print("\n   RETRIEVAL — rehearse this BEFORE launching, per the "
          "retrieval-first rule:")
    print(f"     rsync -avP <user>@<host>:<remote>/runs/  ./runs_pulled/")
    print("     then md5 the checkpoints on both ends before destroying "
          "anything")

    if not args.keep and clone:
        shutil.rmtree(tmp, ignore_errors=True)
    elif clone:
        print(f"\n   clone kept at {clone}")

    print("\n" + "=" * 70)
    if failures:
        print(f"G1 FAILED — {len(failures)} problem(s). Do not launch.")
        print("=" * 70)
        for f_ in failures:
            print(f"  - {f_}")
        sys.exit(1)
    print("G1 PASSED")
    print("=" * 70)
    print("  A fresh clone of HEAD contains everything the run imports, every")
    print("  file compiles, the entry point starts, and outputs are gitignored.")
    print("  Record this alongside the run.")


if __name__ == "__main__":
    main()
