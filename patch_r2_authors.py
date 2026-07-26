"""
patch_r2_authors.py -- add --authors-spec to realenv_r2_planner_eval.py so the
calibration run uses the EXISTING, already-committed evaluator with the
authors' weights swapped in and nothing else changed.

Swapping only the weights is the entire design of the experiment. Every other
element -- episode selection, goal convention, success rule, budget, CEM
settings, the domain guard, the deviations table -- stays byte-identical to the
run that produced our 72%, so the comparison is clean.

Built against the Run-2 double-apply incident, like the last patch:
  - checks for its own marker first and exits without touching the file
  - every target must appear EXACTLY ONCE or it aborts before writing
  - writes a .bak, prints the diff, byte-compiles the result

Usage (from the tinylab folder):
    python3 patch_r2_authors.py
"""
from __future__ import annotations

import difflib
import py_compile
import sys
from pathlib import Path

TARGET = Path("realenv_r2_planner_eval.py")
MARKER = "--authors-spec"

EDITS = [
    ("argparse flag",
     '''    p.add_argument("--unsafe-skip-guard", action="store_true",''',
     '''    p.add_argument("--authors-spec", default=None,
                   help="authors_driving_spec.json: evaluate the AUTHORS' "
                        "released checkpoint instead of ours, with every "
                        "other protocol element unchanged")
    p.add_argument("--unsafe-skip-guard", action="store_true",'''),

    ("model construction",
     '''    ckpt = run_dir / ("ckpt_best.pt" if (run_dir / "ckpt_best.pt").exists()
                      else "ckpt.pt")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    say(f"model: {m.get('encoder', 'cnn')} at {m.get('img_size', 32)}px "
        f"({ckpt.name}, epoch field {ck.get('epoch')})")''',
     '''    if args.authors_spec:
        from authors_adapter import load_authors_model
        say("MODEL: the AUTHORS' released checkpoint (calibration run)")
        model = load_authors_model(args.authors_spec)
        ckpt = Path(args.authors_spec)
        ck = {"epoch": -1}
        say("  every other protocol element is unchanged from our own run")
    else:
        ckpt = run_dir / ("ckpt_best.pt"
                          if (run_dir / "ckpt_best.pt").exists() else "ckpt.pt")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        say(f"model: {m.get('encoder', 'cnn')} at {m.get('img_size', 32)}px "
            f"({ckpt.name}, epoch field {ck.get('epoch')})")'''),

    ("tag the outputs so they never collide with ours",
     '''    tag = "random" if args.random else "cem"''',
     '''    tag = ("authors_" if args.authors_spec else "") + \\
          ("random" if args.random else "cem")'''),
]


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()

    if MARKER in src:
        print(f"'{MARKER}' already present in {TARGET} -- patch was applied "
              f"before. No change made.")
        return

    out = src
    for name, old, new in EDITS:
        cnt = out.count(old)
        if cnt != 1:
            sys.exit(f"ABORT: target for '{name}' appears {cnt} times "
                     f"(need exactly 1). Nothing written.")
        out = out.replace(old, new)

    if out.count(MARKER) != 1:
        sys.exit(f"ABORT: post-patch marker count {out.count(MARKER)}, "
                 f"expected 1. Nothing written.")

    TARGET.with_suffix(".py.bak2").write_text(src)
    TARGET.write_text(out)
    print("--- diff applied ---")
    for line in difflib.unified_diff(src.splitlines(), out.splitlines(),
                                     "before", "after", lineterm="", n=1):
        print(line)
    py_compile.compile(str(TARGET), doraise=True)
    print(f"\nOK: {TARGET} patched and byte-compiles. Backup at "
          f"{TARGET}.bak2")
    print("Re-run to confirm it now reports 'already present'.")


if __name__ == "__main__":
    main()
