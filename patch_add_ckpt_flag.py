"""
patch_add_ckpt_flag.py -- let the evaluation tools load a named checkpoint.

Recalibration writes a NEW checkpoint (ckpt_best_recal.pt) beside the original.
Three scripts hard-code "ckpt_best.pt, else ckpt.pt" and so cannot see it:

    verify_phase2_driving.py     the driving-spec measurement
    probe_encoder_comparison.py  the encoder probe
    realenv_r2_planner_eval.py   the planner

This adds --ckpt to each. Default behaviour is unchanged, so every previously
committed result reproduces exactly.

Each file is patched independently: a file that is missing, already patched, or
whose target text does not match is reported and skipped, not aborted. Run it
again after fixing anything it reports.

Usage (from the tinylab folder):
    python3 patch_add_ckpt_flag.py
"""
from __future__ import annotations

import py_compile
from pathlib import Path

# (file, marker, [(name, old, new), ...])
JOBS = [
    ("verify_phase2_driving.py", "--ckpt", [
        ("argparse",
         '''    ap.add_argument("--seed", type=int, default=7)''',
         '''    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ckpt", default=None,
                   help="checkpoint filename inside the run dir; defaults to "
                        "ckpt_best.pt, else ckpt.pt")'''),
        ("load",
         '''    ck = torch.load(run_dir / ("ckpt_best.pt"
                               if (run_dir / "ckpt_best.pt").exists()
                               else "ckpt.pt"),
                    map_location="cpu", weights_only=False)''',
         '''    _ck_name = args.ckpt or ("ckpt_best.pt"
                            if (run_dir / "ckpt_best.pt").exists()
                            else "ckpt.pt")
    print(f"  checkpoint: {_ck_name}")
    ck = torch.load(run_dir / _ck_name, map_location="cpu",
                    weights_only=False)'''),
    ]),

    ("probe_encoder_comparison.py", "def load_encoder(run_dir, ckpt=None)", [
        ("signature",
         '''def load_encoder(run_dir):''',
         '''def load_encoder(run_dir, ckpt=None):'''),
        ("load",
         '''    ck = torch.load(run_dir / ("ckpt_best.pt"
                               if (run_dir / "ckpt_best.pt").exists()
                               else "ckpt.pt"),
                    map_location="cpu", weights_only=False)''',
         '''    _ck_name = ckpt or ("ckpt_best.pt"
                        if (run_dir / "ckpt_best.pt").exists() else "ckpt.pt")
    ck = torch.load(run_dir / _ck_name, map_location="cpu",
                    weights_only=False)'''),
        ("argparse",
         '''    ap.add_argument("--seed", type=int, default=0)''',
         '''    ap.add_argument("--ckpt-a", default=None,
                   help="checkpoint filename inside --run-a")
    ap.add_argument("--ckpt-b", default=None,
                   help="checkpoint filename inside --run-b")
    ap.add_argument("--seed", type=int, default=0)'''),
        ("call site",
         '''    for label, run in ((args.label_a, args.run_a), (args.label_b, args.run_b)):
        model, conv, m = load_encoder(run)''',
         '''    for label, run, _ck in ((args.label_a, args.run_a, args.ckpt_a),
                            (args.label_b, args.run_b, args.ckpt_b)):
        model, conv, m = load_encoder(run, _ck)'''),
    ]),

    ("realenv_r2_planner_eval.py", "--ckpt", [
        ("argparse",
         '''    p.add_argument("--action-scale", type=float, default=1.0,''',
         '''    p.add_argument("--ckpt", default=None,
                   help="checkpoint filename inside the run dir; defaults to "
                        "ckpt_best.pt, else ckpt.pt")
    p.add_argument("--action-scale", type=float, default=1.0,'''),
        ("load",
         '''        ckpt = run_dir / ("ckpt_best.pt"
                          if (run_dir / "ckpt_best.pt").exists() else "ckpt.pt")''',
         '''        ckpt = run_dir / (args.ckpt or
                          ("ckpt_best.pt"
                           if (run_dir / "ckpt_best.pt").exists()
                           else "ckpt.pt"))'''),
    ]),
]


def main():
    any_done = False
    for fname, marker, edits in JOBS:
        f = Path(fname)
        print(f"\n{fname}")
        if not f.exists():
            print("  SKIP — not found here")
            continue
        src = f.read_text()
        if marker in src:
            print("  SKIP — already patched")
            continue
        out, bad = src, []
        for name, old, new in edits:
            c = out.count(old)
            if c != 1:
                bad.append(f"{name} appears {c}x")
                continue
            out = out.replace(old, new)
        if bad:
            print(f"  SKIP — targets did not match: {'; '.join(bad)}")
            print("  Send me this file's argparse block and its checkpoint "
                  "load and I will retarget the patch.")
            continue
        f.with_suffix(".py.bak_ckpt").write_text(src)
        f.write_text(out)
        try:
            py_compile.compile(str(f), doraise=True)
        except Exception as ex:
            f.write_text(src)
            print(f"  REVERTED — patched file did not compile: {ex}")
            continue
        print(f"  OK — --ckpt added, compiles, backup at {f}.bak_ckpt")
        any_done = True

    if any_done:
        print("\nNow, on the recalibrated checkpoint:")
        print("  python3 verify_phase2_driving.py --run <phase2 run> "
              "--ckpt ckpt_best_recal.pt")


if __name__ == "__main__":
    main()
