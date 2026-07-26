"""
patch_r2_authors_fix.py -- finish the job patch_r2_authors.py started.

WHAT I GOT WRONG
----------------
My first patch swapped the MODEL correctly but left the manifest read and our
ToyJEPA construction above the branch, so the evaluator still demanded a valid
run directory even when it was about to throw that model away. With the
authors' weights the run directory carries no information at all, so it should
not be required -- and because it was, an unset $RUN2 turned `--run ../$RUN2`
into `--run ..` and the run died on a missing manifest.json.

This patch moves the manifest read and the ToyJEPA construction inside the
else-branch, so:

  * with --authors-spec, no manifest, no checkpoint, no run directory needed
  * without it, behaviour is byte-identical to before

It also pins action_dim to 2 for the planner in that mode, which is what our
planner emits; the adapter widens it to their 10 on the way in.

Same construction as the previous patches, for the same reason (the Run-2
double-apply incident): marker check first, exact-once assertions, backup,
diff, byte-compile.

Usage (from the tinylab folder):
    python3 patch_r2_authors_fix.py
"""
from __future__ import annotations

import difflib
import py_compile
import sys
from pathlib import Path

TARGET = Path("realenv_r2_planner_eval.py")
MARKER = "m, model = {\"action_dim\": 2}, None"

OLD = '''    cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
    m = cfg["model"]
    model = ToyJEPA(embed_dim=m["embed_dim"], action_dim=m["action_dim"],
                    history_size=m["history_size"], depth=m["depth"],
                    heads=m["heads"], dim_head=m["dim_head"],
                    mlp_dim=m["mlp_dim"], proj_hidden=m["proj_hidden"],
                    dropout=m["dropout"], enc_width=m["enc_width"],
                    encoder=m.get("encoder", "cnn"),
                    img_size=m.get("img_size", 32),
                    patch_size=m.get("patch_size", 4),
                    enc_depth=m.get("enc_depth", 12),
                    enc_heads=m.get("enc_heads", 3))'''

NEW = '''    if args.authors_spec:
        # their weights carry their own architecture; the run dir is unused
        m, model = {"action_dim": 2}, None
    else:
        cfg = json.loads((run_dir / "manifest.json").read_text())["config"]
        m = cfg["model"]
        model = ToyJEPA(embed_dim=m["embed_dim"], action_dim=m["action_dim"],
                        history_size=m["history_size"], depth=m["depth"],
                        heads=m["heads"], dim_head=m["dim_head"],
                        mlp_dim=m["mlp_dim"], proj_hidden=m["proj_hidden"],
                        dropout=m["dropout"], enc_width=m["enc_width"],
                        encoder=m.get("encoder", "cnn"),
                        img_size=m.get("img_size", 32),
                        patch_size=m.get("patch_size", 4),
                        enc_depth=m.get("enc_depth", 12),
                        enc_heads=m.get("enc_heads", 3))'''


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()

    if "--authors-spec" not in src:
        sys.exit("run patch_r2_authors.py first; this patch builds on it.")
    if MARKER in src:
        print(f"already applied -- {TARGET} needs no run directory when "
              f"--authors-spec is used. No change made.")
        return

    cnt = src.count(OLD)
    if cnt != 1:
        sys.exit(f"ABORT: the model-construction block appears {cnt} times "
                 f"(need exactly 1). Nothing written.")

    out = src.replace(OLD, NEW)
    if out.count(MARKER) != 1:
        sys.exit("ABORT: post-patch marker count wrong. Nothing written.")

    TARGET.with_suffix(".py.bak3").write_text(src)
    TARGET.write_text(out)
    print("--- diff applied ---")
    for line in difflib.unified_diff(src.splitlines(), out.splitlines(),
                                     "before", "after", lineterm="", n=1):
        print(line)
    py_compile.compile(str(TARGET), doraise=True)
    print(f"\nOK: patched and byte-compiles. Backup at {TARGET}.bak3")
    print("--run is now ignored when --authors-spec is given.")


if __name__ == "__main__":
    main()
