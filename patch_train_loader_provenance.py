"""
patch_train_loader_provenance.py -- make the loader convention config-driven,
and record what was ACTUALLY used in the manifest.

THE GAP THIS CLOSES
-------------------
`tworoom_data.py` carries three module-level flags -- DENSE_ACTIONS,
IMAGENET_PIXELS, ZSCORE_ACTIONS -- that completely determine what the model
sees. None of them reaches the manifest. So a future reader of
`runs/<name>/manifest.json` cannot tell which data convention produced that
checkpoint.

That is not hypothetical. "Which convention did Run 0 use?" is exactly the
question this project could not answer from its own artifacts, and answering it
took a source audit three days into a deadline.

WHAT THIS PATCH DOES
--------------------
1. Reads the three flags from the config's `data:` block, if present, and SETS
   the module flags from them. The config becomes the source of truth instead
   of documentation that can silently disagree with the code.
2. Reads the flags back OFF THE MODULE afterwards and puts those values in the
   manifest. Reading back rather than echoing the config means the manifest
   records what was really in force, even if a flag was left unset and the
   module default applied.
3. Prints the convention at startup so it appears in the run log too.

Defaults are the module's own, so a config without a `data:` flag block behaves
exactly as before.

Usage (from the tinylab folder):
    python3 patch_train_loader_provenance.py
"""
from __future__ import annotations

import difflib
import py_compile
import sys
from pathlib import Path

TARGET = Path("train_toy_lewm.py")
MARKER = "loader_convention"

OLD = '''        run_dir, log, close = lablog.start_run(
            cfg, tag=m["name"],
            extra={"data_sha256": data_fingerprint(index, d["h5_path"]),
                   "n_train_clips": int(len(train_idx)),
                   "n_val_clips": int(len(val_idx)),
                   "n_params": sum(p.numel() for p in model.parameters()),
                   "requirements_file": cfg.get("requirements_file", "requirements.txt")})'''

NEW = '''        run_dir, log, close = lablog.start_run(
            cfg, tag=m["name"],
            extra={"data_sha256": data_fingerprint(index, d["h5_path"]),
                   "n_train_clips": int(len(train_idx)),
                   "n_val_clips": int(len(val_idx)),
                   "n_params": sum(p.numel() for p in model.parameters()),
                   # Read back off the module, not echoed from the config, so
                   # the manifest records what was actually in force.
                   "loader_convention": _loader_convention(),
                   "requirements_file": cfg.get("requirements_file", "requirements.txt")})'''

OLD_IMPORTS = '''from tworoom_data import ClipSpec, TwoRoomClips, TwoRoomIndex'''

NEW_IMPORTS = '''import tworoom_data
from tworoom_data import ClipSpec, TwoRoomClips, TwoRoomIndex

# --- loader convention: config is the source of truth ----------------------
# tworoom_data carries module-level flags that fully determine what the model
# sees. Left implicit they are invisible to the manifest, which is how "which
# convention did Run 0 use?" became a three-day source audit.
_LOADER_FLAGS = ("dense_actions", "imagenet_pixels", "zscore_actions")


def _apply_loader_convention(cfg) -> None:
    """Set tworoom_data's flags from cfg['data'], where the config says so."""
    data_cfg = cfg.get("data", {}) or {}
    for name in _LOADER_FLAGS:
        if name in data_cfg:
            attr = name.upper()
            if not hasattr(tworoom_data, attr):
                raise SystemExit(
                    f"config sets data.{name} but tworoom_data has no {attr}; "
                    f"apply the loader patches first")
            setattr(tworoom_data, attr, bool(data_cfg[name]))


def _loader_convention() -> dict:
    """What is actually in force right now, read off the module."""
    return {name: bool(getattr(tworoom_data, name.upper()))
            for name in _LOADER_FLAGS
            if hasattr(tworoom_data, name.upper())}'''

OLD_CALL = '''    cfg["seed"] = args.seed'''

NEW_CALL = '''    cfg["seed"] = args.seed
    _apply_loader_convention(cfg)
    print(f"loader convention: {_loader_convention()}", flush=True)'''


def main():
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found -- run this from the tinylab folder.")
    src = TARGET.read_text()
    if MARKER in src:
        print(f"'{MARKER}' already present in {TARGET}. No change made.")
        return

    out = src
    for name, old, new in (("imports + helpers", OLD_IMPORTS, NEW_IMPORTS),
                           ("apply at startup", OLD_CALL, NEW_CALL),
                           ("record in manifest", OLD, NEW)):
        cnt = out.count(old)
        if cnt != 1:
            sys.exit(f"ABORT: target for '{name}' appears {cnt} times "
                     f"(need exactly 1). Nothing written.")
        out = out.replace(old, new)

    TARGET.with_suffix(".py.bak").write_text(src)
    TARGET.write_text(out)
    print("--- diff applied ---")
    for line in difflib.unified_diff(src.splitlines(), out.splitlines(),
                                     "before", "after", lineterm="", n=1):
        print(line)
    py_compile.compile(str(TARGET), doraise=True)
    print(f"\nOK: patched and byte-compiles. Backup at {TARGET}.bak")
    print("\nFrom now on every manifest carries loader_convention, and the")
    print("config -- not the module -- decides what the loader does.")


if __name__ == "__main__":
    main()
