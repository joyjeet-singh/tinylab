"""
patch_toy_plan_banner.py -- fix the closing banner in toy_plan.py. One anchored
replacement; refuses loudly if the anchor is not found exactly once. Idempotent.

WHY THE OLD BANNER IS HALF WRONG AND HALF LOAD-BEARING
------------------------------------------------------
Old text:  "This is the TOY world with a CNN encoder -- the number here is NOT
            comparable to 87 and is not a result. It only tells us the planning
            loop works end to end before we pay for compute."

  - "CNN encoder" is STALE: the script builds whatever encoder the run's
    manifest says (the Phase 1 runs use the ViT at 224px). Fixed by reading
    the manifest instead of hardcoding.
  - "TOY world" is TRUE and must stay: toy_plan.evaluate() does
    `env = ToyTwoRoom()` -- every planning episode, every rendered frame the
    encoder sees at eval time, comes from the synthetic simulator in
    make_toy_tworoom.py, whose own docstring calls it "a debugging fixture,
    not data". For a model TRAINED on the real tworoom.h5, planner numbers
    therefore include a train->eval rendering-domain gap on top of model
    quality. Deleting that sentence as "stale" would have erased the one line
    that flags the confound.
  - "not a result ... before we pay for compute" is stale in the other
    direction: this script is now run on paid checkpoints, so the banner
    should state the conditions of the measurement, not call it a smoke test.

Run from the tinylab folder:  python3 patch_toy_plan_banner.py
"""
from pathlib import Path

p = Path("toy_plan.py")
src = p.read_text()

old = '''    print("  Reference LeWM scores 87% on the REAL TwoRoom (PLDM/DINO-WM: 97-100).")
    print("  This is the TOY world with a CNN encoder -- the number here is NOT")
    print("  comparable to 87 and is not a result. It only tells us the planning")
    print("  loop works end to end before we pay for compute.")
'''

new = '''    _m = json.loads((run_dir / "manifest.json").read_text())["config"]
    print("  Reference LeWM reports 87% on the REAL TwoRoom env (PLDM/DINO-WM: 97-100).")
    print(f"  THIS eval runs in the synthetic ToyTwoRoom simulator, encoded by the")
    print(f"  run's own {_m['model'].get('encoder', 'cnn')} encoder at "
          f"{_m['model'].get('img_size', 32)}px (trained on {Path(_m['data']['h5_path']).name}).")
    print("  If training data was the real tworoom.h5, numbers here include a")
    print("  train->eval RENDERING-DOMAIN GAP and are not comparable to 87.")
    print("  Success criteria: success_radius=3.0, budget 50 env steps, start and")
    print("  goal sampled in opposite rooms.")
'''

if new in src and old not in src:
    print("already applied: banner is the manifest-driven version")
    raise SystemExit(0)

assert src.count(old) == 1, (
    "STOP: old banner not found exactly once in toy_plan.py; file unchanged")

src = src.replace(old, new)
p.write_text(src)
print("applied: banner now reads encoder/img/data from the run manifest and")
print("states the eval world (synthetic ToyTwoRoom) and the success criteria.")
