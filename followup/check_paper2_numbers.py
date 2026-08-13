"""Gate: every load-bearing number in followup/PAPER2.md is read from a file.

Same rule as the reproduction it builds on -- a figure that cannot be traced
to an artifact on disk does not belong in the paper. Run from the repository
root:

    python3 followup/check_paper2_numbers.py
"""
import json
import re
import sys
from pathlib import Path

PAPER = Path("followup/PAPER2.md").read_text()
fail, checked = [], 0


def check(label, value, source, note=""):
    """Assert the paper states `value`.

    Compares the number, not the glyphs: the paper sets a typographic minus
    (U+2212) where an artifact writes a hyphen, and rounds 0.8191 to 0.819.
    Neither is a discrepancy. A different VALUE still fails.
    """
    global checked
    checked += 1
    variants = {value, value.replace("-", "\u2212")}
    m = re.fullmatch(r"([+\u2212-]?)([\d.]+)(%?)", value)
    if m:
        sign, num, pct = m.groups()
        for dp in (2, 3):
            try:
                r = f"{round(float(num), dp):.{dp}f}".rstrip("0").rstrip(".")
            except ValueError:
                continue
            for sg in {sign, sign.replace("-", "\u2212")}:
                variants.add(f"{sg}{r}{pct}")
                variants.add(f"{sg}{float(num):.{dp}f}{pct}")
    hit = next((v for v in variants if v in PAPER), None)
    print(f"  {'OK  ' if hit else 'MISS'}  {label:<44} {value:<12} {source}{note}")
    if not hit:
        fail.append(f"{label} = {value} (from {source})")


def grab(path, pattern, group=1):
    m = re.search(pattern, Path(path).read_text(), re.M)
    assert m, f"pattern not found in {path}: {pattern}"
    return m.group(group)


print("PAPER2.md -- every load-bearing figure against its artifact\n")

# ---- §3 rollout ------------------------------------------------------
f = "followup/rollout_h15_phase2_recal.txt"
for h, col in ((1, "0.066"), (5, "0.090"), (10, "0.152"), (15, "0.189")):
    v = grab(f, rf"^\s+{h}\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", 1)
    check(f"rollout err/static, horizon {h}", v, f)

# ---- §4 latent metric ------------------------------------------------
f = "followup/latent_metric_phase2_recal.txt"
check("latent metric Pearson (phase2)",
      f"{float(grab(f, r'Pearson r over all pairs\s*:\s*([\d.]+)')):.3f}", f)
check("latent far-band spread (phase2)",
      grab(f, r"changes by ([+-][\d.]+)%") + "%", f)

# ---- §5.1 authors' metric -------------------------------------------
f = "followup/latent_metric_authors.txt"
check("authors metric Pearson",
      f"{float(grab(f, r'Pearson r over all pairs\s*:\s*([\d.]+)')):.3f}", f)
check("authors metric Spearman",
      f"{float(grab(f, r'Spearman \(rank\) over all pairs\s*:\s*([\d.]+)')):.3f}", f)
check("authors far-band spread",
      grab(f, r"changes by ([+-][\d.]+)%") + "%", f)

# ---- §5.2 the four-checkpoint table ---------------------------------
for name, f in (("Run 2", "followup/latent_metric_run2_recal.txt"),
                ("Run 0", "followup/latent_metric_run0_recal.txt")):
    check(f"{name} metric Spearman",
          f"{float(grab(f, r'Spearman \(rank\) over all pairs\s*:\s*([\d.]+)')):.3f}", f)
    check(f"{name} far-band spread",
          grab(f, r"changes by ([+-][\d.]+)%") + "%", f)

# ---- §5.2 one-step errors -------------------------------------------
check("phase2 one-step error",
      grab("runs_archive/verified/driving_spec_phase2_recal.txt",
           r"repeat of MEAN \(displacement-matched\)\s+[\d.]+\s+([\d.]+)"),
      "runs_archive/verified/driving_spec_phase2_recal.txt")
check("authors one-step error",
      grab("runs_archive/verified/repeat_encoding_authors.txt",
           r"repeat \(constant-action planner\)\s+[\d.]+\s+([\d.]+)"),
      "runs_archive/verified/repeat_encoding_authors.txt")
check("Run 2 one-step error",
      grab("runs_archive/verified/check_a_step0_run2.txt",
           r"step0_err\s+[\d.]+\s+real_step\s+[\d.]+\s+ratio\s+([\d.]+)"),
      "runs_archive/verified/check_a_step0_run2.txt")

# ---- §5.3 effective rank --------------------------------------------
f = "runs_archive/verified/encoder_probe_both_recal.txt"
ranks = re.findall(r"effective rank\s+([\d.]+) of 192", Path(f).read_text())
check("Run 0 effective rank", ranks[0], f)
check("phase2 effective rank", ranks[1], f)

# ---- §6 probe --------------------------------------------------------
f = "followup/probe_metric_phase2_recal.txt"
check("probe R2 held out (phase2)", grab(f, r"probe position R\^2 \(held out\) : ([\d.]+)"), f)
check("probe MAE (phase2)", grab(f, r"mean absolute error\s+:\s+([\d.]+)"), f)
check("decoded-position r (phase2)",
      grab(f, r"decoded position : ([\d.]+)"), f)
f = "followup/probe_metric_authors.txt"
check("probe R2 held out (authors)", grab(f, r"probe position R\^2 \(held out\) : ([\d.]+)"), f)
check("decoded-position r (authors)", grab(f, r"decoded position : ([\d.]+)"), f)

# ---- §7 temporal head -----------------------------------------------
f = "followup/temporal_head_phase2.txt"
check("temporal head r vs true distance",
      grab(f, r"learned temporal head\s+:\s+([\d.]+)"), f)
check("temporal head latent-L2 r",
      grab(f, r"latent L2 \(the current objective\) : ([\d.]+)"), f)
check("temporal head held-out MAE",
      grab(f, r"held-out MAE\s+:\s+([\d.]+)"), f)

# ---- §8 planning results, and the baselines they are paired against --
def rate(path, key="results"):
    r = json.load(open(path))[key]
    return sum(e["success"] for e in r), len(r)


for label, path in (("baseline offset 25", "exp_phase2_recal_25/realenv_plan_cem.json"),
                    ("baseline offset 100", "exp_ref_p2/realenv_plan_cem.json"),
                    ("baseline offset 100 budget 50",
                     "exp_phase2_recal/realenv_plan_cem.json"),
                    ("baseline authors offset 100",
                     "exp_ref_protocol/realenv_plan_authors_cem.json")):
    k, n = rate(path)
    check(label, f"{k}/{n}", path, f"  ({100*k/n:.1f}%)")

for label, d in (("follow-up offset 25", "followup/probe_off25"),
                 ("follow-up offset 100", "followup/probe_off100"),
                 ("follow-up authors offset 100", "followup/probe_authors_off100"),
                 ("temporal-head offset 100", "followup/temporal_off100"),
                 ("temporal-head offset 25", "followup/temporal_off25"),
                 ("temporal-head offset 100 budget 50",
                  "followup/temporal_off100_b50"),
                 ("temporal-head v2 authors offset 100",
                  "followup/temporal_v2_authors_off100")):
    log = Path(f"{d}/run.log")
    if not log.exists():
        continue
    rows = re.findall(r"ep\s+\d+/\d+ \(#\s*\d+\): (REACHED|missed )", log.read_text())
    k, n = sum(r == "REACHED" for r in rows), len(rows)
    if n:
        check(label, f"{k}/{n}", str(log), f"  ({100*k/n:.1f}%)")

# ---- §7.4 the distribution condition
for label, f, pat in (
        ("v1 head on ours, real", "followup/temporal_head_v2_phase2_smoke.txt",
         r"v1 MAE, real x real\s+:\s+([\d.]+)"),
        ("v1 head on ours, imagined", "followup/temporal_head_v2_phase2_smoke.txt",
         r"v1 MAE, imagined x real\s+:\s+([\d.]+)"),
        ("v1 head on authors, real", "followup/temporal_head_v2_authors.txt",
         r"v1 MAE, real x real\s+:\s+([\d.]+)"),
        ("v1 head on authors, imagined", "followup/temporal_head_v2_authors.txt",
         r"v1 MAE, imagined x real\s+:\s+([\d.]+)"),
        ("v2 head on authors, real", "followup/temporal_head_v2_authors.txt",
         r"held-out MAE, real x real\s+:\s+([\d.]+)"),
        ("v2 head on authors, imagined", "followup/temporal_head_v2_authors.txt",
         r"held-out MAE, imagined x real\s+:\s+([\d.]+)")):
    if Path(f).exists():
        check(label, grab(f, pat), f)

# ---- §7.3 reachability
f = "followup/wall_reachability.txt"
if Path(f).exists():
    check("temporal head cross/same ratio",
          grab(f, r"temporal head : ([\d.]+)"), f)
    check("latent L2 cross/same ratio",
          grab(f, r"latent L2     : ([\d.]+)"), f)

print()
if fail:
    print(f"{len(fail)} of {checked} figures could not be matched to the paper:")
    for x in fail:
        print("  -", x)
    sys.exit(1)
print(f"all {checked} load-bearing figures in PAPER2.md trace to an artifact on disk.")
