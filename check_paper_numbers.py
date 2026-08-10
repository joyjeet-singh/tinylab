"""Gate: every percentage in PAPER.md is either a disk-verified planning figure
or an explicitly allowlisted exception. Run before submission."""
import csv, json, pathlib, re, sys

paper = pathlib.Path('docs/paper/PAPER.md').read_text()
disk = {f"{float(r['pct']):.1f}" for r in
        csv.DictReader(open('docs/paper/results_from_disk.csv'))
        if r.get('nested_copy') == 'no'}
allow_path = pathlib.Path('docs/paper/number_allowlist.json')
allow = json.loads(allow_path.read_text()) if allow_path.exists() else {}

unmatched = {}
for i, line in enumerate(paper.split('\n'), 1):
    for m in re.finditer(r'(\d+\.\d)\s*%', line):
        v = m.group(1)
        if v in disk or v in allow:
            continue
        unmatched.setdefault(v, []).append((i, line.strip()[:78]))

if unmatched:
    print(f"{len(unmatched)} percentage(s) neither disk-verified nor allowlisted:\n")
    for v, hits in sorted(unmatched.items(), key=lambda kv: float(kv[0])):
        print(f"  {v}%")
        for i, l in hits:
            print(f"      line {i}: {l}")
    print('\nAdd each to docs/paper/number_allowlist.json, e.g.')
    print('  {"79.1": "same-room rate, §5.2 arm 1, exp_balanced_wall/realenv_plan_cem_report.txt"}')
    sys.exit(1)
print(f"OK — {len(disk)} disk figures, {len(allow)} allowlisted, 0 unaccounted.")
