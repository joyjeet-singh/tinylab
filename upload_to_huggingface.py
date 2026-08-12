"""Stage and upload the release to HuggingFace.

Decision S3: six files -- the three BatchNorm-recalibrated checkpoints and the
three un-recalibrated originals. The paper's §8 promises six in public, so
fewer would make a published claim false.

    ./.venv/bin/python upload_to_huggingface.py --arxiv 2608.XXXXX --stage
    ./.venv/bin/python upload_to_huggingface.py --arxiv 2608.XXXXX --upload

--stage builds and checks build/hf_upload/ without touching the network.
--upload pushes it, and refuses if anything is unresolved.

The card becomes README.md, which is what HuggingFace renders. Every
checkpoint is re-hashed against the committed manifest immediately before
upload: the manifest is the tracked artifact the paper cites, and a file that
does not match it is not the file the paper describes.
"""
import argparse
import hashlib
import re
import shutil
from pathlib import Path

CARD = Path("MODEL_CARD.md")
RELEASE = Path("runs_archive/release")
MANIFEST = Path("runs_archive/verified/ckpt_md5.txt")
STAGE = Path("build/hf_upload")
DEFAULT_REPO = "joyjeet-singh/tinylab-tworoom-lewm"

ap = argparse.ArgumentParser()
ap.add_argument("--arxiv", required=True,
                help="the arXiv identifier, e.g. 2608.01234")
ap.add_argument("--repo", default=DEFAULT_REPO)
ap.add_argument("--stage", action="store_true")
ap.add_argument("--upload", action="store_true")
ap.add_argument("--private", action="store_true")
args = ap.parse_args()

assert re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", args.arxiv), (
    f"{args.arxiv!r} does not look like an arXiv identifier (NNNN.NNNNN)")

# ---- md5s, against the manifest the paper cites ----------------------
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


man = MANIFEST.read_text()
files = sorted(RELEASE.glob("*.pt"))
assert len(files) == 6, f"expected 6 checkpoints, found {len(files)}"
print("verifying against the committed manifest:")
for p in files:
    m = re.search(rf"^{re.escape(p.name)}\n.*?md5 after\s+(\w+)", man, re.S | re.M)
    assert m, f"{p.name} is not in {MANIFEST}"
    got = md5(p)
    assert got == m.group(1), (
        f"{p.name} does not match the manifest: {got} vs {m.group(1)}. This is "
        f"not the file the paper describes; stop and find out why.")
    print(f"  OK  {p.name:<34} {got[:16]}…")

# ---- the card ---------------------------------------------------------
card = CARD.read_text()
assert "<ARXIV_ID>" in card, "the card has no <ARXIV_ID> left to fill"
card = card.replace("<ARXIV_ID>", args.arxiv)
card = card.replace("arXiv ID of this reproduction", args.arxiv)
left = re.findall(r"<[A-Z_]+>", card)
assert not left, f"placeholders remain in the card: {sorted(set(left))}"

# HuggingFace reads YAML frontmatter as the model's metadata: without it the
# page carries no licence badge, no link to the paper and no tags. The licence
# is MIT to match the reference implementation (decision S1). The arxiv tag is
# how HF links a model to its paper.
FRONTMATTER = f"""---
license: mit
library_name: pytorch
tags:
  - world-model
  - jepa
  - planning
  - reproducibility
  - reproduction-study
  - arxiv:{args.arxiv}
---

"""

if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)
(STAGE / "README.md").write_text(FRONTMATTER + card)
shutil.copy2(MANIFEST, STAGE / "ckpt_md5.txt")
for p in files:
    shutil.copy2(p, STAGE / p.name)

total = sum(f.stat().st_size for f in STAGE.iterdir())
print(f"\nstaged {STAGE}  ({total / 1e6:.0f} MB)")
for f in sorted(STAGE.iterdir()):
    print(f"  {f.name:<34} {f.stat().st_size / 1e6:8.1f} MB")
print(f"\narXiv identifier written into the card: {args.arxiv}")

# The dataset is the authors'. Releasing it here would be redistributing
# someone else's data under our name.
assert not any(f.suffix in (".h5", ".hdf5") for f in STAGE.iterdir()), \
    "the dataset must not be uploaded; link to the authors' release instead"
assert "not redistribute" in card.lower() or "do not redistribute" in card.lower(), \
    "the card must say the dataset is not redistributed"
print("dataset is not in the upload, and the card says so")

if not args.upload:
    print("\n--stage only; nothing was uploaded.")
    raise SystemExit(0)

# ---- upload -----------------------------------------------------------
from huggingface_hub import HfApi

api = HfApi()
who = api.whoami()            # raises if not authenticated
print(f"\nauthenticated as {who.get('name')}")
url = api.create_repo(args.repo, repo_type="model", exist_ok=True,
                      private=args.private)
print(f"repo: {url}")
api.upload_folder(folder_path=str(STAGE), repo_id=args.repo, repo_type="model",
                  commit_message=f"Release six checkpoints for arXiv:{args.arxiv}")
print(f"\nuploaded to https://huggingface.co/{args.repo}"
      f"  ({'private' if args.private else 'PUBLIC'})")
