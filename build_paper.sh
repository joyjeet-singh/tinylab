#!/bin/zsh
# Build the paper from docs/paper/PAPER.md.
#
#   ./build_paper.sh          convert and build the PDF
#   ./build_paper.sh --tex    convert to LaTeX only, no PDF
#
# The Markdown is the source of truth. paper.tex is generated and should never
# be hand-edited -- edits belong in PAPER.md, patched with the harness in the
# work order's Appendix A, so that the numbers gate still guards them.
#
# Figures live at the repository root; PAPER.md references them as
# ../../fig*.png so they resolve when the Markdown is rendered from
# docs/paper/. --resource-path lets pandoc resolve the same references, and
# \graphicspath lets LaTeX find them at build time.

set -e
cd "$(dirname "$0")"

SRC="docs/paper/PAPER.md"
OUT="build"
TEX="$OUT/paper.tex"

command -v pandoc >/dev/null || { echo "pandoc is not installed"; exit 1; }
mkdir -p "$OUT"
cp docs/paper/paper.bib "$OUT/"

pandoc "$SRC" \
  --from=markdown+yaml_metadata_block+pipe_tables+tex_math_dollars \
  --to=latex \
  --standalone \
  --resource-path="docs/paper:." \
  --bibliography=docs/paper/paper.bib \
  --top-level-division=section \
  --variable=documentclass:article \
  --variable=classoption:11pt \
  --variable=geometry:margin=1in \
  --variable=colorlinks:true \
  --variable=graphics:true \
  --output="$TEX"

# Figures are referenced relative to docs/paper/; tell LaTeX where they are.
/usr/bin/sed -i '' 's|\\begin{document}|\\graphicspath{{../}}\n\\begin{document}|' "$TEX"

echo "wrote $TEX"
[ "$1" = "--tex" ] && exit 0

cd "$OUT"
latexmk -pdf -interaction=nonstopmode -halt-on-error paper.tex
echo
echo "built $OUT/paper.pdf"
