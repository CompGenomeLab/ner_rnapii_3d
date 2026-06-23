#!/usr/bin/env bash
#
# Build the supplementary methods as both PDF and DOCX from a single .tex source.
#
#   PDF  : tectonic           -> keeps the vector (PDF) figure, full LaTeX fidelity.
#   DOCX : pandoc             -> editable Word file. Word cannot embed the vector PDF
#                                figure, so it is rasterised to PNG and the DOCX is
#                                built from a TEMPORARY copy of the .tex whose figure
#                                path points at that PNG. The original .tex is never
#                                modified.
#
# Tools (tectonic, pandoc, pdftocairo) are taken from the micromamba 'docs' env.
# Usage:  bash compile_tex.sh
#
set -euo pipefail
cd "$(dirname "$0")"

TEX="supplementary_methods.tex"         # source
OUT="supplementary_methods"             # output basename (.pdf / .docx)
FIG="transcription_distributions_pub"   # figure basename; LaTeX uses ${FIG}.pdf
RUN="micromamba run -n docs"            # env providing tectonic, pandoc, pdftocairo

TMP_TEX="._docx_build.tex"              # temporary .tex copy used only for the DOCX build
TMP_PNG="${FIG}.png"                    # rasterised figure used only for the DOCX build
cleanup() { rm -f "$TMP_TEX" "$TMP_PNG"; }
trap cleanup EXIT                       # always remove the build artefacts, even on error

# --------------------------------------------------------------------- PDF ----
echo ">> PDF  (tectonic): ${OUT}.pdf"
$RUN tectonic "$TEX"

# -------------------------------------------------------------------- DOCX ----
echo ">> DOCX (pandoc):   ${OUT}.docx"
# 1. rasterise the vector figure (PDF -> PNG) so Word can embed it
$RUN pdftocairo -png -r 150 -singlefile "${FIG}.pdf" "${FIG}"
# 2. temp copy of the source with the figure path swapped to the PNG (original .tex untouched)
sed "s|${FIG}\.pdf|${FIG}.png|g" "$TEX" > "$TMP_TEX"
# 3. convert to DOCX (numbered sections + table of contents)
$RUN pandoc "$TMP_TEX" \
    --from=latex --to=docx \
    --toc --number-sections \
    --resource-path=. \
    -o "${OUT}.docx"

echo ">> done."
ls -la "${OUT}.pdf" "${OUT}.docx"
