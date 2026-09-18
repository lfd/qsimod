#! /bin/bash

# Renders the TikZ figures into standalone PDFs: img-tikz/*.tex -> img-gen/*.pdf.
#
# plot.r is deliberately not run here.  The Makefile's `plot` target does that and `all`
# sequences the two, so running it again from this script would drive R twice per build.
# Invoke this on its own only when img-tikz/ is already current.

set -euo pipefail
# An unmatched glob must not reach lualatex as a literal filename.
shopt -s nullglob

# Work from the directory holding this script, so it runs the same from anywhere.
cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p img-gen build

figures=(img-tikz/*.tex)
if [ ${#figures[@]} -eq 0 ]; then
    echo "gen_img.sh: no figures in img-tikz/; run 'make plot' first" >&2
    exit 1
fi

for file in "${figures[@]}"; do
    file_base=$(basename "$file")
    job=$(basename "$file_base" .tex)

    ## NOTE: We use lualatex on purpose here because it lifts any memory
    ## limitations that classical TeX implementations have, which is important
    ## for complicated TikZ pictures.
    sed -e "s/FILE/$file_base/" img.tex |
	lualatex -output-directory build/ -jobname "$job"
    mv "build/$job.pdf" img-gen/
done
