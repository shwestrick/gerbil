#!/usr/bin/env bash
# Apply each gerbil-produced patch in order, as a series of real commits.
#
# Usage: apply-gerbil.sh [DIR]   (DIR defaults to the current directory)
#
# Each .gerbil/gerbil-*.patch is a `git format-patch` mbox (title + message +
# diff), so `git am` replays it as a proper commit -- message and authorship
# included, no separate commit-message file needed. Zero-padded numbering means
# the glob is already in session order. Run it from (or point it at) the Lean
# project repo gerbil worked on.
set -euo pipefail

cd "${1:-.}"

shopt -s nullglob
patches=(.gerbil/gerbil-*.patch)
(( ${#patches[@]} )) || { echo "no .gerbil/gerbil-*.patch files found" >&2; exit 1; }

for patch in "${patches[@]}"; do
    echo "applying $patch"
    git am "$patch"
done

echo "done"
