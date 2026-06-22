#!/usr/bin/env bash
# Apply and commit each gerbil-produced patch in order, as a series of commits.
#
# Usage: apply-gerbil.sh [DIR]   (DIR defaults to the current directory)
#
# Walks the .gerbil/gerbil-*.commit files (zero-padded, so they sort in session
# order) and for each one:
#   - applies the matching .patch, staging only its changes
#   - also stages the matching .jsonl session log (committed alongside)
#   - commits with the generated message
#   - deletes the .patch and .commit files (the .jsonl stays, now tracked)
# Run it from (or point it at) the Lean project repo gerbil worked on.
set -euo pipefail

cd "${1:-.}"

shopt -s nullglob
commits=(.gerbil/gerbil-*.commit)
(( ${#commits[@]} )) || { echo "no .gerbil/gerbil-*.commit files found" >&2; exit 1; }

for commit in "${commits[@]}"; do
    base="${commit%.commit}"
    patch="${base}.patch"
    jsonl="${base}.jsonl"
    [ -f "$patch" ] || { echo "skip $commit: no matching $patch" >&2; continue; }
    echo "applying $patch"
    # --index stages exactly the patch's changes (no `git add -A`, so the other
    # gerbil-* files are never swept in).
    git apply --index "$patch"
    [ -f "$jsonl" ] && git add "$jsonl"
    git commit -q -F "$commit"
    echo "  committed: $(head -1 "$commit")"
    rm -f "$patch" "$commit"
done

echo "done"
