#!/usr/bin/env bash
# Regenerates results-x64.tsv (native x86_64, MARS, through tools/nas.sh) and results-arm64.tsv
# (this Mac, a linux/arm64 container). Nothing is built or run on the host itself.
#
#   docs/spikes/pdf-normalization/run.sh [pvec-checkout] [candidates-dir] [acadsharp-samples-dir]
#
# PVec is staged from a pinned revision with `git archive`, so no credential goes anywhere and the
# result names the exact tree it measured. The fixtures are public-domain USACE sheets from
# pdf-candidates (licence evidence in its arch/CANDIDATES.md); their sha256 goes into the results.
# Needs DOCKER_CONFIG=/tmp/dc16 if ~/.docker is a symlink the build image refuses.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
tools=$repo/../tools
pvec=${1:-$repo/../pvec}
cand=${2:-$repo/../pdf-candidates}
rev=1f7561e
image=rust:1.99.1-bookworm
files="usace_hnc_yc_11-18_large_2019.pdf usace_hnc_std_drawings_fire_station_set1_2021.pdf usace_tulsa_bt_bnhq_sample_set.pdf"
stage=$(mktemp -d)
trap 'mv "$stage" "$stage.superseded" 2>/dev/null || true' EXIT
mkdir -p "$stage/harness" "$stage/pvec-src" "$stage/fx"
tar -C "$here/harness" --exclude target -cf - . | tar -C "$stage/harness" -xf -
git -C "$pvec" archive "$rev" | tar -C "$stage/pvec-src" -xf -
for f in $files; do cp "$cand/arch/$f" "$stage/fx/$f"; done
shas=$(for f in $files; do printf '%s\tsha256\t%s\n' "$f" "$(shasum -a 256 "$stage/fx/$f" | cut -d' ' -f1)"; done)

emit() { # arch-label, runner output on stdin
  { echo "# $1, pvec@$rev, lopdf 0.36.0, fixtures: $files"; cat; echo "$shas"; }
}

name=n16-pdf-normalization
"$tools/nas.sh" push "$stage" "$name"
"$tools/nas.sh" run "$name" "$image" \
  'cd harness && CARGO_TARGET_DIR=/tmp/target cargo run -q --release -- /work/fx 2>/dev/null; echo "# rustc $(rustc --version) $(uname -m)"' \
  | { out=$(cat); { echo "# native x86_64 (MARS), pvec@$rev, lopdf 0.36.0"; echo "$out" | grep -v '^# rustc'; echo "$shas"; echo "$out" | grep '^# rustc'; } > "$here/results-x64.tsv"; }
"$tools/nas.sh" clean "$name"

out=$(docker run --rm --platform linux/arm64 -v "$stage:/work" -w /work/harness -e CARGO_TARGET_DIR=/tmp/target "$image" \
  sh -c 'cargo run -q --release -- /work/fx 2>/dev/null; echo "# rustc $(rustc --version) $(uname -m)"')
{ echo "# arm64 (this Mac, linux/arm64 container), pvec@$rev, lopdf 0.36.0"; echo "$out" | grep -v '^# rustc'; echo "$shas"; echo "$out" | grep '^# rustc'; } > "$here/results-arm64.tsv"
