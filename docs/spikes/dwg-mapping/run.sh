#!/usr/bin/env bash
# Regenerates the measurement files, always in a container, never on this machine's toolchain.
#
#   docs/spikes/dwg-mapping/run.sh [x64|arm64] [pvec-checkout] [gold-checkout]
#
#   x64    results-x64.tsv, native x86_64 on MARS through tools/nas.sh
#   arm64  results-arm64.tsv, `docker run --platform linux/arm64` on this Apple Silicon Mac (native, no emulation)
#
# PVec and the corpus are staged from pinned revisions with `git archive`, so no credential goes
# anywhere and the result names the exact trees it measured. Needs DOCKER_CONFIG=/tmp/dc13 (or any
# directory with an empty config.json) if ~/.docker is a symlink the build image refuses.
set -euo pipefail
arch=${1:-x64}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
tools=$repo/../tools
pvec=${2:-$repo/../pvec}
gold=${3:-$repo/../pvec-gold-fixtures}
pvec_rev=d399d76
corpus_tag=corpus-r0-open
corpus_rev=f94f7f2
image=rust:1.99.1-bookworm
stage=$(mktemp -d)
trap 'mv "$stage" "$stage.superseded" 2>/dev/null || true' EXIT
mkdir -p "$stage/harness" "$stage/pvec-src" "$stage/gold"
tar -C "$here/harness" --exclude target -cf - . | tar -C "$stage/harness" -xf -
git -C "$pvec" archive "$pvec_rev" | tar -C "$stage/pvec-src" -xf -
test "$(git -C "$gold" rev-parse --short=7 "$corpus_tag^{}")" = "$corpus_rev"
git -C "$gold" archive "$corpus_tag" corpus/r0-open | tar -C "$stage/gold" -xf -
cmd='cd harness && CARGO_TARGET_DIR=/tmp/target cargo run -q --release -- ../gold/corpus/r0-open; echo "# rustc $(rustc --version) $(uname -m)"'
case $arch in
  x64)
    name=n13-dwg-mapping
    "$tools/nas.sh" push "$stage" "$name"
    {
      echo "# native x86_64 (MARS), pvec@$pvec_rev, corpus $corpus_tag@$corpus_rev (r0-open, READER_MAPPING_ONLY)"
      "$tools/nas.sh" run "$name" "$image" "$cmd"
    } > "$here/results-x64.tsv"
    "$tools/nas.sh" clean "$name"
    ;;
  arm64)
    {
      echo "# native arm64 (Apple Silicon, docker --platform linux/arm64), pvec@$pvec_rev, corpus $corpus_tag@$corpus_rev (r0-open, READER_MAPPING_ONLY)"
      docker run --rm --platform linux/arm64 -v "$stage:/work" -w /work "$image" sh -c "$cmd"
    } > "$here/results-arm64.tsv"
    ;;
  *) echo "usage: run.sh [x64|arm64] [pvec-checkout] [gold-checkout]" >&2; exit 2 ;;
esac
