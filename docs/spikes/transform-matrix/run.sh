#!/usr/bin/env bash
# Regenerates the measurement files, always in a container, never on this machine's toolchain.
#
#   docs/spikes/transform-matrix/run.sh [x64|arm64] [pvec-checkout]   (default x64, ../pvec next to this repo)
#
#   x64    results-x64.tsv, native x86_64 on MARS through tools/nas.sh
#   arm64  results-arm64.tsv, `docker run --platform linux/arm64` on this Apple Silicon Mac (native, no emulation)
#
# PVec is staged from a pinned revision with `git archive`, so no credential goes anywhere and the
# result names the exact tree it measured. Needs DOCKER_CONFIG=/tmp/dc if ~/.docker is a symlink
# the build image refuses.
set -euo pipefail
arch=${1:-x64}
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
tools=$repo/../tools
pvec=${2:-$repo/../pvec}
rev=1f7561e
image=rust:1.99.1-bookworm
stage=$(mktemp -d)
trap 'mv "$stage" "$stage.superseded" 2>/dev/null || true' EXIT
mkdir -p "$stage/harness" "$stage/pvec-src"
tar -C "$here/harness" --exclude target -cf - . | tar -C "$stage/harness" -xf -
git -C "$pvec" archive "$rev" | tar -C "$stage/pvec-src" -xf -
cmd='cd harness && CARGO_TARGET_DIR=/tmp/target cargo run -q --release; echo "# rustc $(rustc --version) $(uname -m)"'
case $arch in
  x64)
    name=n15-transform-matrix
    "$tools/nas.sh" push "$stage" "$name"
    {
      echo "# native x86_64 (MARS), pvec@$rev, pvec-geometry transform path (P3.8)"
      "$tools/nas.sh" run "$name" "$image" "$cmd"
    } > "$here/results-x64.tsv"
    "$tools/nas.sh" clean "$name"
    ;;
  arm64)
    {
      echo "# native arm64 (Apple Silicon, docker --platform linux/arm64), pvec@$rev, pvec-geometry transform path (P3.8)"
      docker run --rm --platform linux/arm64 -v "$stage:/work" -w /work "$image" sh -c "$cmd"
    } > "$here/results-arm64.tsv"
    ;;
  *) echo "usage: run.sh [x64|arm64] [pvec-checkout]" >&2; exit 2 ;;
esac
