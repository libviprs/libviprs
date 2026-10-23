#!/usr/bin/env bash
# Regenerates results-x64.tsv on native x86_64 (MARS) through tools/nas.sh, never on this machine.
#
#   docs/spikes/transform-matrix/run.sh [pvec-checkout]     (default: ../pvec next to this repo)
#
# PVec is staged from a pinned revision with `git archive`, so no credential goes anywhere and the
# result names the exact tree it measured. Needs DOCKER_CONFIG=/tmp/dc if ~/.docker is a symlink
# the build image refuses.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
tools=$repo/../tools
pvec=${1:-$repo/../pvec}
rev=1f7561e
stage=$(mktemp -d)
trap 'mv "$stage" "$stage.superseded" 2>/dev/null || true' EXIT
mkdir -p "$stage/harness" "$stage/pvec-src"
tar -C "$here/harness" --exclude target -cf - . | tar -C "$stage/harness" -xf -
git -C "$pvec" archive "$rev" | tar -C "$stage/pvec-src" -xf -
name=n15-transform-matrix
"$tools/nas.sh" push "$stage" "$name"
{
  echo "# native x86_64 (MARS), pvec@$rev, pvec-geometry transform path (P3.8)"
  "$tools/nas.sh" run "$name" rust:1.99.1-bookworm \
    'cd harness && CARGO_TARGET_DIR=/tmp/target cargo run -q --release; echo "# rustc $(rustc --version) $(uname -m)"'
} > "$here/results-x64.tsv"
"$tools/nas.sh" clean "$name"
