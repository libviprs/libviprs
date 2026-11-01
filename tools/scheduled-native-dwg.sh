#!/usr/bin/env bash
# Run the native DWG cells on a schedule, on MARS (issue #1295).
#
# The host only runs docker. Everything else, the clone, the download, the
# SHA-256 check and cargo, happens in one throw-away container that clones the
# public repo's main branch (no token on disk) and calls
# tools/native-dwg-cell.sh, which is the same script the workflow runs.
#
#   tools/scheduled-native-dwg.sh [REF]     REF defaults to main
#
# Install it with tools/native-dwg.cron (see CONTRIBUTING.md). Each run appends
# one line to $OUT/results.tsv and keeps its log as $OUT/<utc-stamp>.log. Nothing
# is deleted here; old logs are renamed to <name>.superseded by hand when the
# directory gets big. Exit status is the run's status, so cron mail carries it.
set -euo pipefail

ref=${1:-main}
image=${NATIVE_DWG_IMAGE:-rust:1.99.0-bookworm}
out=${OUT:-/work/nas-work/native-dwg-sched}
repo=${NATIVE_DWG_REPO:-https://github.com/libviprs/libviprs.git}
stamp=$(date -u +%Y%m%dT%H%M%SZ)

mkdir -p "$out"
# One run at a time on this shared host.
exec 9>"$out/lock"
flock -n 9 || { echo "native-dwg: a run is already in progress" >&2; exit 75; }

start=$(date +%s)
set +e
docker run --rm --platform linux/amd64 --cpus 4 --memory 8g \
  -e REF="$ref" -e REPO="$repo" \
  -v native-dwg-cargo:/usr/local/cargo/registry \
  "$image" sh -c '
    set -e
    git clone --quiet "$REPO" /src
    cd /src
    git checkout --quiet --detach "$REF"
    echo "native-dwg: $(git rev-parse HEAD)"
    CARGO_TARGET_DIR=/tmp/target tools/native-dwg-cell.sh
  ' >"$out/$stamp.log" 2>&1
status=$?
set -e

printf '%s\t%s\t%s\t%ss\n' "$stamp" "$ref" "$([ $status -eq 0 ] && echo pass || echo FAIL)" "$(( $(date +%s) - start ))" >>"$out/results.tsv"
[ $status -eq 0 ] || { tail -n 40 "$out/$stamp.log" >&2; }
exit $status
