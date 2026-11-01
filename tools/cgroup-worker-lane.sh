#!/usr/bin/env bash
# The cgroup form of the worker test (issue #1392, part of #1256).
#
# tests/vector_1036_worker.rs proves the `ulimit -v` form of the memory ceiling.
# A deployer gets a cgroup limit instead, where the child is OOM-killed. This
# builds that test binary and runs its "unbudgeted" child (a zoom 24 tile
# explosion with every budget off) under `docker run --memory=256m`, and
# checks what the docs promise:
#
#   * the container exits 137 and docker reports OOMKilled;
#   * the destination (a.pmtiles) does not exist;
#   * only the staging pair a.pmtiles.tmp.* is left behind.
#
# The host only runs docker. The clone and the build happen in a throw-away
# container, the run in another one with the memory limit.
#
#   tools/cgroup-worker-lane.sh [REF]     REF defaults to main
set -euo pipefail

ref=${1:-main}
image=${NATIVE_DWG_IMAGE:-rust:1.99.0-bookworm}
out=${OUT:-/work/nas-work/native-dwg-sched}
repo=${NATIVE_DWG_REPO:-https://github.com/libviprs/libviprs.git}
name=cgroup-lane-$$
dir=$out/cgroup-$$

mkdir -p "$dir/dest"

docker run --rm --platform linux/amd64 --cpus 4 --memory 8g \
  -e REF="$ref" -e REPO="$repo" -v native-dwg-cargo:/usr/local/cargo/registry -v "$dir":/o \
  "$image" sh -c '
    set -e
    git clone --quiet "$REPO" /src
    cd /src
    git checkout --quiet --detach "$REF"
    CARGO_TARGET_DIR=/tmp/target cargo test --no-run --test vector_1036_worker 2>&1 | tee /tmp/build.log
    exe=$(sed -n "s/.*Executable .*(\(.*\))/\1/p" /tmp/build.log | head -n 1)
    cp "$exe" /o/worker
  '

set +e
docker run --name "$name" --platform linux/amd64 --memory=256m --memory-swap=256m \
  -e VIPRS_L23_MODE=unbudgeted -e VIPRS_L23_OUT=/o/dest/a.pmtiles -v "$dir":/o \
  "$image" /o/worker child_entry --exact --nocapture
status=$?
set -e
oom=$(docker inspect --format '{{.State.OOMKilled}}' "$name")
docker rm "$name" >/dev/null

left=$(ls "$dir/dest" | tr '\n' ' ')
echo "cgroup lane: exit=$status OOMKilled=$oom left='$left'"

fail=0
[ "$status" -eq 137 ] || { echo "expected exit 137" >&2; fail=1; }
[ "$oom" = true ] || { echo "expected OOMKilled=true" >&2; fail=1; }
[ ! -e "$dir/dest/a.pmtiles" ] || { echo "the destination exists" >&2; fail=1; }
for f in $left; do
  case $f in a.pmtiles.tmp.*) ;; *) echo "unexpected leftover $f" >&2; fail=1 ;; esac
done

printf '%s\tcgroup\t%s\t%s\n' "$(date -u +%Y%m%dT%H%M%SZ)" "$ref" "$([ $fail -eq 0 ] && echo pass || echo FAIL)" >>"$out/results.tsv"
exit $fail
