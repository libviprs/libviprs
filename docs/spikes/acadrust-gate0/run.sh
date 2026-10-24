#!/usr/bin/env bash
# Gate 0 of N4.1 (#1072): acadrust, through AcadrustProvider, over the corpus.
#
#   run.sh                 run on the native x86_64 docker host via tools/nas.sh and
#                          write results-x64.tsv beside this file
#
# Everything executes in containers. Nothing builds on this machine, and no
# credential is read or written. Inputs:
#
#   $CORPUS     a directory holding the 22 AutoCAD-authored DWGs of ACadSharp
#               `samples/` at d7dc111023477d8a9fffc2153139459c95b4f345 (MIT),
#               flat, under their own names. Default: ../../../../review-acadsharp-upstream
#               laid out by this script. The sha256 of each ends up in the table, so a
#               substituted file shows.
#   the tree    this repository, pushed to the scratch tree $NAME on the docker host.
#
# Tier A is the AutoCAD-authored corpus and decides. Tier B is libviprs-dep's
# ACadSharp-written regression set (tests/fixtures/cad/g13) and Tier S the two
# spike fixtures; both are reported, neither decides.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
NAME=${NAME:-n41}
IN_NAME=${IN_NAME:-n41in}
NAS_SH=${NAS_SH:-$REPO/../tools/nas.sh}
NAS=${NAS:-devshell@192.168.0.100}
UPSTREAM=${UPSTREAM:-$REPO/../review-acadsharp-upstream}
IMG=rust:1.99.1-bookworm
REV=$(sed -n 's/^acadrust = { git = [^,]*, rev = "\([0-9a-f]*\)".*/\1/p' "$REPO/Cargo.toml" | head -1)
LIBREV=$(git -C "$REPO" rev-parse HEAD)
[ ${#REV} -eq 40 ] || { echo "no acadrust rev in Cargo.toml" >&2; exit 1; }
[ "$(git -C "$UPSTREAM" rev-parse HEAD)" = d7dc111023477d8a9fffc2153139459c95b4f345 ] \
  || { echo "$UPSTREAM is not at d7dc111" >&2; exit 1; }

# A scratch directory this script makes under /tmp and removes by its full path.
STAGE=$(mktemp -d /tmp/gate0.XXXXXX)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/A"
S="$UPSTREAM/samples"
cp "$S"/sample_AC10*.dwg "$S"/sample_base/sample_base.dwg "$S"/sample_base/empty.dwg \
   "$S"/dynamic-blocks/BLOCK*.dwg "$S"/aec_objects/AecObjects.dwg "$S"/geolocation/geoloc.dwg \
   "$S"/svg/export_sample.dwg "$STAGE/A/"
[ "$(ls "$STAGE/A" | wc -l | tr -d ' ')" = 22 ] || { echo "expected 22 Tier A files" >&2; exit 1; }

"$NAS_SH" push "$REPO" "$NAME" >/dev/null
"$NAS_SH" push "$STAGE" "$IN_NAME" >/dev/null

OUT=$STAGE/out.tsv
ssh -o BatchMode=yes -o LogLevel=ERROR "$NAS" "docker run --rm --platform linux/amd64 \
  -v ${NAME}-cargo:/cargo -v ${NAME}-target:/target \
  -v /work/nas-work/$NAME:/work -v /work/nas-work/$IN_NAME:/in:ro -w /work \
  -e CARGO_HOME=/cargo -e CARGO_TARGET_DIR=/target -e CARGO_TERM_COLOR=never \
  $IMG bash -c $(printf '%q' '
set -u
cargo build --release --example dwg_gate0 --features dwg-acadrust 2>&1 | tail -2 >&2
BIN=/target/release/examples/dwg_gate0
echo "file	tier	source	sha256	version	open	layers	layouts	blocks	model_entities	entities_all_scopes	unmapped	reread_identical	error"
row() { # tier source path
  out=$("$BIN" "$1" "$2" "$3" 2>/dev/null); rc=$?
  if [ $rc -ne 0 ] || [ -z "$out" ]; then
    sha=$(sha256sum "$3" | cut -d" " -f1)
    printf "%s\t%s\t%s\t%s\t%s\tOPEN_FAILED\t-\t-\t-\t-\t-\t-\t-\tprocess exit %s\n" \
      "$(basename "$3")" "$1" "$2" "$sha" "$(head -c 6 "$3")" "$rc"
  else
    printf "%s\n" "$out"
  fi
}
for f in $(ls /in/A | sort); do row A autocad "/in/A/$f"; done
for f in $(ls tests/fixtures/cad/g13/*.dwg | sort); do row B acadsharp-writer "$f"; done
for f in tests/fixtures/pvec/provider/vc_layers.dwg tests/fixtures/pvec/provider/vc_nested.dwg; do row S acadsharp-writer "$f"; done
')" > "$OUT"

{
  echo "# acadrust $REV"
  echo "# libviprs $LIBREV (AcadrustProvider over acadrust, release build, example dwg_gate0)"
  echo "# host devshell@192.168.0.100 (MARS), native x86_64, $IMG, one process per file"
  echo "# corpus ACadSharp samples at d7dc111023477d8a9fffc2153139459c95b4f345 (MIT); tier A AutoCAD-authored, B and S ACadSharp-writer"
  cat "$OUT"
} > "$HERE/results-x64.tsv"
echo "wrote $HERE/results-x64.tsv"
"$NAS_SH" clean "$NAME" >/dev/null
"$NAS_SH" clean "$IN_NAME" >/dev/null
