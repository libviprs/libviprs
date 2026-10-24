#!/usr/bin/env bash
# Lay out the inputs the probes build from, in one directory:
#
#   acs/        ACadSharp 3.7.1 source with libviprs-dep's patches applied
#   dep/native  libviprs-dep acadsharp/native (the shim sources) at 0d475bf
#   acadrust/   libviprs/acadrust at d35691f
#   probe/ acadrust-probe/   this directory's two probes
#   in/         the drawings the document names
#
# Usage: stage.sh <out-dir> <acadsharp-3.7.1-source> <libviprs-dep> <acadrust> <real-dwg-dir> <repo>
# The caller supplies the checkouts, so this file pins nothing it cannot check:
# it refuses a checkout that is not at the commit the document records.
set -euo pipefail
OUT=$1 ACS=$2 DEP=$3 ACR=$4 REAL=$5 REPO=$6
want() { [ "$(git -C "$1" rev-parse --short=7 HEAD)" = "$2" ] || { echo "$1 is not at $2" >&2; exit 1; }; }
want "$DEP" 0d475bf
want "$ACR" d35691f
mkdir -p "$OUT/dep" "$OUT/in"
cp -R "$ACS" "$OUT/acs"
python3 -I "$DEP/acadsharp/patches/allocation_ceiling.py" "$OUT/acs" >/dev/null
cp -R "$DEP/acadsharp/native" "$OUT/dep/native"
mkdir -p "$OUT/acadrust"
(cd "$ACR" && git archive HEAD) | tar -x -C "$OUT/acadrust"
cp -R "$(dirname "$0")/probe" "$OUT/probe"
cp -R "$(dirname "$0")/acadrust-probe" "$OUT/acadrust-probe"
cp "$REAL"/*.dwg "$OUT/in/"
cp "$REPO"/tests/fixtures/cad/g13/g13_xref.dwg "$REPO"/tests/fixtures/cad/g13/g13_insert.dwg "$OUT/in/"
