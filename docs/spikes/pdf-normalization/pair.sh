#!/usr/bin/env bash
# Regenerates results-pair.tsv for #1240: the open-exporter pair, not AutoCAD-class.
#
#   docs/spikes/pdf-normalization/pair.sh [pvec-checkout] [acadrust-checkout]
#
# 1. pair-gen (acadrust DwgWriter) writes pair.dwg, reads it back with DwgReader, counts entities per
#    class, and writes the DXF from that read-back. It also writes one DWG/DXF per class.
# 2. LibreCAD (Debian package, headless, offscreen Qt) plots each DXF to PDF with `dxf2pdf -a`.
# 3. The N1.6 harness measures every PDF.
# Everything runs in linux/arm64 containers; nothing is built or run on the host. The scratch tree is
# renamed to .superseded on exit. Needs DOCKER_CONFIG=/tmp/dcpair if ~/.docker is a symlink the image refuses.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
pvec=${1:-$repo/../pvec}
acad=${2:-$repo/../acadrust}
rev=1f7561e
rimg=rust:1.99.1-bookworm
dimg=debian:bookworm-slim
stage=$(mktemp -d)
trap 'mv "$stage" "$stage.superseded" 2>/dev/null || true' EXIT
mkdir -p "$stage"/{pair,harness,pvec-src,acadrust-src,out,fx}
tar -C "$here/pair" --exclude target -cf - . | tar -C "$stage/pair" -xf -
tar -C "$here/harness" --exclude target -cf - . | tar -C "$stage/harness" -xf -
tar -C "$acad" --exclude target --exclude .git -cf - . | tar -C "$stage/acadrust-src" -xf -
git -C "$pvec" archive "$rev" | tar -C "$stage/pvec-src" -xf -
acadrev=$(git -C "$acad" rev-parse --short HEAD)

dwgrows=$(docker run --rm --platform linux/arm64 -v "$stage:/work" -w /work/pair -e CARGO_TARGET_DIR=/tmp/target "$rimg" \
  sh -c 'cargo run -q --release -- /work/out 2>/dev/null')
libre=$(docker run --rm --platform linux/arm64 -v "$stage/out:/w" -w /w "$dimg" sh -c '
  apt-get update >/dev/null 2>&1
  apt-get install -y --no-install-recommends librecad xauth >/dev/null 2>&1
  export QT_QPA_PLATFORM=offscreen
  for f in /w/*.dxf; do b=$(basename "$f" .dxf); timeout 300 librecad dxf2pdf -a -o "/w/$b.pdf" "$f" >/dev/null 2>&1 || echo "plot failed: $b" >&2; done
  echo "librecad $(dpkg -s librecad | sed -n "s/^Version: //p") (Debian bookworm, $(uname -m)), Qt $(dpkg -s libqt5core5a | sed -n "s/^Version: //p")"')
cp "$stage"/out/*.pdf "$stage/fx/"
pdfrows=$(docker run --rm --platform linux/arm64 -v "$stage:/work" -w /work/harness -e CARGO_TARGET_DIR=/tmp/target "$rimg" \
  sh -c 'cargo run -q --release -- /work/fx 2>/dev/null; echo "# rustc $(rustc --version) $(uname -m)"' | grep -a -v '^synthetic' | grep -a -v '^fixture' | LC_ALL=C tr -d '\000\376\377' | perl -CSD -pe 's/\x{FFFD}//g')
shas=$(cd "$stage/out" && for f in pair.dwg pair.dxf pair.pdf; do printf '%s\tsha256\t%s\n' "$f" "$(shasum -a 256 "$f" | cut -d' ' -f1)"; done)
{
  echo "# open-exporter pair, not AutoCAD-class: the loss measured is that exporter's"
  echo "# dwg written by acadrust $acadrev (DwgWriter, AC1032), dxf written from the read-back DWG"
  echo "# pdf plotted by $libre via 'dxf2pdf -a'; harness pvec@$rev, lopdf 0.36.0"
  echo "$dwgrows"
  echo "$pdfrows" | grep -a -v '^# rustc'
  echo "$shas"
  echo "$pdfrows" | grep -a '^# rustc'
} > "$here/results-pair.tsv"
