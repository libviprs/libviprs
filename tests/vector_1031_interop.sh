#!/usr/bin/env bash
# Independent-tool check for the vector PMTiles writer (issue #1031).
#
# Runs on a docker host, never on the machine it was typed on. Three throw-away
# containers, one scratch directory:
#
#   1. rust:      libviprs writes golden.pmtiles and expected.json
#   2. go-pmtiles (pinned v1.31.2 release, sha256 checked): verify, show, tile
#   3. python:    the Mapbox `mapbox-vector-tile` package decodes the tile bytes
#                 go-pmtiles hands back, and every geometry and property is
#                 compared with expected.json
#
# RUST_DOCKER_ARGS adds docker flags to the build container, e.g. cache volumes.
# usage: vector_1031_interop.sh <scratch-dir-on-the-docker-host> <libviprs-tree>
set -euo pipefail

SCRATCH=${1:?scratch dir}
TREE=${2:?libviprs tree}
GO_PMTILES_SHA=3ed7dbf4ec2e6dfe5e25b6f70d1ffc932729f93c86db353bf514dd71010a312f
GO_PMTILES_URL=https://github.com/protomaps/go-pmtiles/releases/download/v1.31.2/go-pmtiles_1.31.2_Linux_x86_64.tar.gz

mkdir -p "$SCRATCH/out" "$SCRATCH/tools"

docker run --rm --platform linux/amd64 \
  -v "$TREE:/work" -v "$SCRATCH/out:/out" -w /work ${RUST_DOCKER_ARGS:-} \
  -e CARGO_TARGET_DIR=/tmp/target -e LIBVIPRS_VECTOR_GOLDEN_DIR=/out \
  rust:1.99.1-bookworm \
  cargo test --test vector_1031_archive write_the_interop_golden_when_asked

docker run --rm --platform linux/amd64 \
  -v "$SCRATCH/out:/out" -v "$SCRATCH/tools:/tools" -w /tools \
  debian:bookworm-slim sh -ec "
    apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null
    curl -fsSL -o gp.tgz '$GO_PMTILES_URL'
    echo '$GO_PMTILES_SHA  gp.tgz' | sha256sum -c -
    tar xzf gp.tgz pmtiles
    ./pmtiles version
    ./pmtiles verify /out/golden.pmtiles
    ./pmtiles show /out/golden.pmtiles
    for t in 0/0/0 1/1/0 1/0/1 2/3/3; do
      ./pmtiles tile /out/golden.pmtiles \$(echo \$t | tr / ' ') > /out/tile_\$(echo \$t | tr / _).mvt
      echo \"tile \$t resolved\"
    done
    ./pmtiles tile /out/golden.pmtiles 2 1 1 > /out/absent.bin 2>/dev/null || true
    if [ -s /out/absent.bin ]; then echo 'absent tile resolved to bytes'; exit 1; fi
    echo 'absent tile 2/1/1 returned nothing'
  "

docker run --rm --platform linux/amd64 \
  -v "$SCRATCH/out:/out" -v "$(cd "$(dirname "$0")" && pwd):/check:ro" -w /out python:3.12-slim sh -ec "
    pip install -q mapbox-vector-tile==2.1.0 >/dev/null
    python3 -I /check/vector_1031_mvt_check.py
  "
echo 'interop OK'
