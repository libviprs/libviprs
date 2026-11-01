#!/usr/bin/env bash
# The native DWG cells, in one place (issue #1295).
#
# Run it from a libviprs checkout, inside a container that has the Rust
# toolchain, curl, tar and sha256sum. `.github/workflows/native-dwg.yml` calls
# it (so tools/local-ci.py does too) and so does the MARS scheduler. Do not run
# it on a host directly.
#
# It downloads the archive pinned in tools/acadsharp-native.pin, refuses it if
# the SHA-256 differs, unpacks it under $TMPDIR, and runs every test file that
# insists on the archive with LIBVIPRS_REQUIRE_ACADSHARP_NATIVE=1, so a missing
# or unlinked archive is a failure and never a quiet skip.
# tests/local_gate_native_dwg_1295.rs fails if a test file starts requiring the
# archive and is not listed here.
set -euo pipefail

cd "$(dirname "$0")/.."
pin() { sed -n "s/^$1=//p" tools/acadsharp-native.pin; }
url=$(pin url)
want=$(pin sha256)

work=$(mktemp -d "${TMPDIR:-/tmp}/native-dwg.XXXXXX")
curl --fail --silent --show-error --location --output "$work/archive.tgz" "$url"
echo "$want  $work/archive.tgz" | sha256sum -c -
mkdir "$work/native"
tar -xzf "$work/archive.tgz" -C "$work/native"
# The archive holds one directory with lib/ and metadata/LINKINFO.json in it.
dir=$(dirname "$(find "$work/native" -name LINKINFO.json | head -n 1)")
dir=$(dirname "$dir")
test -d "$dir/lib" || { echo "no lib/ in the archive at $dir" >&2; exit 1; }

export ACADSHARP_NATIVE_DIR="$dir"
# The test binaries link the shared library, so the loader has to find it.
export LD_LIBRARY_PATH="$dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBVIPRS_REQUIRE_ACADSHARP_NATIVE=1

cargo test --features dwg-acadsharp \
  --test cad_1030_acadsharp_decoder \
  --test vector_1032_dwg_e2e \
  --test vector_1034_fidelity \
  --test vector_1035_api \
  --test vector_1036_dwg_corpus \
  --test vector_1292_hardened \
  --test vector_1038_entity_matrix \
  --test vector_1391_limits
