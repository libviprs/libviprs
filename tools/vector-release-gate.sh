#!/usr/bin/env bash
# The vector release gate (issue #1402, part of #1038).
#
# Run from a libviprs checkout inside a container with the Rust toolchain, curl,
# tar, sha256sum and python3 (the same one tools/native-dwg-cell.sh runs in; on
# MARS that is `rust:1.99.0-bookworm`). It runs what can be run, then prints
# release/vector-gate.json and exits non-zero unless every criterion in it is met:
#
#   * the native cells (real drawings through the real decoder, which includes
#     the supported-entity matrix);
#   * the raster regression suite (the default `cargo test`);
#   * the gate record's own checks.
#
# It never edits the record. Flipping a row to met is a reviewed change with
# evidence, and tests/vector_1038_release_gate.rs refuses a row without it.
set -uo pipefail

cd "$(dirname "$0")/.."
fail=0
step() {
  local what=$1
  shift
  echo "==> $what"
  "$@" || { echo "FAILED: $what" >&2; fail=1; }
}

step "native cells" tools/native-dwg-cell.sh
step "raster regression suite" cargo test
step "gate record" cargo test --test vector_1038_release_gate

echo
echo "==> release/vector-gate.json"
python3 -I - <<'PY'
import json
g = json.load(open("release/vector-gate.json"))
for c in g["criteria"]:
    print(f'{c["status"]:8} {c["id"]}')
    if c["status"] != "met":
        print(f'         {c["blocker"]}')
print("release_ready:", g["release_ready"])
raise SystemExit(0 if g["release_ready"] else 1)
PY
ready=$?
[ $fail -eq 0 ] && [ $ready -eq 0 ]
