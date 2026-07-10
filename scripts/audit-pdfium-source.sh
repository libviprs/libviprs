#!/usr/bin/env bash
#
# audit-pdfium-source.sh — release gate for issue #149.
#
# The `pdfium-render` thread-safety fork (per-call locking in
# `ThreadSafePdfiumBindings`) is applied via `[patch.crates-io]` in this
# crate's `Cargo.toml`. A `[patch]` table only takes effect from the root
# of the workspace performing the build, so any sibling crate or crates.io
# consumer that does NOT replicate the patch silently links the unpatched
# `pdfium-render 0.8.x` wrapper, which segfaults under concurrent access.
#
# This script resolves the dependency graph for a given manifest and fails
# if `pdfium-render` is sourced from the crates.io registry instead of the
# pinned libviprs git fork. Run it in every repo that enables the `pdfium`
# feature before publishing / releasing.
#
# Usage:
#   scripts/audit-pdfium-source.sh [MANIFEST_DIR] [-- <extra cargo metadata args>]
#
# MANIFEST_DIR defaults to the current directory. Exit status:
#   0  pdfium-render resolves from the libviprs git fork (or is absent).
#   1  pdfium-render resolves from the crates.io registry (unpatched).
#   2  usage / tooling error.

set -euo pipefail

EXPECTED_HOST="github.com/libviprs/pdfium-render"

manifest_dir="."
extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --) shift; extra_args=("$@"); break ;;
    -h|--help)
      sed -n '2,26p' "$0"; exit 0 ;;
    *) manifest_dir="$1"; shift ;;
  esac
done

manifest_path="${manifest_dir%/}/Cargo.toml"
if [[ ! -f "$manifest_path" ]]; then
  echo "audit-pdfium-source: no Cargo.toml at '$manifest_path'" >&2
  exit 2
fi

metadata="$(cargo metadata --format-version 1 --manifest-path "$manifest_path" \
  "${extra_args[@]}" 2>/dev/null)" || {
  echo "audit-pdfium-source: 'cargo metadata' failed for '$manifest_path'" >&2
  exit 2
}

# Extract the resolved `source` field for the `pdfium-render` package.
source_field="$(printf '%s' "$metadata" | python3 -c '
import json, sys
data = json.load(sys.stdin)
hits = [p.get("source") or "" for p in data["packages"] if p["name"] == "pdfium-render"]
print("ABSENT" if not hits else hits[0])
')"

case "$source_field" in
  ABSENT)
    echo "audit-pdfium-source: OK — pdfium-render is not in the graph for '$manifest_path'"
    exit 0 ;;
  git+*"$EXPECTED_HOST"*)
    echo "audit-pdfium-source: OK — pdfium-render resolves from the libviprs fork:"
    echo "  $source_field"
    exit 0 ;;
  *)
    echo "audit-pdfium-source: FAIL — pdfium-render does NOT resolve from the libviprs fork (issue #149)." >&2
    echo "  resolved source: ${source_field:-<empty/registry>}" >&2
    echo "  expected a git source containing: $EXPECTED_HOST" >&2
    echo "  add the [patch.crates-io] pdfium-render fork to this manifest's workspace root." >&2
    exit 1 ;;
esac
