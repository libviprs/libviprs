#!/usr/bin/env bash
#
# audit-pdfium-source.sh — release gate for issues #149 and #981.
#
# This gate used to require the opposite of what it requires now, and the
# reversal is the point, so here is why.
#
# It was written for #149: `pdfium-render` 0.9.0 through 0.9.3 deleted
# `src/bindings/thread_safe.rs` and left the `thread_safe` feature gating a
# bare `unsafe impl Send + Sync` with nothing behind it, so the gate demanded
# the libviprs fork, which carried per-call locking. Cargo strips a git source
# on publish, so that demand could never be met by a crates.io consumer, and
# the split it created is exactly what #981 is about: whoever built from git
# got the fork, whoever installed from the registry got the unpatched wrapper,
# and only the first was ever tested.
#
# Upstream reinstated the locking in 0.9.4 (2026-09-06). It is not complete:
# 290 of its 484 binding methods take the lock, and
# `FPDF_RenderPageBitmapWithMatrix` is one of the ones that does not. libviprs
# does not rely on it either way, because it holds `pdfium_lock()` across whole
# operations itself and exposes no pdfium-render type in its public API, so a
# consumer cannot reach the wrapper's `Send + Sync` through libviprs at all.
#
# So the invariant worth gating flipped. One source for everybody beats two
# sources wearing one name, and this script now fails if `pdfium-render`
# resolves from anywhere other than the registry. `tests/pdfium_dependency_
# contract.rs` guards the manifest side of the same claim.
#
# Usage:
#   scripts/audit-pdfium-source.sh [MANIFEST_DIR] [-- <extra cargo metadata args>]
#
# MANIFEST_DIR defaults to the current directory. Exit status:
#   0  pdfium-render resolves from the crates.io registry (or is absent).
#   1  pdfium-render resolves from somewhere else, usually a git fork.
#   2  usage / tooling error.

set -euo pipefail

FORK_HOST="github.com/libviprs/pdfium-render"

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
  registry+*crates.io*)
    echo "audit-pdfium-source: OK — pdfium-render resolves from crates.io:"
    echo "  $source_field"
    exit 0 ;;
  git+*"$FORK_HOST"*)
    echo "audit-pdfium-source: FAIL — pdfium-render resolves from the libviprs fork (issue #981)." >&2
    echo "  resolved source: $source_field" >&2
    echo "  The fork was retired: upstream reinstated the per-call locking in" >&2
    echo "  0.9.4, and what the fork still carried over it is nothing libviprs" >&2
    echo "  calls. A git source cannot survive publish, so it makes the crate" >&2
    echo "  everyone builds different from the crate everyone installs." >&2
    echo "  Depend on the registry: pdfium-render = { version = \"0.9.4\", ... }" >&2
    exit 1 ;;
  *)
    echo "audit-pdfium-source: FAIL — pdfium-render resolves from neither crates.io nor a known fork." >&2
    echo "  resolved source: ${source_field:-<empty>}" >&2
    exit 1 ;;
esac
