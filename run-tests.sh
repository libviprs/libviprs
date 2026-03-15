#!/usr/bin/env bash
# Delegates to libviprs-tests/run-tests.sh — can be invoked from this repo.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/../libviprs-tests/run-tests.sh" "$@"
