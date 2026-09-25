#!/bin/sh
set -e
BASE="$1"
sed -i '' \
  -e "s/PLACEHOLDER_BRANCH/1762 passed, 0 failed, 8 ignored (36 suites)/" \
  -e "s/PLACEHOLDER_BASELINE/$BASE/" \
  .pr-631-float-premultiply.md
grep -n "1762 passed\|$BASE" .pr-631-float-premultiply.md
