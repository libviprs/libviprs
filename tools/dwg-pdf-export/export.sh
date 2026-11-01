#!/usr/bin/env bash
# Export DWGs to vector PDFs with LibreDWG's dwg2SVG + rsvg-convert (+ qpdf for a page rotation),
# in a throw-away container on MARS (issue #1405). Nothing runs on the host but ssh and docker.
#
#   export.sh OUT_DIR ID=PATH_TO_DWG[:ROTATE] ...
#
# For each ID it writes OUT_DIR/ID/exported.pdf and OUT_DIR/ID/exporter.txt (what ran, with
# versions). ROTATE is 90, 180 or 270 and sets /Rotate on the page with `qpdf --rotate`.
# Scratch on MARS is renamed to <name>.superseded at the end, never deleted.
set -euo pipefail
HOST=${EXPORT_HOST:-devshell@192.168.0.100}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1; shift
NAME=dwgpdf-$$-$(date +%s)
W=/work/nas-work/$NAME
SSH=(ssh -o BatchMode=yes -o LogLevel=ERROR "$HOST")
"${SSH[@]}" "mkdir -p $W/in $W/out"
"${SSH[@]}" "cat > $W/Dockerfile" < "$HERE/Dockerfile"
list=""
for spec in "$@"; do
  id=${spec%%=*}; rest=${spec#*=}; path=${rest%%:*}; rot=""
  [ "$rest" != "$path" ] && rot=${rest#*:}
  "${SSH[@]}" "cat > $W/in/$id.dwg" < "$path"
  list="$list $id:${rot:-0}"
done
"${SSH[@]}" "export DOCKER_CONFIG=/tmp/dkcfg-$NAME; mkdir -p \$DOCKER_CONFIG; cd $W \
 && DOCKER_BUILDKIT=0 docker build -q --platform linux/amd64 -t viprs-dwg-pdf-export:1 . >/dev/null \
 && docker run --rm --platform linux/amd64 -v $W:/w -w /w viprs-dwg-pdf-export:1 sh -c '
set -e
for item in $list; do
  id=\${item%%:*}; rot=\${item#*:}
  mkdir -p out/\$id
  dwg2SVG in/\$id.dwg > out/\$id/exported.svg
  rsvg-convert -f pdf -o out/\$id/raw.pdf out/\$id/exported.svg
  if [ \"\$rot\" != 0 ]; then qpdf --rotate=+\$rot:1 out/\$id/raw.pdf out/\$id/exported.pdf; else cp out/\$id/raw.pdf out/\$id/exported.pdf; fi
  {
    echo \"exporter: LibreDWG dwg2SVG \$LIBREDWG_VERSION, rsvg-convert \$(rsvg-convert --version | sed s/.*version.//), qpdf \$(qpdf --version | head -1 | sed s/.*version.//)\"
    echo \"options: dwg2SVG defaults; rsvg-convert -f pdf; page rotation \$rot\"
    echo \"platform: linux/amd64 native (MARS)\"
  } > out/\$id/exporter.txt
done
chmod -R a+rX out'"
for spec in "$@"; do
  id=${spec%%=*}; mkdir -p "$OUT/$id"
  for f in exported.pdf exporter.txt; do "${SSH[@]}" "cat $W/out/$id/$f" > "$OUT/$id/$f"; done
done
"${SSH[@]}" "mv $W $W.superseded"
