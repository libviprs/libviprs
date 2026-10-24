#!/usr/bin/env bash
# The N1.4 view-context probes, built and run on the native x86_64 docker host
# through tools/nas.sh, in containers and nowhere else. Nothing here touches a
# credential, and nothing is built on this machine.
#
#   stage.sh <dir>           lay the inputs out in <dir> (see stage.sh)
#   NAS_SH=... run.sh push <dir>        copy it to the docker host as scratch tree $NAME
#   run.sh build             build the .NET probe (vcprobe) against the staged sources
#   run.sh <mode> <path>     vcprobe author|describe|shim|rule <path>   (paths are /work/...)
#   run.sh acadrust-build    build the acadrust probe
#   run.sh acadrust <path>   what acadrust exposes for that drawing
#   run.sh clean             remove the scratch tree
set -euo pipefail
NAME=${NAME:-n14dn}
NAS_SH=${NAS_SH:-$(dirname "$0")/../../../../tools/nas.sh}
DOTNET=mcr.microsoft.com/dotnet/sdk:10.0.401-noble
RUST=rust:1.99.1-bookworm
DNENV='export HOME=/tmp/h DOTNET_CLI_HOME=/tmp/h DOTNET_NOLOGO=1 DOTNET_CLI_TELEMETRY_OPTOUT=1 NUGET_PACKAGES=/tmp/nuget'
case ${1:-} in
  push)  "$NAS_SH" push "$2" "$NAME" ;;
  clean) "$NAS_SH" clean "$NAME" ;;
  build)
    "$NAS_SH" run "$NAME" "$DOTNET" "$DNENV; dotnet build probe/ViewContextProbe.csproj -c Release -p:DepDir=/work/dep -p:AcadSharpProject=/work/acs/src/ACadSharp/ACadSharp.csproj --artifacts-path /tmp/art 2>&1 | grep -E ': error |Build succeeded|Error\(s\)' | head -30; mkdir -p /work/bin && cp -R /tmp/art/bin/ViewContextProbe/release/. /work/bin/" ;;
  acadrust-build)
    "$NAS_SH" run "$NAME" "$RUST" "export CARGO_HOME=/tmp/cargo CARGO_TARGET_DIR=/tmp/target; cd /work/acadrust-probe && cargo build --release 2>&1 | tail -3; mkdir -p /work/bin && cp /tmp/target/release/vc-acadrust-probe /work/bin/" ;;
  acadrust)
    "$NAS_SH" run "$NAME" "$RUST" "/work/bin/vc-acadrust-probe $2" ;;
  *)
    "$NAS_SH" run "$NAME" "$DOTNET" "$DNENV; dotnet /work/bin/vcprobe.dll $*" ;;
esac
