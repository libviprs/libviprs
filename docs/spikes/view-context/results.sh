#!/usr/bin/env bash
# Produce results-x64.txt: every probe run the document cites, from one native
# x86_64 host. Run after `run.sh push` and both builds, and after `run.sh author /work/out`.
set -euo pipefail
cd "$(dirname "$0")"
R=./run.sh
{
echo "# vcprobe and vc-acadrust-probe on $(ssh -o BatchMode=yes -o LogLevel=ERROR "${NAS:-devshell@192.168.0.100}" 'hostname; uname -m' | tr '\n' ' ')"
echo "# ACadSharp 3.7.1 (libviprs-dep patches applied), shim sources libviprs-dep 0d475bf (3.7.1-viprs.2), acadrust 0.5.5 d35691f, .NET SDK 10.0.401, JIT not NativeAOT"
echo
for f in out/vc_layers out/vc_nested out/vc_paper out/vc_anonymous in/sample_AC1032 in/BLOCKVISIBILITYPARAMETER in/g13_xref; do
  for m in describe shim rule; do
    echo "== $m /work/$f.dwg"
    $R $m "/work/$f.dwg" 2>&1
    echo
  done
done
echo "== rule, every other drawing (summary lines only)"
for f in in/AecObjects in/BLOCKALIGNMENTPARAMETER in/BLOCKBASEPOINTPARAMETER in/BLOCKFLIPPARAMETER in/BLOCKLINEARPARAMETER \
         in/BLOCKLOOKUPPARAMETER in/BLOCKPOINTPARAMETER in/BLOCKPOLARPARAMETER in/BLOCKROTATIONPARAMETER in/BLOCKXYPARAMETER \
         in/empty in/export_sample in/g13_insert in/geoloc in/sample_AC1014 in/sample_AC1015 in/sample_AC1018 in/sample_AC1021 \
         in/sample_AC1024 in/sample_AC1027 in/sample_base; do
  echo "-- /work/$f.dwg"
  $R rule "/work/$f.dwg" 2>&1 | grep -E '^(rule_|shim_)'
done
echo
for f in out/vc_layers out/vc_paper in/sample_AC1032; do
  echo "== acadrust /work/$f.dwg"
  $R acadrust "/work/$f.dwg" 2>&1
  echo
done
} > results-x64.txt
