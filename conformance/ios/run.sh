#!/bin/bash
# iOS half of #1037. Runs natively on a Mac with Xcode (the simulator cannot run in a container).
# Builds the SwiftPM executable for the simulator, wraps it into an .app, starts the Range/CORS store
# (web/vector-viewer/e2e/server.mjs, unchanged), runs the app in a booted simulator, and leaves the
# evidence in conformance/ios/evidence/.
#   DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer conformance/ios/run.sh [SIMULATOR_NAME]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
SIM=${1:-iPhone 17}
SCRATCH=${SCRATCH:-$(mktemp -d)}
OUT=$HERE/evidence
APP=$SCRATCH/Conformance.app
BUNDLE=org.libviprs.conformance

SDK=$(xcrun --sdk iphonesimulator --show-sdk-path)
( cd "$HERE" && swift build --sdk "$SDK" --triple arm64-apple-ios15.0-simulator --scratch-path "$SCRATCH" 2>&1 | tail -5 )
PRODUCTS=$SCRATCH/out/Products/Debug-iphonesimulator

rm -rf "$APP"; mkdir -p "$APP/Frameworks"
cp "$PRODUCTS/Conformance" "$APP/Conformance"
cp -R "$PRODUCTS/MapLibre.framework" "$APP/Frameworks/"
# the shared style, copied at build time and never edited
cp "$REPO/src/vector/black_on_white.style.json" "$APP/"
cat > "$APP/Info.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>$BUNDLE</string>
<key>CFBundleExecutable</key><string>Conformance</string>
<key>CFBundleName</key><string>Conformance</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleVersion</key><string>1</string>
<key>CFBundleShortVersionString</key><string>1</string>
<key>UIDeviceFamily</key><array><integer>1</integer></array>
<key>UILaunchScreen</key><dict/>
<key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/><key>NSAllowsArbitraryLoads</key><true/></dict>
</dict></plist>
PL
codesign --force --sign - "$APP/Frameworks/MapLibre.framework"
codesign --force --sign - "$APP"

# the store and the glyph files, as the Android run does
VV=$SCRATCH/vv
mkdir -p "$VV/dist"
cp -R "$REPO/web/vector-viewer/e2e" "$VV/e2e"
cp -R "$REPO/web/vector-viewer/public/fonts" "$VV/dist/fonts"
node "$VV/e2e/server.mjs" >"$SCRATCH/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf -o /dev/null http://127.0.0.1:4174/golden.pmtiles; do sleep 1; done

UDID=$(xcrun simctl list devices available -j | node -e '
const d = JSON.parse(require("fs").readFileSync(0, "utf8")).devices;
const name = process.argv[1];
for (const [rt, list] of Object.entries(d)) if (rt.includes("iOS")) for (const x of list) if (x.name === name) { console.log(x.udid); process.exit(0); }
process.exit(1);' "$SIM")
WAS_BOOTED=$(xcrun simctl list devices | grep "$UDID" | grep -c Booted || true)
xcrun simctl bootstatus "$UDID" -b >/dev/null
xcrun simctl install "$UDID" "$APP"
rm -rf "$OUT"/*.json "$OUT"/*.png
mkdir -p "$OUT"
SIMCTL_CHILD_EVIDENCE_DIR=$OUT xcrun simctl launch --console-pty --terminate-running-process "$UDID" $BUNDLE >"$SCRATCH/app.log" 2>&1 || true
xcrun simctl io "$UDID" screenshot "$OUT/simulator-final.png" >/dev/null 2>&1 || true
xcrun simctl uninstall "$UDID" $BUNDLE
RUNTIME=$(xcrun simctl list devices -j | node -e '
const d = JSON.parse(require("fs").readFileSync(0, "utf8")).devices;
for (const [rt, list] of Object.entries(d)) for (const x of list) if (x.udid === process.argv[1]) console.log(rt.split(".").pop());' "$UDID")
[ "$WAS_BOOTED" = 1 ] || xcrun simctl shutdown "$UDID"

node - "$OUT" "$REPO/web/vector-viewer/e2e/fixtures/golden.pmtiles" "$HERE/Package.swift" "$SIM" "$RUNTIME" <<'JS'
const fs = require('fs');
const [out, golden, pkg, sim, runtime] = process.argv.slice(2);
const t = JSON.parse(fs.readFileSync(out + '/tests.json', 'utf8'));
const tests = {};
for (const k of ['remoteRangeMode', 'localFileMode', 'missingArchiveRendersBlank']) {
  tests[k] = String(t[k] ?? 'missing').startsWith('passed') ? 'passed' : String(t[k] ?? 'missing');
}
const metrics = {};
for (const [name, file] of [['remoteRangeMode', 'remote-range'], ['localFileMode', 'local-file'], ['missingArchiveRendersBlank', 'missing-archive']]) {
  metrics[name] = JSON.parse(fs.readFileSync(`${out}/${file}.json`, 'utf8'));
}
const result = {
  issue: 1037,
  platform: 'ios',
  maplibre_native_ios: /exact: "([\d.]+)"/.exec(fs.readFileSync(pkg, 'utf8'))[1],
  ios_release: t.system,
  simulator: `${sim} (${runtime}), arm64 Mac, no GPU passthrough beyond the host's Metal`,
  golden_bytes: fs.statSync(golden).size,
  tests,
  metrics,
};
fs.writeFileSync(out + '/result.json', JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result.tests));
JS
rm -f "$OUT/tests.json"
grep -q '"failed' <(node -e 'console.log(JSON.stringify(require(process.argv[1]).tests))' "$OUT/result.json") && exit 1 || exit 0
