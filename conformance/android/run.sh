#!/bin/bash
# Runs inside the g6-android image, with the repo mounted at /repo (read-write copy).
# Starts the Range/CORS store (web/vector-viewer/e2e/server.mjs, unchanged), boots the emulator
# headless with software GL, runs the instrumented test, and leaves evidence in conformance/android/evidence/.
set -euo pipefail
cd /repo
VV=/srv/vv
mkdir -p $VV/dist
cp -r web/vector-viewer/e2e $VV/e2e
cp -r web/vector-viewer/public/fonts $VV/dist/fonts
node $VV/e2e/server.mjs >/tmp/server.log 2>&1 &
until curl -sf -o /dev/null http://127.0.0.1:4174/golden.pmtiles; do sleep 1; done

test -c /dev/kvm
emulator -avd g6 -no-window -no-audio -no-boot-anim -no-snapshot -gpu swiftshader_indirect \
  -memory 3072 -cores 4 >/tmp/emulator.log 2>&1 &
adb wait-for-device
until [ "$(adb shell getprop sys.boot_completed | tr -d '\r')" = 1 ]; do sleep 3; done
adb shell settings put global window_animation_scale 0
adb shell settings put global transition_animation_scale 0
adb shell settings put global animator_duration_scale 0

cd conformance/android
gradle --no-daemon --console=plain :app:assembleDebug :app:assembleDebugAndroidTest 2>&1 | tail -15
adb install -r app/build/outputs/apk/debug/app-debug.apk >/dev/null
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk >/dev/null

OUT=/repo/conformance/android/evidence
mkdir -p $OUT
RC=0
adb shell am instrument -w -r -e class org.libviprs.conformance.Issue1037ConformanceTest \
  org.libviprs.conformance.test/androidx.test.runner.AndroidJUnitRunner > $OUT/instrumentation.txt || RC=$?
tail -5 $OUT/instrumentation.txt

# the app's files are removed when it is uninstalled, so pull them first
adb pull /sdcard/Android/data/org.libviprs.conformance/files/evidence/. $OUT/ >/dev/null
adb exec-out screencap -p > $OUT/emulator-final.png
adb shell getprop ro.build.version.release | tr -d '\r' > $OUT/android-version.txt
node - "$OUT" /repo/web/vector-viewer/e2e/fixtures/golden.pmtiles /repo/conformance/android/app/build.gradle <<'JS'
const fs = require('fs');
const [out, golden, gradle] = process.argv.slice(2);
const raw = fs.readFileSync(out + '/instrumentation.txt', 'utf8');
const tests = {};
for (const blk of raw.split('INSTRUMENTATION_STATUS: class=').slice(1)) {
  const name = /test=(\w+)/.exec(blk)?.[1];
  const code = /INSTRUMENTATION_STATUS_CODE: (-?\d+)/.exec(blk)?.[1];
  if (name && code !== '1') tests[name] = code === '0' ? 'passed' : 'failed';
}
const metrics = {};
for (const [name, file] of [['remoteRangeMode', 'remote-range'], ['localFileMode', 'local-file'], ['missingArchiveRendersBlank', 'missing-archive']]) {
  metrics[name] = JSON.parse(fs.readFileSync(`${out}/${file}.json`, 'utf8'));
}
const result = {
  issue: 1037,
  platform: 'android',
  maplibre_native_android: /android-sdk:([\d.]+)/.exec(fs.readFileSync(gradle, 'utf8'))[1],
  android_release: fs.readFileSync(out + '/android-version.txt', 'utf8').trim(),
  emulator: 'Android 34 default x86_64 (pixel_5 AVD), KVM, swiftshader_indirect',
  golden_bytes: fs.statSync(golden).size,
  tests,
  metrics,
};
fs.writeFileSync(out + '/result.json', JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result.tests));
JS
adb emu kill >/dev/null 2>&1 || true
kill %1 2>/dev/null || true
exit $RC
