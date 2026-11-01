# PMTiles on MapLibre Native (Android and iOS)

The golden vector archive (`web/vector-viewer/e2e/fixtures/golden.pmtiles`) and the shared style
(`src/vector/black_on_white.style.json`) are the same files on the web, on Android and on iOS. This page
is about the two ways a phone can reach the archive, and about what "offline" has to mean.

## Remote Range mode

Point the style's source at `pmtiles://https://host/drawing.pmtiles`. MapLibre Native reads the header, the
directories and then single tiles with `Range: bytes=a-b` requests. The host has to answer them with `206`
and a `Content-Range`, the same as for the web viewer (CORS does not matter to a native client, only the
Range support does). `web/vector-viewer/e2e/server.mjs` is the reference store.

## Local file mode

Point the source at `pmtiles://file:///absolute/path/drawing.pmtiles`. The app has to put the file there
itself.

## Offline

PMTiles is not handled by MapLibre Native offline-pack caching.

MapLibre Native's own PMTiles page says the same thing ("PMTiles sources do not support offline pack downloads or caching", in
`platform/android/docs/data/PMTiles.md` of maplibre-native), so don't expect an offline region
(`OfflineManager` on Android, `MLNOfflineStorage` on iOS) to download or hold the drawing. (The ambient cache that newer releases keep for PMTiles
sources is a size-limited cache of what you looked at, not something you can download ahead of time.)

So offline means the app downloads the whole archive itself, once, into its own storage, checks it, and
then opens it with the local `pmtiles://file://` form above. Glyphs have the same problem: if the style's
`glyphs` URL is remote, ship the font files with the app (`asset://`) or download them with the archive.

## What was checked

On Android the instrumented test in `conformance/android/` loads the golden archive with the shared style
in both modes in an emulator, and checks rendered pixels (black linework on white and nothing coloured),
queried features, and the store's request log (ranged `206` reads only in remote mode, one whole-file
download and no further requests in local mode). `conformance/android/evidence/` has the screenshots and
the numbers. The iOS run is tracked in #1037.
