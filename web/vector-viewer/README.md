# Vector viewer, a reference integration

A DWG-derived vector PMTiles archive, shown as black linework and text on white, streamed from
wherever the file lives with HTTP Range requests. The browser never sees the source DWG.

```text
.dwg  ->  libviprs (decode, tile)  ->  drawing.pmtiles  ->  this app (MapLibre GL JS + pmtiles)
```

## Use it

```sh
npm install
npm run build
npx playwright test              # needs the Playwright browsers; the CI image has them
```

`?archive=<url>` picks the archive. The default is `http://<host>:4174/golden.pmtiles`, the object
store the specs start beside the app (`e2e/server.mjs`).

The archive's host has to do two things, which `e2e/server.mjs` models and the specs check:

* answer `Range: bytes=a-b` with `206` and a `Content-Range`, not the whole object with `200`;
* send CORS headers that let the app's origin read it: `Access-Control-Allow-Origin`, and
  `Access-Control-Expose-Headers: Content-Range, Content-Length, Accept-Ranges, ETag`. `Range` is
  allowed as a request header on the preflight.

A store that gets the second one wrong makes the app show an alert instead of a blank map.

## The style

`../../src/vector/black_on_white.style.json` is the one style, shared with the Rust crate
(`libviprs::vector::style::black_on_white_style`) and with the mobile clients. It has three layers:
white paper, the `lines` layer in black and the `text` layer in black. It uses only properties that
MapLibre GL JS, MapLibre Native Android and MapLibre Native iOS all implement, and `text-allow-overlap:
true` rather than the web-only `text-overlap: always`. `tests/vector_1033_style.rs` holds the list.

Label size follows the drawing: the tiler writes `h0`, the label's height in pixels at zoom 0, and the
style interpolates it exponentially with base 2 from zoom 0 to 24, which is exactly `h0 * 2^zoom`. MapLibre
logs a warning that `text-size` is at or above 255 for that last stop. It is advisory: the size at any zoom
a label is actually shown at is `height` drawing units, which is what a drawing should do.

Glyphs are served from `public/fonts/` (provenance in `public/fonts/PROVENANCE.md`).

## How crisp, and for how long

Lines are 1.25 px wide at every zoom, so the stroke does not thicken or blur. The archive is generated to
a `max_zoom` and MapLibre scales its tiles beyond that: the specs hold the stroke to one crisp stroke
through every generated zoom and two zooms of over-zoom, and the viewer caps zoom at `max_zoom + 2`. See
the precision budget in `libviprs::vector::tiler` for what a tile unit is in drawing units at each zoom.

## Themes and contrast

The chrome has a light and a dark theme (`prefers-color-scheme`, or the toggle, or `?theme=`). The drawing
is always black on white paper. `e2e/contrast.spec.ts` measures every text and background pair, the button
and map-frame borders and the focus ring from the computed styles of the rendered page, in both themes, and
fails under WCAG AA (4.5:1 for text, 3:1 for the boundaries of controls).

## The fixture

`e2e/fixtures/golden.pmtiles` is built by `tests/vector_1033_fixture.rs` in the Rust crate, and that test
fails if the committed bytes differ from what the tiler makes now. After an intended change:

```sh
LIBVIPRS_UPDATE_VIEWER_FIXTURE=1 cargo test --test vector_1033_fixture
npx playwright test --update-snapshots
```

Snapshots are Chromium on Linux with software GL (`e2e/__screenshots__/`).

## On a phone

The same archive and the same style file work in MapLibre Native. The details, and what a mobile app has to
do for offline use, are in the L2.4 conformance work.
