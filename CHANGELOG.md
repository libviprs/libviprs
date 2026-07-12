# Changelog

All notable changes to libviprs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Colour-space and ICC operations ported from libvips (next batch of the
  ported-tests operation surface, libviprs-tests issue #55), in the new
  `colour` module: `Raster::colourspace` (with the typed
  `try_colourspace` form and the new `ColourError`) routing through D65
  XYZ across the libvips route table (Lab, XYZ, LCh, CMC, LabS, scRGB,
  HSV, sRGB, Yxy, Oklab, OkLCh, mono `b-w`/`grey16`, `rgb16`, and the
  no-lcms CMYK approximation), accepting either an `Interpretation` or a
  libvips space nickname (`"srgb"`, `"scrgb"`, ...; `Interpretation` now
  implements `FromStr`); the colour-difference metrics `Raster::de76`,
  `Raster::de00` (CIEDE2000), and `Raster::de_cmc` (the published
  CMC(l:c) 1:1 formula); the `Raster::constant` fixture constructor; and
  real ICC transforms on the pure-Rust moxcms CMS in `Raster::icc_import`
  / `icc_import_with` (with the new `Intent` and `Pcs` enums),
  `Raster::icc_export` / `icc_export_with` (8- or 16-bit device output),
  and `Raster::icc_transform`. Matrix-shaper RGB and grey-TRC profiles
  evaluate exactly from the parsed TRC curves and colorant matrix; LUT
  profiles (CMYK and other table-based classes) run through the moxcms
  LUT engine. Import keeps the source profile attached so a following
  export round-trips through it, and export attaches the profile it
  used, both mirroring libvips.
- Image IO, metadata fields, and library-level free functions ported
  from libvips (seventh batch of the ported-tests operation surface,
  libviprs-tests issue #55), in the new `imageio` module:
  `Raster::save` / `Raster::save_stripped` (extension-dispatched encode
  to PNG, JPEG, or the native `.v` container, with `save_stripped`
  writing pixels only), `Raster::encode_vips`, the named metadata field
  system (`get_field` / `set_field` / `try_set_field` / `get_fields` /
  `get_typeof` / `set_typeof` over the new `MetadataValue` enum, with
  the built-in header fields reading through to the raster header),
  `Raster::set_icc_profile` / `Raster::icc_profile`, and the free
  functions `tokenize`, `parse_thumbnail_geometry` (returning the new
  `ThumbnailGeometry`), `get_max_coord` / `set_max_coord`, and
  `init_from_env` (re-exported at the crate root, where the ported
  tests import them). JPEG save embeds the `icc-profile-data` and raw
  `exif-data` blobs as APP2/APP1 segments and the decoder captures them
  back; `.v` files round-trip the full header (both byte orders) plus
  every attached field, and reject unsupported band formats (float `.v`
  arrives with the float-format batch) and header geometry past the
  `max_coord` ceiling. `decode_file` now also records the source path
  in the `filename` field. Structured EXIF tag encoding (`exif-ifd*-*`
  into the JPEG APP1 TIFF directory) and PNG iCCP embedding are
  deferred to the foreign-format batch.
- Porter-Duff and PDF blend-mode compositing ported from libvips, in
  the new `composite` module: `Raster::composite` /
  `Raster::composite2` with the typed `try_composite2` form
  (`CompositeError`) and the `CompositeMode` enum covering the full
  libvips `VipsBlendMode` set (Clear through Exclusion) plus the four
  PDF non-separable modes (Hue, Saturation, Colour, Luminosity) the
  ported conversion suite exercises. Inputs blend premultiplied in
  `f64` on the libvips `max_alpha` scale (255 unless both inputs are
  16-bit) and write back at the deeper input's depth, so every mode is
  exact to the output quantisation with no float `PixelFormat`
  required.
- Band operations ported from libvips (first batch of the ported-tests
  operation surface, libviprs-tests issue #55), in the new `bands` module:
  `bandjoin`, `bandjoin_const`, `bandjoin_vec`, `bandfold`, `bandunfold`,
  `bandmean`, `bandrank`, `bandand`, `bandor`, `bandeor`, `extract_band`,
  and `extract_bands` as inherent `Raster` methods. Each op also has a
  fallible `try_*` form returning the new typed `BandError` (re-exported at
  the crate root); the plain forms panic on invalid input, matching the
  ported-test call surface. `extract_band`/`extract_bands` accept negative
  indices (from the end), mixed-depth joins promote numerically to 16-bit,
  and constants clamp to the format range with round-to-nearest.
- `PixelFormat` gains `Multi8(n)` / `Multi16(n)` variants so band
  operations can represent results with band counts other than 1, 3, or 4
  (for example 2 bands from `extract_bands`, or width-many bands from
  `bandfold`), plus the canonicalizing constructor
  `PixelFormat::with_channels`. The enum was already `#[non_exhaustive]`,
  so external matches are unaffected; multiband rasters are compute
  intermediates and the tile-encoding sinks reject them with a typed
  `SinkError`. Manifest serialization round-trips the new variants as
  `"multi8:N"` / `"multi16:N"` while the six named formats keep their
  historical tags.

### Fixed

- A `Resume` refused on a plan-hash mismatch now returns
  `EngineError::PlanHashMismatch` before anything touches the output
  directory: the plan-hash gate runs ahead of the advisory run-lock
  acquisition, so a refused resume no longer materialises
  `.libviprs-job.lock` (or any other file) as a side effect.
- `RunLock` removes its `.libviprs-job.lock` file when the guard drops, so a
  finished run leaves no bookkeeping behind in the output tree and a
  crash+resume run now produces a pyramid byte-identical to a clean single
  run. The unlink happens while the exclusive lock is still held, and
  `RunLock::acquire` revalidates after locking that the path still names the
  inode it locked (retrying against the fresh file otherwise), which keeps
  the removal safe against concurrent acquirers.

## [0.4.0] — 2026-07-11

### Breaking

- Tile-lifecycle `EngineEvent` variants carry optional worker attribution
  (issue #67). `TileCompleted`, `TileFailed`, `TileSkippedOnResume`, and
  `RetryAttempted` each gain `worker_id: Option<WorkerId>` and
  `timestamp: Option<SystemTime>`. The in-tree engines set `worker_id: None`
  and stamp `timestamp: Some(_)` on the coordinating thread at emit time; an
  out-of-tree executor layer fills `worker_id` to route events back to the
  worker that produced them. Construction moves to the stamping helpers
  `EngineEvent::tile_completed`, `::tile_failed`, `::tile_skipped_on_resume`,
  and `::retry_attempted`. Code that pattern-matched these variants must add
  `..` (e.g. `TileCompleted { coord, .. }`); code that constructed them by
  struct literal must supply the new fields or use the helpers. `EngineEvent`
  is `#[non_exhaustive]`, so `matches!(e, EngineEvent::TileCompleted { .. })`
  filters are unaffected.

### Added

- Extensible raster drawing framework in the new `draw` module (issue #55).
  A `DrawOp` trait (`fn apply(&self, &mut Raster)`) is the seam: new shapes and
  paint effects plug in as `impl DrawOp` without touching the core `Raster`.
  Ships `Circle` and `Rectangle` ops (outline + filled), the inherent
  convenience methods `Raster::draw`, `draw_circle`, `draw_circle_filled`,
  `draw_rect`, `draw_rect_filled`, and `put_pixel` (clipping, format-agnostic
  pixel write). All drawing clips to the raster bounds and is infallible.
- Pixel utilities on `Raster` (issue #55): `getpoint(x, y) -> Vec<f64>` reads a
  pixel as one `f64` per band (native byte order for 16-bit), and
  `add(&other) -> Raster` combines two rasters pixel-by-pixel, promoting 8-bit
  results to 16-bit so sums do not wrap (16-bit sums saturate at `65535`).
- `WorkExecutor` trait, `StripWorkUnit`, `WorkContext`, and `LocalWorkExecutor`
  in `streaming_mapreduce` (re-exported at the crate root): a plug-in seam at
  the MapReduce MAP-phase strip dispatch, so an out-of-tree executor (process
  pool, distributed worker layer) can substitute for the built-in in-process
  rendering (issue #67). Installed via `EngineBuilder::with_executor(...)`;
  the default `LocalWorkExecutor` is byte-identical to the previous engine.
- `EngineKind::MapReduceHotCache`: a local-only MapReduce variant that holds
  every produced tile in RAM and drains the caller's sink in one batched pass
  at the end of the run, in canonical `(level, row, col)` order, followed by
  a single `finish()` (issue #67). Byte-identical output to the streaming and
  MapReduce engines; explicit opt-in only, never selected by
  `EngineKind::Auto`.
- Worker attribution on `EngineEvent` (issue #67): new `WorkerId` newtype and
  new event variants `StripDispatched`, `StripExecutorDone` (emitted by the
  MapReduce engines on the coordinating thread, in canonical dispatch order,
  stamped with the installed executor's self-reported
  `WorkExecutor::worker_id`, `None` for `LocalWorkExecutor`), plus
  `WorkerJoined`, `WorkerLeft`, and `MemorySnapshot` as vocabulary for
  out-of-tree executor layers (never emitted by the in-tree engines).
  `WorkExecutor::worker_id` has a `None` default, so existing executor
  implementations are unaffected.
- `EngineBuilder::with_observers(Vec<Arc<dyn EngineObserver>>)` and the
  `FanOutObserver` composition behind it (issue #67): every event (and the
  `on_extensions` hatch) fans out to each registered observer in order.
  `with_observer` stays as the single-observer shorthand.

## [0.3.1] — 2026-04-25

Documentation-only patch: the README and crate-root rustdoc shipped on
crates.io with 0.3.0 still showed the removed `generate_pyramid` /
`EngineConfig` entry points. 0.3.1 reships the same code with the README,
`src/lib.rs` rustdoc, `CHANGELOG.md`, and `MIGRATION.md` aligned with the
0.3.0 `EngineBuilder` API.

### Changed

- `README.md` — usage example uses `EngineBuilder` and 2-arg `FsSink::new`
  + `.with_format(...)`; modules and features tables extended for 0.3.0.
- `src/lib.rs` crate-root rustdoc — workflow step 3 references
  `EngineBuilder` / `EngineKind` / `EngineObserver`; cargo features
  documented (`pdfium`, `pdfium-static`, `s3`, `tracing`, `packfile`).

## [0.3.0] — 2026-04-25

The headline change in 0.3.0 is `EngineBuilder`: the five free-function entry
points (`generate_pyramid`, `generate_pyramid_observed`,
`generate_pyramid_streaming`, `generate_pyramid_mapreduce`,
`generate_pyramid_mapreduce_auto`) and `generate_pyramid_resumable` are gone,
replaced by a single typed builder that routes to the monolithic, streaming,
or map-reduce engines. `FsSink` also moved to a 2-arg constructor plus a
`with_format` builder.

See [MIGRATION.md](./MIGRATION.md) for before/after snippets for the most
common 0.2.0 call sites.

### Added

- `EngineBuilder` and `EngineKind` (`Auto`, `Monolithic`, `Streaming`,
  `MapReduce`) — single typed entry point that routes to every engine.
- `EngineSource` enum and `IntoEngineSource` trait so `EngineBuilder::new`
  accepts either `&Raster` or any `T: StripSource` directly.
- `extensions` module and `EngineBuilder::with_extension::<T>(value)` typed
  extension bag.
- `EngineError::IncompatibleSource` for engines that cannot accept the source
  kind they were handed (e.g. `Monolithic` with a strip source).
- `ResumePolicy` (factories `overwrite()`, `resume()`, `verify()`, plus
  `with_checkpoint_every` / `with_checkpoint_root`) wired through the builder
  via `with_resume(...)`. Honored by every engine — Monolithic, Streaming, and
  MapReduce.
- `RetryPolicy::fail_fast()` and `RetryPolicy::with_max(n)` convenience
  constructors; `FailurePolicy` and `RetryPolicy` are first-class on the
  builder via `with_retry(...)` / `with_failure_policy(...)`.
- `DedupeStrategy` exposed on the builder via `with_dedupe(...)` and on
  `FsSink` via `.with_dedupe(...)`.
- Full lifecycle `EngineEvent` variants: `SourceLoadStarted`, `SourceLoaded`,
  `PlanCreated`, `LevelStarted`, `LevelCompleted`, `StripRendered`,
  `BatchStarted`, `BatchCompleted`, `Finished`, `PipelineComplete` (alongside
  the existing `TileCompleted`).
- `EngineObserver` is threaded through every engine via
  `EngineBuilder::with_observer(...)` and `with_observer_arc(...)`.
- `PackfileSink::builder(path)` + `PackfileSinkBuilder` fluent form
  (`.plan(...).format(...).tile_format(...).build()`); `SinkError::MissingField`
  variant surfaced when a required field is omitted.
- `FsSink::with_format(TileFormat)` builder method.
- `stream_verify` module — verify pyramid output against the source.
- `pixel::PixelFormat` is now public and re-exported at the crate root (used by
  `EngineEvent::SourceLoaded`).
- Blanket `impl TileSink for &T` so `&FsSink` and friends work where a
  `TileSink` is required.
- `pdfium-static` cargo feature (pulls in `pdfium` plus
  `pdfium-render/static`) for statically linking libpdfium.

### Changed (breaking)

- `FsSink::new` is now 2-arg: `FsSink::new(dir, plan)`. Format is set via
  `.with_format(TileFormat)`; default remains `TileFormat::Png`.
- `EngineBuilder::with_config(EngineConfig)` is the supported way to override
  the full `EngineConfig`; the old per-call `&EngineConfig` argument is no
  longer accepted by free functions because the free functions no longer
  exist.

### Removed (breaking)

- Free functions `generate_pyramid`, `generate_pyramid_observed`,
  `generate_pyramid_streaming`, `generate_pyramid_mapreduce`,
  `generate_pyramid_mapreduce_auto` — use `EngineBuilder` with the
  appropriate `EngineKind`.
- `generate_pyramid_resumable` — absorbed into
  `EngineBuilder::with_resume(ResumePolicy::resume())`. The internal helper
  is now `pub(crate)`.
- `FsSink::new_with_format(...)` — kept as a deprecated alias of
  `FsSink::new(...).with_format(...)`; will be removed in a future release.

### Fixed

- `pdf_info` honors `/Rotate` when computing page dimensions.
- `PdfiumStripSource` now renders the full page once, then slices strips,
  fixing inconsistencies at strip boundaries on rotated pages.
- `RetryPolicy` backoff rounds nanoseconds to avoid floating-point truncation.
- Phase-3 filesystem tests are gated under Miri via
  `#[cfg_attr(miri, ignore)]`.

### Internal

- Build patches `pdfium-render` against the `libviprs/pdfium-render` fork.
- Version bumped to 0.3.0.

## [0.2.0] — 2025

Phase-3 hardening: manifest v1, sinks, resume, retry, dedupe, tracing.

[0.3.0]: https://github.com/libviprs/libviprs/releases/tag/v0.3.0
[0.2.0]: https://github.com/libviprs/libviprs/releases/tag/v0.2.0
