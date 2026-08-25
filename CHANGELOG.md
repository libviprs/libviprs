# Changelog

All notable changes to libviprs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- A `.v` file tagged `OkLab` or `OkLch` now carries the real libvips
  interpretation codes `30` and `31` in its header `Type` word, so it
  interoperates with vips instead of only with libviprs (issue #535). libvips
  8.18 assigned those codes (`VIPS_INTERPRETATION_OKLAB` and
  `VIPS_INTERPRETATION_OKLCH`, `libvips/include/vips/image.h:115-116`), but
  libviprs still wrote the private extension codes `1000` and `1001` it had
  picked while libvips had none. The consequence ran both ways: a `.v` written
  by real vips came back tagged `Multiband`, because `30` matched no arm of the
  reader and the raster fell through to format inference, and a `.v` written by
  libviprs was unreadable as OkLab anywhere else. This changes what goes on
  disk: newly written files hold `30` / `31` where they used to hold `1000` /
  `1001`. The change is one-way. The reader keeps `1000` and `1001` as legacy
  aliases, so files libviprs has already written still load with their
  OkLab/OkLch tag intact, but nothing emits those codes any more, and a file
  written by this version does not read as OkLab on an older libviprs.

## [0.4.0] — 2026-07-20

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
  pixel write). All drawing clips to the raster bounds. (Ink-width validation
  was added later; see the **Breaking (drawing)** entry under _Changed_ — the
  draw ops now reject a wrong-width ink rather than broadcasting it.)
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
- Image generators ported from libvips (create batch of the ported-tests
  operation surface, libviprs-tests issue #55), in the new `create`
  module, all as associated constructors on `Raster` with `try_*`
  fallible forms and the new `CreateError`: the basic patterns `black` /
  `black_bands`, `xyz`, `eye`, `zone`, and `sines`; the seeded noise
  generators `gaussnoise`, `perlin`, and `worley` (bit-exact
  reproductions of the libvips FNV pixel-hash RNG, with `*_seeded` forms;
  the default seed is pinned to 0 so unseeded output is deterministic)
  and `fractsurf` (noise shaped by a fractal power spectrum through an
  internal 2D DFT, the `vips_freqmult` pipeline); the LUT and matrix
  constructors `buildlut`, `tonelut`, and `from_matrix`; the full
  frequency-mask family `mask_ideal` / `mask_ideal_ring` /
  `mask_ideal_band`, `mask_gaussian` / `mask_gaussian_ring` /
  `mask_gaussian_band`, `mask_butterworth` / `mask_butterworth_ring` /
  `mask_butterworth_band`, and `mask_fractal`, reproducing the
  `create/mask.c` semantics exactly (FFT layout with an `optical`
  quadrant swap, the DC component forced to 1.0 unless `nodc`, uchar
  truncation); signed distance fields `sdf` (circle, box, rounded-box,
  line) with the new `SdfParams`; and real text rendering `text` on the
  pure-Rust `ab_glyph` rasteriser with the bundled Bitstream Vera Sans
  face (`fonts/`), honouring dpi, word/char/word-char/none wrapping, and
  the width x height auto-fit search. `tests/create_ported_surface.rs`
  pins the ported call surface from the external-crate position.
- The arithmetic reductions (`avg`, `deviate`, `min` / `max`,
  `minpos` / `maxpos`) and relational maps now read `f32` rasters, so
  they work on the float images the create generators emit; the
  spellings `Raster::max_value` / `Raster::min_value` the ported suites
  use are provided as aliases. Mutating arithmetic ops still reject
  float inputs loudly.
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
  functions `tokenize` and `parse_thumbnail_geometry` (returning the new
  `ThumbnailGeometry`). JPEG save embeds the `icc-profile-data` and raw
  `exif-data` blobs as APP2/APP1 segments and the decoder captures them
  back; `.v` files round-trip the full header (both byte orders) plus
  every attached field, and reject unsupported band formats (float `.v`
  arrives with the float-format batch) and header geometry past the
  configured coordinate ceiling. `decode_file` now also records the source
  path in the `filename` field. Structured EXIF tag encoding (`exif-ifd*-*`
  into the JPEG APP1 TIFF directory) and PNG iCCP embedding are
  deferred to the foreign-format batch.
- `DecodeLimits` gains a `max_coord` field (with the `with_max_coord`
  builder setter) carrying the libvips `VIPS_MAX_COORD` single-axis
  dimension ceiling (10,000,000 px per axis default) as per-decode state.
  It is enforced on untrusted header geometry — before any pixel
  allocation — by **every** decoder: the native `.v` reader and the
  `image`-crate raster path (PNG/JPEG/TIFF) alike, returning the new
  typed `SourceError::CoordLimitExceeded` on an over-ceiling axis.
  `DecodeLimits` remains `#[non_exhaustive]`, so the field is additive.
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

### Changed

- Image-image `Raster::sub` / `try_sub` now promotes to a `Float32` raster
  instead of routing through the integer round-and-saturate writer, so
  negative differences are preserved instead of saturating to `0`
  (e.g. `10 - 200` now reads back `-190`, not `0`). This matches the
  `vips_subtract` promotion table, which outputs signed `short` for `uchar`
  input — carried here as float so the signed result survives. Because the
  output floats, `try_sub` no longer returns
  `ArithmeticError::FloatUnsupported` for float inputs; a cast-then-subtract
  chain over float intermediates now works. The constant / per-band
  `sub_const` / `sub_vec` forms are unchanged — they still saturate at `0`,
  matching `vips_linear`'s requested-format (integer) output (libviprs#282).
- The cargo feature that gates the `sink_object_store` module
  (`ObjectStoreSink`) has been **renamed** from `s3` to `object-store-sink`
  (libviprs#345). The new name reflects that the module is a generic
  object-store sink driven by a user-injected `ObjectStore` backend — it pulls
  in no extra crates and ships no built-in S3 transport — rather than a
  ready-to-use S3 client. `s3` is kept as a deprecated alias
  (`s3 = ["object-store-sink"]`) so consumers pinned to the old feature name
  keep building; see the _Deprecated_ entry below. The README feature/module
  tables and the crate-root "Feature flags" rustdoc now present
  `object-store-sink` as the canonical name.
- The averaging resamplers `Raster::reduce` / `reduceh` / `reducev`,
  `shrink` / `shrinkh` / `shrinkv`, and `resize` now premultiply alpha
  around the whole operation for images with an alpha band: the source is
  premultiplied once into a float working buffer, every separable box /
  kernel / affine pass runs in that premultiplied space, and the result is
  un-premultiplied once at the end (the `vips_resize` bracket). This
  coverage-weights the colour so the meaningless RGB of transparent pixels
  can no longer bleed into opaque neighbours (the dark fringe at
  transparency boundaries), and it fixes a mixed downscale-one-axis /
  upscale-other-axis `resize` that previously fed straight-alpha colour to
  the affine enlargement. Bracketing once into float (instead of once per
  axis into a same-bit-depth integer intermediate) also removes the
  low-alpha colour banding a per-axis integer round-trip introduced. The
  single-tap Nearest kernel is unchanged — it stays an exact pick with no
  premultiply. This is a deliberate divergence from the bare `vips_reduce*`
  / `vips_shrink*` namesakes, which do not premultiply (only `vips_resize`
  does); it matches a premultiplied vips pipeline
  (`premultiply | reduce/shrink | unpremultiply`).
- The `max_coord` single-axis coordinate ceiling (libvips `VIPS_MAX_COORD`)
  is now per-decode state on `DecodeLimits::max_coord` instead of a mutable
  process-global, and it is enforced uniformly by **every** decoder. The
  `image`-crate raster path (PNG/JPEG/TIFF) previously honoured only
  `max_width` / `max_height` / `max_alloc_bytes` and silently ignored the
  coordinate ceiling; it now rejects an over-ceiling declared dimension
  with the typed `SourceError::CoordLimitExceeded`, on the header geometry
  before the frame is allocated, matching the native `.v` reader
  (libviprs#349). **Security note:** the former process-global
  (`set_max_coord` / `VIPS_MAX_COORD`) is now inert — a deployment that
  relied on it to *tighten* the ceiling below the 10,000,000-px default
  now decodes at that default on every non-`_with_limits` entry point with
  no runtime signal. Reinstate a tighter bound by passing a `DecodeLimits`
  with the desired `max_coord` to a `*_with_limits` decode call.
- The total `max_pixels` ceiling is now enforced on the raster path
  (PNG/JPEG/TIFF) against the declared header geometry *before* the output
  frame is allocated — matching the native `.v` reader — in addition to the
  existing post-decode re-verification in raster construction. Previously an
  over-`max_pixels` raster was only rejected after the frame was materialised.

### Deprecated

- The `s3` cargo feature is deprecated in favour of `object-store-sink`
  (libviprs#345). It survives only as an alias (`s3 = ["object-store-sink"]`)
  that enables the same `sink_object_store` module, so a consumer pinned to the
  old feature name keeps building unchanged. It carries no `since` version
  because the rename and deprecation happen within the same post-0.4.0
  unreleased cycle — no released version ever gated the module under a name
  other than `s3`, and the alias is scheduled for removal in a future release.
- **Breaking (drawing): a wrong-width ink is now rejected instead of broadcast.**
  The `draw` module entry points validate that `ink` is exactly one whole pixel
  wide for the target raster (issues #294/#346). Earlier the ops cycled a
  too-short ink to fill the pixel, so a single-value ink such as `&[128]`
  broadcast to a uniform shade on a multi-band raster (and a 3-byte RGB ink
  silently painted an `Rgba8` alpha band from the red byte). That silent cycle
  is a channel-corruption route, so a mismatched ink now fails up front: the
  panicking `Raster::draw_*` forms and the generic `Raster::draw(&op)` path
  panic, and the `try_draw_*` forms (plus the inherently fallible `draw_flood` /
  `draw_flood_blob`) return `Err(DrawError::InkLengthMismatch)`. The raw
  `Raster::put_pixel` escape hatch is unchanged and remains the deliberate
  opt-in for the shorter-ink cycling broadcast.

### Removed

- The inert process-global coordinate-ceiling shims `get_max_coord`,
  `set_max_coord`, and `init_from_env`, the backing `MAX_COORD` static, and
  their crate-root re-exports (libviprs#462). They were deprecated on
  arrival and consulted by no decoder; the single-axis ceiling lives solely
  on the per-decode `DecodeLimits::max_coord` field, enforced by every
  decoder — build it with `DecodeLimits::with_max_coord`. No `since` version
  applied because no released version ever exposed them as live API. The
  `DEFAULT_MAX_COORD` constant is retained as the `DecodeLimits::max_coord`
  default.

### Fixed

- A Resume/Overwrite run now takes the advisory run lock on the *union* of the
  directories it mutates — the resolved checkpoint root **and** the sink's own
  output dir — instead of only one. Locking a single directory left the other
  exposed: guarding just the checkpoint root reopened the issue #126 output-wipe
  hazard, guarding just the sink dir reopened the issue #276 checkpoint-flush
  race. The locks are taken in a deterministic order so contending jobs cannot
  each grab one directory and deadlock; acquisition stays non-blocking. The two
  entries are de-duplicated by *canonical* path, so an explicit `checkpoint_root`
  that names the same physical directory as the sink through a different spelling
  (`out` vs `./out`, relative vs absolute, `a/../out`, a symlink alias) collapses
  to a single lock rather than making the run acquire the one `.libviprs-job.lock`
  twice and refuse *itself* with `ResumeError::Locked`. That error's wording is
  generalised from "checkpoint root is locked" to "run directory is locked",
  since it can now name either guarded directory.
- `downscale_to` and the 2x box halver `downscale_half` (via
  `downscale_half_alpha`) now round the alpha-weighted colour and averaged
  alpha half-up, matching their own no-alpha branches, so a fully-opaque
  RGBA image downscales bit-identically to its RGB twin — and identically
  through both paths — instead of carrying the systematic -0.5 LSB
  truncation bias the alpha path previously introduced. Both alpha-weighted
  regions are also scanned with integer `u64` accumulators that stay exact
  where the former `f64` colour accumulator would drop low bits on extreme
  16-bit downscales.
- `composite2` mixed bit-depth blends now key the 0..65535 vs 0..255
  read/write scale on each input's *resolved* interpretation
  (`Raster::interpretation`, gated on the actual 2-byte storage depth),
  matching libvips' per-input `max_band` from `vips_interpretation_max_alpha`
  after `formatalike`. A genuine 16-bit layer (e.g. a decoded 16-bit PNG,
  untagged through `Raster::new`) is honoured on the 65535 scale so it blends
  against an 8-bit layer at a true ratio instead of 257:1 and its alpha is not
  capped at 255; a genuine-16 result is stamped `Rgb16` / `Grey16` so a
  re-composite reads it on the same scale (issues #443–#449). To keep the
  crate's promoted-container idiom working across the module boundary, the
  constant arithmetic ops (`add_const` / `mul_const` / `pow_const` /
  `add_vec`, and the widening binary path) now stamp their widened 16-bit
  output with the source interpretation, so an 8-bit input promoted into a
  16-bit container resolves to a non-genuine-16 space and a fully-opaque
  promoted overlay stays visible over an 8-bit base instead of collapsing to
  ~0.4%. The genuine-16 write-back scale and interpretation stamp are gated to
  integer output containers, so a genuine-16 input composited against a float
  raster no longer inflates the float output ~257x or mis-tags it as USHORT.
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
