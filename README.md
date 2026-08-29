<p align="center">
  <img src="images/libviprs-logo-claws.svg" alt="libviprs" width="200">
</p>

<h1 align="center">libviprs</h1>

<p align="center">
  <a href="https://github.com/libviprs/libviprs/actions/workflows/ci.yml"><img src="https://github.com/libviprs/libviprs/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/libviprs/libviprs/actions/workflows/merge-gate.yml"><img src="https://github.com/libviprs/libviprs/actions/workflows/merge-gate.yml/badge.svg" alt="Merge Gate"></a>
  <img src="https://img.shields.io/badge/rust-1.97%2B-orange?logo=rust" alt="Rust 1.97+">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
</p>

A pure-Rust, thread-safe image pyramiding engine. Inspired by [libvips](https://www.libvips.org/), built from scratch for the AEC/construction domain.

Takes blueprint PDFs and images, extracts raster data, optionally geo-references it, and generates tile pyramids (DeepZoom, XYZ, Google Maps) suitable for web-based viewers.

**Try it interactively:** the [CLI docs page](https://libviprs.org/cli/) lets you tick flags and copy a complete Rust program — start from the [pyramid base setup](https://libviprs.org/cli/#pyramid-base-setup) and toggle features in the [generator panel](https://libviprs.org/cli/#cli-generator).

## Features

- **PDF extraction** — extract embedded raster images from scanned blueprint PDFs via lopdf (pure Rust, no C dependencies)
- **PDF rendering** — render vector PDFs (AutoCAD exports, text, paths) via PDFium, with optional [memory-budgeted rendering](https://libviprs.org/cli/#flag-memory-budget) (optional `pdfium` feature)
- **Image decoding** — 17 containers, each identified from its own magic bytes rather than the file extension: `.v` (native libvips), Ultra HDR, JPEG, PNG, TIFF, GIF, WebP, JPEG XL, Radiance, FITS, OpenEXR, NIfTI, AVIF, JPEG 2000, MATLAB, Netpbm and Analyze. SVG rasterises too, behind the `svg` feature, and is the one that cannot be sniffed. Most of these are decoders in this crate; the `image` crate carries JPEG and PNG
- **Tile pyramid generation** — three engines (Monolithic, Streaming, MapReduce) routed through `EngineBuilder` / `EngineKind` (`Auto` by default), with backpressure and configurable tile size and overlap (see [`--parallel`](https://libviprs.org/cli/#flag-parallel))
- **Layout formats** — DeepZoom (`.dzi` + directory tree), XYZ (`z/x/y`), and Google Maps (`z/y/x`, power-of-2 grids)
- **Centre support** — centre image within the tile grid with even background padding on all sides
- **Tile encoding** — PNG, JPEG (configurable quality), or raw pixel output
- **Blank tile optimization** — configurable `BlankTileStrategy` to either emit full tiles or write 1-byte placeholders (`BLANK_TILE_MARKER`) for uniform-color regions, reducing disk usage for sparse images
- **Edge tile background** — configurable background color (`background_rgb`) for padding partial tiles at image edges (defaults to white)
- **Geo-referencing** — affine transform mapping pixel coordinates to geographic coordinates, GCP support ([`--geo-reference`](https://libviprs.org/cli/#flag-geo-reference))
- **Restart-safe runs** — checkpoint and [resume](https://libviprs.org/cli/#flag-resume) interrupted jobs, with content-addressed [tile dedupe](https://libviprs.org/cli/#flag-dedupe) and per-tile [checksums](https://libviprs.org/cli/#flag-checksums)
- **Sinks** — filesystem, [packfile](https://libviprs.org/cli/#flag-packfile) (tar/zip), and [S3-compatible](https://libviprs.org/cli/#flag-s3) object stores
- **Observability** — progress events, per-level callbacks, peak memory tracking, optional structured [tracing](https://libviprs.org/cli/#flag-tracing)

## Usage

```rust
use libviprs::{
    extract_page_image,
    BlankTileStrategy,
    EngineBuilder,
    EngineKind,
    FsSink,
    Layout,
    PyramidPlanner,
    TileFormat,
};
use std::path::Path;

// ──────────────────────────────────────────────────────────────────────
// 1. Decode the source.
//    extract_page_image pulls the embedded raster out of a scanned PDF;
//    use libviprs::decode_file for plain image inputs (PNG/JPEG/TIFF).
// ──────────────────────────────────────────────────────────────────────
let raster = extract_page_image(
    Path::new("blueprint.pdf"),  // input path (PDF here; any image works too)
    1,                           // 1-based PDF page number
).unwrap();

// ──────────────────────────────────────────────────────────────────────
// 2. Plan the pyramid.
//    PyramidPlanner computes per-level dimensions, tile counts, and
//    canvas size — no pixels are touched yet.
// ──────────────────────────────────────────────────────────────────────
let planner = PyramidPlanner::new(
    raster.width(),    // source width in pixels
    raster.height(),   // source height in pixels
    256,               // tile size (DeepZoom default; 512 for HiDPI)
    0,                 // pixel overlap between adjacent tiles
    Layout::DeepZoom,  // tile naming convention (also: Xyz, Google)
).unwrap();

let plan = planner.plan();

// ──────────────────────────────────────────────────────────────────────
// 3. Configure where the tiles get written.
//    FsSink writes to a local directory; libviprs also ships
//    ObjectStoreSink (S3) and PackfileSink (.tar/.tar.gz/.zip).
// ──────────────────────────────────────────────────────────────────────
let sink = FsSink::new(
    "output_tiles",  // output directory (created if missing)
    plan.clone(),    // pyramid plan tile paths are derived from
)
.with_format(TileFormat::Png);  // also: TileFormat::Jpeg { quality: u8 } | Raw

// ──────────────────────────────────────────────────────────────────────
// 4. Run the engine.
//    EngineKind::Auto picks monolithic / streaming / mapreduce based on
//    the source kind and any with_memory_budget value supplied.
// ──────────────────────────────────────────────────────────────────────
let result = EngineBuilder::new(
    &raster,  // source raster from step 1
    plan,     // pyramid plan from step 2
    sink,     // tile sink from step 3
)
.with_engine(EngineKind::Auto)                        // auto-select engine
.with_concurrency(4)                                   // worker threads for tile extraction
.with_blank_strategy(BlankTileStrategy::Placeholder)   // collapse uniform tiles into 1-byte placeholders
.run()
.unwrap();

println!(
    "{} tiles across {} levels ({} blank tiles skipped)",
    result.tiles_produced,    // total tile files written
    result.levels_processed,  // number of pyramid levels
    result.tiles_skipped,     // tiles emitted as blank placeholders
);
```

> See [interactive example](https://libviprs.org/cli/#cli-generator) — tick flags on the CLI docs page to generate a tailored version of this snippet.

## Modules

Every public module, in four groups. `tests/crate_doc_matches_the_crate.rs`
holds this list against `pub mod` in `src/lib.rs`, so a new module is missing
from here for exactly one commit.

### Pipeline

| Module | Description |
|---|---|
| `source` | Content sniffing and decode into a canonical `Raster` (see the decode list above) |
| `pdf` | PDF parsing (lopdf) and optional rendering (PDFium), including budgeted render |
| `raster` | Pixel buffer, region views, format normalization |
| `pixel` | Pixel format definitions: the 14 `PixelFormat` carriers listed below |
| `planner` | Tile math, level computation, layout generation |
| `resize` | Downscaling for pyramid levels |
| `engine` | Monolithic in-memory tile extraction with backpressure, blank tile detection |
| `engine_builder` | Typed `EngineBuilder` / `EngineKind` entry point routing to Monolithic, Streaming, or MapReduce engines |
| `streaming` | Sequential strip engine, `StripSource` trait, `RasterStripSource`, memory-budget helpers |
| `streaming_mapreduce` | Parallel strip engine and `MapReduceConfig` |
| `sink` | Tile output (filesystem, memory, slow sink for testing) |
| `sink_packfile` | `PackfileSink` writing tiles into a tar/zip archive (gated by `packfile`) |
| `sink_object_store` | `ObjectStoreSink` for user-injected object storage backends (gated by `object-store-sink`; the deprecated `s3` alias also enables it) |
| `resume` | Job checkpoints and resume policy for restart-safe runs |
| `retry` | Failure / retry policy and `RetryingSink` wrapper |
| `dedupe` | Content-addressed tile deduplication |
| `manifest` | `Manifest` v1 schema and `ManifestBuilder` describing the produced pyramid |
| `checksum` | Tile checksum modes and verification reports |
| `stream_verify` | Verify pyramid output against the original source |
| `geo` | Affine geo-transform, GCP solving, bounding box computation |
| `observe` | Progress events, lifecycle observers, memory tracking |

### Formats and containers

| Module | Description |
|---|---|
| `codec` | The shared error taxonomy (`DecodeError`, `EncodeError`) and format-option enums every codec below uses |
| `connection` | Streaming IO connections: `Source`, `Target`, and the format-name save route `encode_to_buffer` / `encode_to_target` |
| `imageio` | `Raster::save`'s extension route, the metadata field system, and the native `.v` container |
| `encode` | The JPEG and PNG encoders behind `Raster::encode_jpeg`, `Raster::encode_png` and the `jpegsave_buffer` family |
| `encode_tiff` | TIFF load and save, including multi-page reads and the `.tif` / `.tiff` save route |
| `gif` | GIF still-image load and save, and the `gifsave` option surface |
| `webp` | WebP load and lossless save, still and animated |
| `jxl` | JPEG XL load and lossless save (gated by `jxl`) |
| `jp2k` | JPEG 2000 load and save (gated by `jp2k`) |
| `avif` | AVIF still-image load, an AV1 keyframe in an ISOBMFF container (gated by `avif`) |
| `svg` | SVG rasterisation to an RGBA raster (gated by `svg`) |
| `uhdr` | Ultra HDR: the gain-map JPEG libvips reads with `uhdrload` and writes with `uhdrsave` |
| `radiance` | Radiance HDR load and save: RGBE bytes in, three-band float out |
| `exr` | OpenEXR load: scene-linear samples in, float bands out |
| `fits` | FITS load and save, 80-column ASCII header and all |
| `mat` | MATLAB level 5 load |
| `nifti` | NIfTI load: a fixed-size header in, a raw voxel array out |
| `analyze` | Analyze 7.5 load, from the `.hdr` half of the `.hdr` + `.img` pair |
| `textio` | The libvips `matrix` and `csv` text codecs, plus the binary Netpbm containers |
| `frames` | The page model for multi-frame images: how a raster's rows divide into pages |
| `foreign_stubs` | Typed stubs for the genuinely external formats this build does not link (HEIF/AVIF encode, ImageMagick, DeepZoom buffers) |

### Image operations

| Module | Description |
|---|---|
| `arithmetic` | Arithmetic and whole-image statistics, ported from libvips |
| `bands` | Band (channel) operations |
| `colour` | Colour-space conversion and ICC transforms |
| `composite` | Alpha compositing (`vips_composite2`) |
| `conversion` | Conversion, orientation, and the colour-adjacent operations |
| `convolution` | Convolution and correlation |
| `create` | Image generators (the libvips `create` family) |
| `draw` | In-place raster drawing |
| `extract` | Extract, crop, and geometry placement |
| `freqfilt` | Frequency-domain filters |
| `histogram` | Histogram operations |
| `matrix` | Matrix-image and LUT-inversion operations |
| `morphology` | Morphological operations |
| `mosaicing` | Mosaicing operations |
| `resample` | Resampling: resize, reduce, shrink, affine, thumbnail |

### Support

| Module | Description |
|---|---|
| `cancel` | Cooperative cancellation for long-running generation |
| `error` | Crate-level umbrella error over the per-module operation errors |
| `extensions` | Typed extension map for pipeline-level context |
| `verify` | Verify-mode entry points |

### Pixel formats

`PixelFormat` has 14 carriers, and the signed and 32-bit ones are this release's headline break rather than a footnote: `Gray8`, `Gray16`, `Rgb8`, `Rgba8`, `Rgb16`, `Rgba16`, `RgbaF32`, `Multi8`, `Multi16`, `FloatF32`, `Uint32`, `Int8`, `Int16` and `Int32`.

## Features

| Feature | Default | Description |
|---|---|---|
| `pdfium` | off | Enables `render_page_pdfium()`, `render_page_pdfium_budgeted()`, and `PdfiumStripSource` for vector PDF rendering. Requires libpdfium at runtime. |
| `pdfium-static` | off | Implies `pdfium` and links libpdfium statically via `pdfium-render/static`. |
| `object-store-sink` | off | Enables the `sink_object_store` module (`ObjectStoreSink` against a user-injected `ObjectStore`). Ships no built-in S3 transport — a backend must be injected. |
| `s3` | off | **Deprecated alias** for `object-store-sink`, retained so consumers pinned to the old feature name keep building. Prefer `object-store-sink`; the `s3` alias will be removed in a future release. |
| `tracing` | off | Emits structured `tracing` spans and events from the engine pipeline. |
| `packfile` | off | Enables `PackfileSink` for writing tiles into a tar or zip archive. |
| `svg` | off | Enables the SVG rasteriser behind `decode_svg`. Costs 29 crates (`resvg` and its tree), which is why it is opt-in; without it `decode_svg` returns a typed `Unsupported`. |
| `jxl` | off | Enables the JPEG XL loader and lossless encoder: `decode_jxl`, `Raster::encode_jxl`, `Raster::save_jxl`, the `.jxl` row in `Raster::save` and in `encode_to_buffer`. Costs 21 crates (`jxl-oxide`, `zune-jpegxl` and their trees, including `tracing`), which is why it is opt-in; without it every entry point still exists and returns a typed refusal. |
| `jp2k` | off | Enables the JPEG 2000 loader and encoder: `decode_jp2k`, `Raster::encode_jp2k`, `Raster::save_jp2k`, and the `.jp2` / `.j2k` rows in the content sniffer. Costs **2 crates** (`hayro-jpeg2000` and `openjpeg2-pure-rs`, neither of which has a dependency of its own), so what it buys back is compile time rather than crate count; without it every entry point still exists and returns a typed refusal. |
| `avif` | off | Enables the AV1 decode inside `decode_avif` (still images only). Costs 16 crates (`rav1d` and its runtime tree); without it the entry point still parses the container, checks the codec and applies all three decode limits, and refuses the decode itself. |
| `serde` | off | Adds public `Serialize` / `Deserialize` derives to the wire and config types (`PyramidPlan`, `EngineConfig`, `TileCoord`, `Layout`, ...) so an out-of-process caller can rebuild a job from JSON. Adds no dependencies. |
| `test-util` | off | Exposes the crate's test-only sink doubles (`SlowSink`) to dependent crates, chiefly the external `libviprs-tests` suite. Adds no dependencies. |

## Requirements

- Rust 1.97+ (edition 2024)
- libpdfium shared library (only if using the `pdfium` feature)

### PDFium setup

The `pdfium` feature requires `libpdfium.so` at runtime. Pre-compiled binaries built from source are available from [libviprs-dep](https://github.com/libviprs/libviprs-dep/releases):

```bash
# x86_64
curl -L -o pdfium.tgz \
  https://github.com/libviprs/libviprs-dep/releases/download/pdfium-7881/pdfium-linux-x64.tgz

# arm64
curl -L -o pdfium.tgz \
  https://github.com/libviprs/libviprs-dep/releases/download/pdfium-7881/pdfium-linux-arm64.tgz

# Extract and install
tar xzf pdfium.tgz
sudo cp pdfium-linux-*/lib/libpdfium.so /usr/local/lib/
sudo ldconfig
```

See the [libviprs-dep pdfium README](https://github.com/libviprs/libviprs-dep/tree/main/pdfium) for building PDFium from source or finding other versions.

## Related Crates

| Crate | Description |
|---|---|
| [libviprs-cli](../libviprs-cli) | Command-line interface (`viprs` binary) |
| [libviprs-tests](../libviprs-tests) | Integration tests and fixtures, including end-to-end PDF-to-pyramid tests for `blueprint.pdf` and `blueprint-mix.pdf` |

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) has the dependency rule: what this crate will
and will not take on, why `build.rs`, `links =` and a `-sys` suffix are none of
them the thing that decides it, and where the two carve-outs (`packfile` and
`pdfium`) sit. Read it before adding a dependency.

Two of its three clauses are mechanical, and `tests/dependency_policy.rs` checks
those two against the graph cargo actually resolves, on every `cargo test`: a
dependency that goes looking for a library on the build machine, or that
compiles native source that did not come down with it, turns the suite red
rather than getting caught in review. The third clause, no linking a
third-party library somebody has to install first, has no mechanical check and
cannot have one, because nothing in a manifest tells a crate that needs an
installed library apart from one that does not. That one is applied by hand,
with the checklist in CONTRIBUTING.md.

## CI

GitHub Actions runs two workflows:

**CI** (every push and PR) — `.github/workflows/ci.yml`:
- `cargo fmt --check` — formatting
- `cargo clippy -D warnings` — lint, once per feature. Since #844 that is the default build plus `pdfium`, `object-store-sink`, `tracing`, `avif`, `svg`, `jxl`, `packfile`, `serde` and `jp2k`, because code behind any other `cfg` used to be linted by nothing
- `cargo test` — unit tests

**Merge Gate** (PRs targeting `release`, required to merge) — `.github/workflows/merge-gate.yml`:
- `cargo +nightly miri test` — undefined behavior detection
- Loom concurrency tests

### Running CI locally

A `Makefile` mirrors the full CI pipeline. Run everything with:

```sh
make ci
```

Or run individual checks:

```sh
make fmt      # check formatting
make clippy   # clippy over the default build and each of the nine features CI lints
make test     # unit tests
make miri     # miri (requires nightly + miri component)
make loom     # loom concurrency tests
```

> **Prerequisites:** `make miri` requires the nightly toolchain with the miri component.
> Install with: `rustup toolchain install nightly --component miri`

