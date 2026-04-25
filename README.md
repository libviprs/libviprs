<p align="center">
  <img src="images/libviprs-logo-claws.svg" alt="libviprs" width="200">
</p>

<h1 align="center">libviprs</h1>

<p align="center">
  <a href="https://github.com/libviprs/libviprs/actions/workflows/ci.yml"><img src="https://github.com/libviprs/libviprs/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/libviprs/libviprs/actions/workflows/merge-gate.yml"><img src="https://github.com/libviprs/libviprs/actions/workflows/merge-gate.yml/badge.svg" alt="Merge Gate"></a>
  <img src="https://img.shields.io/badge/rust-1.85%2B-orange?logo=rust" alt="Rust 1.85+">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
</p>

A pure-Rust, thread-safe image pyramiding engine. Inspired by [libvips](https://www.libvips.org/), built from scratch for the AEC/construction domain.

Takes blueprint PDFs and images, extracts raster data, optionally geo-references it, and generates tile pyramids (DeepZoom, XYZ, Google Maps) suitable for web-based viewers.

## Features

- **PDF extraction** — extract embedded raster images from scanned blueprint PDFs via lopdf (pure Rust, no C dependencies)
- **PDF rendering** — render vector PDFs (AutoCAD exports, text, paths) via PDFium, with optional memory-budgeted rendering (optional `pdfium` feature)
- **Image decoding** — JPEG, PNG, TIFF via the `image` crate
- **Tile pyramid generation** — three engines (Monolithic, Streaming, MapReduce) routed through `EngineBuilder` / `EngineKind` (`Auto` by default), with backpressure and configurable tile size and overlap
- **Layout formats** — DeepZoom (`.dzi` + directory tree), XYZ (`z/x/y`), and Google Maps (`z/y/x`, power-of-2 grids)
- **Centre support** — centre image within the tile grid with even background padding on all sides
- **Tile encoding** — PNG, JPEG (configurable quality), or raw pixel output
- **Blank tile optimization** — configurable `BlankTileStrategy` to either emit full tiles or write 1-byte placeholders (`BLANK_TILE_MARKER`) for uniform-color regions, reducing disk usage for sparse images
- **Edge tile background** — configurable background color (`background_rgb`) for padding partial tiles at image edges (defaults to white)
- **Geo-referencing** — affine transform mapping pixel coordinates to geographic coordinates, GCP support
- **Observability** — progress events, per-level callbacks, peak memory tracking

## Usage

```rust
use libviprs::{
    extract_page_image, BlankTileStrategy, EngineBuilder, EngineKind,
    FsSink, Layout, PyramidPlanner, TileFormat,
};
use std::path::Path;

// Extract raster from a scanned blueprint PDF
let raster = extract_page_image(Path::new("blueprint.pdf"), 1).unwrap();

// Plan the pyramid
let planner = PyramidPlanner::new(
    raster.width(), raster.height(),
    256,  // tile size
    0,    // overlap
    Layout::DeepZoom,
).unwrap();
let plan = planner.plan();

// Generate tiles to disk
let sink = FsSink::new("output_tiles", plan.clone()).with_format(TileFormat::Png);
let result = EngineBuilder::new(&raster, plan, sink)
    .with_engine(EngineKind::Auto)
    .with_concurrency(4)
    .with_blank_strategy(BlankTileStrategy::Placeholder)
    .run()
    .unwrap();

println!(
    "{} tiles across {} levels ({} blank tiles skipped)",
    result.tiles_produced, result.levels_processed, result.tiles_skipped,
);
```

## Modules

| Module | Description |
|---|---|
| `source` | Image decoding (JPEG, PNG, TIFF) into canonical `Raster` |
| `pdf` | PDF parsing (lopdf) and optional rendering (PDFium), including budgeted render |
| `raster` | Pixel buffer, region views, format normalization |
| `pixel` | Pixel format definitions (Gray8, RGB8, RGBA8, 16-bit variants) |
| `planner` | Tile math, level computation, layout generation |
| `resize` | Downscaling for pyramid levels |
| `engine` | Monolithic in-memory tile extraction with backpressure, blank tile detection |
| `engine_builder` | Typed `EngineBuilder` / `EngineKind` entry point routing to Monolithic, Streaming, or MapReduce engines |
| `streaming` | Sequential strip engine, `StripSource` trait, `RasterStripSource`, memory-budget helpers |
| `streaming_mapreduce` | Parallel strip engine and `MapReduceConfig` |
| `sink` | Tile output (filesystem, memory, slow sink for testing) |
| `sink_packfile` | `PackfileSink` writing tiles into a tar/zip archive (gated by `packfile`) |
| `sink_object_store` | `ObjectStoreSink` for user-injected object storage backends (gated by `s3`) |
| `resume` | Job checkpoints and resume policy for restart-safe runs |
| `retry` | Failure / retry policy and `RetryingSink` wrapper |
| `dedupe` | Content-addressed tile deduplication |
| `manifest` | `Manifest` v1 schema and `ManifestBuilder` describing the produced pyramid |
| `checksum` | Tile checksum modes and verification reports |
| `stream_verify` | Verify pyramid output against the original source |
| `geo` | Affine geo-transform, GCP solving, bounding box computation |
| `observe` | Progress events, lifecycle observers, memory tracking |

## Features

| Feature | Default | Description |
|---|---|---|
| `pdfium` | off | Enables `render_page_pdfium()`, `render_page_pdfium_budgeted()`, and `PdfiumStripSource` for vector PDF rendering. Requires libpdfium at runtime. |
| `pdfium-static` | off | Implies `pdfium` and links libpdfium statically via `pdfium-render/static`. |
| `s3` | off | Enables the `sink_object_store` module (`ObjectStoreSink` against a user-injected `ObjectStore`). |
| `tracing` | off | Emits structured `tracing` spans and events from the engine pipeline. |
| `packfile` | off | Enables `PackfileSink` for writing tiles into a tar or zip archive. |

## Requirements

- Rust 1.85+ (edition 2024)
- libpdfium shared library (only if using the `pdfium` feature)

### PDFium setup

The `pdfium` feature requires `libpdfium.so` at runtime. Pre-compiled binaries built from source are available from [libviprs-dep](https://github.com/libviprs/libviprs-dep/releases):

```bash
# x86_64
curl -L -o pdfium.tgz \
  https://github.com/libviprs/libviprs-dep/releases/download/pdfium-7725/pdfium-linux-x64.tgz

# arm64
curl -L -o pdfium.tgz \
  https://github.com/libviprs/libviprs-dep/releases/download/pdfium-7725/pdfium-linux-arm64.tgz

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

## CI

GitHub Actions runs two workflows:

**CI** (every push and PR) — `.github/workflows/ci.yml`:
- `cargo fmt --check` — formatting
- `cargo clippy -D warnings` — lint (default + `pdfium` feature)
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
make clippy   # clippy (default + pdfium features)
make test     # unit tests
make miri     # miri (requires nightly + miri component)
make loom     # loom concurrency tests
```

> **Prerequisites:** `make miri` requires the nightly toolchain with the miri component.
> Install with: `rustup toolchain install nightly --component miri`

