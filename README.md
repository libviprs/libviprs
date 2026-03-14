# libviprs

A pure-Rust, thread-safe image pyramiding engine. Inspired by [libvips](https://www.libvips.org/), built from scratch for the AEC/construction domain.

Takes blueprint PDFs and images, extracts raster data, optionally geo-references it, and generates tile pyramids (DeepZoom, XYZ) suitable for web-based viewers.

## Features

- **PDF extraction** — extract embedded raster images from scanned blueprint PDFs via lopdf (zero runtime dependencies)
- **PDF rendering** — render vector PDFs (AutoCAD exports, text, paths) via PDFium (optional `pdfium` feature)
- **Image decoding** — JPEG, PNG, TIFF via the `image` crate
- **Tile pyramid generation** — multi-threaded engine with backpressure, configurable tile size and overlap
- **Layout formats** — DeepZoom (`.dzi` + directory tree) and XYZ (`z/x/y`)
- **Tile encoding** — PNG, JPEG (configurable quality), or raw pixel output
- **Geo-referencing** — affine transform mapping pixel coordinates to geographic coordinates, GCP support
- **Observability** — progress events, per-level callbacks, peak memory tracking

## Usage

```rust
use libviprs::{
    extract_page_image, generate_pyramid, EngineConfig, FsSink,
    Layout, PyramidPlanner, TileFormat,
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
let sink = FsSink::new("output_tiles", plan.clone(), TileFormat::Png);
let config = EngineConfig::default().with_concurrency(4);
let result = generate_pyramid(&raster, &plan, &sink, &config).unwrap();

println!("{} tiles across {} levels", result.tiles_produced, result.levels_processed);
```

## Modules

| Module | Description |
|---|---|
| `source` | Image decoding (JPEG, PNG, TIFF) into canonical `Raster` |
| `pdf` | PDF parsing (lopdf) and optional rendering (PDFium) |
| `raster` | Pixel buffer, region views, format normalization |
| `pixel` | Pixel format definitions (Gray8, RGB8, RGBA8, 16-bit variants) |
| `planner` | Tile math, level computation, layout generation |
| `resize` | Downscaling for pyramid levels |
| `engine` | Multi-threaded tile extraction with backpressure |
| `sink` | Tile output (filesystem, memory, slow sink for testing) |
| `geo` | Affine geo-transform, GCP solving, bounding box computation |
| `observe` | Progress events, memory tracking |

## Features

| Feature | Default | Description |
|---|---|---|
| `pdfium` | off | Enables `render_page_pdfium()` for vector PDF rendering. Requires libpdfium at runtime. |

## Requirements

- Rust 1.85+ (edition 2024)
- libpdfium shared library (only if using the `pdfium` feature)

## Related Crates

| Crate | Description |
|---|---|
| [libviprs-cli](../libviprs-cli) | Command-line interface (`viprs` binary) |
| [libviprs-tests](../libviprs-tests) | Integration tests, fixtures, and system checks |

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
