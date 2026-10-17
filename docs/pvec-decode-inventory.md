# Decode architecture inventory (Epic N, N1.1)

This is a map of what the crate's decode, traversal, output and memory machinery
is at commit `adcf1ff21699ded673af74297a4a83104b96abd1` (`adcf1ff2`), written so
that N1.2 through N1.8 design against the crate as it is and not as I remember
it. Every path, type and function named below resolves at that commit, with line
numbers taken from it. `git show adcf1ff2:<path>` opens any of them.

I wrote this as a reading of the tree, so nothing here is a design and nothing in
it describes a future API.

## The short version

Every decode path in the crate ends in a `Raster`. Nothing reads a DWG, and
nothing extracts vector geometry from a PDF. There is no `CadDecoder`, no
`PrimitiveSink` and no CAD provider anywhere in the tree at this commit. That
contract is Epic L's #1029, which was in flight when this was written, so Epic N
depends on it and does not assume it landed.

## What is there

| area | what is there at `adcf1ff2` |
|---|---|
| PDF | `src/pdf.rs` (2,976 lines). `extract_page_image` (line 687) and its variants `extract_page_image_dpi`, `extract_page_image_with_background`, `extract_page_image_with_background_typed` and `extract_page_image_with_password` read embedded images through `lopdf`. `render_page_pdfium` (1576) and `render_page_pdfium_budgeted` (1793) rasterise a page through pdfium and exist only behind the `pdfium` feature. `pdf_info` (282) and `pdf_info_with_password` (342) read page metadata through `lopdf::Document::load`. Every entry point that produces pixels returns a `Raster`. No content-stream path extraction exists. |
| raster sources | `src/source.rs`. `decode_file` (1176), `decode_file_with_limits` (2224), `decode_bytes` (2275) and `decode_bytes_with_limits` (2285), identification by leading magic bytes, `DecodeLimits` (792). A process-global LRU load cache is bounded by `DEFAULT_LOAD_CACHE_MAX_ENTRIES = 100` (line 17) and `DEFAULT_LOAD_CACHE_MAX_BYTES = 100 * 1024 * 1024` (line 23). |
| SVG | `src/svg.rs` (`svg` feature, off by default) rasterises an SVG through `resvg` into an RGBA `Raster`. It is vector in and raster out, so it is not a geometry path either. |
| traversal | `src/streaming.rs`. `StripSource` (152) with `render_strip(y_offset, height)`, `permits_concurrent_strips` and `RasterStripSource` as the in-memory implementation. `compute_strip_height` (814) divides a memory budget by a per-unit cost estimate and returns `None` when one unit does not fit. `EngineKind` lives in `src/engine_builder.rs` (113): `Auto`, `Monolithic`, `Streaming`, `MapReduce`, `MapReduceHotCache`, marked `#[non_exhaustive]`. |
| output | `src/sink.rs`. `TileSink` (227) with `write_tile(&self, &Tile)` and a defaulted `finish`, `MemorySink` (525), `FsSink` (771). `src/sink_pmtiles.rs` has `PmTilesSink` (156) over the ten-file format layer in `src/pmtiles/`: `directory.rs`, `header.rs`, `metadata.rs`, `mod.rs`, `range.rs`, `reader.rs`, `tileid.rs`, `validate.rs`, `varint.rs`, `writer.rs`. |
| cancellation | `src/cancel.rs`. `CancelToken` (50) is a cloneable handle around an `Arc<AtomicBool>`. The monolithic engine polls it per level and per tile, the streaming engine per strip, the map-reduce engine per batch, and the retry backoff in slices. It surfaces as `EngineError::Cancelled`. |
| georeference | `src/geo.rs`. `PixelCoord` (30), `GeoCoord` (55) and `GeoTransform` (82), a row-major GDAL-style affine of six `f64`. See the note below on what it does and does not do. |
| layout | One crate. Sources sit flat under `src/*.rs` with the `pmtiles/` subdirectory as the only module directory. `[workspace] members = ["fuzz"]` in `Cargo.toml`. There is no `crates/` directory. |

### A correction to the obvious reading of `geo.rs`

`GeoTransform` is not pixel-to-geo only. It has `pixel_to_geo` (250), `geo_to_pixel`
(273, returns `Option`) and `inverse` (293, returns `Option`), plus constructors
from an origin and scale and from three ground control points
(`from_gcps_exact`, 186). What it lacks is anything for composing transforms: there
is no `compose`, no multiply and no notion of a source-space transform. It is also
a pixel/geo affine and not a general 2D geometry type, and it is the only matrix
the crate has for coordinates.

It also carries a local numeric threshold already. `DET_REL_EPS = 1e-12` (line 102)
is the relative conditioning cut-off for the singularity test. I am recording
that as an existing fact about this file. Whatever policy N1.2 and N1.5 settle on
for geometry numerics, this constant is a raster georeferencing detail and not a
precedent for it.

## No CadDecoder, no PrimitiveSink, no DWG reader

This is the check, run against the tree at `adcf1ff2`:

```
git grep -n -i -E "CadDecoder|PrimitiveSink|acadsharp" adcf1ff2 -- src tests Cargo.toml
```

It prints nothing. Searching `src/` for `dwg` as a whole word, case-insensitive,
and for `cad` as a whole word, also finds nothing.

So at this commit:

- there is no decoder trait, and no primitive type a decoder could emit;
- there is no provider dependency in `Cargo.toml`, direct or optional;
- `lopdf = "0.36"` and the optional `pdfium-render` are the only PDF-facing
  dependencies.

### Where the contract is now

After `adcf1ff2`, #1029 (L1.1) landed on `main`. At `f57920ac` there is a
`src/cad/` module (`mod.rs`, `decoder.rs`, `primitive.rs`, `report.rs`) holding
`CadSource`, `CadDecoder`, `CadDrawing`, `CadView`, `PrimitiveSink`, a primitive
IR and a `DecodeReport`. I have not inventoried it here, because this document
is pinned to `adcf1ff2` on purpose and the later notes in N1.2 to N1.7 read that
module directly. Anyone reusing this document should treat the table above as
true at `adcf1ff2` and re-check the CAD rows against `main`.

## PDF is raster-only, so vector normalization is new capability

`src/pdf.rs` at `adcf1ff2` produces rasters and page metadata, and that is all.

- The `extract_page_image*` family walks a page's resources through `lopdf`,
  finds an image XObject and decodes it. It does not read a content stream's
  operators.
- `render_page_pdfium*` hands the page to pdfium and gets pixels back.
- `pdf_info` reads the page count, sizes and encryption state.

I searched `src/pdf.rs` for content-stream handling (`content stream`,
`decode_content`, `get_page_content`, `Operation`) and for the operator tokens
themselves, and found none. Reading `m`, `l`, `c`, `re`, `h`, `cm`, `q` and
`Q` out of a content stream is therefore new capability. It is not a refactor of
something that exists, and N1.6 and phase 5 should cost it that way.

## Memory: the in-tree precedent

The PMTiles writer in `src/pmtiles/writer.rs` is the one place the crate has
already solved "accept things in any order, hold a bounded amount, produce a
sorted artifact". It is the precedent N1.7 measures against.

- **Append-only index log.** The per-tile index is spilled to an append-only log
  on disk (`Spill`, line 512: `tile_id`, a dense `payload` index, `length`).
  `Writer::add_tile` accepts tiles in any order.
- **External sort at finalize.** The log is sorted in runs and merged when the
  archive is finished. The merge reads through a single handle and caps its
  fan-in with `MAX_MERGE_FANIN`, so a long job costs merge passes and not file
  descriptors.
- **A 24 MB sort buffer.** `SORT_RUN_RECORDS = 1 << 20` (line 206). A record is
  20 bytes on disk and 24 in memory, so the buffer is 1,048,576 records, 20 MB
  written and 24 MB resident. The writer's own comment makes the point that the
  buffer is the larger of the two figures.
- **Staged publish.** The archive is written to a staged temp file, `fsync`ed and
  renamed into place, so a partial archive never wears the final name.

The writer's module documentation records peak RSS for one process per row,
every payload distinct:

| distinct payloads | peak RSS |
|---|---|
| 500,000 | 79.0 MB |
| 1,000,000 | 155.4 MB |
| 2,000,000 | 288.5 MB |
| 4,000,000 | 563.2 MB |
| 10,000,000 | 1,083.8 MB |

I want to be careful about what that table supports. Peak memory does **not** grow
with the number of tiles, because the index lives on disk. It **does** grow
linearly with the number of distinct payloads, mostly through the content-hash
table that stores each payload once. The 24 MB sort buffer is about two percent of
the ten-million figure, and the writer's documentation says outright that it is
not the memory ceiling. So the precedent is "the index is spilled and sorted
externally", not "the whole writer is bounded independent of input size".
Those figures are quoted from the source comment at `adcf1ff2`; I did not re-run
them for this document.

The other memory machinery is the strip budget (`compute_strip_height`, above), the
decode-time `DecodeLimits`, and the load cache with its two bounds.

## Gaps: concepts the crate cannot represent

These are listed as gaps. None of them is a design.

- Blocks and inserts: no instancing, no nesting, no per-instance transform.
- Layers: no layer identity, state or override.
- Model space versus paper space, and viewports.
- Source transforms: no matrix composition and no source-space coordinate type.
- Entity handles or any stable identity for a source item.
- Curves of any kind: no arc, ellipse or spline value anywhere in the crate.
- Units and drawing extents as data.

`src/draw.rs` is an in-place raster drawing seam (`DrawOp` applied to a
`Raster`), which is output-side painting and not a geometry model.

## Checks run

All of the following were read or run against a checkout of `adcf1ff2`:

- every file path, type name and line number in the tables above, by `git grep`
  and by opening the lines;
- the line count of `src/pdf.rs` (2,976);
- the absence grep in the previous section;
- the ten files under `src/pmtiles/` by `git ls-tree`;
- the absence of a `crates/` directory.

No benchmark was run for this document and no source file was changed.
