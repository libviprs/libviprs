# Tiles PVec stores (RL1.1, issue #1330)

This file is generated from `TABLE` in `src/raster_pvec/tiles.rs`. Do not edit it by hand: a test renders the table and fails if the committed file differs by a byte. To change it, change the table and run `PVEC_RASTER_TILES_BLESS=1 cargo test --test pvec_1330_raster_tiles`.

A row is a tile libviprs writes for a PVec raster, and the tags PVec stores for it (`tile_codec` and `pixel_format` of `docs/RASTER.md` section 1). The bit depth is 1 or 8. A 1-bit tile is an encoding of `Gray8` (`Raster::encode_png_bilevel`), not a pixel format.

| tile format | source pixels | bit depth | `tile_codec` | `pixel_format` |
|---|---|---|---|---|
| PNG | `Gray8` | 1 | 1 (PNG) | 0 (gray) |
| PNG | `Gray8` | 8 | 1 (PNG) | 0 (gray) |
| PNG | `Rgb8` | 8 | 1 (PNG) | 2 (rgb) |
| PNG | `Rgba8` | 8 | 1 (PNG) | 3 (rgba) |
| JPEG | `Gray8` | 8 | 2 (JPEG) | 0 (gray) |
| JPEG | `Rgb8` | 8 | 2 (JPEG) | 2 (rgb) |
| WebP | `Gray8` | 8 | 3 (WebP) | 0 (gray) |
| WebP | `Rgb8` | 8 | 3 (WebP) | 2 (rgb) |
| WebP | `Rgba8` | 8 | 3 (WebP) | 3 (rgba) |

Everything else is a typed `RasterTileError`:

| refusal | when |
|---|---|
| `RawTiles` | `TileFormat::Raw`: no dimensions, no pixel format (RASTER-04) |
| `JpegAlpha` | JPEG from `Rgba8`; flatten first or use PNG or WebP (RASTER-04) |
| `SixteenBit` | `Gray16`, `Rgb16`, `Rgba16`: the producer converts to 8 bits first and reports `converted-bit-depth` |
| `PixelFormat` | float and multiband sources |
| `BitDepth` | a depth the pair has no row for; 1 bit is gray PNG only |
| `TileEdge` | an edge other than 256, 512 or 1024 (RASTER-01) |
| `TooLong` | a stored tile above `2 * 4 * edge^2 + 65,536` bytes: 589,824 at 256, 2,162,688 at 512, 8,454,144 at 1024 (LIMIT-15) |

Interlaced PNG and palette PNG are never produced on this path, though `Raster::encode_png_interlaced` and `Raster::encode_png_palette` exist: PVec's sniffer refuses both.
