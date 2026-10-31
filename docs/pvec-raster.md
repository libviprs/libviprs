# Rasters in a PVec, from libviprs's side

I put a scanned underlay, an aerial or a PDF's embedded images into the same `.pvec` as the linework. The pixels go in exactly as libviprs encoded them, and a 2x3 affine places them in the drawing's source space. This page is the one place that lists what libviprs promises when it does that. The format and the reasons are in [ADR 0018](https://github.com/libviprs/pvec/blob/main/docs/adr/0018-raster-in-pvec.md) in `libviprs/pvec` (`docs/adr/0018-raster-in-pvec.md` there), and `docs/RASTER.md` in the same repository has the byte layout.

The rule underneath all of it: libviprs encodes, PVec stores. PVec never decodes a pixel, so every tile that reaches its writer has to be one its bounded header sniffer accepts, and nothing is resampled on the way. A 16-bit or palette source is converted before tiling and the report says so.

## The producer rules

This table is generated from `RASTER_PRODUCER_RULES` in `src/raster_pvec/rules.rs`. A test (`tests/pvec_1333_raster_rules_doc.rs`) fails if the page and the constant differ, and every row names the ADR item or the `RASTER-NN` code it comes from.

<!-- producer-rules:begin -->
| rule | what the producer does | comes from |
|---|---|---|
| `centring-off` | Centring is off: the tile grid is anchored at pixel (0, 0) with no offset. | ADR 0018 Producers; RASTER-01 |
| `overlap-0` | Tile overlap is 0. | ADR 0018 Producers; RASTER-01 |
| `google-tiles-past-the-grid-dropped` | The plan is the Google layout, whose grid is a power of two on a side, so the tiles beyond `cols` by `rows` of a level are padding and are dropped (counted in the report). | ADR 0018 Producers; RASTER-02 |
| `skip-blanks-not-stored` | `skip_blanks` leaves a tile out. PVec stores that as a zero-length index entry, a not-stored tile that draws nothing. | ADR 0018 Producers; RASTER-02 |
| `placeholder-refused` | `BlankTileStrategy::Placeholder` writes a marker that is not an image, so it is refused. | ADR 0018 Producers; RASTER-03 |
| `raw-refused` | `TileFormat::Raw` carries no dimensions and no pixel format, so it is refused. | ADR 0018 Producers; RASTER-04 |
| `bit-depth-16-converted` | A 16-bit source is converted to 8 bits before tiling (`v * 255 / 65535`, rounded) and the report says `converted-bit-depth`. | ADR 0018 pixel_format; RASTER-03 |
| `palette-converted` | A palette source is expanded to gray or RGB before tiling. A palette PNG is not a legal tile. | ADR 0018 pixel_format; RASTER-03 |
| `legal-codec-and-format-pairs` | Only the rows of `pvec-raster-tiles.md` are written: PNG in gray (1 or 8 bit), RGB and RGBA, JPEG in gray and RGB, lossless WebP in gray, RGB and RGBA. JPEG never carries alpha. | ADR 0018 pixel_format; RASTER-04 |
| `tile-edge` | The tile edge is 256, 512 or 1024. | ADR 0018 tile_edge; RASTER-01 |
| `limit-15` | A stored tile is at most `2 * 4 * edge^2 + 65,536` bytes, checked at encode time, before anything reaches PVec's writer. | LIMIT-15 (docs/SECURITY_LIMITS.md in libviprs/pvec) |
| `no-interlace` | PNG tiles are not interlaced (no Adam7), and the pyramid never writes a palette PNG. | ADR 0018 Producers; RASTER-03 |
| `raster-id-draw-order` | A `RasterId` is dense and assigned in source draw order (for a PDF, content-stream order). | ADR 0018 Producers (M16) |
| `negative-zero-normalised` | `-0.0` in a placement is normalised to `+0.0`. | ADR 0018 placement; RASTER-05; NP-03 |
| `attribution-plain-text` | Attribution is plain text of at most 255 bytes, with no control characters, line separators or bidirectional overrides. | ADR 0018 Licensing; RASTER-07 |
| `source-digest-sha-256` | `source_digest` is the SHA-256 of the samples as the producer received them (32 bytes). | ADR 0018 Licensing; RASTER-07 |
| `placement-np-19` | The placement passes NP-19: every coefficient finite, the four linear entries at most 2^20, the map invertible. libviprs checks it for a PDF placement before the write, and PVec's writer checks every placement. | ADR 0018 placement; RASTER-05; NP-19 |
| `geo-transform-permutation` | libviprs's `GeoTransform` `[a b c d e f]` becomes PVec's `Transform2` order `[a b d e c f]`. | ADR 0018 Georeference |
| `pixel-origin-corner` | The pixel origin is the top-left corner of pixel (0, 0). A world file or a `PixelIsPoint` transform is shifted by half a pixel first. | ADR 0018 Georeference |
| `composition-precondition` | The placement is composed only when the raster's CRS is the calibration's native CRS and the calibration is affine, with metres per unit folded in. Anything else is a typed `PlacementError`, and nothing is resampled. | ADR 0018 Georeference |
| `geo-transform-in-pmtiles-metadata` | The `GeoTransform` is stored in the PMTiles metadata under `vnd.libviprs.geo_transform` (option D'). No CRS and no geographic transform is written into a `.pvec`. | ADR 0018 Georeference (option D') |
| `clip-baked-into-alpha` | A clip that is not a pixel-aligned rectangle is baked into the alpha channel and `visible_box` is its bounding box. An aligned rectangle sets `visible_box` only. | ADR 0018 Producers (M16) |
| `clipped-image-lossless-tiles` | An image with a baked alpha channel is written as PNG or WebP, never JPEG. | ADR 0018 Producers; RASTER-04 |
| `pdf-colour-and-mask-bake` | PDF images are baked before tiling: `/SMask` into alpha, `/ImageMask` into RGBA in the fill colour, `/Indexed`, `/ICCBased`, `/CalGray`, `/CalRGB` and `/Lab` into sRGB (approximated, reported), `/Decode` applied, the unit square through the CTM. `JPXDecode`, `JBIG2Decode`, `CCITTFaxDecode`, `/Separation`, `/DeviceN`, a graphics-state soft mask and a singular CTM are refused by name. | ADR 0018 Producers (M17) |
| `band-from-paint-order` | An image painted before every painted path is band 0 (underlay), after every one band 1 (overlay), and between paths it is band 0 with `interleaved-draw-order` reported. | ADR 0018 band (M16) |
| `dwg-image-is-a-frame` | A DWG `IMAGE` is reported with its frame and no placement is written, because no reader here opens a path a drawing names. | ADR 0018 Producers |
| `nothing-dropped-silently` | The image constructs of a page equal the rasters written plus the refusals, and each refusal is a report line with its reason. | ADR 0018 Producers; EPIC R-L |

<!-- producer-rules:end -->

## Tiles

The legal `(tile format, pixel format, bit depth)` rows are rendered in [pvec-raster-tiles.md](pvec-raster-tiles.md), also from code. `Raster::encode_png_bilevel` writes the 1-bit gray PNG that bilevel scans want, as an encoding of `Gray8` and not a new pixel format. The cut pyramid, with the report of what was forced, converted and dropped, is described in [pvec-raster-producer.md](pvec-raster-producer.md).

## Georeference

The `GeoTransform` of a raster is stored once, out of band, in the PMTiles metadata (`vnd.libviprs.geo_transform`: six coefficients in `GeoTransform` order and the pixel origin by name, `corner` or `centre`). `Reader::geo_transform` hands it back bit for bit, and the header bounds stay the whole world because the transform describes the source raster, not the tile square. The module docs of `src/pmtiles/metadata.rs` have the wire shape.

To place the raster in a drawing, `raster_pvec::placement::pixel_to_source` composes

```text
pixel -> source = (1 / metres_per_unit) * inverse(metres -> native) o (pixel -> native)
```

and returns PVec's `Transform2` order `[a b c d tx ty]`. libviprs's `[a b c d e f]` goes to `[a b d e c f]`, the pixel origin is the top-left corner of pixel (0, 0), and an upright scan has a negative determinant, which is the normal case in a y-up source space. It only composes when the raster's CRS is the calibration's native CRS and the calibration is affine. The refusals are typed and none of them resamples:

| `PlacementError` | when |
|---|---|
| `CrsMismatch` | the raster is in another CRS than the calibration's native one |
| `ProjectiveCalibration` | the calibration is a homography, so `source -> native` is not affine |
| `SingularCalibration` | the calibration's linear part has no inverse |
| `SingularRaster` | the raster's own transform has no inverse |
| `InvalidUnitScale` | `metres_per_unit` is not finite and above zero |
| `NonFinite` | a coefficient of an input or of the result is not finite |

Until the georeference hand-off of #1048 can hand libviprs a calibration, the calibration is an explicit argument (`--source-to-geo` on the commands below).

## The PVec sink

`sink_pvec::PvecSink` (feature `pvec-raster`) is the `TileSink` the commands drive. It streams a pyramid into a PVec writer in any emission order and from any number of workers, with the same artifact bytes whichever, and `finish_to` stages the artifact under a root you give it and renames it into place. PVec's writer holds the encoded pyramid, so the sink is not flat in memory the way the PMTiles sink is; the measured peak is in [pmtiles-benchmarks.md](pmtiles-benchmarks.md).

## PDF images

`pvec::pdf_image::bake_page_images` turns the images of a PDF page into rasters. It decodes `DeviceGray`, `DeviceRGB` and `DeviceCMYK` at 1 to 16 bits, expands `/Indexed`, reads `/ICCBased` as the device space of its `/N`, converts `/CalGray`, `/CalRGB` and `/Lab` into sRGB, applies `/Decode`, paints `/ImageMask` in the fill colour, writes `/SMask` into alpha, and places the unit square through the CTM: `[A/w -C/h B/w -D/h C+E D+F]`. Clips become `visible_box` and alpha. A `DCTDecode` image is decoded and re-tiled, and the report says the tiles are a re-encode of a lossy source.

What it will not take is refused by name, with a report line and no placement: `JPXDecode`, `JBIG2Decode` and `CCITTFaxDecode`, `/Separation`, `/DeviceN` and `/Pattern`, an image under a graphics-state soft mask, and an image whose CTM is singular or not finite. The report counts written and refused, and the two always add up to the images on the page. The rules and their limits are written out in [pvec-raster-producer.md](pvec-raster-producer.md) and [adr/pvec-pdf-policy.md](adr/pvec-pdf-policy.md).

## DWG images

A DWG `IMAGE` names a file path, and no reader here opens a path a drawing names. It shows up in the conversion report as a frame (insertion point, U and V vectors, size, `show_image`, `definition_loaded`) and no placement is written. The same goes for an image with `SHOW_IMAGE` clear or no loaded definition.

## The commands

Three commands in `libviprs-cli` put this on the command line (libviprs/libviprs-cli#130):

| command | what it does |
|---|---|
| `viprs pyramid --storage pvec` | cuts a raster into a pyramid and writes it into a `.pvec` beside a drawing |
| `viprs pmtiles to-pvec` | passes a PMTiles archive into a `.pvec` byte for byte, with its stored `GeoTransform` |
| `viprs pvec rasters` | lists the rasters of a `.pvec` |

They are not released yet. This page will get their flags and one example each from the regenerated CLI reference when they land.
