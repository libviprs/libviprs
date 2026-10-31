# Rasters towards a PVec (issue #1328)

ADR 0018 in libviprs/pvec puts raster graphics inside a PVec artifact: encoded tiles stored untouched, plus a placement that maps pixels into the drawing's source space. This document is the libviprs side of that. It records what is here, and says plainly what is not.

## The stored GeoTransform (option D')

libviprs cuts a raster into a pyramid and packs it into PMTiles, and it used to forget where the raster was. `viprs pyramid --geo-origin/--geo-scale` built a `GeoTransform`, printed its bounds, and threw it away. Now `PmTilesSinkBuilder::geo_transform` writes it into `vnd.libviprs.geo_transform` (see the module docs of `src/pmtiles/metadata.rs` for the wire shape): six coefficients in `GeoTransform` order and the pixel origin convention by name, `corner` or `centre`. `Reader::geo_transform` hands it back bit for bit. The header bounds are untouched (whole world), because the transform describes the source raster, not the Web Mercator square the tiles are laid on.

## The composed placement

`pvec::raster_placement::pixel_to_source` composes

```text
pixel -> source = (1 / metres_per_unit) * inverse(metres -> native) o (pixel -> native)
```

and returns PVec's `Transform2` order `[a b c d tx ty]` (rows `[a b tx]` and `[c d ty]`), so libviprs's `[a b c d e f]` goes to `[a b d e c f]`. The pixel origin is the top-left corner of pixel (0, 0); a `centre` transform is shifted by half a pixel first. `-0.0` is normalised to `+0.0`. Source space is y-up and rows run down, so an upright scan has a negative determinant: that is the normal case.

It is affine only when the raster's CRS is the calibration's native CRS and the calibration is affine. A raster in another CRS (`CrsMismatch`), a projective calibration (`ProjectiveCalibration`), a singular calibration or raster, a bad unit scale and a non-finite coefficient are each a typed `PlacementError`. Nothing resamples. The calibration is an explicit argument until the georeference handoff of #1048 produces one.

## Images in the conversion report

`PdfVectorPage::images` lists every painted image in content-stream order, forms expanded, with the CTM that maps the image's unit square into page space and, for an image inside a form, the form path. `FillReport` counts them beside the fill constructs (`PdfImageXObject`, `PdfInlineImage`). A DWG `IMAGE` arrives as `RawGeometry::Unmapped` with its frame in the reason (insertion point, U and V vectors, size, `show_image`, `definition_loaded`) and is counted as `DwgImage`; an image with `SHOW_IMAGE` clear or no loaded definition is reported the same way. No placement is written for any of them: the PDF bake rules (`/SMask`, `/ImageMask`, `/Indexed`, ICC, Lab and CalRGB, `/Decode`, the CTM and the page flip) are the first follow-up, and no reader here fetches the pixels a drawing names.

One limit I did not change: `lopdf` only parses inline images with an uncompressed body and a spelled-out device colour space (`/Gray`, `/RGB`, `/CMYK`), so a page with a filtered inline image still fails extraction as it did before.

## The pyramid producer (`pvec::raster_pyramid`)

`produce_raster_pyramid` cuts a raster into the tiles a PVec raster stream stores and hands back the encoded bytes with their `(level, col, row)` and the placement fields. It needs nothing from PVec, so every decision is tested without a container; the thin adapter that gives the result to PVec's writer is the part that waits on libviprs/pvec#218 and #219.

- The plan is `Layout::Google` with centring off and overlap 0, whatever the requested plan said. The forced settings are listed in the report. Google tiles are full squares (an edge tile is padded with the background), which is what RASTER-01 and the tile sniffer expect.
- `tile_size` is 256, 512 or 1024. Anything else is `TileSize`.
- The Google grid is a power of two on a side, so the tiles past `ceil(width_L / tile_size)` by `ceil(height_L / tile_size)` of a level are padding and are dropped (counted in the report).
- `skip_blanks` leaves a tile out, which is a not-stored tile (a zero-length index entry in PVec). `BlankTileStrategy::Placeholder` writes a one-byte marker that is not an image, so it is refused (`PlaceholderBlanks`), and so is `TileFormat::Raw`.
- Codec comes from the tile format (PNG, JPEG, lossless WebP). Samples are gray, RGB or RGBA at eight bits. A 16-bit source is converted by rounded scaling (`v * 255 / 65535`, not the clipping cast) and the report says `Gray16 -> Gray8`. A JPEG tile has no alpha, so an RGBA source is flattened onto the background and the tile says RGB; the report names that too. libviprs has no 1-bit tile format, so a bilevel scan is written as eight-bit gray for now.
- The placement is `[a b c d tx ty]`, given outright or composed from a stored `GeoTransform` and a calibration (`raster_placement`), with `-0.0` normalised. `visible_box` is the whole image, `source_digest` the SHA-256 of the samples as received.
- A lossless tile decodes to exactly the pixels the producer cut, which is the producer-side claim of ADR 0018; `tests/pvec_1328_raster_pyramid.rs` checks it pixel by pixel against the source.
