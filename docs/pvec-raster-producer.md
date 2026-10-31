# Rasters towards a PVec (issue #1328)

ADR 0018 in libviprs/pvec puts raster graphics inside a PVec artifact: encoded tiles stored untouched, plus a placement that maps pixels into the drawing's source space. This document is the libviprs side of that. It records what is here, and says plainly what is not.

## The stored GeoTransform (option D')

libviprs cuts a raster into a pyramid and packs it into PMTiles, and it used to forget where the raster was. `viprs pyramid --geo-origin/--geo-scale` built a `GeoTransform`, printed its bounds, and threw it away. Now `PmTilesSinkBuilder::geo_transform` writes it into `vnd.libviprs.geo_transform` (see the module docs of `src/pmtiles/metadata.rs` for the wire shape): six coefficients in `GeoTransform` order and the pixel origin convention by name, `corner` or `centre`. `Reader::geo_transform` hands it back bit for bit. The header bounds are untouched (whole world), because the transform describes the source raster, not the Web Mercator square the tiles are laid on.

## The composed placement

`raster_pvec::placement::pixel_to_source` composes

```text
pixel -> source = (1 / metres_per_unit) * inverse(metres -> native) o (pixel -> native)
```

and returns PVec's `Transform2` order `[a b c d tx ty]` (rows `[a b tx]` and `[c d ty]`), so libviprs's `[a b c d e f]` goes to `[a b d e c f]`. The pixel origin is the top-left corner of pixel (0, 0); a `centre` transform is shifted by half a pixel first. `-0.0` is normalised to `+0.0`. Source space is y-up and rows run down, so an upright scan has a negative determinant: that is the normal case.

It is affine only when the raster's CRS is the calibration's native CRS and the calibration is affine. A raster in another CRS (`CrsMismatch`), a projective calibration (`ProjectiveCalibration`), a singular calibration or raster, a bad unit scale and a non-finite coefficient are each a typed `PlacementError`. Nothing resamples. The calibration is an explicit argument until the georeference handoff of #1048 produces one.

## Images in the conversion report

`PdfVectorPage::images` lists every painted image in content-stream order, forms expanded, with the CTM that maps the image's unit square into page space, the form path, the clips and soft-mask flag in force and the entries the bake reads. `FillReport` counts them beside the fill constructs (`PdfImageXObject`, `PdfInlineImage`). A DWG `IMAGE` arrives as `RawGeometry::Unmapped` with its frame in the reason (insertion point, U and V vectors, size, `show_image`, `definition_loaded`) and is counted as `DwgImage`; an image with `SHOW_IMAGE` clear or no loaded definition is reported the same way, and no placement is written for it because no reader here fetches the pixels a drawing names. A PDF image gets a placement from `pvec::pdf_image` (next section).

One limit I did not change: `lopdf` only parses inline images with an uncompressed body and a spelled-out device colour space (`/Gray`, `/RGB`, `/CMYK`), so a page with a filtered inline image still fails extraction as it did before.

## PDF image placements (issue #1332)

`pvec::pdf_image::bake_page_images` takes the images `PdfVectorPage::images` recorded and, for each one, decodes the stream, applies the bake rules of ADR 0018 and cuts the pyramid with `produce_raster_pyramid`. It returns the written rasters in content order (a `RasterId` is a position in that list), the refused ones with a typed error each, and a report where the images on the page equal the written plus the refused.

- Colour: `DeviceGray`, `DeviceRGB` and `DeviceCMYK` at 1, 2, 4, 8 and 16 bits, `/Indexed` through its lookup, `/ICCBased` as the device space of its `/N`, `/CalGray`, `/CalRGB` and `/Lab` into sRGB by the standard formulas. 16 bits are reduced to 8 and reported (`bit-depth-reduced`); ICC, Cal, Lab and CMYK are reported `colour-space-approximated`. `/Decode` is applied to every component.
- `/ImageMask true` is painted in the fill colour in force at the `Do` or `BI` (tracked from `g`, `rg`, `k` and `sc`/`scn` with one, three or four numbers), into RGBA, with `/Decode [1 0]` honoured. The unpainted pixels keep the fill colour and get alpha 0.
- `/SMask` is resampled to the image size (nearest pixel centre) and written into alpha. An explicit `/Mask` is not applied and the note `explicit-mask-ignored` says so.
- Bilevel: a `DeviceGray` (or `/Indexed` over gray) image at one bit per component, with no alpha, is cut into 1-bit PNG tiles and the report counts it (`bilevel-1-bit-tiles`). With a soft mask or a clip the image has alpha and stays RGBA at eight bits, and under JPEG tiles it stays eight-bit gray. The levels below full resolution are averaged and thresholded again at 128, so they are a rendering of the bilevel image.
- Streams: no filter, `FlateDecode` with the TIFF and PNG predictors, `DCTDecode` (decoded and re-tiled, `re-encoded-lossy`) and Flate over DCT. `JPXDecode`, `JBIG2Decode`, `CCITTFaxDecode` and every other filter are refused by name (`unsupported-filter:<name>`). The issue allows a `pdfium` decode route for those three and I did not wire it, so a scan stored as CCITT, JBIG2 or JPEG 2000 is a typed refusal (`PdfImageError::UnsupportedFilter`, counted under `unsupported-filter:<name>`, tested for each name) and the page's linework still converts. I counted the filter names in the 35 CAD-export PDFs of the pdf-candidates corpus (a byte count of the file, so a filter named inside a compressed object stream is missed and these are lower bounds): `JPXDecode` in 2 files (27 uses), `CCITTFaxDecode` in 1 (7), `JBIG2Decode` in none, against `DCTDecode` in 12 files (90) that the bake does decode. I would not require `pdfium` for image placements on those numbers: a refused image is a report line and the linework still converts. Whether to wire the route anyway is Rom's call (epic #1329, open question 5).
- Refused by name too: `/Separation`, `/DeviceN` and `/Pattern`, an image under a graphics-state soft mask (`gs-soft-mask`), a singular or non-finite CTM (`singular-ctm`), a fully clipped image, and more pixels than `PdfImageOptions::max_pixels`.
- Placement: the unit square through the CTM, `[A/w -C/h B/w -D/h C+E D+F]` in `Transform2` order, composed with `PdfImageOptions::page_to_source` (the identity by default, because page space is already y-up with the displayed page's lower-left corner as the origin), `-0.0` normalised and checked against NP-19 (finite, linear entries at most 2^20, invertible).
- Clips: every clip in force is intersected. A rectangle that is axis-aligned in pixel space and lands on pixel edges sets `visible_box` and leaves the pixels alone. Any other clip is rasterised at image resolution into alpha (pixel centres, the clip's own nonzero or even-odd rule, curves cut into 16 chords) and `visible_box` is the clips' bounding box inside the image. There is no anti-aliasing at the clip edge. A baked alpha needs PNG or WebP tiles, so JPEG tiles are refused for it (`raster-04-alpha-needs-lossless`).
- Band: an image painted before every painted path is the underlay, after every one the overlay, and between paths it is the underlay with `interleaved-draw-order` in the report. That is the v1 limit the ADR records.
- Limits: `PdfVectorLimits::max_images` and `max_image_bytes` bound what the page keeps for the bake.

One limit that comes from `lopdf` and not from here: its content parser refuses an inline image that has no `/ColorSpace` (an inline image mask) as well as a filtered one, and that fails the page extraction as it did before this task.

## The pyramid producer (`raster_pvec::pyramid`)

`produce_raster_pyramid` cuts a raster into the tiles a PVec raster stream stores and hands back the encoded bytes with their `(level, col, row)` and the placement fields. It needs nothing from PVec, so every decision is tested without a container; the thin adapter that gives the result to PVec's writer is the part that waits on libviprs/pvec#218 and #219.

- The plan is `Layout::Google` with centring off and overlap 0, whatever the requested plan said. The forced settings are listed in the report. Google tiles are full squares (an edge tile is padded with the background), which is what RASTER-01 and the tile sniffer expect.
- `tile_size` is 256, 512 or 1024. Anything else is `TileSize`.
- The Google grid is a power of two on a side, so the tiles past `ceil(width_L / tile_size)` by `ceil(height_L / tile_size)` of a level are padding and are dropped (counted in the report).
- `skip_blanks` leaves a tile out, which is a not-stored tile (a zero-length index entry in PVec). `BlankTileStrategy::Placeholder` writes a one-byte marker that is not an image, so it is refused (`PlaceholderBlanks`), and so is `TileFormat::Raw`.
- Codec comes from the tile format (PNG, JPEG, lossless WebP). Samples are gray, RGB or RGBA at eight bits. A 16-bit source is converted by rounded scaling (`v * 255 / 65535`, not the clipping cast) and the report says `Gray16 -> Gray8`. A JPEG tile has no alpha, so an RGBA source is flattened onto the background and the tile says RGB; the report names that too. `RasterPyramidOptions::bilevel_threshold` cuts gray PNG tiles at one bit per pixel through `Raster::encode_png_bilevel` (the pyramid says `bit_depth` 1, and anything but a `Gray8` source with PNG tiles is refused as a tile PVec has no row for); without it a gray scan is eight-bit gray.
- The placement is `[a b c d tx ty]`, given outright or composed from a stored `GeoTransform` and a calibration (`raster_pvec::placement`), with `-0.0` normalised. `visible_box` is the whole image, `source_digest` the SHA-256 of the samples as received.
- A lossless tile decodes to exactly the pixels the producer cut, which is the producer-side claim of ADR 0018; `tests/pvec_1328_raster_pyramid.rs` checks it pixel by pixel against the source.
