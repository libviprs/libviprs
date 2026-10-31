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
