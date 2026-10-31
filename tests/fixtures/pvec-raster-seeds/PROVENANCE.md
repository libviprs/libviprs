# PVec raster tile seeds

One 256 px tile of each variant libviprs writes for a PVec raster stream, for the fuzz targets of
libviprs/pvec#219 (the three tile sniffers). Issue libviprs/libviprs#1330.

The generator is the test `tests/pvec_1330_seed_corpus.rs`. It builds one drawing-like tile (a grid,
a diagonal and a flat patch, so the same bytes come out on every host) and encodes it with libviprs's
own writers. The test fails if a file here differs from what the generator makes today, and if
`SHA256SUMS` does not match the files. To regenerate after a deliberate writer change:

```
PVEC_SEEDS_BLESS=1 cargo test --test pvec_1330_seed_corpus
```

| file | writer | what the header says |
|---|---|---|
| `png-gray1.png` | `Raster::encode_png_bilevel(128)` | gray, 1 bit |
| `png-gray8.png` | `Raster::encode_png(6)` | gray, 8 bit |
| `png-rgb8.png` | `Raster::encode_png(6)` | RGB, 8 bit |
| `png-rgba8.png` | `Raster::encode_png(6)` | RGBA, 8 bit |
| `jpeg-gray8-baseline.jpg` | `Raster::encode_jpeg(85)` | SOF0, 1 component |
| `jpeg-rgb8-baseline.jpg` | `Raster::encode_jpeg(85)` | SOF0, 3 components |
| `webp-vp8l-rgb8.webp` | `Raster::encode_webp` | `VP8L` |
| `webp-vp8l-rgba8.webp` | `Raster::encode_webp` | `VP8L` |

Not here, because libviprs does not write them: progressive JPEG (the hand-rolled writer has no
progressive mode), `VP8` and `VP8X` WebP, interlaced and palette PNG, 16-bit and 2 or 4 bit PNG.
