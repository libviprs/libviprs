# Changelog

All notable changes to libviprs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- `Raster::arrayjoin` no longer clamps `across` to the number of images
  (issue #577). A value larger than the list used to collapse into one full
  row; it now lays out that many cells wide and leaves the trailing ones
  background, which is what vips does. Anyone passing an `across` above their
  image count gets different output geometry out of the same call, with no
  error to notice it by, so it is worth grepping for.

  Measured against vips 8.18.4 on two inputs whose sizes differ, a 3x2 and a
  2x3, so the grid cell is 3x3 (`vips black a.v 3 2; vips black b.v 2 3;
  vips arrayjoin "a.v b.v" o.v --across N`):

  | `across` | vips | libviprs before | libviprs now |
  |---|---|---|---|
  | 1 | 3x6 | 3x6 | 3x6 |
  | 2 | 6x3 | 6x3 | 6x3 |
  | 3 | 9x3 | 6x3 | 9x3 |
  | 5 | 15x3 | 6x3 | 15x3 |
  | 10 | 30x3 | 6x3 | 30x3 |

  `shim` follows `across` rather than the image count with it, since
  `arrayjoin.c:259-260` sizes the row as `hspacing * across + shim *
  (across - 1)`: the same pair with `--across 4 --shim 2` is 18x3, where the
  clamp gave 8x3.

  An explicit `across` outside `1..=1000000` is now the typed
  `ConversionError::AcrossOutOfRange` instead of being silently clamped.
  That is the range libvips declares on the property
  (`VIPS_ARG_INT(class, "across", 4, ..., 1, 1000000, 1)` in
  `arrayjoin.c:400-406`), and GObject refuses both ends before the operation
  is built, so `--across 0` and `--across 1000001` never produce a grid there
  either. The default is unchanged and is not range checked, because vips
  assigns `join->across = n` straight to the struct field and bypasses its own
  property check the same way.

- `decode_tiff_page` indexes pages from **zero**, where it used to index from
  one (issue #566). `decode_tiff_page(p, 0)` is now the first image and used to
  be an error; `decode_tiff_page(p, 1)` is now the *second* image and used to be
  the first. Every call site that passed a page number has to lose one, and the
  break is silent on a multi-page file, so it is worth grepping for rather than
  waiting to be told.

  The old numbering disagreed with libvips, whose `page` argument is `min: 0`
  on `tiffload`, `pdfload`, `gifload`, `heifload` and `webpload` alike
  (measured against 8.18.4: `vips tiffload --page 0` loads the first image, and
  `--page 1` on a single-page file fails with "TIFF does not contain page 1").
  Anyone moving a pipeline across read the wrong page with no error to show for
  it. It also disagreed with the `tiff` crate underneath, where
  `seek_to_image(0)` is the first IFD, so the function was converting between
  the two conventions for nobody's benefit. A TIFF has no page numbers of its
  own to justify the offset either: the IFD chain is a linked list.

  The frames/page model in #564 is the reason to move it now rather than later.
  That model exposes frames as a sequence, a sequence in Rust is indexed from
  zero, and a `frames()` accessor starting at 0 sitting next to a
  `decode_tiff_page` starting at 1 would be a permanent source of off-by-ones.

  **PDF page numbers are unchanged and remain 1-based.** `extract_page_image`
  and its siblings read a numbering the document carries itself, `PdfInfo`
  reports that numbering, and the CLI's `--page` exposes it to users on those
  terms. The rule across the crate is that a document's own page number is
  1-based and a position in a sequence of frames is 0-based.

  A raster from `decode_tiff_page` now also carries `n-pages`, so the count
  that bounds `page` comes back with the pixels and is readable through
  `Raster::get_n_pages`. vips attaches the same field on every TIFF load,
  single-page files included. The out-of-range error names both the index and
  the count instead of relaying the `tiff` crate's seek failure.

- `Raster::encode_webp` takes a `webp::SaveOptions` carrying a `Compression`
  and a `Keep`, where it used to take a bare `quality: u8` (issue #568). There
  is no lossy WebP encoder reachable in pure Rust: `image-webp` 0.2.4's
  `encoder.rs` writes a `VP8L` chunk and has no quality knob anywhere in it. A
  `quality` argument the encoder throws away inverts the contract — ask for
  quality 10, get a lossless file possibly larger than the PNG you started
  with — and it is a semver time bomb, because the day a lossy encoder lands
  every existing `encode_webp(10)` would silently start emitting small lossy
  files in a patch release. Making quality unrepresentable turns that into a
  compile error now instead. `Compression` is `#[non_exhaustive]`, so
  `Lossy { .. }` can join it as a minor bump.

  A 16-bit raster is also refused rather than narrowed, with a message naming
  the remedy. vips narrows the same input by a right shift of 8, silently
  (measured: 255 becomes 0, 65535 becomes 255). The reason not to copy that is
  internal consistency: `Raster::cast` to an 8-bit format *clips*, so an
  automatic narrow inside the encoder would disagree with the crate's own cast
  while looking like it did the same thing. Cast first and the narrowing is
  yours.

- `decode_svg` takes a `SvgOptions` instead of a bare `Option<f64>` DPI
  (issue #502). It used to be `decode_svg(data, Some(144.0))`, and it is now
  `decode_svg(data, SvgOptions { dpi: 144.0, ..Default::default() })`. The old
  shape had nowhere to put `scale` or `unlimited`, which are two of the three
  load options vips `svgload` actually takes, and growing it to a third
  positional argument would have broken every call site anyway. The function
  also moved from `crate::foreign_stubs` to the new `crate::svg`, but the
  crate-root spelling `libviprs::decode_svg` is unchanged.

- `colourspace` to `labs` now truncates the LabS code toward zero instead of
  rounding it, so the output bytes of every conversion into `labs` change
  (issue #556). `Lab [50, 0, 0]` came out as `16384` and is now `16383`, which
  is what vips 8.18.4 prints: `50 * 32767/100` is exactly `16383.5`, and
  `colour/Lab2LabS.c:66-68` clips in `double` and then assigns into a
  `signed short`, so C drops the fraction rather than rounding it. The `a` and
  `b` channels had the same defect at the `256` scale. Truncation here is
  toward zero and not floor, which is a distinction LabS is the only space in
  this module to make, being the only signed carrier: `a = -0.501953125`
  scales to `-128.5` and vips answers `-128`, not `-129`.

  Every route into `labs` is affected, not only `Lab -> Labs`, because
  `vips_Lab2LabS` is the last stage of all of them (`colour/colourspace.c:229`
  onward). Expect codes to move by one count, and by at most one.

- `Lab <-> Labs` takes the direct route libvips gives it
  (`{ LAB, LABS, { vips_Lab2LabS } }` at `colour/colourspace.c:246` and
  `{ LABS, LAB, { vips_LabS2Lab } }` at `:310`) instead of meeting at the XYZ
  hub (issue #556). This is not the optimisation it looks like. `Lab -> XYZ ->
  Lab` leaves a residue of a few parts in 1e6, because `lab_f` switches at
  `t < 0.008856` and `lab_to_xyz` at `L < 8.0` and those rounded decimals are
  not mutual inverses. Rounding used to absorb that residue; truncation cannot,
  so a code that should land on a whole number loses a whole count whenever the
  residue is negative. `Lab [0, -128, 1]` is `[0, -32768, 256]` in vips and was
  `[0, -32767, 255]` through the hub. Coming back the other way the hub drifts
  in the shadow branch instead: `Labs [983, 256, -256]` is
  `[2.999969482421875, 1, -1]` in vips and was `[2.99994, 1.00052, -1.00021]`.

  `Lch <-> Labs` and `Cmc <-> Labs` are hub-free in libvips too (`:280`,
  `:297`, `:312`, `:313`); they took the hub until issue #583, below.

- `Lch <-> Labs` and `Cmc <-> Labs` take the direct routes too, so the output
  bytes of conversions between those spaces change (issue #583). libvips joins
  them without an XYZ step (`{ LCH, LABS }` at `colour/colourspace.c:280`,
  `{ CMC, LABS }` at `:297`, `{ LABS, LCH }` at `:312`, `{ LABS, CMC }` at
  `:313`) and this port sent them round the hub, which costs the same whole
  count issue #556 found on `Lab <-> Labs`: the `Lab -> XYZ -> Lab` residue is
  a few parts in 1e6 and truncation turns that into a lost count whenever the
  exact code is a whole number.

  Coming back out of `labs` the damage was larger and more visible, because a
  neutral LabS code is *exactly* neutral. vips answers `C = 0, h = 0` for
  `LabS [0, 0, 0]`; the hub pushed `a` and `b` off zero and then read a hue off
  the noise, giving `C = 5.57e-4, h = 338.199` at every `L`. Both `Lch` and
  `Cmc` inherited that. On a 700-pixel swept grid the hub missed vips on 181 of
  the 2100 `Lch -> Labs` channels, 748 of the `Labs -> Lch` ones and 681 of the
  `Labs -> Cmc` ones; all three are now exact.

  The LabS quantiser also rounds its input to `f32` before scaling, which moves
  a further count on some inputs into `labs` from any space. `Lab2LabS.c:59`
  reads a `float` image and every libvips route into LabS hands it one, so the
  quantiser never sees double precision. Under the old rounding that was
  invisible; under truncation it decides counts. `LCh [0, 1, 30]` is the case
  that shows it: `sin(30 deg)` is `0.49999999999999994` in `f64` and exactly
  `0.5` as `f32`, so `b * 256` is either `127.99999999999999` or `128.0`, and
  vips answers `128`.

  `Cmc -> Labs` is exact on the neutral axis and wherever libvips' CMC inverse
  tables agree with this crate's bisection, but not everywhere: libvips inverts
  `Lcmc`, `Ccmc` and `hcmc` through tables sampled every 0.1
  (`colour/UCS2LCh.c:66-135`) and this crate bisects the forward function
  instead, which is the more accurate of the two. Where a LabS code sits a hair
  above a whole number the two land on opposite sides of it. That divergence
  predates this change and is unrelated to the routing: handed libvips' own
  `Cmc -> Lch` output, this crate reproduces libvips' `Cmc -> Labs` codes on
  all 2100 channels of the same grid.

### Added

- Still-image WebP load and lossless save (issues #567 and #568). `decode_webp`
  reads every WebP this build can meet — lossy `VP8`, lossless `VP8L`, alpha,
  and the extended `VP8X` container — and lifts the `ICCP`, `EXIF` and `XMP `
  chunks onto the raster as `icc-profile-data`, `exif-data` and `xmp-data`, the
  same names the JPEG loader uses, so `Raster::icc_profile` finds a WebP
  profile without knowing where it came from. `Raster::encode_webp` and
  `Raster::save_webp` write the lossless form, and `.webp` is now a live row in
  both shared dispatchers: `Raster::save` by extension and
  `Raster::encode_to_buffer` by format name. `save_stripped` drops the metadata
  chunks, which is `webpsave --keep none`.

  The lossy decode is bit-exact against libwebp rather than merely close, and
  the lossless round trip is the identity, so `decode_webp(encode_webp(x))` is
  `x` for every 8-bit raster. Both directions of the differential are pinned in
  `oracle-captures/foreign-webp/`, including vips 8.18.4 reading four files
  libviprs wrote.

  Two things worth knowing before you use it. A one-band raster comes back as
  three bands, because WebP stores no greyscale at all and `vips webpsave` does
  the same. And an animated WebP loads **frame 0 only**, at one frame's size,
  with `n-pages` set to how many frames the original had — which is what a
  default `vips webpload` does too. Reading every frame is issue #569 and waits
  on the page model.

- Still-image GIF load and save (issues #570, #571). `gifload` was routed
  through the `image` crate's facade and `encode_gif` was a typed stub; both
  now go straight to the `gif` crate, because the facade cannot reach what
  either half needs. `image::codecs::gif::GifDecoder` hard-codes `Rgba8`
  where vips emits three bands for a GIF with no transparent index anywhere,
  and its `GifEncoder` has no interlace, no dither, and no palette control at
  all.

  Load produces frame 0 at the logical screen size, which is exactly what
  `vips gifload` does by default (`page = 0`, `n = 1`), tagged sRGB and
  carrying `n-pages`, `loop`, `bits-per-sample`, `palette`, and `interlaced`.
  Seven of the eight GIFs in the libvips reference suite decode
  byte-identically to vips, over 3.25 MB of pixels; the eighth is
  `truncated.gif`, where libviprs recovers sixteen more rows out of the
  broken tail before it gives up and the first 784 rows still match exactly.

  Three details are worth knowing because they are easy to get wrong and all
  three were settled against the binary rather than the spec. The canvas
  around frame 0 is transparent black, not the background colour the header
  reports. `loop` is not the NETSCAPE repeat count: no application extension
  means 1, a stored count of 0 means 0 (forever), and a stored count of `n`
  means `n + 1`. And a frame whose pixel data runs off the end of the file
  counts as declaring transparency, because the rows that never arrived stay
  uncomposited, so a truncated GIF loads with four bands where the intact one
  has three.

  Save writes a single-frame GIF89a and takes `interlace`, `dither` and
  `bitdepth`. `Raster::save`, `encode_to_buffer` and `encode_to_target` all
  route `.gif` to it, so the extension dispatch works alongside the direct
  `encode_gif` / `save_gif` pair.

  **Output is not byte-identical to `vips gifsave`, and it never will be.**
  LZW is exactly lossless and deterministic both ways, so the bitstream is
  not where the two disagree; palette quantisation is. vips quantises with
  libimagequant and libviprs with the median-cut quantiser that already backs
  `encode_png_palette`, and two algorithms pick two different palettes for
  the same image. What is matched instead is structural, and all of it is
  checked: the colour table is `2^bitdepth` entries with the same LZW minimum
  code size vips writes, a transparent index is reserved at 0 under exactly
  the same condition (measured over twelve bitdepth and colour-count pairs,
  so an opaque source with palette headroom reloads as four bands here too),
  alpha is thresholded at 128 with the sub-threshold pixel zeroed outright,
  and interlaced rows go out in GIF's four-pass order. Where the palette
  already fits, the round trip is exact.

  The quantiser gap is bounded rather than waved at. On a 48x32 gradient of
  1536 distinct colours vips scores `avg_abs_diff 3.895` and
  `max_abs_diff 22` against its own input, and libviprs scores `3.457` and
  `12`; on the reference `synth_rgb8` fixture vips scores `3.366` and `23`
  and libviprs `3.944` and `18`. Neither dominates.

  Encoding is byte-reproducible. It was not, before: the shared quantiser
  gathered distinct colours through a `HashMap` whose iteration order the
  default `RandomState` reseeds per process, so identical input produced
  differently ordered palettes and different bytes on every run. That
  affected `encode_png_palette` too, and is fixed for both.

  Animated GIF is not included. `decode_gif` loads frame 0 and attaches
  `n-pages` so a caller can see the rest is there; multi-page load and save
  arrive with the page model. For the same reason the array-valued fields
  `delay`, `background` and `gif-palette` are read but not attached, since
  `MetadataValue` has no array variant yet. `gifsave`'s `effort`, `reuse`,
  `interpalette-maxerror`, `interframe-maxerror` and `keep-duplicate-frames`
  are cgif-specific palette-reuse and frame-coalescing machinery with no
  pure-Rust equivalent and are not modelled.

- SVG rasterisation, behind a new non-default `svg` feature (issue #502).
  `decode_svg` was a typed stub reporting that librsvg was missing; it now
  renders through `resvg` and returns a 4-band 8-bit sRGB raster with
  unpremultiplied alpha, matching what vips gets out of librsvg and cairo.
  `dpi`, `scale` and `unlimited` are implemented and pinned against vips
  8.18.4: `total_scale = scale * dpi / 72`, output geometry rounds half up,
  and `Xres`/`Yres` become `dpi / 25.4` pixels per millimetre. `scale`
  deliberately does not move the resolution, and a physically-sized document
  takes `dpi` twice (once converting millimetres to user units, once through
  `total_scale`) because that is what vips measurably does.

  The feature is off by default because it costs 29 crates. All of them are
  pure Rust: no `-sys` crates and nothing that compiles C, so enabling it does
  not put a C toolchain in your build.

  `<image xlink:href>` never resolves. usvg's stock resolver reads local files
  and, with no `resources_dir` set, takes the href verbatim, so an untrusted
  document could read arbitrary files and probe for their existence. Both
  halves of the resolver are overridden to refuse every href, which means
  `<image>` elements do not render at all, from any source including `data:`
  URIs. That is a deliberate divergence from vips.

  Not implemented: `stylesheet`, `high_bitdepth` (resvg has no float surface
  to render scRGB into), and text fidelity. Text renders against the bundled
  Bitstream Vera face so it is deterministic, but vips shapes through pango
  against system fonts and the two do not match: measured, 12.8% of pixels
  differ on a short line of text and the advance width moves. SVG is also not
  added to the content-sniffing route table, because it has no fixed leading
  magic; `decode_svg` is the entry point.

- Radiance HDR (`.hdr`) load and save (issue #506): `decode_radiance` reads a
  Radiance file into a `FloatF32(3)` raster tagged `ScRgb`, and
  `Raster::encode_radiance` / `Raster::save_radiance` write one back out.
  `decode_file` and `decode_bytes` route `.hdr` there automatically from the
  magic, so an existing caller needs no change. This is the first raster
  format libviprs decodes itself rather than through the `image` facade.

  It is hand-rolled for a specific reason. `image` decodes RGBE as
  `mantissa * 2^(e-136)` where libvips uses the half-bit-centred
  `(mantissa + 0.5) * 2^(e-136)`, and that is not a rounding difference: the
  error is `0.5/mantissa`, so 0.44% at mantissa 161, 33% at mantissa 1, and
  100% at mantissa 0. A saturated red HDR pixel has green and blue mantissas
  of zero, so vips reports a small positive floor there and `image` reports a
  hard zero, which silently breaks any downstream divide, log, or tone map.
  The encode side is the matching half, `frexp(max) * 255.9999 / max` with a
  `1e-32` floor. Together the two are the identity on any RGBE quadruple whose
  largest mantissa is at least 128 and whose exponent byte is in `23..=255`,
  which is exactly the normalised form an encoder emits, so a `.hdr` written
  by vips or by libviprs round-trips byte for byte. Verified against the real
  vips 8.18.4 binary over the reference suite's two `sample.hdr` images,
  3,057,600 pixels, decode and encode both byte-identical.

  The carrier is float and never RGBE. vips models Radiance as a *coding*, a
  4-band uchar raster tagged `VIPS_CODING_RAD` that it unpacks to 3-band float
  scRGB the moment any operation touches it. libviprs has no coding concept
  and decodes straight to the float. The visible consequence is that
  `vipsheader` on a libviprs-loaded `.hdr` reports `bands 3 / float /
  coding none` where vips reports `bands 4 / uchar / coding rad`; the 4-band
  spelling was rejected on correctness rather than taste, because `resample`
  premultiplies on `has_alpha()` and `resize` forks its downscale kernel on
  it, so an RGBE raster tagged `Rgba8` would be premultiplied by its own
  exponent byte.

  One divergence from the vips binary is deliberate and documented at the
  entry point: `save_radiance` preserves high dynamic range, where a bare
  `vips radsave` converts to sRGB and clips (measured on `sample.hdr`, max
  7728 becomes 254.5). The equivalent vips invocation is `float2rad` *then*
  `radsave`, and that is the pair `save_radiance` matches.

  The `FORMAT=` header line is read past and ignored on load, so every
  `.hdr` is tagged `ScRgb` and `rad-format` always reads back as
  `32-bit_rle_rgbe`, `32-bit_rle_xyze` files included. That is a port of a
  libvips defect, not an oversight: `radiance.c:693-698` picks the colour
  tag from the parsed format, but the arm is unreachable, because
  `radiance.c:636` calls `formatval(line, read->format)` while
  `radiance.c:314` declares `formatval(char fmt[MAXFMTLEN], const char *s)`
  with `fmt` as the output buffer. The arguments are swapped, nothing is
  parsed, and the `COLRFMT` default survives. Honouring the line would put
  a third behaviour in the world, matching neither the source nor the
  binary, and the interpretation tag is consumed by `colourspace`, so it
  would move pixels rather than just the header. If upstream fixes
  `formatval`, libviprs should follow. The save side is unaffected and
  still writes `32-bit_rle_xyze` for an `Xyz` raster.

  One honest limitation, stated at `decode_radiance`: a `FloatF32(3)` raster
  is rejected by `resize` with `RasterError::FloatUnsupported`, so a loaded
  `.hdr` cannot enter the pyramid engine. The resampling and op surface handle
  float fine; only the tiled-pyramid path is closed.

- `Raster::join` and its `try_join` twin (issue #551): the port of libvips
  `vips_join`, the generic two-image spatial join. `a.join(&b,
  JoinDirection::Horizontal, expand, shim, background, align)` puts `b` to the
  right of `a` (or below it, with `JoinDirection::Vertical`), separated by
  `shim` pixels and lined up on the edge `align` names. libviprs already had
  `arrayjoin` for a whole grid and `insert` for an explicit offset, but
  nothing for the ordinary "put these two next to each other" case.

  `expand` is the flag worth reading twice, because it does not mean what its
  name suggests it might. With `expand` false, which is the libvips default,
  the result is cropped back to the smaller of the two images along the shared
  axis: joining a 3x2 and a 2x3 horizontally gives 5x2, not 5x3, and the
  bottom row of the taller image is gone. Pass `expand` true to keep every
  input pixel, and `background` then fills whatever neither image covers,
  including the shim gap.

  `align` is `Align::Low`, `Align::Centre`, or `Align::High`, and it parses
  from the libvips nicknames (`"low"`, `"centre"`, `"high"`, and `"center"`)
  through `FromStr`. `Centre` is computed the way libvips computes it, as two
  separate truncating integer divisions `in1 / 2 - in2 / 2`, which is not the
  same as `(in1 - in2) / 2`: for a 4-high image joined to a 3-high one the
  first form offsets by 1 and the second by 0. Matching the C exactly here
  means a libviprs join lands on the same pixel a vips join does.

  Band counts and depths unify exactly as `insert` does, since that is what
  libvips leans on too, so a one-band image joined to a three-band one gives
  three bands and an 8-bit joined to a 16-bit gives 16-bit. Failures from the
  delegated insert and crop arrive as the new
  `ConversionError::Extract(ExtractError)` variant, and a bad align nickname
  as `ConversionError::UnknownAlign`. `ConversionError`, `JoinDirection` and
  `Align` are all `#[non_exhaustive]`.

  Three things are refused up front rather than delegated. A float raster on
  either side is `ConversionError::FloatFormatUnsupported`, because the
  placement path underneath reads samples as `u8` or `u16` and panics on
  4-byte ones; that is not an exotic input, since every `colourspace` result
  for Lab, Lch, OkLab, OkLCh, XYZ, scRGB and Yxy is a float raster, so
  `im.colourspace(Lab).join(&other, ..)` would otherwise panic out of a
  fallible method. A `shim` above `1000000` is
  `ConversionError::ShimTooLarge`, matching the bound libvips declares on the
  property (`VIPS_ARG_INT(class, "shim", 5, ..., 0, 1000000, 0)`, so
  `vips join --shim 1000001` is refused before the operation runs). Widening
  the argument to `u32` had carried the lower bound into the type and dropped
  the upper one, and without the check `shim = 2147483644` on a 3x2 and a 2x3
  asks for a 6.44 GB canvas, each raster still under the per-raster
  allocation budget so the budget never fires. An offset outside `i32`, the
  range libvips places images in, is
  `ConversionError::PlacementOffsetOverflow`, and it reports the offset
  `(x, y)` that did not fit rather than a result size. Note that a `join`
  canvas too large for `u32` still arrives as
  `ConversionError::Extract(ExtractError::SizeOverflow)`, so a caller that
  wants every "too big" outcome matches both.

  A band-promoting join drops the first image's interpretation instead of
  copying it onto a result it no longer describes. `vips join` of a 1-band
  `b-w` with a 3-band `srgb` reports `3 bands, srgb`, and keeping `b-w` is
  not cosmetic: `space_bands(Bw) == 1`, so a later `colourspace(Lab)` reads
  two of the three bands as passthrough extras and hands back 5 bands instead
  of 3. Dropping the tag lets the getter infer one from the result format,
  the same re-stamp `composite2` already does for the same reason. A
  depth-only promotion keeps the tag, matching `vips join` of `b-w` uchar
  with `grey16`, which still reports `b-w`.

- `Raster::sobel`, `Raster::scharr` and `Raster::prewitt`, with their
  `try_*` twins (issues #537, #549, #550): the port of libvips `vips_sobel`,
  `vips_scharr` and `vips_prewitt`. They are one abstract op in libvips
  (`convolution/edge.c`) differing only in a 3x3 mask, they take no arguments
  at all, and each convolves with its mask and with the mask rotated 90
  degrees before combining the two gradients into an edge map.

  How the gradients combine depends on the input format, and the two rules are
  not approximations of each other. A uchar input takes the fast arm: the mask
  is stamped `scale = 2, offset = 128` so a signed response lands centred in
  the unsigned range, both convolutions run at integer precision, and the
  result is `|Gx| + |Gy|` clipped at 255. Every other format takes the
  accurate arm: two float convolutions with the raw mask, then
  `sqrt(Gx^2 + Gy^2)`, then a truncating cast down to 8 bits. On a corner
  where `Gx` and `Gy` are equal the abs sum is `2 * g` where the magnitude is
  `sqrt(2) * g`, so the same picture reads 58 through the uchar arm and 42
  through the float one.

  The output is uchar whatever went in, so a 16-bit or float source comes
  back narrowed four bytes per sample to one. Width, height, band count,
  interpretation, resolution and the attached metadata (EXIF blob, ICC
  profile, arbitrary attachments) all survive, matching what `vips sobel`
  hands through. Alpha is convolved as an ordinary band, exactly as libvips
  does it, so a fully opaque RGBA input comes back fully transparent except
  along its edges. Saturation on the uchar arm happens twice, once inside
  each convolution and once on the abs sum, which is what makes the
  recovered gradient span an asymmetric `-256..=254`.

  These three inherit the integer-convolution divergence described under
  **Changed** below, at a bound of 4.

- `Raster::matrixmultiply` and its `try_matrixmultiply` twin (issue #533): the
  port of libvips `vips_matrixmultiply`, the dense product of two matrix
  images. `left.matrixmultiply(&right)` needs `left.width() ==
  right.height()` and gives a `right.width()` x `left.height()` one-band float
  matrix stamped `Interpretation::Matrix`, accumulated in `f64` with no scale
  and no offset (libvips ignores the scale and offset members of both inputs,
  and libviprs matrices carry neither). Shapes that do not chain are the new
  `MatrixError::ShapeMismatch` variant rather than a panic, and either operand
  failing the `vips_check_matrix` gate is the existing
  `MatrixError::NotOneBand` / `MatrixError::TooLarge`. `MatrixError` is
  `#[non_exhaustive]`, so the added variant is not a breaking change.
  The output's width and height come from two independent operands, each
  capped only at 100000, so the product can be enormously larger than either
  input: a pair of 400 KB matrices shaped `1 x 100000` and `100000 x 1` asks
  for a 40 GB result. That size is checked before anything is allocated, so it
  comes back as `MatrixError::Raster(RasterError::ByteBudgetExceeded)` instead
  of committing the memory first (the abort class issues #280 and #433 removed
  elsewhere in the crate).

- `Raster::remainder` / `Raster::try_remainder`, the generic two-image
  remainder (issue #536). This is the image-image companion to the existing
  constant form `rem_const`, and it ports libvips `vips_remainder`: each
  sample of the result is `self` mod the matching sample of `other`. Output
  depth is the wider of the two input depths, matching the identity promotion
  table libvips applies after formatalike, so `uchar % uchar` stays 8-bit and
  `uchar % ushort` promotes to 16-bit.

  The kernel is C's truncating `%`, and it lives in one shared `remainder_vips`
  function that both `remainder` and `rem_const` run, so the image-image and
  constant forms cannot disagree for identical operands. libvips does not pick
  one definition, it dispatches on format: `IREMAINDER` truncates for the
  integer formats, `FREMAINDER` floors for `float` and `double`. Every carrier
  the crate has today is an unsigned integer one, so truncating is the branch
  that matches vips, on both forms, including the negative constant `rem_const`
  can be handed. The choice is invisible to the image-image form in any case,
  since the two definitions agree on every non-negative operand pair (checked
  exhaustively over all 4,294,836,225 pairs with `a` in `0..=65535` and `b` in
  `1..=65535`, zero disagreements). A float carrier will need the floored
  branch added, which is spelled out where the kernel is defined.

  Three deliberate divergences from libvips, all spelled out on the method's
  docs. A zero divisor gives `0` here where libvips gives `-1` (which reads
  back as `255` through a uchar carrier), since libviprs has no signed carrier
  and `x % 0 == 0` is already the crate-wide convention. There is no band
  broadcast and no size alignment: the two rasters must agree exactly on
  width, height, and band count, the same contract every other image-image
  operation in the arithmetic module has, rather than libvips's
  bandalike-then-sizealike. And float rasters are rejected on either side,
  since the operation rounds and saturates into an unsigned output, so there
  is no representable place for a fractional or negative sample; cast to an
  unsigned 8- or 16-bit format first.

- `ConvolutionError::NonFiniteMaskParameter { param, value }` rejects a `NaN`
  or infinite mask scale at the convolution boundary (issue #534). `conv`,
  `convsep`, `compass`, `gaussblur` and `sharpen` all reach the engine through
  that one check, so they all get it. The enum is `#[non_exhaustive]`, so the
  new variant is additive.

### Changed

- The edge detectors answer both gradients in one traversal instead of two,
  and combine them without materialising either (issue #562). `sobel`,
  `scharr` and `prewitt` used to run the convolution engine twice over the
  same source: each pass widened every sample to `f64` on its own, walked
  every window on its own, and wrote a full-image intermediate raster, and a
  third pass then combined the two. Every output sample's two responses come
  off the same nine source values, so it collapses into one pass with two
  accumulators. Not one output byte moves; the same 24 hard-coded digests and
  the same vips captures pin it.

  Measured on aarch64 at `opt-level = 3`, best of 21 runs against the same
  fixtures built by the same code, alternating the two binaries so they share
  the machine's noise:

  | fixture | before | after | |
  |---|---|---|---|
  | 2048x2048 `Gray8` | 59.3 ms | 23.0 ms | 2.6x |
  | 4096x4096 `Gray8` | 249 ms | 96.7 ms | 2.6x |
  | 1024x1024 `Rgb8` | 43.6 ms | 16.7 ms | 2.6x |
  | 1024x1024 `FloatF32(1)` | 16.6 ms | 2.5 ms | 6.6x |
  | 512x512 `FloatF32(3)` | 11.8 ms | 1.9 ms | 6.3x |

  Peak resident size on a 4096x4096 `sobel` goes from 326 MB to 166 MB on
  `Gray8` and from 806 MB to 278 MB on `FloatF32(1)`.

  A plain `conv` gets most of it too, 2.5x on 8-bit and 4.7x on float, because
  the same rework took the edge clamp out of the inner loop. The clamped
  source index is now a pair of small lookup tables built once, which is
  `vips_embed(..., VIPS_EXTEND_COPY)` written as indices instead of as pixels,
  and the interior of each row is a contiguous run the compiler can vectorise.
  Zero taps are skipped as well, which is the #574 fix paying for itself:
  sobel's mask is six taps, not nine.

- **The integer-convolution parity contract is now stated on
  `Precision::Integer`, and the divergence against a stock libvips is
  unbounded rather than "at most 2" or "at most 4"** (issue #558). No
  convolution arithmetic changed here. What changed is that the claim is
  written down honestly and is checkable, because two earlier statements of
  it in this changelog and in the module docs were wrong.

  libviprs ports `vips_convi_gen`, the portable C integer-convolution loop.
  libvips's own documentation names that loop as the specification and flags
  the alternative as a deviation from it: "`@mask` is converted to an integer
  mask with `rint()` of each element ... For `UCHAR` images, `vips_convi` uses
  a fast vector path based on half-float arithmetic. **This can produce
  slightly different results.** Disable the vector path with
  `--vips-novector` or `VIPS_NOVECTOR`" (`convi.c:1276-1284`). It is also what
  libvips falls back to whenever `vips_convi_intize` declines a mask, which it
  does on ordinary input, so it is the floor rather than one of two options.
  `VIPS_NOVECTOR=1 vips` reproduces libviprs byte for byte.

  The first correction is the **mechanism**. The old wording said the two
  paths differ in how they round the final divide, so that "an
  integer-precision convolution of an unsigned image whose window sum is
  negative and even reads one lower from libviprs". That is not the dominant
  effect and the rule does not hold. `vips_convi_intize` rebuilds the mask
  over a power-of-two denominator, so the vector path **convolves with
  different coefficients**: a 3x3 box blur of scale 9 is applied as
  `57/512 = 0.111328`, not `1/9`. On a window summing to 1147 the C path gives
  `(1147 + 4) / 9 = 127`, flooring also gives 127, and the vector path gives
  `(57 * 1147 + 256) >> 9 = 128`. Changing how libviprs rounds would move zero
  bytes for `gaussblur`, for `conv` with a non-negative mask, and for
  `canny`'s first stage.

  The second correction is the **bound**, and it is the one that matters if
  you are writing a comparison. `vips_convi_intize`'s accuracy check
  (`convi.c:1096-1113`) is not a bound on the two paths at all: it compares
  the requantised mask against exact real arithmetic, at one grey level, on a
  flat field, so it constrains DC gain and says nothing about per-pixel error.
  Of 400 random 3x3 masks, 301 were accepted onto the vector path and 179 of
  those diverge by more than 2. One accepted mask,
  `[45 -17 -25 / -33 -15 -34 / 55 53 -26]` at scale 3, has been measured
  **128 of 255** apart over a near-binary noise field, and 73 and 2 over two
  other inputs, so even that is a fixture's number and not a bound. Use
  `VIPS_NOVECTOR=1` rather than a tolerance.

  Three regimes exist, not two, and nothing on this API surface tells you
  which one a mask is in: the vector path can run and disagree; it can run and
  agree (scale 1, or any exact requantisation, including every power-of-two
  scale); or libvips can decline the mask and run the C path itself. Sigma
  1.4, the `gaussblur` default, is lucky only for the *separable* gaussmat,
  whose scale is 64. The 2D gaussmat at the same sigma has scale 216 and is
  not, which is why a suite pinned only at the default sees none of this.

  This is a property of the **library**, not of the `vips` command line.
  pyvips, sharp, ruby-vips and anything linking a distro libvips all hit the
  identical gap, and `VIPS_NOVECTOR` is read once at library init, so the
  escape hatch works for a CLI comparison and not for a caller who already
  holds an `Image`.

  It reaches `conv` and `convsep` at integer precision, `compass`,
  `gaussblur`, and the uchar arm of `sobel` / `scharr` / `prewitt`. On the
  edge detectors the gap is **quadrupled, not doubled**: the uchar arm
  recovers each response as `2 * (p - 128)`, which doubles a one-unit gap, and
  `Gx` and `Gy` can both be off at once. Measured on an 8x3 `Gray8` image,
  `prewitt` at pixel (4,0) reads 106 from libviprs and from
  `VIPS_NOVECTOR=1 vips`, and 110 from the same binary with the vector path
  live. The float arm has no such gap and is bit-exact either way.

  `sharpen` is **not** in that list any more, and dropping it is a third
  correction. It convolves the `L` of `LabS`, which is 16-bit, and the vector
  path is gated on `BandFmt == VIPS_FORMAT_UCHAR` (`convi.c:1151`), so both
  libvips builds take the C path, and `VIPS_INFO=1` says so. Any `sharpen`
  deviation is a separate libviprs bug, tracked as issue #581.

- The edge detectors' float arm now closes through `Raster::try_cast` instead
  of a private `cast_uchar_truncating` helper (issues #558, #561). #561 made
  `try_cast` truncate float samples toward zero, which is exactly what the
  helper existed to work around, so the helper was a second copy of
  `vips_cast`'s rule with nothing left to add, and its doc comment still
  claimed libviprs rounded, which stopped being true the moment #561 landed.
  libvips builds that arm out of a `vips_cast` call on the whole magnitude
  image (`edge.c:174`), so going through `try_cast` also matches the shape of
  the original. `sobel`, `scharr` and `prewitt` produce identical bytes; the
  pinned impulse, vertical-step and truncation fixtures all still pass. The
  one visible consequence is a new
  `ConvolutionError::Conversion(ConversionError)` variant, which only the
  allocation inside `try_cast` can reach in practice. `ConvolutionError` is
  `#[non_exhaustive]`, so matching code is unaffected.

- **Breaking (cast): a float sample narrowing to an integer format is now
  truncated toward zero, where it used to be rounded to nearest** (issue #561).
  `Raster::cast` and `Raster::try_cast` are the operations that move, and so is
  anything that narrows through them, `Raster::freqmult` included. This changes
  output bytes for a public API shipped in 0.4.0: casting `1.7` to `Gray8` now
  gives `1` where it used to give `2`, and `254.6` gives `254` where it used to
  give `255`. Roughly half of all fractional samples shift down by one.

  The old behaviour was simply wrong against libvips. `cast.c:566-567` says
  "Floats are truncated (not rounded). Out of range values are clipped", and
  vips 8.18.4 agrees on every row I measured: `1.7` to `1`, `2.5` to `2`,
  `3.999` to `3`, `254.6` to `254`, and on the wider target `300.9` to `300`.
  libviprs answered one above vips on all five. The rustdoc made it worse by
  claiming the rounding and claiming parity with `vips_cast` in the same
  paragraph, so it promised libvips compatibility while describing
  libvips-incompatible behaviour; both halves of that are corrected, and the
  doc now scopes the parity claim to the formats `PixelFormat` can actually
  carry.

  Clipping and the `NaN` pin do not move. Those already matched vips (below
  range to `0`, above range to `255` or `65535`, `NaN` to `0`), and there are
  now tests pinning each so the next change to this arm cannot quietly take
  them with it. The truncation is `f64::trunc`, not `f64::floor`, which reads
  as a distinction without a difference today because every carrier here is
  unsigned and a negative sample clips to `0` before the rounding mode can
  show. C's `(int)` conversion truncates toward zero, so `trunc` is the form
  that stays correct once a signed carrier lands (#516).

- **Breaking (WebP and GIF encode): `Raster::encode_webp` now takes a
  `webp::SaveOptions` instead of a `quality: u8`, and the three GIF stubs
  `encode_gif`, `encode_gif_interlaced`, and `encode_gif_dither` collapse into
  one `Raster::encode_gif(gif::SaveOptions)`** (issue #563). Both still return
  the same typed `EncodeError::Unsupported` they always have, so nothing that
  worked stops working, but the call sites have to be updated.

  The WebP change is the one with teeth. vips `webpsave` takes a `Q` factor
  *and* a `lossless` flag, and quality only means anything on the lossy path.
  The only pure-Rust WebP encoder libviprs can reach is lossless-only and has
  no quality knob at all, so the `quality` argument was going to be accepted
  and thrown away. That inverts the contract (ask for quality 10, get a
  lossless file possibly larger than the PNG you started from) and it is a
  semver time bomb: the day a lossy encoder lands, every existing
  `encode_webp(10)` silently starts producing small lossy files in a patch
  release. Quality is now unrepresentable rather than ignored, as a
  `#[non_exhaustive] webp::Compression` whose only variant is `Lossless`, so
  `Compression::Lossy { .. }` can be added later as a minor bump.

  **Upgrading:** `im.encode_webp(80)` becomes
  `im.encode_webp(webp::SaveOptions::default())`; `im.encode_gif()` becomes
  `im.encode_gif(gif::SaveOptions::default())`; `im.encode_gif_interlaced()`
  becomes `im.encode_gif(gif::SaveOptions { interlaced: true,
  ..Default::default() })`; and `im.encode_gif_dither(d)` becomes
  `im.encode_gif(gif::SaveOptions { dither: d, ..Default::default() })`.
  Neither options struct is `#[non_exhaustive]`, so struct literals and
  `..Default::default()` both work from outside the crate.

- GIF and WebP files now decode. The `image` dependency is built with its
  `gif` and `webp` features on (issue #563), where before it carried only
  `jpeg`, `png`, and `tiff`, so `decode_file` and `decode_bytes` read those
  two containers instead of reporting an undecodable format. The `hdr` feature
  is still off and stays off: the crate decodes RGBE as `mantissa * 2^(e-136)`
  where vips uses the half-bit-centred `(mantissa + 0.5) * 2^(e-136)`, which
  is a 100% error at mantissa 0, so it was never usable for parity, and
  leaving it off is also what keeps the unchecked RLE multiply in its Radiance
  decoder unreachable.

- **Breaking (`.v` container): a file tagged `OkLab` or `OkLch` now carries the
  real libvips interpretation codes `30` and `31` in its header `Type` word,**
  so it interoperates with vips instead of only with libviprs (issue #535).
  libvips 8.18 assigned those codes (`VIPS_INTERPRETATION_OKLAB` and
  `VIPS_INTERPRETATION_OKLCH`, `libvips/include/vips/image.h:115-116`), but
  libviprs still wrote the private extension codes `1000` and `1001` it had
  picked while libvips had none. The consequence ran both ways: a `.v` written
  by real vips came back tagged `Multiband`, because `30` matched no arm of the
  reader and the raster fell through to format inference, and a `.v` written by
  libviprs was unreadable as OkLab anywhere else. This changes what goes on
  disk: newly written files hold `30` / `31` where they used to hold `1000` /
  `1001`. The change is one-way. The reader keeps `1000` and `1001` as legacy
  aliases, so files libviprs has already written still load with their
  OkLab/OkLch tag intact, but nothing emits those codes any more, and a file
  written by this version does not read as OkLab on libviprs 0.4.0 or earlier.
  **Upgrading:** nothing to do to keep reading the files you already have. The
  aliases are permanent, not a deprecation window: `1000` and `1001` stay
  reserved for OkLab/OkLch forever and will never be reused, because retiring
  them would silently re-break every `.v` libviprs has already written. To make
  an already-written file readable by vips, re-encode it with this version
  (load it and save it again); there is no in-place header rewrite.

- The panicking matrix operations no longer double the operation name in their
  panic message (issue #339's class, found while reviewing #533). Every
  `MatrixError` variant except the transparent `Raster` tail already opens with
  the operation that failed, so the wrapper's own `"<op>: "` prefix produced
  `matrixinvert: matrixinvert: non-square matrix (3x2)`. The prefix is now
  applied only to the `Raster` tail, whose message names no operation. Code
  matching on the typed errors is unaffected; only the panic text changes.

### Fixed

- The TIFF page readers honour `DecodeLimits` instead of bypassing it (issue
  #540). `decode_tiff_page` and `tiff_page_count` took no limits at all and
  handed the decoded result straight to `Raster::new`, whose
  `DEFAULT_MAX_ALLOC_BYTES` is 8 GiB, sixteen times looser than the 512 MiB
  `DecodeLimits::max_alloc_bytes` the rest of the crate publishes and honours.
  `src/source.rs` publishes a table of which decoder enforces which field, and
  these two were not in it.

  Four things on that path were sized by the file, not the one the issue
  describes, because #566 added two more while sourcing `n-pages`:

  * The whole-file `std::fs::read`, unbounded. It is now capped at
    `max_alloc_bytes`, checked against the declared length before the read and
    against what actually arrived after it, so a file that grows in between
    cannot slip past.
  * `normalize_multiband_photometric`, which clones the entire buffer when the
    vips multiband relabel applies, doubling the peak footprint. The page
    readers own their buffer, so they now patch it in place and never pay for
    the copy.
  * The IFD walk, which ran to the end of the chain with no ceiling on every
    single page decode. `DecodeLimits` grows a `max_pages` field, default
    `100_000`, and the walk stops there with `SourceError::PageLimitExceeded`
    rather than counting on to find out how far past it the file goes. The
    default is the ceiling libvips puts on both the page index and the page
    count on every multi-page loader it has: measured against 8.18.4,
    `vips tiffload x.tif o.v --page 100001` and `--n 100001` are both refused
    before the loader runs.
  * The pixel buffer, which now goes through `check_coord`, then
    `check_pixels`, then an explicit
    `width * height * bands * bytes_per_sample` budget, all on the declared
    geometry and all before anything is reserved. That last one is the only
    check that can see the band count and the sample depth, so it is the one
    that catches a frame `max_pixels` waves through.

  `decode_tiff_page_with_limits`, `tiff_page_count_with_limits` and
  `Raster::tiff_load_with_limits` take the ceilings explicitly; the existing
  three delegate to `DecodeLimits::default()`. `DecodeLimits` is
  `#[non_exhaustive]` with `with_*` builder setters, so `max_pages` is
  additive, and `SourceError` is `#[non_exhaustive]`, so the two new variants
  (`AllocLimitExceeded` and `PageLimitExceeded`) are too. A file that was
  already decoding keeps decoding: the new ceilings are all far above anything
  a real TIFF carries, and the `tiff` crate's own 256 MiB decode buffer
  default is only ever tightened by `max_alloc_bytes`, never loosened.

  One thing the issue asserts that no longer holds, recorded so nobody chases
  it: it argued this was "the template" the other format modules would copy,
  and they did not. `gif.rs`'s `decode_gif` and `webp.rs`'s `decode_webp` both
  take a `DecodeLimits` and apply `check_coord`, `check_pixels` and
  `max_alloc_bytes` already. A cyclic IFD chain is not an unbounded loop
  either: `tiff` 0.10.3 runs union-find over the IFD edges and returns
  `CycleInOffsets` on a back edge. The ceiling is for the chain that is merely
  very long, which nothing below it bounds.

- A zero mask coefficient no longer poisons a non-finite sample (issue #574).
  libvips squeezes zero taps out of a mask before it convolves, in both cores
  (`convolution/convf.c:314-321` and `convolution/convi.c:1189-1197`), and
  this port iterated every tap instead. `0.0 * inf` is `NaN`, so a structural
  zero sitting over an infinity poisoned the whole response, survived the
  square and the root, and clipped to 0. On a 5x5 float image that is all zero
  except for an infinity at its centre, `sobel` read `0` at four cells of the
  impulse ring where vips reads `255`, and `scharr` and `prewitt` did the same
  thing for the same reason: all three masks have structural zeros. The taps
  are now compacted after the scale division, exactly where vips does it, so
  the answers match.

  It reaches further than the edge detectors, because the same engine serves
  `conv`, `convsep`, `compass` and `gaussblur`. Any mask with a zero
  coefficient over a non-finite sample had the property, and a `.v` file can
  carry `inf` or `NaN`. Finite input is untouched: dropping `+ 0.0 * x` can
  only change the sign of a zero, and a signed zero does not survive `a * a`.

  An all-zero mask keeps one tap rather than none, which is not the same as
  skipping everything. Both C cores force the tap count back up to 1 at mask
  index 0 when the whole mask squeezed away, so an all-zero mask still answers
  `NaN` over an infinity, but only at the single output pixel whose window
  top-left is the infinity. Measured on vips 8.18.4 and pinned.

- The edge detectors no longer build float or unsigned intermediates through
  the byte-budgeted `Raster::new`, so `sobel()`, `scharr()` and `prewitt()`
  cannot panic on a legal input any more (issue #575). A 16-bit source above
  about 4 GiB implied a float intermediate over the 8 GiB
  `DEFAULT_MAX_ALLOC_BYTES` ceiling, and the panicking twins turned that
  rejection into a process-ending `expect` on an input the crate accepts. The
  convolution buffers now go through `alloc_op_output` and
  `Raster::from_op_output`, the fallible budget-free pair `arithmetic.rs`
  already uses and that issue #279 exists to provide. The edge detectors go
  further and never build the intermediates at all. The `try_reserve` and
  per-row streaming halves of #575 are untouched and stay open.

- `Raster::try_arrayjoin` no longer panics on a float input (issue #551). Its
  sample copy is unsigned-only and panicked on 4-byte samples, so a fallible
  method aborted the process on ordinary input: `space_depth` maps Lab, Lch,
  OkLab, OkLCh, XYZ, scRGB and Yxy all to F32, which makes every `colourspace`
  result for those spaces a float raster. It is now
  `ConversionError::FloatFormatUnsupported { op: "arrayjoin" }`, the same
  guard `join` got, and the panicking `arrayjoin` twin still panics through
  the usual `expect` path. Real vips handles float on both operations, so this
  is a libviprs limitation reported honestly rather than parity, and it goes
  away when the unsigned-only sample helpers grow a float arm.

- `Raster::arrayjoin` now rejects a `shim` above `1000000` with
  `ConversionError::ShimTooLarge` (issue #551), the same bound `join` got.
  Both operations declare the property as `VIPS_ARG_INT(class, "shim", 5, ...,
  0, 1000000, 0)`, and the binary refuses `vips arrayjoin --shim 1000001` with
  the identical GObject CRITICAL, so the two now agree with each other and
  with vips. `--shim 1000000` still builds the grid vips builds.

- `Raster::arrayjoin` no longer tags a band-promoted grid with the first
  image's interpretation (issue #551). `bandalike` promotes a one-band input
  up to the widest band count in the list, so a grid built from a 1-band
  `b-w` and a 3-band `srgb` has 3 bands while the copied tag still says
  `b-w`. `vips arrayjoin` reports `srgb` for that pair. The mis-tag is not
  cosmetic, since `space_bands(Bw) == 1`: a later colourspace conversion
  reads bands 1 and 2 of the grid as passthrough extras rather than colour,
  and returns a different band count with different numbers in it. The tag is
  now cleared whenever the grid's band count differs from the first image's,
  which lets the getter infer one from the result format. A depth-only
  promotion still keeps the tag, matching `vips arrayjoin` of `b-w` uchar
  with `grey16`. Nothing changes for a grid whose inputs all already share a
  band count, which is the common case.

- `decode_file` and `decode_bytes` now identify a format the same way, so the
  same bytes decode to the same raster whichever entry point you call
  (issue #563). They disagreed: the file path read a four-byte head, handled
  `.v` and JPEG itself, and handed everything else to `ImageReader::open`,
  which resolves the format from the **path extension** and never reads the
  file, while the in-memory path resolved it from the **content**. A
  PNG-encoded file named `photo.jpg` therefore failed through `decode_file`
  and succeeded through `decode_bytes`, and a file with no extension at all
  failed through `decode_file` with "the image format could not be
  determined" while decoding perfectly from a buffer. Both entry points now
  share one magic-byte sniff and one route table, and the extension is not
  consulted anywhere in the decode path. libvips has always behaved this way:
  `vips_foreign_find_load` asks each loader's `is_a` in priority order and
  does not trust the filename.

  The sniff head also grew from 4 bytes to 16. Four is not enough to identify
  the containers the format work needs next: WebP's magic is `RIFF????WEBP`,
  which is 12 bytes with a four-byte file-specific length in the middle, and
  Radiance's is the 10-byte `#?RADIANCE`. Sixteen is what `image`'s own
  content guess reads, so the sniff never sees less of a file than the
  fallback does. `.v` and JPEG still decode from a whole in-memory buffer,
  because their decoders parse the container themselves, and every other
  format still streams, so no format's memory profile changed.

- A `.v` written by real vips and tagged `OkLab` or `OkLch` now reads back with
  that tag instead of falling through to format inference and reporting
  `Multiband` (issue #535). See the **Breaking (`.v` container)** entry under
  _Changed_ for what moved on disk and what an upgrader has to do.

- A convolution at `Precision::Integer` over a float image with a negative mask
  scale wrote `-0.0` where vips 8.18.4 writes `+0.0` (issue #534). The integer
  path divides by the intized scale and then adds the mask offset summand, and
  C promotes the `int 0` and rewrites the sign along with it; libviprs was
  skipping the add entirely, so a sum of zero over a negative scale kept its
  sign bit and reached `data()` as `-0.0`. It does not take a negative scale
  from the caller to hit: `vips__image_intize`'s brightness nudge turns an
  ordinary positive scale negative, so `Kernel { data: vec![vec![1.0, 1.0]],
  scale: 0.4 }` is enough. libviprs' own float-precision arm already wrote
  `+0.0` for the same input, so the two arms now agree with each other as well
  as with vips.
- A `NaN` mask scale used to panic with an integer divide by zero on the
  `Precision::Integer` path over an unsigned image (issue #534). The
  `scale == 0.0` guard let it past, and `rint(NaN) as i64` is `0`, so the
  rounded scale the arithmetic divides by was zero after all. An infinite scale
  got past the same guard and produced an all-zero image with no diagnostic.
  Both are now the typed `ConvolutionError::NonFiniteMaskParameter`, on both
  precisions.

- `colourspace` between `OkLab` and `OkLch`, and between `Lab` and `Lch`, now
  takes the direct route libvips gives those pairs instead of detouring through
  the XYZ hub (issue #552). libvips joins each cartesian space to its polar
  form with a single transform and nothing else in the pipeline
  (`colour/colourspace.c:244,276,478,494`), so routing them through XYZ added a
  cube-root round trip real vips never runs. On both pairs the two halves of
  that round trip fail to invert each other, so a neutral colour picked up a
  chroma out of nowhere and the hue read off that chroma was meaningless.

  On the Oklab pair the culprit is the matrix: the published inverse is only an
  8-decimal approximation (it carries the `1.00000001` and `1.00000005` quirk
  digits), so the round trip pushed a neutral colour's `a` and `b` off zero by
  about 2e-9, and OkLab `[0.5, 0, 0]` came back as OkLCh
  `[0.5, 1.9e-9, 94.489]` where vips 8.18.4 returns `[0.5, 0, 0]`.

  On the Lab pair the culprit is the shadow branch, and it is the bigger of the
  two: `XYZ2Lab` switches to its linear segment at 0.008856 while `Lab2XYZ`
  switches at `L < 8`, and those rounded decimal constants are not mutual
  inverses. Dark neutrals came out about 3e5 times further off than the Oklab
  ones in raw units, so `Lab [5, 0, 0]` converted to LCh
  `(4.99996, 5.571e-4, 338.199)` where vips returns `5 0 0`. Above about
  `L = 10` the residue rounds away, which is why the defect only ever showed in
  the shadows and why a mid-grey fixture says nothing about it. Both pairs
  convert in place now, so the hue is exact and `OkLab -> OkLch -> OkLab` gives
  back the value it started with.

- Hue no longer comes out as 180 degrees for a colour whose `a` is `-0.0`
  (issue #552). Anything that reads a hue off Lab-like coordinates was
  affected: `colourspace` into `Lch` and `OkLch`, plus the hue term inside
  `de00` and `de_cmc`, which read the same ladder. (`Cmc` reaches its hue
  through the XYZ hub, and the hub cannot produce a `-0.0` there.) libvips'
  `vips_col_ab2h` (`colour/Lab2LCh.c:61-89`) tests `a == 0` and answers
  0 / 90 / 270 from an explicit branch, and in C that test is true for `-0.0`
  as well; libviprs was
  taking `atan2` at its word instead, and `atan2(±0.0, -0.0)` is `±PI`. Against
  the binary, OkLab `[0.5, -0.0, 0.0]` is OkLCh `0.5 0 0` in vips 8.18.4 and
  was `[0.5, 0.0, 180.0]` here. The branch is now transcribed from the C, so
  the whole `a` axis answers the way vips does whichever zero it is handed.

## [0.4.0] — 2026-07-20

### Breaking

- Tile-lifecycle `EngineEvent` variants carry optional worker attribution
  (issue #67). `TileCompleted`, `TileFailed`, `TileSkippedOnResume`, and
  `RetryAttempted` each gain `worker_id: Option<WorkerId>` and
  `timestamp: Option<SystemTime>`. The in-tree engines set `worker_id: None`
  and stamp `timestamp: Some(_)` on the coordinating thread at emit time; an
  out-of-tree executor layer fills `worker_id` to route events back to the
  worker that produced them. Construction moves to the stamping helpers
  `EngineEvent::tile_completed`, `::tile_failed`, `::tile_skipped_on_resume`,
  and `::retry_attempted`. Code that pattern-matched these variants must add
  `..` (e.g. `TileCompleted { coord, .. }`); code that constructed them by
  struct literal must supply the new fields or use the helpers. `EngineEvent`
  is `#[non_exhaustive]`, so `matches!(e, EngineEvent::TileCompleted { .. })`
  filters are unaffected.

### Added

- Extensible raster drawing framework in the new `draw` module (issue #55).
  A `DrawOp` trait (`fn apply(&self, &mut Raster)`) is the seam: new shapes and
  paint effects plug in as `impl DrawOp` without touching the core `Raster`.
  Ships `Circle` and `Rectangle` ops (outline + filled), the inherent
  convenience methods `Raster::draw`, `draw_circle`, `draw_circle_filled`,
  `draw_rect`, `draw_rect_filled`, and `put_pixel` (clipping, format-agnostic
  pixel write). All drawing clips to the raster bounds. (Ink-width validation
  was added later; see the **Breaking (drawing)** entry under _Changed_ — the
  draw ops now reject a wrong-width ink rather than broadcasting it.)
- Pixel utilities on `Raster` (issue #55): `getpoint(x, y) -> Vec<f64>` reads a
  pixel as one `f64` per band (native byte order for 16-bit), and
  `add(&other) -> Raster` combines two rasters pixel-by-pixel, promoting 8-bit
  results to 16-bit so sums do not wrap (16-bit sums saturate at `65535`).
- `WorkExecutor` trait, `StripWorkUnit`, `WorkContext`, and `LocalWorkExecutor`
  in `streaming_mapreduce` (re-exported at the crate root): a plug-in seam at
  the MapReduce MAP-phase strip dispatch, so an out-of-tree executor (process
  pool, distributed worker layer) can substitute for the built-in in-process
  rendering (issue #67). Installed via `EngineBuilder::with_executor(...)`;
  the default `LocalWorkExecutor` is byte-identical to the previous engine.
- `EngineKind::MapReduceHotCache`: a local-only MapReduce variant that holds
  every produced tile in RAM and drains the caller's sink in one batched pass
  at the end of the run, in canonical `(level, row, col)` order, followed by
  a single `finish()` (issue #67). Byte-identical output to the streaming and
  MapReduce engines; explicit opt-in only, never selected by
  `EngineKind::Auto`.
- Worker attribution on `EngineEvent` (issue #67): new `WorkerId` newtype and
  new event variants `StripDispatched`, `StripExecutorDone` (emitted by the
  MapReduce engines on the coordinating thread, in canonical dispatch order,
  stamped with the installed executor's self-reported
  `WorkExecutor::worker_id`, `None` for `LocalWorkExecutor`), plus
  `WorkerJoined`, `WorkerLeft`, and `MemorySnapshot` as vocabulary for
  out-of-tree executor layers (never emitted by the in-tree engines).
  `WorkExecutor::worker_id` has a `None` default, so existing executor
  implementations are unaffected.
- `EngineBuilder::with_observers(Vec<Arc<dyn EngineObserver>>)` and the
  `FanOutObserver` composition behind it (issue #67): every event (and the
  `on_extensions` hatch) fans out to each registered observer in order.
  `with_observer` stays as the single-observer shorthand.
- Image generators ported from libvips (create batch of the ported-tests
  operation surface, libviprs-tests issue #55), in the new `create`
  module, all as associated constructors on `Raster` with `try_*`
  fallible forms and the new `CreateError`: the basic patterns `black` /
  `black_bands`, `xyz`, `eye`, `zone`, and `sines`; the seeded noise
  generators `gaussnoise`, `perlin`, and `worley` (bit-exact
  reproductions of the libvips FNV pixel-hash RNG, with `*_seeded` forms;
  the default seed is pinned to 0 so unseeded output is deterministic)
  and `fractsurf` (noise shaped by a fractal power spectrum through an
  internal 2D DFT, the `vips_freqmult` pipeline); the LUT and matrix
  constructors `buildlut`, `tonelut`, and `from_matrix`; the full
  frequency-mask family `mask_ideal` / `mask_ideal_ring` /
  `mask_ideal_band`, `mask_gaussian` / `mask_gaussian_ring` /
  `mask_gaussian_band`, `mask_butterworth` / `mask_butterworth_ring` /
  `mask_butterworth_band`, and `mask_fractal`, reproducing the
  `create/mask.c` semantics exactly (FFT layout with an `optical`
  quadrant swap, the DC component forced to 1.0 unless `nodc`, uchar
  truncation); signed distance fields `sdf` (circle, box, rounded-box,
  line) with the new `SdfParams`; and real text rendering `text` on the
  pure-Rust `ab_glyph` rasteriser with the bundled Bitstream Vera Sans
  face (`fonts/`), honouring dpi, word/char/word-char/none wrapping, and
  the width x height auto-fit search. `tests/create_ported_surface.rs`
  pins the ported call surface from the external-crate position.
- The arithmetic reductions (`avg`, `deviate`, `min` / `max`,
  `minpos` / `maxpos`) and relational maps now read `f32` rasters, so
  they work on the float images the create generators emit; the
  spellings `Raster::max_value` / `Raster::min_value` the ported suites
  use are provided as aliases. Mutating arithmetic ops still reject
  float inputs loudly.
- Colour-space and ICC operations ported from libvips (next batch of the
  ported-tests operation surface, libviprs-tests issue #55), in the new
  `colour` module: `Raster::colourspace` (with the typed
  `try_colourspace` form and the new `ColourError`) routing through D65
  XYZ across the libvips route table (Lab, XYZ, LCh, CMC, LabS, scRGB,
  HSV, sRGB, Yxy, Oklab, OkLCh, mono `b-w`/`grey16`, `rgb16`, and the
  no-lcms CMYK approximation), accepting either an `Interpretation` or a
  libvips space nickname (`"srgb"`, `"scrgb"`, ...; `Interpretation` now
  implements `FromStr`); the colour-difference metrics `Raster::de76`,
  `Raster::de00` (CIEDE2000), and `Raster::de_cmc` (the published
  CMC(l:c) 1:1 formula); the `Raster::constant` fixture constructor; and
  real ICC transforms on the pure-Rust moxcms CMS in `Raster::icc_import`
  / `icc_import_with` (with the new `Intent` and `Pcs` enums),
  `Raster::icc_export` / `icc_export_with` (8- or 16-bit device output),
  and `Raster::icc_transform`. Matrix-shaper RGB and grey-TRC profiles
  evaluate exactly from the parsed TRC curves and colorant matrix; LUT
  profiles (CMYK and other table-based classes) run through the moxcms
  LUT engine. Import keeps the source profile attached so a following
  export round-trips through it, and export attaches the profile it
  used, both mirroring libvips.
- Image IO, metadata fields, and library-level free functions ported
  from libvips (seventh batch of the ported-tests operation surface,
  libviprs-tests issue #55), in the new `imageio` module:
  `Raster::save` / `Raster::save_stripped` (extension-dispatched encode
  to PNG, JPEG, or the native `.v` container, with `save_stripped`
  writing pixels only), `Raster::encode_vips`, the named metadata field
  system (`get_field` / `set_field` / `try_set_field` / `get_fields` /
  `get_typeof` / `set_typeof` over the new `MetadataValue` enum, with
  the built-in header fields reading through to the raster header),
  `Raster::set_icc_profile` / `Raster::icc_profile`, and the free
  functions `tokenize` and `parse_thumbnail_geometry` (returning the new
  `ThumbnailGeometry`). JPEG save embeds the `icc-profile-data` and raw
  `exif-data` blobs as APP2/APP1 segments and the decoder captures them
  back; `.v` files round-trip the full header (both byte orders) plus
  every attached field, and reject unsupported band formats (float `.v`
  arrives with the float-format batch) and header geometry past the
  configured coordinate ceiling. `decode_file` now also records the source
  path in the `filename` field. Structured EXIF tag encoding (`exif-ifd*-*`
  into the JPEG APP1 TIFF directory) and PNG iCCP embedding are
  deferred to the foreign-format batch.
- `DecodeLimits` gains a `max_coord` field (with the `with_max_coord`
  builder setter) carrying the libvips `VIPS_MAX_COORD` single-axis
  dimension ceiling (10,000,000 px per axis default) as per-decode state.
  It is enforced on untrusted header geometry — before any pixel
  allocation — by **every** decoder: the native `.v` reader and the
  `image`-crate raster path (PNG/JPEG/TIFF) alike, returning the new
  typed `SourceError::CoordLimitExceeded` on an over-ceiling axis.
  `DecodeLimits` remains `#[non_exhaustive]`, so the field is additive.
- Porter-Duff and PDF blend-mode compositing ported from libvips, in
  the new `composite` module: `Raster::composite` /
  `Raster::composite2` with the typed `try_composite2` form
  (`CompositeError`) and the `CompositeMode` enum covering the full
  libvips `VipsBlendMode` set (Clear through Exclusion) plus the four
  PDF non-separable modes (Hue, Saturation, Colour, Luminosity) the
  ported conversion suite exercises. Inputs blend premultiplied in
  `f64` on the libvips `max_alpha` scale (255 unless both inputs are
  16-bit) and write back at the deeper input's depth, so every mode is
  exact to the output quantisation with no float `PixelFormat`
  required.
- Band operations ported from libvips (first batch of the ported-tests
  operation surface, libviprs-tests issue #55), in the new `bands` module:
  `bandjoin`, `bandjoin_const`, `bandjoin_vec`, `bandfold`, `bandunfold`,
  `bandmean`, `bandrank`, `bandand`, `bandor`, `bandeor`, `extract_band`,
  and `extract_bands` as inherent `Raster` methods. Each op also has a
  fallible `try_*` form returning the new typed `BandError` (re-exported at
  the crate root); the plain forms panic on invalid input, matching the
  ported-test call surface. `extract_band`/`extract_bands` accept negative
  indices (from the end), mixed-depth joins promote numerically to 16-bit,
  and constants clamp to the format range with round-to-nearest.
- `PixelFormat` gains `Multi8(n)` / `Multi16(n)` variants so band
  operations can represent results with band counts other than 1, 3, or 4
  (for example 2 bands from `extract_bands`, or width-many bands from
  `bandfold`), plus the canonicalizing constructor
  `PixelFormat::with_channels`. The enum was already `#[non_exhaustive]`,
  so external matches are unaffected; multiband rasters are compute
  intermediates and the tile-encoding sinks reject them with a typed
  `SinkError`. Manifest serialization round-trips the new variants as
  `"multi8:N"` / `"multi16:N"` while the six named formats keep their
  historical tags.

### Changed

- Image-image `Raster::sub` / `try_sub` now promotes to a `Float32` raster
  instead of routing through the integer round-and-saturate writer, so
  negative differences are preserved instead of saturating to `0`
  (e.g. `10 - 200` now reads back `-190`, not `0`). This matches the
  `vips_subtract` promotion table, which outputs signed `short` for `uchar`
  input — carried here as float so the signed result survives. Because the
  output floats, `try_sub` no longer returns
  `ArithmeticError::FloatUnsupported` for float inputs; a cast-then-subtract
  chain over float intermediates now works. The constant / per-band
  `sub_const` / `sub_vec` forms are unchanged — they still saturate at `0`,
  matching `vips_linear`'s requested-format (integer) output (libviprs#282).
- The cargo feature that gates the `sink_object_store` module
  (`ObjectStoreSink`) has been **renamed** from `s3` to `object-store-sink`
  (libviprs#345). The new name reflects that the module is a generic
  object-store sink driven by a user-injected `ObjectStore` backend — it pulls
  in no extra crates and ships no built-in S3 transport — rather than a
  ready-to-use S3 client. `s3` is kept as a deprecated alias
  (`s3 = ["object-store-sink"]`) so consumers pinned to the old feature name
  keep building; see the _Deprecated_ entry below. The README feature/module
  tables and the crate-root "Feature flags" rustdoc now present
  `object-store-sink` as the canonical name.
- The averaging resamplers `Raster::reduce` / `reduceh` / `reducev`,
  `shrink` / `shrinkh` / `shrinkv`, and `resize` now premultiply alpha
  around the whole operation for images with an alpha band: the source is
  premultiplied once into a float working buffer, every separable box /
  kernel / affine pass runs in that premultiplied space, and the result is
  un-premultiplied once at the end (the `vips_resize` bracket). This
  coverage-weights the colour so the meaningless RGB of transparent pixels
  can no longer bleed into opaque neighbours (the dark fringe at
  transparency boundaries), and it fixes a mixed downscale-one-axis /
  upscale-other-axis `resize` that previously fed straight-alpha colour to
  the affine enlargement. Bracketing once into float (instead of once per
  axis into a same-bit-depth integer intermediate) also removes the
  low-alpha colour banding a per-axis integer round-trip introduced. The
  single-tap Nearest kernel is unchanged — it stays an exact pick with no
  premultiply. This is a deliberate divergence from the bare `vips_reduce*`
  / `vips_shrink*` namesakes, which do not premultiply (only `vips_resize`
  does); it matches a premultiplied vips pipeline
  (`premultiply | reduce/shrink | unpremultiply`).
- The `max_coord` single-axis coordinate ceiling (libvips `VIPS_MAX_COORD`)
  is now per-decode state on `DecodeLimits::max_coord` instead of a mutable
  process-global, and it is enforced uniformly by **every** decoder. The
  `image`-crate raster path (PNG/JPEG/TIFF) previously honoured only
  `max_width` / `max_height` / `max_alloc_bytes` and silently ignored the
  coordinate ceiling; it now rejects an over-ceiling declared dimension
  with the typed `SourceError::CoordLimitExceeded`, on the header geometry
  before the frame is allocated, matching the native `.v` reader
  (libviprs#349). **Security note:** the former process-global
  (`set_max_coord` / `VIPS_MAX_COORD`) is now inert — a deployment that
  relied on it to *tighten* the ceiling below the 10,000,000-px default
  now decodes at that default on every non-`_with_limits` entry point with
  no runtime signal. Reinstate a tighter bound by passing a `DecodeLimits`
  with the desired `max_coord` to a `*_with_limits` decode call.
- The total `max_pixels` ceiling is now enforced on the raster path
  (PNG/JPEG/TIFF) against the declared header geometry *before* the output
  frame is allocated — matching the native `.v` reader — in addition to the
  existing post-decode re-verification in raster construction. Previously an
  over-`max_pixels` raster was only rejected after the frame was materialised.

### Deprecated

- The `s3` cargo feature is deprecated in favour of `object-store-sink`
  (libviprs#345). It survives only as an alias (`s3 = ["object-store-sink"]`)
  that enables the same `sink_object_store` module, so a consumer pinned to the
  old feature name keeps building unchanged. It carries no `since` version
  because the rename and deprecation happen within the same post-0.4.0
  unreleased cycle — no released version ever gated the module under a name
  other than `s3`, and the alias is scheduled for removal in a future release.
- **Breaking (drawing): a wrong-width ink is now rejected instead of broadcast.**
  The `draw` module entry points validate that `ink` is exactly one whole pixel
  wide for the target raster (issues #294/#346). Earlier the ops cycled a
  too-short ink to fill the pixel, so a single-value ink such as `&[128]`
  broadcast to a uniform shade on a multi-band raster (and a 3-byte RGB ink
  silently painted an `Rgba8` alpha band from the red byte). That silent cycle
  is a channel-corruption route, so a mismatched ink now fails up front: the
  panicking `Raster::draw_*` forms and the generic `Raster::draw(&op)` path
  panic, and the `try_draw_*` forms (plus the inherently fallible `draw_flood` /
  `draw_flood_blob`) return `Err(DrawError::InkLengthMismatch)`. The raw
  `Raster::put_pixel` escape hatch is unchanged and remains the deliberate
  opt-in for the shorter-ink cycling broadcast.

### Removed

- The inert process-global coordinate-ceiling shims `get_max_coord`,
  `set_max_coord`, and `init_from_env`, the backing `MAX_COORD` static, and
  their crate-root re-exports (libviprs#462). They were deprecated on
  arrival and consulted by no decoder; the single-axis ceiling lives solely
  on the per-decode `DecodeLimits::max_coord` field, enforced by every
  decoder — build it with `DecodeLimits::with_max_coord`. No `since` version
  applied because no released version ever exposed them as live API. The
  `DEFAULT_MAX_COORD` constant is retained as the `DecodeLimits::max_coord`
  default.

### Fixed

- A Resume/Overwrite run now takes the advisory run lock on the *union* of the
  directories it mutates — the resolved checkpoint root **and** the sink's own
  output dir — instead of only one. Locking a single directory left the other
  exposed: guarding just the checkpoint root reopened the issue #126 output-wipe
  hazard, guarding just the sink dir reopened the issue #276 checkpoint-flush
  race. The locks are taken in a deterministic order so contending jobs cannot
  each grab one directory and deadlock; acquisition stays non-blocking. The two
  entries are de-duplicated by *canonical* path, so an explicit `checkpoint_root`
  that names the same physical directory as the sink through a different spelling
  (`out` vs `./out`, relative vs absolute, `a/../out`, a symlink alias) collapses
  to a single lock rather than making the run acquire the one `.libviprs-job.lock`
  twice and refuse *itself* with `ResumeError::Locked`. That error's wording is
  generalised from "checkpoint root is locked" to "run directory is locked",
  since it can now name either guarded directory.
- `downscale_to` and the 2x box halver `downscale_half` (via
  `downscale_half_alpha`) now round the alpha-weighted colour and averaged
  alpha half-up, matching their own no-alpha branches, so a fully-opaque
  RGBA image downscales bit-identically to its RGB twin — and identically
  through both paths — instead of carrying the systematic -0.5 LSB
  truncation bias the alpha path previously introduced. Both alpha-weighted
  regions are also scanned with integer `u64` accumulators that stay exact
  where the former `f64` colour accumulator would drop low bits on extreme
  16-bit downscales.
- `composite2` mixed bit-depth blends now key the 0..65535 vs 0..255
  read/write scale on each input's *resolved* interpretation
  (`Raster::interpretation`, gated on the actual 2-byte storage depth),
  matching libvips' per-input `max_band` from `vips_interpretation_max_alpha`
  after `formatalike`. A genuine 16-bit layer (e.g. a decoded 16-bit PNG,
  untagged through `Raster::new`) is honoured on the 65535 scale so it blends
  against an 8-bit layer at a true ratio instead of 257:1 and its alpha is not
  capped at 255; a genuine-16 result is stamped `Rgb16` / `Grey16` so a
  re-composite reads it on the same scale (issues #443–#449). To keep the
  crate's promoted-container idiom working across the module boundary, the
  constant arithmetic ops (`add_const` / `mul_const` / `pow_const` /
  `add_vec`, and the widening binary path) now stamp their widened 16-bit
  output with the source interpretation, so an 8-bit input promoted into a
  16-bit container resolves to a non-genuine-16 space and a fully-opaque
  promoted overlay stays visible over an 8-bit base instead of collapsing to
  ~0.4%. The genuine-16 write-back scale and interpretation stamp are gated to
  integer output containers, so a genuine-16 input composited against a float
  raster no longer inflates the float output ~257x or mis-tags it as USHORT.
- A `Resume` refused on a plan-hash mismatch now returns
  `EngineError::PlanHashMismatch` before anything touches the output
  directory: the plan-hash gate runs ahead of the advisory run-lock
  acquisition, so a refused resume no longer materialises
  `.libviprs-job.lock` (or any other file) as a side effect.
- `RunLock` removes its `.libviprs-job.lock` file when the guard drops, so a
  finished run leaves no bookkeeping behind in the output tree and a
  crash+resume run now produces a pyramid byte-identical to a clean single
  run. The unlink happens while the exclusive lock is still held, and
  `RunLock::acquire` revalidates after locking that the path still names the
  inode it locked (retrying against the fresh file otherwise), which keeps
  the removal safe against concurrent acquirers.

## [0.3.1] — 2026-04-25

Documentation-only patch: the README and crate-root rustdoc shipped on
crates.io with 0.3.0 still showed the removed `generate_pyramid` /
`EngineConfig` entry points. 0.3.1 reships the same code with the README,
`src/lib.rs` rustdoc, `CHANGELOG.md`, and `MIGRATION.md` aligned with the
0.3.0 `EngineBuilder` API.

### Changed

- `README.md` — usage example uses `EngineBuilder` and 2-arg `FsSink::new`
  + `.with_format(...)`; modules and features tables extended for 0.3.0.
- `src/lib.rs` crate-root rustdoc — workflow step 3 references
  `EngineBuilder` / `EngineKind` / `EngineObserver`; cargo features
  documented (`pdfium`, `pdfium-static`, `s3`, `tracing`, `packfile`).

## [0.3.0] — 2026-04-25

The headline change in 0.3.0 is `EngineBuilder`: the five free-function entry
points (`generate_pyramid`, `generate_pyramid_observed`,
`generate_pyramid_streaming`, `generate_pyramid_mapreduce`,
`generate_pyramid_mapreduce_auto`) and `generate_pyramid_resumable` are gone,
replaced by a single typed builder that routes to the monolithic, streaming,
or map-reduce engines. `FsSink` also moved to a 2-arg constructor plus a
`with_format` builder.

See [MIGRATION.md](./MIGRATION.md) for before/after snippets for the most
common 0.2.0 call sites.

### Added

- `EngineBuilder` and `EngineKind` (`Auto`, `Monolithic`, `Streaming`,
  `MapReduce`) — single typed entry point that routes to every engine.
- `EngineSource` enum and `IntoEngineSource` trait so `EngineBuilder::new`
  accepts either `&Raster` or any `T: StripSource` directly.
- `extensions` module and `EngineBuilder::with_extension::<T>(value)` typed
  extension bag.
- `EngineError::IncompatibleSource` for engines that cannot accept the source
  kind they were handed (e.g. `Monolithic` with a strip source).
- `ResumePolicy` (factories `overwrite()`, `resume()`, `verify()`, plus
  `with_checkpoint_every` / `with_checkpoint_root`) wired through the builder
  via `with_resume(...)`. Honored by every engine — Monolithic, Streaming, and
  MapReduce.
- `RetryPolicy::fail_fast()` and `RetryPolicy::with_max(n)` convenience
  constructors; `FailurePolicy` and `RetryPolicy` are first-class on the
  builder via `with_retry(...)` / `with_failure_policy(...)`.
- `DedupeStrategy` exposed on the builder via `with_dedupe(...)` and on
  `FsSink` via `.with_dedupe(...)`.
- Full lifecycle `EngineEvent` variants: `SourceLoadStarted`, `SourceLoaded`,
  `PlanCreated`, `LevelStarted`, `LevelCompleted`, `StripRendered`,
  `BatchStarted`, `BatchCompleted`, `Finished`, `PipelineComplete` (alongside
  the existing `TileCompleted`).
- `EngineObserver` is threaded through every engine via
  `EngineBuilder::with_observer(...)` and `with_observer_arc(...)`.
- `PackfileSink::builder(path)` + `PackfileSinkBuilder` fluent form
  (`.plan(...).format(...).tile_format(...).build()`); `SinkError::MissingField`
  variant surfaced when a required field is omitted.
- `FsSink::with_format(TileFormat)` builder method.
- `stream_verify` module — verify pyramid output against the source.
- `pixel::PixelFormat` is now public and re-exported at the crate root (used by
  `EngineEvent::SourceLoaded`).
- Blanket `impl TileSink for &T` so `&FsSink` and friends work where a
  `TileSink` is required.
- `pdfium-static` cargo feature (pulls in `pdfium` plus
  `pdfium-render/static`) for statically linking libpdfium.

### Changed (breaking)

- `FsSink::new` is now 2-arg: `FsSink::new(dir, plan)`. Format is set via
  `.with_format(TileFormat)`; default remains `TileFormat::Png`.
- `EngineBuilder::with_config(EngineConfig)` is the supported way to override
  the full `EngineConfig`; the old per-call `&EngineConfig` argument is no
  longer accepted by free functions because the free functions no longer
  exist.

### Removed (breaking)

- Free functions `generate_pyramid`, `generate_pyramid_observed`,
  `generate_pyramid_streaming`, `generate_pyramid_mapreduce`,
  `generate_pyramid_mapreduce_auto` — use `EngineBuilder` with the
  appropriate `EngineKind`.
- `generate_pyramid_resumable` — absorbed into
  `EngineBuilder::with_resume(ResumePolicy::resume())`. The internal helper
  is now `pub(crate)`.
- `FsSink::new_with_format(...)` — kept as a deprecated alias of
  `FsSink::new(...).with_format(...)`; will be removed in a future release.

### Fixed

- `pdf_info` honors `/Rotate` when computing page dimensions.
- `PdfiumStripSource` now renders the full page once, then slices strips,
  fixing inconsistencies at strip boundaries on rotated pages.
- `RetryPolicy` backoff rounds nanoseconds to avoid floating-point truncation.
- Phase-3 filesystem tests are gated under Miri via
  `#[cfg_attr(miri, ignore)]`.

### Internal

- Build patches `pdfium-render` against the `libviprs/pdfium-render` fork.
- Version bumped to 0.3.0.

## [0.2.0] — 2025

Phase-3 hardening: manifest v1, sinks, resume, retry, dedupe, tracing.

[0.3.0]: https://github.com/libviprs/libviprs/releases/tag/v0.3.0
[0.2.0]: https://github.com/libviprs/libviprs/releases/tag/v0.2.0
