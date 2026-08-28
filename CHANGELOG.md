# Changelog

All notable changes to libviprs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- `resize`, `shrink`, `reduce` and `affine` take the premultiply bracket's
  alpha ceiling from the raster's interpretation instead of from its storage
  depth, so a float raster tagged `ScRgb` brackets against `1.0` and one tagged
  `Rgb16` against `65535` (issue #664). An untagged raster resolves to the same
  ceiling it always had, and the unsigned carriers stay on the depth rule
  deliberately, so an 8-bit buffer someone labelled `Rgb16` still premultiplies
  against 255 rather than coming back black.

  `max_alpha` was derived from the depth, which is right for the unsigned
  carriers by accident: an untagged `Rgba8` resolves to `Srgb` and 255, an
  untagged `Rgba16` to `Rgb16` and 65535. A float carrier has no depth-implied
  ceiling at all, so the tag is the only thing that can say what "fully opaque"
  means, and `colourspace(Interpretation::ScRgb)` hands back exactly that
  combination: an `RgbaF32` of scene-linear 0..1 samples, which the old rule
  bracketed against 255.

  `vips_resize` premultiplies nothing of its own, and the binary confirms it:
  the same float RGBA resizes to identical bytes under `multiband`, `b-w`,
  `srgb`, `scrgb` and `rgb16`. The bracket lives in `vips_affine`
  (`affine.c:553`) and `vips_thumbnail` (`thumbnail.c:835`), both of which
  reach it through `vips_premultiply` / `vips_unpremultiply`, and those default
  `max_alpha` from `vips_interpretation_max_alpha` (`header.c:195`). Measured
  on vips 8.18.6, an 8x8 constant float RGBA `(100, 20, 3, 1.5)` through
  `premultiply | resize 0.5 | unpremultiply` comes back `100 20 3 1.5` untagged
  and `66.666671752929688 13.333333969116211 2 1` under scRGB, and
  `affine "0.5 0 0 0.5"` on its own gives the same three-tag table because
  `vips_affine` calls the pair itself.

  You do not need an out-of-range alpha to see it. lanczos3 rings, so resizing
  a hard transparency edge pushes the resampled alpha above the source's
  maximum, and the stored alpha is clipped to the ceiling on the way out: a
  16x2 edge fixture puts `1.0152533054351807` in the output untagged and
  exactly `1` under scRGB. So committed reference images of a resized float
  scRGB raster with alpha will need regenerating.

- The premultiply bracket rounds through `f32` where it used to compute in
  `f64` and round once at the store, and that moves **unsigned** output bytes
  as well as float ones (issue #664, found while measuring the above).
  `vips_premultiply` writes `OUT nalpha = (OUT) clip_alpha / max_alpha` with
  `OUT` = float for every carrier this crate has (only a DOUBLE input widens to
  DOUBLE, `premultiply.c:229-232`), and multiplies the colour by that
  already-rounded value; `vips_unpremultiply` mirrors it with `OUT factor`. So
  the bracket rounds twice on an 8-bit RGBA exactly as it does on a float one,
  the same shape #631 found in the standalone pair.

  Measured against the code this branched from, on pseudo-random data, a 64x64
  `Rgba8` `resize(2.0)` moves 6 of 65536 samples and a 32x32 `Rgba16`
  `resize(0.5)` moves 4 of 1024, every one of them by a single count. Those are
  the ones that were wrong: against `premultiply | resize | unpremultiply` on
  vips 8.18.6, read back as FLOAT and quantised the way libviprs quantises, the
  new bytes agree on 65536 of 65536 and 1024 of 1024 where the old ones agreed
  on 65530 and 1020. The two 8-bit samples that move in the pinned fixture sit
  on `59.500003814697266` and `226.5` in vips's float, and the old `f64`
  expression put both a hair below and rounded them the wrong way.

  Float output moves by an ulp on the same fixtures, and `affine` moves with
  it. `vips_affine` premultiplies into a FLOAT image, interpolates that, and
  lets `vips_unpremultiply` read the FLOAT result back, so there are three
  `f32` rounding points on that path rather than two, and the interpolation
  itself accumulates in `f64` because `BILINEAR_FLOAT` uses `double`
  coefficients. All three are reproduced now, and the 8x8 constant table and a
  4x4 `affine "0.8 0.15 -0.15 0.8"` fixture are both bit-exact against the
  binary. Dropping any one of them shows: without the accumulator rounding, 9
  of that fixture's 64 samples move off the binary's value.

- Every conversion into `srgb`, `rgb16`, `b-w`, `grey16` and `hsv` now
  produces different output bytes, because the linear -> sRGB store goes
  through the libvips lookup table instead of evaluating the transfer
  function (issue #581). This is a parity fix, not a change of intent: the
  new bytes are the ones vips 8.18.4 produces.

  libvips never evaluates the sRGB curve per pixel. `calcul_tables`
  (`colour/LabQ2sRGB.c:126-146`) rounds `range` samples of it to integer
  codes once, in `float`, and `vips_col_scRGB2sRGB` (`:282-353`) and
  `vips_col_scRGB2BW` (`:385-428`) then interpolate linearly between two of
  those already-rounded entries and finish the chord with `rintf`, which
  rounds halves to even. Three quantisations, none of which an analytic
  `f64` encode has. Sweeping the neutral LabS L codes against the binary,
  `Labs -> b-w` used to differ on 5434 of 32768 and `Labs -> sRGB` on 16295
  of 98304, always by exactly one count and in both directions; both are now
  exact. On a 21x21x21 Lab grid `grey16` went from 1323 of 9261 to 0 and
  `rgb16` from 3223 of 27783 to 25, the remainder being an unrelated
  `f32`-versus-`f64` difference in the scRGB value itself, which only a
  16-bit carrier is fine enough to see.

  `sharpen` moves with it. It was carrying its own tolerance on the grounds
  that it convolves 16-bit LabS, but the deviation was never in the
  convolution: it was in the sRGB store the result comes back through. The
  same goes for two `colourspace` cells that were pinned at one LSB. All
  three are exact now and the tolerances are gone.

  If you have committed reference images produced by an older libviprs, they
  will need regenerating against vips rather than against the previous
  output.

- sRGB -> HSV truncates its hue and saturation codes instead of rounding
  them (issue #581, found while fixing the above). `sRGB2HSV.c:113-117`
  writes both into an `unsigned char`, and the C drops the fraction on that
  store; libviprs was handing them out unrounded and letting the writer
  round, which missed vips on about a third of the two bands (measured
  45370 of 138528 codes over a 46176-pixel sRGB grid, now 0). The hue's
  ratio is an `f32` division there too, which decides another 299 of them.
  It stayed hidden until now because the analytic sRGB encode produced flat
  greys where the table produces a real spread, and a flat grey has no hue
  or saturation to get wrong.
- `MetadataValue` is `#[non_exhaustive]` (issue #609). An exhaustive `match`
  on it downstream needs a `_ =>` arm now. Nothing else changes: the attribute
  is on the enum rather than on its variants, so `MetadataValue::Int(3)`, the
  `From` impls and the `as_*` accessors all keep working untouched.

  Four is not the number of types a vips metadata field can have. A `.v`
  trailer this crate reads today can carry `VipsArrayInt`, `VipsArrayDouble`
  and `gboolean` fields, which it can only forward opaquely, and #573 needs an
  array variant for the per-frame GIF delays. Adding that variant to an
  exhaustive enum is a major bump, and it would be a major bump for a reason
  nobody enjoys explaining. Doing it now, while the cost is one `_` arm, is
  the cheap moment, and it puts the enum where every other growable public
  enum in the crate already sits: `tests/non_exhaustive_enums.rs` registers 21
  of them and this one had simply escaped the list.

  Unlike the on-disk half of the same question (#565), this break is one
  `cargo semver-checks` can see, which is exactly why it was safe to leave
  until it was worth doing and why it is worth doing before the variant lands
  rather than after.
- Integer-precision convolution divides by `rint(kernel.scale)`, the scale on
  the mask that was passed in, where it used to divide by the
  brightness-corrected scale `vips__image_intize` derives from it (issue #547).
  `conv`, `convsep` and `compass` move output bytes wherever rounding the
  coefficients changes the mask's overall gain, which takes at least one
  coefficient that does not already round to itself, and they move on every
  carrier: uchar, ushort and the float-input arm alike. `sobel`, `scharr`,
  `prewitt`, `canny`, `gaussblur` and `sharpen` do not move at all, because
  every mask they build has integer coefficients over an integer scale and the
  correction is a no-op there.

  The correction was never the divisor. `vips_convi_gen` reads the scale and
  the offset off `convolution->M`, the mask the caller handed in
  (`convolution/convi.c:757-760`); `vips_convi_build` shadows `M` with the
  intized copy only for as long as it takes to harvest the integer
  coefficients (`convi.c:1179-1181`) and never writes that copy back. So the
  `out_scale` `vips__image_intize` computes is dead for `convi` and live only
  for the approximate paths this crate does not implement, `conva` and
  `convasep`. libviprs threaded it into the division, which is a different
  operation from the one libvips performs.

  It was not a rounding nit either. On `Kernel { data: [[3.0, 0.4, 0.4, 0.4,
  0.4]], scale: 1.0 }` the correction lands on `-1`, so a 5x1 grey field of
  100s came back `[0, 0, 0, 0, 0]` where vips 8.18.4 answers
  `255 255 255 255 255`. Black where the reference is white, from a mask any
  caller can build out of the public two-field `Kernel`. Measured over 126
  fractional masks on four fixtures, 191 of 504 outputs disagreed with
  `VIPS_NOVECTOR=1 vips conv --precision integer` before this change and 30 do
  after, and all 30 are the corner below.

  That corner is the one place the new divisor cannot follow libvips, because
  there is nothing there to follow. `vips_convi_gen` holds the scale in an
  `int`, so a mask scale below 0.5 leaves it at `0` and C divides by it:
  measured on 8.18.4 at scale 0.4, the two integer arms answer `0` (aarch64
  `sdiv` returns zero instead of trapping, which is not a defined result, and
  x86 would trap) and the float-input arm prints `inf`. libviprs nudges a
  divisor of zero to `1`, which is the guard `vips__image_intize` writes for
  its own copy at `convi.c:895-897` and the only total answer available. A
  caller who wants a sub-unit scale at integer precision should scale the
  coefficients instead, or use `Precision::Float`, which has no `int` in the
  path and divides exactly.

- `Raster::compass` refuses a `times` outside `1..=1000` where it used to
  accept anything above zero (issue #547, found in review). That is the range
  libvips declares on the property,
  `VIPS_ARG_INT(class, "times", 101, ..., 1, 1000, 2)` at
  `convolution/compass.c:162-167`. GObject refuses to *set* an out-of-range
  value, which is not the same as refusing the call. Measured on 8.18.4 with
  a 3x3 ones mask over a 4x4 black image,
  `vips compass a.v o.v m.mat --times 1` and `--times 1000` run, while
  `--times 0`, `--times 1001` and `--times 100000` each draw
  `value "N" of type 'gint' is invalid or out of range for property 'times'
  of type 'gint'` out of GObject. The CLI carries on at the property's
  default of 2 rather than exiting, so the out-of-range number never reaches
  a convolution in vips at all.

  Checking the low end only left the high end wide open, and `times` is a
  `u32` on this surface. `u32::MAX` reserved a result vector of 4.29 billion
  rasters, on the order of 400 GB of address space, and then started that many
  whole-image convolutions: no error, no ceiling, and nothing back inside half
  a minute. The refusal is the typed
  `ConvolutionError::TimesOutOfRange { times, min, max }`, which replaces
  `ConvolutionError::ZeroTimes` and covers both ends of the range at once.
  `ZeroTimes` has never been in a release, and the enum is
  `#[non_exhaustive]`.

  Worth being explicit that this is a divergence rather than a match, because
  the accepted range is identical and that makes it easy to skim past: vips
  hands back an image for `--times 1001`, computed at 2, and libviprs hands
  back an error. Convolving twice when a thousand rounds were asked for is a
  wrong answer wearing a warning, and the warning goes to stderr where a
  library caller never sees it. The `# Divergence from stock libvips` section
  in `crate::convolution` carries the measurement and the reasoning.
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

- `Raster::encode_jxl` takes a `jxl::SaveOptions` carrying a `Compression`,
  where it used to take a bare `lossless: bool` and always return
  `EncodeError::Unsupported` (issue #620). It now encodes, and the argument it
  used to take could only ever have meant one thing: there is no lossy JPEG XL
  encoder reachable in pure Rust, because `zune-jpegxl` is a lossless modular
  encoder with no VarDCT path anywhere in it. `encode_jxl(true)` becomes
  `encode_jxl(jxl::SaveOptions::default())` and `encode_jxl(false)` has no
  spelling at all, which is the point: `jxlsave`'s `distance`, `Q`, `tier` and
  `effort` have nothing behind them here, so none of them is a field this crate
  accepts and discards. `Compression` is `#[non_exhaustive]`, so
  `Lossy { distance }` can join it as a minor bump the day there is an encoder
  for it.

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

- `smartcrop(Attention)` picks a different crop on any image that carries an
  alpha band, and the pixels of `resize` / `reduce` / `shrink` / `affine` move
  wherever an alpha lands near zero (issues #603, #604). Both are parity fixes
  towards vips, so output that was wrong is now right, but output bytes for
  shipped operations do move and a caller pinning them will see it.

  `smartcrop` first. `vips_smartcrop_build` premultiplies once into float and
  hands the result to `vips_resize`, which explicitly does **not** premultiply
  ("This operation does not premultiply alpha. If your image has an alpha
  channel, you should use premultiply on it first", `libvips/resample/resize.c`),
  so the analysis image is still premultiplied when the argmax is taken and
  every transparent pixel is still at colour 0. libviprs' `resize` premultiplies
  on its own — a deliberate divergence from the C namesake, issue #458 — so it
  was un-premultiplying the already-premultiplied analysis image on the way out,
  the colour hiding behind transparent pixels came back amplified by
  `max / alpha`, and those bright fake regions dominated the edge and skin
  scores. On `rgba.png` at 80x60 that is the difference between (124, 84) and
  vips' (20, 124), which is now what you get. The fix drops the alpha band
  before the shrink, which is exactly what vips computes: it discards the band
  immediately after the resize anyway (`extract_band(0, "n", 3)`), and a
  resample that does not premultiply is per-band independent.

  That asymmetry is the real trap and it is now written down in the
  `resample` module docs, because it will catch the next operation that composes
  on `resize`: **never hand `resize` an image you have already premultiplied.**

  Second, the un-premultiply guards. libvips damps the factor to zero inside a
  dead zone, `factor = fabs(alpha) < 0.01 ? 0 : max_alpha / alpha`, and clips
  the alpha it stores with `VIPS_CLIP(0, alpha, max_alpha)`
  (`libvips/conversion/unpremultiply.c`). libviprs tested only `alpha == 0.0`
  and clipped neither end. That is not a theoretical gap: a lanczos resample
  undershoots, so an alpha dipping to 0.003 or going slightly negative at a hard
  transparency edge is ordinary, and dividing by it multiplies the colour by
  ~333 or flips its sign into a saturated result. The literal `0.01` is absolute
  in whatever units the alpha band carries and is deliberately not scaled by
  `max` — measured on 8.18.4, `alpha = 0.02` on a `(100, 100, 100, alpha)` pixel
  gives 5000 under scRGB, 1275000 under the 255 default and 327675008 under
  RGB16 — so it works out to `0.01 / 255` of full scale on the 8-bit and float
  carriers and `0.01 / 65535` on the 16-bit ones. The two guards stay separate,
  as they are in C: the factor divides by the raw alpha so that an alpha
  overshoot and the colour overshoot that came with it still cancel, and only
  the stored alpha is clipped.

  `premultiply` keeps no dead zone, and that asymmetry is libvips' rather than
  an omission: premultiply multiplies by the alpha, so a near-zero one damps
  instead of amplifying and there is no division to blow up. What it does have
  is the mirror-image clip — the normalising factor is built from a clipped
  alpha while the stored alpha stays raw, the other way round from
  un-premultiply — and that is now ported too, so the bracket cancels on a round
  trip. On the unsigned 8- and 16-bit carriers every one of these guards is
  inert, which is why nothing but the float resample paths moved.

- A pixel layout now has one spelling everywhere it is observed, so
  `PixelFormat::has_alpha`, `with_alpha`, `without_alpha` and
  `Interpretation::for_format` answer differently for the tuple spellings of a
  layout that also has a named variant (issue #531). `FloatF32(4).has_alpha()`
  was `false` and is now `true`; `Multi8(1).with_alpha()` was `Multi8(1)` and is
  now `Rgba8`; `Interpretation::for_format(FloatF32(4))` was `Multiband` and is
  now `Srgb`. `Raster::new`, `Raster::zeroed` and the decoders behind them store
  the canonical spelling, so `decode_exr` on an RGBA file reports `RgbaF32`
  rather than `FloatF32(4)`, and the manifest writes `"rgbaf32"` rather than
  `"floatf32:4"`.

  `PixelFormat`'s tuple variants are public, so `FloatF32(4)` is constructible
  and names exactly the layout `RgbaF32` names. `with_channels` canonicalises and
  direct construction did not, and nothing reconciled the two, so which answer
  you got depended on which spelling you happened to be holding. That is not
  only a wart in memory: `PixelFormat` is written into the persisted manifest,
  the writer emitted `"floatf32:4"` and the reader turned it back into
  `RgbaF32`, so the value read off disk was not the value written to it, hashed
  differently, and disagreed about alpha. Two places in this crate were already
  minting the alias: `decode_exr` built `FloatF32(n)` straight from the channel
  count, so a four-channel EXR reported no alpha while `resize` consults exactly
  that to decide whether to premultiply, and `invertlut` did the same from its
  column count.

  Reading is unchanged and deliberately so: `"floatf32:4"`, `"multi8:3"` and the
  rest still load, and still canonicalise, so a manifest written by an older
  build keeps working. Refusing them would have turned a silent mismatch into a
  hard failure on data already on disk.

  `PixelFormat::canonical` and `PixelFormat::is_canonical` are new and public.
  You need them only if you built a format yourself and want to compare it with
  one of ours, since `PartialEq` and `Hash` stay derived and so still tell the
  two spellings apart.

- `Extend::White` inks its fill from the raster's interpretation instead of
  from its sample depth, so a float raster tagged `ScRgb` fills with `1.0` and
  one tagged `Rgb16` with `65535` where both used to fill with `255` (issue
  #667). That covers `embed` and `gravity` in `extract`, and the white taps
  `affine` and the interpolating resamplers read outside the input. An untagged
  raster keeps the fill it always had.

  vips inks a white border with `(int) vips_interpretation_max_alpha(in->Type)`
  (`embed.c:280`), so the tag picks the ink and the depth never does. libviprs
  was reading the depth in two places that did not even agree with each other:
  `TapFetch::fill_value` gave 255 for every float carrier, and `embed` computed
  `bpc == 1 ? 255 : 65535`. Both go through one `white_ink` now, and
  `colourspace(ScRgb)` hands back exactly the raster that made it visible, an
  `RgbaF32` of scene-linear 0..1 samples that used to get a border 255 times
  too bright.

  There is a second half, and it is why the integer carriers move at all.
  `vips_region_paint` only writes that ink as a number when the image is float
  (`FILL_LINE(float, ...)`, `region.c:936`); on an integer image it `memset`s
  the buffer with it (`region.c:922`), which keeps the low byte of the ink and
  repeats that byte across every byte of the sample. On the ordinary tags it is
  invisible, since 255 memset over a `u16` is 65535 again, which is why a
  depth-derived ceiling has served this long. On scRGB it is very visible: the
  ink is 1, so a `u8` raster tagged scRGB fills with 1 and a `u16` one with
  `0x0101`, which is **257**.

  I ported the 257. It is not white in any sense and it is plainly an artefact
  of the paint mechanism, but it is what the oracle produces, and the other
  reading of the intent (clamp the ink into the carrier's range, giving 1) is
  not whiter, it is black. A `u16` buffer tagged scRGB is an incoherent thing
  to be holding in the first place, so neither answer serves anybody... and a
  port that quietly improves on the binary is one you can no longer check
  against it.

  Measured on vips 8.18.6, `vips embed in.v out.v 1 1 10 10 --extend white`,
  reading the corner:

  | carrier | multiband | srgb | rgb16 | grey16 | scrgb |
  |---|---|---|---|---|---|
  | uchar | 255 | 255 | 255 | 255 | 1 |
  | ushort | 65535 | 65535 | 65535 | 65535 | 257 |
  | float | 255 | 255 | 65535 | 65535 | 1 |

  Float `embed` stays unimplemented rather than newly wrong: `read_s` and
  `write_s` still panic on any depth that is not 1 or 2 bytes, so the float row
  is `resample`'s alone for now (issue #694).

  `vips affine --extend white` reproduces that table cell for cell **on a
  raster without an alpha band**, because it builds its resampling border with
  `vips_embed` (`affine.c:534`). It cannot once the raster carries alpha:
  `vips_image_hasalpha()` sends `vips_affine` through a premultiply into a
  **float** image before it paints the border, so `FILL_LINE(float, ...)` runs,
  the memset never happens, and the border lands on the plain interpretation
  maximum. Measured on 8.18.6 by solving the ink back out of a half-pixel
  bicubic shift over a constant input, since a plain identity never samples
  past the edge and shows no border at all:

  | bands | tag | alpha | `embed` | `affine` |
  |---|---|---|---|---|
  | 3 | `srgb` | no | 65535 | 65534.7 |
  | 4 | `srgb` | yes | 65535 | collapses to 255 |
  | 3 | `scrgb` | no | 257 | 256.0 |
  | 4 | `scrgb` | yes | 257 | collapses to 1 |
  | 1 | `b-w` | no | 65535 | 65534.7 |
  | 2 | `b-w` | yes | 65535 | collapses to 255 |

  The alpha rows differ in kind and not in degree, since `--extend white` and
  `--extend black` produce the same output there. libviprs paints the ink into
  the raster's own domain and premultiplies afterwards, so on an alpha raster
  it keeps the memset values, and `affine_white_on_an_alpha_raster_keeps_the_memset_ink`
  pins all three cells rather than leaving the gap implied. Moving the paint to
  the other side of the premultiply is a change to the ordering rather than to
  the ink, so it is issue #692's and not this one's.

### Added

- JPEG XL load and lossless save, behind a new non-default **`jxl`** feature
  (issues #500, #619, #620, #622). Build with `--features jxl` and `decode_jxl`
  reads both container forms, the bare `FF 0A` codestream and the boxed ISOBMFF
  one, `.jxl` becomes a live row in the content sniffer, in `Raster::save`'s
  extension route and in `Raster::encode_to_buffer`'s format route, and
  `Raster::encode_jxl` and `Raster::save_jxl` write the lossless modular form.
  The `Raster::encode_jxl(lossless: bool)` typed-`Unsupported` stub is gone.

  Without the feature nothing about the surface moves: every one of those entry
  points still exists at the same signature and returns a typed refusal naming
  the feature. `decode_jxl` reports `JxlError::FeatureNotEnabled`, both encoders
  report `EncodeError::Unsupported { format: "jxl" }`, and `.jxl` leaves the
  extension route entirely, so `save("x.jxl")` reports an unsupported extension
  like any other format with no encoder behind it. Consumer code compiles
  against either build.

  Decode goes to `jxl-oxide`, which targets the same JPEG XL conformance suite
  libjxl does, so this is a parity port rather than an approximation, and the
  two paths land in different places. The **lossless modular** path is a true
  identity against vips 8.18.4 for all three carriers, 8-bit, 16-bit and float,
  so its pins carry no tolerance at all. The **VarDCT** path agrees to within
  one count per channel and is pinned with exactly that and no more. vips also
  reads back what libviprs writes, at the same band counts and the same
  interpretations: `oracle-captures/foreign-jxl/` records both directions.

  The carrier follows the file rather than a fixed choice, the way
  `jxlload.c:679-696` picks one, so a 16-bit file comes back `Rgb16` and a
  float one comes back `FloatF32(3)` tagged `scrgb` instead of being quantised
  on the way in. A greyscale file stays one band, which is where JPEG XL and
  WebP part company: `webpsave` promotes `b-w` to three bands because the
  format stores no greyscale and `jxlsave` does not, because it does.
  `icc-profile-data`, `exif-data`, `xmp-data` and `bits-per-sample` come across
  under the field names `jxlload` uses.

  Two behaviours are worth knowing before you wire it in. The EXIF box needs a
  fix-up, because JPEG XL stores the TIFF block behind a big-endian 4-byte
  offset and without the `Exif\0\0` prefix a JPEG APP1 segment carries; the
  loader skips the offset and puts the prefix back, which is what makes a JXL
  `exif-data` blob compare equal to the JPEG one for the same image. And when
  that box is malformed, libviprs drops the blob and keeps the image where vips
  fails the whole load (measured: `vipsheader` exits 1 and prints nothing).
  Refusing an otherwise-valid image over a metadata box is the wrong trade for
  a decoder reading untrusted bytes.

  Save is lossless and nothing else, and there is no `quality` or `distance` to
  pass. `zune-jpegxl` is a lossless modular encoder with no VarDCT path
  anywhere in it, so `jxl::SaveOptions` carries a `Compression` whose one
  variant is `Lossless`, for the reason #568 gave for WebP: an argument the
  encoder throws away inverts the contract now and changes behaviour silently
  in a patch release later. `Compression` is `#[non_exhaustive]`, so
  `Lossy { distance }` can join it as a minor bump.

  16-bit encodes, unlike WebP, because the format and the encoder both hold
  16-bit samples and there is no narrowing question to answer. Float is refused
  with a message naming the remedy. There is one floor vips does not have:
  `zune-jpegxl` rejects a single-pixel row or column outright, where
  `vips jxlsave` writes an 18-byte 1x1 file happily, so `MIN_DIMENSION` is 2 on
  each axis and the refusal says so.

  No metadata is written on save. The encoder emits a bare codestream with no
  box container, so there is nowhere for an ICC profile, an EXIF block or an
  XMP packet to go, and `save` and `save_stripped` write identical `.jxl`
  bytes. `vips jxlsave --keep none` writes the same bare form; `--keep all` has
  no encoder behind it here. Animated JPEG XL loads frame 0 and reports
  `n-pages`, which is what a default `vips jxlload` does; reading every frame
  is #621 and waits on the page model in #564.

  The feature gate is there because of what the codec costs, and the two
  numbers that matter disagree, which is why it is worth spelling both out.
  `Cargo.lock` grows by 17 entries (260 to 277, measured), but a consumer's
  *compiled* graph grows by 21: `tracing`, `tracing-core`, `once_cell` and
  `pin-project-lite` were already in the lock through the `tracing-subscriber`
  dev-dependency and were not in anybody's build. Counting the lock is what
  undercounts it. Measured on `cargo tree -p libviprs -e normal`, a default
  build stays at 115 crates and `--features jxl` takes it to 136, and a release
  binary reaching the whole codec surface goes from 2,381,616 to 4,781,648
  bytes, +100.8%.

  One of the 21 is `tracing`, which `jxl-oxide` and `jxl-bitstream` both depend
  on unconditionally. Unconditional JXL therefore put `tracing` in the default
  graph of a crate whose own `tracing = ["dep:tracing"]` feature is deliberately
  opt-in and whose `default` is empty, and `default-features = false` could not
  get it back out. Behind `jxl` the opt-in holds again:
  `cargo tree -e normal -i tracing` finds nothing in a default build.

  All 21 are pure Rust, none with a `links =` key or a C compile. `jxl-oxide`
  is floored at 0.12.6 because every release at or below 0.12.5 carries
  GHSA-66m8-c62j-h6v5, an unchecked `usize` multiply in `FrameBuffer::new` that
  hands out oversized slices from an undersized buffer.
  `fuzz/fuzz_targets/fuzz_jxl.rs` and a 26-seed corpus ship with it.
- `JxlError`, the JPEG XL loader's own error type, reached through a new
  `SourceError::Jxl` variant (issue #634). JPEG XL was the only one of the three
  codecs in this release with no typed error of its own, so its refusals came
  back as `SourceError::Decode` wrapping an `image::ImageError` with a
  hand-spelled `"JPEG XL"` format hint, and telling a CMYK refusal from a
  truncated file from an over-budget one meant matching on the message text.
  That is exactly what `ExrError`, `FitsError`, `GifError` and `RadianceError`
  exist to avoid, and `JxlError` is the same shape: `#[non_exhaustive]`, struct
  variants with named fields, and an `#[error(transparent)] Raster(RasterError)`
  tail.

  Nine variants. `FeatureNotEnabled` for a build without the `jxl` feature,
  `Decode` for a bitstream `jxl-oxide` refuses, `Truncated` for one that simply
  runs out (the two-phase feed makes those different answers, and the variant
  names which of the header and the first frame was still missing),
  `CmykNotSupported` for a file with a black ink channel,
  `UnsupportedChannelCount` and `ChannelCountMismatch` for the two defensive
  channel checks, and `Raster` for a frame that cannot be wrapped.

  The two allocation refusals stayed separate rather than collapsing into one,
  and that is the change with teeth. `AllocLimitExceeded` is the crate's own
  ceiling, priced from the declared header geometry before the decoder reserves
  a thing, and it reports the geometry, the bytes needed and the budget.
  `DecoderAllocLimitExceeded` is `jxl-oxide`'s `AllocTracker` refusing an
  internal buffer part-way through, where the size is the decoder's business and
  never reaches us. Both used to arrive as the same
  `image::ImageError::Limits(InsufficientMemory)`, so the test covering them
  passed whichever one fired. Measured now that they are distinguishable: a
  4x3 file under an 8-byte budget answers with the tracker, because even a
  header's working buffers are over 8 bytes, while a 512x512 one under 256 KiB
  answers with the pre-check. Both are pinned, one per test.

  `JxlError` and `SourceError::Jxl` are declared whether or not the feature is
  on, so a caller's `match` has the same arms in either build and none of them
  names a type that is not there. Without the feature `FeatureNotEnabled` is
  the only reachable variant, and with it it is the only unreachable one, which
  is what lets a caller tell "this build has no JPEG XL" from "these bytes are
  not JPEG XL" without reading a message. `decode_jxl` used to report the
  feature-off case as an `Unsupported` I/O error, the way `crate::svg` still
  does; that is the one behaviour change here and it only affects a build
  without `jxl`.

  The encoder is deliberately not on this enum. `Raster::encode_jxl` and
  `Raster::save_jxl` stay on the shared `EncodeError` spine, which is where
  `gif`, `radiance` and `fits` leave their save refusals too, so JPEG XL does
  not become the one codec with a third convention.
- OpenEXR load: `decode_exr`, plus the sniff route so `decode_bytes` and
  `decode_file` reach it from the magic bytes rather than the extension
  (issues #504, #614 and #615). An `.exr` decodes to `FloatF32(n)` holding the
  file's own scene-linear samples, one band per selected channel, tagged
  `ScRgb` for an R/G/B selection and `Multiband` otherwise.

  **There is no save half and none is coming, because libvips has never
  shipped an EXR writer.** `vips -l` registers `openexrload` and no saver at
  all, and `vips copy src.png out.exr` answers `"out.exr" is not a known file
  format`. Nothing is deferred here; there is nothing to be parity with. Both
  facts are captured rather than asserted, in
  `oracle-captures/foreign-exr/oracle.json`.

  The load side goes further than vips does, and that is deliberate rather
  than accidental. `openexr2vips.c` drives the OpenEXR **C RGBA wrapper**
  (`ImfCRgbaFile.h`), which hands back four `half` samples per pixel and
  nothing else, so vips flattens every EXR before it sees a float. The file's
  own TODO block says so: "more of OpenEXR's pixel formats", "more than just
  RGBA channels", "best redo with the C++ API now we support C++ operations".
  Three measured consequences, all of which libviprs avoids:

  * A **FLOAT** channel comes back from vips rounded to half. Measured on a
    file holding `7/3`, vips reports `2.333984375`; libviprs returns the
    stored `f32`.
  * A file with **no R/G/B/Y channels**, a depth pass for instance, loads in
    vips as four bands of `(0, 0, 0, 1)`: an entirely black image, with no
    error and no warning. libviprs selects channels by name and returns the
    depth.
  * **Band count follows the file**, so an R/G/B file is three bands and a
    luminance file is one, where vips is always four with a synthesised alpha.
    The selected names come back as `exr-channels` so a band is never a guess.

  Parity with vips is nonetheless **exact, with no tolerance anywhere**: project
  a libviprs decode through that RGBA-half funnel and it reproduces the
  `vips rawsave` payload byte for byte on all twenty fixtures, lossy B44 and
  DWA codings included. The fixtures are written by the OpenEXR reference
  implementation 3.4.15, so no capture is circular.

  Known ceilings, stated at the entry point and not only here. **UINT channels
  do not load**: they need the unsigned sample carrier from issue #517, and
  `ExrError::UnsupportedSampleType` names it. vips does not refuse them, it
  converts them to half, so an object ID above 65504 reads back there as
  infinity. **Multi-part files decode their first part only**, which is also
  all vips can reach; the real count comes back as `exr-parts`, deliberately
  not the shared `n-pages`, because vips attaches no page count to an EXR and
  an EXR part is a layer rather than a page a caller could ask `decode_exr`
  for. **Deep EXR does not load** in either. Chroma-subsampled channels do not
  load. And a `FloatF32(n)` raster is rejected by the pyramid engine, as a
  loaded `.hdr` already is, so cast to an integer format first if you need
  tiles.

  The decode allocation budget, `DecodeLimits::max_alloc_bytes`, is priced
  off the channels the header **declares** and not off the bands the
  selection keeps. An EXR body is compressed and the decoder builds one
  full-resolution buffer per declared channel before it decompresses
  anything, so an ordinary compositing render declaring sixty-four channels
  costs sixty-four buffers however few of them survive selection. Pricing off
  the selection would under-count that by `declared / selected`, with nothing
  bounding the ratio.

  This costs **ten net-new lock entries**: `exr` 1.74.2 with
  `default-features = false`, plus `bit_field`, `lebe`, `libm`, `paste`,
  `pulp`, `pulp-wasm-simd-flag`, `raw-cpuid`, `reborrow` and `zune-inflate`.
  BSD-3-Clause with MIT / Apache-2.0 / BSD-3 / Zlib transitives, no `links =`
  key and no C source anywhere in the tree, and `exr` itself is
  `#![forbid(unsafe_code)]`. `image` 0.25's `exr` feature is exactly
  `dep:exr`, so naming the crate directly costs nothing extra and buys the
  channel list, the per-channel sample type and the data window, none of which
  survive `image::DynamicImage`.
- FITS load and save, hand-rolled with no new dependency: `decode_fits`, the
  `FitsError` it fails with, and `Raster::encode_fits` / `Raster::save_fits`,
  matching `vips fitsload` and `vips fitssave` (issue #505). `.fits`, `.fit`
  and `.fts` all reach it, through `Raster::save`, through
  `Raster::encode_to_buffer`, and through the content sniff, which never looks
  at the file name.

  FITS is worth writing out by hand because the container is a sequence of
  2880-byte blocks of fixed-width 80-column ASCII cards followed by a plain
  big-endian sample array, and the part that actually has to be right is the
  behaviour on the *vips* side, which no FITS crate models. There are three
  pieces of that. The scan order is bottom-up, because vips wraps the codec in
  a vertical flip in both directions. Bands are planes rather than interleaved
  samples, so `NAXIS3` names the band count and each band occupies a whole
  plane. And the header a save writes is generated by cfitsio rather than by
  vips, so libviprs spells those cards column for column; a file `vips
  fitssave` writes and a file `Raster::encode_fits` writes from the same pixels
  are byte-identical, and that is checked both ways against the reference
  binary rather than reasoned about.

  Header cards come back as `fits-0`, `fits-1` and so on in file order, and go
  back out on save filtered the way vips filters them: the cards cfitsio
  regenerates are not written twice, and a keyword is written once unless FITS
  lets it repeat. A header unit that declares no data is walked past, so the
  common layout of a metadata-only primary unit in front of an image extension
  loads, and the records you get are the loaded unit's.

  **Three of the six BITPIX values load; the rest are refused by name.**
  BITPIX 8 gives `Gray8` and its multi-band siblings, BITPIX 16 in the
  standard's unsigned spelling (`BZERO = 32768`) gives `Gray16`, and BITPIX -32
  gives `FloatF32`, with `BSCALE` / `BZERO` applied. What is missing is a
  carrier rather than the parsing: signed 16-bit is issue #516, 32-bit integer
  is issue #517, and double is issue #518. Those come back as
  `FitsError::UnsupportedCarrier` naming the sample kind and the issue, because
  narrowing a 16-bit array into 8 bits would lose data silently, which is worse
  than failing. BITPIX 64 is refused by vips too. The sample-kind spine
  (issue #607) is what lifts the ceiling.

  Save has no such ceiling and is total over `PixelFormat`, because vips's own
  promotion table already sends every signed integer format to its unsigned
  twin and libviprs has only the unsigned ones.

  The parser is bounded, which matters more here than for a binary container
  because FITS states its geometry in ASCII and a few dozen bytes can claim a
  gigapixel image: a cap on header blocks per unit, a cap on units walked
  looking for an image, the `NAXIS <= 10` ceiling vips applies, and the
  geometry checked through `DecodeLimits` before anything is allocated. There
  is a seeded fuzz corpus and a `fuzz_fits` target driving the decoder directly.

  Pinned against `oracle-captures/foreign-fits/`, captured on vips 8.18.4 built
  with `cfitsio: true` against cfitsio 4.6.4.

- Canny edge detection: `Raster::canny` and `Raster::try_canny`, matching
  `vips_canny` (issues #511, #559 and #560). It takes `sigma` and `precision`
  and nothing else, because libvips's canny **stops after non-maximum
  suppression**: it blurs, takes a 2x2 `[-1 1; -1 1]` gradient, converts to
  `(G, theta)`, thins along the gradient direction, and that is the whole
  operation. There is no double-thresholding and no edge tracking by
  connectivity, which is why there are no hysteresis thresholds to pass. Expect
  the result to look thinner and greyer than a textbook Canny, because it is a
  suppressed gradient magnitude rather than a binary edge map.

  The output format is the surprising part, so it is worth stating before you
  wire it into a pipeline: `precision` reaches only the blur, and the gradient
  stage then picks its own arm from the format of the *blurred* image. On the
  float arm the blur has already promoted a uchar input by then, so canny
  answers a float raster whose values run past 500 and do not fit a byte. A
  uchar input comes back uchar only at integer precision or below sigma 0.2,
  where the blur short-circuits to a copy. Everything 16-bit or float comes
  back float at every precision. Size, band count, interpretation, resolution
  and the attached metadata always round-trip.

  One deliberate divergence from the `vips` CLI: `vips canny --sigma 0` does
  not fail. GObject refuses any value outside `0.01..1000`, silently leaves
  sigma at its 1.4 default and still exits 0, so the CLI quietly runs a
  different blur than the one asked for. `try_canny` honours what it is given,
  as `try_gaussblur` already does, so a sigma below 0.2 is a no-blur request.

  Pinned against `oracle-captures/convolution/canny/`, which captured 42 vips
  8.18.4 outputs on both libvips paths. Where the two disagree libviprs is the
  portable C one, as issue #558 settled, and the suite says so at sigma 0.8 and
  1.6 rather than only at the 1.4 default, which is one of the few sigmas where
  the two implementations happen to agree.

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

- Every edit that adds a format to `src/source.rs` is checked by `cargo build`
  now, where two of the six used to fail silently (issue #633). It is still
  more than one edit, and worth being exact about which: the variant itself,
  its arm in `SniffedFormat::next`, the two lengths on `SniffedFormat::ALL`,
  and its row in `SniffedFormat::route`. What changed is that leaving any of
  them out stops the build. The two that used to be silent, the magic in
  `sniff` and the memory profile in `decodes_from_memory`, are not edits any
  more at all, because both are read off the row.

  Every container has a single row in a route table: the magic signatures
  `sniff` matches on, and the decoder the bytes go to. `sniff` walks
  `SniffedFormat::ALL` and reads the signatures off the rows,
  `decodes_from_memory` and `image_format` are derived from the row's decoder,
  and the chain of `if sniffed == Some(..)` arms at the top of
  `decode_bytes_with_limits` is gone, because the arm is the row.

  I reproduced the problem before fixing it rather than taking the issue's word
  for it. On `a356c50` I added an eleventh container the way a format lane
  would, wiring every site the compiler insists on and every list the tests
  count, and leaving out the two that are silent: the magic in `sniff` and the
  memory profile in `decodes_from_memory`. It compiled, and all 1794 tests
  passed, over a container nothing could ever detect and that would have been
  streamed to an `image` decoder it does not have. The same eleventh variant on
  this branch fails `cargo build` with two errors naming `SniffedFormat::next`
  and `SniffedFormat::route`, and the `ALL` length assert fires once those are
  filled...

  That it is `cargo build` and not `cargo test` is part of the change.
  `SniffedFormat::ALL` and `next` were `#[cfg(test)]`, so the library itself
  compiled happily with a variant nothing could reach. `sniff` walks `ALL` now,
  so the enum is load-bearing in an ordinary build.

  The magics are data rather than a hand-written chain, which is what lets
  `sniff` be driven from the table at all. Three shapes cover everything
  libviprs routes, because a signature is not always a leading prefix: WebP's
  `RIFF????WEBP` is split either side of a file-specific chunk length, and
  Radiance's `#?RADIANCE` is a whole first line rather than a prefix of one.
  Measured on vips 8.18.6, `#?RADIANCE\n` loads through `radload` while the
  near-misses `#?RGBE\n` and `#?RADIANCEX\n` both fall past it to `magickload`.
  That is `vips__rad_israd` (`radiance.c:568-577`) comparing the whole line,
  and it is what the `Line` shape encodes.

  Two tests carry the new guarantees. One builds the shortest head every
  signature accepts and runs it back through `sniff`, so a row with no magic, a
  magic `sniff` cannot match, a magic longer than the 16 bytes a file entry
  point ever reads, and a magic some earlier row shadows all fail. The other
  writes those same heads to disk and compares `decode_file` against
  `decode_bytes` for all ten containers, which is what pins the memory profile:
  a native codec whose row said "stream me" answers one way from a buffer and a
  different way from a path, and only the second answer is wrong. The old
  route-table test kept two hand-written lists of variants and both are gone,
  since a list kept by hand beside a table is the shape this is retiring.

  A third test names, per variant and by hand, which kind of decoder its row
  has to carry. That one is redundant with the table on purpose, because one
  row being wrong is a different failure from one row being missing: a missing
  row stops the build, a wrong row is consistent with itself. Swapping WebP's
  row for the streaming `image` facade bypasses `crate::webp` and everything
  issue #567 put there, and every other test in `src/source.rs` stays green.
  The suite as a whole does catch it, in `webp::tests`, and I checked it
  catches the same swap on every other row too, so nothing was going to merge
  silently... but the red landed three modules from the edit that caused it.
  Now it lands beside the table. The `match` inside is exhaustive, so a new
  container has to be named there or the crate does not compile.

  `Magic::matches` grew three `debug_assert!`s for the shapes that would
  otherwise be self-consistent and wrong: an empty `Prefix` matches every
  buffer and would shadow every row declared after it, and a `Split` whose
  prefix runs into its own tag builds the very probe that then matches it. The
  public doc on `decode_file_with_limits` names the containers held whole
  again, too, rather than pointing a caller sizing `max_alloc_bytes` at a
  routing table that is `pub(crate)` and renders nowhere, and the same new test
  pins that list so the prose cannot drift.

  Nothing about detection or decoding moves. Same signatures, same decoders,
  same answers. The only ordering change is that FITS is tried before OpenEXR
  now because that is their declaration order, and their signatures share no
  bytes. One live doc drift went with it: `image_format`'s doc named five of
  the seven containers libviprs decodes itself, never having been updated when
  FITS and OpenEXR landed. It names none of them now, because the row says.

  `crate::imageio::is_vips_bytes` is gone and `VIPS_MAGIC_LE` / `VIPS_MAGIC_BE`
  are `pub(crate)` in its place, so the `.v` signature is owned by the module
  that owns the container, the way `exr::MAGIC`, `fits::MAGIC` and
  `radiance::MAGIC` already were. Everything here is crate-internal, so no
  public API moves.

- `n-pages` has one documented meaning, and `Raster::get_n_pages` now ports the
  whole of the libvips sanity check that guards it (issue #635). The panel that
  filed the issue counted four meanings behind the one accessor. Re-measured
  after #626 moved the OpenEXR multi-part count out to `exr-parts`, there is one
  meaning left and four loaders honouring it: `n-pages` is how many pages the
  original **file** holds, where a page is something a zero-based `page`
  argument can select. GIF counts frames, TIFF counts IFDs, WebP and JPEG XL
  count frames in the original, and every one of them agrees with the vips
  loader it ports, on the value and on whether the field is attached at all.
  `vipsheader -a` on 8.18.6 reports `n-pages: 1` for a still GIF and a one-page
  TIFF and nothing at all for a still WebP or a single-frame JPEG XL, which is
  exactly what libviprs does. So the answer is one shared key rather than
  per-format ones, and a count that no page index can reach keeps getting its
  own name the way `exr-parts` did.

  What actually changes for a caller is the accessor. `vips_image_get_n_pages`
  (`iofuncs/header.c:917-928`) reports a single page unless the stored field is
  an int strictly between 1 and 10000; libviprs only had half of that, accepting
  anything positive and additionally parsing a string-typed field. Measured
  against the C on 8.18.6, a stored `9999` reads back as `9999` while `10000`,
  `65536` and `2000000000` all read back as `1`, and a `gchararray` `"3"` reads
  back as `1` because `vips_image_get_int` does not coerce one. `get_n_pages`
  now matches on all of those. The ceiling is reachable rather than theoretical:
  `DecodeLimits::max_pages` defaults to 100000, so a TIFF with 12000 IFDs
  decodes fine here and used to report 12000 where vips reports 1, and the same
  goes for a GIF or an animation with that many frames. Nothing is lost, only
  moved: the field itself is never rewritten, so `get_field("n-pages")` still
  hands back the real number and `tiff_page_count` still walks the chain. The
  string arm has no producer in the crate at all, since every loader stores an
  int and the `.v` trailer round trip preserves the `gint` type.

  The PDF readers still attach no `n-pages`, and that is now written down with
  its reason rather than left as a silence: vips's `pdfload` does attach one
  (measured: 3 for a three-page document), but its `page` is zero-based where
  this crate's PDF page numbers are deliberately one-based, so a caller sweeping
  `0..get_n_pages()` would be off by one. `PdfInfo::page_count` is the count for
  a PDF.

  Two doc blocks in `encode_tiff` that promised the count travels back on the
  raster and reads out of `get_n_pages` are corrected rather than left to
  describe the old behaviour. They now point at `tiff_page_count` as the
  uncapped page count for a TIFF and say where the accessor caps, and the
  matching promises in the WebP and JPEG XL loader docs say the same. A `0..n`
  sweep is still safe on a long chain, because the capped answer is 1 rather
  than something longer than the file.

  `get_n_pages` and `get_int` also stopped deep-copying to read a number.
  Both resolved through `get_field`, which hands back an **owned**
  `MetadataValue` cloned out of the field list, and any name can hold a
  `Blob`: `try_set_field` stores whatever type it is given outside the
  built-ins, and the `.v` trailer restores arbitrary named fields with
  arbitrary types out of an untrusted file. Measured in release with 64 MiB
  under the key, `get_n_pages` cost 1.296 ms and a 64 MiB alloc-and-free per
  call, against 2 ns now; with an ordinary `Int` under it the same call went
  from 18 ns to 2 ns. Both accessors borrow the stored value now, and a
  counting global allocator in `tests/n_pages_meaning.rs` asserts zero
  allocations across each call so a regression fails on the mechanism rather
  than on a timing threshold.

- `SaveError::UnsupportedExtension`'s message names the extensions the build
  in front of you can actually write, instead of a fixed list (issue #500).
  It used to end "libviprs encodes png, jpg/jpeg, gif, webp, and v/vips",
  which stopped being true the moment `.jxl` became a live save arm:
  `save("x.avif")` told you JPEG XL was unsupported at the moment it became
  supported. The tail is now computed from the same set the extension route
  dispatches on, so `jxl` appears exactly when the `jxl` feature is on, and a
  test parses the list back out of a rendered message and saves under every
  name in it, so a future arm that forgets the message fails rather than
  drifting. Anything matching on the exact string will need to stop; the
  variant and its `extension` field are unchanged.

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

- The six decoders that price a frame buffer from declared geometry (GIF,
  Radiance, FITS, OpenEXR, JPEG XL and the TIFF page reader) do it with one
  shared, saturating `width * height * bands * sample_bytes` and hand the answer
  to one shared comparison, instead of five near-copies of the same arithmetic
  that did not agree on what to do when the product overflows (issue #632). #612
  shipped that comparison as `DecodeLimits::check_alloc` in the same batch that
  added the FITS, OpenEXR and JPEG XL loaders, and none of the three used it,
  because all three were written in parallel against a `main` that did not have
  it yet. WebP still does neither half and is deliberately left alone: it prices
  off the decoder's own `output_buffer_size()` rather than a declared-geometry
  product and reports `SourceError::Decode` carrying an `image` `LimitError`, so
  nothing in the shared pair fits it.

  The one that actually diverged is JPEG XL. It saturated the first three
  multiplicands in `usize` and only then widened, so on a 32-bit target the
  sample count pins at `u32::MAX` before the sample size is applied, and no
  frame can be priced above `u32::MAX * sample_bytes`, about 16 GiB, however
  large the header says it is. Reaching that wants `max_pixels` above 2^32 and
  `max_alloc_bytes` above ~8.6 GB, both far past their defaults, on a target
  with a 4 GiB address space, so no such decode was ever going to succeed: what
  differed is which typed refusal came back, the budget's or the allocator's. On
  a 64-bit build `usize` is `u64` and all five spellings agree to the byte, so
  nothing moves for anyone on x86_64 or aarch64... it was a latent divergence
  rather than a live one, and it is the exact hazard `Raster::buffer_len`
  already documents and guards with `checked_mul` two functions further up.

  GIF and Radiance were the other two shapes, both a plain `*` with no
  saturation at all. Neither can overflow today, but only because a GIF states
  its logical screen in `u16` and `parse_resolution` bounds a `.hdr` axis below
  `DEFAULT_MAX_COORD` before `DecodeLimits` is consulted. Neither guarantee is
  written anywhere near the expression that leans on it, and the three codecs
  that copied the shape do not have one: their axes are `u32` and both ceilings
  above them are caller-settable, which is what turned a safe idiom into an
  unsafe one on the way across.

  The typed per-format variants stay. `FitsError::AllocLimitExceeded`,
  `ExrError::AllocLimitExceeded`, `JxlError::AllocLimitExceeded`,
  `GifError::AllocLimitExceeded` and `RadianceError::AllocLimitExceeded` are all
  still what a caller sees, because collapsing them onto
  `SourceError::AllocLimitExceeded` is a breaking change to five public enums;
  #632 deferred it and issue #686 carries it for 0.5.0. They are built from the
  budget's answer rather than retagged off its error, through the new
  `DecodeLimits::exceeds_alloc_budget`: `check_alloc`'s `what` label is only
  ever observable through a decoder that propagates the `SourceError` whole,
  which is the file-body read and the TIFF page reader, so the five formats that
  rebuild the message were constructing a label nobody could see.

  A saturated price is now refused whatever the budget says. `check_alloc` was a
  plain `needed > max_alloc_bytes`, which is false when both sides are
  `u64::MAX`, and `with_max_alloc_bytes(u64::MAX)` is the idiomatic spelling of
  "no limit" against a public unclamped field. So the one value saturation
  produces was the one value that budget waved through, and OpenEXR would then
  have sized its buffer from a plain `usize` product of the same untrusted
  geometry. `u64::MAX` is a sentinel meaning "this did not fit a `u64`", not a
  price, and the comparison treats it as one. The arm costs exactly one
  accepted value, and that value is 16 EiB.

  GIF, OpenEXR and Radiance also size their output buffers through
  `raster::buffer_len` rather than a bare `usize` multiply. Clearing the budget
  says the byte count fits a `u64`, which on a 32-bit target is not the same as
  fitting the address space: a caller who raises `max_alloc_bytes` past 4 GiB
  clears the check and then wraps the product two lines lower, in release. Same
  defect as the price, one line down, and it now answers
  `RasterError::SizeOverflow` on both widths. JPEG XL's sample count moves the
  same way, which also retires a comment claiming the pre-check made a `usize`
  chain safe; it only does so while `max_alloc_bytes <= usize::MAX`, and nothing
  enforces that.

  Two of the existing budget tests could not fail for the reason they claimed.
  The TIFF one decoded a 64x64 **gray8** page, where the band count and the
  sample depth are both 1, so it priced the same whether or not the check saw
  them, and that check is the only one of the three ceilings that can see them
  at all. The JPEG XL one had the same hole on the sample size, with a 512x512
  `Rgb8` frame. Both now carry a second case on a wider carrier (a 64x64 RGBA
  16-bit TIFF page at 32768 bytes, a 256x256 `Rgb16` JPEG XL frame at 393216)
  where dropping either multiplicand changes the answer. Every format also
  pins the budget at exactly the byte its geometry costs and one byte below it,
  which is what fixes the comparison at `>` rather than `>=`, and the overflow
  boundary itself is pinned once on the shared price rather than three times in
  three dialects. The sentinel gets its own boundary test, offering a saturated
  price to a `u64::MAX` budget and pinning both the refusal and the one value
  below it that is still accepted.

- `try_premultiply` and `try_unpremultiply` handle float rasters instead of
  panicking on them (issue #631). They used to fall into `depth_max`'s "the
  arithmetic operations do not support float rasters yet" panic from inside
  the fallible form, which is the one thing a `try_` method is not allowed to
  do, and the panicking twins then panicked with a reason that had nothing to
  do with their own contract.

  The panic was always reachable, but this release made it easy: OpenEXR and
  FITS both hand back float pixel data straight out of a file, so loading an
  EXR and calling a premultiply helper now hits it on ordinary input rather
  than on a raster you built on purpose.

  I implemented the float carriers rather than refusing them with a typed
  error, because there was nothing left to guess at. libvips defines both ops
  on float and this build runs them, `unpremultiply_factor` already carried
  the dead-zone and alpha-clip rules from #611, and the resize path has been
  doing the same arithmetic internally since #604. All of it checks against
  the binary, so I pinned it there rather than inventing a refusal.

  Three things about the float arm are worth knowing. Its `max_alpha` comes
  from the raster's `Interpretation` and not from the sample depth, the way
  `vips_interpretation_max_alpha` supplies it, so an scRGB raster (what an RGB
  OpenEXR load is tagged) divides by `1.0` where an untagged one divides by
  `255` and an `Rgb16`-tagged one by `65535`. Get that wrong and an EXR's 0..1
  samples premultiply to roughly black. The arithmetic runs in `f32` rather
  than `f64`, because the C macros land the multiplier in a `float` before the
  colour multiply, so the result rounds twice: `(100, 100, 100, 0.5)` comes
  out `0.19607845` through the float intermediate and `0.19607843` without it.
  And NaN and the infinities propagate the way `VIPS_CLIP`'s plain ternaries
  make them, so a NaN alpha gives a NaN pixel instead of being quietly
  rewritten.

  Both ops keep the input format, so an unsigned raster still comes back
  unsigned, rounded and saturated, and the arithmetic on that path is
  untouched. vips itself always writes `FLOAT` output here, and that
  divergence is unchanged and now written down on both methods.

  One thing on the unsigned path *did* change, and it is worth stating rather
  than filing under "nothing": both ops now copy the input's interpretation
  onto the output. An `Rgba16` explicitly tagged `Srgb` used to come back
  resolving to `Rgb16`, because the result was left untagged and a 4-band
  16-bit buffer resolves to the genuine 16-bit space. It comes back `Srgb`
  now. That is the correct answer and it is what vips does: measured on
  8.18.6, `vips premultiply` and `vips unpremultiply` both hand a `1x1 ushort,
  4 bands, srgb` input straight back as `srgb`, and a `multiband` one as
  `multiband`, because `vips_premultiply` copies the header. The tag matters
  downstream, since `composite2` keys its 0..255 against 0..65535 scale on the
  resolved interpretation, so this is a behaviour change rather than
  bookkeeping.

- `try_colourspace` no longer aborts the process when it cannot allocate its
  output, so both ends of the LabS round trip `try_sharpen` opens and closes
  report the failure instead of taking the process down with them (issue #672).
  There is a new `ColourError::Raster` variant carrying the `RasterError` that
  says why.

  Every colour result is one image-sized `Vec<u8>`, and every one of them was a
  plain `vec![0u8; ..]`. An over-capacity request there reaches
  `handle_alloc_error`, which ends the process instead of returning, and no `?`
  catches an abort. So `try_colourspace` handed back a `Result` that did not
  cover the failure a caller most reasonably assumes it covers, and
  `try_sharpen` inherited that however its own signature read: it converts to
  LabS on the way in and back on the way out, so both ends of it were the same
  abort. #627 is the same problem one module over, in the `raster.rs` widening,
  and it descoped this round trip on purpose, because the abort was not in
  `convolution.rs` or `raster.rs` at all.

  Both image-sized sites now reserve through `Vec::try_reserve_exact` and
  report `RasterError::AllocationFailed`: the conversion buffer the
  `try_colourspace` loop writes samples into, and the quantisation buffer the
  colour-difference and ICC arms finish through. `ColourError` is
  `#[non_exhaustive]`, so the new variant is additive and a downstream match
  with the wildcard arm the attribute asks for keeps compiling. The panicking
  twins, `colourspace`, `de76`, `icc_import` and the rest, keep panicking on
  it, which is what they do with every other `ColourError` and which at least
  unwinds where the abort did not.

  The wrap that follows the allocation moved to the op-output constructor at
  the same time, so a legal widening conversion is no longer rejected for
  exceeding the 8 GiB construction budget. `Srgb -> Lab` turns 8-bit bands into
  `f32`, a 4x, and an input at the budget ceiling produced an output over it;
  `Raster::new` refused that and the `.expect` around it turned the refusal
  into a panic out of a `try_` form. An op output derives from an input that
  was budget-checked at its own construction, which is the whole reason
  `Raster::from_op_output` exists (issue #279).

  The remaining infallible allocations in `colour.rs` are the `Vec<f64>` sample
  staging on the colour-difference path and the ICC fallback buffers. None of
  them is on the `try_colourspace` route, and each needs its own way to be
  driven honestly, so they went to issue #685 rather than being converted here
  on the assumption that they are reachable. They are converted in this same
  release, in the entry below.

  **This does not make `try_sharpen` abort-free**, and the claim is deliberately
  narrower than that. Its own body still widens through `Raster::f32_samples`
  and still keeps five image-sized `vec![]` and `clone` scratch buffers of its
  own, so an allocation failure in any of those ends the process before it can
  be reported. That set is issue #627's, PR #669 is open against it, and
  `try_sharpen`'s `# Errors` now names the five sites so a caller reading the
  API docs gets the same answer. What changed here is only the two `colour.rs`
  allocations the round trip reaches, which is all #672 was ever about.

- The fourteen image-sized buffers the colour-difference and ICC paths allocate
  are now reserved fallibly, so `try_de76`, `try_de00`, `try_de00_sharma`,
  `try_de_cmc`, `try_icc_import_with`, `try_icc_export_with` and
  `try_icc_transform` report a host that cannot serve one of them as
  `ColourError::Raster` instead of aborting the process (issue #685). No new
  public error variant: this reports through the one #672 added.

  **This does not make the ICC paths abort-free**, and the claim is deliberately
  the fourteen sites rather than the call. On a LUT profile both directions hand
  the pixels to a moxcms transform, and three katana stages inside it size
  intermediates from the image and allocate them with a plain `vec![]`
  (`conversions/katana/md3x3.rs`, `md4x3.rs` and `md_nx3.rs` in 0.8.1), so that
  route still reaches `handle_alloc_error`. moxcms is a required dependency with
  `any_to_any` on, and `any_to_any` is exactly what turns the katana engine on,
  so this is the default build and not a corner. The fallible spelling already
  exists upstream, a `try_vec!` over `try_reserve_exact` returning
  `CmsError::OutOfMemory`, and those three stages just do not use it; issue #693
  tracks the fix there. The matrix-shaper and grey-TRC routes evaluate in this
  crate and never reach any of it, and the module docs now carry the same
  boundary so the API and the CHANGELOG say one thing.

  #678 made `try_colourspace`'s output fallible and deliberately converted only
  the sites on that route. Fourteen image-sized allocations were left over
  everywhere else in `colour.rs`, spelled `Vec::with_capacity`, `vec![0.0; ..]`,
  `collect()` and `clone()`, and every one of them reaches `handle_alloc_error`
  on a request the host cannot serve, which no `?` catches. The largest is the
  colour-difference plane at an `f64` a sample, so a dE asked for more memory
  than either of the two Lab conversions ahead of it and asked for it
  infallibly... the `try_` form's `Result` covered the small allocations and not
  the big one. The ICC paths carry four or five each: the normalised device
  plane, the `Vec<[f64; 3]>` Lab staging on both directions, the `f64` sample
  buffer, the copy `try_icc_export_with` takes of an already-Lab input, and the
  two moxcms buffers each LUT-profile fallback fills.

  All fourteen now reserve through `Vec::try_reserve_exact` and report
  `RasterError::AllocationFailed`, via one `alloc_colour_plane` helper that
  prices a plane the way `Raster::new` prices a buffer, so a geometry whose
  element count does not fit a `usize` comes back as `SizeOverflow` on a 32-bit
  target instead of a wrapped product. The copy goes through
  `Raster::try_clone`, which exists for exactly this. The panicking twins keep
  panicking, as they do on every other `ColourError`.

  Reserving once and pushing is only abort-free while the reserve and the fill
  agree, and the four ICC conversions are the ones where they could drift: they
  size a plane from a `(width, height)` and fill it from a slice they were
  handed, which are two independent inputs. Each of the four now opens with a
  `debug_assert_eq!` tying the slice back to the geometry, because a `push` past
  the reservation grows through the infallible path on the largest buffers in
  the module and every allocation test would still pass, since those starve the
  reserve rather than filling it.

  Testing this is the whole difficulty and it is worth writing down, because a
  byte ceiling cannot reach any of it. Both dE operands convert to Lab first,
  and after #678 those conversions are themselves fallible, so any ceiling low
  enough to starve the difference plane returns from the first
  `try_colourspace` and the check goes green having never run the line it
  names. #678 hit the same wall on `try_sharpen` and answered it with a counter
  on the existing hook: wave the first `n` over-ceiling requests through, then
  refuse. That counter is what these tests use, so `spare` is the index of the
  site along the path and the byte count in the resulting error says which
  buffer it was. The fixtures carry an extra band on purpose, so that every
  site on a path has a size no neighbour can produce and an assertion cannot
  be satisfied by the wrong allocation.

  One pair cannot be separated that way. The export fallback's PCS buffer and
  its device buffer are both three-and-one-ink f32 over the same pixels, and
  the only device space the suite can build a profile for is RGB, so both are
  the same size. They are covered jointly by a check that counts the refusals
  the function offers up instead of sizing them, which is what notices if
  either site quietly goes back to an infallible `Vec::with_capacity`.

  The ceiling has a blind spot of its own, and it took the review to find it: it
  answers *before* `try_reserve_exact` runs, so every one of those site checks
  stays green with the reservation put back to an infallible `reserve_exact` and
  the copy back to `Clone::clone`. The whole change undone, 1562 lib tests
  passing. They pin the routing, which is worth having, and say nothing about
  the helper being fallible. Two checks say that directly now.
  `colour_plane_allocation_reports_failure_rather_than_aborting` asks
  `alloc_colour_plane` for a 512 PiB plane with no ceiling in play, so the
  refusal is the real allocator's and the infallible spelling aborts on it. The
  export's copy of an already-Lab input cannot be reached that way at any size a
  test can build, so `raster.rs` keeps a `cfg(test)` counter on
  `Raster::try_clone` and the export check counts the delegation instead of
  starving it.

  On the zeroing cost #672's entry records: twelve of the fourteen dodge it
  entirely, because they reserve and then push or copy and never touch a byte
  they do not write. The two moxcms buffers do pay it. `vec![0.0f32; n]` hits
  std's zero specialisation and lowers to `alloc_zeroed`, and reserve plus
  `resize` is a `malloc` and a full `memset`, so those two acquire the same
  34%-at-4-GiB regression `alloc_colour_output` documents, and the fill is dead
  in both, since the transform writes every element. Same follow-up (#460),
  same reason: std has no fallible zeroed `Vec` today.

- `try_recomb`, `try_stdif`, `try_bitand`, `try_bitor` and `try_bitxor` return
  `ArithmeticError::FloatUnsupported` on a float raster instead of panicking
  (issue #631). They reached the same `depth_max` panic the alpha pair did, on
  the same input: an OpenEXR or FITS load is a float raster, so
  `decode_file("x.exr")?.try_recomb(&m)` took the process down. If you were
  matching on the error you get one now; if you were relying on the panic, you
  were relying on a bug.

  These five refuse rather than compute, where the alpha pair computes, and
  the reason is that vips gives no float answer to port for four of them. I
  measured all three families on 8.18.6 rather than assuming: `vips_boolean`
  casts a float operand to `int` before the bitwise op and never operates on
  float at all (`(100.5, 100.5, 100.5, 0.5)` AND itself comes back as an
  `int` image of `100 100 100 0`), and `vips stdif` refuses anything that is
  not `uchar`, `ushort` included. `recomb` is the exception: vips does compute
  it on float and keeps it float, so libviprs is deliberately narrower there,
  because this port writes into the input depth and a float carrier has no
  unsigned spelling of that. It is written up on the method.

  What closes this properly is not the five fixes but the test behind them: a
  property test now calls every `try_*` method in `arithmetic.rs` on an
  `RgbaF32` raster and fails if any of them unwinds, with a companion check
  that reads the module's own source and fails if a `try_` method exists that
  the sweep does not call. A sixth one cannot arrive quietly.

- `decode_file` bounds the whole-file read it does for the formats that are
  decoded from memory, so `DecodeLimits::max_alloc_bytes` is now in the path
  of that read instead of being consulted after it (issue #629). `.v`, JPEG,
  GIF, WebP, JPEG XL, Radiance, FITS and OpenEXR all need the bytes
  addressable end to end rather than streamed, and the read that got them was
  a plain `std::fs::read`. That sizes its buffer from the file and then grows
  infallibly, so `max_coord`, `max_pixels` and `max_alloc_bytes` were every
  one of them consulted after the whole file was already resident, and on a
  constrained host the failure was a process abort rather than a returned
  error.

  So the read had no ceiling at all, and a file could name any size it liked.
  A 3 GiB FITS declaring a 4x3 image decoded successfully at 3.01 GiB
  resident under the default 512 MiB ceiling; the same file is now refused at
  6 MiB resident with `AllocLimitExceeded { needed_bytes: 3221225472,
  max_alloc_bytes: 536870912 }`. That is the whole of what changed: the worst
  case went from unbounded to `max_alloc_bytes`, and the failure went from an
  abort on a constrained host to a returned error everywhere.

  The read now stats first and refuses anything longer than
  `max_alloc_bytes`, then caps the read as well, so a source that yields more
  bytes than its `stat` declared is refused rather than silently truncated
  and handed to the decoder as a whole file. It is the same
  `read_file_bounded` the TIFF page readers have used since #612, lifted into
  `source` so both call sites share one implementation rather than drifting
  apart.

  **What this does not do is make a cheap file cheap to decode.** The ceiling
  is a byte count, not a ratio between what a file costs to store and what it
  costs to decode, so anything under it is untouched. Measured on APFS, where
  a file grown with `set_len` leaves the tail as a hole, so it declares 400
  MiB and occupies 8 KiB: it decodes to a 4x3 image at 406 MiB resident, and
  that number is the same before and after this change. At exactly the
  ceiling it is 518 MiB resident from the same 8 KiB on disk. If you serve
  untrusted files, `max_alloc_bytes` is now the number that bounds what one
  decode can cost you, and the default 512 MiB is a lot to hand a file you
  did not write.

  **This can refuse a file that used to decode.** A file longer than
  `max_alloc_bytes` in one of the formats above is now
  `SourceError::AllocLimitExceeded { what: "image file body", .. }`, which
  is the same variant the declared-geometry checks already raised, so
  nothing has to tell "too big by header" from "too big by file length".
  Raise the ceiling with `DecodeLimits::with_max_alloc_bytes` if you
  legitimately load files bigger than that. The ceiling is inclusive, so a
  file of exactly `max_alloc_bytes` still decodes. The streaming decoders are
  untouched: they never read the whole file, so bounding them by its length
  would refuse work that costs nothing.

- `MemoryTracker::alloc` saturates at `u64::MAX` instead of overflowing (the
  `alloc` half of issue #114, which fixed the same thing on `dealloc`). It read
  `self.current.fetch_add(bytes, Relaxed) + bytes`, and that `+ bytes` panics
  with "attempt to add with overflow" in debug builds and under Miri once the
  counter is high enough, which is not a thing an observability counter should
  do to a run. Saturating the local sum alone would have been worse than the
  bug: `fetch_add` wraps the stored counter, so `current` would come back small
  while `peak` ratcheted to `u64::MAX` permanently, which is exactly the
  corruption #114 removed from the other side. The counter now clamps through
  the same saturating `fetch_update` `dealloc` uses, so the two ends match and
  `current` and `peak` stay consistent. In-tree call sites never get near the
  ceiling, but the type is `pub` with a `Clone`-able `Arc` inside, so a caller
  can put it there.
- A `.v` file libviprs writes is now readable by real vips, metadata and all,
  and no longer makes it print a warning on every open (issue #546). The
  trailer after the pixel data was libviprs's own JSON. libvips parses that
  slot as XML, so `vipsheader -a` on any file the crate wrote answered

      VIPS-WARNING **: error reading vips image metadata: VipsImage: XML parse error

  and then threw the whole metadata block away. Since `.v` exists for vips
  interop, and is the only format here that round-trips a float raster, that
  hit exactly the people moving compute intermediates between the two tools:
  they lost their ICC profile, their EXIF blob and their orientation, and got
  a warning they could not act on.

  The warning fired even for a raster with no metadata at all, because the
  writer always appended the 41 bytes of
  `{"orientation":1,"fields":{"entries":[]}}`. Nothing is written there now
  when there is nothing to say, which fixes the common case on its own.

  Everything else goes out as the XML document vips writes, `<root>` with a
  `<header>` and a `<meta>` block of `<field type="..." name="...">` elements.
  The four `MetadataValue` variants land on the four GTypes vips can
  round-trip: `gint`, `gdouble`, `VipsRefString`, and `VipsBlob` as base64.
  The reader takes both that and the old JSON form, so every `.v` already
  written keeps its metadata, and a `.v` vips itself wrote now reads whole
  rather than down to its orientation tag.

  Two places where this deliberately does not copy vips byte for byte. It
  escapes only what XML needs, so non-ASCII text survives: vips's own writer
  tests `*p < 32` on a signed `char` (`libvips/iofuncs/target.c:821`), which
  catches every byte of a multi-byte UTF-8 sequence, and `vips copy` over a
  `.v` carrying `café ☃ 日本` rewrites it as `caf&#x23c3;&#x23a9; …`
  irreversibly. And a field name containing a quote is escaped as `&quot;`
  where vips writes a backslash and leaves the attribute unterminated.

  **libviprs 0.4.0 reads a `.v` written now for its pixels, its geometry and
  its orientation, and not for its attached fields.** Its reader only takes a
  trailer as metadata when the first non-whitespace byte is `{`, and no byte
  sequence is both that and the XML vips requires, so vips interop and full
  field recovery on 0.4.0 cannot both hold. Nothing errors, and it runs one
  way only: this build reads every older file completely.

  Forward compatibility is kept and is better than it was. A `<field>` whose
  `type` this build does not know is carried opaquely and written back byte
  for byte, same as before, but now the carrier is vips's own encoding, so
  vips reads the carried field too. The one thing that cannot survive the
  format change is a value carried out of an *old JSON* trailer: spelling it
  in XML would mean interpreting it, which is the one thing a carried value
  does not allow, so a raster still holding one keeps the JSON trailer rather
  than losing it.
- A fallible convolution reports an allocation failure instead of aborting the
  process (issue #575). `samples_f64` widens every sample to `f64` before the
  traversal, eight bytes where the source carries one or two, and it did that
  with a plain `.collect()`: on failure that reaches `handle_alloc_error` and
  kills the process outright. Every `try_` entry point in the module sits on
  top of it, so none of them could report the failure and no caller could
  catch it. A `Result` that does not cover allocation is worse than no
  `Result`, because callers reasonably assume it does, and the rest of the
  crate had already settled the question the other way: `Raster` reserves with
  `try_reserve_exact` and returns `RasterError::AllocationFailed`, documented
  as never an abort.

  The widening now reserves fallibly and surfaces
  `ConvolutionError::Raster(RasterError::AllocationFailed { .. })`, and so do
  the other image-sized intermediates in the same functions: the combine
  buffers in `compass` and the output planes in `spcor` and `fastcor`, whose
  `# Errors` sections already promised the variant, the polar buffers in
  `canny`, and the whole-image copy `gaussblur` hands back for a `sigma` under
  0.2. That last one was a bare `self.clone()`, an image-sized allocation on
  the one branch of the operation that touches no other allocator, so it was
  the whole of what kept `try_gaussblur` abortable, and `canny` inherited it
  because `canny_gradient` blurs through `try_gaussblur` before it does
  anything else. It goes through the new crate-internal `Raster::try_clone`,
  which reserves with `try_reserve_exact` and carries the interpretation, the
  resolution and the attached fields exactly as `Clone` does.

  `try_conv`, `try_convsep`, `try_compass`, `try_gaussblur`, `try_spcor`,
  `try_fastcor`, `try_sobel`, `try_scharr` and `try_prewitt` are abort-free end
  to end as a result, and `try_canny` is on its uchar arm.

  Two things are deliberately not on that list, so the claim is not read wider
  than it goes. `try_canny`'s float arm and `try_sharpen` both widen through
  `Raster::f32_samples`, which still collects infallibly, and `try_sharpen`
  keeps five image-sized `vec![]` and `clone` scratch buffers of its own on top
  of that. Neither is one allocation away from the list, and pretending
  otherwise would be the same failure as a `try_` API that aborts, so both stay
  off it until the widening itself goes. `try_sharpen`'s `# Errors` says so in
  as many words, so the exclusion is where a caller reading the API docs will
  find it. The LabS round trip it makes through `colour.rs` was a third reason
  when this landed; that half is fixed in this same release under issue #672,
  and the `# Errors` block was rewritten there rather than left pointing at a
  claim that had stopped being true.

  It matters more than it reads: measured on a 4000x4000 `Rgb8` at integer
  precision, the widened buffer is 384 MB of a 486 MB peak for 48 MB of input,
  so it is by some distance the request most likely to be the one that fails.
  Removing the widening rather than making it fallible is the streaming work
  in #575's third item, which stays open.

- A `.v` file written by a newer libviprs no longer loses every metadata field
  when it is read by an older one (issue #565). The trailer was read as one
  `serde_json::from_slice` onto a struct holding a plain externally tagged
  `MetadataValue`, and serde errors on a variant it has never heard of, so the
  first field a future version added would fail the whole parse. The `if let
  Ok(..)` around it then swallowed the failure, and the image came back with
  no ICC profile, no EXIF blob and an orientation of 1, with nothing said. It
  is a data-loss break that `cargo semver-checks` cannot see, because it lives
  in the file format rather than in the API, and it was blocking the animated
  formats: a per-frame delay array is a new `MetadataValue` variant, so adding
  one would have started corrupting metadata for everyone on the current
  release.

  The trailer is now read entry by entry. An entry this build cannot represent
  is carried opaquely rather than dropped, so it survives being written back
  out and an old build that opens a new file and re-saves it does not strip
  what it could not read. Those fields stay out of `get_field` and
  `get_fields`, because this build can say the field was there but not what it
  means; setting or removing a field of the same name supersedes the carried
  one, so stripping still strips. Unknown trailer keys are ignored and missing
  ones default, so the shape can grow too, and the bytes written are unchanged,
  which is what keeps every already-released reader working.

  A trailer that opens with `{` and is not valid JSON is now reported as a
  corrupt `.v` rather than ignored. That is the one case left where metadata is
  genuinely unrecoverable, and it is narrow enough that no libviprs or libvips
  writer can produce it: libvips writes XML in the same slot, and a trailer
  that never claimed to be libviprs JSON is still read as absent, exactly as
  before.

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

- The `foreign-radiance` and `foreign-uhdr` oracle captures are JSON a standard
  parser will read (issue #674). Both carried bare `NaN` and `Infinity`
  literals, which RFC 8259 has no spelling for, so `serde_json`, `jq` and
  `JSON.parse` rejected the whole file rather than the one record that needed
  them: one `Infinity` in the radiance `encode_setcolr` sweep, and six `NaN`
  across the two degenerate-metadata arms of `uhdr2scRGB`.

  Nothing was failing over it, because Python writes these files and Python
  read them back. `json.dump` emits the bare literals by default and
  `json.load` takes them again as a documented non-standard extension, so a
  capture round-trips perfectly on the machine that produced it and breaks for
  a consumer in any other language... which is the moment someone ports a
  radiance or uhdr differential test to Rust, and starts by suspecting their
  own code rather than the fixture.

  Both files now quote the token `json.dump` would have written bare: `"NaN"`,
  `"Infinity"` and `"-Infinity"`, with every finite value staying an ordinary
  JSON number. That convention is introduced here rather than inherited.
  `foreign-nifti` is the only other capture that records a non-finite float and
  it carries both spellings at once: `"Infinity"` and `"NaN"` from its
  `probe.c`, and `"inf"`, `"-inf"` and `"nan"` from a `str(v)` in its
  `capture.py`. Bringing that file onto one spelling means re-capturing it, so
  it belongs with the #650 / #673 repin rather than here, and
  `tests/oracle_capture_json.rs` says which spelling it would have to move to.
  I picked quoting over `null` because those two records exist precisely to say
  which non-finite value libvips produced, and `null` folds all three onto one
  answer. Each `capture.py` sanitises on the way out and then dumps with
  `allow_nan=False`, so a value the sanitiser misses stops the capture rather
  than writing a file nobody outside Python can parse.

  I rewrote the two files in place instead of re-running the captures, because
  a re-run would have moved each area's recorded vips version too (issue #650)
  and the repair is worth exactly three lines. I drove the committed writer
  functions over the parsed documents to produce them, so what landed is
  byte-for-byte what a fresh capture emits and the diff is the added quotes and
  nothing else.

  `tests/oracle_capture_json.rs` is what keeps it shut. It walks the whole
  capture tree and parses every `oracle.json` with `serde_json`, reporting all
  the offenders rather than the first, so the next capture that reaches for a
  bare literal goes red in CI instead of waiting for someone to try to read it.
  Two more tests sit next to it: one checks the three tokens are pairwise
  distinct and each comes back as itself rather than as one of the others,
  which is the property `null` would lose, and one asserts `serde_json` really
  does refuse the bare literals, so the guard cannot quietly become a check
  that passes for the wrong reason. A fourth reads the two repaired files back
  and checks the values really are a `+inf` and a `NaN` in the rows that are
  supposed to carry them, which is the only one of the four that tests what the
  Python writer actually emitted.

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
