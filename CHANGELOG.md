# Changelog

All notable changes to libviprs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- Every public options struct is `#[non_exhaustive]` and grows a `with_*`
  builder setter per field, so a downstream struct literal no longer compiles
  (issue #630). Ten types: `gif::SaveOptions`, `jp2k::SaveOptions`,
  `jxl::SaveOptions`, `radiance::SaveOptions`, `uhdr::SaveOptions`,
  `webp::SaveOptions`, `SvgOptions`, `AffineOptions`, `ResizeOptions` and
  `MagickLoadOptions`.

  Five of them carried a doc line promising that later fields could be added
  without a breaking change, and two went further and said they were
  *deliberately* not `#[non_exhaustive]` so `..Default::default()` would keep
  working downstream. Half of that was true and the half that mattered was not:
  a `..Default::default()` literal does survive a new field, an exhaustive one
  stops compiling with `E0063`, and the docs advertised the exhaustive spelling.
  So the guarantee held only for callers who happened to pick the other form,
  with nothing making them.

  The crate proved it against itself: four integration tests were called
  `save_options_are_constructible_downstream`, each built its options both ways,
  and integration tests compile as an external crate. Measured on this tree,
  adding one field to the five save-options structs breaks eight of their twelve
  construction sites with `E0063`. `#[non_exhaustive]` refuses all twelve up
  front instead, which is the trade: one break now, before 0.5.0 and at a time
  somebody picks, against a break later at a time nobody does.

  **Migration.** Start from `default()` and chain a setter per field:

  ```rust
  // before
  let o = gif::SaveOptions { interlaced: true, ..Default::default() };
  let r = ResizeOptions { vscale: Some(0.5), ..ResizeOptions::default() };

  // after
  let o = gif::SaveOptions::default().with_interlaced(true);
  let r = ResizeOptions::default().with_vscale(Some(0.5));
  ```

  Reading a field is unchanged; only construction moves. `DecodeLimits` has had
  this shape since it was written and is where it comes from.

  Measured downstream cost inside this workspace: 12 sites in this repo's own
  `tests/`, all migrated here, plus three outside it, all `ResizeOptions`, in
  `libviprs-cli` (`src/ops/resample.rs`) and `libviprs-tests`
  (`tests/resample_nearest_alpha.rs`,
  `tests/resample_premultiplied_alpha_reference.rs`). Both are path
  dependencies, so they need the one-line change above; `libviprs-cli#46` and
  `libviprs-tests#181` carry the exact sites.

  `tests/non_exhaustive_options.rs` holds the attribute on all ten and
  exercises every setter from outside the crate, so a setter that goes missing
  fails to compile rather than failing quietly. `jp2k::SaveOptions` is the
  tenth: it landed in #783 while this was in flight, carrying the same promise
  word for word, which is how a rule with no check behind it spreads.

- `Raster::encode_jp2k(quality: u8, lossless: bool)` and
  `Raster::encode_jp2k_chroma(quality, lossless, subsample)` are **gone**,
  replaced by `Raster::encode_jp2k(options: jp2k::SaveOptions)` and its new
  sibling `Raster::save_jp2k` (issue #501). The two old ones lived in
  `crate::foreign_stubs` and always returned `EncodeError::Unsupported`, so
  nothing that called them ever produced bytes; the new one does.

  **Migration.** `encode_jp2k(q, true)` becomes
  `encode_jp2k(jp2k::SaveOptions::default())`. `encode_jp2k(q, false)` becomes
  `encode_jp2k(jp2k::SaveOptions { compression: jp2k::Compression::Lossy { ratio } })`,
  and `ratio` is a compression ratio rather than vips's `Q`: see Added below
  for why there is no `Q` to pass. `encode_jp2k_chroma`'s `subsample` argument
  has no replacement, because `openjpeg2-pure-rs` exposes no subsampling knob;
  it never did anything either way.


- `ConversionError::FloatFormatUnsupported` is renamed
  `ConversionError::FloatUnsupported` (issue #730). Three of the crate's four
  float-refusal variants already spelled it that way
  (`RasterError::FloatUnsupported`, `ArithmeticError::FloatUnsupported`,
  `ExtractError::FloatUnsupported`), and the odd one out meant a caller asking
  "did this refuse a float raster" had to carry an exception in the one
  `matches!` they wanted. The enum is `#[non_exhaustive]`, so a caller with a
  wildcard arm is unaffected; a caller matching the variant by name renames it.

  Four enums is not the thing being fixed: each module owning its error type is
  what gives a single-family caller a tight surface. A predicate per enum, the
  shape `SourceError::is_alloc_limit` took in #686, was considered and not
  taken: that one composes because it collapses five variants of *one* enum onto
  a question, where this is one variant of each of four, so it would be four
  impls that still cannot be called through a single type without a trait. The
  set is now written down in one place, `src/error.rs`'s `OpError` module doc,
  next to the existing note on matching raster failures.

- Catching "the decode allocation budget refused this file" takes one call
  instead of seven match arms. `GifError::AllocLimitExceeded`,
  `FitsError::AllocLimitExceeded`, `ExrError::AllocLimitExceeded`,
  `RadianceError::AllocLimitExceeded` and `JxlError::AllocLimitExceeded` are
  **gone**, collapsed onto `SourceError::AllocLimitExceeded`, which grows a
  `geometry: Option<DeclaredGeometry>` field carrying the width, height and
  band count the five used to carry separately (issue #686). There is a new
  `SourceError::is_alloc_limit()` that answers for every shape the budget can
  refuse in.

  #632 put one price and one comparison behind every declared-geometry
  decoder. That left five variants doing nothing but re-tagging a refusal
  computed elsewhere, in two field vocabularies (`needed` against
  `needed_bytes`, `channels` against `bands`), which is the cheapest they will
  ever be to delete. They now all go through one
  `DecodeLimits::check_image_alloc`, and so does the TIFF page reader, which
  reported no geometry before and reports its page's now.

  **Migration.** Match `SourceError::AllocLimitExceeded { .. }` where you
  matched any of the five, or call `err.is_alloc_limit()` and stop matching.
  `needed` becomes `needed_bytes`, and `width` / `height` / `bands` /
  `channels` move inside `geometry`. Both that struct and the enum are
  `#[non_exhaustive]`, so a destructuring match needs `..` in two places and
  the compiler error if you forget will not obviously say why:

  ```rust
  Err(SourceError::AllocLimitExceeded {
      geometry: Some(DeclaredGeometry { width, height, .. }),
      needed_bytes,
      ..
  }) => ...
  ```

  `DeclaredGeometry::new` builds one, so a caller can still construct the
  error in their own tests.

  The `what` label says which buffer was refused: `"GIF canvas"`,
  `"FITS pixel buffer"`, `"OpenEXR sample buffers"`,
  `"Radiance pixel buffer"`, `"JPEG XL frame buffer"`,
  `"WebP frame buffer"`, `"TIFF page pixel buffer"`, `"TIFF file body"`,
  `"image file body"`. It is a human-readable label rather than a
  compatibility promise: the wording may change and new decoders add new
  labels, so branch on `geometry` or on the variant, never on the string.

  **WebP comes along too**, which is what #686 asked for and what I initially
  got wrong. It looked like one of four formats reporting the `image` crate's
  shape, but the four are not alike underneath: JPEG, PNG and single-image
  TIFF are refused inside `image`'s own decoder through `Limits::reserve`, so
  there is genuinely no libviprs price and no declared geometry behind them.
  WebP had both, from `decoder.dimensions()` and
  `decoder.output_buffer_size()`, and fabricated an `image::ImageError` to
  look like the other three. Since the frames refused are set by the
  comparison and not by the error type, that consistency was costing a caller
  the geometry and the price and buying nothing.

  **Two things this does not do.** JPEG, PNG and single-image TIFF keep the
  `image` shape, for the reason above. `JxlError::DecoderAllocLimitExceeded`
  also stays, because it is `jxl-oxide`'s own tracker refusing an internal
  buffer at a size it does not report out, and a file can trip either without
  tripping the other. `is_alloc_limit` covers all three so a caller does not
  have to know the split, and it answers the same in a build with or without
  the `jxl` feature, since that variant exists in both.

  **What `is_alloc_limit` deliberately says no to**, since one of them looks
  like a false negative: `DimensionLimitExceeded` and `PageLimitExceeded` are
  different ceilings, and so is
  `SourceError::Raster(RasterError::ByteBudgetExceeded)`, which
  `Raster::ppm_load`, `csv_load` and `matrix_load` return through this same
  enum with a message reading "needs N bytes, exceeding the M-byte allocation
  budget". That M is `DEFAULT_MAX_ALLOC_BYTES`, the raster construction
  ceiling, not `DecodeLimits::max_alloc_bytes`, so raising the decode limit
  does nothing about it. The predicate's whole test is "does raising
  `max_alloc_bytes` fix this", and all three fail it.

  `geometry` is an `Option` rather than three flat fields because the
  whole-file read prices a file's length on disk, which says nothing about the
  geometry declared inside it. Reporting `0x0x0` there would have been a lie
  in a field a caller reads.

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

- **CI runs the non-default features it had been compiling out**, and a guard
  so the next one cannot be forgotten (issues #772, #816). `jp2k`, `avif`,
  `packfile`, `serde` and `tracing` all gate code behind
  `#[cfg(feature = ...)]`, and no job named any of them, so those bodies were
  compiled *out* and 53 assertions never ran: 24 for `jp2k`, 8 for `packfile`,
  6 for `serde` (all of `tests/serde_wire.rs`, which opens
  `#![cfg(feature = "serde")]`), 5 for `tracing`, and 10 for `avif` that are the
  codec's entire oracle comparison and are `ignore`d without the feature.

  The lint half was not hypothetical either: the first
  `cargo clippy --all-targets --features packfile` ever run on this tree came
  back **red**, on a `collapsible_if` in `src/sink_packfile.rs` that no job had
  ever compiled. That is fixed here too.

  The MSRV cells are measured rather than assumed. A feature needs one when it
  pulls in a crate declaring no `rust-version`, because that is exactly what the
  MSRV-aware resolver cannot see and `Cargo.lock` is not committed: `svg` adds
  12 such crates, `avif` 9, `packfile` 5 and `jp2k` 1 (`openjpeg2-pure-rs`
  0.1.1). All four pass `cargo +1.97 check --all-targets` today, so the cells
  are a guard rather than a fix.

  The same issue had already been filed three times, once per format (#502 for
  `svg`, #500 for `jxl`, #772 for `jp2k`), because nothing checked the class.
  `tests/ci_feature_coverage.rs` now does: it reads `Cargo.toml` and `ci.yml` at
  compile time and asserts that `[features]` holds exactly the names an explicit
  table covers, and that every cell the table claims is in the right job. A new
  feature fails there until somebody writes down which jobs it belongs in, and
  why.

- **Animated GIF save** (issue #573). `Raster::encode_gif` splits the raster by
  its page height and writes one GIF frame per page, taking the per-frame
  delays out of the `delay` field and the NETSCAPE loop block out of `loop`,
  which is where vips reads them too: `gifsave` has no argument for either and
  `cgifsave.c:753` reads them back off the image.

  A still built from scratch saves to the same bytes it always did, because a
  raster with no page split is one page and a raster with no `loop` field
  loops forever, which is the block cgif already wrote. A still that was
  *loaded* from a GIF is a different matter, and worth saying out loud: the
  loader attaches `loop`, so re-encoding one now honours it, and a source file
  carrying no NETSCAPE block round-trips to a file carrying none, where before
  it gained a block holding zero. That is the better match for vips, which
  writes no block for `loop = 1`, and
  `a_reloaded_still_carries_its_loop_count_back_out` pins all three cases.

  The frame geometry is the page, not the roll, so the GIF axis limit applies
  per frame: a 1x80000 roll of two 40000-row pages saves, and a 1x70000 still
  does not.

  **Four measured divergences from vips, each with a test carrying the
  measurement.** A `delay` array whose length is not the page count is refused
  where vips pads with zeros or truncates (a two-entry array on a four-page
  roll wrote `2 3 0 0`, a six-entry one wrote `2 3 4 5`). A negative delay or
  `loop` is refused where vips casts it unsigned, turning -10 ms into 655
  seconds and `loop = -1` into 65536 plays. A delay past what the wire holds
  saturates where vips wraps, turning 655360 ms into no delay at all and
  700000 ms into 44.64 seconds. And a stored `page-height` that does not
  divide the raster is refused at `try_set_page_height` rather than silently
  collapsing at the save, which is what `vips gifsave --page-height 5` on a
  12-row image does.

  A field of the *wrong type* is ignored rather than refused, matching the
  `page-height` and `n-pages` readers: an untrusted `.v` can leave anything
  under any name, so a wrong type means "this is not the field I read", where
  a negative integer means "this is the field and its value is impossible".

  Disposal follows cgif, measured over five files: restore-to-background on
  every frame but the last when the animation carries transparency, keep
  otherwise. Under "keep" a transparent pixel on page 2 would show page 1
  through it, because every frame written here covers the whole screen.

  **`Raster` drops `delay` wherever it drops `page-height`.** The array holds
  one entry per page, so it describes the page split rather than the image,
  and an op that changes the shape hands on an array that no longer indexes
  anything. `roll.extract_page(0).encode_gif(..)` found it: the extracted page
  arrived carrying the roll's four delays and the save refused it. Dropping is
  the same call the page split already gets, and for the same reason; keeping
  it would have been worse than refusing, since the first delay would then be
  written onto a page that is not the first. `merge_fields_from` refuses to
  import one for the matching reason, so joining a still to an animation
  cannot give the still that animation's timings.

- **Animated GIF load** (issue #572). `decode_gif_with` takes `gif::LoadOptions`
  carrying vips's `page` and `n`, composites the frames it selects and stacks
  them into one raster whose `page-height` is the logical screen height, which
  is the page roll `src/frames.rs` landed for. `decode_gif` is that with the
  vips defaults, so a still load is unchanged down to the bytes.

  **Delays come back in milliseconds**, which is the whole point of the issue.
  The graphic control extension counts centiseconds and vips's `delay` counts
  milliseconds, so a decoder that passes the number through is a silent factor
  of ten that every other assertion still agrees with. `FrameDelay` carries the
  unit in the type across that boundary, and `4 6 8 10` on the wire has to come
  back `40 60 80 100`.

  **The delay array covers the pages the raster holds**, one entry per page,
  and that diverges from vips deliberately. vips reports the whole file's array
  whatever window was loaded: `anim4.gif[page=2,n=2]` loads frames 2 and 3 and
  still says `delay: 40 60 80 100`, so re-saving it writes 40 and 60
  centiseconds onto frames whose real delays are 80 and 100. Both halves of
  that were measured on the pinned 8.18.6 binary. Making `delay[i]` loaded page
  `i`'s delay is what makes the array usable on the raster carrying it, and it
  is the split `n-pages` already has: `n-pages` describes the file,
  `pages_loaded` describes the raster.

  Disposal and blending are libnsgif's, each rule measured by building the
  fixture, running it through vips and pinning what came back. The one that
  needs saying out loud is restore-to-background, which has two arms a single
  fixture cannot see: the clear is transparent when the disposed frame declares
  a transparent index and the background colour when it does not, and an index
  past the end of the colour table is black. Reserved disposal codes 5, 6 and 7
  keep the canvas, matching vips; code 4 is a tracked divergence (issue #827),
  because the `gif` crate maps every code it does not know onto
  `DisposalMethod::Any` and it arrives here indistinguishable from 0.

  A window the file cannot serve is `GifError::BadPageNumber` rather than a
  clamp, matching vips, which fails `[page=4]`, `[n=99]`, `[n=0]` and
  `[page=3,n=3]` on a four-frame file with `bad page number`.

  **The frame walk is bounded now, where the still loader's was not.** A GIF's
  frame list has no count in its header, so the only way to know how long it
  is, is to walk it, which is the exposure `DecodeLimits::max_pages` exists
  for and which `decode_tiff_page` already honours for the IFD chain. GIF was
  the one multi-page loader that did not consult it, and now does. The
  per-frame index buffer is priced too: a frame may declare a rectangle far
  larger than the logical screen (libnsgif clips such a frame rather than
  refusing the file, and libviprs matches that), so a forty-byte file
  declaring a 65535x65535 frame on a 1x1 screen used to allocate 4 GiB
  through a budget that had only seen the 3-byte screen.

- **Animated WebP and animated JPEG XL load** (issues #569, #621).
  `webp::decode_webp_with` and `jxl::decode_jxl_with` take a `LoadOptions`
  carrying libvips's `page` and `n`, decode the frames asked for and stack
  them into one toilet-roll raster with the page geometry #564's model
  derives. `decode_webp` and `decode_jxl` are those functions at their
  default, which is page 0 and one frame, so nothing about the still path
  moved.

  Both structs are `#[non_exhaustive]` with `with_page` / `with_n` builders,
  as #630 requires, and they are `gif::LoadOptions` field for field: same
  `page: u32`, same `n: i32` with `-1` meaning every remaining page, same
  argument order on the entry point. Three sibling loaders spelling one
  libvips argument two ways is worse than carrying its sentinel, and
  `non_exhaustive_options.rs` now asserts the three defaults against each
  other rather than restating them.

  An animation now carries `page-height` (when more than one page was
  loaded), `delay` as a `MetadataValue::IntArray` of milliseconds, `loop`,
  and the `gif-delay` / `gif-loop` compatibility fields vips attaches beside
  them. `n-pages` still counts the pages of the *file*, as #635 pinned it.

  **The delay array is subset to the pages actually loaded, and vips's is
  not.** Measured on 8.18.6, `vipsheader -f delay 'anim4.webp[page=1,n=2]'`
  prints the file's whole `45 67 200 12` onto a raster holding pages 1 and
  2. Nothing on that raster records the offset, so that array cannot be
  lined up with the pages that are there and a saver reading it writes the
  wrong two delays, silently. Here `delay[i]` is the delay of loaded page
  `i` and `delay.len() == pages_loaded()` always holds.

  **Both formats are read-only and that is a decision, not an oversight.**
  No pure-Rust encoder writes a WebP `ANIM`/`ANMF` or a JPEG XL animation
  header, so an animation can be loaded and transformed and not saved back
  in its own format. `encode_webp` and `encode_jxl` write a roll as **one
  tall still image** rather than refusing it, which is a divergence from
  `vips webpsave` and `vips jxlsave` on the same raster and is pinned as
  one; refusing would fire on the ordinary path of loading two pages and
  saving the result, and the pixels are a perfectly good image. A caller
  who wants one frame uses `Raster::try_extract_page`, and a caller who
  wants an animation saves GIF, which is the one animated format in this
  crate with a pure-Rust encoder behind it.

- `SourceError::PageOutOfRange` is the typed refusal for a `page` or
  `page + n` naming pages a file does not have, shared by the animated
  loaders (issues #569, #621). Distinct from `SourceError::PageLimitExceeded`,
  which is the configured ceiling rather than the file's own count. vips
  refuses the same requests, with `webp: bad page number`, and clamps none of
  them: `[page=4]`, `[page=2,n=5]` and `[n=0]` on a four-page file all fail
  there too.

- **Analyze 7.5 (`.hdr` + `.img`) load** (issues #510, #640, #764).
  `decode_analyze_file` takes either half of the pair or the bare stem and
  resolves the other, `analyze::decode_analyze` takes the two buffers, and a
  `.hdr` becomes a live row in the content sniffer so `decode_file` loads an
  Analyze image without being told what it is. There is no save half: `vips`
  registers no `analyzesave`.

  **The decode seam grew a route kind for it**, which is the part of this
  worth reading. Analyze is the only container in the crate that is
  inherently two files: a `.hdr` has a geometry and no pixels, an `.img` has
  pixels and no geometry, and `Decoder::Native(fn(&[u8], DecodeLimits))`
  cannot express either. The route table now has a `Paired` kind carrying two
  function pointers, one the file entry point calls with the path and one the
  buffer entry point calls with the header half alone; the alternatives, a
  path-only entry point with no sniff row and a sniff row that always
  refuses, both leave `decode_file` unable to load an Analyze image at all,
  which is the format's whole normal use.

  `decode_bytes` on a `.hdr` therefore reports
  `AnalyzeError::PixelsAreInASiblingFile`, after validating the header in
  full, so a malformed one still reports its malformation. And one divergence
  falls out that is unavoidable rather than chosen: `vips` loads `fred.img`
  as well, because its `is_a` rewrites whatever name it is handed, and a
  content sniff has nothing to look at in a raw pixel array.
  `decode_analyze_file` takes all three names, so only the sniffing entry
  point is narrower.

  **Big-endian, always, with no flag and no escape hatch.** Every field of
  the 348-byte header and every pixel of the `.img` is big-endian whatever
  the host is; a little-endian `.hdr` is refused because its `sizeof_hdr`
  reads back as 0x5C010000. This is the single most likely thing for a port
  on a little-endian host to get backwards and both halves are pinned.

  The rest of the measured contract: the rank is `dim[0]` and must be 2..=7,
  the width is `dim[1]` and the height is `dim[2]` multiplied by every extent
  up to the rank, so a volume flattens into a toilet roll with nothing but
  the `dsr-image_dimension.dim[]` metadata recording that it was ever 3-D.
  `vox_offset` is parsed, attached and then ignored, so the pixels come from
  byte 0 of the `.img` on every file that sets one. `bitpix` is attached and
  never consulted. `DT_RGB` is the only multi-band datatype and its `.img` is
  interleaved, not planar. A short `.img` is an error and a long one is not.

  63 `dsr-<section>.<member>` metadata fields and the 348-byte `dsr` blob are
  attached, with both of `getstr`'s traps reproduced: an 80-byte `descrip`
  loses its last byte to `g_strlcpy`'s size argument, and every byte that is
  not printable ASCII becomes `@`, which is lossy and not reversible. The
  capture's own prose states that second rule with an `||` where its measured
  data needs an `&&`; that is issue #797, fixed in the same wave.

  Three of the nine datatypes `analyzeload` reads have a carrier here
  (`DT_UNSIGNED_CHAR`, `DT_FLOAT`, `DT_RGB`) and the rest are refused **by
  name**: `DT_SIGNED_SHORT` and `DT_SIGNED_INT` need #516, `DT_DOUBLE` needs
  #518, and `DT_COMPLEX` has no carrier and no issue. `DT_SIGNED_SHORT` is
  what most real Analyze volumes use, so it is the refusal a caller meets
  first.

  One deliberate divergence, the same one `matload` carries: a zero or
  negative dimension is refused rather than clamped to 1 by GObject's
  property range check, which in vips leaves the load exiting 0 with a
  silently wrong geometry.

  The declared geometry is priced against every `DecodeLimits` ceiling
  **before the `.img` is opened**, because a 348-byte header can declare
  1.07 gigapixels in front of a six-byte image, so a header that prices past
  the budget costs no second read.

  No new dependency.

- **MATLAB level 5 (`.mat`) load** (issues #510, #640, #763). `decode_mat`
  reads the first variable of rank 1, 2 or 3 out of a MAT-5 container, in
  either byte order, bare or inside a `miCOMPRESSED` zlib element, and `.mat`
  becomes a live row in the content sniffer so `decode_bytes` and
  `decode_file` reach it without being told what the bytes are. There is no
  save half: `vips` registers no `matsave`.

  **The sniff is the shipped binary's, not the C source's**, and that is the
  sharp edge of this port. `vips__mat_ismat` in the reference checkout reads
  ten bytes and compares them with `MATLAB 5.0`; the 8.18.6 dylib that
  shipped reads 128 and validates the version word and the endian indicator
  as well, and the 8.18.4 it replaced did not (issue #650). A port written
  from the source would claim `MATLAB 5.1`, `matlab 5.0`, `MATLAB_5.0`, a
  file with a bogus endian indicator and a 127-byte file, all of which
  8.18.6 refuses. The whole predicate lands as two route-table rows, because
  the version and the indicator are one four-byte constant per byte order and
  the 128-byte length floor falls out of the offset.

  The container is a transpose and a de-planarisation, not a copy.
  `mat2vips_get_header` takes the height from `dims[0]` and the width from
  `dims[1]`, so a MATLAB 2x3 becomes a 3x2 image and element `(r, c)` is
  pixel `(c, r)`; rank 3 makes `dims[2]` the band count and the file holds
  the planes one after another where a libviprs raster is interleaved.

  The behaviours a spec reading gets wrong are the point. One variable loads
  and there is no way to pick it. The rank filter runs in the search loop and
  the class check runs *after* it, so a loadable `uint8` variable behind an
  `int64` one fails outright. The logical flag is read and ignored. And
  read-info validates the array-flags, dimensions and name subelements and
  never the data one, so a file truncated mid-element reports a full header
  and fails only at the pixels.

  Four deliberate divergences, all refusals where `matload` carries on.
  A complex array is refused: vips never reads the complex bit and memcpys
  out of a `mat_complex_split_t`, so its pixels are the raw bytes of two heap
  addresses and change from run to run under ASLR. A non-positive dimension
  is refused rather than clamped to 1 by GObject. A band count other than 1,
  3 or 4 is refused rather than pushed onto a multiband carrier the decode
  path does not produce. And a stored element type that does not match the
  array class is refused rather than widened.

  Three of the eight classes `matload` reads have a carrier here (`mxUINT8`,
  `mxUINT16`, `mxSINGLE`) and the other five are refused **by name** with the
  issue that would add the carrier: `mxINT8`, `mxINT16` and `mxINT32` need
  #516, `mxUINT32` needs #517, and `mxDOUBLE`, which is what MATLAB writes
  unless told otherwise, needs #518.

  The allocation budget matters twice here rather than once.
  `dims_100000x100000.mat` declares ten gigapixels behind eight bytes of
  data, so the declared geometry goes through `DecodeLimits::check_coord`,
  `check_pixels` and `check_image_alloc` before anything is reserved; and a
  `miCOMPRESSED` element's inflated size is not declared anywhere in the
  container, so every inflate stops at `max_alloc_bytes` and is refused
  rather than grown past it.

  No new dependency. `flate2` was already a required dependency of this
  crate, and nothing else in the format needs one.

- `UhdrError::BadSaveInput`, so `uhdr::encode_uhdr`'s input refusal names the
  operation that actually failed (issue #810). It reused `UhdrError::BadInput`,
  whose Display is `uhdr2scRGB: {reason}`, so a failed **save** reported the
  **expand** operation and reported it first:
  `uhdr2scRGB: uhdrsave needs a 3-band float image, got Rgb8`. It now reads
  `uhdrsave: needs a 3-band float image, got Rgb8`. `UhdrError` is
  `#[non_exhaustive]`, so a caller with a wildcard arm is unaffected; a caller
  matching `BadInput` to catch a save refusal moves to the new variant.

- `Raster::encode_uhdr(quality)` and `Raster::encode_uhdr_gainmap_scale(quality,
  scale_factor)` **write an Ultra HDR container** instead of returning
  `EncodeError::Unsupported` (issue #757). #508 landed the writer in
  `crate::uhdr` with no new dependency and libvips reads its output back
  (`vipsheader -a` reports `vips-loader: uhdrload`), but the documented
  `Raster` surface still refused, so a caller was told this build cannot write
  Ultra HDR while the crate demonstrably could.

  The input is a **3-band `f32`** raster holding linear-light scRGB, which is
  what a gain map is computed from. Anything else is
  `EncodeError::InvalidParameter` naming the format it got, not `Unsupported`:
  the build can write the format, this raster is the wrong shape for it, and
  those are different answers. libvips gates on the interpretation tag instead
  and it does not buy correctness there. Measured on 8.18.6: a 1-band scRGB
  float image saves as an all-black container, and a `uchar` scRGB image is
  re-linearised on the way in, so a constant 128 comes back as 0.2137 rather
  than 0.502.

  `scale_factor` is the libvips `gainmap-scale-factor` and is refused outside
  1..=128. libvips declares that same range and then silently substitutes the
  default: `--gainmap-scale-factor 0` and `--gainmap-scale-factor 200` both
  exit 0 and write the same 2630 bytes as the plain call, with
  `gainmap-scale-factor: 2` in the header. `quality` is clamped to 1..=100 the
  way `Raster::encode_jpeg` clamps its own.

- A page model for multi-frame images (issue #564). A multi-frame image is one
  `Raster` whose rows are a whole number of equal-height pages stacked top to
  bottom, the layout libvips calls a toilet roll, and the split is now a
  derived, checked value rather than an integer riding along in the metadata.
  `Raster::page_layout`, `Raster::get_page_height`, `Raster::pages_loaded`,
  `Raster::page`, `Raster::try_extract_page` / `Raster::extract_page` and
  `Raster::try_set_page_height` / `Raster::set_page_height` /
  `Raster::clear_page_height` are the surface, and the new `frames` module
  holds `PageLayout`, `FrameDelay` and `LoopCount`.

  `Raster::get_page_height` ports `vips_image_get_page_height`, sanity check
  included: a stored `page-height` counts only when it is positive and divides
  the raster's height exactly, and otherwise the raster is one page. Measured
  against 8.18.6 through `ctypes` on a 4x12 image, where every divisor of 12
  comes back as stored and 5, 7, 11, 13, 24, 100, 0 and the negatives all come
  back as 12. So the split can never fail to tile the rows it describes, and a
  caller sweeping `0..raster.pages_loaded()` cannot land off the end.

  `Raster::pages_loaded` is **not** `Raster::get_n_pages`. The first counts the
  pages this raster holds; the second counts the pages the file held (#635).
  They differ whenever a loader was asked for a subset: `vips copy
  'anim3.webp[n=2]' out.v` reports `n-pages: 3` on a raster holding two pages.

  `FrameDelay` holds milliseconds and says so in the type, because the two wire
  formats disagree: `gifsave` writes `round(ms / 10)` centiseconds with halves
  to even (measured: `35 55 15 25` ms wrote `4 6 2 2`, `45 67 5 1` wrote
  `4 7 0 0`), where `webpsave` writes milliseconds straight into `ANMF` and
  instead clamps anything at or under 10 ms up to 100 ms (measured: `8 9 10 11`
  went out as `100 100 100 11`). `LoopCount` counts plays, `0` meaning forever,
  and carries the GIF off-by-one: the NETSCAPE2.0 block holds
  repeats-after-the-first and a single play carries no block at all, where
  WebP's `ANIM` chunk holds the play count unshifted.

- JPEG 2000 load and save, behind a new non-default **`jp2k`** feature (issue
  #501). Build with `--features jp2k` and `decode_jp2k` reads both container
  forms, the RFC 3745 JP2 box structure and the bare `SOC` + `SIZ` codestream,
  `.jp2` and `.j2k` become live rows in the content sniffer, and
  `Raster::encode_jp2k` and `Raster::save_jp2k` write a JP2 container. The
  `Raster::encode_jp2k(quality, lossless)` and `Raster::encode_jp2k_chroma`
  typed-`Unsupported` stubs are gone; see Breaking below.

  Without the feature nothing about the surface moves: every entry point still
  exists at the same signature and returns a typed refusal naming the feature.
  `decode_jp2k` reports `Jp2kError::FeatureNotEnabled`, the encoders report
  `EncodeError::Unsupported { format: "jp2k" }`, and the sniffer still routes a
  JPEG 2000 file here so it reads as "this build has no JPEG 2000" rather than
  "these bytes are not an image".

  **It is the cheapest codec feature in the crate.** Measured with
  `cargo generate-lockfile` on a clean tree, 288 packages before and 290 after:
  **+2 lock entries, and both of them are the two crates themselves**, because
  neither has a dependency of its own. `svg` costs +29 and `jxl` costs +21.
  Neither compiles C, declares a `links` key, runs a build script or carries a
  `-sys` suffix. It is non-default for compile time alone, which is what `svg`
  argued on its own: 9.7k lines of decoder and 36.9k of translated encoder is
  real build time and nobody who does not read or write JPEG 2000 should pay
  it.

  The two halves are split the way `crate::jxl`'s are, and on the same line.
  `hayro-jpeg2000` decodes, because it is `#![forbid(unsafe_code)]` at the
  settings this build uses and the decoder is the half that eats
  attacker-controlled bytes. `openjpeg2-pure-rs`, a translation of OpenJPEG's
  own C, encodes, because the encoder only ever sees a `Raster` this crate
  already owns. Three other pure-Rust encoders were measured and rejected;
  `justjp2` is the one worth naming, because it looks ideal and is not: its own
  `lossless: true` round trip is not lossless, `hayro-jpeg2000` refuses its
  output outright, and OpenJPEG 2.5.4 through `vips jp2kload` decodes it to a
  flat mid-grey with every coefficient gone.

  Measured against `vips` 8.18.6 over the 27 fixtures in
  `oracle-captures/foreign-jp2k/`, the result splits by wavelet. The
  **reversible 5/3** path is byte-identical to what `vips rawsave` writes, for
  seven fixtures covering greyscale, RGB, RGBA, CMYK, tiled, subsampled and
  multi-resolution, so its pins carry no tolerance at all. The
  **irreversible 9/7** path is float-specified and agrees with OpenJPEG to
  within 4 counts at worst, pinned per fixture at the number each one actually
  reaches. The encoder goes the other way too: every carrier it writes reads
  back through `vips jp2kload` bit for bit, including 16-bit greyscale and
  4-band CMYK.

  Four loader behaviours are ports rather than side effects, and each one is
  invisible to an 8-bit RGB test. A precision-N component is **left-justified**
  into its element, so a 12-bit 4095 comes back as 65520 and the real depth
  survives in `bits-per-sample`. A bare codestream with **subsampled chroma**
  gets OpenJPEG's inverse YCC, coefficient for coefficient and with its
  truncating casts, because a rounding implementation is one count out. The
  **tile geometry** is attached only when the image is more than one tile,
  which is what `vipsheader` shows. And the **ICC profile** comes out of a
  `METH=2` `colr` box verbatim and unvalidated, which is what `jp2kload` does
  and is why the fixture carrying 24 bytes that are not a profile still has
  one.

  Two refusals are carrier gaps in this crate rather than format ones, and both
  would otherwise be silently wrong answers. A **signed component** is refused
  with `Jp2kError::SignedComponent`: `PixelFormat` has no signed carrier and
  the decoder reports every component DC-level-shifted into the unsigned range,
  so decoding one anyway comes back offset by half the range. More than **16
  bits** of precision is refused with `Jp2kError::PrecisionNotSupported`: there
  is no 32-bit integer carrier, and the decoder's `f32` container cannot hold a
  31-bit sample either.

  The resolution count travels as **`jp2k-resolutions`**, not as `n-pages`.
  `vipsheader` calls it `n-pages` and vips's `page` selects a resolution level
  rather than a frame, and this crate reserves that key for counts a zero-based
  `page` argument can select (issue #635), which `decode_jp2k` does not have
  yet.

  Lossy is a **compression ratio and not a `Q`**. `jp2ksave --Q` sets
  OpenJPEG's `cp_fixed_quality` with a distortion ratio in decibels;
  `openjpeg2-pure-rs` exposes `cp_disto_alloc` with a compression ratio and
  keeps the rest `pub(crate)`. Those are different numbers, so
  `Compression::Lossy` carries a `ratio` and there is no `Q` field for this
  crate to accept and reinterpret, which is the same answer `jxl` gave to
  `jxlsave`'s `distance`.

  Known limits, each filed: the image origin is read as the standard defines it
  and vips subtracts it twice (#766); the `colr` box's enumerated colour space
  does not override the component count here (#767); tiled save has no encoder
  parameter behind it (#768); and more than four bands is refused on the way
  out because the loader cannot read it back, though `jp2ksave` writes it
  (#769).

- AVIF still-image load, behind a new non-default **`avif`** feature
  (issue #605). Build with `--features avif` and `decode_avif` reads an AV1
  keyframe out of an ISOBMFF container, with alpha from an `auxl`-linked
  auxiliary item, at 8, 10 and 12 bits; `.avif` becomes a live row in the
  content sniffer, matching `ftyp` + the major brand `avif` at offset 4.

  **It is deliberately not `heifload` parity, and the module says so at its
  own entry point.** `heifload` also reads HEVC, AVC and JPEG payloads and
  `heifsave` writes HEVC by default, so this covers one of four inputs and
  none of the default output. An HEVC payload is refused by name rather than
  as a generic parse failure, because that is the wall issue #498 closed on
  and it has not moved. There is no save side and none is deferred: no
  pure-Rust AV1 encoder is worth shipping in a pyramiding engine.

  Pixels match vips exactly rather than approximately, which almost nothing
  else in the foreign-format roadmap can claim. AV1 decode is bit-exact by
  specification, and the colour step that is *not* fixed by any specification
  is pinned against `oracle-captures/foreign-avif` frame by frame. That step
  turned out to need two implementations: libheif reaches 4:4:4 and 4:2:0
  through different arithmetic, float with round-to-nearest for one and 8.8
  fixed point for the other, and measured over 1024 pixels each way the wrong
  one is wrong on 103 and 124 pixels respectively, always by exactly one.
  Chroma is upsampled nearest-neighbour, deeper bit depths left-justify the
  way `heifload` does, and a monochrome AVIF still returns three bands.
  Colour encodings that nothing in the tree can measure, which is 4:2:2,
  limited range, BT.709 and BT.601 above 8 bits, are refused rather than
  approximated.

  The decoder is `rav1d` (BSD-2-Clause), +16 lock entries, cheaper than `jxl`
  at +21 and `svg` at +29. It is taken with `default-features = false` so its
  `asm` feature stays off, which means no assembler is required and no native
  code is compiled: a debug build emits zero object files under
  `target/debug/build/rav1d-*`. The ISOBMFF container walk is hand-rolled
  rather than taken from `avif-parse`, which is MPL-2.0.

- **`MetadataValue::IntArray`, the array variant every animated codec was
  waiting on** (issue #787). `MetadataValue` had four variants and none of
  them could hold a per-frame `delay`, so #572, #573, #569 and #621 all had a
  page-geometry half they could land and a delay half they could not. It now
  has five, and the fifth is an ordered list of `i64`.

  The spelling is measured against the pinned vips 8.18.6 rather than read out
  of the C. `vips copy 'anim3.webp[n=-1]' out.v` writes
  `<field type="VipsArrayInt" name="delay">100 100 100 </field>`, one space
  after every element including the last, so that is what the writer produces
  and it is pinned as bytes. The reader is looser, because vips's is: a
  trailer carrying `40 60 80`, `40 60 80 ` or `  40   60   80  ` reads back as
  the same three elements in both libraries, and an empty element list is an
  empty array rather than a missing field.

  Two answers here are libviprs's own, and both are measured:

  - **an element that will not parse keeps the whole field opaque.** vips
    hands back an *empty* array for `40 x 80` (`vipsheader -f delay` prints
    nothing and `vips copy` writes the field back out empty), losing the two
    elements that did parse. libviprs carries the text through untouched, the
    same rule `gint`, `gdouble` and `VipsBlob` already follow when their text
    will not parse.
  - **the elements are `i64`, not `u32` or `i32`.** vips's `gint` is 32 bits
    and wraps rather than refusing: a trailer carrying `3000000000` reads back
    through vips as `-1294967296`, and
    `9223372036854775807 -9223372036854775808` as `-1 0`. A narrower carrier
    would lose data on a file libviprs did not write and could not warn about.

  `Raster::get_int_array` reads one borrowed, the way `get_int` does since
  #635, so reading a delay does not deep-copy whatever blob happens to sit
  under the same name. `MetadataValue::as_int_array` is the panicking
  accessor beside `as_blob`, `type_code` gets a fifth code, and `len` reports
  the element count.

  Naming the variant also releases files from the legacy JSON trailer. The
  fallback is keyed on what is *still* carried, so a `.v` whose only
  unnameable value was an `{"IntArray":[...]}` delay is read as a value now
  and its rewrite comes back out as the XML vips reads. Nothing about the
  format moved: #565's trailer already carried this exact field opaquely, and
  #609's `#[non_exhaustive]` already made the variant additive.

- **NIfTI (`.nii`) load** (issues #510, #641). `decode_nifti` reads both
  versions of the single-file form, NIfTI-1 and NIfTI-2, in either byte order,
  and `.nii` becomes a live row in the content sniffer, so `decode_bytes` and
  `decode_file` reach it without being told what the bytes are. There is no
  save half: the format is load-only here, the way Analyze and MAT are
  load-only in libvips.

  **The oracle is deliberately not libvips**, and that is measured rather than
  assumed. The pinned `vips` 8.18.6 reports `NIfTI load/save with libnifti:
  false` and registers neither `niftiload` nor `niftisave`, so a `.nii` handed
  to it falls through the sniffing chain to `magickload`, which guesses TGA.
  The reference is `nifti_clib` (`v3.0.1-91-g8f72d11`, the NIH implementation
  and the library libvips itself would have linked), captured in
  `oracle-captures/foreign-nifti/`, which re-measures the vips half on every
  run so a build that gains libnifti announces itself.

  What that buys is the *repair* rules, which are the part of this format a
  spec reading gets wrong. Non-finite `FLOAT32` samples are rewritten to zero
  before a caller sees them, so an infinity or a NaN stored in a file never
  comes back. `vox_offset` is truncated toward zero and floored at the header
  length, so `-8` and `100` both mean 348. `bitpix` is decoration and the
  datatype alone fixes the sample width. `scl_slope` and `scl_inter` are
  carried and never applied, because the scaling rule lives in FSL rather than
  in the reference. Rank 0 is a one-voxel image, a non-positive `dim[1]` is
  refused, and a zero extent on any higher axis is silently clamped to 1.

  And one where the capture's own prose was wrong and its measurements were
  right: on NIfTI-1 the byte order comes from `dim[0]`, not from the
  `sizeof_hdr` sentinel, with the sentinel only as a fallback. A file with
  only its four sentinel bytes swapped loads little-endian. That prose is
  corrected in the capture (issue #752) and the correction is held against
  this module by a test rather than by hope.

  NIfTI is a volume format and `Raster` is two-dimensional, so the axes above
  the second fold into the height, `dim[1]` wide by `dim[2] * .. * dim[rank]`
  high. That is `analyzeload`'s measured rule for the sibling format rather
  than an invention, it moves no bytes, and the collapsed axes stay readable
  as `nifti-dim[N]` metadata beside every other header field.

  Five datatypes have a carrier here (`UINT8`, `UINT16`, `FLOAT32`, `RGB24`,
  `RGBA32`) and the rest are refused **by name** through
  `NiftiError::UnsupportedCarrier`, naming the issue that would add the
  carrier, exactly as `crate::fits` refuses a signed BITPIX. `INT16` is the
  most common datatype in real NIfTI files and it is one of them: it needs
  #516. Narrowing it into 8 bits would lose data silently, which is worse than
  failing.

  The allocation budget is the interesting part rather than a checkbox. 348
  bytes can declare a 35-teravoxel volume in front of a 12-byte payload, so
  the declared geometry goes through `DecodeLimits::check_coord`,
  `check_pixels` and `check_image_alloc` before anything is reserved, and the
  refusal is the shared `SourceError::AllocLimitExceeded` rather than a sixth
  per-format variant.

  No new dependency. The whole format is a fixed-offset header and a raw
  array, so `crate::nifti` is field offsets, a byte-order flag and a copy
  loop; a NIfTI crate would supply the free half and leave every measured
  repair here anyway.

- `PixelFormat::kind()` and the `SampleKind` enum it returns (`U8`, `U16`,
  `F32`), plus `PixelFormat::with_kind()` alongside `with_channels()` (issue
  #607). Reach for `kind()` whenever the question is how to *interpret* a
  sample, and keep `bytes_per_channel()` for a stride or a buffer size.

  Byte width has been standing in for sample kind throughout the crate, and it
  cannot: four bytes means `f32` today and would mean `u32` under a uint
  carrier (issue #517) or `i32` under the signed ones (issue #516). A `match`
  keyed on the width needs a trailing `_` arm, and that arm reads a four-byte
  integer as a float without a word from the compiler. `SampleKind` gives the
  question one answer that a new carrier cannot slip past: every mapping off
  it is a total match.

  `SampleKind` also carries the per-kind constants the sample code used to
  keep private copies of: `bytes()`, `is_float()`, `max_value()`,
  `hist_bins()`, and `promote()`, which is the `vips__formatalike` order for
  a two-image op whose inputs disagree. `max_value()` and `hist_bins()` are
  `Option`, and `None` on `F32` is a statement rather than a gap: a float
  carrier has no depth-implied ceiling and no value-indexed bin table.

  `src/arithmetic.rs` and `src/histogram.rs` are converted and no longer name
  a byte width at all: no `bytes_per_channel()`, and no `with_channels()`
  either, since handing a width *back* to the constructor is the same
  ambiguity in the other direction. Nothing they do changes; what changes is that
  their sample readers and writers now fail to compile, rather than silently
  misread, the day a carrier arrives. The other 22 modules still key on the
  width and are tracked separately.

  `SampleKind` lives at `libviprs::pixel::SampleKind`.

- `SampleKind` names the four sample kinds no `PixelFormat` carries yet:
  `I8`, `I16`, `I32` and `U32` (issue #798, towards #516 and #517). Two new
  accessors come with them, `is_signed()` and `range()`, and `max_value()` is
  now derived from `range()` so the two cannot drift.

  The point of naming them before the carriers exist is that the answers are
  the part that has to be *measured*, and measuring costs nothing now while
  the carriers cost a crate-wide refactor. `promote()` is the case in point.
  It is `vips__formatalike`, swept on vips 8.18.6 with
  `vips boolean <a> <b> out and`, whose format table maps every integer format
  to itself so the output format is the formatalike result rather than a
  promotion of it. Four of the 36 integer pairs are ones "the wider kind wins"
  gets wrong: `(U8, I8)` is two one-byte kinds promoting to a **two**-byte one,
  `(I8, U16)` and `(U16, I16)` promote to **four** bytes, and `(U32, I8)` takes
  its sign from the one-byte operand.

  `PixelFormat::with_kind()` now returns `None` for a kind no format carries,
  rather than falling through to `with_channels(channels, kind.bytes())`, which
  would answer `Rgb16` for three bands of `I16` and `FloatF32(3)` for three
  bands of `U32`. That silent retag is exactly what `with_kind()` exists to
  prevent, so it refuses instead. `with_kind()` therefore has two reasons to
  answer `None` and a caller that needs to tell them apart has to look at the
  kind.

  `src/arithmetic.rs` and `src/histogram.rs` handle the new kinds for real
  rather than leaving a hole. Two behaviours are worth knowing. The rounding,
  saturating write in `arithmetic` takes its floor from `range()` instead of a
  literal `0.0`, since zero is the right floor for only three of the six
  integer kinds; nothing moves on the carriers that exist. And `histogram`'s
  bin-index read *folds* rather than widens, matching the `VipsStatisticClass`
  input cast, measured: a `char` image of `[-128, -1, 0, 127]` histograms to
  `bin 0 = 3` and `bin 127 = 1`, and a `uint` image whose largest sample is
  70000 gives a 65536-wide histogram.

  No `PixelFormat` produces any of the four, so nothing in the crate's
  behaviour moves. What moves is that the decisions are made, measured and
  pinned, so the carrier work in #516 and #517 is the `PixelFormat` variant and
  the 22 modules of #748, and not this as well.

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

- The crate has one fallible-plane helper instead of three private copies of
  it: `raster::try_plane`, its `_len` and `_filled` forms, and a single
  `cfg(test)` probe over all of them that a check addresses **by site label**
  rather than by position along a path (issue #696).

  `arithmetic.rs`, `convolution.rs` and `colour.rs` had each grown their own
  version of "reserve `len` elements fallibly and report
  `RasterError::AllocationFailed`", with three signatures and three separate
  test ceilings. `convolution.rs`'s had no ceiling at all when it was written,
  so that module's fallible paths could not be driven the way `colour.rs`'s
  were; `colour.rs`'s refused the *Nth* over-ceiling request on the thread, so a
  dozen checks that read as naming a buffer were really naming a position, told
  apart only by the byte sizes on the path happening to be unique. Three of the
  colour fixtures carry an extra band for no other reason, one pair of buffers
  could not be separated at all, and a check on either side of the boundary
  could only ever see the sites inside its own module.

  What lands here: `convolution.rs` and `colour.rs` on the shared helper, with
  `raster::alloc_op_output` and `Raster::try_f32_samples` reserving through it
  too, so an op output and a sample widening are on the same funnel as the
  intermediates. Every site now carries a label like
  `colour.import.lab_staging`, and `with_plane_cap_at` starves that one buffer
  and nothing else. Two checks that could not be written before are: the
  export fallback's PCS plane and its device buffer are the same 768 bytes on
  an RGB profile, and under the ordinal the first of them could be reverted to
  an infallible `Vec::with_capacity` with the suite green, because the other
  took the refusal and reported the same number.

  The ceiling also stopped answering before the reservation. It used to return
  early, which left `try_reserve_exact` and an infallible `reserve_exact`
  indistinguishable to every check that drove it, and is how fourteen of #689's
  guards came to pass with the fallibility they guarded reverted; it now
  poisons the request instead, so the same revert turns those checks red.

  `raster.rs` gains a cross-module funnel check that pins, per entry point and
  per module, exactly how many planes a path reserves. `try_sharpen` is the row
  it exists for: one call crosses all three modules, and its six reservations
  are the same six image-sized allocations
  `tests/convolution_image_sized_allocations.rs` charges to it from the
  allocator, so between the two nothing image-sized on that path is outside the
  fallible helper.

  One thing the mutation pass turned up while the funnel was being written, and
  it is fixed here rather than filed: `try_plane_filled` fills to a length
  computed from the geometry and its doc has said since it was
  `alloc_colour_plane_filled` that this is a contract and not `capacity()`,
  because `Vec::try_reserve_exact` may hand back more room than asked for. On
  this allocator at these sizes it never does, so `out.resize(out.capacity(),
  fill)` passed all eighty-one allocation checks in the crate. The probe now has
  an over-reserve knob that makes the allocator's licence happen on purpose, and
  the substitution goes red.

  No public API moves and no behaviour changes: everything here is
  `pub(crate)` or `cfg(test)`, and the error payloads each site reports are the
  ones it reported before. `arithmetic.rs`'s `try_scratch` is the copy still
  outstanding; that file is held by another lane, so #696 stays open on it.

- `src/resample.rs` records a fourth deliberate quantisation divergence from
  stock libvips and pins it from both sides (issue #777). `vips_reduce_make_mask`
  keeps a `short` fixed-point copy of every mask, truncated toward zero and
  **not renormalised**, and the reduce generators read it on both integer
  carriers. This module keeps the `f64` masks. Nothing about the behaviour
  moves; what lands is the measurement, three tests, and the argument in the
  module header, so the gap can neither grow nor quietly vanish.

  The short version of why: a constant image survives `reduce` here and does
  not survive it in libvips. Over a 32x32 constant 65535 `ushort`, six of
  fifteen kernel-by-shrink cells come back short on 8.18.6, `lanczos3` at
  shrink 4 by 128 of 65535. Against the convolution evaluated in compensated
  arithmetic at the same table offset, this module's mean absolute error is
  0.2558 of a level on the 16-bit carrier and libvips' is 10.1088, one-directional
  at a signed mean of -5.81, and libvips is the closer of the two on 0 of 43889
  interior samples.

- No test in the tree reaches the filesystem without `#[cfg_attr(miri, ignore)]`
  any more, and `UNANNOTATED_FS_EXCEPTIONS` is empty (issue #756).

  The four that were left are in `src/resample.rs`, which had four pull requests
  open against it while #739's sweep ran and so was the one module the sweep
  could not touch. Those merged, so these are annotated and their four rows in
  `tests/miri_fs_test_inventory.txt` flip to `annotated`. The file now records
  272 `annotated fs-detected` and 14 `annotated not-detected` tests, and nothing
  else.

  Emptying the list cost one further edit and no change to any assertion, which
  is the difference between an exception list and the floor it replaced. The
  floor, `assert!(unannotated_fs > 0)`, would have gone red here and demanded
  rewriting. What did go red, on purpose, is
  `merge_gate_states_the_backlog_as_a_bound_it_still_meets`: it has a separate
  arm for zero, because at zero the bound holds and `merge-gate.yml`'s sentence
  about a named handful of unannotated tests becomes false with nothing to catch
  it. That sentence is rewritten, once, and the failure named it.

- The Miri filesystem detector follows a call into a test helper, one file deep
  and to a fixed point, and 73 more tests over nine files carry
  `#[cfg_attr(miri, ignore)]` because of it (issue #781). 39 of those were the
  population when the change was written; the other 34 are `src/nifti.rs`,
  `tests/uhdr_ported_surface.rs` and `tests/page_model.rs`, which reached `main`
  while it was in flight and were caught by the new detector on the merge rather
  than by a re-read.

  `tests/page_model.rs` is the one worth naming, because its own module doc had
  written the gap down and deferred it: "three tests here reach the filesystem
  to read `src/`, and none carries `#[cfg_attr(miri, ignore)]` ... it belongs in
  that lane's sweep rather than here". It is one test, not three. The other two
  it counted go through `encode_vips` and `decode_bytes`, which are in memory,
  and through string literals declared inline. The detector was right about
  those and the note was not; it now says what was measured.

  It read one function body and stopped, which the guard's module docs listed as
  a known blind spot without ever measuring it. Measured: on the tree where
  every inventory row was annotated,
  `cargo +nightly-2026-08-20 miri test --test exr_ported_surface` still died in
  one second on `channel_names_and_compression_are_readable_downstream`, which
  calls `sample()` six lines above it, whose body is `std::fs::read(path)`. The
  same shape killed `tests/n_pages_meaning.rs`.

  `process_spawning_fns` had already solved this for `std::process`, so it
  becomes `reaching_fns`, parameterised on the marker list and on a scope
  predicate. The filesystem arm passes a predicate that accepts only test
  scaffolding: every function in an integration test, and only the
  `#[cfg(test)]` modules of a `src/` file. That restriction is the interesting
  part and it is measured rather than argued: with every function in scope, the
  way the process arm has it, the follower finds 85 unannotated tests over
  eleven files; with only scaffolding in scope it finds 39 over six. The
  difference is almost all one arm, `src/colour.rs` reading an ICC profile off
  disk inside the library, which would have marked all 23 colour tests that
  reach the loader whether or not any of them passes it a path.

  The `annotated not-detected` class halves as a result, from 22 rows to 13:
  those were annotations the detector could not have asked for, and nine of them
  it can now. What is left is the library boundary and the helper in another
  file.

- The filesystem half of the Miri convention is enforced rather than recorded:
  134 tests across 28 files carry `#[cfg_attr(miri, ignore)]` that did not, and
  `tests/miri_ignore_convention.rs` now refuses any filesystem-touching test
  that is neither annotated nor named in a four-entry exception list (issues
  #712, #739).

  #711 took `-Zmiri-disable-isolation` off the job. Under isolation a
  filesystem call is an unsupported operation and Miri ends the whole session
  on the first one rather than failing that test, so the 138 rows
  `tests/miri_fs_test_inventory.txt` carried as `unannotated fs-detected` ceased
  to be recorded debt and became 138 ways to take the gate down. Measured on
  `bd4bb1d`, `cargo +nightly-2026-08-20 miri test --test workspace_layout` died
  on `fuzz_crate_is_a_member_of_the_root_workspace` having run nothing.

  The guard's own floor had to go with it. It ended with
  `assert!(unannotated_fs > 0)` and a message saying that if the count ever
  reached zero the ledger had stopped being a ledger, which is a floor that
  goes red on the change that clears the debt. It is now a refusal with an
  exception list, checked in both directions: a filesystem test that is neither
  annotated nor named fails, and a named entry that is no longer an unannotated
  filesystem test fails too, so the list cannot rot into decoration. It carries
  four names, all in `src/resample.rs`, which had four pull requests open
  against it while the sweep ran; issue #756 carries them.

  This does not make the `miri` job report, and the reason is worth writing
  down because it is the half of #675 nobody had measured. `cargo miri test`
  runs the `--lib` target first and libtest runs it in sorted order, so the
  first two tests of the whole invocation are
  `arithmetic::proptests::every_try_method_in_the_module_is_in_the_sweep` and
  `arithmetic::proptests::no_try_method_panics_on_a_float_raster`. Neither
  touches the filesystem, so no annotation sweep can reach them, and the second
  is the proptest already measured at over twenty minutes without finishing.
  The unannotated filesystem tests were never the first wall of the whole-suite
  run, they were the first wall of every target after it.

- The doc gate denies `rustdoc::private_intra_doc_links`, and the 33 public doc
  comments that pointed at `pub(crate)` items no longer do (issue #697). That
  lint is warn-by-default and neither invocation denied it, so a public doc
  comment could link to a private helper, rustdoc would silently drop the link
  and render it as inert bracketed text on docs.rs, and both `make doc` and the
  CI docs job stayed green while publishing a dead pointer.

  At `9b1ade6` that had happened 33 times across 13 files: `sink.rs` 7,
  `source.rs` 6, `resume.rs` 5, two each in `colour.rs`, `dedupe.rs`,
  `encode_tiff.rs`, `engine.rs` and `gif.rs`, one each in `composite.rs`,
  `manifest.rs`, `pdf.rs`, `raster.rs` and `streaming_mapreduce.rs`. Every
  target is a private helper, a crate-internal constant or a `pub(crate)`
  cache, and none of them is worth making public just to satisfy a link, so
  each site inlines the sentence the public reader needed and keeps the
  identifier in plain backticks for anyone reading the source. `cargo doc
  --no-deps --all-features` goes from 47 warnings to 13, the remaining 13 all
  being `rustdoc::redundant_explicit_links` (issue #795).

  `tests/doc_link_gate.rs` holds the `Makefile` recipe and the `ci.yml` docs
  job to the same deny set and the same `cargo doc` arguments, so tightening
  one file alone fails there rather than quietly un-mirroring the local gate,
  and it holds the docs job's own `name:` to naming every lint it denies.

- The doc gate denies `rustdoc::redundant_explicit_links` too, and the 13 links
  that carried a redundant explicit target no longer do (issue #795). Each was
  written `[`Foo`](crate::path::Foo)` where the label alone already resolves to
  the same destination, in `engine_builder.rs` (4), `engine.rs` (2), `jxl.rs`
  (2) and one each in `draw.rs`, `sink.rs`, `sink_object_store.rs`,
  `stream_verify.rs` and `verify.rs`.

  Nothing rendered wrong, so this is not a rendering fix. It is that 13 standing
  warnings is a floor which hides the fourteenth, and a warning stream nobody
  reads is not a gate: that is exactly how the 33 private links above
  accumulated unnoticed. `cargo doc --no-deps --all-features` is now **silent**,
  so anything it prints is new.


- `spcor` and `fastcor` stopped widening the whole image and stopped
  materialising their results twice. Both read the image as a sliding window of
  the template's rows, which is the same access pattern the convolution
  traversal has, so both now share its row window; and both filled a whole
  `Vec<f64>` in output order only to hand it to a builder that walked it once,
  so both write into the output raster directly instead. At 4000x4000 `Rgb8`
  with a 32x32 template, `spcor` peaks at 238 MiB rather than 967 MiB, 5.2 times
  the input rather than 21, and `fastcor` reads the same. No output byte moves
  (issue #791).

  What is left whole is the **template**, which both read in full at every
  output sample. That one is bounded by the operand a caller passes rather than
  by the image.

- `compass` stopped keeping a widened copy of every result. It convolves
  `times` times and combines the absolute results, and it used to widen each of
  those results to `f64` first and hold all `times` widenings live at once, to
  read each sample once. That made it the most expensive operation in the crate:
  at 4000x4000 `Rgb8` with a 3x3 box mask and `Combine::Max`, the libvips
  default `times = 2` peaked at 1.61 GiB over a 48 MB input, 36 times what it
  was handed, and `times = 4` at 2.42 GiB (issue #790).

  Each result is folded into the accumulator off its own bytes now, and the
  integer branch builds its output raster from an iterator rather than from a
  whole `Vec<i64>` of clipped samples. `times = 2` peaks at 556 MiB and
  `times = 4` at 647 MiB, so 12x and 14x, and no output byte moves.

  Holding every result is inherent to `vips_compass` and is what is left:
  `times * bands` bytes a pixel, bounded by the `1..=1000` range the operation
  already enforced on `times`.

- Convolution stopped widening the whole image. `conv`, `convsep`, `gaussblur`,
  `compass`, `sobel`, `scharr`, `prewitt` and canny's gradient stage all run one
  shared traversal, and that traversal used to decode the entire source to `f64`
  before it started: eight bytes a sample where a uchar carries one, which made
  it the largest allocation in the crate. It keeps a rolling window of the rows
  the mask actually reaches instead, `min(h, mask height)` of them, each source
  row widened exactly once on the way past (issue #575).

  Measured on a 4000x4000 `Rgb8` input, release, peak resident set:

  | operation | before | after |
  |---|---|---|
  | `conv`, 3x3 box, integer | 464 MiB, 10.1x the input | 98 MiB, 2.1x |
  | `conv`, 3x3 box, float, `Rgb16` | 693 MiB, 7.6x | 327 MiB, 3.6x |
  | `sobel` | 464 MiB, 10.1x | 98 MiB, 2.1x |
  | `gaussblur`, sigma 3, integer | 510 MiB | 145 MiB |

  Not one output byte moves. The window holds the same values in the same
  order and the accumulation is untouched, so every pinned oracle capture and
  every FNV hash in the module reads exactly what it read before.

  The allocation counts moved with it: a `conv` at integer precision now makes
  **one** image-sized allocation, its own output, and `canny` makes eight rather
  than eleven. `tests/convolution_image_sized_allocations.rs` (renamed from
  `sharpen_canny_image_sized_allocations.rs`, since the budgets are no longer
  only those two) pins all sixteen rows at two image sizes.

  `Raster::try_compass` is the one operation that did not move, because its
  combine reads all `times` results at the same sample and so has no row window
  to keep. It has a row in the budget file saying so, at 159 bytes a pixel over
  a three-byte-a-pixel input, and an issue of its own.

- The page split no longer survives an operation that changes the raster's
  height, and is no longer imported from a second input by a multi-input op
  (issue #564). Both are deliberate divergences from libvips and both have a
  measured counter-example on 8.18.6:

  - `vips resize` on a four-page 4x12 roll writes a 2x6 result still claiming
    `page-height: 3`, and `gifsave` then writes that as a **two**-frame
    animation whose frames are two half-height frames stacked, silently.
    `Raster::carry_meta_from` drops the split on a height change instead, so
    the same pipeline yields a still image: the safe half of the two wrong
    answers, and the caller can see it in `pages_loaded`.
  - `vips join plain.v paged.v out.v horizontal`, where only the **second**
    input is a four-page roll, produces an 8x12 output carrying
    `page-height: 3`, `n-pages: 4` and the roll's delay array, so an unpaged
    image silently becomes a four-frame animation.
    `Raster::merge_fields_from` is the one name the field union does not
    import.

  Nothing in the crate attaches `page-height` yet, so this changes no current
  behaviour; it is the contract the animated GIF, WebP and JPEG XL lanes are
  written against.

- The `miri` job in `.github/workflows/merge-gate.yml` no longer runs with
  `-Zmiri-disable-isolation`, and `make miri` is now a local mirror of it that
  actually runs (issues #675, #707). Neither change makes the job pass. What
  they change is that it fails in a couple of minutes with something to act on,
  instead of running to the 90 minute ceiling and reporting `cancelled`.

  The flag was added with the claim that it was not a coverage loss, on the
  reasoning that only *unannotated* filesystem tests stop aborting the run so
  the job covers strictly more. That was never measured and it is false:
  letting those tests execute under the interpreter is what pushed the run past
  the ceiling, so the job covered nothing at all. Three consecutive dispatched
  runs died at 90 minutes, and the run before the ceiling existed went 4h13m.

  `make miri` could not run at all before this. It was missing `-A deprecated`,
  so it died on the denied `AtomicU64::fetch_update` rename under nightly
  (#643), and it invoked the floating `+nightly`, which on the machine this was
  written on resolves to 1.96.0-nightly, below the crate's 1.97 MSRV, so cargo
  refused to build before Miri was reached. It now takes a `MIRI_TOOLCHAIN`
  that defaults to a dated nightly, and checks whatever it resolves to against
  the MSRV read out of `Cargo.toml`, so a toolchain that cannot work says so in
  one line instead of printing the MSRV refusal once per target.

  `RUSTFLAGS` gains `--cfg sha2_backend="soft"`, which pins sha2's portable
  backend. Under Miri, `cpufeatures` compiles to `cpufeatures-0.3.1/src/miri.rs`
  and chooses nothing at runtime: the detection macro becomes
  `cfg!(all(target_feature = ...))` and the probe a constant `false`, so sha2's
  backend is fixed at compile time by the target's baseline features. Per
  `rustc --print cfg`, `aarch64-apple-darwin` carries `target_feature="sha2"`
  and `x86_64-unknown-linux-gnu` carries none of `sha`, `ssse3` or `sse4.1`. So
  the pin is what keeps the run off sha2's NEON path locally, where it reaches
  `vld1q_u32(&K32[0])` and aborts about 30 seconds in on a Stacked Borrows
  violation, a 16 byte load through a `&u32` whose retag covers four; and on
  the hosted x86_64 runner it changes nothing, because the portable backend was
  already what got compiled. That shape is on file against Miri itself as
  rust-lang/miri#3900, closed as not planned. `-Zmiri-tree-borrows` clears it
  too, measured, and would be a defensible answer; the backend pin is simply
  the smaller of the two changes.

  What the job did after this change was abort on the first filesystem test
  that had no `#[cfg_attr(miri, ignore)]`, of which
  `tests/miri_fs_test_inventory.txt` recorded 138. Annotating them is #712 and
  #739, below. Whether the suite then fits inside `timeout-minutes: 90` was
  open, and one measurement said not to assume it would: the single proptest
  `arithmetic::proptests::no_try_method_panics_on_a_float_raster` ran over
  twenty minutes under the interpreter without finishing, and it touches no
  filesystem, so no annotation sweep will ever reach it. That is the one that
  turned out to decide the answer.

  Three claims in that workflow file were false when I got here and are gone.
  It said Miri "cannot run on the dev machine", which was true of the reason
  given and stopped being true of the conclusion. It said the tree carries 48
  annotations across seven modules, where the inventory recorded 53 across
  eight. And it said dropping the isolation flag was a coverage win. It now
  quotes no count at all: `tests/miri_invocation_parity.rs` holds it to a bound
  and sends the reader to the inventory for the number, because quoting the
  live figure made an unrelated workflow file a mandatory edit for every pull
  request that adds a filesystem test.

  `tests/miri_invocation_parity.rs` also holds the two invocations to the same
  command and the same `MIRIFLAGS`/`RUSTFLAGS`, merged across every scope they
  can arrive from: the workflow, job and step `env:` blocks on one side, and
  file-level make variables on the other. It compares the command rather than
  the compiler, which it cannot: the hosted job resolves
  `dtolnay/rust-toolchain@nightly` on the day and the local mirror pins a date,
  so a local green is evidence about the crate rather than a prediction of the
  hosted run.

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

- **Six tests reached the filesystem with no `#[cfg_attr(miri, ignore)]`, and
  `merge-gate.yml` said none did** (issue #765). Any one of them ends the whole
  Miri session on its first syscall, because #711 turned isolation on and Miri
  aborts the run rather than failing the test.

  The scanner in `tests/miri_ignore_convention.rs` could not have asked for
  those annotations. It refuses to follow a call into the library on purpose,
  since a production function that *can* open a path is not evidence that this
  caller hands it one, and the measurement behind that choice is 46 spurious
  marks if it does. So the six read as pure: three in `src/analyze.rs`, whose
  two-file entry point resolves the `.img` from the `.hdr`'s path and therefore
  has no buffer form to test, and three in `src/colour.rs` and `src/pdf.rs` that
  hand an entry point a path which does not exist and assert on the error. Those
  last three look like tests that never reach disk, and the `open` still
  happens: Miri refuses the syscall before the kernel can answer `NotFound`.

  Found by measuring rather than by reading. I ran every test binary single
  threaded under a `DYLD_INSERT_LIBRARIES` interposer on `open`, `openat`,
  `opendir`, `stat`, `lstat`, `access`, `mkdir`, `unlink`, `rename`, `symlink`,
  `link`, `rmdir`, `readlink` and `chmod`, printing each path between libtest's
  own `test NAME ...` and `ok` so every syscall lands on the test that made it.
  2143 tests ran, 264 touched the filesystem, and six were in neither the
  inventory nor the annotated set.

  The same measurement retires the rest of #765's claim. It was filed when 21 of
  `src/exr.rs`'s 22 tests and all of `src/nifti.rs` reached fixtures through a
  `fixture()` helper the detector could not follow; #781 closed that, and the
  interposer confirms it, since not one `exr` or `nifti` test comes back
  untracked.

- **A `#[cfg(test)]` helper that is not inside a `#[cfg(test)] mod` was outside
  the filesystem follower's call graph entirely** (issue #833). The scope
  predicate matched the attribute only when it sat on a `mod`, so a free `fn`
  under it was filtered out and a test calling it read as pure however plainly
  the helper called `std::fs::read`. Thirteen such helpers exist in `src/`
  today, in eight files; none touches the filesystem, so widening the predicate
  to any `#[cfg(test)]` item moved no count and no inventory row, and that is
  luck rather than design.

- **The Miri guard annotated one of its own tests for a filesystem access it
  never makes** (issue #832). `the_filesystem_detector_follows_a_test_helper_but_not_the_library`
  carried `// reads the repository source tree`, copied off the four siblings
  that call `scan_repo()`; it calls `scan_source` on two inline `&str` fixtures
  and made zero filesystem syscalls under the interposer, against thousands for
  each of those siblings. That is one test the Miri gate can now actually run,
  and one ledger row that had stopped meaning anything.

- The morphology walkers dispatch on `SampleKind` instead of on the byte width
  (issue #831, part of #748 and #607 step (b)). `sample_u32` and its write side
  stepped **two bytes per sample for every width that is not one**, which is the
  right stride for `u16` and half the right stride for any four-byte kind, so a
  32-bit carrier would have walked the wrong pixels rather than merely read the
  wrong type.

  The half-stride walk was not reachable today and would not have been reachable
  under a `U32` carrier either: `try_rank`, `try_countlines` and
  `try_label_regions` all refuse `bytes_per_channel() == 4` first. The kinds a
  width test cannot see are the signed ones, and those pass every one of those
  guards: a one-byte signed raster would have been read as unsigned, and in the
  rank window every negative sample sorts above every positive one. `morph`'s
  own 8-bit guard had the same shape one width down. All four are keyed on the
  kind now, through one `unsigned_8_or_16` predicate that is total over the
  enum, and the two sample helpers match the kind with no wildcard arm.

  Nothing moves for the three kinds a `PixelFormat` carries today. The float
  refusal for `rank`, `countlines` and `label_regions` had no test before this,
  which is why breaking the guard stayed green; it has one now.

- Six modules read and write samples through the sample kind instead of the
  byte width (issue #840, part of #748 and #607 step (b)). `composite`,
  `create`, `freqfilt`, `mosaicing`, `raster_ops` and `textio` each carried
  their own copy of the same three-arm `match`, whose trailing arm reads four
  bytes as an `f32` whatever those bytes are, so a `u32` sample of `1` came
  back as `1.4e-45`. Six copies of one function is the reason a new carrier
  would be a six-place edit; they go through one `read_sample_f64` now, whose
  match has no wildcard arm.

  Three of the sites do more than read. `composite` takes its output depth from
  `SampleKind::promote` rather than the wider of the two byte widths, and its
  write-back clamp from `SampleKind::range` rather than a literal ceiling per
  width, so a signed carrier would saturate at its own floor instead of at
  zero. `mosaicing`'s merge dispatches the feathered blend on the kind, and the
  four kinds with no `BlendSample` implementation get the new typed
  `MosaicError::UnsupportedSampleKind` instead of being blended as float.
  `mosaicing` also wrote `bytes_per_channel()` into the `VMJ1`
  `mosaic-join-tree` header and read it back through the width-keyed
  constructor, which is the `.v` `BandFmt` shape one layer in; that byte is a
  sample-kind code now, keeping `1`, `2` and `4` for the three kinds that exist
  so every blob already written still parses.

  Nothing moves for the three kinds a `PixelFormat` carries today, and the
  mutation sweep that says so found three gaps on the way: breaking the 16-bit
  write in `new_from_image`, dropping the stride in `elem_f64`, and blending a
  `Gray16` merge as `u8` all left the whole suite green, because every merge
  fixture in the module was 8-bit or float. All three have a test now.

- `hist_find` sizes a 16-bit histogram from the data instead of from the depth,
  and `hist_equal` follows it (issues #803, #823). Measured on vips 8.18.6,
  `vips hist_find` of a `ushort` `[4096, 4096, 9]` gives width **4097** where
  libviprs gave 65536: 65536 is the ceiling of the rule, not the rule. `uchar`
  really is a fixed 256 even when the data maxes out at 3, and that half is
  unchanged.

  It follows the band selection too, which is the case a whole-image test
  cannot separate: on a 16-bit image whose band 0 maxes at 10 and band 1 at
  5000, `hist_find` is 5001 wide over both bands, `hist_find_band(0)` is 11 and
  `hist_find_band(1)` is 5001. `hist_find_indexed` is sized the same way from
  its index image.

  `hist_equal` fuses `maplut(hist_norm(hist_cum(hist_find)))` into one pass and
  was taking its table width from the depth, so it stopped being that
  composition the moment `hist_find` moved. The visible consequence is at the
  constant image: measured, a constant `uchar` band equalises to `255` and a
  constant `ushort` band equalises to **itself**, because a table one value wide
  normalises that value's single cumulative entry back to it. The doc said "a
  constant band maps to the depth maximum" without the qualifier.

  `bins_for` stays as it was, and the two functions now answer different
  questions on purpose: `hist_find_ndim` uses it as the value **range** it
  scales samples by, and that range is the depth's rather than the data's,
  measured (a `ushort` `[0, 5, 10]` and a `uchar` `[0, 5, 10]` both put all
  three samples in bin 0 at 10 bins).

  **Migration.** A caller reading `hist_find`'s width, or indexing bins beyond
  its own data's maximum, gets a narrower image for 16-bit input. `maplut`
  already clips an out-of-range index to the last LUT entry, as libvips does,
  so the equalisation chain absorbs the narrowing on its own.

- `hist_plot` plots one row too many for every histogram that is not 8-bit
  (issue #802). Measured on vips 8.18.6, `vips hist_plot` of a `ushort`
  `[2, 0, 3]` gives a **3x3** image where libviprs gave 3x4: the height is the
  largest count, floored at one, not `max + 1`. The 8-bit fixed height of 256
  was right and is unchanged.

  The doc said the old number matched libvips, and nothing checked that.
  `hist_plot_bar_geometry` pinned libviprs's own answer instead, so the claim
  and the test agreed with each other and with the code, and with nothing else.
  Both now compare against a measured sweep: `[0, 1]`, `[1, 1]` and `[0, 0, 0]`
  plot 1 row, `[3, 9]` plots 9 (not 6, so the floor is a literal zero rather
  than the smallest count), and `[65535, 0]` plots 65535.

  **Migration.** A caller reading the plot's height, or indexing rows from the
  top, gets one row fewer for a 16-bit histogram. Bars still grow from the
  bottom.

- **`uhdr::uhdr_to_scrgb` scales the gain map through `crate::resample`**
  instead of a private linear interpolator (issue #760). `uhdr2scRGB` scales
  the gain map with `vips_resize(..., VIPS_KERNEL_LINEAR)`, and `vips_resize`
  is not a bilinear point sample: below 1.0 it runs `reduce`, which averages
  every input sample an output covers. #508's copy interpolated between two
  neighbours at any scale, which is right at scale 1 and scale 2 (the only
  ones the oracle capture pins, and both still bit-exact) and wrong anywhere
  else. Measured against `vips resize` 8.18.6 on a 12x9 gain map scaled onto a
  4x3 base: the copy missed **12 of 12** levels, the worst by 87 of 255; the
  shared resampler misses none.

  A gain map larger than its base is reachable, because `from_container` reads
  whatever a file holds even though nothing writes one.

  New `UhdrError::Resample` for a ratio the resize refuses, and
  `UhdrError::BadInput` when the resize lands on a size other than the base's,
  which would otherwise have been a short read.

  This also corrects the attribution in #508's own measurement: the residual
  between a container expanded here and by `vips uhdr2scRGB` is the two JPEG
  decoders, not the resampler. Hand this module the halves vips decoded and
  the two agree to `f32` ulp. See the `crate::uhdr` module docs for the table.

- **Every operation in `crate::resample` carries the input's metadata onto its
  output** (issue #789). `resize`, `shrink`, `shrinkh`, `shrinkv`, `reduce`,
  `reduceh`, `reducev`, `affine`, `similarity`, `rotate`, `mapim` and
  `thumbnail_image` all built their result with a bare `Raster::new` and
  carried nothing: no interpretation, no resolution, no orientation, no ICC
  profile and no field a caller attached. vips carries all of them, measured on
  8.18.6 across two image shapes and fifteen ops.

  The resolution is carried **verbatim** rather than rescaled with the factor,
  which is what vips does and what #690 already measured for `zoom` and
  `subsample`.

  It is not only tags. #664 made the premultiply bracket read the
  interpretation on a float carrier, because scRGB's alpha maximum is 1.0 where
  sRGB's is 255, so while the tag was being dropped `resize(0.5).resize(0.5)`
  read a different alpha ceiling on the second call from the first. An 8x8
  `RgbaF32` chequerboard resized to half, twice, differed in 33 of 256 output
  bytes on the tag alone, with both outputs coming back untagged.

  `crate::resample`'s two thumbnail paths lose the `copy().interpretation(...)`
  restamps they carried to work around this, which also removes two
  image-sized clones from the linear and ICC thumbnail pipelines.

- **Animated WebP frames after the first no longer decode one grey level
  low.** `image-webp` 0.2.4 runs its approximate alpha blend on fully opaque
  pixels, where libwebp explicitly does not (`demux/anim_decode.c`,
  `BlendPixelRowNonPremult`, which tests `src_alpha != 0xff` before blending).
  With `src_a = 255` the approximation is `(s * 255 * ((1 << 24) / 255)) >> 24`,
  which is `s - 1` for every `s` from 1 to 255, so every opaque channel of a
  blended frame came back one low. `vips webpsave` writes frame 0 with
  blending off and every later frame with it on, so a four-page roll read back
  `74 20 38` where `vips rawsave 'x.webp[n=-1]'` read `75 21 39`, on every
  page but the first.

  libviprs now switches blending off on the frames that provably carry no
  transparency before handing the bytes to the decoder: a `VP8 ` frame is
  lossy and has no alpha channel, and a `VP8L` frame declares one in its
  `alpha_is_used` header bit. Blending a fully opaque frame is the identity,
  so clearing the bit cannot change the image and it routes the decoder onto
  its exact copy path. The input is cloned only when there is a frame to
  rewrite.

  What is left is a frame that declares alpha *and* asks to be blended, where
  the opaque pixels inside it are still one low. `vips webpsave` does not
  write that combination (a transparent roll comes out with blending off on
  every frame, measured), so no oracle fixture reaches it, and it is written
  down at `disable_blending_on_opaque_frames` rather than hidden.

- `EncodeError::Unsupported`'s own documentation no longer names four formats
  this crate encodes (issue #758). The variant's doc listed UHDR, FITS,
  JPEG-XL and JP2K as "genuinely-external formats that have no mature pure-Rust
  encoder", which made the variant's *contract* wrong rather than merely stale:
  `crate::uhdr` has written an Ultra HDR container since #508 with no new
  dependency at all, `crate::fits` hand-rolls FITS, and `crate::jxl` and
  `crate::jp2k` carry real pure-Rust codecs behind their features. The type's
  own doc block carried the same list.

  A new guard, `the_unsupported_doc_lists_name_no_format_this_build_encodes`,
  extracts both lists from the source and probes each named format by calling
  its encoder, so "this build encodes it" is measured rather than declared and
  the lists cannot drift again.

- **Every `capture.py` under `oracle-captures/` now checks `ORACLE_PIN.json`
  before it writes anything. Two of the fourteen did** (issue #796), both of
  them the convolution scripts `oracle_pin.py` was factored out of. The pin
  file said "capture.py refuses to run against a binary that disagrees with
  it", `oracle_pin.py` opened "The oracle pin every capture.py under
  oracle-captures/ checks", and `tests/oracle_capture_pins.rs` said "each
  area's". All three were true for `convolution`.

  #650 left a two-sided guard: the capture script stops a bad capture being
  taken, and the Rust test stops one being kept. Only the second side existed
  for the twelve `foreign-*` areas, so re-running any of them on a machine
  whose vips had moved wrote a whole capture with the new version stamped
  through it and told nobody, which is the exact failure #650 was filed for.
  All six areas still marked `pre_pin`, the ones most likely to be re-run,
  were in the unguarded twelve.

  `every_capture_script_checks_the_oracle_pin` is what stops it coming back.
  It reads the scripts through `include_str!` and matches at column zero,
  because `oracle_pin.py`'s docstring shows callers the exact lines to write
  and a substring scan would read that example as an adoption. No committed
  capture changes: re-running `foreign-avif` with the check in reproduced its
  `oracle.json`, `commands.sh` and all thirteen fixtures byte for byte.

- **The AVIF oracle recorded a sha256 for an `rgb8.avif` that was not the file
  in the tree** (issue #779). Two records in
  `oracle-captures/foreign-avif/capture.py` wrote different images to
  `fixtures/rgb8.avif`: the bit-depth carrier saved the 16-bit ramp narrowed to
  8 bits, and the lossless-identity record then saved the 8-bit ramp over the
  top of it. The later write won, so the carrier's row went on recording
  `d5a55b1a…` / 323 bytes for a file that was `c1f34aad…` / 355 bytes, and its
  `read_back` and `source_16bit` arrays described an artefact nobody could
  open.

  The narrowed image is now `fixtures/rgb8_narrowed.avif` and it is committed.
  Re-running the capture against the pinned vips 8.18.6 reproduced
  `d5a55b1a…` / 323 bytes exactly, so the carrier row was measured against the
  narrowed image all along and only lost the file; the identity row was
  measured against the committed `rgb8.avif` and was right. Of 1890 leaves in
  that `oracle.json`, the re-run moved two, both the 8-bit row's file name, and
  left all twelve existing fixtures byte-identical. `capture.py` now refuses to
  write any name under `fixtures/` twice, so the next collision stops the
  capture instead of quietly losing an artefact.

  The half that matters is the guard, because nothing was looking.
  `tests/oracle_capture_pins.rs` now hashes every committed file a capture
  names and compares it to what was recorded, across every area: 95 rows, of
  which exactly one disagreed. A second test reads the same defect off the JSON
  alone, so a collision under `outputs/` or on a path outside the repository is
  caught too, with no file to compare against. A green suite used to mean "the
  recorded vips versions line up"; it now also means the pins describe the
  tree.

- `SourceError::is_alloc_limit`'s documentation no longer lists WebP among the
  containers whose allocation refusal is spent inside the `image` crate (issue
  #782). It has not been one since #686: WebP is decoded by libviprs, prices its
  own frame, and reports `SourceError::AllocLimitExceeded` with the declared
  geometry attached. The predicate itself was right the whole time, so nothing a
  caller wrote against it breaks; the bullet list beside it sent anyone matching
  by shape to the wrong arm.

  The list is pinned to the tables in `tests/decode_alloc_refusal_shape.rs` now.
  Nothing held it before, because those tables pin their own size and what their
  rows report, and neither of those sees a format moving out of one and leaving
  its description behind.

- **`profile`'s docs claimed its 16-bit saturating output matched "the libvips
  `ushort` output". libvips emits `VIPS_FORMAT_INT`** (issue #759), measured on
  8.18.6 for every one of the eight input formats. The word matters more than
  it looks: `INT` is the *signed* 32-bit carrier, so `profile` is a payoff of
  the signed carriers (issue #516), not of the uint one (issue #517).

  Two neighbouring claims were under-specified in the same direction and are
  corrected with the measured tables. `project` promotes to `UINT` for the
  unsigned inputs, `INT` for the signed ones and `DOUBLE` for the float ones,
  so it needs both carrier families rather than just uint. The histogram
  module's "libvips stores counts in 32-bit unsigned samples" swept in
  `hist_find_indexed`, which emits `DOUBLE` for every input format and either
  `combine` mode, and `hist_cum`, which follows its input across all four.

  No value or format changes here: the saturation at `65535` stays until a
  wider carrier lands. What changes is that the claims now have checks under
  them. `profile` and `project` had no assertion on their output format
  anywhere in the crate and `profile` had no saturation test at all, which is
  how the wrong sentence survived. Six counter ops get a format pin and two
  get a ceiling pin carrying the measured vips answer beside the libviprs one.

- The native `.v` reader applies `DecodeLimits::max_alloc_bytes` to the pixel
  body it copies out of the file, priced from the declared header geometry
  through the same `DecodeLimits::check_image_alloc` every other self-priced
  decoder uses (issue #710). It applied `max_coord` and `max_pixels` and then
  nothing else, so a 36-byte raster decoded clean under a 35-byte ceiling and
  `.v` was the one container out of ten where setting the budget bought a
  caller nothing.

  **`.v` was never a decompression-bomb vector**, and that is worth saying
  because the obvious reading is wrong. The reader refuses a header promising
  more pixel data than the file physically holds, so the allocation was already
  bounded by the input length, and no crafted small file ever got past it. What
  was missing was the contract, in two visible ways. `Raster::new`'s 8 GiB
  construction budget was the only ceiling in force, fifteen times the 512 MiB
  decode default. And the two decode entry points disagreed about the same run
  of bytes: `decode_file_with_limits` spends the budget on the bounded
  whole-file read, `decode_bytes_with_limits` has no file to spend it on.
  Measured before the change:

  ```text
  bytes 4x4 budget=47 (price 48) -> Ok((4, 4))
  file  4x4 budget=47 (price 48) -> Err(AllocLimitExceeded {
      what: "image file body", needed_bytes: 112, max_alloc_bytes: 47 })
  ```

  **What changes for a caller.** Only `decode_bytes_with_limits` and
  `decode_bytes`, and only on a `.v` whose pixel body is over the budget. The
  file entry points cannot change: a `.v` file is always its 64-byte header
  plus the body plus any trailer, so a budget under the body's price is under
  the file's length too and the whole-file read refuses first. On the in-memory
  path a `.v` body over `max_alloc_bytes` now comes back as
  `SourceError::AllocLimitExceeded { what: ".v pixel buffer", .. }` with the
  declared geometry attached, where it used to decode. At the 512 MiB default
  that is a `.v` over half a gigabyte handed to the crate as bytes.

- `affine`, `mapim` and any `resize` above 1.0 with a bicubic upsize kernel are
  now byte-identical to `vips affine --interpolate bicubic` on a `uchar` raster
  with no alpha band (issue #704). `vips_interpolate_bicubic_interpolate` sends
  that carrier to `bicubic_unsigned_int_tab`, which reads
  `vips_bicubic_matrixi` (the Catmull-Rom coefficients truncated to 12-bit
  fixed point) and accumulates as integers a row at a time, closing each row
  and the column combine with `unsigned_fixed_round`. This module evaluated the coefficients in
  `f64` at the grid offset #668 put them on, which is the last systematic
  divergence on that path.

  **This is deliberately less accurate, and that is the trade.** Against
  Catmull-Rom evaluated at the true sub-pixel offset in exact rational
  arithmetic, over 17814 interior samples of random `uchar` images, the mean
  absolute error goes from 0.4371 LSB to 0.4798 and the worst case stays at
  1 LSB. Some samples move the other way: vips is the closer of the two on
  1355 of those 17814. The error both spellings already share from #668's
  1/64 offset grid is 0.44 LSB, ten times the difference this makes.

  What it buys is a gate that can see a regression. The bicubic allowance in
  `affine_interpolators_match_libvips_oracle` goes from 30 bytes at delta 1 to
  **zero**, joining `nohalo` and `lbb`, so a future 1-LSB drift on this path
  goes red instead of landing inside a tolerance. That is the failure #668
  itself documented: a false comment plus a tolerance wide enough to absorb it
  is how a 2.3-magnitude divergence survived.

  It is one carrier, not "the integer carriers". `USHORT` and `SHORT` take
  `bicubic_unsigned_int32_tab`, which reads the `double` table, and an alpha
  band routes through a premultiply into FLOAT first, so neither ever sees the
  fixed point. Three tests pin those carriers so the new path cannot spread.

- `affine`, `mapim` and `resize` are now byte-identical to
  `vips affine --interpolate bicubic` on a **float** raster too, and on any
  raster with an alpha band (issue #705). `bicubic_float<T>` sums each of the
  four rows through `cubic_float<T>` and combines them through `cubic_float<T>`
  again, and that helper returns `T`. Its arithmetic is `double` either way, so
  with `T = float` all five sums are computed in `f64` and narrowed to `f32` on
  the way out. This module accumulated in `f64` and narrowed once at the store.

  The issue asked for the accumulation *order*, and that turned out to be a red
  herring worth exactly zero bits: flat 16-term `f64` and row-then-column `f64`
  are bit-identical, 0 of 1764 samples apart on a random 24x24, and both miss
  the binary by the same 1.5259e-05 in the same 356 samples. Adding the per-row
  narrowing takes that to 0 of 1764.

  An alpha band comes along because `vips_affine_build` premultiplies into a
  FLOAT image before it resamples, so an `Rgba16` raster takes the narrowing as
  well. That is worth about 3 samples in 480 on real data, always on a rounding
  boundary, and an `Rgba8` raster cannot see it at all because an 8-bit quantum
  swallows an `f32` ulp whole.

  Nothing else moves: the 16- and 32-bit integer carriers reach
  `bicubic_float<double>`, which narrows nothing, and `BILINEAR_FLOAT`, `lbb`
  and `nohalo` are one expression with a single narrowing at the store and were
  already bit-exact.

- Two more places where libvips quantises more coarsely than this module are
  now measured, pinned and **kept** (issues #732 and #733), and the rule that
  decided them, and that decided #704 the other way, is written into the module
  docs. Against the exact answer in rational arithmetic, on real `affine`
  output:

  | | this module | libvips | libvips closer |
  |---|---|---|---|
  | #704 bicubic coefficients, `uchar` | 0.4371 LSB | 0.4798 LSB | 1355 of 17814 |
  | #732 bicubic store, `ushort` | 0.0000 LSB | 0.4680 LSB | 0 of 1017 |
  | #733 bilinear weights, `uchar` | 0.0000 LSB | 0.0252 LSB | 0 of 1113 |
  | #733 bilinear weights, `ushort` | 0.0000 LSB | 6.2848 LSB | 0 of 1113 |

  #704 was a coin toss taken for parity. These two are not: this module is
  exact and libvips is not, on every sample. `bicubic_unsigned_int32_tab`
  truncates its `double` store, a one-directional bias of -0.499 LSB that
  darkens every resampled `ushort` image by half a level, and `BILINEAR_INT`
  builds its four weights as 12-bit fixed point, worth up to 26 of 65535.

  The pins are on a linear ramp, which both bilinear and Catmull-Rom reproduce
  exactly, so the right answer is closed form and the tests do not have to
  reimplement an interpolator to know it. Both directions are asserted, so the
  divergence can neither grow nor quietly vanish.

- `affine` and `mapim` convert the caller's `background` to the carrier once
  before they resample, the way `vips_affine_build` runs `vips__vector_to_ink`
  once before it embeds (issue #736). `vips_cast` clips and then truncates
  toward zero on an integer carrier and narrows on a float one, so every tap
  past the edge and every output pixel outside the transformed input is already
  a carrier value in vips; this module carried the raw `f64` into both.

  It was worth up to **75 of 255** on a byte carrier: `--background 400.9` is
  ink 255 in vips and 400.9 in a `f64` convolution, and the difference survives
  wherever the ink is weighted against real pixels. Measured over a 6x6
  constant with five interpolators and three carriers, the whole table is now 0
  differences except the two cells that belong to other issues (#732, #733) and
  two float samples in a degenerate constant-ramp fixture that land exactly on
  an `f32` rounding midpoint.

  Callers passing an in-range integral background see no change. A fractional
  one now truncates rather than rounding, and an out-of-range one clips, which
  is what vips does and what the docs claimed the module already did.

- The `resample` module docs said `Extend::White` diverges on an alpha raster
  because `vips_affine` "premultiplies into a float image before it paints that
  border", so `FILL_LINE(float, ...)` runs and the byte `memset` never does.
  That is not what happens (issue #692). `vips_affine_build` embeds **before**
  it premultiplies, so the ink is memset into the raster's own domain either
  way. What moves the value is that the premultiply pair does not cancel on that
  pixel: `vips_premultiply` takes a clipped alpha into its multiplier and
  `vips_unpremultiply` takes the raw one into its reciprocal, so a border pixel
  whose every band holds the same ink `E` comes back as `clip(E, 0, max_alpha)`.

  **The divergence stays**, and that is now a decision with numbers behind it
  rather than a to-do. The border follows whichever ceiling the premultiply
  bracket uses, and this module's is the depth's on an unsigned carrier
  (issue #664), so the two answers differ only where a tag's ceiling sits below
  its carrier's depth: three cells out of eleven measured, all of them a 16-bit
  raster wearing an 8-bit tag. Adopting vips' ceiling to close them costs the
  whole image, not the border: `vips affine` on a constant-25000 `ushort` RGBA
  tagged `srgb` returns **255 for every interior sample**, tagged `scrgb` it
  returns 1, and with alpha 65535 a colour of 25000 comes back as 97. Clipping
  only the border fill instead would fix the pure-ink pixel and leave every
  blended one wrong, because the two premultiplied spaces are scaled
  differently.

  Both halves are pinned now: the agreeing cells so the divergence is bounded
  to those three rather than assumed, and the interior round-trip so the price
  of the other reading is a number.

- `affine_interpolators_match_libvips_oracle` explained its 1-byte `bilinear`
  allowance as "a single `.5` rounding tie". It is not: `SWITCH_INTERPOLATE`
  sends `uchar` and `ushort` rasters to `BILINEAR_INT`, whose four weights are
  12-bit fixed point as well. Modelling that reproduces the binary exactly and
  modelling a tie does not. The comment now says so and issue #733 carries the
  measurement.

- `try_embed`, `try_gravity`, `try_insert` and `smartcrop`'s `Entropy` and
  `Attention` strategies return a new `ExtractError::FloatUnsupported` on a
  float raster instead of **panicking** out of a `Result` signature
  (issue #694). The enum is `#[non_exhaustive]`, so the variant is additive
  and this is not a breaking change.

  #667 made the panic easy to walk into rather than creating it. It put the
  float column of the white-ink table on the public `Extend::White` rustdoc, so
  a caller holding a float raster from an EXR, FITS or `.v` decode reads that
  the ink is `1.0` for `ScRgb`, calls `try_embed`, and gets a process-visible
  panic out of a signature that promised an `Err`. That doc now says the float
  column belongs to the resamplers.

  The issue names two entry points. It is four, plus two of `smartcrop`'s six
  strategies, and the split is not per operation, it is whether the operation
  copies whole pixels byte-wise or reads individual samples. `extract_area`,
  `crop`, `replicate`, `zoom`, `subsample` and `smartcrop`'s four pure-geometry
  strategies (`Centre`, `Low`, `High`, `All`) take a float raster unchanged and
  always did, so the guard is deliberately not at the `try_smartcrop` entry
  point: putting it there would break four working strategies to fix two.

  `insert` checks **both** inputs. The result takes the wider of the two
  depths, so a float `sub` under an unsigned `main` reaches the same sample
  copy. I found that by mutating the second check away and watching the tests
  stay green.

- `Raster::extract` carries its input's metadata, and so do the pyramid
  downscale and the padded-tile path, so every tile of an engine run keeps the
  interpretation, the resolution, the orientation and the attached fields
  (issue #740). `Raster::extract` is the crate's physical crop: `src/engine.rs`
  and `src/streaming.rs` call it per tile and per strip, and
  `Raster::extract_area` is built on it and was the only one of the two that
  carried, since #690.

  It is not cosmetic, and it is only visible on the float carriers. #664 makes
  the premultiply bracket take its alpha ceiling from the interpretation on
  float and from the storage depth on unsigned, so a float raster that lost its
  tag brackets against 255 rather than 1.0. A 32x32 `RgbaF32` tagged `ScRgb`,
  cropped 16x16 through each method and then `resize(0.5)`, differs in **98 of
  1024 bytes**, and an explicitly `Srgb`-retagged copy of the same pixels
  differs by exactly the same 98, so the loss is precisely equivalent to a
  retag. The same fixture as `Rgba8` differs in **0 of 256**, which is the trap:
  measuring this on the obvious 8-bit carrier reports no effect.

  Three more sites had to carry for the engine to keep it end to end:
  `resize::downscale_half` and `downscale_to`, which build every pyramid level
  below the first, and the three padded-tile constructions in
  `engine::extract_tile`, which build a tile from a fresh background buffer.
  Without those, only the top two levels of a pyramid carried anything. vips
  agrees: `shrink`, `reduce` and `resize` all hand the whole block on, and none
  of them rescales the resolution with the pixel grid.

  **A correction to the issue.** It says a pyramid of a *float* scRGB source
  through the region entry point would not match a whole-image one. That is not
  reachable: the engine refuses a float source outright with
  `RasterError::FloatUnsupported { op: "downscale_half" }`, so the pixel
  divergence above is a public-API consequence and not a pyramid one. What the
  pyramid lost was the metadata, on every tile.

  The origin offset is **carried** here, not stamped. `extract_area` still
  stamps `(-left, -top)` to match `vips_extract_area` (#690), and `extract` is
  not that operation: it is the physical crop, vips has no method it
  corresponds to, and a pyramid tile is not a crop of a larger image in the
  sense `Xoffset` means. Stamping there would have put a non-zero origin into
  every tile header on an analogy rather than a measurement.

  The cost is one bounded copy of the attached fields per crop, per downscale
  and per padded tile, which for an ICC profile is a real allocation that was
  not there before. Measured across four image sizes, it is **O(tiles), not
  O(pixels)**: 3.22 profile copies per tile at 32x32 falling to 2.04 at
  256x256, while the pixel count grows 64x. At a realistic 256px tile a
  1024x1024 run makes 78 copies of a 3144-byte profile, about 4% of the bytes of
  a single tile buffer.

- `Raster::try_join`'s float guard and its documentation say what they are for
  (issue #730). The comment claimed the placement path underneath panics on
  4-byte samples, which stopped being true when #694 moved that guard into
  `try_insert`. The guard stays, because without it a float input surfaces as
  `ConversionError::Extract(ExtractError::FloatUnsupported { op: "insert" })`,
  naming an operation the caller did not call; with it they get
  `ConversionError::FloatUnsupported { op: "join" }`. It runs before the
  delegation, so `try_insert`'s refusal is unreachable from `join`.

  `try_join`'s rustdoc lists the variants delegated from `try_insert`, including
  ones that are not reachable, because "a `#[non_exhaustive]` match should expect
  them". `ExtractError::FloatUnsupported` was missing from that list by the
  module's own rule, and is now in it.

  `try_arrayjoin`'s guard is **not** the same animal, and its comment used to say
  it was: `arrayjoin` blits its cells with `read_flat` / `write_flat` itself
  rather than delegating, so nothing underneath it would catch a float and
  removing that guard restores a panic rather than changing an error type.

- An attached ICC profile is dropped when the interpretation is retagged to a
  space it cannot describe, matching vips (issue #720). `try_colourspace(Bw)`
  used to hand back a one-band grey raster with a three-channel RGB profile
  still attached, and the next `icc_transform` read that profile as if it
  described the samples.

  Measured against the pinned vips 8.18.6 on three **real** profiles, sRGB
  (3144 bytes), Generic Gray (2020) and Generic CMYK (55280), against every
  interpretation. The rule is the band count the new tag implies versus the
  profile's own colour space: `b-w` and `grey16` imply one, `cmyk` four, and
  everything else three. Swapping the profile swaps which targets lose it,
  which is what makes it a rule rather than a list of unlucky interpretations.

  It reads the **tag** and not the image. `vips bandmean` and
  `vips extract_band 0` both take a three-band `scrgb` raster to one band,
  leave the tag alone, and keep the three-channel profile, so an
  implementation comparing the profile against `format().channels()` would be
  wrong.

  Setting the interpretation through `Raster::set_field` still keeps the
  profile, and that split is vips's rather than a gap:
  `vipsedit --interpretation b-w` keeps it and `vips copy --interpretation b-w`
  drops it. A header write describes what the file already holds, so
  revalidating there would drop a profile the file legitimately carries; the
  decoders assign the tag directly for the same reason.

  A profile this build cannot read is kept: the colour space lives at bytes
  16..20 of the ICC header, and a blob too short to hold one, or carrying a
  signature this build does not know, has no verdict. Dropping an attachment
  because the parser could not reach one is worse than keeping one that may
  not apply, and it is the same call `imageio` makes for `.v` trailer values it
  cannot interpret (#565).

  `invfft`, `invfft_real` and `freqmult` still drop the profile through an
  explicit call rather than through the general rule, and that is now written
  down with its reason: libviprs tags them `None` where vips tags them `B_W`,
  so the rule looks at `Multiband` (three channels) and keeps what vips drops.

- The operations that reposition an image stamp the origin offset instead of
  carrying the input's, matching vips (issue #721). `fliphor`, `flipver`,
  `rot`, `wrap`, `autorot`, `conv`, `convsep`, `compass`, `gaussblur`, `sobel`,
  `scharr`, `prewitt` and `canny` all reported the input's `xoffset` / `yoffset`
  where vips reports a value derived from the transform. `Raster::xoffset` and
  `Raster::yoffset` are public and both go into the `.v` header, so a pipeline
  that flipped and saved recorded an origin saying the image was where it had
  been before the flip.

  #706 found the first instance of this split in the other direction, where
  `extract_area` and `crop` stamp `(-left, -top)` and the six other extract ops
  carry.

  Measured against the pinned vips 8.18.6 at three image shapes per operation,
  and for the convolving ones at nine mask shapes as well:

  | op | rule |
  |---|---|
  | `fliphor` | `(width, 0)` |
  | `flipver` | `(0, height)` |
  | `rot` D90 / D180 / D270 | `(out width, 0)` / `(width, height)` / `(0, out height)` |
  | `wrap` | `(w - w/2, h - h/2)` |
  | `conv` | `(-(mask width / 2), -(mask height / 2))` |
  | `convsep`, `compass`, `gaussblur` | the same rule, inherited |
  | `sobel`, `scharr`, `prewitt` | `(-1, -1)` from the 3x3 gradient mask |
  | `canny` | `(-1, -1)` from the 2x2 gradient mask, at every sigma |
  | `autorot` | whichever transform it finishes on |

  None of them reads the input's offsets: the same sweep from a source at
  `0 / 0` gives byte-identical numbers, and `rot45` at all seven angles, `grid`,
  `cast`, `gamma`, `join`, `arrayjoin`, `fwfft`, `colourspace`, `composite2`,
  `spcor`, `fastcor` and every op in `src/bands.rs` still hand the input's
  offsets straight back through the same `.v` writer. That last list is the
  positive control, and it is a test rather than a remark.

  **`convsep` is the cell that says what the rule is.** A 3-wide, 1-tall mask
  stamps `0 / -1`, not the `-1 / 0` the mask itself implies, because `convsep`
  finishes on the mask's 90-degree rotation. So there is one rule, `conv`'s, and
  `convsep`, `compass` and `gaussblur` inherit it by composition rather than
  each carrying a copy. `canny` is the counterexample that keeps that honest:
  its offset follows its 2x2 gradient and not its blur, so at sigma 3 it reports
  `-1 / -1` where `gaussblur` alone reports `0 / -5`.

  **`autorot` at orientation 4 is the one cell composition gets wrong.**
  `vips_autorot` reaches it as a 180-degree rotation followed by a horizontal
  flip and stamps the flip's `(width, 0)`; libviprs does the same pixels in one
  vertical flip, whose own rule is `(0, height)`. The offset is corrected there
  rather than paying for a second pass over the image to make the composition
  match.

- Every operation in `src/bands.rs` carries the input's metadata onto its
  result: `bandjoin`, `bandjoin_const`, `bandjoin_vec`, `bandfold`,
  `bandunfold`, `bandmean`, `bandrank`, `bandand`, `bandor`, `bandeor`,
  `extract_band` and `extract_bands` all used to finish on a bare
  `Raster::new` and hand back `RasterMeta::default()` with an empty field map
  (issue #727). That is the last module in the crate with the defect, after
  #717 and #719.

  Measured against the pinned vips 8.18.6 from an 8x8 `rgb` source carrying
  `xres 5`, `yres 7`, `xoffset 11`, `yoffset 13`, `orientation 6`, an attached
  string and a real 3144-byte sRGB ICC profile. All twelve calls report the
  whole lot back.

  Two cells needed measuring rather than assuming. `bandfold` and `bandunfold`
  reshape the pixel grid and do **not** rescale the resolution: 8x8 3-band
  folds to 1x8 24-band and still reports `xres 5 yres 7`, and unfolds to 24x8
  1-band reporting the same, which is the shape `zoom` and `subsample` had in
  #690. And nothing in this module stamps the origin offset, so all twelve
  report `11 / 13` straight through, unlike `flip`, `rot`, `wrap` and the
  convolving ops (#721).

  `bandjoin` and `bandrank` take more than one input and follow the union rule
  #718 established: the header block comes from the first input alone and the
  attached fields are the union of every input, the first winning a name they
  share. `bandjoin` is measured both ways round, and reversing the arguments
  flips the header block and the shared name while the ICC profile still
  crosses from whichever input has one. `bandrank` is measured over three
  sources, so the union is not a two-way merge in disguise.

- `conv`, `convsep`, `compass`, `gaussblur`, `spcor` and `fastcor` carry the
  input's metadata onto their results, where all six used to hand back a raster
  built from `RasterMeta::default()` and an empty field map (issue #719). They
  lost the interpretation, the resolution, the offsets and the orientation as
  well as every attachment, which is a step worse than the sites #717 fixed.
  `try_sobel` named the six in a comment and nothing tracked it.

  Measured against the pinned vips 8.18.6 from an 8x8 `rgb` source carrying
  `xres 5`, `yres 7`, `orientation 6`, an attached string and a real 3144-byte
  sRGB ICC profile: `conv` at 3x3 and 5x5, `convsep` at 1x3, `compass` at 3x3
  and `gaussblur` at sigma 1 and 3 all report the tag, the resolution, the
  orientation, the string and the 3144 bytes back. `spcor` and `fastcor` want a
  one-band input, so those two were measured on a `b-w` source carrying a real
  2020-byte grey profile and hand all of it on as well. The profile has to match
  the tag there, because a three-channel profile under a `b-w` tag is removed by
  a rule about the retag rather than about these ops (issue #720).

  `sharpen` is **not** one of the six and does not change here. It blurs through
  `convsep` on a LabS intermediate, so it looks like it should inherit this, but
  its output metadata comes from the `colourspace` on the way back, which issue
  #717 already carries. I had that the wrong way round until the mutation sweep
  said so, and there is a test that says which change it belongs to.

  The origin offsets are **not** fixed by this. `conv`, `convsep`, `compass` and
  `gaussblur` stamp a mask-relative origin (`-1 / -1` for a 3x3, `-2 / -2` for a
  5x5, `0 / -1` for a separable 1x3) that does not depend on the input's at all,
  and now that they carry, they carry the input's instead of stamping. That is
  issue #721, it is the same shape `flip`, `rot` and `wrap` have, and the test
  deliberately asserts nothing about the offsets so it does not pin behaviour
  this change leaves wrong.

- Every operation that builds a fresh raster carries its input's metadata onto
  it, not just the header block: the interpretation, the resolution, the
  offsets and the orientation as before, and now the ICC profile, the EXIF blob
  and every attached field with them (issue #717). `cast`, `gamma`,
  `falsecolour`, `addalpha`, `arrayjoin`, `join`, `fliphor`, `flipver`, `rot`,
  `rot45`, `grid`, `wrap`, `fwfft`, `invfft`, `invfft_real` and `freqmult` all
  used to copy `RasterMeta` and leave the field map behind, so a profile that
  survived a load went missing the moment you cast the depth.

  There were eighteen open-coded carries in `src/`, eleven of which wrote only
  the first of the two lines. They now go through one `Raster::carry_meta_from`,
  and #690's private `carry_extract_meta` folds into it. The name takes
  `&mut self` (`out.carry_meta_from(src)`) because it reads in the direction the
  data moves and works on a result a helper already built, where a returning
  form puts the construction inside the carry's own argument list.

  Measured against the pinned vips 8.18.6 across nineteen operations, from an
  8x8 `rgb` source carrying `xres 5`, `yres 7`, `xoffset 11`, `yoffset 13`,
  `orientation 6`, a `VipsRefString` and a real 3144-byte sRGB ICC profile. The
  tag is `rgb` rather than `scrgb` on purpose: `vips gamma` on an `scrgb` or
  `rgb16` source hands back `srgb` because it retags off the output's sample
  format, and pinning the carry against a source that trips an unrelated retag
  rule would measure the wrong thing.

  Two cells are not a wholesale carry.

  `invfft`, `invfft --real` and `freqmult` **drop the ICC profile** and keep
  every other attachment. It is the profile specifically rather than blobs in
  general: a second plain 48-byte `VipsBlob` attached alongside survives all
  three. The cause is the retag those three do, not the transform:
  `vips copy in.v out.v --interpretation b-w` removes the same profile, and
  sweeping every interpretation shows the rule is a band-count match against
  the profile's own colour space (a 3-channel profile is removed by `b-w`,
  `grey16` and `cmyk` and kept by the rest; a 1-channel one is kept by `b-w`
  and `grey16` and removed by `srgb` and `cmyk`). The general rule is issue
  #720; these three measured cells are handled where they happen.

  `new_from_image` carries the header block **without** the fields, which is
  what its doc already claimed and is now measured rather than asserted. There
  is no CLI for `vips_image_new_from_image`, so I called it against the same
  8.18.6 through `ctypes` on `libvips.42.dylib`. It also drops the
  **orientation**, which libviprs was carrying: vips holds orientation as an
  attached field, libviprs holds it in `RasterMeta`, so it used to ride along
  with the header block and a constant image arrived claiming the source's
  rotation.

- `Raster::try_insert` carries the metadata, where it used to hand back a
  raster with none of it, and `join` and `arrayjoin` take the same rule (issue
  #718). It was written down as a known gap in `src/extract.rs`'s module doc
  and in this file, and tracked by nothing.

  Two rules, both measured on vips 8.18.6 from two sources chosen to disagree
  on every field. The header block comes from `main` alone: an scRGB `sub`
  under an sRGB `main` reports sRGB, and the resolution, the offsets and the
  orientation are all `main`'s. The attached fields are the **union** of both,
  with `main` winning a name they share, so a profile only `sub` carries still
  reaches the output. I ran it in both directions rather than reading one cell.

  `vips join`, `vips arrayjoin` and `vips bandjoin` follow the same rule, so
  `join` and `arrayjoin` merge here too and `out.fields = self.fields.clone()`
  would have been wrong for both. `bandjoin` lives in `src/bands.rs` and is not
  in this change.

  The merge is a new `MetadataFields::merge_under` in `imageio`. Values this
  build cannot interpret merge on the same terms, so a `.v` trailer field an
  older build wrote still travels through an insert.

- Every operation in `src/extract.rs` carries its input's metadata through to
  its result: `extract_area`, `crop`, `embed`, `gravity`, `replicate`, `zoom`,
  `subsample` and `smartcrop` all keep the interpretation, the resolution, the
  orientation and every attached field, where each of them used to hand back a
  raster rebuilt from a default header block and an empty field map (issue
  #690). An ICC profile, an EXIF blob and the colour tag survived a load and
  then went missing the moment you cropped.

  It turned load-bearing with #667, which makes `Extend::White` ink from the
  interpretation. `embed(.., White)` painted the right ink and then handed back
  a raster that no longer said what it was, so embedding that result a second
  time inked it differently: an scRGB source came back 1, and its own output
  came back 255.

  The origin offset is the one field the operations disagree on, so it is not a
  verbatim carry. `vips_extract_area` writes `Xoffset = -left` and
  `Yoffset = -top` and throws the source's away, while the placement and tiling
  ops leave the source's alone, and `smartcrop` inherits the crop rule by being
  `extract_area` underneath. Measured on the pinned vips 8.18.6 by sweeping
  `left` over 0/1/3/4 against `top` over 0/2/5, and `embed`'s `x` over 0/2/-2
  against `y` over 0/3/-3, rather than by reading one cell.

  `zoom` and `subsample` do **not** rescale the resolution with the pixel grid,
  which is the part worth measuring rather than assuming: `zoom` by 2x3 on
  `xres=5 yres=7` reports 5 and 7 back, not 10 and 21.

  `Raster::try_insert` still drops the metadata and is deliberately not in this
  change. Its rule is a two-input one and a different shape: the header block
  comes from `main` alone while the attached fields are the union of both with
  `main` winning a shared name, and carrying that union needs a merge on
  `MetadataFields`, which lives in `imageio`. Issue #718 does that.

- `Raster::try_sharpen` and `Raster::try_canny`'s float arm no longer abort the
  process when an allocation fails, and there is a new
  `Raster::try_f32_samples` for the widening they sit on (issue #627).

  `Raster::f32_samples` is built on `.collect()`, and a `.collect()` sized from
  an exact-size iterator allocates through `handle_alloc_error`, which ends the
  process rather than returning. Nothing catches that, so both entry points
  carried an unavoidable process exit however their signatures read. #575 had
  taken the other nine convolution entry points abort-free and these two could
  not follow, because the abort was not in `convolution.rs` at all.

  `try_f32_samples` returns `Result<Vec<f32>, RasterError>`, reserving with
  `try_reserve_exact` and reporting `RasterError::AllocationFailed`. Reach for
  it wherever an allocation failure should arrive as a value. `f32_samples`
  keeps its signature and its meaning, so no existing call has to change:
  `None` still means only "not a float format". It now delegates to the
  fallible form and **panics** rather than aborting if the widening fails,
  which at least unwinds and can be caught.

  `try_sharpen` also stopped cloning: the widened samples are written back in
  place, and the LabS raster is moved into the result instead of copied, so two
  image-sized allocations are gone rather than made fallible. Its three
  remaining scratch planes go through the fallible reservation the rest of the
  module uses. With #672 and #685 having done the same for the LabS round
  trip's own buffers, no image-sized allocation on either entry point's path
  is infallible any more. What is left is smaller than an image and stays out
  of scope: the `fields.clone()` that carries an input's attachments onto each
  result, an embedded ICC profile among them, and the mask, table and per-row
  buffers the mask generator and the convolution scan build.

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

- The ICC LUT routes hand the CMS bounded slices instead of the whole plane, so
  the buffer moxcms allocates for itself no longer follows the image size
  (issue #693). `try_icc_import_with`, `try_icc_export_with` and
  `try_icc_transform` drive `xf.transform` in 16384-pixel chunks, which caps a
  single moxcms intermediate at 192 KiB on any geometry and any device space.
  Measured on the 256-square LUT fixture the CMS asked for 786432 bytes before
  and 196608 after; on the 512-square one, 3145728 before and the same 196608
  after.

  **This still does not make the LUT routes abort-free**, and the difference
  matters. moxcms sizes the katana engine's intermediates from the slice it is
  handed and allocates them with a plain `vec![0f32; n]`
  (`conversions/katana/md3x3.rs:176`, `md4x3.rs:164`, `md_nx3.rs:160` and
  `md_pipeline.rs:90` in 0.8.1), so a request the host cannot serve still
  reaches `handle_alloc_error` and still ends the process. Nothing in this crate
  can change that: the fallible spelling is upstream's to adopt, a `try_vec!`
  over `try_reserve_exact` returning `CmsError::OutOfMemory` that
  `katana/rgb_xyz.rs:56` already uses and the `md*` stages do not. What changed
  is that the request is now a fixed 192 KiB rather than an attacker-chosen
  fraction of the address space, so an image big enough to exhaust the host
  fails at one of this crate's own fallible reservations, which report
  `ColourError::Raster`, rather than at moxcms's infallible one.

  `tests/icc_lut_alloc.rs` keeps that distinction on the record rather than
  letting the docs quietly widen past it. It drives the routes at two
  geometries under a `GlobalAlloc` that logs and can refuse, insists the largest
  *zeroed* request is the same number at both sizes and under the 512 KiB the
  module promises, then refuses that request too and asserts the child dies on
  SIGABRT with a `handle_alloc_error` message naming a size inside the bound.
  Zeroed is what separates the two kinds of allocation: every buffer this crate
  reserves arrives as a plain `alloc` through `try_reserve_exact`, and the four
  katana sites are `vec![0f32; n]`, which std lowers to `alloc_zeroed`. So the
  ceiling cannot answer ahead of the call it is aimed at, which is the trap the
  #685 tests fell into.

  Splitting the plane changes no sample. Every stage moxcms runs reads only the
  pixel it is writing (`conversions/katana/stages.rs:73`), and
  `chunking_the_cms_transform_reproduces_the_whole_plane_result` asserts the
  chunked buffer is bit-identical to one whole-plane call over a fused LUT
  profile, a katana LUT profile and a four-channel source layout, on a geometry
  that ends on a short chunk. The oracle captures are unchanged.

  `transform_in_chunks` refuses two sides that disagree on pixel count instead
  of asserting it. `zip` stops at the shorter one, so a mismatch would transform
  a prefix, leave the rest of the destination holding whatever it was reserved
  with, and return `Ok(())`. Both callers derive both planes from one
  `(width, height)` so it cannot happen today, which is why a `debug_assert!`
  was the wrong tool: nothing would ever exercise it, and the release build
  would have neither the assert nor an error.

  16384 is a cache choice, and the constant's doc now says which retunes are
  free and which are not. Anything from 43 to 43690 pixels is green; below that
  the intermediate stops clearing the test's zeroed logging floor, and above it
  the intermediate passes the 512 KiB bound this module advertises. Both ends
  are one named number, and a chunk outside the window now fails saying so
  rather than sending the reader upstream to bump a moxcms pin.

- `cargo +nightly miri test` gets past `tests/dependency_policy.rs` (issue #714).
  It used to die there on the first test that shells out, with `unsupported
  operation: can't call foreign function `fork``, and Miri ends the whole session
  on one unsupported operation rather than failing that one test. So the gate
  reported "Miri failed" having run none of the code it exists to check.

  Ten tests over three files now carry `#[cfg_attr(miri, ignore)]`: the five in
  `tests/dependency_policy.rs`, the three in `tests/pdfium_source_audit.rs` and
  two of the three in `tests/workspace_layout.rs`. None of them calls into
  libviprs at all, so nothing is lost by keeping them out of Miri.

  This is not #707, which is a Stacked Borrows violation in `sha2`'s aarch64
  NEON backend and so never executes on the hosted `ubuntu-latest` job. Miri
  supports process spawning on no target and under no flag, and
  `-Zmiri-disable-isolation` does nothing about it, so this one was taking the
  hosted gate down as well.

  `tests/miri_ignore_convention.rs` enforces it from here, and enforces it
  differently from the filesystem convention it was built for. The filesystem
  rows were a ledger: an `unannotated fs-detected` test was allowed to stand,
  because `-Zmiri-disable-isolation` made its call come back rather than abort.
  A spawning test is a flat refusal, because there is no configuration in which
  it runs.

  #711 removed that flag after this was written, so the filesystem class aborts
  now too and the asymmetry has narrowed. It has not gone. This class is
  enforceable today because its population is 17 and all 17 are annotated; the
  filesystem population is 138 tests over 29 files, 8 of them `src/` modules,
  and that is issue #739 rather than something to fold in here. Measured on
  `800c699` with nothing applied, `cargo miri test --test workspace_layout`
  aborts on its first test, so the suite has not reached a second target since
  #711 landed.

  The detector had to learn to follow a call to see any of them, since not one
  of the ten spells `Command::new` in its own body: they call `cells()`, which
  calls `graphs()`, which spawns cargo. It now parses every `fn` in a file,
  marks the ones that reach `std::process`, and repeats to a fixed point,
  matching a callee by name on identifier boundaries rather than as `name(`,
  because `graphs()` reaches cargo through `CELLS.iter().map(resolve)` where the
  callee never sits beside a paren.

  That parse reads a `;` as the end of a bodyless declaration, which is right
  for a trait method and wrong for `fn fingerprint() -> [u8; 32]`. Measured
  across `src/` and `tests/`, the naive test dropped 133 function headers where
  a bracket-aware one drops 17, so **116 real functions were invisible to the
  call graph**. None of them spawns, so nothing was missed in fact, but the
  failure was silent and in the under-approximating direction, which is the one
  that costs the gate rather than an annotation.

  The count of spawning tests is pinned at 17, which is the positive control the
  rest of it needs: every other assertion here says a set is empty, and a
  detector that has stopped finding anything satisfies all of them. It earned
  its place twice over. It caught a miscount the first time it ran, and under
  the `name(` matching the count goes to 14 while every other check stays green.

  Four shapes still reach `std::process` unseen, none of them in the tree: an
  aliased `use ... as Cmd`, a spawn inside a `macro_rules!` body, a closure held
  in a `static`, and a helper in another file, since the scan is per file. The
  module docs list them and a test pins all three of the single-file ones as
  misses, so one being fixed shows up as a failure rather than as documentation
  quietly going stale.

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

- `resize`, `reduce` and the bicubic interpolator round the sub-pixel offset
  onto libvips' coefficient-table grid, so they stop diverging from the binary
  at non-dyadic scales (issue #668). Output moves wherever the offset used to
  miss that grid, which is every scale that is not a power of two, so committed
  reference images of a fractional resize will need regenerating. The dyadic
  scales are untouched by construction.

  **The full list of operations whose output can move**, because `table_offset`
  sits in two places and reaches further than the three named above.
  `Raster::reduce` and `Raster::resize` go through the reduce mask, and so does
  `Raster::shrink` on its residual reduce and `Raster::thumbnail` /
  `thumbnail_buffer`, which do their heavy shrink through `resize`. The
  interpolator half reaches `affine`, `mapim`, `rotate_with` and
  `similarity_with`, at an explicit `Interpolator::Bicubic` in each case. The
  bare `rotate` and `similarity` default to bilinear and are genuinely
  untouched, as are `nohalo` and `lbb`, because none of the three reads a table
  in libvips either. Anyone with pinned `mapim` or `thumbnail` output needs this
  list as much as anyone with a pinned `resize`.

  libvips never evaluates a resampling kernel at the true offset. `vips_reduceh`
  and `vips_reducev` index a 65-entry table of masks, and
  `vips_interpolate_bicubic_interpolate` indexes a 65-entry table of Catmull-Rom
  coefficients, both built at `(float) x / VIPS_TRANSFORM_SCALE` with
  `VIPS_TRANSFORM_SCALE` = 64, and both spelling the index the same way
  (`reduceh.cpp:270-276`, `bicubic.cpp:496-503`). We computed the mask at the
  exact offset and called that the same convolution without the quantisation
  error. It is not: it is the mask for a different sub-pixel position.

  Dyadic scales hide it because their offsets land on the grid. A reduce by 2
  has offset 0 at every output position and one by 2.5 alternates 48/64 and
  16/64, so the lookup and the exact evaluation agree and always did. A reduce
  by 4/3 has offsets in thirds and 0.6667 * 64 is 42.67, which does not.

  Measured on vips 8.18.6 over a 64x64 float raster with three bands and no
  alpha, so no premultiply bracket is involved anywhere. `vips shrinkh` and
  `vips shrinkv` already agreed exactly at factors 2, 3, 4, 5 and 7, so the box
  shrink and the split point that picks it were never the problem. `vips resize`
  went from 6144 of 6912 samples wrong at 0.75 (max 1.54), 1728 of 1728 at 0.37
  (max 0.46), 936 of 1083 at 0.3 (max 0.21) and 25259 of 27648 at 1.5 (max 2.27)
  to bit-exact at every downscale and within one f32 ulp at every upscale.

  Rounding has to floor where the C truncates. `(int)(X * 128)` rounds toward
  zero and `& 127` reads two's complement, so on a negative coordinate that pair
  lands one bucket above `floor`. vips never meets the case, because
  `vips_affine_gen` hands the interpolator a coordinate in the embedded space
  shifted by `window_offset` (`affine.c:361-362`); we interpolate in the input's
  own coordinates, which go negative on the first output column of any
  enlargement past 2x.

  Bilinear, nohalo and lbb keep the exact offset, because none of them reads a
  table in libvips either.

- Untracked the two compiled Python files under `oracle-captures/`,
  `foreign-analyze/__pycache__/capture.cpython-314.pyc` and the matching
  `foreign-mat` one (issue #681). They are build artefacts of the capture
  scripts next to them, tied to CPython 3.14, and nothing reads them.

  The ordering is the part worth writing down. #673 adds an
  `oracle-captures/.gitignore` that ignores `__pycache__/`, and an ignore rule
  does nothing at all to a path that is already in the index. So that file
  landing on its own would have left both of these exactly where they were and
  stopped `git status` mentioning them, which is worse than either half by
  itself. `git rm --cached` is what actually moves them, and it keeps the
  working copies, so nobody loses a cache they were using.

  `tests/oracle_capture_pins.rs` now asks git what it tracks under
  `oracle-captures/` and fails on anything ending in `.pyc` or `.pyo` or
  sitting under a `__pycache__/`. It has to ask git rather than walk the
  directory, because the question is about the index and the filesystem cannot
  answer it either way round: `git rm --cached` leaves the file on disk, and a
  fresh clone does not have it whether or not anyone ran that command, so a
  walk would go green in CI for a reason unrelated to the fix. The listing is
  the thing that can come back empty and take the guard with it, so the test
  anchors on capture scripts it knows are tracked before it reads anything into
  an absence.

- Every `json.dump` and `json.dumps` in `oracle-captures/` now passes
  `allow_nan=False`, so a capture that measures a non-finite value stops at the
  write instead of putting a bare `NaN`, `Infinity` or `-Infinity` into a file
  no strict parser will read (issue #682). Two of the twenty call sites already
  had it, both from #674. The other eighteen are across twelve scripts.

  Nothing was broken. Every committed capture parses strictly today, which is
  what #674 fixed. The problem is where the failure lands: `json.dump` writes
  the bare literal by default and Python's own `json.load` reads it straight
  back, so a capture round-trips perfectly on the machine that took it and only
  falls over for a reader in another language, months later, in a file nobody
  suspects. The flag moves that to the moment somebody runs `python3
  capture.py`, which costs a re-run and no investigation.

  It is eighteen call sites and not twelve because `foreign-avif` and
  `foreign-jp2k` hand-roll an encoder that keeps a leaf array on one line, and a
  leaf is exactly where a float lives. Guarding only their top-level dump would
  have left every pixel row unguarded. The rule is blanket for the same reason,
  down to the two calls that only serialise a dict key: an exemption needs a
  rule for who qualifies, and any such rule is something to argue past later.

  No `oracle.json` changed. I drove each area's committed writer over its own
  parsed capture with the flag on, including the two hand-rolled encoders, and
  all fourteen come back byte for byte identical, so nothing in the tree was
  passing a non-finite float to begin with.

  `tests/oracle_capture_json.rs` holds it shut. It blanks Python comments and
  the insides of string literals first, then finds each `json.dump(` in what is
  left and reads that call's own bracketed argument list, so prose cannot answer
  for code and a file with one guarded call out of four still fails, with the
  line number of the one that is missing it. F-string fields count as code,
  because `foreign-jp2k` keeps one of its four dumps inside one. A companion
  test feeds the scanner a source whose only `allow_nan=False` is in a docstring
  and a comment and fails if that reads as guarded.

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
