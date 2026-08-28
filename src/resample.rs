//! Resampling operations ported from libvips.
//!
//! This module is the next batch of the libvips operation surface required
//! by the ported integration tests (after [`crate::bands`],
//! [`crate::arithmetic`], [`crate::extract`], [`crate::conversion`],
//! [`crate::draw`], [`crate::histogram`], [`crate::imageio`],
//! [`crate::composite`], [`crate::colour`], [`crate::morphology`],
//! [`crate::mosaicing`], and [`crate::convolution`]): box shrink, kernel
//! reduce, resize, affine transforms driven by an interpolator, the
//! similarity and rotate convenience forms, and coordinate-image remapping.
//! Operations that can fail on caller input exist in two forms, following
//! the established convention:
//!
//! * a fallible `try_*` method returning `Result<_, ResampleError>` with
//!   typed errors for bad factors, singular matrices, and unknown kernel or
//!   interpolator names; and
//! * a panicking convenience method matching the ported-test call surface
//!   (`shrink`, `reduce`, `resize`, `affine`, `similarity`, `rotate`,
//!   `mapim`) exactly, delegating to the `try_*` form. Where the ported
//!   surface passes libvips nicknames (`"bilinear"`, `"lanczos3"`), the
//!   panicking form takes `&str` and parses it; the `try_*` form takes the
//!   typed [`Interpolator`] / [`ReduceKernel`] enum.
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::shrink`] | `vips_shrink` | box-filter downsample |
//! | [`Raster::shrinkh`] / [`Raster::shrinkv`] | `vips_shrinkh` / `vips_shrinkv` | one-axis integer box shrink |
//! | [`Raster::reduce`] | `vips_reduce` | kernel downsample |
//! | [`Raster::reduceh`] / [`Raster::reducev`] | `vips_reduceh` / `vips_reducev` | one-axis kernel downsample |
//! | [`Raster::resize`] | `vips_resize` | scale by a factor |
//! | [`Raster::affine`] | `vips_affine` | 2x2 matrix transform |
//! | [`Raster::similarity`] | `vips_similarity` | rotate + scale |
//! | [`Raster::rotate`] | `vips_rotate` | rotate by an angle in degrees |
//! | [`Raster::mapim`] | `vips_mapim` | remap through a coordinate image |
//! | [`Raster::constant_u8`] | `vips_black` + `linear` | constant one-band image |
//!
//! # Semantics shared with libvips
//!
//! * **Output sizes.** Downsampled dimensions round to nearest, half away
//!   from zero (`VIPS_ROUND_UINT`): `reduce` and fractional `shrink` produce
//!   `round(dim / factor)` and `resize` produces `round(dim * scale)`. The
//!   internal box-shrink passes that `reduce` runs for large factors use
//!   ceiling rounding, exactly as `vips_reduceh` invokes `vips_shrinkh`
//!   with `ceil` set.
//! * **`shrink` composition.** Integer factors run the plain box filter
//!   (`shrinkv` then `shrinkh`, integer mean with round-half-up). Fractional
//!   factors delegate to `reduce` with the default `lanczos3` kernel and a
//!   reducing gap of 1, reproducing `vips_shrink_build`.
//! * **`reduce` kernels.** Each output sample is a 1D convolution of
//!   `vips_reduce_get_points(kernel, shrink)` input samples with the kernel
//!   stretched by the shrink factor and normalised to unit sum, evaluated at
//!   the sub-pixel offset rounded onto libvips' 65-entry table grid by
//!   `table_offset`. That rounding is part of the answer and not a speed
//!   optimisation, because `vips_reduceh_gen` never evaluates the kernel
//!   anywhere but on the grid; treating it as one is what made `resize`
//!   disagree with the binary by up to 2.3 at every non-dyadic scale while
//!   staying exact at the dyadic ones (issue #668). The masks themselves stay
//!   in `f64` and normalised to unit sum, where libvips carries a `short`
//!   fixed-point copy it reads on **both** integer carriers and does not
//!   renormalise after quantising. That divergence is kept on purpose and the
//!   argument is in the divergence section below (issue #777). Edges extend by
//!   replication
//!   (`VIPS_EXTEND_COPY`), so constant images are preserved exactly. `reduce`
//!   itself runs with gap 0 (no box pre-pass); `shrink` passes gap 1 and
//!   `resize` gap 2, as in libvips.
//! * **Interpolator offsets.** The bicubic interpolator reads the same kind of
//!   table (`bicubic.cpp:496-519`), so it rounds its offset through
//!   `table_offset` too. Bilinear and the nonlinear nohalo and lbb have no
//!   offset table and keep the exact offset, which is what
//!   `vips_interpolate_*` does for them.
//! * **Bicubic coefficients, per carrier.** `vips_interpolate_bicubic_interpolate`
//!   picks its arithmetic from the band format, and this module follows it
//!   (issue #704):
//!
//!   | `BandFmt` | libvips function | coefficients |
//!   |---|---|---|
//!   | `UCHAR` / `CHAR` | `bicubic_unsigned_int_tab` | `vips_bicubic_matrixi`, 12-bit fixed point |
//!   | `USHORT` / `SHORT` / `UINT` / `INT` | `bicubic_unsigned_int32_tab` | `vips_bicubic_matrixf`, `double` |
//!   | `FLOAT` | `bicubic_float_tab<float>` | `vips_bicubic_matrixf`, `double` |
//!   | `DOUBLE` | `bicubic_notab` | computed at the exact offset, no table |
//!
//!   So the fixed point is the `uchar` arithmetic and only the `uchar`
//!   arithmetic, and an alpha band takes the decision away from the stored
//!   depth entirely, because `vips_affine` premultiplies into a FLOAT image
//!   before it resamples.
//!
//!   The `FLOAT` row carries a second consequence, which is issue #705.
//!   `bicubic_float<T>` sums each of the four rows through `cubic_float<T>` and
//!   then combines them through `cubic_float<T>` again, and that helper
//!   **returns `T`**. Its arithmetic is `double` either way, because the
//!   coefficients are, so with `T = float` all five sums are computed in `f64`
//!   and narrowed to `f32` on the way out, and with `T = double` nothing
//!   narrows. This module does the same, keyed on the same carrier rule. The
//!   accumulation *order* is not part of it: flat 16-term `f64` and
//!   row-then-column `f64` are bit-identical here (measured 0 of 1764 apart on
//!   a random 24x24 float raster, where both miss the binary by the same
//!   1.5259e-05 in the same 356 samples), so it is the narrowing and not the
//!   reassociation that closes it. Porting the fixed point is a deliberate loss of
//!   accuracy in exchange for parity: measured against Catmull-Rom evaluated at
//!   the true offset in exact rational arithmetic, the mean absolute error over
//!   17814 interior samples of random `uchar` images goes from 0.4371 LSB to
//!   0.4798, worst case 1 LSB either way, and vips is the closer of the two on
//!   1355 of those samples. The shared error from the 1/64 offset grid above is
//!   0.44 LSB, ten times larger.
//! * **Metadata.** Every operation carries the input's header block
//!   (interpretation, resolution, offsets, orientation) and its attached
//!   fields (ICC profile, EXIF blob, anything a caller set) onto its output,
//!   through `Raster::carry_meta_from` (issue #789). The resolution goes
//!   across **verbatim**: measured on 8.18.6, `vips resize in.v out.v 0.5`
//!   and the same at 2 both return the input's `xres` and `yres` to the last
//!   bit rather than rescaling them with the factor, which is the answer #690
//!   measured for `zoom` and `subsample` too. `mapim` takes the block from the
//!   image being remapped and not from the index, which is a coordinate field
//!   rather than a picture, and [`Raster::constant_u8`] has no input to carry
//!   from.
//!
//!   The internal steps carry it as well, so the premultiply bracket sees the
//!   same interpretation between the vertical and horizontal passes that it
//!   saw at the start. That is not cosmetic: #664 made the bracket read the
//!   tag on a float carrier, so while the tag was being dropped the second
//!   call of `resize(0.5).resize(0.5)` read a different alpha ceiling from the
//!   first, whichever way the input was tagged.
//! * **`resize` composition.** The scale is split per axis: any downscale
//!   runs `reducev` / `reduceh` with the chosen kernel (default `lanczos3`,
//!   gap 2), any residual upscale runs `affine` with the interpolator
//!   mapped from the kernel (`nearest` to nearest, `linear` to bilinear,
//!   everything else to bicubic), input displacement 0.5 for centre
//!   sampling, copy extension, and premultiplication skipped. The `nearest`
//!   kernel subsamples by the integer part first and enlarges integral
//!   factors by pixel replication (`vips_zoom`).
//! * **`affine` geometry.** The matrix `[a, b, c, d]` maps input to output
//!   as `x' = a*x + b*y + odx`, `y' = c*x + d*y + ody`. The default output
//!   area is the bounding box of the transformed input corners, rounded to
//!   nearest, computed from the matrix alone (the `odx` / `ody` / `idx` /
//!   `idy` displacements do not move the default area, matching the
//!   `vips_affine_build` ordering). Each output pixel is inverse-mapped and
//!   interpolated; positions whose floor falls outside `[-1, dim - 1]` are
//!   painted with the background, and interpolation taps outside the image
//!   read the [`Extend`] mode (background 0 by default), reproducing the
//!   one-pixel anti-aliased border of `vips_affine_gen`. Both inks are
//!   converted to the carrier once before any resampling, the way
//!   `vips_affine_build` runs `vips__vector_to_ink` once before it embeds:
//!   clipped and truncated toward zero on an integer carrier, narrowed to
//!   `f32` on a float one (issue #736). Carrying
//!   the caller's `f64` into the convolution instead was worth up to 75 of 255
//!   on a byte carrier with an out-of-range background. `vips_affine` grows
//!   that border by embedding the input with the caller's extend mode before
//!   it resamples (`affine.c:534`), so on a raster **without** an alpha band
//!   an [`Extend::White`] tap is inked the way `vips_embed` inks one, from the
//!   interpretation (`white_ink`, issue #667) and not from the sample depth,
//!   and the measured cells match `vips embed --extend white` cell for cell.
//!   They stop matching once the raster carries alpha, and the reason is not
//!   the paint order (issue #692). `vips_affine_build` embeds **before** it
//!   premultiplies (`affine.c:529`, then `affine.c:551`), so the border is
//!   painted in the raster's own domain either way and `vips_region_paint`
//!   memsets an integer carrier exactly as it does for a bare `vips_embed`.
//!   What moves the value afterwards is that the premultiply / un-premultiply
//!   pair does **not** cancel on that pixel: `vips_premultiply` builds its
//!   multiplier from a **clipped** alpha, `nalpha = clip(a, 0, M) / M`, while
//!   `vips_unpremultiply` builds its reciprocal from the **raw** one,
//!   `factor = M / a`, deliberately ("we want over and undershoots on alpha
//!   and RGB to cancel", `unpremultiply.c:78`). A border pixel holds the same
//!   ink `E` in every band including alpha, so the round trip is
//!   `E * clip(E, 0, M) / M * M / E`, which is `clip(E, 0, M)`: the ink comes
//!   back clipped to the **interpretation's** ceiling.
//!
//!   libviprs does the same arithmetic against its own ceiling, which is the
//!   **depth's** on an unsigned carrier (`bracket_max_alpha`, issue #664), and
//!   the white ink never exceeds that, so `clip(E, 0, D)` is just `E`. The two
//!   therefore agree everywhere except where the tag's ceiling sits below the
//!   carrier's depth. Measured on 8.18.6 with `--interpolate nearest`, whose
//!   window is one pixel, so an output shifted one step off the input reads the
//!   **pure** ink with no blend to solve back out:
//!
//!   | carrier | bands | tag | `embed white` | `affine white` | libviprs |
//!   |---|---|---|---|---|---|
//!   | `uchar` | 3 | none / `srgb` | 255 | 255 | 255 |
//!   | `uchar` | 4 | `srgb` | 255 | 255 | 255 |
//!   | `uchar` | 4 | `scrgb` | 1 | 1 | 1 |
//!   | `ushort` | 3 | `srgb` | 65535 | 65535 | 65535 |
//!   | `ushort` | 4 | `srgb` | 65535 | **255** | 65535 |
//!   | `ushort` | 3 | `scrgb` | 257 | 257 | 257 |
//!   | `ushort` | 4 | `scrgb` | 257 | **1** | 257 |
//!   | `ushort` | 4 | `rgb16` | 65535 | 65535 | 65535 |
//!   | `ushort` | 2 | `b-w` | 65535 | **255** | 65535 |
//!   | `float` | 4 | `srgb` / `scrgb` / `rgb16` | 255 / 1 / 65535 | same | same |
//!
//!   The three bold cells are the whole divergence, and every one of them is a
//!   16-bit raster wearing an 8-bit tag. **libviprs does not follow vips here,
//!   and that is decided rather than pending.** Two reasons, both measured:
//!
//!   1. The border ink is not separately settable. What comes out is
//!      `clip(E, 0, ceiling)` for whatever ceiling the bracket uses, so
//!      matching the pure-ink cell means either changing the ceiling or
//!      pre-clipping the fill. Pre-clipping fixes the pure-ink pixels and
//!      leaves every *blended* one wrong, because the two premultiplied spaces
//!      are scaled differently: on a `ushort` `srgb` raster with alpha 200 a
//!      real pixel of 25000 sits at 19608 in vips' premultiplied image and at
//!      76.3 in this module's. A change that fixes the corner and not the
//!      fringe looks like a fix and is not one.
//!   2. Changing the ceiling means `bracket_max_alpha` following the tag on
//!      unsigned carriers, which is exactly what #664 decided against, and the
//!      price is not theoretical. `vips affine` on a constant-25000 `ushort`
//!      RGBA tagged `srgb` returns **255 for every interior sample**, not just
//!      at the border; tagged `scrgb` it returns 1; and with alpha 65535 a
//!      colour of 25000 comes back as 97. libviprs returns 25000 in all three.
//!
//!   So the border follows the ceiling, the ceiling is #664's, and this module
//!   is self-consistent under it. The suite pins both halves: the agreeing
//!   cells so the divergence stays bounded to those three, and the interior
//!   round-trip so the cost of adopting vips' reading is a number rather than
//!   an assertion.
//! * **Premultiplied alpha.** Like `vips_affine`, images with an alpha band
//!   are premultiplied before interpolation and unpremultiplied afterwards
//!   unless [`AffineOptions::premultiplied`] says the input already is. The
//!   alpha ceiling is the one `vips_premultiply` defaults to, and that is a
//!   property of the **interpretation** rather than of the sample depth
//!   (`vips_interpretation_max_alpha`, issue #664): 65535 for
//!   [`Interpretation::Rgb16`] / [`Interpretation::Grey16`], 1.0 for
//!   [`Interpretation::ScRgb`], 255 otherwise. The unsigned carriers keep the
//!   depth ceiling, where an untagged raster gives the same answer either way;
//!   only a float carrier, which has no depth-implied ceiling at all, reads the
//!   tag. Both ends of the bracket round through `f32` exactly where the C
//!   does, because
//!   `OUT nalpha` and `OUT factor` are `float` for every carrier but DOUBLE
//!   (`premultiply.c:229-232`), so the multiplier is quantised before the
//!   colour multiply even on an 8-bit input. The affine half has a third such
//!   point, since `vips_affine` premultiplies into a **FLOAT** image and
//!   `vips_unpremultiply` reads that image back, so the interpolated pixel is
//!   quantised at the seam between them; the interpolation itself accumulates
//!   in `f64`, which is what `BILINEAR_FLOAT` does with its `double`
//!   coefficients (`interpolate.c:462`). Bicubic is the exception, and only
//!   because the premultiply moved the carrier: `bicubic_float<float>` narrows
//!   each row sum to `f32` (issue #705, and the per-carrier table above), so a
//!   premultiplied raster takes that narrowing whatever its stored depth was. The averaging resamplers —
//!   `reduce` / `reduceh` / `reducev`, `shrink` / `shrinkh` / `shrinkv`, and
//!   `resize` — do the same: an alpha image is premultiplied once into a float
//!   working buffer, the separable box / kernel / affine passes all run in that
//!   premultiplied space, and the result is unpremultiplied once at the end
//!   (the `vips_resize` bracket). This coverage-weights the colour so the
//!   meaningless RGB of transparent pixels cannot bleed into opaque neighbours
//!   (the dark fringe at transparency boundaries). Note this is a deliberate
//!   divergence from the bare `vips_reduce*` / `vips_shrink*` namesakes, which
//!   do *not* premultiply — only `vips_resize` does — but it is the behaviour
//!   the pyramid pipeline needs by default and matches a premultiplied vips
//!   pipeline (`premultiply | reduce/shrink | unpremultiply`). The single-tap
//!   Nearest kernel is exempt: it does no averaging, so it stays an exact pick
//!   with no premultiply round-trip.
//! * **Do not hand an already-premultiplied image to `resize`.** This is the
//!   trap the bullet above sets, and it has already been walked into once
//!   (issue #603), so it is worth spelling out. `vips_resize` does *not*
//!   premultiply — "This operation does not premultiply alpha. If your image
//!   has an alpha channel, you should use premultiply on it first",
//!   `libvips/resample/resize.c` — which means a vips caller that premultiplies
//!   first, as `vips_smartcrop_build` does, gets an image that is *still*
//!   premultiplied on the other side. libviprs' `resize` premultiplies on its
//!   own, so the same call here un-premultiplies it instead, and the colour
//!   sitting behind transparent pixels comes back out amplified by
//!   `max / alpha`. Any op that ports a vips pipeline of the form
//!   `premultiply | resize` must therefore either drop the alpha band before
//!   the resize (what [`crate::extract`]'s attention smartcrop does, since vips
//!   discards it right after the resize anyway) or not premultiply first.
//!   The un-premultiply dead zone (issue #604) bounds the damage but does
//!   not remove it.
//! * **`similarity` / `rotate`.** `similarity(angle, scale)` builds the
//!   matrix `a = scale*cos, b = -scale*sin, c = -b, d = a` and calls
//!   `affine` with the default bilinear interpolator; `rotate(angle)` is
//!   `similarity(angle, 1.0)`. Note that libvips affine rotations sample on
//!   a grid displaced by the bounding-box rounding, so `rotate(90.0)` is
//!   the exact `rot90` permutation shifted one column right with a
//!   background seam in column 0. The unit tests pin this faithfully.
//! * **`mapim`.** The index image must have exactly two bands (band 0 is
//!   the source x, band 1 the source y). Coordinates inside
//!   `[-1, dim + 1)` are interpolated with background-extended taps (edge
//!   antialiasing); everything else, including NaN, paints the background.
//! * **Interpolators.** `nearest`, `bilinear`, `bicubic` (Catmull-Rom,
//!   the libvips `VipsInterpolateBicubic` coefficients), `nohalo`, and
//!   `lbb` are all implemented. `nohalo` and `lbb` are faithful ports of
//!   the libvips `nohalo.cpp` and `lbb.cpp` minmod-subdivision resamplers:
//!   `lbb` is locally bounded bicubic (a nonlinear Catmull-Rom variant
//!   whose reconstruction stays within the range of the 16 nearest input
//!   samples, so it never overshoots), and `nohalo` is level-1 co-monotone
//!   subdivision (minmod slopes) finished with `lbb`. Both centre and
//!   reflect their stencils exactly as the C interpolators do (`lbb` at
//!   window offset 1, `nohalo` at window offset 2 with round-to-nearest
//!   centring), so on samples that land exactly on the input grid they
//!   return the input pixel unchanged, which keeps the 4x rotation
//!   round-trip an identity.
//!
//! # Divergence from stock libvips
//!
//! Four gaps are open between this module and a stock libvips, all four are
//! quantisation rather than a different convolution, and all four are kept on
//! purpose. The rule that decided them, and that decided #704 the other way, is
//! measured rather than stylistic:
//!
//! > Adopt libvips' arithmetic when the difference is inside the carrier's own
//! > noise **and** points both ways, so neither implementation is the more
//! > accurate one. Keep this module's when the difference is large or
//! > one-directional, and pin libvips' answer so the gap stays visible.
//!
//! Against the exact answer in rational arithmetic, on real `affine` output:
//!
//! | | this module | libvips | libvips closer |
//! |---|---|---|---|
//! | #704 bicubic coefficients, `uchar` | 0.4371 LSB | 0.4798 LSB | 1355 of 17814 |
//! | #732 bicubic store, `ushort` | 0.0000 LSB | 0.4680 LSB | 0 of 1017 |
//! | #733 bilinear weights, `uchar` | 0.0000 LSB | 0.0252 LSB | 0 of 1113 |
//! | #733 bilinear weights, `ushort` | 0.0000 LSB | 6.2848 LSB | 0 of 1113 |
//! | #777 reduce mask, `uchar` | 0.2628 LSB | 0.2667 LSB | 51 of 62685 |
//! | #777 reduce mask, `ushort` | 0.2558 LSB | 10.1088 LSB | 0 of 43889 |
//!
//! #704 was a coin toss the project took for parity, because both spellings
//! were within a hair of each other and this module was not implementing either
//! one cleanly after #668 put the offsets on libvips' grid. The others are not
//! coin tosses: on the interpolator rows this module is exact and libvips is
//! not, on every sample, and on the two `reduce` rows the byte column is a hair
//! and the 16-bit one is 40 times #704's whole disagreement, all of it in the
//! same direction.
//!
//! * **`reduce` reads a `short` mask on the integer carriers.**
//!   `vips_reduce_make_mask` builds every mask in `double` and keeps a second
//!   copy scaled by `VIPS_INTERPOLATE_SCALE` and truncated toward zero, and
//!   the reduce generators read that copy for `UCHAR`, `CHAR`, `USHORT` and
//!   `SHORT`. It is **not** renormalised, so its taps no longer sum to one and
//!   the whole image is scaled by whatever the quantisation left behind. This
//!   module keeps the `f64` masks (issue #777).
//!
//!   The sharpest form of it needs no fixture at all: a constant image
//!   survives `reduce` here and does not survive it there. Measured on 8.18.6
//!   over a 32x32 constant 65535 `ushort`, six of fifteen kernel-by-shrink
//!   cells come back short, `lanczos3` at shrink 4 by **128 of 65535**, with
//!   `vips min` equal to `vips max` in every cell so the output really is
//!   flat. The byte carrier is clean on both sides, and that is the same
//!   measurement rather than a second one: `128 / 65535` is 0.498 of a byte
//!   level, so round-half-up absorbs it with two thousandths of a level to
//!   spare.
//!
//!   On real content, 94080 `uchar` and 66240 `ushort` samples over three
//!   fixtures crossed with five kernels and seven shrink factors, this module
//!   differs from the binary on 7.57% of the byte samples at 1 level and on
//!   61.8% of the 16-bit ones at up to 123. The table above is the accuracy
//!   side of that, and the `ushort` row's signed mean is **-5.81 LSB**: the
//!   `short` mask darkens every 16-bit image it touches.
//!
//!   Porting it would also not close the gate, which is the second reason and
//!   the one that separates this from #704. The copy is `matrixf[x][i] * 4096`
//!   truncated by a C `double`-to-`int` conversion, over a mask normalised by
//!   a floating-point sum, so a coefficient whose exact value is a multiple of
//!   1/4096 falls either side of the boundary on its last bit. `mitchell` at
//!   shrink 4 has one: taps 1 and 14 are exactly `-27/4096`, this module's
//!   `f64` normalisation lands 3 ulp below it and truncates to -27 where the
//!   binary has -26. A full port reproduces 94035 of 94080 byte samples and
//!   leaves **45** at 1 level for that reason alone, so the 1-level allowance
//!   stays whichever way this goes. #704 bought parity by giving up 0.043 of a
//!   level; here 0.004 of a level buys 99.95% of it and not the allowance.
//!
//! * **`vips_cast` truncates where this module rounds.** Whenever libvips
//!   brackets a resample in a premultiply it works in FLOAT and casts back to
//!   the input format afterwards (`vips_affine`'s
//!   `vips_cast(t[5], &t[6], unpremultiplied_format)`, `affine.c:616`), and
//!   `vips_cast` converts a float to an integer with a plain C cast, which
//!   truncates towards zero: `q[x] = CAST((double) p[x])` where `CAST` only
//!   clips (`cast.c:237`), and the file's own header note says "now does
//!   floor(), not rint() ... you'll need to round yourself". libviprs stores
//!   through its own sample writer, which rounds half up like
//!   `VIPS_ROUND_UINT`. So on an unsigned carrier **with an alpha band** the
//!   two disagree wherever the premultiply round-trip lands a hair below the
//!   value it started from: vips floors the step away and libviprs keeps it.
//!   Measured on 8.18.6, `affine "0.5 0 0 0.5"` bilinear over a 64x64
//!   pseudo-random `Rgba8` moves 240 of 4096 samples, every one of them
//!   libviprs one **above** vips and equal to the input sample, so this module
//!   is the one that round-trips. It is unchanged by issue #664 and predates
//!   it. Float carriers are unaffected, because nothing is requantised.
//!
//!   It matters when reading the oracle: a comparison run as
//!   `premultiply | resize | unpremultiply | cast uchar` disagrees with this
//!   module on about half of all samples for that reason alone. Read the
//!   unpremultiplied result as FLOAT and quantise it the same way instead,
//!   which is what `resize_unsigned_bracket_matches_the_vips_oracle_on_varying_data`
//!   does.
//!
//! * **`affine` bicubic truncates its own store on a `ushort` carrier**
//!   (issue #732). `bicubic_unsigned_int32_tab` finishes `out[z] = bicubic`
//!   with `out` an `unsigned short *` and `bicubic` a `double`, so that store
//!   truncates too, and this one needs no alpha band and no premultiply to
//!   fire. Measured on 8.18.6 over a random 24x24: 691 of 1764 samples at
//!   exactly 1 LSB, and modelling the truncation instead reproduces the binary
//!   in 0 of 1764. It is the same shape as the `vips_cast` gap above and it
//!   goes the same way: truncation is a one-directional bias of -0.499 LSB
//!   where round-half-up is +0.0006, so libvips darkens every resampled `ushort`
//!   image by half a level and this module round-trips.
//!   `affine_bicubic_rounds_the_ushort_store_where_vips_truncates` pins it on a
//!   linear ramp, where Catmull-Rom's answer is closed form: libvips hits
//!   `floor(exact)` on 441 of 441 interior samples and this module hits
//!   `round(exact)` on 441 of 441.
//!
//! * **`bilinear` keeps `f64` weights on the carriers where vips uses 12-bit
//!   fixed point** (issue #733). `SWITCH_INTERPOLATE` sends `UCHAR`, `CHAR`,
//!   `USHORT` and `SHORT` to `BILINEAR_INT`, whose four weights are
//!   `(x - ix) * VIPS_INTERPOLATE_SCALE` truncated to an `int`. That is worth
//!   1 LSB on a byte carrier and up to **26** on a 16-bit one, because a weight
//!   quantised to 1/4096 costs `65535 / 4096` of a sample. Measured over the
//!   same random 24x24: 31 of 1764 at `uchar`, 1345 of 1764 at `ushort`, and a
//!   `BILINEAR_INT` model reproduces the binary in 0 and 5 respectively (those
//!   5 are the equidistant-neighbour ties every interpolator shares).
//!
//!   This one stays too, and it is the least close of the three calls. Bilinear
//!   reproduces a linear function exactly, so there is a closed-form right
//!   answer, and this module hits it on 529 of 529 interior samples of a ramp
//!   where libvips misses it on 529 of 529. Adopting the 12-bit weights would
//!   introduce a mean error of 6.28 LSB on a 16-bit carrier where there is
//!   currently none, and libvips is closer on 0 of 1113 samples. What the
//!   argument for adopting it does buy, a gate that can see a regression, comes
//!   from pinning the divergence instead:
//!   `affine_bilinear_divergence_from_the_12_bit_weights_is_bounded` asserts it
//!   from both sides, so it can neither grow nor quietly vanish.
//!
//! # Example usage
//!
//! * [ported_resample tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/ported_resample.rs)

use crate::arithmetic::{interpretation_max_alpha, unpremultiply_factor};
use crate::colour::{ColourError, Intent, Pcs};
use crate::conversion::Interpretation;
use crate::extract::{Extend, ExtractError, white_ink};
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use crate::source::SourceError;
use std::f64::consts::PI;
use std::path::Path;
use thiserror::Error;

/// Largest reduce mask supported, the libvips `MAX_POINT` from
/// `resample/presample.h`.
const MAX_POINT: usize = 2000;

/// Largest accepted shrink / reduce factor, the libvips argument ceiling on
/// `vips_shrink` and `vips_reduce`.
const MAX_FACTOR: f64 = 1_000_000.0;

/// Determinant threshold below which an affine matrix is treated as
/// singular, the libvips `TOO_SMALL` from `resample/transform.c`.
const TOO_SMALL: f64 = 2.0 * f64::MIN_POSITIVE;

/// Typed errors for the resampling operations in [`crate::resample`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ResampleError {
    /// The interpolator name is not a libvips interpolator nickname.
    #[error(
        "unknown interpolator {name:?}; expected \"nearest\", \"bilinear\", \"bicubic\", \"nohalo\" or \"lbb\""
    )]
    UnknownInterpolator { name: String },
    /// The kernel name is not a libvips `VipsKernel` nickname.
    #[error(
        "unknown kernel {name:?}; expected \"nearest\", \"linear\", \"cubic\", \"mitchell\", \"lanczos2\" or \"lanczos3\""
    )]
    UnknownKernel { name: String },
    /// A shrink or reduce factor is not a finite number in
    /// `1.0..=1_000_000.0`.
    #[error("{op} factor should be in 1.0..=1000000.0, got {factor}")]
    BadFactor { op: &'static str, factor: f64 },
    /// The reducing gap is below 1.0.
    #[error("reduce gap should be >= 1.0, got {gap}")]
    GapTooSmall { gap: f64 },
    /// The reduce mask would exceed the libvips `MAX_POINT` limit.
    #[error("reduce factor too large: {n_point}-point mask exceeds {max}")]
    FactorTooLarge { n_point: usize, max: usize },
    /// A resize scale is not a finite positive number.
    #[error("resize scale should be a finite positive number, got {scale}")]
    BadScale { scale: f64 },
    /// The output would have a zero dimension.
    #[error("image has shrunk to nothing")]
    ShrunkToNothing,
    /// The affine matrix is singular or near-singular.
    #[error("singular or near-singular matrix")]
    SingularMatrix,
    /// The affine output area is empty or does not fit in `u32` dimensions.
    #[error("bad affine output area {width}x{height}")]
    BadOutputArea { width: i64, height: i64 },
    /// The mapim index image does not have exactly two bands.
    #[error("mapim index image must have 2 bands, got {bands}")]
    IndexBands { bands: usize },
    /// Constructing the result raster failed (allocation budget, size
    /// overflow).
    #[error(transparent)]
    Raster(#[from] RasterError),
}

#[track_caller]
fn expect_resample<T>(op: &str, r: Result<T, ResampleError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

/// Typed errors for the [`Raster::thumbnail`] family (libvips
/// `vips_thumbnail`): decode, resample, colour, and crop failures folded
/// into one surface so the panicking forms report a single cause.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ThumbnailError {
    /// The target width (or height) is zero; a thumbnail box must be at
    /// least one pixel on each side.
    #[error("thumbnail target size must be at least 1 pixel, got {size}")]
    BadSize { size: u32 },
    /// The output profile name is not a recognised built-in
    /// ([`Raster::thumbnail_with_profile`] currently accepts only
    /// `"srgb"`).
    #[error("unknown output profile {name:?}; expected \"srgb\"")]
    UnknownProfile { name: String },
    /// The built-in sRGB profile could not be encoded.
    #[error("could not build the built-in sRGB profile: {0}")]
    Profile(String),
    /// Decoding the source file or buffer failed.
    #[error(transparent)]
    Decode(#[from] SourceError),
    /// A resampling step (reduce / resize / affine) failed.
    #[error(transparent)]
    Resample(#[from] ResampleError),
    /// A colour step (linear import, ICC import / export) failed.
    #[error(transparent)]
    Colour(#[from] ColourError),
    /// The crop-to-box step failed.
    #[error(transparent)]
    Extract(#[from] ExtractError),
}

#[track_caller]
fn expect_thumbnail(r: Result<Raster, ThumbnailError>) -> Raster {
    match r {
        Ok(v) => v,
        Err(e) => panic!("thumbnail: {e}"),
    }
}

/// A point resampler for [`Raster::affine`] and [`Raster::mapim`] (libvips
/// `VipsInterpolate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Interpolator {
    /// Nearest neighbour: the sample whose floor position contains the
    /// point.
    Nearest,
    /// Bilinear blend of the surrounding 2x2 samples.
    Bilinear,
    /// Catmull-Rom bicubic over the surrounding 4x4 samples, the libvips
    /// `VipsInterpolateBicubic` coefficients.
    Bicubic,
    /// Nohalo level-1 co-monotone subdivision finished with LBB, the
    /// libvips `VipsInterpolateNohalo` (`nohalo.cpp`). A halo-reducing,
    /// edge-sharpening resampler that stays within the range of nearby
    /// input samples.
    Nohalo,
    /// Locally bounded bicubic, the libvips `VipsInterpolateLbb`
    /// (`lbb.cpp`). A nonlinear Catmull-Rom variant whose reconstruction
    /// is bounded by the 16 nearest input samples, so it produces no
    /// overshoot.
    Lbb,
}

impl Interpolator {
    /// Parse a libvips interpolator nickname.
    ///
    /// # Errors
    ///
    /// [`ResampleError::UnknownInterpolator`] for any name that is not a
    /// recognised libvips interpolator nickname.
    pub fn from_name(name: &str) -> Result<Self, ResampleError> {
        match name {
            "nearest" => Ok(Self::Nearest),
            "bilinear" => Ok(Self::Bilinear),
            "bicubic" => Ok(Self::Bicubic),
            "nohalo" => Ok(Self::Nohalo),
            "lbb" => Ok(Self::Lbb),
            _ => Err(ResampleError::UnknownInterpolator {
                name: name.to_string(),
            }),
        }
    }
}

/// A downsampling kernel for [`Raster::reduce`] and [`Raster::resize`]
/// (libvips `VipsKernel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReduceKernel {
    /// Point sample.
    Nearest,
    /// Triangle (tent) filter.
    Linear,
    /// Catmull-Rom cubic.
    Cubic,
    /// Mitchell-Netravali cubic (B = C = 1/3).
    Mitchell,
    /// Two-lobe Lanczos windowed sinc.
    Lanczos2,
    /// Three-lobe Lanczos windowed sinc, the libvips default.
    Lanczos3,
}

impl ReduceKernel {
    /// Parse a libvips `VipsKernel` nickname.
    ///
    /// # Errors
    ///
    /// [`ResampleError::UnknownKernel`] when the name is not one of
    /// `"nearest"`, `"linear"`, `"cubic"`, `"mitchell"`, `"lanczos2"`,
    /// `"lanczos3"`.
    pub fn from_name(name: &str) -> Result<Self, ResampleError> {
        match name {
            "nearest" => Ok(Self::Nearest),
            "linear" => Ok(Self::Linear),
            "cubic" => Ok(Self::Cubic),
            "mitchell" => Ok(Self::Mitchell),
            "lanczos2" => Ok(Self::Lanczos2),
            "lanczos3" => Ok(Self::Lanczos3),
            _ => Err(ResampleError::UnknownKernel {
                name: name.to_string(),
            }),
        }
    }

    /// Mask size for a shrink factor (`vips_reduce_get_points`). Always odd.
    fn points(self, shrink: f64) -> usize {
        match self {
            Self::Nearest => 1,
            Self::Linear => 2 * shrink.round_ties_even() as usize + 1,
            Self::Cubic | Self::Mitchell | Self::Lanczos2 => {
                2 * (2.0 * shrink).round_ties_even() as usize + 1
            }
            Self::Lanczos3 => 2 * (3.0 * shrink).round_ties_even() as usize + 1,
        }
    }

    /// The kernel function at distance `x` from the centre
    /// (`resample/templates.h` `filter<K>`).
    fn filter(self, x: f64) -> f64 {
        match self {
            // Nearest masks are built directly in `mask`.
            Self::Nearest => 0.0,
            Self::Linear => {
                let ax = x.abs();
                if ax < 1.0 { 1.0 - ax } else { 0.0 }
            }
            Self::Cubic => cubic_filter(x, 0.0, 0.5),
            Self::Mitchell => cubic_filter(x, 1.0 / 3.0, 1.0 / 3.0),
            Self::Lanczos2 => {
                if (-2.0..=2.0).contains(&x) {
                    sinc_filter(x) * sinc_filter(x / 2.0)
                } else {
                    0.0
                }
            }
            Self::Lanczos3 => {
                if (-3.0..=3.0).contains(&x) {
                    sinc_filter(x) * sinc_filter(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }

    /// Fill `c` with the mask for sub-pixel offset `x` in `[0, 1]`,
    /// normalised to unit sum (`vips_reduce_make_mask` over
    /// `calculate_coefficients`).
    fn mask(self, c: &mut [f64], shrink: f64, x: f64) {
        if self == Self::Nearest {
            c[0] = 1.0;
            return;
        }
        let n = c.len();
        let half = x + n as f64 / 2.0 - 1.0;
        let mut sum = 0.0;
        for (i, ci) in c.iter_mut().enumerate() {
            let xp = (i as f64 - half) / shrink;
            *ci = self.filter(xp);
            sum += *ci;
        }
        for ci in c.iter_mut() {
            *ci /= sum;
        }
    }

    /// The interpolator `vips_resize` upsizes with for this kernel
    /// (`vips_resize_interpolate`).
    fn upsize_interpolator(self) -> Interpolator {
        match self {
            Self::Nearest => Interpolator::Nearest,
            Self::Linear => Interpolator::Bilinear,
            _ => Interpolator::Bicubic,
        }
    }
}

/// The two-parameter cubic family from `resample/templates.h`
/// (`cubic_filter`): B = 0, C = 0.5 is Catmull-Rom, B = C = 1/3 is
/// Mitchell-Netravali.
fn cubic_filter(x: f64, b: f64, c: f64) -> f64 {
    let ax = x.abs();
    let ax2 = ax * ax;
    let ax3 = ax2 * ax;
    if ax <= 1.0 {
        ((12.0 - 9.0 * b - 6.0 * c) * ax3 + (-18.0 + 12.0 * b + 6.0 * c) * ax2 + (6.0 - 2.0 * b))
            / 6.0
    } else if ax <= 2.0 {
        ((-b - 6.0 * c) * ax3
            + (6.0 * b + 30.0 * c) * ax2
            + (-12.0 * b - 48.0 * c) * ax
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

/// Normalised sinc (`resample/templates.h` `sinc_filter`).
fn sinc_filter(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * PI;
    x.sin() / x
}

/// Options for [`Raster::try_affine_with`], mirroring the optional
/// arguments of `vips_affine`.
///
/// `#[non_exhaustive]` and `Default`, the same shape as
/// [`DecodeLimits`](crate::source::DecodeLimits): start from
/// [`AffineOptions::default`] and set what you need with the `with_*`
/// builders. `vips_affine` grows optional arguments, so this one will grow
/// fields, and taking the struct literal away now is what stops that being a
/// breaking change later (issue #630).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct AffineOptions {
    /// Horizontal output displacement (`odx`).
    pub odx: f64,
    /// Vertical output displacement (`ody`).
    pub ody: f64,
    /// Horizontal input displacement (`idx`).
    pub idx: f64,
    /// Vertical input displacement (`idy`).
    pub idy: f64,
    /// Output rectangle `[left, top, width, height]` (`oarea`); the
    /// bounding box of the transformed input when `None`.
    pub oarea: Option<[i32; 4]>,
    /// How interpolation taps outside the input read (`extend`); the
    /// libvips default is [`Extend::Background`].
    pub extend: Extend,
    /// Background sample value for [`Extend::Background`] taps and for
    /// output pixels outside the transformed input (`background`,
    /// broadcast to every band).
    pub background: f64,
    /// The input already has premultiplied alpha, so skip the
    /// premultiply / unpremultiply pair (`premultiplied`).
    pub premultiplied: bool,
}

impl Default for AffineOptions {
    fn default() -> Self {
        Self {
            odx: 0.0,
            ody: 0.0,
            idx: 0.0,
            idy: 0.0,
            oarea: None,
            extend: Extend::Background,
            background: 0.0,
            premultiplied: false,
        }
    }
}

impl AffineOptions {
    /// Set the horizontal output displacement (`odx`), returning the updated
    /// options.
    #[must_use]
    pub fn with_odx(mut self, odx: f64) -> Self {
        self.odx = odx;
        self
    }

    /// Set the vertical output displacement (`ody`), returning the updated
    /// options.
    #[must_use]
    pub fn with_ody(mut self, ody: f64) -> Self {
        self.ody = ody;
        self
    }

    /// Set the horizontal input displacement (`idx`), returning the updated
    /// options.
    #[must_use]
    pub fn with_idx(mut self, idx: f64) -> Self {
        self.idx = idx;
        self
    }

    /// Set the vertical input displacement (`idy`), returning the updated
    /// options.
    #[must_use]
    pub fn with_idy(mut self, idy: f64) -> Self {
        self.idy = idy;
        self
    }

    /// Set the output rectangle `[left, top, width, height]` (`oarea`),
    /// returning the updated options. `None` uses the bounding box of the
    /// transformed input.
    #[must_use]
    pub fn with_oarea(mut self, oarea: Option<[i32; 4]>) -> Self {
        self.oarea = oarea;
        self
    }

    /// Set how interpolation taps outside the input read (`extend`),
    /// returning the updated options.
    #[must_use]
    pub fn with_extend(mut self, extend: Extend) -> Self {
        self.extend = extend;
        self
    }

    /// Set the background sample value (`background`), returning the updated
    /// options.
    #[must_use]
    pub fn with_background(mut self, background: f64) -> Self {
        self.background = background;
        self
    }

    /// Declare that the input already has premultiplied alpha
    /// (`premultiplied`), returning the updated options.
    #[must_use]
    pub fn with_premultiplied(mut self, premultiplied: bool) -> Self {
        self.premultiplied = premultiplied;
        self
    }
}

/// Options for [`Raster::try_resize_with`], mirroring the optional
/// arguments of `vips_resize`.
///
/// `#[non_exhaustive]` and `Default`, the same shape as
/// [`DecodeLimits`](crate::source::DecodeLimits): start from
/// [`ResizeOptions::default`] and set what you need with the `with_*`
/// builders, e.g. `ResizeOptions::default().with_vscale(Some(0.5))`
/// (issue #630).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ResizeOptions {
    /// Vertical scale factor; the horizontal scale when `None`.
    pub vscale: Option<f64>,
    /// Downsampling kernel (libvips default `lanczos3`).
    pub kernel: ReduceKernel,
    /// Reducing gap (libvips default 2.0).
    pub gap: f64,
}

impl Default for ResizeOptions {
    fn default() -> Self {
        Self {
            vscale: None,
            kernel: ReduceKernel::Lanczos3,
            gap: 2.0,
        }
    }
}

impl ResizeOptions {
    /// Set the vertical scale factor, returning the updated options. `None`
    /// reuses the horizontal scale.
    #[must_use]
    pub fn with_vscale(mut self, vscale: Option<f64>) -> Self {
        self.vscale = vscale;
        self
    }

    /// Set the downsampling kernel, returning the updated options.
    #[must_use]
    pub fn with_kernel(mut self, kernel: ReduceKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the reducing gap, returning the updated options.
    #[must_use]
    pub fn with_gap(mut self, gap: f64) -> Self {
        self.gap = gap;
        self
    }
}

// ---------------------------------------------------------------------------
// Sample plumbing
// ---------------------------------------------------------------------------

/// `VIPS_ROUND_UINT`: round a non-negative quantity to nearest, half up.
fn round_uint(v: f64) -> i64 {
    (v + 0.5).floor() as i64
}

/// `VIPS_ROUND_INT`: round to nearest, half away from zero, with the C
/// truncation-toward-zero cast.
fn round_int(v: f64) -> i64 {
    if v >= 0.0 {
        (v + 0.5) as i64
    } else {
        (v - 0.5) as i64
    }
}

/// `VIPS_TRANSFORM_SCALE` (`interpolate.h:109`): libvips precomputes its
/// resampling coefficients at 65 sub-pixel positions, one every 1/64 of a
/// pixel.
const TRANSFORM_SCALE: i64 = 64;

/// Round the sub-pixel part of a continuous source coordinate onto the grid
/// libvips builds those tables on, so a mask evaluated here is the mask a table
/// lookup would have returned.
///
/// This is not a precision detail, it is the answer. Neither `vips_reduceh_gen`
/// nor `vips_interpolate_bicubic_interpolate` ever evaluates its kernel at the
/// true offset; both index a table built at `(float) x / VIPS_TRANSFORM_SCALE`
/// and both spell the index with the same five lines (`reduceh.cpp:270-276`,
/// `bicubic.cpp:496-503`):
///
/// ```c
/// const int sx = X * VIPS_TRANSFORM_SCALE * 2;
/// const int six = sx & (VIPS_TRANSFORM_SCALE * 2 - 1);
/// const int tx = (six + 1) >> 1;
/// ```
///
/// A dyadic scale lands every offset on the grid and the rounding is invisible,
/// which is why [`Raster::resize`] agreed with the binary at 0.5, 0.25, 0.125
/// and 2.0 while missing 1.5, 0.75, 0.37 and 0.3 by up to 2.3 (issue #668).
///
/// The `floor` is deliberate where the C truncates. `(int)(X * 128)` rounds
/// toward zero and `& 127` reads two's complement, so on a negative coordinate
/// that pair picks the bucket above the one `floor` picks. vips never meets the
/// case: `vips_affine_gen` hands the interpolator a coordinate in the embedded
/// space, shifted by `window_offset` and so never below 1 (`affine.c:361-362`),
/// while libviprs interpolates in the input's own coordinates, which go
/// negative on the first output column of any enlargement past 2x. Flooring
/// agrees with vips on both signs; truncating agrees only on the positive one.
fn table_offset(x: f64) -> f64 {
    let sx = (x * (TRANSFORM_SCALE * 2) as f64).floor() as i64;
    let six = sx.rem_euclid(TRANSFORM_SCALE * 2);
    ((six + 1) >> 1) as f64 / TRANSFORM_SCALE as f64
}

/// `VIPS_INTERPOLATE_SHIFT` / `VIPS_INTERPOLATE_SCALE` (`interpolate.h:117`):
/// the 12-bit fixed point libvips accumulates its `uchar` interpolators in.
const INTERPOLATE_SHIFT: u32 = 12;
const INTERPOLATE_SCALE: i64 = 1 << INTERPOLATE_SHIFT;

/// One row of `vips_bicubic_matrixi`: the Catmull-Rom coefficients for a grid
/// offset, scaled by [`INTERPOLATE_SCALE`] and **truncated toward zero**, which
/// is what the `double` to `int` assignment in
/// `vips_interpolate_bicubic_class_init` does:
///
/// ```c
/// vips_bicubic_matrixi[x][i] = vips_bicubic_matrixf[x][i] * VIPS_INTERPOLATE_SCALE;
/// ```
///
/// Two of the four Catmull-Rom coefficients are negative, so truncating toward
/// zero is not the same as flooring and the difference shows up as a bias in
/// the reconstructed sample. Rust's `as` cast truncates toward zero too, so the
/// spelling carries over directly.
fn fixed_catmull(offset: f64) -> [i64; 4] {
    let mut c = [0.0f64; 4];
    catmull_coefficients(&mut c, offset);
    c.map(|v| (v * INTERPOLATE_SCALE as f64) as i64)
}

/// `unsigned_fixed_round` (`resample/templates.h:152`): bring a fixed-point
/// accumulator back to sample units, rounding half up. The C spells the divide
/// as `>>` on a signed `int`, an arithmetic shift, so a negative accumulator
/// floors rather than truncating; `>>` on `i64` does the same.
fn fixed_round(v: i64) -> i64 {
    (v + (INTERPOLATE_SCALE >> 1)) >> INTERPOLATE_SHIFT
}

/// Per-format sample layout: bytes per channel and float flag.
#[derive(Clone, Copy)]
struct SampleLayout {
    bpc: usize,
    is_float: bool,
    /// Sample ceiling for the **storage** arithmetic: what [`write`] rounds
    /// and clamps an unsigned sample into. 255 for 8-bit and float, 65535 for
    /// 16-bit.
    ///
    /// Not the premultiply denominator, which is a property of the
    /// interpretation rather than of the depth and comes from
    /// [`bracket_max_alpha`] instead (issue #664), and not the
    /// [`Extend::White`] ink either, which is a property of the interpretation
    /// too and comes from [`white_ink`] (issue #667). This ceiling and the
    /// bracket's agree on the unsigned carriers and part only on a float one.
    /// The white ink can differ from both on any carrier, because it follows
    /// the tag the whole way down: `Rgb16` tagged `ScRgb` inks **257** where
    /// this ceiling and the bracket's are both 65535.
    ///
    /// [`write`]: SampleLayout::write
    max: f64,
}

impl SampleLayout {
    fn of(format: PixelFormat) -> Self {
        let bpc = format.bytes_per_channel();
        let is_float = format.is_float();
        let max = if is_float {
            255.0
        } else if bpc == 2 {
            65535.0
        } else {
            255.0
        };
        Self { bpc, is_float, max }
    }

    /// Read sample `i` (flat sample index, not byte index) as `f64`.
    fn read(self, data: &[u8], i: usize) -> f64 {
        let o = i * self.bpc;
        if self.is_float {
            f64::from(f32::from_ne_bytes([
                data[o],
                data[o + 1],
                data[o + 2],
                data[o + 3],
            ]))
        } else if self.bpc == 2 {
            f64::from(u16::from_ne_bytes([data[o], data[o + 1]]))
        } else {
            f64::from(data[o])
        }
    }

    /// Write sample `i` from `f64`, rounding half up and clamping for the
    /// unsigned formats and storing raw `f32` for the float formats.
    fn write(self, data: &mut [u8], i: usize, v: f64) {
        let o = i * self.bpc;
        if self.is_float {
            data[o..o + 4].copy_from_slice(&(v as f32).to_ne_bytes());
        } else {
            let r = (v + 0.5).floor().clamp(0.0, self.max);
            if self.bpc == 2 {
                data[o..o + 2].copy_from_slice(&(r as u16).to_ne_bytes());
            } else {
                data[o] = r as u8;
            }
        }
    }
}

impl SampleLayout {
    /// Convert a caller-supplied ink to the carrier the way
    /// `vips__vector_to_ink` does (issue #736): it builds a one-pixel image of
    /// the doubles and casts it to the input's band format, and `vips_cast`
    /// **clips and then truncates toward zero** on an integer carrier
    /// (`cast.c:237`, and the file's own header note: "now does floor(), not
    /// rint() ... you'll need to round yourself"), and narrows without clipping
    /// on a float one.
    ///
    /// `vips_affine_build` runs that once, before the embed and before the
    /// resample, so both the taps the interpolator reads past the edge and the
    /// pixels `vips_affine_gen` paints outside the transformed input are
    /// already carrier values. Carrying the raw `f64` instead is worth up to
    /// 75 of 255 on a byte carrier with an out-of-range background, because the
    /// convolution then weights a value the carrier cannot hold.
    ///
    /// Note this is **not** [`write`](SampleLayout::write), which rounds half
    /// up. The two disagree on every fractional ink, and that is the whole
    /// point: vips quantises the ink one way and the resampled sample the
    /// other.
    fn cast_ink(self, v: f64) -> f64 {
        if self.is_float {
            f64::from(v as f32)
        } else {
            v.clamp(0.0, self.max).trunc()
        }
    }
}

/// The alpha ceiling the premultiply bracket divides by, the default
/// `vips_premultiply` / `vips_unpremultiply` read from
/// `vips_interpretation_max_alpha` (issue #664).
///
/// `vips_resize` premultiplies nothing of its own ("This operation does not
/// premultiply alpha. If your image has an alpha channel, you should use
/// premultiply on it first", `libvips/resample/resize.c`), and the binary
/// agrees: a float RGBA raster resized on 8.18.6 comes back byte-identical
/// under `multiband`, `b-w`, `srgb`, `scrgb` and `rgb16`. The bracket lives in
/// the callers, `vips_affine` (`affine.c:553`) and `vips_thumbnail`
/// (`thumbnail.c:835`), and both reach it through the premultiply pair. So the
/// oracle for the bracket this module wraps around its own resamplers is
/// `premultiply | resize | unpremultiply`, and it takes the interpretation.
///
/// Float carriers read [`interpretation_max_alpha`]; the unsigned ones keep
/// [`SampleLayout::max`], mirroring what #631 did for the standalone pair. On
/// an untagged raster the two answers are the same, so the only thing routing
/// the unsigned carriers through the tag would change is a raster whose tag
/// disagrees with its bytes, and
/// [`RasterCopyBuilder::interpretation`](crate::conversion::RasterCopyBuilder::interpretation)
/// accepts any tag without checking the depth, so an 8-bit buffer labelled
/// `Rgb16` would premultiply against 65535 and come back black. [`crate::composite`]
/// navigates the same trap for its own normalisation.
fn bracket_max_alpha(format: PixelFormat, interpretation: Interpretation) -> f64 {
    if format.is_float() {
        interpretation_max_alpha(interpretation)
    } else {
        SampleLayout::of(format).max
    }
}

/// The axis a one-dimensional shrink or reduce runs along.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Axis {
    Horizontal,
    Vertical,
}

// ---------------------------------------------------------------------------
// Box shrink
// ---------------------------------------------------------------------------

/// One-axis integer box shrink (`vips_shrinkh` / `vips_shrinkv`): each
/// output sample is the mean of `factor` consecutive input samples, with
/// round-half-up integer arithmetic for the unsigned formats. Blocks past
/// the edge replicate the edge sample (`VIPS_EXTEND_COPY`).
fn shrink_axis(src: &Raster, factor: u32, ceil: bool, axis: Axis) -> Result<Raster, ResampleError> {
    let op = match axis {
        Axis::Horizontal => "shrinkh",
        Axis::Vertical => "shrinkv",
    };
    if factor < 1 || f64::from(factor) > MAX_FACTOR {
        return Err(ResampleError::BadFactor {
            op,
            factor: f64::from(factor),
        });
    }
    if factor == 1 {
        return Ok(src.clone());
    }

    let (w, h) = (src.width() as usize, src.height() as usize);
    let dim = match axis {
        Axis::Horizontal => w,
        Axis::Vertical => h,
    };
    let f = factor as usize;
    let out_dim = if ceil {
        dim.div_ceil(f)
    } else {
        usize::try_from(round_uint(dim as f64 / f as f64)).unwrap_or(0)
    };
    if out_dim == 0 {
        return Err(ResampleError::ShrunkToNothing);
    }
    let (ow, oh) = match axis {
        Axis::Horizontal => (out_dim, h),
        Axis::Vertical => (w, out_dim),
    };

    let format = src.format();
    let layout = SampleLayout::of(format);
    let bands = format.channels();
    let data = src.data();
    let mut out = vec![0u8; ow * oh * format.bytes_per_pixel()];

    for oy in 0..oh {
        for ox in 0..ow {
            for band in 0..bands {
                let sample = |k: usize| -> f64 {
                    let (sx, sy) = match axis {
                        Axis::Horizontal => ((ox * f + k).min(w - 1), oy),
                        Axis::Vertical => (ox, (oy * f + k).min(h - 1)),
                    };
                    layout.read(data, (sy * w + sx) * bands + band)
                };
                let oi = (oy * ow + ox) * bands + band;
                if layout.is_float {
                    let sum: f64 = (0..f).map(sample).sum();
                    layout.write(&mut out, oi, sum / f as f64);
                } else {
                    // Integer mean with round-half-up, the libvips
                    // `(sum + hshrink / 2) / hshrink`.
                    let sum: u64 = (0..f).map(|k| sample(k) as u64).sum();
                    let mean = (sum + f as u64 / 2) / f as u64;
                    layout.write(&mut out, oi, mean as f64);
                }
            }
        }
    }

    let mut out = Raster::new(ow as u32, oh as u32, format, out)?;
    out.carry_meta_from(src);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Kernel reduce
// ---------------------------------------------------------------------------

/// One-axis kernel reduce (`vips_reduceh` / `vips_reducev`), including the
/// gap-driven integer box pre-shrink.
fn reduce_axis(
    src: &Raster,
    shrink: f64,
    kernel: ReduceKernel,
    gap: f64,
    axis: Axis,
) -> Result<Raster, ResampleError> {
    let op = match axis {
        Axis::Horizontal => "reduceh",
        Axis::Vertical => "reducev",
    };
    if !shrink.is_finite() || !(1.0..=MAX_FACTOR).contains(&shrink) {
        return Err(ResampleError::BadFactor { op, factor: shrink });
    }

    let dim = match axis {
        Axis::Horizontal => src.width() as usize,
        Axis::Vertical => src.height() as usize,
    };
    let out_dim = round_uint(dim as f64 / shrink);
    if out_dim <= 0 {
        return Err(ResampleError::ShrunkToNothing);
    }
    let out_dim = out_dim as usize;

    // How many samples we invent in the input, negative for discarding
    // (`extra_pixels` in vips_reduceh_build).
    let mut extra = out_dim as f64 * shrink - dim as f64;
    let mut shrink = shrink;

    // Alpha coverage-weighting (#288/#348) is bracketed once by the caller in
    // [`with_premultiply`], not here: an alpha image arrives already
    // premultiplied in a float working raster, so `reduce_axis` just convolves
    // its input linearly, whatever it is — straight colour for the no-alpha
    // (and Nearest single-tap) paths, premultiplied float for the bracketed
    // alpha paths. Keeping the premultiply outside means the inter-axis
    // intermediate stays premultiplied and full-precision instead of being
    // un-premultiplied and requantised to straight 8/16-bit alpha between the
    // vertical and horizontal passes (the low-alpha colour-banding of a
    // per-axis integer bracket).

    // Gap-driven integer box shrink first (`vips_shrinkh` with ceil), then
    // reduce the residual.
    let mut boxed: Option<Raster> = None;
    if gap > 0.0 && kernel != ReduceKernel::Nearest {
        if gap < 1.0 {
            return Err(ResampleError::GapTooSmall { gap });
        }
        let int_shrink = (dim as f64 / out_dim as f64 / gap).floor().max(1.0) as u32;
        if int_shrink > 1 {
            boxed = Some(shrink_axis(src, int_shrink, true, axis)?);
            shrink /= f64::from(int_shrink);
            extra /= f64::from(int_shrink);
        }
    }
    let cur = boxed.as_ref().unwrap_or(src);

    if shrink == 1.0 {
        // The integer box pre-shrink consumed the whole factor, or this is a
        // pure passthrough: no residual convolution to run.
        return Ok(cur.clone());
    }

    let n = kernel.points(shrink);
    if n > MAX_POINT {
        return Err(ResampleError::FactorTooLarge {
            n_point: n,
            max: MAX_POINT,
        });
    }
    // The embed margin, `VIPS_CEIL(n_point / 2.0) - 1`; `n` is always odd.
    let margin = (n - 1) / 2;
    // Discard invented pixels equally from both ends
    // (`hoffset` / `voffset` in the vips builds).
    let offset = (1.0 + extra) / 2.0 - 1.0;

    let (w, h) = (cur.width() as usize, cur.height() as usize);
    let cur_dim = match axis {
        Axis::Horizontal => w,
        Axis::Vertical => h,
    };
    let (ow, oh) = match axis {
        Axis::Horizontal => (out_dim, h),
        Axis::Vertical => (w, out_dim),
    };

    // Precompute the mask and first tap for every output position along the
    // axis. The masks stay in f64 where libvips carries one `double` table and
    // one `short` one, but the offset each mask is built at goes through
    // `table_offset` first: the table is not a precision detail, and evaluating
    // the kernel at the true offset is a different answer wherever that offset
    // misses the 1/64 grid (issue #668).
    let mut masks: Vec<(i64, Vec<f64>)> = Vec::with_capacity(out_dim);
    for i in 0..out_dim {
        let x = (i as f64 + 0.5) * shrink - 0.5 - offset;
        let ix = x.floor();
        let mut c = vec![0.0f64; n];
        kernel.mask(&mut c, shrink, table_offset(x));
        masks.push((ix as i64 - margin as i64, c));
    }

    let format = cur.format();
    let layout = SampleLayout::of(format);
    let bands = format.channels();
    let data = cur.data();
    let mut out = vec![0u8; ow * oh * format.bytes_per_pixel()];

    let clamp_dim = |v: i64| -> usize { v.clamp(0, cur_dim as i64 - 1) as usize };
    // Accumulate every band of a destination pixel before writing.
    let mut px = vec![0.0f64; bands];
    for oy in 0..oh {
        for ox in 0..ow {
            let (start, c) = match axis {
                Axis::Horizontal => &masks[ox],
                Axis::Vertical => &masks[oy],
            };
            for (band, p) in px.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for (k, ck) in c.iter().enumerate() {
                    let tap = clamp_dim(start + k as i64);
                    let (sx, sy) = match axis {
                        Axis::Horizontal => (tap, oy),
                        Axis::Vertical => (ox, tap),
                    };
                    acc += ck * layout.read(data, (sy * w + sx) * bands + band);
                }
                *p = acc;
            }
            let obase = (oy * ow + ox) * bands;
            for (band, &p) in px.iter().enumerate() {
                layout.write(&mut out, obase + band, p);
            }
        }
    }

    let mut out = Raster::new(ow as u32, oh as u32, format, out)?;
    out.carry_meta_from(src);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Point sampling for affine / mapim
// ---------------------------------------------------------------------------

/// Bounds-aware sample fetch applying an [`Extend`] rule to taps outside
/// the image, the equivalent of the `vips_embed` borders the libvips
/// resamplers add.
struct TapFetch<'a> {
    data: &'a [u8],
    w: i64,
    h: i64,
    bands: usize,
    layout: SampleLayout,
    /// The premultiply denominator, from [`bracket_max_alpha`]; deliberately
    /// not [`SampleLayout::max`], which stays the storage ceiling (#664).
    alpha_max: f64,
    /// The sample an [`Extend::White`] tap reads, from [`white_ink`]; also a
    /// property of the interpretation and not of the depth (#667).
    white: f64,
    extend: Extend,
    background: f64,
}

impl TapFetch<'_> {
    fn new(src: &Raster, extend: Extend, background: f64) -> TapFetch<'_> {
        let layout = SampleLayout::of(src.format());
        TapFetch {
            data: src.data(),
            w: i64::from(src.width()),
            h: i64::from(src.height()),
            bands: src.format().channels(),
            layout,
            alpha_max: bracket_max_alpha(src.format(), src.interpretation()),
            // Both inks go through the carrier once, here, exactly as
            // `vips_affine_build` runs `vips__vector_to_ink` once before it
            // embeds (issue #736). The white ink is already integral and in
            // range on every carrier, so this is a no-op for it and #667's
            // table does not move; the caller's background is the one that
            // needs it.
            white: layout.cast_ink(white_ink(src.format(), src.interpretation())),
            extend,
            background: layout.cast_ink(background),
        }
    }

    /// Fold a coordinate into `0..dim` per the extend rule; `None` paints
    /// the fill value.
    fn resolve(&self, v: i64, dim: i64) -> Option<i64> {
        if (0..dim).contains(&v) {
            return Some(v);
        }
        match self.extend {
            Extend::Copy => Some(v.clamp(0, dim - 1)),
            Extend::Repeat => Some(v.rem_euclid(dim)),
            Extend::Mirror => {
                // Reflect with the edge sample duplicated, period 2 * dim.
                let m = v.rem_euclid(2 * dim);
                Some(if m < dim { m } else { 2 * dim - 1 - m })
            }
            Extend::Black | Extend::White | Extend::Background => None,
        }
    }

    /// True when `vips_interpolate_bicubic_interpolate` would dispatch this
    /// raster to `bicubic_unsigned_int_tab` and read the 12-bit
    /// `vips_bicubic_matrixi` rather than the `double` table (issue #704).
    ///
    /// That is the `uchar` carrier and nothing else. `USHORT` and `SHORT` go to
    /// `bicubic_unsigned_int32_tab` / `bicubic_signed_int32_tab`, which take
    /// `cxf`/`cyf`; `FLOAT` goes to `bicubic_float_tab<float>`, likewise. An
    /// alpha band takes the decision away from the stored depth altogether,
    /// because `vips_affine_build` premultiplies into a FLOAT image before it
    /// resamples (`affine.c:551`), which is what `premultiply` stands for here.
    fn bicubic_is_fixed_point(&self, premultiply: bool) -> bool {
        !premultiply && !self.layout.is_float && self.layout.bpc == 1
    }

    /// True when `vips_interpolate_bicubic_interpolate` would run
    /// `bicubic_float<T>` with `T = float`, so each of the four row sums and
    /// the column combine are narrowed to `f32` on the way out of
    /// `cubic_float<T>` (issue #705).
    ///
    /// That is `VIPS_FORMAT_FLOAT` and `VIPS_FORMAT_COMPLEX`, plus anything
    /// with an alpha band, because `vips_affine_build` premultiplies into a
    /// FLOAT image before it resamples whatever the stored depth was. The
    /// 16- and 32-bit integer carriers take `bicubic_float<double>` instead
    /// and narrow nothing, and the `uchar` carrier never reaches this at all
    /// (see [`bicubic_is_fixed_point`]).
    ///
    /// [`bicubic_is_fixed_point`]: TapFetch::bicubic_is_fixed_point
    fn bicubic_narrows_rows(&self, premultiply: bool) -> bool {
        premultiply || self.layout.is_float
    }

    fn fill_value(&self) -> f64 {
        match self.extend {
            Extend::White => self.white,
            Extend::Background => self.background,
            _ => 0.0,
        }
    }

    /// Fetch the full pixel at `(x, y)` into `px`, applying the extend
    /// rule, and premultiply the colour bands when asked.
    ///
    /// The normalising alpha is clipped to `0..=max` and the alpha band itself
    /// is left raw, the `vips_premultiply` guard; [`unpremultiply`] mirrors it
    /// on the way back out (issue #604).
    fn fetch(&self, x: i64, y: i64, premultiply: bool, px: &mut [f64]) {
        match (self.resolve(x, self.w), self.resolve(y, self.h)) {
            (Some(x), Some(y)) => {
                let base = (y as usize * self.w as usize + x as usize) * self.bands;
                for (b, v) in px.iter_mut().enumerate() {
                    *v = self.layout.read(self.data, base + b);
                }
            }
            _ => px.fill(self.fill_value()),
        }
        if premultiply {
            // `OUT nalpha = (OUT) clip_alpha / max_alpha` with `OUT` = float,
            // then `q[i] = p[i] * nalpha` as a float multiply: both round to
            // `f32`, so the tap matches the FLOAT image `vips_premultiply`
            // hands `vips_affine_gen` (issue #664).
            let nalpha = (px[self.bands - 1].clamp(0.0, self.alpha_max) / self.alpha_max) as f32;
            for v in px.iter_mut().take(self.bands - 1) {
                *v = f64::from((*v as f32) * nalpha);
            }
        }
    }
}

/// Per-operation scratch for [`interpolate_at`], allocated once by the caller
/// rather than once per output pixel.
struct InterpScratch {
    /// One fetched tap, `bands` long.
    tap: Vec<f64>,
    /// The four row sums of the fixed-point bicubic path in
    /// [`INTERPOLATE_SCALE`] units, `4 * bands` long and row-major. Only that
    /// path uses it; every other kernel leaves it untouched.
    rows: Vec<i64>,
    /// The same four row sums for the floating-point bicubic path, which
    /// carries them separately because vips rounds each one to `f32` on a
    /// float carrier and that has to happen between the two stages
    /// (issue #705).
    rows_f: Vec<f64>,
}

impl InterpScratch {
    fn new(bands: usize) -> Self {
        Self {
            tap: vec![0.0f64; bands],
            rows: vec![0i64; 4 * bands],
            rows_f: vec![0.0f64; 4 * bands],
        }
    }
}

/// Interpolate every band at the continuous position `(x, y)`, writing the
/// per-band result (premultiplied when `premultiply` is set) into `out`.
fn interpolate_at(
    fetch: &TapFetch<'_>,
    interp: Interpolator,
    x: f64,
    y: f64,
    premultiply: bool,
    scratch: &mut InterpScratch,
    out: &mut [f64],
) {
    let InterpScratch { tap, rows, rows_f } = scratch;
    let px = &mut tap[..];
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    match interp {
        Interpolator::Nearest => {
            fetch.fetch(x0, y0, premultiply, out);
        }
        Interpolator::Bilinear => {
            let u = x - x0 as f64;
            let v = y - y0 as f64;
            let wx = [1.0 - u, u];
            let wy = [1.0 - v, v];
            out.fill(0.0);
            for (j, wyj) in wy.iter().enumerate() {
                for (i, wxi) in wx.iter().enumerate() {
                    let wgt = wyj * wxi;
                    if wgt == 0.0 {
                        continue;
                    }
                    fetch.fetch(x0 + i as i64, y0 + j as i64, premultiply, px);
                    for (o, p) in out.iter_mut().zip(px.iter()) {
                        *o += wgt * p;
                    }
                }
            }
        }
        Interpolator::Bicubic if fetch.bicubic_is_fixed_point(premultiply) => {
            // `vips_interpolate_bicubic_interpolate` sends a `uchar` carrier
            // to `bicubic_unsigned_int_tab`, which reads the *other* table,
            // `vips_bicubic_matrixi`: the coefficients themselves as 12-bit
            // fixed point, accumulated as integers a row at a time with
            // `unsigned_fixed_round` closing each row and the column combine
            // (issue #704). The offset is on the same 1/64 grid either way
            // (issue #668); this is the second quantisation on top of it, and
            // it is the only one that is carrier-dependent.
            let cx = fixed_catmull(table_offset(x));
            let cy = fixed_catmull(table_offset(y));
            let bands = out.len();
            rows.fill(0);
            for (j, row) in rows.chunks_exact_mut(bands).enumerate() {
                for (i, cxi) in cx.iter().enumerate() {
                    fetch.fetch(x0 - 1 + i as i64, y0 - 1 + j as i64, premultiply, px);
                    for (r, p) in row.iter_mut().zip(px.iter()) {
                        // The taps vips reads are the stored samples of the
                        // embedded image, so they are integral even at the
                        // border: `vips__vector_to_ink` casts the background
                        // through `vips_cast`, which clips and truncates
                        // (`affine.c:565`).
                        *r += cxi * (p.clamp(0.0, fetch.layout.max) as i64);
                    }
                }
            }
            for (b, o) in out.iter_mut().enumerate() {
                let acc: i64 = (0..4)
                    .map(|j| cy[j] * fixed_round(rows[j * bands + b]))
                    .sum();
                // `VIPS_CLIP(0, bicubic, max_value)` is the next line in the C
                // and it is not spelled here, because [`SampleLayout::write`]
                // already applies exactly that bound to every sample this
                // function returns and the value is integral by now, so
                // `(v + 0.5).floor().clamp(0, max)` and `VIPS_CLIP` agree.
                // Catmull-Rom rings past both ends at a hard edge, so this is
                // a live path rather than a theoretical one:
                // `affine_bicubic_clips_the_fixed_point_overshoot_like_vips`
                // pins a fixture where 15 of 36 samples land outside the
                // carrier's range.
                *o = fixed_round(acc) as f64;
            }
        }
        Interpolator::Bicubic => {
            let mut cx = [0.0f64; 4];
            let mut cy = [0.0f64; 4];
            // The bicubic tables are the reduce tables' twin, so the offset
            // is quantised the same way; bilinear and the nonlinear
            // interpolators below have no tables and keep the exact offset
            // (issue #668).
            catmull_coefficients(&mut cx, table_offset(x));
            catmull_coefficients(&mut cy, table_offset(y));
            // `bicubic_float` runs the four rows through `cubic_float<T>` and
            // then combines them through `cubic_float<T>` again, and that
            // helper *returns* `T`. With `T = float` every one of those five
            // sums is computed in `double` and narrowed to `f32` on the way
            // out; with `T = double`, which is what the 16- and 32-bit integer
            // carriers get from `bicubic_unsigned_int32_tab`, nothing narrows.
            // Reassociating alone changes no bits (measured: 0 of 1764), so
            // the narrowing is the whole of issue #705.
            let narrow = fetch.bicubic_narrows_rows(premultiply);
            let bands = out.len();
            rows_f.fill(0.0);
            for (j, row) in rows_f.chunks_exact_mut(bands).enumerate() {
                if cy[j] == 0.0 {
                    continue;
                }
                for (i, cxi) in cx.iter().enumerate() {
                    if *cxi == 0.0 {
                        continue;
                    }
                    fetch.fetch(x0 - 1 + i as i64, y0 - 1 + j as i64, premultiply, px);
                    for (r, p) in row.iter_mut().zip(px.iter()) {
                        *r += cxi * p;
                    }
                }
                if narrow {
                    for r in row.iter_mut() {
                        *r = f64::from(*r as f32);
                    }
                }
            }
            for (b, o) in out.iter_mut().enumerate() {
                // `cubic_float<T>` narrows the column combine too, and that
                // narrowing is deliberately not spelled here: on a float
                // carrier [`SampleLayout::write`] stores an `f32` anyway, and
                // on a premultiplied one [`Raster::try_affine_with`] quantises
                // the accumulator to `f32` at the un-premultiply seam (#664).
                // Mutation N5 in the PR adds it back and every test stays
                // green, which is what says the two spellings are the same
                // bits rather than an assumption that they are.
                *o = (0..4).map(|j| cy[j] * rows_f[j * bands + b]).sum();
            }
        }
        Interpolator::Lbb => {
            // LBB samples the 4x4 block at (x0-1..x0+2, y0-1..y0+2), the
            // patch corner at (x0, y0), relative offset in [0, 1]; the
            // same stencil geometry as bicubic (window_offset 1).
            let k = LbbCoeffs::new(x - x0 as f64, y - y0 as f64);
            let offsets = stencil_offsets_4x4();
            let cols = gather_stencil(fetch, x0, y0, &offsets, premultiply, px);
            for (b, o) in out.iter_mut().enumerate() {
                let mut s = [0.0f64; 16];
                for (i, si) in s.iter_mut().enumerate() {
                    *si = cols[i * fetch.bands + b];
                }
                *o = lbbicubic(&k, &s);
            }
        }
        Interpolator::Nohalo => {
            // Nohalo centres on the nearest pixel (window_offset 2, round
            // to nearest), reflects the diamond stencil so the sample sits
            // to the bottom-right of the centre, subdivides to a 4x4 LBB
            // stencil, then finishes with LBB at the reflected offset.
            let ix = (x + 0.5).floor() as i64;
            let iy = (y + 0.5).floor() as i64;
            let rel_x = x - ix as f64;
            let rel_y = y - iy as f64;
            let sx: i64 = if rel_x >= 0.0 { 1 } else { -1 };
            let sy: i64 = if rel_y >= 0.0 { 1 } else { -1 };
            // xp1over2 = 2 * |relative| in [0, 1] after the reflection.
            let k = LbbCoeffs::new(2.0 * rel_x.abs(), 2.0 * rel_y.abs());
            let offsets = nohalo_offsets(sx, sy);
            let cols = gather_stencil(fetch, ix, iy, &offsets, premultiply, px);
            for (b, o) in out.iter_mut().enumerate() {
                let mut diamond = [0.0f64; 21];
                for (i, di) in diamond.iter_mut().enumerate() {
                    *di = cols[i * fetch.bands + b];
                }
                let st = NohaloStencil::from_diamond(&diamond);
                let lbb_stencil = nohalo_subdivision(&st);
                *o = lbbicubic(&k, &lbb_stencil);
            }
        }
    }
}

/// Gather a nonlinear-interpolator stencil into a flat `taps * bands`
/// buffer (tap-major), fetching each `(dx, dy)` offset from `(cx, cy)`
/// through the [`Extend`] rule and premultiplying when asked. `px` is the
/// per-tap scratch. Allocates one small buffer per output pixel, which the
/// nonlinear resamplers need because they cannot be expressed as a fixed
/// weighted sum of taps.
fn gather_stencil(
    fetch: &TapFetch<'_>,
    cx: i64,
    cy: i64,
    offsets: &[(i64, i64)],
    premultiply: bool,
    px: &mut [f64],
) -> Vec<f64> {
    let bands = fetch.bands;
    let mut cols = vec![0.0f64; offsets.len() * bands];
    for (idx, &(dx, dy)) in offsets.iter().enumerate() {
        fetch.fetch(cx + dx, cy + dy, premultiply, px);
        cols[idx * bands..idx * bands + bands].copy_from_slice(px);
    }
    cols
}

/// The 16 `(dx, dy)` offsets of the LBB 4x4 stencil relative to the patch
/// corner `(x0, y0)`, in row-major uno/dos/tre/qua order.
fn stencil_offsets_4x4() -> [(i64, i64); 16] {
    let mut out = [(0i64, 0i64); 16];
    let mut idx = 0;
    let mut dy = -1;
    while dy <= 2 {
        let mut dx = -1;
        while dx <= 2 {
            out[idx] = (dx, dy);
            idx += 1;
            dx += 1;
        }
        dy += 1;
    }
    out
}

/// The 21 `(dx, dy)` offsets of the nohalo diamond stencil relative to the
/// centre pixel, reflected by the sample-position signs `(sx, sy)`, in the
/// order [`NohaloStencil::from_diamond`] expects.
fn nohalo_offsets(sx: i64, sy: i64) -> [(i64, i64); 21] {
    [
        (-sx, -2 * sy), // uno_two
        (0, -2 * sy),   // uno_thr
        (sx, -2 * sy),  // uno_fou
        (-2 * sx, -sy), // dos_one
        (-sx, -sy),     // dos_two
        (0, -sy),       // dos_thr
        (sx, -sy),      // dos_fou
        (2 * sx, -sy),  // dos_fiv
        (-2 * sx, 0),   // tre_one
        (-sx, 0),       // tre_two
        (0, 0),         // tre_thr
        (sx, 0),        // tre_fou
        (2 * sx, 0),    // tre_fiv
        (-2 * sx, sy),  // qua_one
        (-sx, sy),      // qua_two
        (0, sy),        // qua_thr
        (sx, sy),       // qua_fou
        (2 * sx, sy),   // qua_fiv
        (-sx, 2 * sy),  // cin_two
        (0, 2 * sy),    // cin_thr
        (sx, 2 * sy),   // cin_fou
    ]
}

/// Catmull-Rom coefficients for offset `x` in `[0, 1]`
/// (`calculate_coefficients_catmull` in `resample/templates.h`).
fn catmull_coefficients(c: &mut [f64; 4], x: f64) {
    let cr1 = 1.0 - x;
    let cr2 = -0.5 * x;
    let cr3 = cr1 * cr2;
    let cone = cr1 * cr3;
    let cfou = x * cr3;
    let cr4 = cfou - cone;
    c[0] = cone;
    c[1] = cr1 - cone + cr4;
    c[2] = x - cfou - cr4;
    c[3] = cfou;
}

// ---------------------------------------------------------------------------
// Nohalo / LBB (minmod-subdivision resamplers)
// ---------------------------------------------------------------------------
//
// A faithful port of the libvips `nohalo.cpp` and `lbb.cpp` (v8.18) nonlinear
// resamplers by N. Robidoux, C. Racette and J. Cupitt. Nohalo is level-1
// co-monotone subdivision (minmod slopes) producing a 4x4 stencil that feeds
// LBB; LBB is locally bounded bicubic, a nonlinear Hermite variant of
// Catmull-Rom whose reconstruction stays within the range of the 16 nearest
// input samples, so no output clamping is needed to avoid overshoot.

/// The sixteen LBB Hermite coefficients for a sample offset, the
/// `c00 .. c11dxdy` block shared verbatim by `nohalo.cpp` and `lbb.cpp`.
/// `xp1over2` and `yp1over2` are both in `[0, 1]`: for LBB they are the
/// relative offsets directly, for nohalo they are `2 * |relative|` after
/// the stencil reflection.
#[derive(Clone, Copy)]
struct LbbCoeffs {
    c00: f64,
    c10: f64,
    c01: f64,
    c11: f64,
    c00dx: f64,
    c10dx: f64,
    c01dx: f64,
    c11dx: f64,
    c00dy: f64,
    c10dy: f64,
    c01dy: f64,
    c11dy: f64,
    c00dxdy: f64,
    c10dxdy: f64,
    c01dxdy: f64,
    c11dxdy: f64,
}

impl LbbCoeffs {
    fn new(xp1over2: f64, yp1over2: f64) -> Self {
        let xm1over2 = xp1over2 - 1.0;
        let onepx = 0.5 + xp1over2;
        let onemx = 1.5 - xp1over2;
        let xp1over2sq = xp1over2 * xp1over2;

        let ym1over2 = yp1over2 - 1.0;
        let onepy = 0.5 + yp1over2;
        let onemy = 1.5 - yp1over2;
        let yp1over2sq = yp1over2 * yp1over2;

        let xm1over2sq = xm1over2 * xm1over2;
        let ym1over2sq = ym1over2 * ym1over2;

        let twice1px = onepx + onepx;
        let twice1py = onepy + onepy;
        let twice1mx = onemx + onemx;
        let twice1my = onemy + onemy;

        let xm1over2sq_times_ym1over2sq = xm1over2sq * ym1over2sq;
        let xp1over2sq_times_ym1over2sq = xp1over2sq * ym1over2sq;
        let xp1over2sq_times_yp1over2sq = xp1over2sq * yp1over2sq;
        let xm1over2sq_times_yp1over2sq = xm1over2sq * yp1over2sq;

        let four_times_1px_times_1py = twice1px * twice1py;
        let four_times_1mx_times_1py = twice1mx * twice1py;
        let twice_xp1over2_times_1py = xp1over2 * twice1py;
        let twice_xm1over2_times_1py = xm1over2 * twice1py;

        let twice_xm1over2_times_1my = xm1over2 * twice1my;
        let twice_xp1over2_times_1my = xp1over2 * twice1my;
        let four_times_1mx_times_1my = twice1mx * twice1my;
        let four_times_1px_times_1my = twice1px * twice1my;

        let twice_1px_times_ym1over2 = twice1px * ym1over2;
        let twice_1mx_times_ym1over2 = twice1mx * ym1over2;
        let xp1over2_times_ym1over2 = xp1over2 * ym1over2;
        let xm1over2_times_ym1over2 = xm1over2 * ym1over2;

        let xm1over2_times_yp1over2 = xm1over2 * yp1over2;
        let xp1over2_times_yp1over2 = xp1over2 * yp1over2;
        let twice_1mx_times_yp1over2 = twice1mx * yp1over2;
        let twice_1px_times_yp1over2 = twice1px * yp1over2;

        Self {
            c00: four_times_1px_times_1py * xm1over2sq_times_ym1over2sq,
            c00dx: twice_xp1over2_times_1py * xm1over2sq_times_ym1over2sq,
            c00dy: twice_1px_times_yp1over2 * xm1over2sq_times_ym1over2sq,
            c00dxdy: xp1over2_times_yp1over2 * xm1over2sq_times_ym1over2sq,

            c10: four_times_1mx_times_1py * xp1over2sq_times_ym1over2sq,
            c10dx: twice_xm1over2_times_1py * xp1over2sq_times_ym1over2sq,
            c10dy: twice_1mx_times_yp1over2 * xp1over2sq_times_ym1over2sq,
            c10dxdy: xm1over2_times_yp1over2 * xp1over2sq_times_ym1over2sq,

            c01: four_times_1px_times_1my * xm1over2sq_times_yp1over2sq,
            c01dx: twice_xp1over2_times_1my * xm1over2sq_times_yp1over2sq,
            c01dy: twice_1px_times_ym1over2 * xm1over2sq_times_yp1over2sq,
            c01dxdy: xp1over2_times_ym1over2 * xm1over2sq_times_yp1over2sq,

            c11: four_times_1mx_times_1my * xp1over2sq_times_yp1over2sq,
            c11dx: twice_xm1over2_times_1my * xp1over2sq_times_yp1over2sq,
            c11dy: twice_1mx_times_ym1over2 * xp1over2sq_times_yp1over2sq,
            c11dxdy: xm1over2_times_ym1over2 * xp1over2sq_times_yp1over2sq,
        }
    }
}

/// Minmod: the smaller (in absolute value) of two slopes when they share a
/// sign, else zero (`NOHALO_MINMOD`). `aa` is `a * a`, `ab` is `a * b`.
#[inline]
fn nohalo_minmod(a: f64, b: f64, aa: f64, ab: f64) -> f64 {
    if ab >= 0.0 {
        if aa <= ab { a } else { b }
    } else {
        0.0
    }
}

/// Locally bounded bicubic over a 4x4 stencil (`lbbicubic`, the "soft"
/// 3x3-block limiter version, the libvips default). `s` holds the sixteen
/// stencil values in row-major order (uno/dos/tre/qua rows, one/two/thr/fou
/// columns).
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
fn lbbicubic(k: &LbbCoeffs, s: &[f64; 16]) -> f64 {
    let (uno_one, uno_two, uno_thr, uno_fou) = (s[0], s[1], s[2], s[3]);
    let (dos_one, dos_two, dos_thr, dos_fou) = (s[4], s[5], s[6], s[7]);
    let (tre_one, tre_two, tre_thr, tre_fou) = (s[8], s[9], s[10], s[11]);
    let (qua_one, qua_two, qua_thr, qua_fou) = (s[12], s[13], s[14], s[15]);

    // Four min and four max over 3x3 sub-blocks of the 4x4 stencil.
    let m1 = dos_two.min(dos_thr);
    let big_m1 = dos_two.max(dos_thr);
    let m2 = tre_two.min(tre_thr);
    let big_m2 = tre_two.max(tre_thr);
    let m6 = dos_one.min(tre_one);
    let big_m6 = dos_one.max(tre_one);
    let m7 = dos_fou.min(tre_fou);
    let big_m7 = dos_fou.max(tre_fou);
    let m3 = uno_two.min(uno_thr);
    let big_m3 = uno_two.max(uno_thr);
    let m4 = qua_two.min(qua_thr);
    let big_m4 = qua_two.max(qua_thr);
    let m5 = m1.min(m2);
    let big_m5 = big_m1.max(big_m2);
    let m10 = m6.min(uno_one);
    let big_m10 = big_m6.max(uno_one);
    let m11 = m6.min(qua_one);
    let big_m11 = big_m6.max(qua_one);
    let m12 = m7.min(uno_fou);
    let big_m12 = big_m7.max(uno_fou);
    let m13 = m7.min(qua_fou);
    let big_m13 = big_m7.max(qua_fou);
    let m8 = m5.min(m3);
    let big_m8 = big_m5.max(big_m3);
    let m9 = m5.min(m4);
    let big_m9 = big_m5.max(big_m4);
    let min00 = m8.min(m10);
    let max00 = big_m8.max(big_m10);
    let min10 = m8.min(m12);
    let max10 = big_m8.max(big_m12);
    let min01 = m9.min(m11);
    let max01 = big_m9.max(big_m11);
    let min11 = m9.min(m13);
    let max11 = big_m9.max(big_m13);

    // Distances to the local min and max.
    let u00 = dos_two - min00;
    let v00 = max00 - dos_two;
    let u10 = dos_thr - min10;
    let v10 = max10 - dos_thr;
    let u01 = tre_two - min01;
    let v01 = max01 - tre_two;
    let u11 = tre_thr - min11;
    let v11 = max11 - tre_thr;

    // Centred-difference first derivatives (factors of 1/2 folded in later).
    let dble_dzdx00i = dos_thr - dos_one;
    let dble_dzdy11i = qua_thr - dos_thr;
    let dble_dzdx10i = dos_fou - dos_two;
    let dble_dzdy01i = qua_two - dos_two;
    let dble_dzdx01i = tre_thr - tre_one;
    let dble_dzdy10i = tre_thr - uno_thr;
    let dble_dzdx11i = tre_fou - tre_two;
    let dble_dzdy00i = tre_two - uno_two;

    let sign_dzdx00 = if dble_dzdx00i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdx10 = if dble_dzdx10i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdx01 = if dble_dzdx01i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdx11 = if dble_dzdx11i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdy00 = if dble_dzdy00i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdy10 = if dble_dzdy10i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdy01 = if dble_dzdy01i >= 0.0 { 1.0 } else { -1.0 };
    let sign_dzdy11 = if dble_dzdy11i >= 0.0 { 1.0 } else { -1.0 };

    // Centred-difference cross derivatives (factors of 1/4 folded in later).
    let quad_d2zdxdy00i = uno_one - uno_thr + dble_dzdx01i;
    let quad_d2zdxdy10i = uno_two - uno_fou + dble_dzdx11i;
    let quad_d2zdxdy01i = qua_thr - qua_one - dble_dzdx00i;
    let quad_d2zdxdy11i = qua_fou - qua_two - dble_dzdx10i;

    // Slope limiters (key multiplier 3, folded with a factor of 2).
    let dble_slopelimit_00 = 6.0 * u00.min(v00);
    let dble_slopelimit_10 = 6.0 * u10.min(v10);
    let dble_slopelimit_01 = 6.0 * u01.min(v01);
    let dble_slopelimit_11 = 6.0 * u11.min(v11);

    let clamp_slope = |sign: f64, deriv: f64, limit: f64| -> f64 {
        if sign * deriv <= limit {
            deriv
        } else {
            sign * limit
        }
    };
    let dble_dzdx00 = clamp_slope(sign_dzdx00, dble_dzdx00i, dble_slopelimit_00);
    let dble_dzdy00 = clamp_slope(sign_dzdy00, dble_dzdy00i, dble_slopelimit_00);
    let dble_dzdx10 = clamp_slope(sign_dzdx10, dble_dzdx10i, dble_slopelimit_10);
    let dble_dzdy10 = clamp_slope(sign_dzdy10, dble_dzdy10i, dble_slopelimit_10);
    let dble_dzdx01 = clamp_slope(sign_dzdx01, dble_dzdx01i, dble_slopelimit_01);
    let dble_dzdy01 = clamp_slope(sign_dzdy01, dble_dzdy01i, dble_slopelimit_01);
    let dble_dzdx11 = clamp_slope(sign_dzdx11, dble_dzdx11i, dble_slopelimit_11);
    let dble_dzdy11 = clamp_slope(sign_dzdy11, dble_dzdy11i, dble_slopelimit_11);

    // Sums and differences of first derivatives.
    let twelve_sum00 = 6.0 * (dble_dzdx00 + dble_dzdy00);
    let twelve_dif00 = 6.0 * (dble_dzdx00 - dble_dzdy00);
    let twelve_sum10 = 6.0 * (dble_dzdx10 + dble_dzdy10);
    let twelve_dif10 = 6.0 * (dble_dzdx10 - dble_dzdy10);
    let twelve_sum01 = 6.0 * (dble_dzdx01 + dble_dzdy01);
    let twelve_dif01 = 6.0 * (dble_dzdx01 - dble_dzdy01);
    let twelve_sum11 = 6.0 * (dble_dzdx11 + dble_dzdy11);
    let twelve_dif11 = 6.0 * (dble_dzdx11 - dble_dzdy11);

    let twelve_abs_sum00 = twelve_sum00.abs();
    let twelve_abs_sum10 = twelve_sum10.abs();
    let twelve_abs_sum01 = twelve_sum01.abs();
    let twelve_abs_sum11 = twelve_sum11.abs();

    let u00_times_36 = 36.0 * u00;
    let u10_times_36 = 36.0 * u10;
    let u01_times_36 = 36.0 * u01;
    let u11_times_36 = 36.0 * u11;

    let first_limit00 = twelve_abs_sum00 - u00_times_36;
    let first_limit10 = twelve_abs_sum10 - u10_times_36;
    let first_limit01 = twelve_abs_sum01 - u01_times_36;
    let first_limit11 = twelve_abs_sum11 - u11_times_36;

    let quad_d2zdxdy00ii = quad_d2zdxdy00i.max(first_limit00);
    let quad_d2zdxdy10ii = quad_d2zdxdy10i.max(first_limit10);
    let quad_d2zdxdy01ii = quad_d2zdxdy01i.max(first_limit01);
    let quad_d2zdxdy11ii = quad_d2zdxdy11i.max(first_limit11);

    let v00_times_36 = 36.0 * v00;
    let v10_times_36 = 36.0 * v10;
    let v01_times_36 = 36.0 * v01;
    let v11_times_36 = 36.0 * v11;

    let second_limit00 = v00_times_36 - twelve_abs_sum00;
    let second_limit10 = v10_times_36 - twelve_abs_sum10;
    let second_limit01 = v01_times_36 - twelve_abs_sum01;
    let second_limit11 = v11_times_36 - twelve_abs_sum11;

    let quad_d2zdxdy00iii = quad_d2zdxdy00ii.min(second_limit00);
    let quad_d2zdxdy10iii = quad_d2zdxdy10ii.min(second_limit10);
    let quad_d2zdxdy01iii = quad_d2zdxdy01ii.min(second_limit01);
    let quad_d2zdxdy11iii = quad_d2zdxdy11ii.min(second_limit11);

    let twelve_abs_dif00 = twelve_dif00.abs();
    let twelve_abs_dif10 = twelve_dif10.abs();
    let twelve_abs_dif01 = twelve_dif01.abs();
    let twelve_abs_dif11 = twelve_dif11.abs();

    let third_limit00 = twelve_abs_dif00 - v00_times_36;
    let third_limit10 = twelve_abs_dif10 - v10_times_36;
    let third_limit01 = twelve_abs_dif01 - v01_times_36;
    let third_limit11 = twelve_abs_dif11 - v11_times_36;

    let quad_d2zdxdy00iiii = quad_d2zdxdy00iii.max(third_limit00);
    let quad_d2zdxdy10iiii = quad_d2zdxdy10iii.max(third_limit10);
    let quad_d2zdxdy01iiii = quad_d2zdxdy01iii.max(third_limit01);
    let quad_d2zdxdy11iiii = quad_d2zdxdy11iii.max(third_limit11);

    let fourth_limit00 = u00_times_36 - twelve_abs_dif00;
    let fourth_limit10 = u10_times_36 - twelve_abs_dif10;
    let fourth_limit01 = u01_times_36 - twelve_abs_dif01;
    let fourth_limit11 = u11_times_36 - twelve_abs_dif11;

    let quad_d2zdxdy00 = quad_d2zdxdy00iiii.min(fourth_limit00);
    let quad_d2zdxdy10 = quad_d2zdxdy10iiii.min(fourth_limit10);
    let quad_d2zdxdy01 = quad_d2zdxdy01iiii.min(fourth_limit01);
    let quad_d2zdxdy11 = quad_d2zdxdy11iiii.min(fourth_limit11);

    let newval1 = k.c00 * dos_two + k.c10 * dos_thr + k.c01 * tre_two + k.c11 * tre_thr;
    let newval2 = k.c00dx * dble_dzdx00
        + k.c10dx * dble_dzdx10
        + k.c01dx * dble_dzdx01
        + k.c11dx * dble_dzdx11
        + k.c00dy * dble_dzdy00
        + k.c10dy * dble_dzdy10
        + k.c01dy * dble_dzdy01
        + k.c11dy * dble_dzdy11;
    let newval3 = k.c00dxdy * quad_d2zdxdy00
        + k.c10dxdy * quad_d2zdxdy10
        + k.c01dxdy * quad_d2zdxdy01
        + k.c11dxdy * quad_d2zdxdy11;

    // `dble_dzdy11i` participates only in the reference implementation's
    // symmetry; it is unused in the final combination, kept above for a
    // line-by-line correspondence with the C source.
    let _ = dble_dzdy11i;

    newval1 + 0.5 * newval2 + 0.25 * newval3
}

/// Nohalo level-1 subdivision (`nohalo_subdivision`): from the 21-point
/// diamond stencil, compute the twelve new half-density values and return
/// the sixteen LBB stencil values in row-major order. `st` holds the input
/// stencil already reflected so the sample sits to the bottom-right of the
/// centre (`tre_thr`); see [`gather_nohalo_stencil`].
#[allow(clippy::many_single_char_names)]
fn nohalo_subdivision(st: &NohaloStencil) -> [f64; 16] {
    let NohaloStencil {
        uno_two,
        uno_thr,
        uno_fou,
        dos_one,
        dos_two,
        dos_thr,
        dos_fou,
        dos_fiv,
        tre_one,
        tre_two,
        tre_thr,
        tre_fou,
        tre_fiv,
        qua_one,
        qua_two,
        qua_thr,
        qua_fou,
        qua_fiv,
        cin_two,
        cin_thr,
        cin_fou,
    } = *st;

    // Vertical simple differences.
    let d_unodos_two = dos_two - uno_two;
    let d_dostre_two = tre_two - dos_two;
    let d_trequa_two = qua_two - tre_two;
    let d_quacin_two = cin_two - qua_two;
    let d_unodos_thr = dos_thr - uno_thr;
    let d_dostre_thr = tre_thr - dos_thr;
    let d_trequa_thr = qua_thr - tre_thr;
    let d_quacin_thr = cin_thr - qua_thr;
    let d_unodos_fou = dos_fou - uno_fou;
    let d_dostre_fou = tre_fou - dos_fou;
    let d_trequa_fou = qua_fou - tre_fou;
    let d_quacin_fou = cin_fou - qua_fou;
    // Horizontal simple differences.
    let d_dos_onetwo = dos_two - dos_one;
    let d_dos_twothr = dos_thr - dos_two;
    let d_dos_thrfou = dos_fou - dos_thr;
    let d_dos_foufiv = dos_fiv - dos_fou;
    let d_tre_onetwo = tre_two - tre_one;
    let d_tre_twothr = tre_thr - tre_two;
    let d_tre_thrfou = tre_fou - tre_thr;
    let d_tre_foufiv = tre_fiv - tre_fou;
    let d_qua_onetwo = qua_two - qua_one;
    let d_qua_twothr = qua_thr - qua_two;
    let d_qua_thrfou = qua_fou - qua_thr;
    let d_qua_foufiv = qua_fiv - qua_fou;

    // Recyclable vertical products and squares.
    let d_unodos_times_dostre_two = d_unodos_two * d_dostre_two;
    let d_dostre_two_sq = d_dostre_two * d_dostre_two;
    let d_dostre_times_trequa_two = d_dostre_two * d_trequa_two;
    let d_trequa_times_quacin_two = d_quacin_two * d_trequa_two;
    let d_quacin_two_sq = d_quacin_two * d_quacin_two;

    let d_unodos_times_dostre_thr = d_unodos_thr * d_dostre_thr;
    let d_dostre_thr_sq = d_dostre_thr * d_dostre_thr;
    let d_dostre_times_trequa_thr = d_trequa_thr * d_dostre_thr;
    let d_trequa_times_quacin_thr = d_trequa_thr * d_quacin_thr;
    let d_quacin_thr_sq = d_quacin_thr * d_quacin_thr;

    let d_unodos_times_dostre_fou = d_unodos_fou * d_dostre_fou;
    let d_dostre_fou_sq = d_dostre_fou * d_dostre_fou;
    let d_dostre_times_trequa_fou = d_trequa_fou * d_dostre_fou;
    let d_trequa_times_quacin_fou = d_trequa_fou * d_quacin_fou;
    let d_quacin_fou_sq = d_quacin_fou * d_quacin_fou;
    // Recyclable horizontal products and squares.
    let d_dos_onetwo_times_twothr = d_dos_onetwo * d_dos_twothr;
    let d_dos_twothr_sq = d_dos_twothr * d_dos_twothr;
    let d_dos_twothr_times_thrfou = d_dos_twothr * d_dos_thrfou;
    let d_dos_thrfou_times_foufiv = d_dos_thrfou * d_dos_foufiv;
    let d_dos_foufiv_sq = d_dos_foufiv * d_dos_foufiv;

    let d_tre_onetwo_times_twothr = d_tre_onetwo * d_tre_twothr;
    let d_tre_twothr_sq = d_tre_twothr * d_tre_twothr;
    let d_tre_twothr_times_thrfou = d_tre_thrfou * d_tre_twothr;
    let d_tre_thrfou_times_foufiv = d_tre_thrfou * d_tre_foufiv;
    let d_tre_foufiv_sq = d_tre_foufiv * d_tre_foufiv;

    let d_qua_onetwo_times_twothr = d_qua_onetwo * d_qua_twothr;
    let d_qua_twothr_sq = d_qua_twothr * d_qua_twothr;
    let d_qua_twothr_times_thrfou = d_qua_thrfou * d_qua_twothr;
    let d_qua_thrfou_times_foufiv = d_qua_thrfou * d_qua_foufiv;
    let d_qua_foufiv_sq = d_qua_foufiv * d_qua_foufiv;

    // Minmod slopes and first-level pixel values.
    let dos_thr_y = nohalo_minmod(
        d_dostre_thr,
        d_unodos_thr,
        d_dostre_thr_sq,
        d_unodos_times_dostre_thr,
    );
    let tre_thr_y = nohalo_minmod(
        d_dostre_thr,
        d_trequa_thr,
        d_dostre_thr_sq,
        d_dostre_times_trequa_thr,
    );
    let newval_uno_two = 0.5 * (dos_thr + tre_thr) + 0.25 * (dos_thr_y - tre_thr_y);

    let qua_thr_y = nohalo_minmod(
        d_quacin_thr,
        d_trequa_thr,
        d_quacin_thr_sq,
        d_trequa_times_quacin_thr,
    );
    let newval_tre_two = 0.5 * (tre_thr + qua_thr) + 0.25 * (tre_thr_y - qua_thr_y);

    let tre_fou_y = nohalo_minmod(
        d_dostre_fou,
        d_trequa_fou,
        d_dostre_fou_sq,
        d_dostre_times_trequa_fou,
    );
    let qua_fou_y = nohalo_minmod(
        d_quacin_fou,
        d_trequa_fou,
        d_quacin_fou_sq,
        d_trequa_times_quacin_fou,
    );
    let newval_tre_fou = 0.5 * (tre_fou + qua_fou) + 0.25 * (tre_fou_y - qua_fou_y);

    let dos_fou_y = nohalo_minmod(
        d_dostre_fou,
        d_unodos_fou,
        d_dostre_fou_sq,
        d_unodos_times_dostre_fou,
    );
    let newval_uno_fou = 0.5 * (dos_fou + tre_fou) + 0.25 * (dos_fou_y - tre_fou_y);

    let tre_two_x = nohalo_minmod(
        d_tre_twothr,
        d_tre_onetwo,
        d_tre_twothr_sq,
        d_tre_onetwo_times_twothr,
    );
    let tre_thr_x = nohalo_minmod(
        d_tre_twothr,
        d_tre_thrfou,
        d_tre_twothr_sq,
        d_tre_twothr_times_thrfou,
    );
    let newval_dos_one = 0.5 * (tre_two + tre_thr) + 0.25 * (tre_two_x - tre_thr_x);

    let tre_fou_x = nohalo_minmod(
        d_tre_foufiv,
        d_tre_thrfou,
        d_tre_foufiv_sq,
        d_tre_thrfou_times_foufiv,
    );
    let tre_thr_x_minus_tre_fou_x = tre_thr_x - tre_fou_x;
    let newval_dos_thr = 0.5 * (tre_thr + tre_fou) + 0.25 * tre_thr_x_minus_tre_fou_x;

    let qua_thr_x = nohalo_minmod(
        d_qua_twothr,
        d_qua_thrfou,
        d_qua_twothr_sq,
        d_qua_twothr_times_thrfou,
    );
    let qua_fou_x = nohalo_minmod(
        d_qua_foufiv,
        d_qua_thrfou,
        d_qua_foufiv_sq,
        d_qua_thrfou_times_foufiv,
    );
    let qua_thr_x_minus_qua_fou_x = qua_thr_x - qua_fou_x;
    let newval_qua_thr = 0.5 * (qua_thr + qua_fou) + 0.25 * qua_thr_x_minus_qua_fou_x;

    let qua_two_x = nohalo_minmod(
        d_qua_twothr,
        d_qua_onetwo,
        d_qua_twothr_sq,
        d_qua_onetwo_times_twothr,
    );
    let newval_qua_one = 0.5 * (qua_two + qua_thr) + 0.25 * (qua_two_x - qua_thr_x);

    let newval_tre_thr = 0.125 * (tre_thr_x_minus_tre_fou_x + qua_thr_x_minus_qua_fou_x)
        + 0.5 * (newval_tre_two + newval_tre_fou);

    let dos_thr_x = nohalo_minmod(
        d_dos_twothr,
        d_dos_thrfou,
        d_dos_twothr_sq,
        d_dos_twothr_times_thrfou,
    );
    let dos_fou_x = nohalo_minmod(
        d_dos_foufiv,
        d_dos_thrfou,
        d_dos_foufiv_sq,
        d_dos_thrfou_times_foufiv,
    );
    let newval_uno_thr = 0.25 * (dos_fou - tre_thr)
        + 0.125 * (dos_fou_y - tre_fou_y + dos_thr_x - dos_fou_x)
        + 0.5 * (newval_uno_two + newval_dos_thr);

    let tre_two_y = nohalo_minmod(
        d_dostre_two,
        d_trequa_two,
        d_dostre_two_sq,
        d_dostre_times_trequa_two,
    );
    let qua_two_y = nohalo_minmod(
        d_quacin_two,
        d_trequa_two,
        d_quacin_two_sq,
        d_trequa_times_quacin_two,
    );
    let newval_tre_one = 0.25 * (qua_two - tre_thr)
        + 0.125 * (qua_two_x - qua_thr_x + tre_two_y - qua_two_y)
        + 0.5 * (newval_dos_one + newval_tre_two);

    let dos_two_x = nohalo_minmod(
        d_dos_twothr,
        d_dos_onetwo,
        d_dos_twothr_sq,
        d_dos_onetwo_times_twothr,
    );
    let dos_two_y = nohalo_minmod(
        d_dostre_two,
        d_unodos_two,
        d_dostre_two_sq,
        d_unodos_times_dostre_two,
    );
    let newval_uno_one = 0.25 * (dos_two + dos_thr + tre_two + tre_thr)
        + 0.125
            * (dos_two_x - dos_thr_x + tre_two_x - tre_thr_x + dos_two_y + dos_thr_y
                - tre_two_y
                - tre_thr_y);

    [
        newval_uno_one,
        newval_uno_two,
        newval_uno_thr,
        newval_uno_fou,
        newval_dos_one,
        tre_thr,
        newval_dos_thr,
        tre_fou,
        newval_tre_one,
        newval_tre_two,
        newval_tre_thr,
        newval_tre_fou,
        newval_qua_one,
        qua_thr,
        newval_qua_thr,
        qua_fou,
    ]
}

/// The 21-point nohalo input stencil, already reflected so the sampling
/// point lies to the bottom-right of `tre_thr`.
#[derive(Clone, Copy)]
struct NohaloStencil {
    uno_two: f64,
    uno_thr: f64,
    uno_fou: f64,
    dos_one: f64,
    dos_two: f64,
    dos_thr: f64,
    dos_fou: f64,
    dos_fiv: f64,
    tre_one: f64,
    tre_two: f64,
    tre_thr: f64,
    tre_fou: f64,
    tre_fiv: f64,
    qua_one: f64,
    qua_two: f64,
    qua_thr: f64,
    qua_fou: f64,
    qua_fiv: f64,
    cin_two: f64,
    cin_thr: f64,
    cin_fou: f64,
}

impl NohaloStencil {
    /// Build the stencil from the 21 diamond taps in [`nohalo_offsets`]
    /// order.
    fn from_diamond(d: &[f64; 21]) -> Self {
        Self {
            uno_two: d[0],
            uno_thr: d[1],
            uno_fou: d[2],
            dos_one: d[3],
            dos_two: d[4],
            dos_thr: d[5],
            dos_fou: d[6],
            dos_fiv: d[7],
            tre_one: d[8],
            tre_two: d[9],
            tre_thr: d[10],
            tre_fou: d[11],
            tre_fiv: d[12],
            qua_one: d[13],
            qua_two: d[14],
            qua_thr: d[15],
            qua_fou: d[16],
            qua_fiv: d[17],
            cin_two: d[18],
            cin_thr: d[19],
            cin_fou: d[20],
        }
    }
}

/// Unpremultiply an interpolated pixel in place (`vips_unpremultiply`):
/// colour bands scale by [`unpremultiply_factor`] and the stored alpha clips
/// to `0..=max`.
///
/// These are two separate guards and libvips keeps them separate on purpose
/// (issue #604). The factor divides by the **raw** alpha, so an alpha
/// overshoot and the colour overshoot that came with it cancel — "Don't use
/// clip_alpha to calculate factor: we want over and undershoots on alpha and
/// RGB to cancel", `libvips/conversion/unpremultiply.c` — while the alpha that
/// is **stored** is clipped, `VIPS_CLIP(0, alpha, max_alpha)`. Applying the
/// clip to the factor as well would quietly discard the cancellation; applying
/// neither leaves a near-zero alpha to amplify the colour by `max / alpha`.
///
/// This is the float side of the pair, so the dead zone is live here: every
/// caller runs on the [`premultiply_to_float`] working raster or on an
/// interpolated accumulator, where a lanczos undershoot at a hard transparency
/// edge routinely lands alpha in `(0, 0.01)` or just below zero.
fn unpremultiply(px: &mut [f64], max: f64) {
    let bands = px.len();
    let alpha = px[bands - 1];
    // `OUT factor` is a `float` and `q[i] = p[i] * factor` a float multiply,
    // so both round to `f32`, mirroring [`TapFetch::fetch`] (issue #664).
    let factor = unpremultiply_factor(alpha, max) as f32;
    for v in px.iter_mut().take(bands - 1) {
        *v = f64::from((*v as f32) * factor);
    }
    px[bands - 1] = alpha.clamp(0.0, max);
}

/// Premultiply the colour bands of an alpha raster by `alpha / max` into a
/// four-band **float** working raster (`RgbaF32`); the alpha band is copied
/// unchanged. `max` is the alpha ceiling from [`bracket_max_alpha`], which
/// reads the interpretation on a float carrier and the depth on an unsigned
/// one (issue #664).
///
/// Premultiplying into float — the way `vips_resize` premultiplies once into a
/// float buffer — is what lets [`with_premultiply`] bracket the whole separable
/// pipeline (see there): colour is averaged weighted by coverage, the same
/// thing the affine path does per tap via [`TapFetch::fetch`], so the
/// meaningless RGB of transparent pixels cannot bleed into opaque neighbours
/// (dark fringes at transparency boundaries). Unlike a same-bit-depth integer
/// intermediate, the float buffer does not requantise `round(c * a / max)` to a
/// couple of bits for near-transparent pixels — quantisation that
/// un-premultiply would then amplify by `max / a` into visible colour banding.
///
/// The normalising factor is built from a **clipped** alpha and the alpha band
/// is copied through **raw**, which is `vips_premultiply`'s side of the guard
/// pair and the exact mirror of [`unpremultiply`]'s (issue #604): there the
/// factor takes the raw alpha and the stored alpha is clipped. Keeping the two
/// mirrored is what makes the bracket cancel. There is no dead zone on this
/// side, and that is deliberate rather than missing — premultiply multiplies by
/// the alpha, so a near-zero one damps rather than amplifies and no division
/// can blow up. `libvips/conversion/premultiply.c` has a single macro for every
/// band format, with no float variant.
///
/// The arithmetic rounds **twice**, which is not an accident of this port but
/// what the C macro does: `OUT nalpha = (OUT) clip_alpha / max_alpha`, and
/// `OUT` is `float` for every carrier this crate has. `vips_premultiply`
/// widens only a DOUBLE input to DOUBLE output (`premultiply.c:229-232`) and
/// writes FLOAT for everything else, so the multiplier is quantised to `f32`
/// before the colour multiply even when the input is 8-bit, and an `f64`
/// expression rounded once at the store is a different number. Measured on
/// 8.18.6: `100 * f32(0.5 / 255)` un-premultiplied by `f32(255 / 0.5)` comes
/// back `100.00000762939453`, where the single-rounded form gives exactly
/// `100`. Same shape as the finding in #631.
fn premultiply_to_float(src: &Raster, max: f64) -> Result<Raster, ResampleError> {
    let in_layout = SampleLayout::of(src.format());
    let out_fmt = PixelFormat::RgbaF32;
    let out_layout = SampleLayout::of(out_fmt);
    let bands = src.format().channels();
    let data = src.data();
    let count = src.width() as usize * src.height() as usize;
    let mut out = vec![0u8; count * out_fmt.bytes_per_pixel()];
    for p in 0..count {
        let base = p * bands;
        let alpha = in_layout.read(data, base + bands - 1);
        // `OUT nalpha = (OUT) clip_alpha / max_alpha` with `OUT` = float, so
        // the multiplier is rounded to `f32` *before* the colour multiply and
        // the whole thing rounds twice; see the note above.
        let nalpha = (alpha.clamp(0.0, max) / max) as f32;
        for b in 0..bands - 1 {
            let v = in_layout.read(data, base + b) as f32;
            out_layout.write(&mut out, base + b, f64::from(v * nalpha));
        }
        out_layout.write(&mut out, base + bands - 1, alpha);
    }
    let mut out = Raster::new(src.width(), src.height(), out_fmt, out)?;
    out.carry_meta_from(src);
    Ok(out)
}

/// Un-premultiply the float working raster produced by [`premultiply_to_float`]
/// back into `dst_fmt` (the original source format), dividing each colour band
/// by the alpha and requantising exactly once. `max` is the same ceiling the
/// premultiply used, so the round-trip cancels.
///
/// The guard pair is [`unpremultiply`]'s: the factor takes the **raw** alpha so
/// over- and undershoots cancel, and the alpha that is **stored** is clipped to
/// `0..=max`. That clip is where the ceiling shows up on ordinary data rather
/// than only on out-of-range alpha, because lanczos3 rings: resampling a hard
/// transparency edge pushes the alpha above the source's maximum, and an scRGB
/// raster clips it back to `1.0` where the 255 default leaves it (issue #664).
///
/// Rounds through `f32` for the same reason [`premultiply_to_float`] does:
/// `OUT factor` is a `float` in `unpremultiply.c`, so the reciprocal is
/// quantised before the colour multiply.
fn unpremultiply_from_float(
    src: &Raster,
    dst_fmt: PixelFormat,
    max: f64,
) -> Result<Raster, ResampleError> {
    let in_layout = SampleLayout::of(src.format());
    let out_layout = SampleLayout::of(dst_fmt);
    let bands = dst_fmt.channels();
    let data = src.data();
    let count = src.width() as usize * src.height() as usize;
    let mut out = vec![0u8; count * dst_fmt.bytes_per_pixel()];
    for p in 0..count {
        let base = p * bands;
        let alpha = in_layout.read(data, base + bands - 1);
        // The mirror of [`premultiply_to_float`]'s rounding: `OUT factor` is a
        // `float` too, so the reciprocal lands in `f32` before the colour
        // multiply. [`unpremultiply`] keeps the `f64` spelling for the affine
        // accumulator, which is not a stored `f32` buffer.
        let factor = unpremultiply_factor(alpha, max) as f32;
        for b in 0..bands - 1 {
            let v = in_layout.read(data, base + b) as f32;
            out_layout.write(&mut out, base + b, f64::from(v * factor));
        }
        out_layout.write(&mut out, base + bands - 1, alpha.clamp(0.0, max));
    }
    let mut out = Raster::new(src.width(), src.height(), dst_fmt, out)?;
    out.carry_meta_from(src);
    Ok(out)
}

/// Bracket the alpha premultiply exactly once around a separable resample
/// `pipeline`, mirroring how `vips_resize` premultiplies before its reduce and
/// affine passes and un-premultiplies after.
///
/// When `bracket` is set and the source carries an alpha band, the source is
/// premultiplied into a float working raster, `pipeline` runs entirely in
/// premultiplied float space — so the intermediate between the vertical and
/// horizontal passes stays full-precision, with no per-axis requantisation of
/// low-alpha colour and no straight/premultiplied round-trip between axes — and
/// the result is un-premultiplied back to the source format once. Otherwise
/// `pipeline` runs directly on the source: no-alpha images need no coverage
/// weighting, and the Nearest single-tap kernel must stay an exact pick, since
/// bracketing would only requantise its semi-transparent RGB (#287/#288).
///
/// Callers pass `bracket = false` for the pure-passthrough cases (an identity
/// factor that does no work) so a no-op never pays for a premultiply /
/// un-premultiply round-trip.
fn with_premultiply<F>(src: &Raster, bracket: bool, pipeline: F) -> Result<Raster, ResampleError>
where
    F: FnOnce(&Raster) -> Result<Raster, ResampleError>,
{
    if !bracket || !src.format().has_alpha() {
        return pipeline(src);
    }
    let max = bracket_max_alpha(src.format(), src.interpretation());
    let work = premultiply_to_float(src, max)?;
    let reduced = pipeline(&work)?;
    unpremultiply_from_float(&reduced, src.format(), max)
}

// ---------------------------------------------------------------------------
// Nearest-kernel resize helpers
// ---------------------------------------------------------------------------

/// Point-sample every `xfac`-th / `yfac`-th pixel (`vips_subsample`).
/// Output dimensions truncate, matching the libvips integer division.
fn subsample(src: &Raster, xfac: u32, yfac: u32) -> Result<Raster, ResampleError> {
    let ow = src.width() / xfac;
    let oh = src.height() / yfac;
    if ow == 0 || oh == 0 {
        return Err(ResampleError::ShrunkToNothing);
    }
    let format = src.format();
    let bpp = format.bytes_per_pixel();
    let w = src.width() as usize;
    let data = src.data();
    let mut out = vec![0u8; ow as usize * oh as usize * bpp];
    for oy in 0..oh as usize {
        for ox in 0..ow as usize {
            let src_off = (oy * yfac as usize * w + ox * xfac as usize) * bpp;
            let dst_off = (oy * ow as usize + ox) * bpp;
            out[dst_off..dst_off + bpp].copy_from_slice(&data[src_off..src_off + bpp]);
        }
    }
    let mut out = Raster::new(ow, oh, format, out)?;
    out.carry_meta_from(src);
    Ok(out)
}

/// Integral pixel replication (`vips_zoom`).
fn zoom(src: &Raster, xfac: u32, yfac: u32) -> Result<Raster, ResampleError> {
    let ow = u64::from(src.width()) * u64::from(xfac);
    let oh = u64::from(src.height()) * u64::from(yfac);
    let (Ok(ow), Ok(oh)) = (u32::try_from(ow), u32::try_from(oh)) else {
        return Err(ResampleError::BadOutputArea {
            width: (u64::from(src.width()) * u64::from(xfac)) as i64,
            height: (u64::from(src.height()) * u64::from(yfac)) as i64,
        });
    };
    let format = src.format();
    let bpp = format.bytes_per_pixel();
    let w = src.width() as usize;
    let data = src.data();
    let mut out = Raster::zeroed(ow, oh, format)?;
    let ow = ow as usize;
    let buf = out.data_mut();
    for oy in 0..oh as usize {
        let sy = oy / yfac as usize;
        for ox in 0..ow {
            let sx = ox / xfac as usize;
            let src_off = (sy * w + sx) * bpp;
            let dst_off = (oy * ow + ox) * bpp;
            buf[dst_off..dst_off + bpp].copy_from_slice(&data[src_off..src_off + bpp]);
        }
    }
    out.carry_meta_from(src);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Raster methods
// ---------------------------------------------------------------------------

impl Raster {
    /// Fallible form of [`Raster::shrink`].
    ///
    /// # Errors
    ///
    /// [`ResampleError::BadFactor`] unless both factors are finite numbers
    /// in `1.0..=1_000_000.0`, [`ResampleError::ShrunkToNothing`] when a
    /// dimension would reach zero, or [`ResampleError::Raster`] on
    /// allocation failure.
    pub fn try_shrink(&self, hshrink: f64, vshrink: f64) -> Result<Raster, ResampleError> {
        for factor in [hshrink, vshrink] {
            if !factor.is_finite() || !(1.0..=MAX_FACTOR).contains(&factor) {
                return Err(ResampleError::BadFactor {
                    op: "shrink",
                    factor,
                });
            }
        }
        // Alpha is coverage-weighted once around both axes (#348): premultiply
        // into float, run the separable box / kernel passes, un-premultiply.
        if hshrink.fract() != 0.0 || vshrink.fract() != 0.0 {
            // Fractional factors delegate to reduce with the default
            // lanczos3 kernel and gap 1 (`vips_shrink_build`).
            with_premultiply(self, true, |w| {
                let t = reduce_axis(w, vshrink, ReduceKernel::Lanczos3, 1.0, Axis::Vertical)?;
                reduce_axis(&t, hshrink, ReduceKernel::Lanczos3, 1.0, Axis::Horizontal)
            })
        } else {
            // Integer factors run the plain box average on both axes; bracket
            // them the same way so integer and fractional shrink handle alpha
            // consistently (no factor-dependent bleed).
            with_premultiply(self, hshrink > 1.0 || vshrink > 1.0, |w| {
                let t = shrink_axis(w, vshrink as u32, false, Axis::Vertical)?;
                shrink_axis(&t, hshrink as u32, false, Axis::Horizontal)
            })
        }
    }

    /// Shrink by a pair of factors with a box filter (libvips
    /// `vips_shrink`). Integer factors run the plain box average; for
    /// fractional factors the residual is reduced with the default
    /// `lanczos3` kernel, exactly as libvips composes it. Output dimensions
    /// are `round(dim / factor)`. Panicking form of [`Raster::try_shrink`],
    /// matching the ported-test call surface.
    ///
    /// Alpha images are premultiplied around the whole shrink (both the integer
    /// and fractional paths) so transparent colour cannot bleed into opaque
    /// neighbours; see the module-level *Premultiplied alpha* note. This
    /// diverges from bare `vips_shrink`, which does not premultiply.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_shrink`].
    #[track_caller]
    pub fn shrink(&self, hshrink: f64, vshrink: f64) -> Raster {
        expect_resample("shrink", self.try_shrink(hshrink, vshrink))
    }

    /// Fallible form of [`Raster::shrinkh`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_shrink`].
    pub fn try_shrinkh(&self, hshrink: u32) -> Result<Raster, ResampleError> {
        with_premultiply(self, hshrink > 1, |w| {
            shrink_axis(w, hshrink, false, Axis::Horizontal)
        })
    }

    /// Horizontal integer box shrink (libvips `vips_shrinkh`); the output
    /// width is `round(width / hshrink)`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_shrinkh`].
    #[track_caller]
    pub fn shrinkh(&self, hshrink: u32) -> Raster {
        expect_resample("shrinkh", self.try_shrinkh(hshrink))
    }

    /// Fallible form of [`Raster::shrinkv`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_shrink`].
    pub fn try_shrinkv(&self, vshrink: u32) -> Result<Raster, ResampleError> {
        with_premultiply(self, vshrink > 1, |w| {
            shrink_axis(w, vshrink, false, Axis::Vertical)
        })
    }

    /// Vertical integer box shrink (libvips `vips_shrinkv`); the output
    /// height is `round(height / vshrink)`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_shrinkv`].
    #[track_caller]
    pub fn shrinkv(&self, vshrink: u32) -> Raster {
        expect_resample("shrinkv", self.try_shrinkv(vshrink))
    }

    /// Fallible form of [`Raster::reduce`], taking the typed kernel.
    ///
    /// # Errors
    ///
    /// [`ResampleError::BadFactor`] unless both factors are finite numbers
    /// in `1.0..=1_000_000.0`, [`ResampleError::FactorTooLarge`] when the
    /// mask would exceed the libvips `MAX_POINT`,
    /// [`ResampleError::ShrunkToNothing`] when a dimension would reach
    /// zero, or [`ResampleError::Raster`] on allocation failure.
    pub fn try_reduce(
        &self,
        hshrink: f64,
        vshrink: f64,
        kernel: ReduceKernel,
    ) -> Result<Raster, ResampleError> {
        // Bracket the alpha premultiply once around both axes (#348); Nearest
        // is a single-tap pick that must not premultiply, and an identity
        // factor does no work so it needs no bracket either.
        let bracket = kernel != ReduceKernel::Nearest && (hshrink > 1.0 || vshrink > 1.0);
        with_premultiply(self, bracket, |w| {
            let t = reduce_axis(w, vshrink, kernel, 0.0, Axis::Vertical)?;
            reduce_axis(&t, hshrink, kernel, 0.0, Axis::Horizontal)
        })
    }

    /// Downsample with an anti-aliasing kernel (libvips `vips_reduce`):
    /// vertical pass then horizontal pass, no box pre-pass (gap 0), output
    /// dimensions `round(dim / factor)`. The kernel is a libvips nickname:
    /// `"nearest"`, `"linear"`, `"cubic"`, `"mitchell"`, `"lanczos2"`, or
    /// `"lanczos3"`. Panicking form of [`Raster::try_reduce`], matching the
    /// ported-test call surface.
    ///
    /// Alpha images (every kernel except Nearest) are premultiplied once around
    /// both axes so transparent colour cannot bleed into opaque neighbours; see
    /// the module-level *Premultiplied alpha* note. This diverges from bare
    /// `vips_reduce`, which does not premultiply.
    ///
    /// # Panics
    ///
    /// Panics on an unknown kernel name or any [`ResampleError`]; see
    /// [`Raster::try_reduce`].
    #[track_caller]
    pub fn reduce(&self, hshrink: f64, vshrink: f64, kernel: &str) -> Raster {
        let kernel = expect_resample("reduce", ReduceKernel::from_name(kernel));
        expect_resample("reduce", self.try_reduce(hshrink, vshrink, kernel))
    }

    /// Fallible form of [`Raster::reduceh`], taking the typed kernel.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_reduce`].
    pub fn try_reduceh(&self, hshrink: f64, kernel: ReduceKernel) -> Result<Raster, ResampleError> {
        let bracket = kernel != ReduceKernel::Nearest && hshrink > 1.0;
        with_premultiply(self, bracket, |w| {
            reduce_axis(w, hshrink, kernel, 0.0, Axis::Horizontal)
        })
    }

    /// Horizontal kernel reduce (libvips `vips_reduceh`); the kernel is a
    /// libvips nickname as in [`Raster::reduce`].
    ///
    /// # Panics
    ///
    /// Panics on an unknown kernel name or any [`ResampleError`]; see
    /// [`Raster::try_reduceh`].
    #[track_caller]
    pub fn reduceh(&self, hshrink: f64, kernel: &str) -> Raster {
        let kernel = expect_resample("reduceh", ReduceKernel::from_name(kernel));
        expect_resample("reduceh", self.try_reduceh(hshrink, kernel))
    }

    /// Fallible form of [`Raster::reducev`], taking the typed kernel.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_reduce`].
    pub fn try_reducev(&self, vshrink: f64, kernel: ReduceKernel) -> Result<Raster, ResampleError> {
        let bracket = kernel != ReduceKernel::Nearest && vshrink > 1.0;
        with_premultiply(self, bracket, |w| {
            reduce_axis(w, vshrink, kernel, 0.0, Axis::Vertical)
        })
    }

    /// Vertical kernel reduce (libvips `vips_reducev`); the kernel is a
    /// libvips nickname as in [`Raster::reduce`].
    ///
    /// # Panics
    ///
    /// Panics on an unknown kernel name or any [`ResampleError`]; see
    /// [`Raster::try_reducev`].
    #[track_caller]
    pub fn reducev(&self, vshrink: f64, kernel: &str) -> Raster {
        let kernel = expect_resample("reducev", ReduceKernel::from_name(kernel));
        expect_resample("reducev", self.try_reducev(vshrink, kernel))
    }

    /// Fallible form of [`Raster::resize`] with explicit options.
    ///
    /// # Errors
    ///
    /// [`ResampleError::BadScale`] unless the scales are finite positive
    /// numbers, plus any error of the underlying reduce or affine pass.
    pub fn try_resize_with(
        &self,
        scale: f64,
        options: ResizeOptions,
    ) -> Result<Raster, ResampleError> {
        let mut hscale = scale;
        let mut vscale = options.vscale.unwrap_or(scale);
        for s in [hscale, vscale] {
            if !s.is_finite() || s <= 0.0 {
                return Err(ResampleError::BadScale { scale: s });
            }
        }

        let nearest = options.kernel == ReduceKernel::Nearest;
        let mut start = self.clone();

        // The nearest kernel subsamples the integer part first
        // (`vips_resize_build`).
        if nearest {
            let int_shrink = |dim: u32, s: f64| -> u32 {
                let f = if options.gap < 1.0 {
                    (1.0 / s).floor()
                } else {
                    let target = round_uint(f64::from(dim) * s).max(1) as f64;
                    (f64::from(dim) / target / options.gap).floor()
                };
                f.max(1.0) as u32
            };
            let int_h = int_shrink(start.width(), hscale);
            let int_v = int_shrink(start.height(), vscale);
            if int_h > 1 || int_v > 1 {
                start = subsample(&start, int_h, int_v)?;
                hscale *= f64::from(int_h);
                vscale *= f64::from(int_v);
            }
        }

        // Don't let either axis drop below one pixel.
        hscale = hscale.max(1.0 / f64::from(start.width()));
        vscale = vscale.max(1.0 / f64::from(start.height()));

        // Premultiply once around the *whole* resize — the residual reduce
        // passes and the affine enlargement together (#348/#406) — so every
        // separable pass is coverage-weighted and internally consistent. This
        // fixes the mixed downscale-one-axis / upscale-other-axis case (the
        // reduce emits premultiplied colour and the affine, running with
        // `premultiplied: true`, keeps it premultiplied rather than
        // interpolating straight-alpha colour across transparency boundaries)
        // and premultiplies a pure upscale too. The single un-premultiply
        // happens once after, in `with_premultiply`. Nearest never averages, so
        // it is not bracketed; nor is a no-op resize that neither reduces nor
        // enlarges.
        let will_reduce = vscale < 1.0 || hscale < 1.0;
        let will_upscale = hscale > 1.0 || vscale > 1.0;
        let bracket = !nearest && (will_reduce || will_upscale);
        with_premultiply(&start, bracket, |w| {
            let mut cur = w.clone();

            // Any residual downsizing, vertical then horizontal.
            if vscale < 1.0 {
                cur = reduce_axis(
                    &cur,
                    1.0 / vscale,
                    options.kernel,
                    options.gap,
                    Axis::Vertical,
                )?;
            }
            if hscale < 1.0 {
                cur = reduce_axis(
                    &cur,
                    1.0 / hscale,
                    options.kernel,
                    options.gap,
                    Axis::Horizontal,
                )?;
            }

            // Any upsizing: affine with the interpolator mapped from the
            // kernel, or pixel replication for integral nearest enlargement.
            if hscale > 1.0 || vscale > 1.0 {
                if nearest && hscale.fract() == 0.0 && vscale.fract() == 0.0 {
                    cur = zoom(&cur, hscale as u32, vscale as u32)?;
                } else {
                    let id = if nearest { 0.0 } else { 0.5 };
                    let matrix = [hscale.max(1.0), 0.0, 0.0, vscale.max(1.0)];
                    cur = cur.try_affine_with(
                        matrix,
                        options.kernel.upsize_interpolator(),
                        AffineOptions {
                            idx: id,
                            idy: id,
                            extend: Extend::Copy,
                            premultiplied: true,
                            ..AffineOptions::default()
                        },
                    )?;
                }
            }

            Ok(cur)
        })
    }

    /// Fallible form of [`Raster::resize`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_resize_with`].
    pub fn try_resize(&self, scale: f64) -> Result<Raster, ResampleError> {
        self.try_resize_with(scale, ResizeOptions::default())
    }

    /// Resize by a scale factor (libvips `vips_resize`): reduce with the
    /// default `lanczos3` kernel for downscales, affine with bicubic for
    /// upscales. Output dimensions are `round(dim * scale)`. Panicking form
    /// of [`Raster::try_resize`], matching the ported-test call surface.
    ///
    /// As in `vips_resize`, an alpha image is premultiplied once around the
    /// whole operation — the reduce passes and the affine enlargement together
    /// — and unpremultiplied once at the end, so every axis is coverage-weighted
    /// consistently; see the module-level *Premultiplied alpha* note.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_resize`].
    #[track_caller]
    pub fn resize(&self, scale: f64) -> Raster {
        expect_resample("resize", self.try_resize(scale))
    }

    /// Resize with explicit [`ResizeOptions`]. Panicking form of
    /// [`Raster::try_resize_with`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_resize_with`].
    #[track_caller]
    pub fn resize_with(&self, scale: f64, options: ResizeOptions) -> Raster {
        expect_resample("resize", self.try_resize_with(scale, options))
    }

    /// Fallible form of [`Raster::affine`] with explicit options.
    ///
    /// # Errors
    ///
    /// [`ResampleError::SingularMatrix`] when the matrix cannot be
    /// inverted, [`ResampleError::BadOutputArea`] when the output area is
    /// empty or overflows `u32` dimensions, or [`ResampleError::Raster`] on
    /// allocation failure.
    pub fn try_affine_with(
        &self,
        matrix: [f64; 4],
        interpolate: Interpolator,
        options: AffineOptions,
    ) -> Result<Raster, ResampleError> {
        let [a, b, c, d] = matrix;
        let det = a * d - b * c;
        if det.abs() < TOO_SMALL {
            return Err(ResampleError::SingularMatrix);
        }
        let (ia, ib, ic, idd) = (d / det, -b / det, -c / det, a / det);

        let w = i64::from(self.width());
        let h = i64::from(self.height());

        // Default output area: bounding box of the transformed input
        // corners, rounded to nearest. Computed from the matrix alone; the
        // displacement options do not move it (`vips__transform_set_area`
        // runs before they are applied in `vips_affine_build`).
        let default_oarea = {
            let corners = [
                (0.0, 0.0),
                (w as f64, 0.0),
                (0.0, h as f64),
                (w as f64, h as f64),
            ];
            let xs = corners.map(|(x, y)| a * x + b * y);
            let ys = corners.map(|(x, y)| c * x + d * y);
            let fold =
                |v: [f64; 4], f: fn(f64, f64) -> f64| v.into_iter().reduce(f).expect("non-empty");
            let left = fold(xs, f64::min);
            let right = fold(xs, f64::max);
            let top = fold(ys, f64::min);
            let bottom = fold(ys, f64::max);
            [
                round_int(left),
                round_int(top),
                round_int(right - left),
                round_int(bottom - top),
            ]
        };
        let oarea = options
            .oarea
            .map(|o| o.map(i64::from))
            .unwrap_or(default_oarea);
        let (ow, oh) = (oarea[2], oarea[3]);
        if ow < 1 || oh < 1 || u32::try_from(ow).is_err() || u32::try_from(oh).is_err() {
            return Err(ResampleError::BadOutputArea {
                width: ow,
                height: oh,
            });
        }

        // Identity transform writing the full input straight through is a
        // copy (`vips__transform_isidentity` shortcut).
        if matrix == [1.0, 0.0, 0.0, 1.0]
            && options.odx == 0.0
            && options.ody == 0.0
            && options.idx == 0.0
            && options.idy == 0.0
            && oarea == [0, 0, w, h]
        {
            return Ok(self.clone());
        }

        let format = self.format();
        let layout = SampleLayout::of(format);
        let bands = format.channels();
        let premultiply = format.has_alpha() && !options.premultiplied;
        let fetch = TapFetch::new(self, options.extend, options.background);

        let mut out = Raster::zeroed(ow as u32, oh as u32, format)?;
        let buf = out.data_mut();
        let mut scratch = InterpScratch::new(bands);
        let mut acc = vec![0.0f64; bands];

        for y in 0..oh {
            let oy = (y + oarea[1]) as f64 - options.ody;
            for x in 0..ow {
                let ox = (x + oarea[0]) as f64 - options.odx;
                let ix = ia * ox + ib * oy - options.idx;
                let iy = ic * ox + idd * oy - options.idy;
                let oi = (y * ow + x) as usize * bands;
                let (fx, fy) = (ix.floor(), iy.floor());
                if fx >= -1.0 && fx <= (w - 1) as f64 && fy >= -1.0 && fy <= (h - 1) as f64 {
                    interpolate_at(
                        &fetch,
                        interpolate,
                        ix,
                        iy,
                        premultiply,
                        &mut scratch,
                        &mut acc,
                    );
                    if premultiply {
                        // `vips_affine_gen` writes the interpolated
                        // premultiplied pixel into the FLOAT image
                        // `vips_unpremultiply` then reads, so the accumulator
                        // is quantised to `f32` at that seam (issue #664).
                        for v in acc.iter_mut() {
                            *v = f64::from(*v as f32);
                        }
                        unpremultiply(&mut acc, fetch.alpha_max);
                    }
                    for (bi, v) in acc.iter().enumerate() {
                        layout.write(buf, oi + bi, *v);
                    }
                } else {
                    // `vips_affine_gen` paints everything outside the
                    // transformed input with `affine->ink`, the same converted
                    // background the taps read, not with the caller's `f64`
                    // (issue #736).
                    for bi in 0..bands {
                        layout.write(buf, oi + bi, fetch.background);
                    }
                }
            }
        }

        out.carry_meta_from(self);
        Ok(out)
    }

    /// Fallible form of [`Raster::affine`], taking the typed interpolator.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_affine_with`].
    pub fn try_affine(
        &self,
        matrix: [f64; 4],
        interpolate: Interpolator,
    ) -> Result<Raster, ResampleError> {
        self.try_affine_with(matrix, interpolate, AffineOptions::default())
    }

    /// Transform by the 2x2 matrix `[a, b, c, d]` (libvips `vips_affine`):
    /// input `(x, y)` maps to output `(a*x + b*y, c*x + d*y)`, the output
    /// is the rounded bounding box of the transformed input, and each
    /// output pixel is inverse-mapped and interpolated. The interpolator is
    /// a libvips nickname: `"nearest"`, `"bilinear"`, `"bicubic"`,
    /// `"nohalo"`, or `"lbb"`. Panicking form of [`Raster::try_affine`],
    /// matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on an unknown interpolator name or any
    /// [`ResampleError`]; see [`Raster::try_affine`].
    #[track_caller]
    pub fn affine(&self, matrix: [f64; 4], interpolate: &str) -> Raster {
        let interpolate = expect_resample("affine", Interpolator::from_name(interpolate));
        expect_resample("affine", self.try_affine(matrix, interpolate))
    }

    /// Fallible form of [`Raster::similarity`] with an explicit
    /// interpolator.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_affine_with`].
    pub fn try_similarity_with(
        &self,
        angle: f64,
        scale: f64,
        interpolate: Interpolator,
    ) -> Result<Raster, ResampleError> {
        let rad = angle * PI / 180.0;
        let a = scale * rad.cos();
        let b = scale * -rad.sin();
        self.try_affine([a, b, -b, a], interpolate)
    }

    /// Fallible form of [`Raster::similarity`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_affine_with`].
    pub fn try_similarity(&self, angle: f64, scale: f64) -> Result<Raster, ResampleError> {
        self.try_similarity_with(angle, scale, Interpolator::Bilinear)
    }

    /// Rotate by `angle` degrees and scale by `scale` (libvips
    /// `vips_similarity`), expanding the canvas to the rotated bounding
    /// box and interpolating bilinearly, the libvips default. Panicking
    /// form of [`Raster::try_similarity`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_similarity`].
    #[track_caller]
    pub fn similarity(&self, angle: f64, scale: f64) -> Raster {
        expect_resample("similarity", self.try_similarity(angle, scale))
    }

    /// Fallible form of [`Raster::rotate`] with an explicit interpolator.
    ///
    /// # Errors
    ///
    /// See [`Raster::try_affine_with`].
    pub fn try_rotate_with(
        &self,
        angle: f64,
        interpolate: Interpolator,
    ) -> Result<Raster, ResampleError> {
        self.try_similarity_with(angle, 1.0, interpolate)
    }

    /// Fallible form of [`Raster::rotate`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_affine_with`].
    pub fn try_rotate(&self, angle: f64) -> Result<Raster, ResampleError> {
        self.try_similarity(angle, 1.0)
    }

    /// Rotate by an arbitrary angle in degrees (libvips `vips_rotate`,
    /// `vips_similarity` with scale 1), expanding the canvas to the
    /// rotated bounding box. Panicking form of [`Raster::try_rotate`],
    /// matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_rotate`].
    #[track_caller]
    pub fn rotate(&self, angle: f64) -> Raster {
        expect_resample("rotate", self.try_rotate(angle))
    }

    /// Fallible form of [`Raster::mapim`], taking the typed interpolator.
    ///
    /// # Errors
    ///
    /// [`ResampleError::IndexBands`] unless the index image has exactly
    /// two bands, or [`ResampleError::Raster`] on allocation failure.
    pub fn try_mapim(
        &self,
        index: &Raster,
        interpolate: Interpolator,
    ) -> Result<Raster, ResampleError> {
        let index_bands = index.format().channels();
        if index_bands != 2 {
            return Err(ResampleError::IndexBands { bands: index_bands });
        }

        let format = self.format();
        let layout = SampleLayout::of(format);
        let bands = format.channels();
        let (w, h) = (f64::from(self.width()), f64::from(self.height()));
        let background = 0.0f64;
        let fetch = TapFetch::new(self, Extend::Background, background);

        let index_layout = SampleLayout::of(index.format());
        let index_data = index.data();
        let (ow, oh) = (index.width() as usize, index.height() as usize);

        let mut out = Raster::zeroed(index.width(), index.height(), format)?;
        let buf = out.data_mut();
        let mut scratch = InterpScratch::new(bands);
        let mut acc = vec![0.0f64; bands];

        for y in 0..oh {
            for x in 0..ow {
                let ii = (y * ow + x) * 2;
                let sx = index_layout.read(index_data, ii);
                let sy = index_layout.read(index_data, ii + 1);
                let oi = (y * ow + x) * bands;
                // Coordinates inside [-1, dim + 1) interpolate with
                // background-extended taps (edge antialiasing); everything
                // else, including NaN, paints the background
                // (`vips_mapim_gen` clip against `Xsize - window_size` on
                // the embedded input).
                if sx >= -1.0 && sx < w + 1.0 && sy >= -1.0 && sy < h + 1.0 {
                    interpolate_at(&fetch, interpolate, sx, sy, false, &mut scratch, &mut acc);
                    for (bi, v) in acc.iter().enumerate() {
                        layout.write(buf, oi + bi, *v);
                    }
                } else {
                    // The same converted ink the taps read (issue #736).
                    for bi in 0..bands {
                        layout.write(buf, oi + bi, fetch.background);
                    }
                }
            }
        }

        // The header block comes from the image being remapped, not from the
        // index image, which is a coordinate field and not a picture.
        out.carry_meta_from(self);
        Ok(out)
    }

    /// Remap through a two-band coordinate image (libvips `vips_mapim`):
    /// output pixel `(x, y)` samples this image at the position stored in
    /// the index pixel `(x, y)` (band 0 is the source x, band 1 the source
    /// y). The interpolator is a libvips nickname as in
    /// [`Raster::affine`]. Panicking form of [`Raster::try_mapim`],
    /// matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on an unknown interpolator name or any
    /// [`ResampleError`]; see [`Raster::try_mapim`].
    #[track_caller]
    pub fn mapim(&self, index: &Raster, interpolate: &str) -> Raster {
        let interpolate = expect_resample("mapim", Interpolator::from_name(interpolate));
        expect_resample("mapim", self.try_mapim(index, interpolate))
    }

    /// Fallible form of [`Raster::constant_u8`].
    ///
    /// # Errors
    ///
    /// [`ResampleError::Raster`] on zero dimensions or allocation failure.
    pub fn try_constant_u8(width: u32, height: u32, value: u8) -> Result<Raster, ResampleError> {
        let mut out = Raster::zeroed(width, height, PixelFormat::Gray8)?;
        out.data_mut().fill(value);
        Ok(out)
    }

    /// Create a one-band 8-bit image with every sample set to `value`
    /// (libvips `vips_black` plus a constant add). The ported resample
    /// cell uses this to pin constant preservation through
    /// [`Raster::reduce`]; it lives in this module to keep the resample
    /// batch file-disjoint from [`crate::create`]. Panicking form of
    /// [`Raster::try_constant_u8`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ResampleError`]; see [`Raster::try_constant_u8`].
    #[track_caller]
    pub fn constant_u8(width: u32, height: u32, value: u8) -> Raster {
        expect_resample("constant_u8", Self::try_constant_u8(width, height, value))
    }
}

// ---------------------------------------------------------------------------
// Thumbnail (vips_thumbnail / vips_thumbnail_image)
// ---------------------------------------------------------------------------

/// The colour space a thumbnail resamples in, mirroring the `vips_thumbnail`
/// `linear` and `export-profile` options.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ThumbSpace {
    /// Resample directly in the decoded device space (the default).
    Device,
    /// Import to linear-light scRGB, resample, re-encode to sRGB
    /// (`--linear`).
    Linear,
    /// Import through the embedded ICC profile, resample in the PCS, then
    /// export to the built-in sRGB profile (`--export-profile srgb`).
    IccSrgb,
}

/// The built-in sRGB ICC profile bytes (moxcms `ColorProfile::new_srgb`),
/// the target for the [`ThumbSpace::IccSrgb`] export.
fn builtin_srgb_profile() -> Result<Vec<u8>, ThumbnailError> {
    moxcms::ColorProfile::new_srgb()
        .encode()
        .map_err(|e| ThumbnailError::Profile(format!("{e:?}")))
}

/// Resize `src` by `scale`, short-circuiting the exact-unit case so a
/// no-op thumbnail (target == source) keeps the pixels and metadata
/// untouched rather than round-tripping through the resampler.
fn resize_if_needed(src: &Raster, scale: f64) -> Result<Raster, ResampleError> {
    if (scale - 1.0).abs() <= f64::EPSILON {
        Ok(src.clone())
    } else {
        src.try_resize(scale)
    }
}

/// Fit an in-memory raster into a `width` x `height` box, the core of the
/// whole thumbnail family (libvips `vips_thumbnail_image`).
///
/// The shrink factor fits the image inside the bounding box preserving
/// aspect (the larger of the per-axis shrinks) or, when `crop` is set,
/// fills it (the smaller), exactly as `vips_thumbnail_calculate_shrink`.
/// The resample runs in the space `space` selects, and a crop centre-crops
/// the filled image down to the box.
fn thumbnail_fit(
    src: &Raster,
    width: u32,
    height: Option<u32>,
    crop: bool,
    space: ThumbSpace,
) -> Result<Raster, ThumbnailError> {
    if width == 0 {
        return Err(ThumbnailError::BadSize { size: 0 });
    }
    let box_w = width;
    let box_h = match height {
        Some(0) => return Err(ThumbnailError::BadSize { size: 0 }),
        Some(h) => h,
        // The bare-width forms fit a square box, matching the ported
        // `thumbnail(width)` call surface and the vips CLI where a single
        // size bounds both axes.
        None => width,
    };

    let horizontal = f64::from(src.width()) / f64::from(box_w);
    let vertical = f64::from(src.height()) / f64::from(box_h);
    let shrink = if crop {
        horizontal.min(vertical)
    } else {
        horizontal.max(vertical)
    };
    let scale = 1.0 / shrink;

    let fitted = match space {
        ThumbSpace::Device => resize_if_needed(src, scale)?,
        ThumbSpace::Linear => {
            let linear = src.try_colourspace(Interpretation::ScRgb)?;
            // The resample carries the scRGB tag through (issue #789), so the
            // re-encode to sRGB sees the space it was told to expect. This
            // used to restamp the tag with a `copy()` here, which cost a whole
            // extra image-sized clone to undo a bug one call up.
            let small = resize_if_needed(&linear, scale)?;
            small.try_colourspace(Interpretation::Srgb)?
        }
        ThumbSpace::IccSrgb => {
            let lab = src.try_icc_import_with(Intent::Perceptual, None, Some(Pcs::Lab))?;
            // Same again: the Lab tag survives the resample, so the export
            // still takes the direct PCS path. Only the profile is stamped,
            // pointing the export at the built-in sRGB one.
            let mut small = resize_if_needed(&lab, scale)?;
            small.set_icc_profile(&builtin_srgb_profile()?);
            small.try_icc_export_with(8, Intent::Perceptual, None)?
        }
    };

    if crop {
        let (ow, oh) = (fitted.width(), fitted.height());
        let cw = box_w.min(ow);
        let ch = box_h.min(oh);
        let left = (ow - cw) / 2;
        let top = (oh - ch) / 2;
        Ok(fitted.try_extract_area(left, top, cw, ch)?)
    } else {
        Ok(fitted)
    }
}

impl Raster {
    /// Fallible form of [`Raster::thumbnail`].
    ///
    /// # Errors
    ///
    /// [`ThumbnailError::Decode`] when the file cannot be read or decoded,
    /// [`ThumbnailError::BadSize`] for a zero target, or the resample /
    /// crop errors from the fit.
    pub fn try_thumbnail(
        path: &Path,
        width: u32,
        height: Option<u32>,
        crop: bool,
    ) -> Result<Raster, ThumbnailError> {
        let src = crate::source::decode_file(path)?;
        thumbnail_fit(&src, width, height, crop, ThumbSpace::Device)
    }

    /// Make a thumbnail from an image file (libvips `vips_thumbnail`).
    ///
    /// The image is loaded and shrunk to fit inside the `width` x `height`
    /// bounding box preserving aspect ratio; a bare width (`height` is
    /// `None`) fits a square `width` x `width` box, so the bound axis lands
    /// exactly on `width`. With `crop` the image fills the box and is
    /// centre-cropped to it. The heavy shrink runs through
    /// [`Raster::resize`], whose gap-driven box pre-shrink keeps the reduce
    /// mask bounded even for large downscales, matching the shrink-on-load
    /// then residual-reduce shape of `vips_thumbnail`. Panicking form of
    /// [`Raster::try_thumbnail`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`ThumbnailError`]; see [`Raster::try_thumbnail`].
    #[track_caller]
    pub fn thumbnail(path: &Path, width: u32, height: Option<u32>, crop: bool) -> Raster {
        expect_thumbnail(Self::try_thumbnail(path, width, height, crop))
    }

    /// Fallible form of [`Raster::thumbnail_buffer`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_thumbnail`]; decodes from memory instead of a file.
    pub fn try_thumbnail_buffer(data: &[u8], width: u32) -> Result<Raster, ThumbnailError> {
        let src = crate::source::decode_bytes(data)?;
        thumbnail_fit(&src, width, None, false, ThumbSpace::Device)
    }

    /// Make a thumbnail from an in-memory encoded image buffer (libvips
    /// `vips_thumbnail_buffer`), fitting a square `width` x `width` box.
    /// Panicking form of [`Raster::try_thumbnail_buffer`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ThumbnailError`]; see [`Raster::try_thumbnail_buffer`].
    #[track_caller]
    pub fn thumbnail_buffer(data: &[u8], width: u32) -> Raster {
        expect_thumbnail(Self::try_thumbnail_buffer(data, width))
    }

    /// Fallible form of [`Raster::thumbnail_with_options`].
    ///
    /// # Errors
    ///
    /// See [`Raster::try_thumbnail`], plus [`ThumbnailError::Colour`] from
    /// the linear-light import / export.
    pub fn try_thumbnail_with_options(
        path: &Path,
        width: u32,
        linear: bool,
    ) -> Result<Raster, ThumbnailError> {
        let src = crate::source::decode_file(path)?;
        let space = if linear {
            ThumbSpace::Linear
        } else {
            ThumbSpace::Device
        };
        thumbnail_fit(&src, width, None, false, space)
    }

    /// Make a thumbnail with the `vips_thumbnail` `linear` option: when
    /// `linear` is set the reduce runs in linear-light scRGB and the result
    /// is re-encoded to sRGB, which avoids the darkening a naive gamma-space
    /// average produces. Fits a square `width` x `width` box. Panicking form
    /// of [`Raster::try_thumbnail_with_options`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ThumbnailError`].
    #[track_caller]
    pub fn thumbnail_with_options(path: &Path, width: u32, linear: bool) -> Raster {
        expect_thumbnail(Self::try_thumbnail_with_options(path, width, linear))
    }

    /// Fallible form of [`Raster::thumbnail_with_profile`].
    ///
    /// # Errors
    ///
    /// [`ThumbnailError::UnknownProfile`] for an output profile other than
    /// `"srgb"`, plus the decode / colour / resample errors.
    pub fn try_thumbnail_with_profile(
        path: &Path,
        width: u32,
        output_profile: &str,
    ) -> Result<Raster, ThumbnailError> {
        let space = match output_profile {
            "srgb" | "sRGB" => ThumbSpace::IccSrgb,
            other => {
                return Err(ThumbnailError::UnknownProfile {
                    name: other.to_string(),
                });
            }
        };
        let src = crate::source::decode_file(path)?;
        thumbnail_fit(&src, width, None, false, space)
    }

    /// Make a thumbnail through the embedded ICC profile (libvips
    /// `vips_thumbnail` with `export-profile`): the image is imported from
    /// its embedded profile to the Lab PCS, reduced there, and exported to
    /// `output_profile` (only the built-in `"srgb"` today). Fits a square
    /// `width` x `width` box. Panicking form of
    /// [`Raster::try_thumbnail_with_profile`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ThumbnailError`].
    #[track_caller]
    pub fn thumbnail_with_profile(path: &Path, width: u32, output_profile: &str) -> Raster {
        expect_thumbnail(Self::try_thumbnail_with_profile(
            path,
            width,
            output_profile,
        ))
    }

    /// Fallible form of [`Raster::thumbnail_image`].
    ///
    /// # Errors
    ///
    /// [`ThumbnailError::BadSize`] for a zero target, or the resample
    /// errors from the fit.
    pub fn try_thumbnail_image(&self, width: u32) -> Result<Raster, ThumbnailError> {
        thumbnail_fit(self, width, None, false, ThumbSpace::Device)
    }

    /// Make a thumbnail from this already-loaded raster (libvips
    /// `vips_thumbnail_image`), fitting a square `width` x `width` box
    /// preserving aspect ratio. This is the in-memory counterpart to the
    /// file-loading [`Raster::thumbnail`]; the sequential-access ported
    /// cell drives it after a decode. Panicking form of
    /// [`Raster::try_thumbnail_image`].
    ///
    /// # Panics
    ///
    /// Panics on any [`ThumbnailError`]; see [`Raster::try_thumbnail_image`].
    #[track_caller]
    pub fn thumbnail_image(&self, width: u32) -> Raster {
        expect_thumbnail(self.try_thumbnail_image(width))
    }
}

/// Map a [`ThumbnailError`] onto the shared decode error, preserving the
/// decode cause and folding the resample/colour/crop steps into an I/O error.
fn thumbnail_to_decode(err: ThumbnailError) -> crate::codec::DecodeError {
    match err {
        ThumbnailError::Decode(source) => source,
        other => crate::source::SourceError::Io(std::io::Error::other(other.to_string())),
    }
}

/// Make a thumbnail from an image file, bounded by `width` (libvips
/// `vips_thumbnail` bare-width form).
///
/// A convenience free function over [`Raster::try_thumbnail`] returning the
/// shared [`crate::codec::DecodeError`], matching the ported foreign cell's
/// `thumbnail(path, width)` surface. The image is loaded and shrunk to fit
/// inside a `width` x `width` box, preserving aspect ratio.
///
/// # Errors
///
/// A [`crate::codec::DecodeError`] when the file cannot be read or decoded,
/// or when the resample step fails.
pub fn thumbnail(path: &Path, width: u32) -> Result<Raster, crate::codec::DecodeError> {
    Raster::try_thumbnail(path, width, None, false).map_err(thumbnail_to_decode)
}

/// Make a thumbnail from an image file into a `width` x `height` box with a
/// crop mode (libvips `vips_thumbnail` with `crop`).
///
/// `crop` selects the fit: `"none"` (or an empty string) fits inside the box
/// preserving aspect ratio, and any other value (for example `"centre"`)
/// fills the box and centre-crops to it.
///
/// # Errors
///
/// As [`thumbnail`], plus the crop step.
pub fn thumbnail_crop(
    path: &Path,
    width: u32,
    height: u32,
    crop: &str,
) -> Result<Raster, crate::codec::DecodeError> {
    let do_crop = !matches!(crop, "" | "none");
    Raster::try_thumbnail(path, width, Some(height), do_crop).map_err(thumbnail_to_decode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversion::Angle;

    /// A 4x4 Gray8 ramp with distinct values per pixel.
    fn ramp_4x4() -> Raster {
        let data: Vec<u8> = (0..16u8).map(|v| v * 10).collect();
        Raster::new(4, 4, PixelFormat::Gray8, data).unwrap()
    }

    /// shrink by 2 box-averages each 2x2 block with round-half-up
    /// integer arithmetic.
    #[test]
    fn shrink_2x_averages_blocks() {
        let im = ramp_4x4();
        let out = im.shrink(2.0, 2.0);
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 2);
        // Blocks: {0,10,40,50} {20,30,60,70} {80,90,120,130} {100,110,140,150},
        // means with (sum + 2) / 4 rounding.
        assert_eq!(out.data(), &[25, 45, 105, 125]);
    }

    /// Fractional shrink sizes round to nearest and preserve a constant
    /// image exactly through the lanczos3 residual reduce.
    #[test]
    fn shrink_fractional_dims_and_constant() {
        let im = Raster::constant_u8(10, 10, 77);
        let out = im.shrink(2.5, 2.5);
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
        assert!(out.data().iter().all(|&v| v == 77));
    }

    /// shrink factors below 1 are a typed error, as in libvips.
    #[test]
    fn shrink_bad_factor_is_typed_error() {
        let im = ramp_4x4();
        assert!(matches!(
            im.try_shrink(0.5, 2.0),
            Err(ResampleError::BadFactor { op: "shrink", .. })
        ));
    }

    /// Every reduce kernel preserves constant images exactly (the masks
    /// are normalised and the edges extend by replication).
    #[test]
    fn reduce_preserves_constants_for_every_kernel() {
        for &val in &[0u8, 1, 2, 254, 255] {
            let im = Raster::constant_u8(10, 10, val);
            for kernel in [
                "nearest", "linear", "cubic", "mitchell", "lanczos2", "lanczos3",
            ] {
                let out = im.reduce(2.0, 2.0, kernel);
                assert_eq!(out.width(), 5);
                assert_eq!(out.height(), 5);
                assert!(
                    out.data().iter().all(|&v| v == val),
                    "constant {val} not preserved by reduce with {kernel}"
                );
            }
        }
    }

    /// reduce output dimensions round to nearest, and the average of a
    /// smooth ramp stays close through every kernel.
    #[test]
    fn reduce_dims_and_average() {
        let im = crate::source::generate_test_raster(64, 48).unwrap();
        for kernel in [
            "nearest", "linear", "cubic", "mitchell", "lanczos2", "lanczos3",
        ] {
            for &fac in &[1.0f64, 1.1, 1.5, 1.999] {
                let out = im.reduce(fac, fac, kernel);
                assert_eq!(i64::from(out.width()), round_uint(64.0 / fac));
                assert_eq!(i64::from(out.height()), round_uint(48.0 / fac));
                let d = (out.avg() - im.avg()).abs();
                assert!(d < 2.0, "reduce({fac}, {kernel}) moved the average by {d}");
            }
        }
    }

    /// An unknown kernel or interpolator nickname is a typed error, and
    /// every recognised interpolator nickname parses, including the
    /// nohalo and lbb minmod resamplers.
    #[test]
    fn kernel_and_interpolator_parsing() {
        assert!(matches!(
            ReduceKernel::from_name("box"),
            Err(ResampleError::UnknownKernel { .. })
        ));
        assert_eq!(
            Interpolator::from_name("nohalo").unwrap(),
            Interpolator::Nohalo
        );
        assert_eq!(Interpolator::from_name("lbb").unwrap(), Interpolator::Lbb);
        assert!(matches!(
            Interpolator::from_name("vsqbs"),
            Err(ResampleError::UnknownInterpolator { .. })
        ));
        assert_eq!(
            Interpolator::from_name("bilinear").unwrap(),
            Interpolator::Bilinear
        );
        assert_eq!(
            ReduceKernel::from_name("lanczos3").unwrap(),
            ReduceKernel::Lanczos3
        );
    }

    /// resize dimensions follow round(dim * scale), reproducing the
    /// libvips resize sizing rules the ported cell pins.
    #[test]
    fn resize_dims_round_to_nearest() {
        let im = Raster::black(100, 1);
        let out = im.resize(0.5);
        assert_eq!(out.width(), 50);
        assert_eq!(out.height(), 1);

        let im = Raster::black(1600, 1000);
        let out = im.resize(10.0 / 1600.0);
        assert_eq!(out.width(), 10);
        assert_eq!(out.height(), 6);
    }

    /// resize round-trips dimensions: halving then doubling restores the
    /// original size, and the average of a smooth ramp survives a quarter
    /// resize.
    #[test]
    fn resize_round_trip_and_average() {
        let im = crate::source::generate_test_raster(64, 64).unwrap();
        let half = im.resize(0.5);
        assert_eq!(half.width(), 32);
        assert_eq!(half.height(), 32);
        let back = half.resize(2.0);
        assert_eq!(back.width(), 64);
        assert_eq!(back.height(), 64);

        let quarter = im.resize(0.25);
        assert_eq!(quarter.width(), 16);
        assert_eq!(quarter.height(), 16);
        assert!((quarter.avg() - im.avg()).abs() < 1.0);
    }

    /// A nearest-kernel integral upscale replicates pixels exactly
    /// (the vips_zoom path).
    #[test]
    fn nearest_upsample_duplicates_pixels() {
        let im = Raster::new(2, 2, PixelFormat::Gray8, vec![10, 20, 30, 40]).unwrap();
        let out = im.resize_with(
            2.0,
            ResizeOptions {
                kernel: ReduceKernel::Nearest,
                ..ResizeOptions::default()
            },
        );
        assert_eq!(out.width(), 4);
        assert_eq!(out.height(), 4);
        #[rustfmt::skip]
        let expected = [
            10, 10, 20, 20,
            10, 10, 20, 20,
            30, 30, 40, 40,
            30, 30, 40, 40,
        ];
        assert_eq!(out.data(), &expected);
    }

    /// Bilinear interpolation at the midpoint of four samples is their
    /// mean (pinned through mapim with a half-integer coordinate).
    #[test]
    fn bilinear_midpoint_is_mean_of_four() {
        let im = Raster::new(2, 2, PixelFormat::Gray8, vec![10, 20, 30, 40]).unwrap();
        let mut index = Raster::zeroed(
            1,
            1,
            PixelFormat::FloatF32(core::num::NonZeroU16::new(2).unwrap()),
        )
        .unwrap();
        index.data_mut()[0..4].copy_from_slice(&0.5f32.to_ne_bytes());
        index.data_mut()[4..8].copy_from_slice(&0.5f32.to_ne_bytes());
        let out = im.mapim(&index, "bilinear");
        assert_eq!(out.data(), &[25]);
    }

    /// The identity affine transform is a straight copy.
    #[test]
    fn affine_identity_is_copy() {
        let im = crate::source::generate_test_raster(7, 5).unwrap();
        for interp in ["nearest", "bilinear", "bicubic"] {
            let out = im.affine([1.0, 0.0, 0.0, 1.0], interp);
            assert_eq!(out.width(), im.width());
            assert_eq!(out.height(), im.height());
            assert_eq!(out.data(), im.data());
        }
    }

    /// The transpose matrix [0, 1, 1, 0] samples exactly on the input
    /// grid: every interpolator reproduces rot90 + fliphor byte for byte,
    /// and four applications are the identity (the ported test_affine
    /// invariant).
    #[test]
    fn affine_transpose_matches_rot90_fliphor_and_round_trips() {
        let im = crate::source::generate_test_raster(6, 4).unwrap();
        let reference = im.rot(Angle::D90).fliphor();
        for interp in ["nearest", "bilinear", "bicubic"] {
            let t = im.affine([0.0, 1.0, 1.0, 0.0], interp);
            assert_eq!(t.width(), im.height());
            assert_eq!(t.height(), im.width());
            assert_eq!(
                t.data(),
                reference.data(),
                "transpose mismatch for {interp}"
            );

            let mut x = im.clone();
            for _ in 0..4 {
                x = x.affine([0.0, 1.0, 1.0, 0.0], interp);
            }
            assert_eq!(
                x.data(),
                im.data(),
                "4x transpose not identity for {interp}"
            );
        }
    }

    /// rotate(90) is the rot90 permutation displaced one column right by
    /// the bounding-box rounding, with a background seam in column 0: the
    /// faithful libvips affine geometry (`vips_affine_gen` samples
    /// `in(y, h - x)` for the [0, -1, 1, 0] matrix).
    #[test]
    fn rotate_90_is_shifted_rot90() {
        let im = crate::source::generate_test_raster(6, 4).unwrap();
        let rotated = im.rotate(90.0);
        let reference = im.rot(Angle::D90);
        assert_eq!(rotated.width(), reference.width());
        assert_eq!(rotated.height(), reference.height());

        let bands = im.format().channels();
        let (w, h) = (rotated.width() as usize, rotated.height() as usize);
        for y in 0..h {
            for x in 0..w {
                let got = &rotated.data()[(y * w + x) * bands..][..bands];
                if x == 0 {
                    assert!(got.iter().all(|&v| v == 0), "column 0 should be background");
                } else {
                    let want = &reference.data()[(y * w + x - 1) * bands..][..bands];
                    assert_eq!(got, want, "mismatch at ({x}, {y})");
                }
            }
        }
    }

    /// similarity(0, 2) equals affine([2, 0, 0, 2]) exactly, and
    /// similarity(90, 1) equals affine([0, -1, 1, 0]) (the ported
    /// test_similarity bodies).
    #[test]
    fn similarity_matches_affine() {
        let im = crate::source::generate_test_raster(8, 6).unwrap();

        let scaled = im.similarity(0.0, 2.0);
        let affined = im.affine([2.0, 0.0, 0.0, 2.0], "bilinear");
        assert_eq!(scaled.data(), affined.data());

        let rotated = im.similarity(90.0, 1.0);
        let affined = im.affine([0.0, -1.0, 1.0, 0.0], "bilinear");
        assert_eq!(rotated.width(), affined.width());
        assert_eq!(rotated.height(), affined.height());
        let max_diff = rotated
            .data()
            .iter()
            .zip(affined.data().iter())
            .map(|(&p, &q)| (i16::from(p) - i16::from(q)).unsigned_abs())
            .max()
            .unwrap();
        assert!(max_diff < 50, "similarity(90) vs affine: {max_diff}");
    }

    /// An identity coordinate image maps every pixel to itself: bicubic
    /// weights at integer offsets are exactly [0, 1, 0, 0], so mapim
    /// reproduces the input byte for byte.
    #[test]
    fn mapim_identity_is_exact() {
        let im = crate::source::generate_test_raster(9, 7).unwrap();
        let index = Raster::xyz(im.width(), im.height());
        let out = im.mapim(&index, "bicubic");
        assert_eq!(out.data(), im.data());
        assert!((out.avg() - im.avg()).abs() < 0.001);
    }

    /// mapim rejects index images without exactly two bands.
    #[test]
    fn mapim_index_bands_is_typed_error() {
        let im = ramp_4x4();
        let bad = Raster::black(4, 4);
        assert!(matches!(
            im.try_mapim(&bad, Interpolator::Bilinear),
            Err(ResampleError::IndexBands { bands: 1 })
        ));
    }

    /// A singular affine matrix is a typed error.
    #[test]
    fn affine_singular_matrix_is_typed_error() {
        let im = ramp_4x4();
        assert!(matches!(
            im.try_affine([1.0, 2.0, 2.0, 4.0], Interpolator::Bilinear),
            Err(ResampleError::SingularMatrix)
        ));
    }

    /// Alpha images premultiply through affine: a transpose remains exact
    /// for pixels with non-zero alpha.
    #[test]
    fn affine_transpose_premultiplies_alpha_exactly() {
        #[rustfmt::skip]
        let data = vec![
            200, 10, 30, 255,   40, 80, 120, 128,
            10, 20, 30, 64,     90, 60, 30, 1,
        ];
        let im = Raster::new(2, 2, PixelFormat::Rgba8, data).unwrap();
        let reference = im.rot(Angle::D90).fliphor();
        let out = im.affine([0.0, 1.0, 1.0, 0.0], "bilinear");
        assert_eq!(out.data(), reference.data());
    }

    /// constant_u8 fills a one-band image with the value.
    #[test]
    fn constant_u8_fills() {
        let im = Raster::constant_u8(3, 2, 254);
        assert_eq!(im.format(), PixelFormat::Gray8);
        assert_eq!(im.data(), &[254; 6]);
    }

    /// shrinkh / shrinkv shrink one axis only.
    #[test]
    fn shrink_axis_forms() {
        let im = ramp_4x4();
        let h = im.shrinkh(2);
        assert_eq!((h.width(), h.height()), (2, 4));
        let v = im.shrinkv(2);
        assert_eq!((v.width(), v.height()), (4, 2));
        // First row of shrinkh: means of (0,10) and (20,30).
        assert_eq!(&h.data()[0..2], &[5, 25]);
    }

    /// reduceh / reducev reduce one axis only, with round-to-nearest
    /// sizing.
    #[test]
    fn reduce_axis_forms() {
        let im = Raster::constant_u8(10, 10, 33);
        let h = im.reduceh(2.5, "cubic");
        assert_eq!((h.width(), h.height()), (4, 10));
        let v = im.reducev(2.5, "cubic");
        assert_eq!((v.width(), v.height()), (10, 4));
        assert!(h.data().iter().all(|&p| p == 33));
        assert!(v.data().iter().all(|&p| p == 33));
    }

    /// Regression for #288: the reduce (Lanczos) path must premultiply alpha
    /// so the RGB of transparent pixels cannot bleed across a transparency
    /// boundary. A 24x2 RGBA raster is opaque red on the left (cols 0..11) and
    /// *transparent green* on the right (cols 12..23); a Lanczos3 reduceh has
    /// taps that cross the seam. Under straight-alpha convolution (the pre-fix
    /// behaviour) the transparent green leaks into the output (G > 0) and the
    /// opaque red darkens at the seam. With the premultiply bracket the green,
    /// carrying zero coverage, contributes nothing: G stays 0 everywhere and
    /// the opaque colour stays saturated wherever any coverage survives.
    #[test]
    fn reduce_premultiplies_alpha_no_colour_bleed() {
        let w = 24u32;
        let h = 2u32;
        let mut data = Vec::with_capacity((w * h) as usize * 4);
        for _ in 0..h {
            for x in 0..w {
                if x < 12 {
                    data.extend_from_slice(&[255, 0, 0, 255]); // opaque red
                } else {
                    data.extend_from_slice(&[0, 255, 0, 0]); // transparent green
                }
            }
        }
        let im = Raster::new(w, h, PixelFormat::Rgba8, data).unwrap();
        let out = im.reduceh(2.0, "lanczos3");
        assert_eq!((out.width(), out.height()), (12, 2));

        let ow = out.width() as usize;
        let mut saw_opaque = false;
        let mut saw_transparent = false;
        for (i, chunk) in out.data().chunks(4).enumerate() {
            let (r, g, b, a) = (chunk[0], chunk[1], chunk[2], chunk[3]);
            let col = i % ow;
            // The transparent green must never bleed into any output pixel.
            assert!(
                g <= 1,
                "G bled at col {col}: {g} (transparent green leaked)"
            );
            assert!(b <= 1, "B bled at col {col}: {b}");
            // Wherever coverage survives, the opaque red is preserved intact
            // (no dark fringe): un-premultiply restores the saturated colour.
            if a > 0 {
                assert!(r >= 254, "R darkened at col {col}: {r} (dark fringe)");
            }
            // Deep opaque columns stay fully opaque saturated red...
            if col == 0 {
                assert_eq!((r, a), (255, 255), "deep opaque column must survive");
                saw_opaque = true;
            }
            // ...and deep transparent columns stay fully transparent (colour
            // zeroed, no leaked green).
            if col == ow - 1 {
                assert_eq!(a, 0, "deep transparent column must stay transparent");
                assert_eq!((r, g, b), (0, 0, 0), "transparent colour must be zeroed");
                saw_transparent = true;
            }
        }
        assert!(
            saw_opaque && saw_transparent,
            "fixture must span both regions"
        );
    }

    /// Follow-up to #287/#288: the reduce path must NOT premultiply for the
    /// Nearest kernel. Nearest is a single-tap pick with no averaging, so a
    /// premultiply -> un-premultiply round-trip through the same-bit-depth
    /// integer raster would only requantise — and thus corrupt — the straight-
    /// alpha RGB of semi-transparent pixels (e.g. `(200,100,50,10)` round-trips
    /// to `(204,102,51,10)`). A single-tap nearest pick must return each
    /// selected source pixel byte-identically.
    #[test]
    fn reduce_nearest_preserves_exact_alpha_pixels() {
        // Four semi-transparent colours whose RGB does not survive the
        // premultiply/un-premultiply integer round-trip. Tiled on a 2x2 lattice
        // so every source pixel is one of the four; a single-tap pick can only
        // ever return one of them exactly.
        const PALETTE: [[u8; 4]; 4] = [
            [200, 100, 50, 10],
            [30, 220, 140, 3],
            [170, 90, 240, 7],
            [90, 200, 60, 5],
        ];
        let (w, h) = (4u32, 4u32);
        let mut data = Vec::with_capacity((w * h) as usize * 4);
        for y in 0..h {
            for x in 0..w {
                let idx = (x % 2) as usize + 2 * (y % 2) as usize;
                data.extend_from_slice(&PALETTE[idx]);
            }
        }
        let im = Raster::new(w, h, PixelFormat::Rgba8, data).unwrap();

        for (label, out) in [
            ("reduceh", im.reduceh(2.0, "nearest")),
            ("reducev", im.reducev(2.0, "nearest")),
            ("reduce", im.reduce(2.0, 2.0, "nearest")),
            ("reduce-fractional", im.reduce(1.5, 1.5, "nearest")),
        ] {
            assert_eq!(out.format(), PixelFormat::Rgba8);
            for px in out.data().as_chunks::<4>().0 {
                assert!(
                    PALETTE.contains(px),
                    "{label} nearest corrupted a semi-transparent pixel: {px:?} \
                     is not an exact source sample (premultiply round-trip)"
                );
            }
        }
    }

    /// Regression for #604: the float un-premultiply damps a near-zero alpha
    /// to nothing instead of dividing by it. Pinned on the vips 8.18.4 binary
    /// with a 1x1 float pixel `(100, 100, 100, alpha)` under the default
    /// `max_alpha` of 255:
    ///
    /// ```text
    /// vips linear b.v a.v "0 0 0 0" "100 100 100 <alpha>"
    /// vips unpremultiply a.v u.v ; vips getpoint u.v 0 0
    ///   alpha = 0.005  ->  0 0 0 0.005
    ///   alpha = 0.02   ->  1275000 1275000 1275000 0.02
    /// ```
    #[test]
    fn unpremultiply_dead_zone_damps_a_near_zero_alpha() {
        for alpha in [0.0, 0.003, 0.005, -0.005, 0.009] {
            let mut px = [100.0, 100.0, 100.0, alpha];
            unpremultiply(&mut px, 255.0);
            for (b, v) in px[..3].iter().enumerate() {
                assert!(
                    v.abs() < 1e-12,
                    "alpha {alpha} is inside the dead zone: band {b} came out \
                     {v}, want 0"
                );
            }
        }
        // Just outside the dead zone the full `max / alpha` factor applies,
        // amplification and all: 100 * 255 / 0.02 = 1275000.
        let mut px = [100.0, 100.0, 100.0, 0.02];
        unpremultiply(&mut px, 255.0);
        for (b, v) in px[..3].iter().enumerate() {
            assert!(
                (v - 1_275_000.0).abs() < 1e-3,
                "band {b} came out {v}, want vips' 1275000"
            );
        }
    }

    /// Regression for #604: the dead zone is an absolute `0.01` in whatever
    /// units the alpha band carries, never a fraction of `max`. Measured on the
    /// binary, `alpha = 0.02` on the same `(100, 100, 100, alpha)` pixel gives
    /// `5000` under scRGB (`max_alpha` 1), `1275000` under the 255 default and
    /// `327675008` under RGB16 (`max_alpha` 65535), and `alpha = 0.005` gives 0
    /// in all three. So the 16-bit carrier's dead zone is `0.01 / 65535` of
    /// full scale, not `0.01`.
    ///
    /// The RGB16 row is the one that moved with #664. It used to pin
    /// `327675000`, the exact quotient, and carry a note saying the binary
    /// "prints the float32 rounding" of it. That was the divergence, not a
    /// printing artefact: `OUT factor` is a `float`, so vips genuinely computes
    /// `327675008` and this pins the binary now rather than a model of it.
    #[test]
    fn unpremultiply_dead_zone_does_not_scale_with_max() {
        for (max, want) in [
            (1.0, 5000.0),
            (255.0, 1_275_000.0),
            (65535.0, 327_675_008.0),
        ] {
            let mut px = [100.0, 100.0, 100.0, 0.02];
            unpremultiply(&mut px, max);
            assert!(
                (px[0] - want).abs() < want * 1e-9,
                "max {max}: got {}, want {want}",
                px[0]
            );
            let mut px = [100.0, 100.0, 100.0, 0.005];
            unpremultiply(&mut px, max);
            assert!(px[0].abs() < 1e-12, "max {max}: 0.005 must stay damped");
        }
    }

    /// Regression for #604: the stored alpha is clipped to `0..=max`, and the
    /// factor is deliberately *not*, so an alpha overshoot and the colour
    /// overshoot that came with it still cancel. Both pinned on the binary:
    ///
    /// ```text
    ///   alpha = -0.5  ->  -51000 -51000 -51000 0
    ///   alpha = 300   ->  85 85 85 255
    /// ```
    #[test]
    fn unpremultiply_clips_the_stored_alpha_but_not_the_factor() {
        // Negative alpha outside the dead zone: factor 255 / -0.5 = -510, so
        // the colour flips sign rather than being clamped, and only the alpha
        // that is stored clips to 0.
        let mut px = [100.0, 100.0, 100.0, -0.5];
        unpremultiply(&mut px, 255.0);
        for (b, v) in px[..3].iter().enumerate() {
            assert!(
                (v + 51_000.0).abs() < 1e-6,
                "band {b} came out {v}, want vips' -51000"
            );
        }
        assert_eq!(px[3], 0.0, "a negative alpha must store as 0");

        // Alpha above max: factor 255 / 300, stored alpha clips to max.
        let mut px = [100.0, 100.0, 100.0, 300.0];
        unpremultiply(&mut px, 255.0);
        for (b, v) in px[..3].iter().enumerate() {
            assert!(
                (v - 85.0).abs() < 1e-9,
                "band {b} came out {v}, want vips' 85"
            );
        }
        assert_eq!(px[3], 255.0, "an alpha above max must store as max");
    }

    /// Regression for #604 through the float carrier, where the clip is
    /// observable end to end: `unpremultiply_from_float` writing back to
    /// `RgbaF32` must store the clipped alpha, not the raw one. On the unsigned
    /// carriers the sample store clamps anyway, which is why the guard has to
    /// live in the un-premultiply rather than being left to the writer.
    #[test]
    fn unpremultiply_from_float_stores_the_clipped_alpha() {
        let px: Vec<f32> = vec![
            100.0, 100.0, 100.0, -0.5, // undershoot
            100.0, 100.0, 100.0, 300.0, // overshoot
            100.0, 100.0, 100.0, 0.005, // dead zone
        ];
        let data: Vec<u8> = px.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let src = Raster::new(3, 1, PixelFormat::RgbaF32, data).unwrap();
        let out = unpremultiply_from_float(&src, PixelFormat::RgbaF32, 255.0).unwrap();
        assert_eq!(out.getpoint(0, 0)[3], 0.0, "undershoot clips to 0");
        assert_eq!(out.getpoint(1, 0)[3], 255.0, "overshoot clips to max");
        let damped = out.getpoint(2, 0);
        assert!(
            damped[..3].iter().all(|v| v.abs() < 1e-12),
            "dead-zone colour must be damped, got {damped:?}"
        );
    }

    /// The premultiply half of the pair carries the mirror-image guard
    /// (#604): the normalising factor is built from a **clipped** alpha while
    /// the alpha band is copied through **raw**. A float source with alpha
    /// above max therefore scales its colour by 1, not by `alpha / max`, and
    /// still hands the raw alpha to the resample so the round trip cancels.
    #[test]
    fn premultiply_to_float_clips_the_factor_and_keeps_the_raw_alpha() {
        let px: Vec<f32> = vec![100.0, 100.0, 100.0, 300.0, 100.0, 100.0, 100.0, -5.0];
        let data: Vec<u8> = px.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let src = Raster::new(2, 1, PixelFormat::RgbaF32, data).unwrap();
        let out = premultiply_to_float(&src, 255.0).unwrap();
        let over = out.getpoint(0, 0);
        assert!(
            (over[0] - 100.0).abs() < 1e-9,
            "an alpha above max must normalise to 1, got {}",
            over[0]
        );
        assert_eq!(over[3], 300.0, "the stored alpha stays raw");
        let under = out.getpoint(1, 0);
        assert!(
            under[0].abs() < 1e-12,
            "a negative alpha must normalise to 0, got {}",
            under[0]
        );
        assert_eq!(under[3], -5.0, "the stored alpha stays raw");
    }

    /// resize is honest about invalid scales.
    #[test]
    fn resize_bad_scale_is_typed_error() {
        let im = ramp_4x4();
        assert!(matches!(
            im.try_resize(0.0),
            Err(ResampleError::BadScale { .. })
        ));
        assert!(matches!(
            im.try_resize(f64::NAN),
            Err(ResampleError::BadScale { .. })
        ));
    }

    /// 16-bit and float formats resample through the same paths.
    #[test]
    fn shrink_gray16_and_float() {
        let mut data = Vec::new();
        for v in [1000u16, 2000, 3000, 4000] {
            data.extend_from_slice(&v.to_ne_bytes());
        }
        let im = Raster::new(2, 2, PixelFormat::Gray16, data).unwrap();
        let out = im.shrink(2.0, 2.0);
        assert_eq!(out.width(), 1);
        assert_eq!(out.height(), 1);
        assert_eq!(u16::from_ne_bytes([out.data()[0], out.data()[1]]), 2500);

        let mut fdata = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            fdata.extend_from_slice(&v.to_ne_bytes());
        }
        let fim = Raster::new(
            2,
            2,
            PixelFormat::FloatF32(core::num::NonZeroU16::new(1).unwrap()),
            fdata,
        )
        .unwrap();
        let fout = fim.shrink(2.0, 2.0);
        let got = f32::from_ne_bytes([
            fout.data()[0],
            fout.data()[1],
            fout.data()[2],
            fout.data()[3],
        ]);
        assert!((got - 2.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------
    // Thumbnail
    // -----------------------------------------------------------------

    /// A 290x442 portrait RGB raster, the aspect of the libvips `sample.jpg`
    /// fixture, so the thumbnail dimension checks reuse the vips oracle.
    fn portrait_290x442() -> Raster {
        let (w, h) = (290u32, 442u32);
        let mut data = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push((x % 256) as u8);
                data.push((y % 256) as u8);
                data.push(((x + y) % 256) as u8);
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /// A one-pixel black/white checkerboard: every 2x downscale sees equal
    /// black and white, so linear and gamma-space averages diverge sharply.
    fn checker(side: u32) -> Raster {
        let mut data = Vec::with_capacity((side * side * 3) as usize);
        for y in 0..side {
            for x in 0..side {
                let v = if (x + y) % 2 == 0 { 0u8 } else { 255u8 };
                data.extend_from_slice(&[v, v, v]);
            }
        }
        Raster::new(side, side, PixelFormat::Rgb8, data).unwrap()
    }

    /// The fit dimensions match the real `vips thumbnail` / `vipsthumbnail`
    /// oracle for the 290x442 aspect: a bare width fits a square box (the
    /// bound axis lands exactly), a width x height box fits inside it, and
    /// crop fills it.
    #[test]
    fn thumbnail_fit_matches_vips_oracle_dims() {
        let im = portrait_290x442();
        let dims = |w, h, crop| {
            let t = thumbnail_fit(&im, w, h, crop, ThumbSpace::Device).unwrap();
            (t.width(), t.height())
        };
        // Square boxes: the bound (larger) axis lands exactly on the size.
        assert_eq!(dims(100, None, false), (66, 100));
        assert_eq!(dims(128, None, false), (84, 128));
        assert_eq!(dims(442, None, false), (290, 442));
        // Rectangular boxes fit inside, preserving aspect.
        assert_eq!(dims(100, Some(300), false), (100, 152));
        assert_eq!(dims(300, Some(100), false), (66, 100));
        // Crop fills the box exactly.
        assert_eq!(dims(100, Some(300), true), (100, 300));
    }

    /// The bare-width thumbnail always lands the target on the bound axis,
    /// the libvips `for height in range(440, 1, -13)` invariant.
    #[test]
    fn thumbnail_height_series_is_exact() {
        let im = portrait_290x442();
        let mut h = 440u32;
        while h >= 2 {
            let t = thumbnail_fit(&im, h, None, false, ThumbSpace::Device).unwrap();
            assert_eq!(t.height(), h, "bound axis must land exactly for {h}");
            h = h.saturating_sub(13);
        }
    }

    /// A plain shrink preserves the mean, the `|avg_orig - avg_thumb| < 1`
    /// invariant of the ported thumbnail cell.
    #[test]
    fn thumbnail_preserves_average() {
        let im = portrait_290x442();
        let t = thumbnail_fit(&im, 100, None, false, ThumbSpace::Device).unwrap();
        assert!(
            (im.avg() - t.avg()).abs() < 1.0,
            "mean drifted: {}",
            t.avg()
        );
    }

    /// Reducing in linear light lifts the mean of a black/white pattern well
    /// above the gamma-space average, so the linear path is demonstrably not
    /// a plain reduce.
    #[test]
    fn thumbnail_linear_differs_from_naive() {
        let im = checker(64);
        let naive = thumbnail_fit(&im, 8, None, false, ThumbSpace::Device).unwrap();
        let linear = thumbnail_fit(&im, 8, None, false, ThumbSpace::Linear).unwrap();
        assert_eq!((linear.width(), linear.height()), (8, 8));
        assert_eq!(linear.format().channels(), 3);
        // Gamma-space average of 0 and 255 is ~127; linear-light average of
        // the same pair re-encodes to ~188.
        assert!(naive.avg() < 140.0, "naive avg {}", naive.avg());
        assert!(
            linear.avg() > naive.avg() + 30.0,
            "linear {} should exceed naive {} by a wide margin",
            linear.avg(),
            naive.avg()
        );
    }

    /// The file and buffer entry points decode the same bytes and agree, and
    /// the associated `thumbnail` resolves to this inherent method.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn thumbnail_file_and_buffer_agree() {
        let im = portrait_290x442();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("portrait.png");
        im.save(&path).unwrap();

        let by_file = Raster::thumbnail(&path, 100, None, false);
        assert_eq!((by_file.width(), by_file.height()), (66, 100));
        assert_eq!(by_file.format().channels(), 3);

        let buf = std::fs::read(&path).unwrap();
        let by_buf = Raster::thumbnail_buffer(&buf, 100);
        assert_eq!((by_buf.width(), by_buf.height()), (66, 100));
        assert!((by_file.avg() - by_buf.avg()).abs() < 1.0);

        // The in-memory instance form fits the same square box.
        let by_image = im.thumbnail_image(100);
        assert_eq!((by_image.width(), by_image.height()), (66, 100));
    }

    /// The ICC export path imports through the attached profile and exports
    /// to the built-in sRGB profile; a source already in sRGB round-trips
    /// close to identity, exercising the whole import/reduce/export machine.
    #[test]
    fn thumbnail_icc_srgb_roundtrips() {
        let mut im = portrait_290x442();
        im.set_icc_profile(&builtin_srgb_profile().unwrap());
        let t = thumbnail_fit(&im, 442, None, false, ThumbSpace::IccSrgb).unwrap();
        assert_eq!((t.width(), t.height()), (290, 442));
        assert_eq!(t.format().channels(), 3);
        // sRGB -> Lab(D50) -> sRGB is near identity at 8-bit.
        assert!(
            (im.avg() - t.avg()).abs() < 4.0,
            "sRGB ICC round-trip drifted: {} vs {}",
            im.avg(),
            t.avg()
        );
    }

    /// A zero target is a typed error, not a panic in the fit math.
    #[test]
    fn thumbnail_zero_size_is_typed_error() {
        let im = portrait_290x442();
        assert!(matches!(
            im.try_thumbnail_image(0),
            Err(ThumbnailError::BadSize { size: 0 })
        ));
    }

    /// An unknown output profile is reported, not silently treated as sRGB.
    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn thumbnail_unknown_profile_is_typed_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("p.png");
        portrait_290x442().save(&path).unwrap();
        assert!(matches!(
            Raster::try_thumbnail_with_profile(&path, 100, "adobe-rgb"),
            Err(ThumbnailError::UnknownProfile { .. })
        ));
    }

    // -----------------------------------------------------------------
    // Nohalo / LBB interpolators, pinned to a real libvips 8.18.3 oracle
    // -----------------------------------------------------------------

    /// The 16x16 single-band fixture the libvips oracle affines: sharp
    /// 4x4 block-parity edges so the nohalo minmod slopes and the LBB
    /// range limiters both activate. The generator formula is
    /// `base = (x*17 + y*29) % 256; v = 255 - base` on the block-parity
    /// squares, `base` elsewhere.
    fn oracle_16x16() -> Raster {
        let mut data = vec![0u8; 16 * 16];
        for y in 0..16usize {
            for x in 0..16usize {
                let base = ((x * 17 + y * 29) % 256) as u8;
                let bx = (x / 4) % 2;
                let by = (y / 4) % 2;
                data[y * 16 + x] = if bx ^ by != 0 { 255 - base } else { base };
            }
        }
        Raster::new(16, 16, PixelFormat::Gray8, data).unwrap()
    }

    /// The interior 12x12 crop `[6, 6, 12, 12]` of the 28x28 affine of the
    /// oracle fixture by `[1.5, 0.25, -0.25, 1.5]` matches real libvips
    /// 8.18.3 byte for byte, for every interpolator. Pinning nearest,
    /// bilinear, and bicubic confirms the affine geometry and rounding
    /// agree with libvips; pinning nohalo and lbb confirms the two
    /// minmod-subdivision ports are faithful. The interior crop keeps the
    /// stencils off the image edge so the comparison is pure kernel math.
    #[test]
    fn affine_interpolators_match_libvips_oracle() {
        // Captured with: vips affine in.pgm out.v "1.5 0.25 -0.25 1.5"
        //   --interpolate INTERP  (libvips 8.18.3), interior crop [6,6,12,12].
        #[rustfmt::skip]
        let oracle: [(&str, [u8; 144]); 5] = [
            ("nearest", [
                80, 129, 129, 112, 95, 95, 78, 194, 194, 211, 1, 1,
                109, 129, 129, 112, 66, 66, 49, 223, 223, 240, 1, 1,
                138, 100, 100, 83, 66, 66, 49, 223, 3, 242, 242, 225,
                138, 100, 184, 201, 201, 218, 235, 235, 3, 242, 242, 196,
                88, 88, 184, 201, 201, 247, 8, 8, 230, 213, 213, 196,
                59, 59, 213, 230, 230, 247, 8, 8, 201, 201, 184, 167,
                59, 59, 242, 242, 3, 20, 20, 37, 201, 201, 184, 138,
                47, 30, 242, 242, 3, 49, 49, 66, 172, 172, 155, 138,
                18, 1, 15, 15, 32, 49, 49, 66, 112, 112, 129, 129,
                18, 1, 211, 211, 194, 194, 177, 160, 160, 112, 129, 129,
                10, 27, 27, 211, 194, 194, 148, 131, 131, 141, 158, 158,
                39, 56, 56, 182, 165, 165, 148, 131, 131, 170, 170, 187,
            ]),
            ("bilinear", [
                123, 122, 108, 94, 79, 65, 146, 218, 232, 126, 10, 41,
                118, 105, 91, 77, 67, 76, 129, 156, 212, 195, 152, 170,
                123, 129, 138, 151, 169, 192, 149, 3, 165, 231, 217, 202,
                122, 184, 198, 212, 226, 175, 116, 143, 199, 214, 200, 185,
                103, 188, 215, 229, 243, 122, 42, 177, 211, 197, 183, 168,
                76, 185, 187, 115, 101, 62, 37, 148, 194, 180, 166, 151,
                40, 173, 157, 13, 21, 36, 50, 128, 177, 163, 149, 136,
                17, 66, 70, 31, 38, 55, 76, 120, 150, 143, 141, 144,
                7, 33, 84, 106, 123, 135, 143, 139, 112, 126, 140, 154,
                19, 82, 211, 197, 183, 169, 154, 140, 132, 143, 157, 171,
                33, 80, 176, 180, 166, 152, 137, 128, 144, 160, 174, 188,
                50, 77, 150, 163, 149, 135, 120, 108, 147, 177, 191, 205,
            ]),
            ("bicubic", [
                126, 124, 107, 89, 67, 46, 150, 249, 248, 117, 0, 9,
                118, 95, 82, 71, 63, 60, 124, 158, 228, 209, 152, 180,
                124, 122, 133, 148, 172, 225, 155, 3, 162, 253, 235, 228,
                117, 184, 206, 226, 247, 196, 105, 145, 211, 216, 200, 189,
                96, 193, 234, 248, 255, 111, 10, 198, 239, 197, 182, 169,
                69, 202, 210, 109, 96, 46, 33, 155, 207, 180, 165, 149,
                38, 201, 182, 0, 0, 14, 41, 131, 188, 164, 148, 135,
                10, 61, 54, 16, 33, 54, 69, 120, 152, 141, 138, 141,
                2, 17, 64, 105, 129, 143, 151, 141, 112, 121, 137, 153,
                0, 79, 211, 214, 195, 180, 165, 143, 128, 140, 157, 171,
                13, 78, 191, 194, 168, 152, 136, 127, 144, 161, 174, 188,
                44, 73, 156, 171, 149, 134, 117, 107, 147, 181, 193, 208,
            ]),
            ("nohalo", [
                127, 125, 106, 89, 70, 54, 144, 232, 239, 120, 2, 21,
                121, 97, 83, 72, 64, 61, 125, 160, 221, 200, 156, 181,
                126, 121, 130, 144, 166, 205, 159, 3, 168, 240, 226, 211,
                118, 184, 202, 220, 237, 181, 111, 150, 208, 217, 200, 185,
                97, 191, 222, 238, 247, 122, 23, 196, 220, 196, 183, 169,
                69, 195, 193, 111, 99, 56, 33, 155, 200, 180, 163, 144,
                38, 184, 169, 4, 11, 29, 46, 132, 184, 164, 148, 134,
                6, 60, 64, 21, 34, 54, 73, 122, 147, 137, 137, 142,
                2, 22, 81, 112, 131, 143, 151, 138, 112, 119, 136, 150,
                9, 85, 211, 206, 191, 177, 160, 143, 126, 137, 156, 171,
                26, 79, 188, 188, 167, 152, 133, 126, 143, 159, 174, 188,
                50, 75, 156, 167, 148, 135, 117, 108, 145, 179, 194, 208,
            ]),
            ("lbb", [
                126, 124, 106, 89, 69, 47, 154, 233, 238, 116, 2, 19,
                117, 96, 81, 71, 63, 57, 131, 161, 226, 203, 161, 188,
                125, 122, 134, 148, 172, 215, 164, 3, 176, 238, 228, 216,
                118, 184, 206, 222, 238, 184, 104, 154, 214, 215, 200, 189,
                95, 193, 233, 238, 246, 131, 19, 201, 224, 196, 182, 170,
                68, 198, 197, 107, 93, 53, 33, 157, 208, 180, 165, 149,
                38, 196, 181, 4, 9, 17, 40, 132, 188, 164, 148, 135,
                10, 65, 60, 19, 34, 53, 69, 119, 152, 141, 137, 141,
                2, 20, 70, 108, 130, 143, 152, 139, 112, 121, 137, 153,
                8, 85, 211, 203, 194, 180, 165, 145, 128, 140, 157, 171,
                17, 77, 187, 190, 167, 152, 135, 126, 144, 161, 174, 188,
                44, 73, 156, 172, 149, 135, 117, 107, 148, 181, 194, 208,
            ]),
        ];

        let im = oracle_16x16();
        for (name, expected) in oracle {
            let out = im.affine([1.5, 0.25, -0.25, 1.5], name);
            assert_eq!(out.width(), 28, "{name} width");
            assert_eq!(out.height(), 28, "{name} height");
            let interior = out.extract_area(6, 6, 12, 12);
            let got = interior.data();
            let (mismatches, worst) =
                got.iter()
                    .zip(expected.iter())
                    .fold((0usize, 0u8), |(n, worst), (&a, &b)| {
                        let d = a.abs_diff(b);
                        (n + usize::from(d != 0), worst.max(d))
                    });
            // nohalo, lbb and bicubic reproduce libvips byte for byte
            // (0 of 144). nohalo and lbb always did: both compute their
            // Hermite coefficients directly, just like `nohalo.cpp` and
            // `lbb.cpp`. Bicubic took two changes to get there. It was 60
            // bytes at delta 3 while this module evaluated Catmull-Rom at
            // the exact sub-pixel offset and libvips read its 65-entry
            // table; rounding the offset the same way (#668) halved the
            // count and took the worst delta to one LSB; reading the
            // coefficients out of `vips_bicubic_matrixi` as 12-bit fixed
            // point and accumulating them as integers, which is what
            // `bicubic_unsigned_int_tab` does on a `uchar` carrier, closes
            // the rest (#704).
            //
            // The other two kernels still only bound the affine geometry.
            // Nearest differs at 2 equidistant-neighbour ties (a whole-pixel
            // swap, so a large delta but the adjacent sample). Bilinear
            // differs in 1 byte at delta 1, and that one is *not* a rounding
            // tie: `SWITCH_INTERPOLATE` sends a `uchar` raster to
            // `BILINEAR_INT`, whose four weights are 12-bit fixed point too,
            // and modelling that reproduces vips exactly where modelling a
            // tie does not. Issue #733 carries the measurement.
            let (allowed_count, allowed_delta) = match name {
                "nohalo" | "lbb" => (0, 0),
                "bilinear" => (1, 1),
                "nearest" => (2, u8::MAX),
                "bicubic" => (0, 0),
                _ => unreachable!(),
            };
            assert!(
                mismatches <= allowed_count && worst <= allowed_delta,
                "{name} differs from the libvips oracle in {mismatches} bytes \
                 (worst delta {worst}); expected at most {allowed_count} bytes, delta {allowed_delta}"
            );
        }
    }

    // -----------------------------------------------------------------
    // Bicubic on the integer carriers: vips picks a different arithmetic
    // per band format, and the four fixtures below pin all three of them
    // (issue #704).
    // -----------------------------------------------------------------

    /// The 12x12 fixture the four bicubic carrier tests share, generated by
    /// formula so the input never needs pinning: sample `i` is
    /// `(i * 53 + 17) % 251` for the unsigned byte carriers and
    /// `(i * 3719 + 977) % 65413` for the 16-bit one, both coprime strides
    /// over a prime modulus so no row or column repeats.
    fn carrier_fixture(format: PixelFormat) -> Raster {
        let n = 12 * 12 * format.channels();
        let data: Vec<u8> = match format {
            PixelFormat::Gray16 => (0..n)
                .flat_map(|i| (((i * 3719 + 977) % 65413) as u16).to_ne_bytes())
                .collect(),
            PixelFormat::FloatF32(_) => (0..n)
                .flat_map(|i| (((i * 37 + 11) % 251) as f32).to_ne_bytes())
                .collect(),
            _ => (0..n).map(|i| ((i * 53 + 17) % 251) as u8).collect(),
        };
        Raster::new(12, 12, format, data).unwrap()
    }

    /// The three-band float twin of [`carrier_fixture`], sample `i` being
    /// `(i * 37 + 11) % 251` as an `f32`, for the float accumulation pins
    /// (issue #705). Three bands rather than one so the per-band loop is
    /// exercised at the same time.
    fn float_carrier_fixture() -> Raster {
        let data: Vec<u8> = (0..12 * 12 * 3usize)
            .flat_map(|i| (((i * 37 + 11) % 251) as f32).to_ne_bytes())
            .collect();
        Raster::new(
            12,
            12,
            PixelFormat::FloatF32(core::num::NonZeroU16::new(3).unwrap()),
            data,
        )
        .unwrap()
    }

    /// The interior 6x6 crop at `[4, 4]` of the 18x15 affine, as flat
    /// samples. Every stencil that crop reads spans input x 1..8 and
    /// y 1..9, so the whole comparison is kernel arithmetic with no
    /// [`Extend`] rule anywhere in it.
    fn carrier_crop(r: &Raster) -> Vec<u8> {
        assert_eq!(
            (r.width(), r.height()),
            (18, 15),
            "carrier fixture output size"
        );
        r.extract_area(4, 4, 6, 6).data().to_vec()
    }

    fn carrier_crop_u16(r: &Raster) -> Vec<u16> {
        carrier_crop(r)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect()
    }

    fn carrier_crop_f32(r: &Raster) -> Vec<f32> {
        carrier_crop(r)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect()
    }

    /// Count the mismatches and the worst absolute delta between two
    /// integer sample runs.
    fn carrier_diff(got: &[u16], want: &[u16]) -> (usize, u16) {
        assert_eq!(got.len(), want.len(), "sample count");
        got.iter()
            .zip(want.iter())
            .fold((0usize, 0u16), |(n, worst), (&a, &b)| {
                let d = a.abs_diff(b);
                (n + usize::from(d != 0), worst.max(d))
            })
    }

    /// Issue #704. On a `uchar` carrier `vips_interpolate_bicubic_interpolate`
    /// dispatches to `bicubic_unsigned_int_tab`, which reads
    /// `vips_bicubic_matrixi`: the Catmull-Rom coefficients themselves
    /// truncated to 12-bit fixed point (`VIPS_INTERPOLATE_SCALE`, 4096), then
    /// accumulated as integers a row at a time with a fixed-point round after
    /// each row and after the column combine. #668 put the *offset* on the
    /// same 1/64 grid vips uses and left the coefficients in `f64`, which is
    /// the whole of what is left.
    ///
    /// Measured on 8.18.6 as
    /// `vips affine in.v out.v "1.3 0.2 -0.15 1.1" --interpolate bicubic`
    /// over the three-band byte fixture, interior crop `[4, 4, 6, 6]`.
    /// Before this change libviprs missed 25 of those 108 bytes, every one by
    /// exactly 1.
    #[test]
    fn affine_bicubic_reads_the_vips_fixed_point_table_on_a_uchar_carrier() {
        #[rustfmt::skip]
        let want: [u8; 108] = [
        58, 138, 159, 141, 75, 87, 162, 65, 131, 44, 113, 182, 97, 153, 64, 116, 192, 87,
        141, 193, 66, 164, 225, 111, 60, 130, 195, 57, 111, 77, 106, 180, 73, 127, 130, 122,
        158, 213, 113, 74, 115, 195, 16, 68, 90, 118, 184, 42, 124, 140, 105, 156, 57, 134,
        129, 114, 177, 245, 51, 95, 172, 191, 11, 88, 147, 110, 130, 58, 140, 197, 76, 129,
        215, 57, 86, 166, 183, 0, 63, 145, 120, 125, 47, 121, 202, 47, 100, 91, 156, 217,
        179, 146, 196, 61, 127, 143, 129, 34, 80, 194, 45, 98, 100, 166, 219, 30, 88, 127,
        ];
        let out = carrier_fixture(PixelFormat::Rgb8).affine([1.3, 0.2, -0.15, 1.1], "bicubic");
        assert_eq!(
            carrier_crop(&out),
            want,
            "uchar bicubic against vips 8.18.6"
        );
    }

    /// Issue #736. `vips_affine_build` converts the background exactly once,
    /// before any resampling happens:
    ///
    /// ```c
    /// affine->ink = vips__vector_to_ink(class->nickname, in,
    ///     VIPS_AREA(affine->background)->data, NULL,
    ///     VIPS_AREA(affine->background)->n);
    /// ```
    ///
    /// `vips__vector_to_ink` builds a one-pixel image of the doubles and casts
    /// it to the input's band format, and `vips_cast` clips and then truncates
    /// toward zero (`cast.c:237`, and the file's own header note: "now does
    /// floor(), not rint()"). So every tap past the edge, and every output
    /// pixel outside the transformed input, is already an integer inside the
    /// carrier's range by the time the interpolator sees it. This module
    /// carried the raw `f64` into both.
    ///
    /// Verified on 8.18.6 for every interpolator on both integer carriers,
    /// `vips affine in.v out.v "1.3 0.2 -0.15 1.1" --interpolate INTERP
    /// --extend background --background BG` over a 6x6 constant:
    ///
    /// ```text
    ///           200.7 == 200   200.7 != 201   -30.4 == 0   over-range == max
    /// uchar     yes            yes            yes          400.9 == 255
    /// ushort    yes            yes            yes          70000.9 == 65535
    /// ```
    ///
    /// all five interpolators, all four columns. The `200.7 != 201` column is
    /// the positive control: without it the first column would also pass if the
    /// ink were being ignored altogether.
    #[test]
    fn affine_casts_the_background_ink_to_an_integer_carrier() {
        for (format, value, over, ceiling) in [
            (PixelFormat::Gray8, 100u16, 400.9, 255.0),
            (PixelFormat::Gray16, 25000u16, 70000.9, 65535.0),
        ] {
            for name in ["nearest", "bilinear", "bicubic", "nohalo", "lbb"] {
                let bytes: Vec<u8> = if format == PixelFormat::Gray8 {
                    vec![value as u8; 36]
                } else {
                    (0..36).flat_map(|_| value.to_ne_bytes()).collect()
                };
                let im = Raster::new(6, 6, format, bytes).unwrap();
                let interp = Interpolator::from_name(name).unwrap();
                let shift = |background: f64| -> Vec<u8> {
                    im.try_affine_with(
                        [1.3, 0.2, -0.15, 1.1],
                        interp,
                        AffineOptions {
                            extend: Extend::Background,
                            background,
                            ..AffineOptions::default()
                        },
                    )
                    .unwrap()
                    .data()
                    .to_vec()
                };
                assert_eq!(shift(200.7), shift(200.0), "{format:?} {name}: truncates");
                assert_ne!(
                    shift(200.7),
                    shift(201.0),
                    "{format:?} {name}: truncates rather than rounding"
                );
                assert_eq!(shift(-30.4), shift(0.0), "{format:?} {name}: clips low");
                assert_eq!(shift(over), shift(ceiling), "{format:?} {name}: clips high");
            }
        }
    }

    /// Issue #736, the float carrier. `vips_cast` to `VIPS_FORMAT_FLOAT` is
    /// still a narrowing and there is no clip on that arm, so vips reads
    /// `f32(200.7)` where this module read the `f64`. Measured on 8.18.6 for
    /// all five interpolators: `--background 200.7` and
    /// `--background 200.6999969482422` produce identical output, `200.7` and
    /// `200` do not, and an ink below zero is **not** clipped the way it is on
    /// an integer carrier.
    #[test]
    fn affine_narrows_the_background_ink_on_a_float_carrier() {
        let data: Vec<u8> = (0..36).flat_map(|_| 100.0f32.to_ne_bytes()).collect();
        let im = Raster::new(
            6,
            6,
            PixelFormat::FloatF32(core::num::NonZeroU16::new(1).unwrap()),
            data,
        )
        .unwrap();
        for name in ["nearest", "bilinear", "bicubic", "nohalo", "lbb"] {
            let interp = Interpolator::from_name(name).unwrap();
            let shift = |background: f64| -> Vec<f32> {
                float_samples(
                    &im.try_affine_with(
                        [1.3, 0.2, -0.15, 1.1],
                        interp,
                        AffineOptions {
                            extend: Extend::Background,
                            background,
                            ..AffineOptions::default()
                        },
                    )
                    .unwrap(),
                )
            };
            assert_eq!(
                shift(200.7),
                shift(f64::from(200.7f32)),
                "{name}: the ink narrows to f32"
            );
            assert_ne!(shift(200.7), shift(200.0), "{name}: and only to f32");
            assert_ne!(
                shift(-30.4),
                shift(0.0),
                "{name}: a float carrier has no clip on the cast"
            );
        }
    }

    /// Issue #736, the anchors. The equivalences above say the ink is converted
    /// the same way vips converts it; these say the converted value is the one
    /// vips uses. Row 0 of the 9x8 output for five
    /// `(carrier, interpolator, background)` cells, measured on 8.18.6. The
    /// last sample of each row is an output pixel outside the transformed
    /// input, which `vips_affine_gen` paints with `affine->ink` rather than
    /// interpolating, so each row covers both sites the cast reaches.
    ///
    /// No `bilinear` cell here, and that is not an oversight: on an integer
    /// carrier `SWITCH_INTERPOLATE` sends bilinear to `BILINEAR_INT` and its
    /// four weights are 12-bit fixed point, so the row still misses the binary
    /// by a byte for a reason that has nothing to do with the ink (#733). The
    /// equivalences above do cover bilinear, because they compare two libviprs
    /// runs against each other and that divergence cancels.
    #[test]
    fn affine_background_ink_rows_match_the_oracle() {
        let gray8 = |v: u8| Raster::new(6, 6, PixelFormat::Gray8, vec![v; 36]).unwrap();
        let gray16 = |v: u16| {
            Raster::new(
                6,
                6,
                PixelFormat::Gray16,
                (0..36).flat_map(|_| v.to_ne_bytes()).collect::<Vec<u8>>(),
            )
            .unwrap()
        };
        let shift = |im: &Raster, name: &str, background: f64| -> Raster {
            im.try_affine_with(
                [1.3, 0.2, -0.15, 1.1],
                Interpolator::from_name(name).unwrap(),
                AffineOptions {
                    extend: Extend::Background,
                    background,
                    ..AffineOptions::default()
                },
            )
            .unwrap()
        };
        let out = shift(&gray8(100), "bicubic", 200.7);
        assert_eq!((out.width(), out.height()), (9, 8), "output size");
        assert_eq!(
            &out.data()[..9],
            &[193u8, 183, 173, 160, 148, 135, 118, 146, 200],
            "uchar bicubic, background 200.7"
        );
        assert_eq!(
            &shift(&gray8(100), "lbb", 400.9).data()[..9],
            &[250u8, 237, 219, 196, 173, 149, 129, 165, 255],
            "uchar lbb, background 400.9"
        );
        let u16_row = |r: &Raster| -> Vec<u16> {
            r.data().as_chunks::<2>().0[..9]
                .iter()
                .map(|b| u16::from_ne_bytes(*b))
                .collect()
        };
        assert_eq!(
            u16_row(&shift(&gray16(25000), "nearest", 200.7)),
            vec![200u16; 9],
            "ushort nearest, background 200.7"
        );
        assert_eq!(
            u16_row(&shift(&gray16(25000), "nohalo", -30.4)),
            vec![1108u16, 3551, 6799, 10163, 13034, 16155, 19701, 13081, 0],
            "ushort nohalo, background -30.4"
        );
        let data: Vec<u8> = (0..36).flat_map(|_| 100.0f32.to_ne_bytes()).collect();
        let f = Raster::new(
            6,
            6,
            PixelFormat::FloatF32(core::num::NonZeroU16::new(1).unwrap()),
            data,
        )
        .unwrap();
        assert_eq!(
            &float_samples(&shift(&f, "bicubic", 200.7))[..9],
            &[
                193.14478f32,
                183.42047,
                173.28777,
                160.13597,
                148.38359,
                134.8123,
                118.18186,
                146.40132,
                200.7
            ],
            "float bicubic, background 200.7"
        );
    }

    /// Issue #704 and #736. The taps `bicubic_unsigned_int_tab` reads are the
    /// stored samples of the embedded image, so an out-of-band tap is already
    /// an integer inside the carrier's range by the time the interpolator sees
    /// it: `vips_affine_build` converts the background once through
    /// `vips__vector_to_ink`, which casts to the input's band format, and
    /// `vips_cast` clips and then truncates toward zero (`cast.c:237`). The
    /// fixed-point path does that conversion per tap.
    ///
    /// Measured on 8.18.6 with a 6x6 constant-100 `uchar` raster,
    /// `vips affine in.v out.v "1.3 0 0 1.1" --interpolate bicubic
    /// --extend background`: `--background 200.7` produces the same 56 bytes
    /// as `--background 200`, `-30.4` the same as `0`, and `400.9` the same as
    /// `255`. The positive control is that `200.7` and `201` do **not** agree,
    /// so the equivalence is about the conversion and not about the ink being
    /// ignored.
    ///
    /// The matrix is a pure scale on purpose. A sheared one leaves output
    /// pixels outside the transformed input entirely, and those are painted by
    /// `try_affine_with` rather than by the interpolator, from an ink that is
    /// still the raw `f64`. That second site is #736's, not this path's.
    ///
    /// The other four interpolators and the other two carriers still read the
    /// raw `f64` here, worth up to 75 of 255 on a byte carrier; #736 carries
    /// the whole table and generalises the conversion.
    #[test]
    fn affine_bicubic_casts_the_background_ink_to_the_carrier() {
        let im = Raster::new(6, 6, PixelFormat::Gray8, vec![100u8; 36]).unwrap();
        let shift = |background: f64| -> Vec<u8> {
            im.try_affine_with(
                [1.3, 0.0, 0.0, 1.1],
                Interpolator::Bicubic,
                AffineOptions {
                    extend: Extend::Background,
                    background,
                    ..AffineOptions::default()
                },
            )
            .unwrap()
            .data()
            .to_vec()
        };
        assert_eq!(shift(200.7), shift(200.0), "fractional ink truncates");
        assert_ne!(
            shift(200.7),
            shift(201.0),
            "and truncates rather than rounding"
        );
        assert_eq!(shift(-30.4), shift(0.0), "ink below the carrier clips to 0");
        assert_eq!(
            shift(400.9),
            shift(255.0),
            "ink above the carrier clips to 255"
        );
        // The anchor: row 0 of the 8x7 output at `--background 200.7` on
        // vips 8.18.6, so the four equivalences above sit on the binary and
        // not only on each other.
        assert_eq!(
            &shift(200.7)[..8],
            &[100u8, 98, 100, 100, 100, 100, 93, 136],
            "row 0 against vips 8.18.6"
        );
    }

    /// Issue #705. `bicubic_float_tab<float>` reaches `bicubic_float<T>` with
    /// `T = float`, and every row goes through `cubic_float<T>`, whose *return
    /// type is `T`*:
    ///
    /// ```c
    /// template <typename T>
    /// static T inline cubic_float(const T one, ..., const double *cx)
    /// {
    ///     return cx[0] * one + cx[1] * two + cx[2] * thr + cx[3] * fou;
    /// }
    /// ```
    ///
    /// The products and the sum are `double` because the coefficients are, and
    /// then the `return` narrows to `float`. So on a float carrier vips rounds
    /// **each of the four row sums** to `f32` before it combines them, and
    /// rounds the combine to `f32` as well.
    ///
    /// The accumulation *order* is not the cause, which is the whole point of
    /// this test existing separately from the ordering change. Modelling flat
    /// 16-term `f64` and row-then-column `f64` against the same binary gives
    /// **bit-identical** output, 0 of 1764 samples apart on a random 24x24, and
    /// both miss vips in the same 356 of 1764 by the same 1.5259e-05. Adding
    /// the per-row `f32` narrowing takes it to 0 of 1764.
    ///
    /// Measured on 8.18.6 as `vips affine in.v out.v "1.3 0.2 -0.15 1.1"
    /// --interpolate bicubic` over a three-band float ramp, interior crop
    /// `[4, 4, 6, 6]`, and asserted **exactly**: this is a bit-for-bit pin, not
    /// a tolerance. Before this change 16 of the 108 samples were wrong.
    #[test]
    fn affine_bicubic_rounds_each_row_sum_to_f32_on_a_float_carrier() {
        #[rustfmt::skip]
        let want: [f32; 108] = [
            221.05179, 60.94549, 90.7095, 171.81142, 106.03813, 107.21296,
            148.8833, 157.84538, 105.36776, 131.0, 26.8125, 63.8125,
            122.80959, 152.21906, 200.63866, 94.56327, 145.91641, 141.46187,
            117.17962, 160.57845, 114.42711, 97.89925, 148.53372, 117.20308,
            141.41801, 179.07013, 12.738264, 55.36105, 114.84537, 144.2059,
            168.30489, 111.80183, 160.67693, 163.968, 74.43494, 122.49891,
            88.708, 127.681114, 178.83788, 137.05281, 162.68243, 200.89423,
            9.306032, 45.70137, 97.95593, 109.54341, 163.6745, 149.838,
            178.50795, 118.677635, 94.25602, 143.57538, 54.46303, 103.87227,
            151.52507, 110.30661, 161.0822, 224.8973, 15.140279, 52.72461,
            103.42076, 106.68581, 158.4583, 152.80048, 201.55286, 77.08958,
            128.2253, 173.6129, 52.531086, 74.72636, 123.54618, 178.26077,
            183.94586, 211.92632, 47.876602, 69.16237, 79.52507, 107.24806,
            139.33195, 178.72528, 223.97078, 105.58892, 136.03918, 177.21284,
            30.985691, 74.062935, 121.39275, 157.12605, 198.74245, 175.05026,
            72.24145, 109.478065, 55.602993, 87.696976, 143.34001, 187.84433,
            204.26207, 89.05573, 133.19304, 207.79547, 44.639683, 91.34048,
            106.67966, 150.49532, 168.25237, 192.1947, 231.11996, 42.528263,
        ];
        let out = float_carrier_fixture().affine([1.3, 0.2, -0.15, 1.1], "bicubic");
        assert_eq!(
            carrier_crop_f32(&out),
            want,
            "float bicubic against vips 8.18.6"
        );
    }

    /// Issue #705, the third guard: the plain `ushort` carrier must **not**
    /// narrow. `bicubic_unsigned_int32_tab` calls `bicubic_float<double>`, so
    /// `cubic_float` returns `double` there and nothing rounds between the row
    /// sums and the combine.
    ///
    /// No oracle can arbitrate that one head on. An `f32` ulp at 16-bit
    /// magnitudes is about 0.004, so it only moves a sample sitting that close
    /// to a rounding boundary, and vips truncates its own `ushort` store where
    /// this module rounds half up (#732), so the binary quantises the two
    /// candidate answers differently anyway. What *can* arbitrate it is the
    /// float carrier, which is pinned bit for bit against 8.18.6 by
    /// [`affine_bicubic_rounds_each_row_sum_to_f32_on_a_float_carrier`]: run
    /// the same numbers through both carriers and they have to **disagree**
    /// exactly where the narrowing bites. Giving `ushort` the narrowing makes
    /// them agree, and that is the mutation this catches.
    ///
    /// The fixture is `(i * 7907 + 13) % 64007`, picked out of a search over
    /// 330 candidates because one of its 270 output samples lands within an
    /// `f32` ulp of a rounding boundary *and* moves under narrowing the rows
    /// alone. That second condition matters: my first fixture only moved when
    /// the column combine narrowed, so it let a mutation that narrows only the
    /// four row sums through, which is exactly the arm this is meant to guard.
    ///
    /// [`affine_bicubic_rounds_each_row_sum_to_f32_on_a_float_carrier`]: self::affine_bicubic_rounds_each_row_sum_to_f32_on_a_float_carrier
    #[test]
    fn affine_bicubic_keeps_full_f64_rows_on_a_ushort_carrier() {
        let values: Vec<u32> = (0..12 * 12u32).map(|i| (i * 7907 + 13) % 64007).collect();
        let as_u16 = Raster::new(
            12,
            12,
            PixelFormat::Gray16,
            values
                .iter()
                .flat_map(|v| (*v as u16).to_ne_bytes())
                .collect(),
        )
        .unwrap()
        .affine([1.3, 0.2, -0.15, 1.1], "bicubic");
        let as_f32 = Raster::new(
            12,
            12,
            PixelFormat::FloatF32(core::num::NonZeroU16::new(1).unwrap()),
            values
                .iter()
                .flat_map(|v| (*v as f32).to_ne_bytes())
                .collect(),
        )
        .unwrap()
        .affine([1.3, 0.2, -0.15, 1.1], "bicubic");

        let got: Vec<u16> = as_u16
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        // The float result quantised by this module's own writer rule, so the
        // only thing left between the two runs is the accumulation.
        let via_float: Vec<u16> = float_samples(&as_f32)
            .iter()
            .map(|v| (f64::from(*v) + 0.5).floor().clamp(0.0, 65535.0) as u16)
            .collect();
        assert_eq!(got.len(), 18 * 15, "output size");

        let differ: Vec<usize> = (0..got.len()).filter(|&i| got[i] != via_float[i]).collect();
        assert_eq!(
            differ,
            vec![73],
            "the two carriers must disagree exactly where the f32 narrowing bites"
        );
        assert_eq!((got[73], via_float[73]), (38531, 38532));
    }

    /// Issue #705, the premultiplied half. `vips_affine_build` premultiplies
    /// into a **FLOAT** image whenever `vips_image_hasalpha()`, so an alpha
    /// raster takes `bicubic_float_tab<float>` and the per-row narrowing
    /// whatever its stored depth was. Without a case here the `premultiply`
    /// arm of `bicubic_narrows_rows` has nothing behind it: an 8-bit output
    /// quantum swallows an `f32` ulp whole, so an `Rgba8` fixture cannot see
    /// this at all (measured: 0 samples move over 40 random 12x12 `Rgba8`
    /// rasters, against 28 of 40 for `Rgba16`).
    ///
    /// The oracle is `premultiply | affine --premultiplied | unpremultiply` on
    /// 8.18.6 with `--max-alpha 65535`, read back as FLOAT and quantised with
    /// this module's own round-half-up, for the reason
    /// `resize_unsigned_bracket_matches_the_vips_oracle_on_varying_data` gives:
    /// `vips_cast` truncates a float toward zero, so casting the oracle to
    /// `ushort` would ask a different question. Read that way the whole 12x10
    /// output agrees in **480 of 480** samples with the narrowing and 477 of
    /// 480 without it.
    ///
    /// The three that separate them are output pixel `(0, 7)`, bands 0 to 2,
    /// and they are on the rounding boundary, which is the only place an `f32`
    /// ulp at 50000 can reach:
    ///
    /// ```text
    /// vips float 56743.50390625  ->  56744   without the narrowing 56743
    /// vips float 51235.5         ->  51236   without the narrowing 51235
    /// vips float 45727.5         ->  45728   without the narrowing 45727
    /// ```
    ///
    /// Row 7 of the output is pinned rather than only those three, so the test
    /// says what the row is and not just where it bends.
    #[test]
    fn affine_bicubic_narrows_the_rows_when_alpha_premultiplies_to_float() {
        let data: Vec<u8> = (0..8 * 8 * 4usize)
            .flat_map(|i| (((i * 60013 + 977) % 65521) as u16).to_ne_bytes())
            .collect();
        #[rustfmt::skip]
        let want_row7: [u16; 48] = [
            56744, 51236, 45728, 3151, 54961, 49453, 43945, 27834,
            24906, 19398, 13890, 36248, 17693, 12185, 6677, 45305,
            23153, 17645, 12137, 60075, 57805, 52297, 46789, 38234,
            52224, 46716, 41208, 20211, 14154, 8646, 2902, 36135,
            15458, 9950, 6030, 58270, 41591, 36083, 50625, 47886,
            46338, 40830, 48301, 31296, 59024, 53516, 37023, 12280,
        ];
        let out = Raster::new(8, 8, PixelFormat::Rgba16, data)
            .unwrap()
            .affine([1.3, 0.2, -0.15, 1.1], "bicubic");
        assert_eq!((out.width(), out.height()), (12, 10), "output size");
        let got: Vec<u16> = out
            .extract_area(0, 7, 12, 1)
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        assert_eq!(got, want_row7, "premultiplied Rgba16 bicubic row 7");
    }

    /// Issue #705, the over-reach guard. Only bicubic sums rows through a
    /// `T`-returning helper. `BILINEAR_FLOAT` writes
    /// `tq[z] = c1 * tp1[z] + c2 * tp2[z] + c3 * tp3[z] + c4 * tp4[z]` as one
    /// expression with a single narrowing at the store, and `lbb.cpp` and
    /// `nohalo.cpp` compute in `double` throughout and narrow once at the end
    /// too. All three are already bit-exact against 8.18.6 on this fixture and
    /// must stay that way: narrowing an intermediate in any of them would move
    /// these samples by about 1e-05.
    #[test]
    fn affine_float_interpolators_other_than_bicubic_narrow_only_at_the_store() {
        #[rustfmt::skip]
        let want_bilinear: [f32; 108] = [
            197.26852, 84.20548, 108.676674, 163.16832, 110.41781, 110.961815,
            140.3723, 140.0685, 101.942764, 131.0, 42.5, 79.5,
            121.6277, 134.04109, 171.04109, 98.83167, 135.83168, 135.363,
            126.74967, 163.74966, 109.23288, 110.2416, 147.24161, 111.37671,
            126.51567, 163.51567, 41.315067, 79.81601, 116.81601, 132.85617,
            150.39726, 111.09402, 148.09401, 149.10274, 91.8541, 128.8541,
            92.2629, 129.2629, 162.58904, 125.11644, 147.35034, 184.35034,
            27.547945, 63.323326, 100.323326, 112.212326, 149.21233, 135.03734,
            157.59467, 120.41096, 107.29574, 143.20613, 81.29452, 118.29452,
            141.9452, 107.72884, 144.72884, 212.85617, 27.37568, 64.37568,
            112.461624, 111.0274, 148.0274, 140.25183, 177.25183, 91.71918,
            125.933945, 162.93394, 80.10959, 97.65069, 125.583786, 162.58379,
            159.03378, 184.16438, 68.32239, 83.7204, 86.595894, 111.8207,
            130.49222, 164.38356, 201.38356, 105.4589, 133.98076, 170.98076,
            59.46575, 96.46575, 124.98762, 151.00685, 188.00685, 160.7143,
            82.15631, 119.15631, 71.65753, 101.39773, 138.39774, 163.19862,
            180.73973, 98.00957, 135.00957, 193.19862, 66.52345, 103.52345,
            111.879906, 148.8799, 156.81873, 171.86348, 208.86348, 65.02008,
        ];
        #[rustfmt::skip]
        let want_lbb: [f32; 108] = [
            223.97984, 63.182693, 89.980804, 177.65471, 107.6153, 104.81888,
            149.91078, 156.67326, 101.22427, 137.0625, 32.875, 63.8125,
            123.63425, 152.94336, 193.15602, 88.592865, 145.86058, 139.08575,
            118.22841, 164.37238, 113.296364, 97.041115, 148.76147, 117.54855,
            137.64272, 185.66116, 21.03061, 55.103294, 115.21083, 145.20808,
            163.71788, 107.58647, 164.24315, 162.95132, 70.80461, 126.090485,
            88.235245, 126.98186, 176.27092, 136.41968, 162.69293, 205.70947,
            14.805203, 45.779423, 98.33576, 112.73169, 162.11461, 147.71895,
            181.09906, 119.592834, 93.7493, 149.6334, 60.88833, 102.771034,
            149.72311, 108.11873, 160.95882, 224.178, 14.91653, 53.13271,
            103.49911, 108.85715, 159.47243, 151.35829, 202.91455, 79.47844,
            127.9066, 181.5277, 58.553047, 74.95804, 123.53025, 178.6542,
            183.75572, 207.03062, 42.808277, 69.97353, 79.5257, 107.78205,
            139.00935, 178.128, 219.92058, 105.22234, 136.24048, 180.87585,
            37.62609, 73.83081, 120.96473, 158.00949, 196.41797, 174.71162,
            65.08279, 112.96368, 59.87288, 86.777824, 143.21373, 187.29716,
            199.47922, 83.86982, 133.26796, 207.85735, 45.452923, 91.88468,
            106.17024, 150.52966, 168.45255, 191.63763, 234.06316, 43.11054,
        ];
        #[rustfmt::skip]
        let want_nohalo: [f32; 108] = [
            224.97852, 66.48908, 93.30692, 172.79782, 107.53387, 103.74632,
            153.59686, 158.96323, 96.17205, 131.0, 23.25, 60.25,
            122.54222, 143.90665, 200.35365, 95.67481, 150.86491, 138.24774,
            120.79364, 166.54047, 111.22518, 100.58867, 153.37073, 115.44871,
            136.65433, 175.45961, 15.20176, 53.12734, 115.27774, 145.08076,
            170.37251, 111.6232, 167.74391, 158.46616, 69.20872, 126.323364,
            89.768036, 129.33746, 191.13144, 135.12457, 160.80562, 199.78749,
            13.108764, 49.285946, 99.760254, 105.16903, 166.1975, 145.16159,
            185.39957, 119.98104, 88.485245, 141.9015, 52.766975, 109.648346,
            162.18074, 104.56628, 158.36682, 223.42383, 14.497568, 53.48192,
            105.77036, 102.13799, 158.74597, 147.92348, 209.60939, 83.47758,
            120.97823, 169.89148, 50.86215, 78.40687, 124.20311, 186.91504,
            181.91985, 205.4671, 45.594307, 67.353546, 76.229294, 100.768616,
            134.20372, 172.1164, 227.43303, 103.49184, 132.25568, 175.53285,
            32.38869, 81.44474, 127.17772, 162.42912, 204.23552, 172.94606,
            73.97242, 111.93984, 49.92682, 75.99779, 140.74083, 179.31236,
            211.2558, 92.12908, 139.14842, 202.32161, 38.810753, 97.05681,
            108.59659, 156.0834, 170.3779, 189.37865, 233.43689, 46.334454,
        ];
        let im = float_carrier_fixture();
        for (name, want) in [
            ("bilinear", want_bilinear),
            ("lbb", want_lbb),
            ("nohalo", want_nohalo),
        ] {
            let out = im.affine([1.3, 0.2, -0.15, 1.1], name);
            assert_eq!(
                carrier_crop_f32(&out),
                want,
                "float {name} against vips 8.18.6"
            );
        }
    }

    /// The 24x24 `ushort` linear ramp `a * x + b * y + c`. Both bilinear and
    /// Catmull-Rom reproduce a linear function **exactly** (bilinear trivially;
    /// Catmull-Rom because its four coefficients sum to 1 and their first
    /// moment is the offset, which I checked in exact rationals at 0, 1/4,
    /// 33/64, 45/64 and 3/4), so over this fixture the right answer is closed
    /// form and the test does not have to reimplement the interpolator to know
    /// it. Issues #732 and #733 both turn on that.
    fn linear_ramp_u16(a: u16, b: u16, c: u16) -> Raster {
        let data: Vec<u8> = (0..24 * 24usize)
            .flat_map(|i| {
                let (x, y) = ((i % 24) as u16, (i / 24) as u16);
                (a * x + b * y + c).to_ne_bytes()
            })
            .collect();
        Raster::new(24, 24, PixelFormat::Gray16, data).unwrap()
    }

    /// Shift a raster by a sub-pixel displacement with the identity matrix, so
    /// the inverse map is `ix = ox - idx` and nothing about the geometry has to
    /// be recomputed in the test.
    fn subpixel_shift(im: &Raster, name: &str, idx: f64, idy: f64) -> Vec<u16> {
        let out = im
            .try_affine_with(
                [1.0, 0.0, 0.0, 1.0],
                Interpolator::from_name(name).unwrap(),
                AffineOptions {
                    idx,
                    idy,
                    extend: Extend::Copy,
                    ..AffineOptions::default()
                },
            )
            .unwrap();
        assert_eq!((out.width(), out.height()), (24, 24), "shift output size");
        out.data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect()
    }

    /// The 32x32 reduce fixtures. Sample `i` is `(i * 53 + 17) % 251` on the
    /// byte carrier and `(i * 3719 + 977) % 65413` on the 16-bit one, the
    /// same coprime-stride-over-a-prime-modulus shape [`carrier_fixture`]
    /// uses, at a size where a `reduce` stencil can sit strictly inside the
    /// image. Both are reproducible byte for byte on the vips side through
    /// `vips rawload`, which is how the oracle rows below were taken.
    fn reduce_fixture(format: PixelFormat) -> Raster {
        let n = 32 * 32usize;
        let data: Vec<u8> = match format {
            PixelFormat::Gray16 => (0..n)
                .flat_map(|i| (((i * 3719 + 977) % 65413) as u16).to_ne_bytes())
                .collect(),
            _ => (0..n).map(|i| ((i * 53 + 17) % 251) as u8).collect(),
        };
        Raster::new(32, 32, format, data).unwrap()
    }

    /// The interior crop of a `reduceh … 3 --kernel linear` over
    /// [`reduce_fixture`]: rows 0..8 of output columns 1..10. The mask is 7
    /// points wide with a margin of 3, and the sample position is `3 * i + 1`,
    /// so every column in that range reads a stencil strictly inside the
    /// 32-wide input and no [`Extend`] rule is involved in the comparison.
    fn reduce_crop(r: &Raster) -> Vec<u8> {
        assert_eq!(
            (r.width(), r.height()),
            (11, 32),
            "reduce fixture output size"
        );
        r.extract_area(1, 0, 9, 8).data().to_vec()
    }

    fn reduce_crop_u16(r: &Raster) -> Vec<u16> {
        reduce_crop(r)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect()
    }

    /// Issue #777. `vips_reduce_make_mask` builds every mask twice, once in
    /// `double` and once as a `short` fixed-point copy scaled by
    /// `VIPS_INTERPOLATE_SCALE`, and `vips_reduceh_gen` / `vips_reducev_gen`
    /// read the fixed-point one on **both** integer carriers. The copy is a
    /// truncation of each coefficient toward zero and is **not renormalised**,
    /// so its taps no longer sum to one.
    ///
    /// This module keeps the masks in `f64`, normalised to unit sum, and the
    /// most direct consequence is closed form: a constant image comes back
    /// unchanged here and does not in libvips.
    ///
    /// Measured on 8.18.6 over a 32x32 constant, `vips reduceh in.v out.v S
    /// --kernel K` and the same through `reducev`, which agree cell for cell,
    /// with `vips min` equal to `vips max` in every one so the output really
    /// is flat:
    ///
    /// | kernel | shrink 2 | shrink 3 | shrink 4 |
    /// |---|---|---|---|
    /// | `linear` | 65535 | **65471** | 65535 |
    /// | `cubic` | 65535 | 65535 | 65535 |
    /// | `mitchell` | **65503** | 65535 | **65503** |
    /// | `lanczos2` | 65535 | **65503** | 65535 |
    /// | `lanczos3` | 65535 | **65503** | **65407** |
    ///
    /// on a constant of 65535. The six bold cells are up to **128 of 65535**
    /// of pure scale error on an image with no detail in it at all.
    ///
    /// The `uchar` column of the same sweep is 200 for all fifteen cells, and
    /// that is not a second measurement but the same one: the worst deficit is
    /// `128 / 65535`, which at eight bits is `0.498` of a level, so
    /// round-half-up absorbs it with two thousandths of a level to spare. The
    /// test computes that rather than asserting it, since it is the reason the
    /// byte carrier looks clean and it is very nearly not true.
    #[test]
    fn reduce_preserves_a_constant_where_the_vips_short_mask_does_not() {
        // (kernel, shrink, what vips 8.18.6 returns on a constant 65535)
        const VIPS_8_18_6: [(&str, f64, u16); 15] = [
            ("linear", 2.0, 65535),
            ("linear", 3.0, 65471),
            ("linear", 4.0, 65535),
            ("cubic", 2.0, 65535),
            ("cubic", 3.0, 65535),
            ("cubic", 4.0, 65535),
            ("mitchell", 2.0, 65503),
            ("mitchell", 3.0, 65535),
            ("mitchell", 4.0, 65503),
            ("lanczos2", 2.0, 65535),
            ("lanczos2", 3.0, 65503),
            ("lanczos2", 4.0, 65535),
            ("lanczos3", 2.0, 65535),
            ("lanczos3", 3.0, 65503),
            ("lanczos3", 4.0, 65407),
        ];
        let flat = Raster::new(
            32,
            32,
            PixelFormat::Gray16,
            (0..32 * 32).flat_map(|_| 65535u16.to_ne_bytes()).collect(),
        )
        .unwrap();
        let flat8 = Raster::constant_u8(32, 32, 200);

        let mut short_cells = 0usize;
        let mut worst_deficit = 0u16;
        for (kernel, shrink, vips) in VIPS_8_18_6 {
            for out in [flat.reduceh(shrink, kernel), flat.reducev(shrink, kernel)] {
                assert!(
                    out.data()
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .all(|b| u16::from_ne_bytes(*b) == 65535),
                    "{kernel} at shrink {shrink} must leave a constant alone; \
                     vips returns {vips}"
                );
            }
            // The byte carrier agrees with vips, which is what makes the row
            // above a statement about depth rather than about this module.
            for out in [flat8.reduceh(shrink, kernel), flat8.reducev(shrink, kernel)] {
                assert!(out.data().iter().all(|&v| v == 200));
            }
            if vips != 65535 {
                short_cells += 1;
                worst_deficit = worst_deficit.max(65535 - vips);
            }
        }
        assert_eq!(
            (short_cells, worst_deficit),
            (6, 128),
            "the recorded vips answers are the other half of this pin: if they \
             stop being short, either the binary changed or the table did"
        );
        // Why the byte carrier cannot see it, computed rather than asserted.
        let at_eight_bits = f64::from(worst_deficit) / 65535.0 * 255.0;
        assert!(
            at_eight_bits < 0.5,
            "a deficit of {worst_deficit} in 65535 is {at_eight_bits} of a byte \
             level, which round-half-up would no longer absorb"
        );
        assert!(
            at_eight_bits > 0.49,
            "and it is within {} of not being absorbed, which is the point",
            0.5 - at_eight_bits
        );
    }

    /// Issue #777, the magnitude on real content, `ushort` carrier. A constant
    /// only shows the mask's scale error; the taps also move relative to each
    /// other, so the error on detail is larger and signed both ways.
    ///
    /// Measured on 8.18.6 as `vips reduceh in.v out.v 3 --kernel linear` over
    /// [`reduce_fixture`] rebuilt with `vips rawload`, interior crop
    /// `[1, 0, 9, 8]`: all 72 samples differ, worst delta 55. Over the whole
    /// 288-sample interior it is 288 of 288 at worst 55, and the worst cell in
    /// a five-kernel by five-shrink sweep is `lanczos3` at 2.5, 224 of 224 at
    /// worst 52.
    #[test]
    fn reduce_divergence_from_the_12_bit_mask_is_bounded_on_a_ushort_carrier() {
        #[rustfmt::skip]
        let vips_8_18_6: [u16; 72] = [
            13980, 25126, 36272, 47418, 54939, 18878, 15507, 26653, 37800,
            16690, 13319, 24466, 35612, 46758, 54279, 18218, 14847, 25993,
            55716, 34188, 12659, 23805, 34951, 46097, 53618, 17557, 14187,
            43910, 55056, 33527, 11999, 23145, 34291, 45437, 52958, 16897,
            32103, 43249, 54395, 32867, 11338, 22485, 33631, 44777, 55923,
            20297, 31443, 42589, 53735, 32207, 10678, 21824, 32970, 44116,
            12115, 19636, 30782, 41929, 53075, 31546, 10018, 21164, 32310,
            47516, 11455, 18976, 30122, 41268, 52414, 30886, 9357, 20503,
        ];
        let out = reduce_fixture(PixelFormat::Gray16).reduceh(3.0, "linear");
        let (mismatches, worst) = carrier_diff(&reduce_crop_u16(&out), &vips_8_18_6);
        assert!(
            mismatches <= 72 && worst <= 55,
            "ushort reduce differs from vips 8.18.6 in {mismatches} samples \
             (worst delta {worst}); expected at most 72 samples, delta 55"
        );
        assert!(
            mismatches >= 60 && worst >= 30,
            "the short-mask divergence is supposed to be here: {mismatches} \
             samples, worst delta {worst}. If this dropped, either #777 was \
             adopted or the fixture stopped exercising it"
        );
    }

    /// Issue #777, the same cell on a `uchar` carrier, which is the one
    /// `libviprs-tests`' `cli_resample_diff` sees as a 1 in the `reduceh` and
    /// `reducev` rows. The mask error is a fraction of 4096, so what it is
    /// worth scales with the carrier's depth: 1 level at eight bits and up to
    /// 55 at sixteen.
    ///
    /// Measured on 8.18.6 as `vips reduceh in.v out.v 3 --kernel linear` over
    /// the byte [`reduce_fixture`], same interior crop: 12 of 72 samples
    /// differ, every one by exactly 1. Over the whole 288-sample interior it
    /// is 53 of 288, still every one by exactly 1.
    #[test]
    fn reduce_divergence_from_the_12_bit_mask_is_one_level_on_a_uchar_carrier() {
        #[rustfmt::skip]
        let vips_8_18_6: [u8; 72] = [
            147, 124, 144, 122, 99, 119, 138, 116, 164,
            127, 105, 153, 130, 150, 128, 105, 125, 144,
            94, 114, 133, 111, 159, 122, 86, 134, 111,
            145, 123, 100, 120, 139, 117, 165, 128, 92,
            154, 131, 151, 129, 106, 126, 145, 123, 101,
            134, 112, 160, 123, 87, 135, 112, 132, 151,
            101, 121, 140, 118, 166, 129, 93, 141, 118,
            82, 130, 107, 127, 146, 124, 102, 121, 99,
        ];
        let out = reduce_fixture(PixelFormat::Gray8).reduceh(3.0, "linear");
        let got = reduce_crop(&out);
        let (mismatches, worst) = got
            .iter()
            .zip(vips_8_18_6.iter())
            .fold((0usize, 0u8), |(n, w), (&a, &b)| {
                (n + usize::from(a != b), w.max(a.abs_diff(b)))
            });
        assert!(
            mismatches <= 12 && worst <= 1,
            "uchar reduce differs from vips 8.18.6 in {mismatches} samples \
             (worst delta {worst}); expected at most 12 samples, delta 1"
        );
        assert!(
            mismatches >= 6,
            "the short-mask divergence is supposed to be here on the byte \
             carrier too: {mismatches} samples. If this dropped, either #777 \
             was adopted or the fixture stopped exercising it"
        );
    }

    /// A raster wearing every part of the header block plus two attachments,
    /// for the metadata sweep. `Rgb8` rather than a float carrier so the
    /// tag is unambiguously *set* rather than inferred from the format.
    fn meta_probe(w: u32, h: u32) -> Raster {
        let mut r = Raster::new(
            w,
            h,
            PixelFormat::Rgb8,
            (0..(w * h * 3) as usize).map(|i| (i % 251) as u8).collect(),
        )
        .unwrap();
        r.meta.interpretation = Some(Interpretation::ScRgb);
        r.meta.xres = 5.0;
        r.meta.yres = 7.0;
        r.meta.orientation = 6;
        r.set_icc_profile(&[1u8, 2, 3, 4]);
        r.set_field(
            "lane-564",
            crate::imageio::MetadataValue::Str("carried".into()),
        );
        r
    }

    /// Issue #789. Every op in this module built its result with a bare
    /// [`Raster::new`] or [`Raster::zeroed`] and carried nothing: no
    /// interpretation, no resolution, no orientation, no ICC profile and no
    /// field a caller attached. #717 fixed this shape at eighteen sites and
    /// never reached here, because its census was `grep '\.meta = '` and this
    /// module has no such line.
    ///
    /// Measured on 8.18.6 over an 8x8 and a 12x5 `.v` tagged
    /// `--interpretation scrgb --xres 5 --yres 7`, carrying a 560-byte
    /// AdobeRGB profile and a `VipsRefString` named `lane-564`: `resize`,
    /// `shrink`, `shrinkh`, `shrinkv`, `reduce`, `reduceh`, `reducev`,
    /// `affine`, `similarity`, `rotate`, `mapim`, `zoom`, `subsample`,
    /// `thumbnail_image` and `thumbnail` all carry every one of them, at both
    /// shapes and on both an upscale and a downscale.
    ///
    /// **The resolution is carried verbatim and not rescaled with the scale
    /// factor.** `vips resize probe.v out.v 0.5` and `… 2` both come back
    /// `xres: 5, yres: 6.9999606299212607`, the input's values to the last
    /// bit. That is the same answer #690 measured for `zoom` and `subsample`
    /// and it is worth stating, because rescaling is the plausible-looking
    /// thing a resampler might do.
    ///
    /// [`Raster::extract_area`] is the control: it went onto the shared carry
    /// in #740, so a run where it fails too is a broken probe rather than a
    /// finding.
    #[test]
    fn every_resample_op_carries_the_header_block_and_the_attached_fields() {
        for (w, h) in [(8u32, 8u32), (12, 5)] {
            let s = meta_probe(w, h);
            let index = {
                let mut px: Vec<u8> = Vec::new();
                for y in 0..4u32 {
                    for x in 0..4u32 {
                        px.extend((x as f32).to_ne_bytes());
                        px.extend((y as f32).to_ne_bytes());
                    }
                }
                Raster::new(
                    4,
                    4,
                    PixelFormat::FloatF32(core::num::NonZeroU16::new(2).unwrap()),
                    px,
                )
                .unwrap()
            };
            let cases: Vec<(&str, Raster)> = vec![
                ("extract_area (control)", s.extract_area(0, 0, 4, 4)),
                ("shrink 2", s.try_shrink(2.0, 2.0).unwrap()),
                ("shrink 1.5", s.try_shrink(1.5, 1.5).unwrap()),
                ("shrinkh 2", s.try_shrinkh(2).unwrap()),
                ("shrinkv 2", s.try_shrinkv(2).unwrap()),
                (
                    "reduce 2",
                    s.try_reduce(2.0, 2.0, ReduceKernel::Lanczos3).unwrap(),
                ),
                (
                    "reduceh 2",
                    s.try_reduceh(2.0, ReduceKernel::Lanczos3).unwrap(),
                ),
                (
                    "reducev 2",
                    s.try_reducev(2.0, ReduceKernel::Lanczos3).unwrap(),
                ),
                (
                    "reduceh 1 (passthrough)",
                    s.try_reduceh(1.0, ReduceKernel::Lanczos3).unwrap(),
                ),
                ("resize 0.5", s.try_resize(0.5).unwrap()),
                ("resize 2", s.try_resize(2.0).unwrap()),
                ("resize 1 (passthrough)", s.try_resize(1.0).unwrap()),
                (
                    "resize 0.5 nearest",
                    s.try_resize_with(
                        0.5,
                        ResizeOptions {
                            kernel: ReduceKernel::Nearest,
                            ..ResizeOptions::default()
                        },
                    )
                    .unwrap(),
                ),
                (
                    "resize 2 nearest",
                    s.try_resize_with(
                        2.0,
                        ResizeOptions {
                            kernel: ReduceKernel::Nearest,
                            ..ResizeOptions::default()
                        },
                    )
                    .unwrap(),
                ),
                (
                    "affine 1.5",
                    s.try_affine([1.5, 0.0, 0.0, 1.5], Interpolator::Bilinear)
                        .unwrap(),
                ),
                ("similarity 2", s.try_similarity(0.0, 2.0).unwrap()),
                ("rotate 30", s.try_rotate(30.0).unwrap()),
                (
                    "mapim",
                    s.try_mapim(&index, Interpolator::Bilinear).unwrap(),
                ),
                ("thumbnail_image 4", s.try_thumbnail_image(4).unwrap()),
            ];
            for (op, out) in cases {
                let at = format!("{op} on a {w}x{h} raster");
                assert_eq!(
                    out.meta.interpretation,
                    Some(Interpretation::ScRgb),
                    "{at}: interpretation"
                );
                assert!(
                    (out.meta.xres - 5.0).abs() < 1e-12 && (out.meta.yres - 7.0).abs() < 1e-12,
                    "{at}: resolution is {} x {}, and vips carries 5 x 7 verbatim \
                     rather than rescaling it",
                    out.meta.xres,
                    out.meta.yres
                );
                assert_eq!(out.meta.orientation, 6, "{at}: orientation");
                assert_eq!(
                    out.icc_profile(),
                    Some(&[1u8, 2, 3, 4][..]),
                    "{at}: ICC profile"
                );
                assert!(
                    out.get_field("lane-564").is_some(),
                    "{at}: the attached field"
                );
            }
        }
    }

    /// Issue #789, the half that is not cosmetic. #664 made the premultiply
    /// bracket read the **interpretation** on a float carrier, because scRGB's
    /// alpha maximum is 1.0 where sRGB's is 255. While `resize` dropped the
    /// tag, the second call in a chain saw a different carrier from the first
    /// however the input was tagged, so `resize(0.5).resize(0.5)` could not
    /// agree with itself.
    ///
    /// The probe is an 8x8 `RgbaF32` chequerboard alpha resized to half twice,
    /// once from a `ScRgb`-tagged input and once from the same pixels
    /// untagged. Before this change both outputs came back untagged and 33 of
    /// the 256 output bytes differed; after it the tagged chain keeps its tag
    /// and the two chains stay different, which is the correct answer rather
    /// than the absence of one.
    #[test]
    fn a_chained_resize_keeps_reading_the_carrier_the_first_call_read() {
        let px: Vec<u8> = (0..8 * 8usize)
            .flat_map(|p| {
                let a = if (p / 8 + p % 8) % 2 == 0 {
                    1.0f32
                } else {
                    0.25
                };
                [0.8f32, 0.5, 0.2, a]
            })
            .flat_map(f32::to_ne_bytes)
            .collect();
        let untagged = Raster::new(8, 8, PixelFormat::RgbaF32, px).unwrap();
        let tagged = untagged
            .copy()
            .interpretation(Interpretation::ScRgb)
            .build();

        let chain = |r: &Raster| r.try_resize(0.5).unwrap().try_resize(0.5).unwrap();
        let from_tagged = chain(&tagged);
        let from_untagged = chain(&untagged);

        assert_eq!(
            from_tagged.meta.interpretation,
            Some(Interpretation::ScRgb),
            "the tag has to survive the chain or the second call reads the \
             wrong alpha ceiling"
        );
        assert_eq!(
            from_untagged.meta.interpretation, None,
            "and an untagged input stays untagged rather than acquiring one"
        );
        // The positive control on the whole test: the two carriers really do
        // produce different pixels, so "the tag survived" is a claim about
        // something observable.
        let differing = from_tagged
            .data()
            .iter()
            .zip(from_untagged.data())
            .filter(|(a, b)| a != b)
            .count();
        assert!(
            differing > 0,
            "the scRGB and sRGB alpha ceilings must disagree on this fixture, \
             or the tag carrying nothing would look the same as carrying it"
        );
        // And a single half-scale call agrees with the first step of the
        // chain, which is what says the chain is consistent with itself.
        assert_eq!(
            tagged.try_resize(0.5).unwrap().meta.interpretation,
            Some(Interpretation::ScRgb)
        );
    }

    /// Issue #733. `vips_interpolate_bilinear_interpolate` dispatches through
    /// `SWITCH_INTERPOLATE(BandFmt, BILINEAR_INT, BILINEAR_FLOAT)`, and `UCHAR`,
    /// `CHAR`, `USHORT` and `SHORT` all take `BILINEAR_INT`, which builds its
    /// four weights as 12-bit fixed point:
    ///
    /// ```c
    /// const int X = (x - ix) * VIPS_INTERPOLATE_SCALE;
    /// const int Yd = VIPS_INTERPOLATE_SCALE - Y;
    /// const int c4 = (Y * X) >> VIPS_INTERPOLATE_SHIFT;
    /// ```
    ///
    /// This module keeps the weights in `f64` on every carrier, which is what
    /// `BILINEAR_FLOAT` does for `UINT`, `INT`, `FLOAT` and `DOUBLE`. **That
    /// divergence stays**, and this test is the reason it can stay visible.
    ///
    /// The fixture is a linear ramp, which bilinear reproduces exactly, so the
    /// correct answer is `a * ix + b * iy + c` and the identity matrix with a
    /// sub-pixel `idx` makes `ix = ox - idx`. Measured on 8.18.6 as
    /// `vips affine in.v out.v "1 0 0 1" --interpolate bilinear --idx 0.3
    /// --idy 0.6 --extend copy`: this module hits `round(exact)` on **529 of
    /// 529** interior samples and vips misses it on **529 of 529**, every one
    /// of them one low. The offsets are 0.7 and 0.4, and `0.7 * 4096` is
    /// 2867.2, so vips' weight is 2867 and the ramp is reconstructed with a
    /// slope that is short by one part in 20000.
    #[test]
    fn affine_bilinear_reproduces_a_linear_ramp_where_vips_quantises_its_weights() {
        let (a, b, c) = (2001.0f64, 700.0f64, 1000.0f64);
        let (idx, idy) = (0.3f64, 0.6f64);
        let got = subpixel_shift(&linear_ramp_u16(2001, 700, 1000), "bilinear", idx, idy);
        // `ix = ox - idx` and `iy = oy - idy`, so the exact bilinear value is
        // the ramp evaluated there. The constant works out to -20.3, whose
        // fractional part is 0.7: far enough from a rounding boundary that no
        // `f64` accumulation noise can reach it.
        let offset = c - a * idx - b * idy;
        for oy in 1..24usize {
            for ox in 1..24usize {
                let exact = a * ox as f64 + b * oy as f64 + offset;
                assert_eq!(
                    got[oy * 24 + ox],
                    (exact + 0.5).floor() as u16,
                    "sample ({ox}, {oy}): exact bilinear is {exact}"
                );
            }
        }
    }

    /// Issue #733, the magnitude. The ramp above says which implementation is
    /// right; this one says what the divergence is worth, because a smooth
    /// ramp understates it. A 12-bit weight is short by up to `1 / 4096` of a
    /// pixel, and the error that produces is that fraction of the *difference*
    /// between neighbouring taps, so it grows with the local contrast and with
    /// the carrier's depth. On a byte carrier it is 1 LSB; on a 16-bit one it
    /// is worth `65535 / 4096`, about 16.
    ///
    /// Measured on 8.18.6 over the shared 12x12 `Gray16` fixture, interior crop
    /// `[4, 4, 6, 6]`: 31 of 36 samples differ, worst delta **20**. Over a
    /// random 24x24 the whole frame is 1345 of 1764 at worst 26.
    #[test]
    fn affine_bilinear_divergence_from_the_12_bit_weights_is_bounded() {
        #[rustfmt::skip]
        let vips_8_18_6: [u16; 36] = [
            31024, 31689, 32355, 33023, 33692, 34358,
            17374, 24763, 30108, 17518, 14675, 15341,
            56600, 59071, 59736, 26258, 20743, 28121,
            39389, 40055, 40720, 41386, 42052, 42717,
            20368, 21039, 21704, 22370, 23036, 23701,
            43465, 50843, 27059, 3354, 7597, 14986,
        ];
        let out = carrier_fixture(PixelFormat::Gray16).affine([1.3, 0.2, -0.15, 1.1], "bilinear");
        let (mismatches, worst) = carrier_diff(&carrier_crop_u16(&out), &vips_8_18_6);
        assert!(
            mismatches <= 31 && worst <= 20,
            "ushort bilinear differs from vips 8.18.6 in {mismatches} samples \
             (worst delta {worst}); expected at most 31 samples, delta 20"
        );
        assert!(
            mismatches >= 20 && worst >= 8,
            "the 12-bit weight divergence is supposed to be here: {mismatches} \
             samples, worst delta {worst}. If this dropped, either #733 was \
             adopted or the fixture stopped exercising it"
        );
    }

    /// Issue #732. `vips_interpolate_bicubic_interpolate` sends a `ushort`
    /// raster to `bicubic_unsigned_int32_tab`, which reads the same `double`
    /// coefficient table this module uses and then ends:
    ///
    /// ```c
    /// bicubic = VIPS_CLIP(0, bicubic, max_value);
    /// out[z] = bicubic;
    /// ```
    ///
    /// `out` is an `unsigned short *` and `bicubic` a `double`, so that store
    /// is a plain C conversion and **truncates toward zero**.
    /// [`SampleLayout::write`] rounds half up. **That divergence stays too.**
    ///
    /// Same closed-form fixture, with the displacements chosen so both
    /// sub-pixel offsets land on the 1/64 grid (0.75 and 0.5), which makes the
    /// offset quantisation of #668 a no-op and leaves the store as the only
    /// difference. The constant works out to 149.75. Measured on 8.18.6 as
    /// `vips affine in.v out.v "1 0 0 1" --interpolate bicubic --idx 0.25
    /// --idy 0.5 --extend copy`: vips hits `floor(exact)` on **441 of 441**
    /// interior samples and this module hits `round(exact)` on 441 of 441, so
    /// every one of the 441 differs by exactly 1 and this module is the one
    /// 0.25 away rather than 0.75.
    #[test]
    fn affine_bicubic_rounds_the_ushort_store_where_vips_truncates() {
        let (a, b, c) = (2001.0f64, 700.0f64, 1000.0f64);
        let (idx, idy) = (0.25f64, 0.5f64);
        let got = subpixel_shift(&linear_ramp_u16(2001, 700, 1000), "bicubic", idx, idy);
        let offset = c - a * idx - b * idy;
        for oy in 2..23usize {
            for ox in 2..23usize {
                let exact = a * ox as f64 + b * oy as f64 + offset;
                assert_eq!(
                    got[oy * 24 + ox],
                    (exact + 0.5).floor() as u16,
                    "sample ({ox}, {oy}): exact bicubic is {exact}, and vips \
                     stores {} because its store truncates (#732)",
                    exact.floor() as u16
                );
            }
        }
    }

    /// Issue #704, the overshoot regime. Catmull-Rom rings, so a hard edge
    /// drives the fixed-point accumulators **negative** and past the carrier's
    /// ceiling, and that is where two details of the C stop being cosmetic:
    /// `unsigned_fixed_round` divides with `>>` on a signed `int`, an
    /// arithmetic shift, so a negative accumulator floors rather than
    /// truncating; and `vips_bicubic_matrixi` truncates its two negative
    /// coefficients toward zero, which is the opposite direction.
    ///
    /// The 3x3 checkerboard below puts 40 of the 144 row accumulators and 9 of
    /// the 36 column accumulators below zero and drives 15 of the 36 output
    /// samples outside `0..=255`, where the smooth fixture in
    /// `affine_bicubic_reads_the_vips_fixed_point_table_on_a_uchar_carrier`
    /// reaches one negative row accumulator and clips nothing at all.
    ///
    /// Measured on 8.18.6 with the same matrix and the same interior crop.
    #[test]
    fn affine_bicubic_clips_the_fixed_point_overshoot_like_vips() {
        let data: Vec<u8> = (0..12 * 12usize)
            .map(|i| {
                let (x, y) = (i % 12, i / 12);
                if ((x / 3) % 2) ^ ((y / 3) % 2) != 0 {
                    255
                } else {
                    0
                }
            })
            .collect();
        #[rustfmt::skip]
        let want: [u8; 36] = [
            179, 198, 164, 128, 148, 198,
            91, 0, 0, 0, 166, 255,
            137, 0, 0, 0, 123, 255,
            187, 0, 7, 18, 102, 170,
            73, 223, 255, 255, 225, 0,
            0, 223, 255, 255, 246, 30,
        ];
        let out = Raster::new(12, 12, PixelFormat::Gray8, data)
            .unwrap()
            .affine([1.3, 0.2, -0.15, 1.1], "bicubic");
        assert_eq!(
            carrier_crop(&out),
            want,
            "uchar bicubic overshoot against vips 8.18.6"
        );
    }

    /// Issue #704, the first over-reach guard: the fixed point is the `uchar`
    /// arithmetic and **only** the `uchar` arithmetic. `VIPS_FORMAT_USHORT`
    /// goes to `bicubic_unsigned_int32_tab`, which takes `cxf`/`cyf`, the
    /// `double` table, and never looks at `vips_bicubic_matrixi` at all. So a
    /// 16-bit raster must keep exact `f64` coefficients, and quantising them
    /// there moves samples by far more than the residual this test allows
    /// (measured: 1320 of 1764 samples, worst delta 29, on a random 24x24).
    ///
    /// The residual it does allow is a different divergence with a different
    /// cause: `bicubic_unsigned_int32_tab` finishes with `out[z] = bicubic`
    /// where `out` is `unsigned short *` and `bicubic` a `double`, so the C
    /// conversion truncates toward zero, and this module's sample writer
    /// rounds half up. That is the same shape as the `vips_cast` divergence
    /// the module header already records, and it is worth 14 of these 36
    /// samples at exactly 1 LSB. Issue #732 tracks it.
    #[test]
    fn affine_bicubic_keeps_f64_coefficients_on_a_ushort_carrier() {
        #[rustfmt::skip]
        let want: [u16; 36] = [
        30157, 29512, 29971, 33022, 36073, 36532,
        15238, 22199, 27738, 13560, 8704, 10718,
        57377, 62386, 64601, 23414, 16394, 27797,
        40015, 40009, 40694, 43805, 47009, 47537,
        15759, 17450, 21479, 22063, 22070, 22016,
        45560, 57600, 25251, 0, 6247, 11991,
        ];
        let out = carrier_fixture(PixelFormat::Gray16).affine([1.3, 0.2, -0.15, 1.1], "bicubic");
        let (mismatches, worst) = carrier_diff(&carrier_crop_u16(&out), &want);
        assert!(
            mismatches <= 14 && worst <= 1,
            "ushort bicubic differs from vips 8.18.6 in {mismatches} samples \
             (worst delta {worst}); expected at most 14 samples, delta 1"
        );
    }

    /// Issue #704, the second over-reach guard: `VIPS_FORMAT_FLOAT` goes to
    /// `bicubic_float_tab<float>` and reads the `double` table too, so a float
    /// raster keeps exact coefficients as well. Twelve-bit coefficients would
    /// move these samples by around 0.03, four orders of magnitude past the
    /// residual allowed here.
    ///
    /// There is no residual to allow. The accumulation seam that used to leave
    /// 3.8147e-06 on 2 of these 36 samples is closed by #705, so this is a
    /// bit-for-bit pin and a 12-bit coefficient would miss it by four orders of
    /// magnitude.
    #[test]
    fn affine_bicubic_keeps_f64_coefficients_on_a_float_carrier() {
        #[rustfmt::skip]
        let want: [f32; 36] = [
        71.703255, 62.78587, 99.873535, 35.3125, 64.543846, 95.86998,
        185.31021, 227.67528, 227.36438, 108.281334, 44.770844, 7.026093,
        123.21098, 144.45389, 166.65953, 217.43056, 144.26581, 94.96411,
        57.04591, 87.824974, 109.839294, 131.09113, 159.03162, 195.53754,
        178.75148, 26.286255, 44.638718, 73.9971, 93.95278, 118.6862,
        211.47437, 196.00458, 81.3681, -5.6804013, 48.58261, 63.801754,
        ];
        let out = carrier_fixture(PixelFormat::FloatF32(
            core::num::NonZeroU16::new(1).unwrap(),
        ))
        .affine([1.3, 0.2, -0.15, 1.1], "bicubic");
        assert_eq!(
            carrier_crop_f32(&out),
            want,
            "float bicubic against vips 8.18.6"
        );
    }

    /// Issue #704, the third over-reach guard: an alpha band takes the
    /// decision away from the stored depth. `vips_affine_build` premultiplies
    /// whenever `vips_image_hasalpha()` (`affine.c:551`), and
    /// `vips_premultiply` writes a FLOAT image, so a `uchar` RGBA raster is
    /// interpolated by `bicubic_float_tab<float>` and never touches the fixed
    /// point either. Running the integer path here would quantise
    /// premultiplied colour that is no longer integral at all.
    ///
    /// The residual is the `vips_cast` truncation the module header records:
    /// vips casts the un-premultiplied FLOAT back to `uchar` with a plain C
    /// cast and this module's writer rounds half up, worth 64 of these 144
    /// bytes at 1 LSB.
    #[test]
    fn affine_bicubic_keeps_f64_coefficients_when_alpha_forces_the_premultiply() {
        #[rustfmt::skip]
        let want: [u8; 144] = [
        135, 19, 90, 143, 186, 76, 59, 117, 193, 166, 58, 91, 158, 238, 114, 66, 177, 255, 241, 23, 98, 144, 255, 29,
        3, 73, 126, 179, 57, 38, 100, 153, 245, 18, 73, 127, 199, 116, 36, 101, 164, 182, 70, 77, 159, 255, 120, 40,
        50, 113, 166, 233, 65, 83, 136, 190, 213, 57, 110, 163, 226, 13, 84, 137, 224, 53, 50, 111, 174, 127, 37, 85,
        71, 126, 188, 95, 67, 120, 173, 236, 24, 93, 146, 201, 129, 68, 121, 175, 213, 33, 93, 146, 234, 9, 67, 121,
        131, 179, 255, 33, 86, 139, 195, 26, 66, 120, 173, 176, 41, 105, 158, 220, 42, 78, 131, 187, 178, 51, 104, 157,
        163, 255, 86, 69, 152, 224, 236, 36, 128, 174, 255, 17, 61, 115, 167, 68, 59, 112, 165, 216, 13, 86, 139, 191,
        ];
        let out = carrier_fixture(PixelFormat::Rgba8).affine([1.3, 0.2, -0.15, 1.1], "bicubic");
        let got: Vec<u16> = carrier_crop(&out).iter().map(|&b| u16::from(b)).collect();
        let expected: Vec<u16> = want.iter().map(|&b| u16::from(b)).collect();
        let (mismatches, worst) = carrier_diff(&got, &expected);
        assert!(
            mismatches <= 64 && worst <= 1,
            "premultiplied uchar bicubic differs from vips 8.18.6 in {mismatches} bytes \
             (worst delta {worst}); expected at most 64 bytes, delta 1"
        );
    }

    /// The transpose matrix `[0, 1, 1, 0]` samples exactly on the input
    /// grid, so every interpolator (including nohalo and lbb) reproduces
    /// the transpose byte for byte, and four applications are the
    /// identity: the ported test_affine round-trip invariant.
    #[test]
    fn nohalo_lbb_transpose_round_trip_is_identity() {
        let im = crate::source::generate_test_raster(6, 4).unwrap();
        let reference = im.rot(Angle::D90).fliphor();
        for interp in ["nohalo", "lbb"] {
            let t = im.affine([0.0, 1.0, 1.0, 0.0], interp);
            assert_eq!(t.width(), im.height(), "{interp} transpose width");
            assert_eq!(t.height(), im.width(), "{interp} transpose height");
            assert_eq!(t.data(), reference.data(), "{interp} transpose bytes");

            let mut x = im.clone();
            for _ in 0..4 {
                x = x.affine([0.0, 1.0, 1.0, 0.0], interp);
            }
            assert_eq!(x.data(), im.data(), "{interp} 4x transpose not identity");
        }
    }

    /// LBB stays locally bounded: an upscale never overshoots the range of
    /// the input samples, the defining property of the resampler (no
    /// output clamping needed). Nohalo, being co-monotone, likewise keeps
    /// a monotone ramp within its endpoints.
    #[test]
    fn nohalo_lbb_stay_within_input_range() {
        // A ramp with a sharp central step, the classic overshoot probe:
        // a plain bicubic rings above 255 / below 0 at the step, lbb and
        // nohalo may not.
        let mut data = vec![0u8; 12 * 12];
        for y in 0..12usize {
            for x in 0..12usize {
                data[y * 12 + x] = if x < 6 { 30 } else { 220 };
            }
        }
        let im = Raster::new(12, 12, PixelFormat::Gray8, data).unwrap();
        for interp in ["nohalo", "lbb"] {
            // A 2.5x upscale around the step, the regime where cubic
            // resamplers overshoot.
            let up = im.affine([2.5, 0.0, 0.0, 2.5], interp);
            // Sample the interior to avoid the background-extended border.
            let inner = up.extract_area(4, 4, up.width() - 8, up.height() - 8);
            let (lo, hi) = inner
                .data()
                .iter()
                .fold((255u8, 0u8), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            assert!(
                lo >= 30 && hi <= 220,
                "{interp} overshot the [30, 220] input range: got [{lo}, {hi}]"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn thumbnail_free_fn_fits_the_width_box() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("thumb_in.png");
        Raster::new(100, 60, PixelFormat::Rgb8, vec![120u8; 100 * 60 * 3])
            .unwrap()
            .save(&path)
            .unwrap();

        // Bare-width fit into a 50x50 box: shrink = max(100/50, 60/50) = 2.
        let thumb = super::thumbnail(&path, 50).unwrap();
        assert_eq!(thumb.width(), 50);
        assert!(
            (i64::from(thumb.height()) - 30).abs() <= 1,
            "height {} not near 30",
            thumb.height()
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
    fn thumbnail_crop_free_fn_fills_and_crops_the_box() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("thumb_crop_in.png");
        Raster::new(100, 60, PixelFormat::Rgb8, vec![90u8; 100 * 60 * 3])
            .unwrap()
            .save(&path)
            .unwrap();

        // crop="centre" fills a 40x40 box and centre-crops to it.
        let thumb = super::thumbnail_crop(&path, 40, 40, "centre").unwrap();
        assert_eq!(thumb.width(), 40);
        assert_eq!(thumb.height(), 40);

        // crop="none" fits inside the box, preserving aspect ratio.
        let fit = super::thumbnail_crop(&path, 40, 40, "none").unwrap();
        assert!(fit.width() <= 40 && fit.height() <= 40);
        assert!(fit.width() == 40 || fit.height() == 40);
    }

    /// An 8x8 constant `RgbaF32` raster of `(100, 20, 3, alpha)`, optionally
    /// tagged. A constant survives every reduce kernel exactly (unit-sum mask,
    /// replicated edges), so a `resize(0.5)` on one isolates the premultiply
    /// bracket: whatever comes out is `premultiply(max) -> unpremultiply(max)`
    /// and nothing else.
    fn const_rgba_f32(alpha: f32, tag: Option<Interpretation>) -> Raster {
        let mut samples = Vec::with_capacity(8 * 8 * 4);
        for _ in 0..8 * 8 {
            samples.extend_from_slice(&[100.0, 20.0, 3.0, alpha]);
        }
        let im = Raster::from_f32_samples(8, 8, PixelFormat::RgbaF32, &samples).unwrap();
        match tag {
            Some(t) => im.copy().interpretation(t).build(),
            None => im,
        }
    }

    /// The interior sample of `const_rgba_f32(alpha, tag).resize(0.5)`.
    fn resized_const_pixel(alpha: f32, tag: Option<Interpretation>) -> [f32; 4] {
        let out = const_rgba_f32(alpha, tag).resize(0.5);
        assert_eq!(out.width(), 4, "resize(0.5) of an 8x8 is 4 wide");
        assert_eq!(out.height(), 4, "resize(0.5) of an 8x8 is 4 high");
        assert_eq!(out.format(), PixelFormat::RgbaF32, "the carrier survives");
        let s = out.f32_samples().expect("float carrier");
        // Pixel (1, 1) of the 4x4 output, the one vips was read at.
        let base = (4 + 1) * 4;
        [s[base], s[base + 1], s[base + 2], s[base + 3]]
    }

    /// Issue #664. The premultiply bracket's `max_alpha` is a property of the
    /// **interpretation**, not of the storage depth, so a float carrier tagged
    /// scRGB brackets against `1.0` and one tagged RGB16 against `65535`, where
    /// an untagged one keeps the `255` default.
    ///
    /// `vips_resize` premultiplies nothing of its own ("This operation does
    /// not premultiply alpha", `libvips/resample/resize.c`), and measured: the
    /// same float RGBA resizes to identical bytes under every interpretation
    /// tag. The bracket libviprs runs around it is `vips_premultiply` /
    /// `vips_unpremultiply`, which default `max_alpha` from
    /// `vips_interpretation_max_alpha` (`libvips/iofuncs/header.c:195`), so
    /// that pair is the oracle. Measured on vips 8.18.6, an 8x8 constant float
    /// RGBA `(100, 20, 3, alpha)` through
    /// `premultiply | resize 0.5 | unpremultiply`, read at pixel (1, 1):
    ///
    /// ```text
    /// alpha  untagged (255)                        scrgb (1.0)                                rgb16 (65535)
    /// 0.5    100.00000762939453 20 3.0000002384185791 0.5   100 20 3 0.5                       100 20 3 0.5
    /// 1.5    100.00000762939453 20 3.0000002384185791 1.5   66.666671752929688 13.333333969116211 2 1   100 20 3 1.5
    /// 300    85 17 2.5500001907348633 255                   0.3333333432674408 0.066666670143604279 0.0099999997764825821 1   99.999992370605469 20 3 300
    /// ```
    ///
    /// The literals below are the shortest `f32` spelling of each of those
    /// prints, which is the same bit pattern (`clippy::excessive_precision`
    /// rejects the full decimal expansion vips writes).
    ///
    /// The alpha `0.5` row is the regression guard the untagged case needs: an
    /// alpha inside every candidate ceiling makes the bracket cancel, so all
    /// three agree to within an ulp and nothing may move there. `1.5` and `300`
    /// are the discriminating rows, because they sit above one ceiling and
    /// below another: the premultiply clip and the stored-alpha clip both bite,
    /// and the three interpretations separate by a factor of 300.
    #[test]
    fn resize_float_bracket_takes_max_alpha_from_the_interpretation() {
        let cases: [(f32, Option<Interpretation>, [f32; 4]); 9] = [
            (0.5, None, [100.000_01, 20.0, 3.000_000_2, 0.5]),
            (0.5, Some(Interpretation::ScRgb), [100.0, 20.0, 3.0, 0.5]),
            (0.5, Some(Interpretation::Rgb16), [100.0, 20.0, 3.0, 0.5]),
            (1.5, None, [100.000_01, 20.0, 3.000_000_2, 1.5]),
            (
                1.5,
                Some(Interpretation::ScRgb),
                [66.666_67, 13.333_334, 2.0, 1.0],
            ),
            (1.5, Some(Interpretation::Rgb16), [100.0, 20.0, 3.0, 1.5]),
            (300.0, None, [85.0, 17.0, 2.550_000_2, 255.0]),
            (
                300.0,
                Some(Interpretation::ScRgb),
                [0.333_333_34, 0.066_666_67, 0.01, 1.0],
            ),
            (
                300.0,
                Some(Interpretation::Rgb16),
                [99.999_99, 20.0, 3.0, 300.0],
            ),
        ];
        for (alpha, tag, want) in cases {
            let got = resized_const_pixel(alpha, tag);
            for (band, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (f64::from(*g) - f64::from(*w)).abs() <= f64::from(w.abs()) * 1e-6,
                    "alpha {alpha}, tag {tag:?}, band {band}: got {g}, want vips' {w}"
                );
            }
        }
    }

    /// Issue #664, the bit-exact half. `vips_premultiply` lands the multiplier
    /// in a `float` before the colour multiply (`OUT nalpha = (OUT) clip_alpha
    /// / max_alpha` with `OUT` = float, then `q[i] = p[i] * nalpha`), and
    /// `vips_unpremultiply` mirrors it with `OUT factor`. So the bracket rounds
    /// **twice**, exactly as #631 found for the standalone operations, and an
    /// `f64` expression rounded once at the store is a different number.
    ///
    /// The fingerprint is visible in the measured table above and it is what
    /// this pins: `100 * f32(0.5 / 255) * f32(255 / 0.5)` is
    /// `100.00000762939453`, not `100`, and `100 * f32(1.5 / 1) * f32(1 / 1.5)`
    /// is `66.666671752929688`, not `66.66666412353516`. Both are one ulp, and
    /// both are the difference between a value pinned on the binary and a value
    /// pinned on a model of it.
    #[test]
    fn resize_float_bracket_rounds_through_f32_like_the_c_macros() {
        // The untagged carrier: an f64 round trip returns exactly 100.0 here,
        // the C's double rounding does not.
        let got = resized_const_pixel(0.5, None);
        assert_eq!(
            got[0].to_bits(),
            100.000_01f32.to_bits(),
            "band 0 came out {}, want vips' 100.00000762939453 (f64 single-rounded \
             gives exactly 100)",
            got[0]
        );
        // scRGB with an alpha above its ceiling: the clip makes the premultiply
        // exact, so the surviving ulp comes from `f32(1.0 / 1.5)`.
        let got = resized_const_pixel(1.5, Some(Interpretation::ScRgb));
        assert_eq!(
            got[0].to_bits(),
            66.666_67f32.to_bits(),
            "band 0 came out {}, want vips' 66.666671752929688 (f64 single-rounded \
             gives 66.66666412353516)",
            got[0]
        );
    }

    /// Issue #664, the reachable case. Nothing has to be out of range for the
    /// ceiling to show: lanczos3 rings, so resizing a hard transparency edge
    /// pushes the resampled alpha above the source's maximum, and the stored
    /// alpha is clipped to `VIPS_CLIP(0, alpha, max_alpha)` on the way out.
    ///
    /// Measured on vips 8.18.6 with a 16x2 float RGBA of `(100, 20, 3, a)`,
    /// `a = 0` for the left half and `1` for the right, through
    /// `premultiply | resize 0.5 | unpremultiply`: the alpha at x = 5 comes
    /// back `1.0152533054351807` untagged and exactly `1` under scrgb. So an
    /// ordinary 0..1 scRGB raster with no overshoot in it at all still moves
    /// bytes, which is what makes this worth fixing rather than documenting.
    #[test]
    fn resize_clips_the_stored_alpha_to_the_interpretation_ceiling() {
        let mut samples = Vec::with_capacity(16 * 2 * 4);
        for _ in 0..2 {
            for x in 0..16 {
                samples.extend_from_slice(&[100.0, 20.0, 3.0, if x < 8 { 0.0 } else { 1.0 }]);
            }
        }
        let im = Raster::from_f32_samples(16, 2, PixelFormat::RgbaF32, &samples).unwrap();

        let plain = im.resize(0.5);
        let scrgb = im
            .copy()
            .interpretation(Interpretation::ScRgb)
            .build()
            .resize(0.5);
        let (a, b) = (
            plain.f32_samples().expect("float carrier"),
            scrgb.f32_samples().expect("float carrier"),
        );
        let alphas: Vec<f32> = (0..plain.width() as usize).map(|x| a[x * 4 + 3]).collect();
        let overshoot = alphas
            .iter()
            .position(|v| *v > 1.0)
            .unwrap_or_else(|| panic!("lanczos3 must ring above 1.0 here, got {alphas:?}"));

        assert!(
            a[overshoot * 4 + 3] > 1.0,
            "untagged: alpha {} must keep the ringing overshoot",
            a[overshoot * 4 + 3]
        );
        assert_eq!(
            b[overshoot * 4 + 3].to_bits(),
            1.0f32.to_bits(),
            "scRGB: alpha {} must clip to exactly 1.0, the scRGB max_alpha",
            b[overshoot * 4 + 3]
        );
    }

    /// Issue #664. The unsigned carriers stay on the depth-derived ceiling and
    /// do **not** route through the interpretation, mirroring what #631 did for
    /// the standalone premultiply pair. On an untagged raster the two agree
    /// anyway (`Rgba8` resolves to `Srgb` and 255, `Rgba16` to `Rgb16` and
    /// 65535), so the only thing routing them through the tag would change is a
    /// raster whose tag disagrees with its bytes, and there it would be
    /// destructive: an 8-bit buffer mislabelled `Rgb16` would premultiply
    /// against 65535 and come back black. `RasterCopyBuilder::interpretation`
    /// accepts any tag without checking the depth, so that is reachable.
    ///
    /// This one guards the **ceiling** and nothing else. Its raster is
    /// constant, and `max_alpha` cancels across the bracket on a constant, so
    /// it cannot see the `f32` re-rounding at all;
    /// [`resize_unsigned_bracket_matches_the_vips_oracle_on_varying_data`] is
    /// where that is pinned, on data that varies.
    #[test]
    fn resize_unsigned_carriers_keep_the_depth_ceiling() {
        let mut data = Vec::with_capacity(8 * 8 * 4);
        for _ in 0..8 * 8 {
            data.extend_from_slice(&[200u8, 100, 50, 128]);
        }
        let im = Raster::new(8, 8, PixelFormat::Rgba8, data).unwrap();
        let plain = im.resize(0.5);
        for tag in [
            Interpretation::ScRgb,
            Interpretation::Rgb16,
            Interpretation::Grey16,
        ] {
            let tagged = im.copy().interpretation(tag).build().resize(0.5);
            assert_eq!(
                tagged.data(),
                plain.data(),
                "an 8-bit carrier tagged {tag:?} must still bracket against 255"
            );
        }
        // And the value itself: a constant survives the bracket unchanged.
        assert_eq!(&plain.data()[..4], &[200u8, 100, 50, 128]);
    }

    /// The interior sample of
    /// `const_rgba_f32(alpha, tag).affine([0.5, 0, 0, 0.5], "bilinear")`.
    /// Bilinear at an exact half-scale lands every tap on the input grid, and
    /// the raster is constant anyway, so the interpolation contributes nothing
    /// and what comes out is the premultiply bracket alone.
    fn affine_const_pixel(alpha: f32, tag: Option<Interpretation>) -> [f32; 4] {
        let out = const_rgba_f32(alpha, tag).affine([0.5, 0.0, 0.0, 0.5], "bilinear");
        assert_eq!(out.width(), 4, "affine 0.5 of an 8x8 is 4 wide");
        assert_eq!(out.height(), 4, "affine 0.5 of an 8x8 is 4 high");
        assert_eq!(out.format(), PixelFormat::RgbaF32, "the carrier survives");
        let s = out.f32_samples().expect("float carrier");
        // Pixel (1, 1) of the 4x4 output, the one vips was read at.
        let base = (4 + 1) * 4;
        [s[base], s[base + 1], s[base + 2], s[base + 3]]
    }

    /// Issue #664, the affine half. Same rule as
    /// [`resize_float_bracket_takes_max_alpha_from_the_interpretation`], on the
    /// path with the better oracle: `vips_affine` calls `vips_premultiply`
    /// itself (`affine.c:553`), so `vips affine` on its own *is* the bracket
    /// rather than something wrapped around one, and the composed
    /// `premultiply | affine | unpremultiply` agrees with it value for value.
    ///
    /// Measured on vips 8.18.6,
    /// `vips affine in.v out.v "0.5 0 0 0.5" --interpolate bilinear` over the
    /// 8x8 constant float RGBA `(100, 20, 3, alpha)`, read at pixel (1, 1):
    ///
    /// ```text
    /// alpha  srgb (255)                                    scrgb (1.0)                                                   rgb16 (65535)
    /// 0.5    100.00000762939453 20 3.0000002384185791 0.5  100 20 3 0.5                                                  100 20 3 0.5
    /// 1.5    100.00000762939453 20 3.0000002384185791 1.5  66.666671752929688 13.333333969116211 2 1                     100 20 3 1.5
    /// 300    85 17 2.5500001907348633 255                  0.3333333432674408 0.066666670143604279 0.0099999997764825821 1  99.999992370605469 20 3 300
    /// ```
    ///
    /// `srgb` is the untagged row here rather than `multiband`, because
    /// `vips_image_hasalpha` is `bands > vips_interpretation_bands(Type)` and
    /// `vips_interpretation_bands(MULTIBAND)` is `0` (`image.c:3104`,
    /// `header.c:218`), so vips skips the bracket entirely on a 4-band
    /// MULTIBAND image and there is nothing to compare. libviprs decides from
    /// the carrier instead, and an untagged `RgbaF32` reports
    /// [`Interpretation::Srgb`] via `Interpretation::for_format`, so `srgb` is
    /// the row that matches.
    ///
    /// The `0.5` row is the cancellation guard: an alpha inside every candidate
    /// ceiling makes the bracket cancel, so all three tags agree and nothing may
    /// move there. `1.5` and `300` discriminate, because the premultiply clip
    /// and the stored-alpha clip both bite and the three ceilings separate by a
    /// factor of 300.
    #[test]
    fn affine_float_bracket_takes_max_alpha_from_the_interpretation() {
        let cases: [(f32, Option<Interpretation>, [f32; 4]); 9] = [
            (0.5, None, [100.000_01, 20.0, 3.000_000_2, 0.5]),
            (0.5, Some(Interpretation::ScRgb), [100.0, 20.0, 3.0, 0.5]),
            (0.5, Some(Interpretation::Rgb16), [100.0, 20.0, 3.0, 0.5]),
            (1.5, None, [100.000_01, 20.0, 3.000_000_2, 1.5]),
            (
                1.5,
                Some(Interpretation::ScRgb),
                [66.666_67, 13.333_334, 2.0, 1.0],
            ),
            (1.5, Some(Interpretation::Rgb16), [100.0, 20.0, 3.0, 1.5]),
            (300.0, None, [85.0, 17.0, 2.550_000_2, 255.0]),
            (
                300.0,
                Some(Interpretation::ScRgb),
                [0.333_333_34, 0.066_666_67, 0.01, 1.0],
            ),
            (
                300.0,
                Some(Interpretation::Rgb16),
                [99.999_99, 20.0, 3.0, 300.0],
            ),
        ];
        for (alpha, tag, want) in cases {
            let got = affine_const_pixel(alpha, tag);
            for (band, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (f64::from(*g) - f64::from(*w)).abs() <= f64::from(w.abs()) * 1e-6,
                    "alpha {alpha}, tag {tag:?}, band {band}: got {g}, want vips' {w}"
                );
            }
        }
    }

    /// Issue #664, the affine half of the bit-exact claim. `vips_affine`
    /// premultiplies into a **FLOAT** image (`affine.c:553`), interpolates that
    /// with `BILINEAR_FLOAT`, which accumulates in `double` and stores back to
    /// `float` (`interpolate.c:462`), and unpremultiplies the float result. So
    /// three rounding points are `f32` and only the accumulation is `f64`, and
    /// [`TapFetch::fetch`], the accumulator quantisation in
    /// [`Raster::try_affine_with`] and [`unpremultiply`] reproduce that.
    ///
    /// Pinned on the two rows where the `f64` spelling of the same expression
    /// gives a visibly different number: `100 * f32(0.5 / 255) * f32(255 / 0.5)`
    /// is `100.00000762939453` and not `100`, and `100 * f32(1 / 1.5)` is
    /// `66.666671752929688` and not `66.66666412353516`. Both are one ulp, and
    /// both are the difference between a value pinned on the binary and a value
    /// pinned on a model of it.
    #[test]
    fn affine_float_bracket_rounds_through_f32_like_the_c_macros() {
        let got = affine_const_pixel(0.5, None);
        assert_eq!(
            got[0].to_bits(),
            100.000_01f32.to_bits(),
            "band 0 came out {}, want vips' 100.00000762939453 (f64 single-rounded \
             gives exactly 100)",
            got[0]
        );
        let got = affine_const_pixel(1.5, Some(Interpretation::ScRgb));
        assert_eq!(
            got[0].to_bits(),
            66.666_67f32.to_bits(),
            "band 0 came out {}, want vips' 66.666671752929688 (f64 single-rounded \
             gives 66.66666412353516)",
            got[0]
        );
    }

    /// Issue #664. The `f32` re-rounding is **not** confined to the float
    /// carriers: `vips_premultiply` widens only a DOUBLE input to DOUBLE and
    /// writes FLOAT for everything else (`premultiply.c:229-232`), so an 8-bit
    /// or 16-bit RGBA premultiplies through an `f32` multiplier too, and the
    /// bytes that come back move. `resize_unsigned_carriers_keep_the_depth_ceiling`
    /// cannot see that, because its raster is constant and `max_alpha` cancels
    /// across the bracket on a constant. These two fixtures vary, and they are
    /// pinned on the binary.
    ///
    /// The oracle is `premultiply | resize | unpremultiply` on vips 8.18.6, read
    /// back as FLOAT and quantised with libviprs' own `round(v + 0.5)`. It has
    /// to be read as float rather than piped through `vips cast`, because
    /// `vips_cast` **truncates** a float towards zero rather than rounding it
    /// (`cast.c:237`, and the header note "now does floor(), not rint() ...
    /// you'll need to round yourself"). That is a real and separate divergence,
    /// unchanged by this issue, and the module docs record it; holding it fixed
    /// is what lets these fixtures ask only which float value is being
    /// quantised.
    ///
    /// Measured that way, on 64x64 pseudo-random `Rgba8`, the `f32` spelling
    /// agrees with vips on 65536 of 65536 samples and the `f64` spelling on
    /// 65530. The two fixtures below are the smallest cases found that carry
    /// the same discrimination:
    ///
    /// ```text
    /// Rgba8  4x4 -> resize(2.0)  sample 225  vips float 59.500003814697266  f32 -> 60      f64 -> 59
    /// Rgba8  4x4 -> resize(2.0)  sample 226  vips float 226.5               f32 -> 227     f64 -> 226
    /// Rgba16 4x4 -> resize(0.5)  sample 1    vips float 46267.5             f32 -> 46268   f64 -> 46267
    /// Rgba16 4x4 -> resize(0.5)  sample 10   vips float 40023.5             f32 -> 40024   f64 -> 40023
    /// ```
    #[test]
    fn resize_unsigned_bracket_matches_the_vips_oracle_on_varying_data() {
        #[rustfmt::skip]
        let src8: [u8; 64] = [
            83, 124, 157, 3, 66, 96, 173, 188, 252, 118, 220, 58, 6, 83, 160, 155,
            61, 64, 99, 48, 32, 122, 7, 174, 253, 7, 246, 173, 225, 220, 177, 120,
            23, 28, 63, 7, 69, 153, 163, 153, 156, 69, 162, 32, 125, 150, 34, 67,
            95, 49, 202, 30, 92, 7, 104, 102, 178, 158, 238, 79, 169, 55, 51, 198,
        ];
        #[rustfmt::skip]
        let want8: [u8; 256] = [
            62, 78, 167, 0, 255, 255, 255, 0, 63, 94, 182, 103, 68, 95, 183, 189,
            116, 107, 192, 125, 252, 142, 214, 51, 63, 94, 173, 95, 0, 76, 159, 157,
            60, 86, 179, 0, 83, 124, 157, 3, 60, 96, 171, 104, 66, 96, 173, 188,
            118, 103, 186, 129, 252, 118, 220, 58, 80, 93, 178, 98, 6, 83, 160, 155,
            72, 44, 114, 18, 63, 68, 103, 28, 35, 107, 76, 109, 48, 107, 84, 182,
            133, 73, 147, 162, 254, 32, 241, 124, 192, 88, 217, 129, 107, 147, 172, 141,
            69, 48, 124, 40, 61, 64, 99, 48, 17, 121, 4, 111, 32, 122, 7, 174,
            140, 58, 125, 185, 253, 7, 246, 173, 255, 86, 235, 146, 225, 220, 177, 120,
            57, 16, 94, 20, 53, 60, 87, 29, 35, 141, 64, 101, 46, 144, 72, 166,
            120, 86, 136, 146, 240, 7, 233, 107, 250, 92, 206, 91, 214, 229, 133, 83,
            230, 255, 255, 0, 23, 28, 63, 7, 65, 150, 159, 88, 69, 153, 163, 153,
            83, 138, 169, 99, 156, 69, 162, 32, 151, 115, 60, 42, 125, 150, 34, 67,
            85, 0, 217, 9, 84, 41, 187, 16, 79, 88, 154, 76, 81, 97, 153, 126,
            96, 118, 179, 88, 152, 159, 207, 47, 159, 98, 81, 83, 153, 73, 39, 129,
            96, 60, 227, 26, 95, 49, 202, 30, 86, 5, 116, 67, 92, 7, 104, 102,
            125, 76, 177, 88, 178, 158, 238, 79, 175, 91, 109, 137, 169, 55, 51, 198,
        ];
        let out8 = Raster::new(4, 4, PixelFormat::Rgba8, src8.to_vec())
            .unwrap()
            .resize(2.0);
        assert_eq!(
            (out8.width(), out8.height()),
            (8, 8),
            "resize(2.0) of a 4x4"
        );
        assert_eq!(
            out8.data(),
            &want8[..],
            "Rgba8 resize(2.0) must match the vips premultiply bracket; samples \
             225 and 226 are the ones the f64 spelling gets wrong"
        );

        #[rustfmt::skip]
        let src16: [u16; 64] = [
            5811, 45664, 6410, 60453, 41496, 17556, 1957, 5372,
            8592, 58651, 44857, 50526, 63461, 32455, 56025, 31877,
            53299, 13787, 42485, 1040, 22692, 34103, 23367, 28187,
            22750, 30003, 17440, 5032, 27552, 27809, 28992, 61172,
            35549, 36785, 43531, 26206, 57364, 39895, 55208, 19333,
            6402, 40411, 48481, 13289, 61001, 43977, 21054, 60646,
            61466, 51347, 9527, 29216, 18132, 6665, 62257, 26093,
            11912, 1600, 9749, 50747, 28458, 19948, 39624, 56728,
        ];
        #[rustfmt::skip]
        let want16: [u16; 16] = [
            7692, 46268, 15071, 22530, 35811, 42627, 40829, 34163,
            40590, 27893, 40024, 20702, 31511, 21821, 26686, 44624,
        ];
        let bytes16: Vec<u8> = src16.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let out16 = Raster::new(4, 4, PixelFormat::Rgba16, bytes16)
            .unwrap()
            .resize(0.5);
        assert_eq!(
            (out16.width(), out16.height()),
            (2, 2),
            "resize(0.5) of a 4x4"
        );
        let got16: Vec<u16> = out16
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_ne_bytes(*c))
            .collect();
        assert_eq!(
            got16,
            want16.to_vec(),
            "Rgba16 resize(0.5) must match the vips premultiply bracket; samples \
             1 and 10 are the ones the f64 spelling gets wrong"
        );
    }

    /// Issue #664, the affine bracket on data that varies. The constant table
    /// in [`affine_float_bracket_takes_max_alpha_from_the_interpretation`]
    /// isolates the ceiling by making the interpolation contribute nothing,
    /// which is exactly what makes it blind to the third rounding point:
    /// `vips_affine_gen` writes each interpolated premultiplied pixel into a
    /// **FLOAT** image before `vips_unpremultiply` reads it back, so the
    /// accumulator is quantised to `f32` at that seam, and on a constant raster
    /// it already is. Drop the quantisation and 9 of the 64 samples below move.
    ///
    /// Measured on vips 8.18.6:
    /// `vips affine in.v out.v "0.8 0.15 -0.15 0.8" --interpolate bilinear` over
    /// a 4x4 float RGBA tagged `scrgb`, whose alpha runs 0.25, 0.5, 1.25, 0.75
    /// so the `max_alpha` clip bites on a quarter of the taps and the lanczos
    /// -free bilinear stencil still lands off the grid everywhere. All 64
    /// samples are bit-exact, which is the claim the module docs make.
    #[test]
    fn affine_float_bracket_matches_the_vips_oracle_on_varying_data() {
        let mut src = Vec::with_capacity(4 * 4 * 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let alpha = [0.25f32, 0.5, 1.25, 0.75][((x + y) % 4) as usize];
                src.extend_from_slice(&[
                    10.0 * (x + 1) as f32,
                    20.0 * (y + 1) as f32,
                    5.0 * (x + y + 1) as f32,
                    alpha,
                ]);
            }
        }
        #[rustfmt::skip]
        let want: [f32; 64] = [
            0.0, 0.0, 0.0, 0.0,
            22.62857, 17.371428, 11.314285, 0.015574938,
            32.284264, 18.071066, 16.142132, 0.22792809,
            40.000004, 20.000002, 20.000002, 0.053399786,
            10.0, 20.0, 5.0, 0.25,
            20.33662, 23.382473, 11.568194, 0.7667319,
            29.72189, 24.059332, 16.366016, 0.8160377,
            40.0, 28.275862, 22.068966, 0.15485938,
            9.208634, 43.16547, 10.791368, 0.50720894,
            17.1305, 40.88904, 14.472023, 1.0,
            31.869556, 49.369026, 23.277035, 0.41892132,
            40.0, 58.800003, 29.700003, 0.28479886,
            8.597285, 57.556557, 14.389139, 0.57039875,
            15.471698, 66.273186, 19.503305, 0.5518868,
            29.741463, 78.46035, 29.485819, 0.46128514,
            32.0, 64.0, 28.0, 0.93983626,
        ];
        let out = Raster::from_f32_samples(4, 4, PixelFormat::RgbaF32, &src)
            .unwrap()
            .copy()
            .interpretation(Interpretation::ScRgb)
            .build()
            .affine([0.8, 0.15, -0.15, 0.8], "bilinear");
        assert_eq!((out.width(), out.height()), (4, 4), "the vips output area");
        let got = out.f32_samples().expect("float carrier");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "sample {i} (pixel {}, band {}) came out {g}, want vips' {w}",
                i / 4,
                i % 4
            );
        }
    }

    /// Issue #667. `vips_affine` builds the border it interpolates against by
    /// embedding the input with the caller's extend mode before it resamples
    /// (`"extend", affine->extend` in the `vips_embed` call at
    /// `libvips/resample/affine.c:534`), so a white tap is inked exactly the
    /// way `vips_embed` inks one: [`white_ink`], the interpretation's max
    /// alpha laid down by whichever paint mechanism the carrier picks.
    ///
    /// **On a raster without an alpha band**, which is every fixture below.
    /// That scope is the whole content of the claim, not a caveat on it: once
    /// alpha is present vips premultiplies into float before it paints the
    /// border and the two stop agreeing, which
    /// `affine_white_on_an_alpha_raster_keeps_the_memset_ink` below pins.
    ///
    /// Measured on 8.18.6 with `vips affine in.v out.v "1 0 0 1" --extend
    /// white --oarea "-1 -1 4 4" --interpolate nearest`, reading the corner.
    /// It is the same table `vips embed --extend white` gives, cell for cell,
    /// which is the point:
    ///
    /// ```text
    /// carrier  multiband/b-w  srgb   rgb16  grey16  scrgb
    /// uchar    255            255    255    255     1
    /// ushort   65535          65535  65535  65535   257
    /// float    255            255    65535  65535   1
    /// ```
    ///
    /// The `grey16` column is the other one the old depth rule got wrong: its
    /// float cell is 65535 where the depth gave 255, and nothing else in the
    /// suite reaches it.
    ///
    /// Nearest on the `-1` ring reads one tap and nothing else, so the corner
    /// is the ink itself rather than a blend of it, and no premultiply
    /// round-trip stands between the ink and the assertion.
    #[test]
    fn affine_white_taps_ink_from_the_interpretation() {
        use Interpretation as I;
        // (tag, uchar corner, ushort corner, float corner)
        let cases = [
            (None, 255.0, 65535.0, 255.0),
            (Some(I::Srgb), 255.0, 65535.0, 255.0),
            (Some(I::Rgb16), 255.0, 65535.0, 65535.0),
            (Some(I::Grey16), 255.0, 65535.0, 65535.0),
            (Some(I::ScRgb), 1.0, 257.0, 1.0),
        ];
        let float1 = PixelFormat::FloatF32(core::num::NonZeroU16::new(1).unwrap());
        for (tag, want8, want16, wantf) in cases {
            let carriers = [
                (
                    Raster::new(2, 2, PixelFormat::Gray8, vec![7; 4]).unwrap(),
                    want8,
                ),
                (
                    Raster::new(2, 2, PixelFormat::Gray16, 7u16.to_ne_bytes().repeat(4)).unwrap(),
                    want16,
                ),
                (
                    Raster::from_f32_samples(2, 2, float1, &[7.0; 4]).unwrap(),
                    wantf,
                ),
            ];
            for (im, want) in carriers {
                let fmt = im.format();
                let im = match tag {
                    Some(t) => im.copy().interpretation(t).build(),
                    None => im,
                };
                let out = im
                    .try_affine_with(
                        [1.0, 0.0, 0.0, 1.0],
                        Interpolator::Nearest,
                        AffineOptions {
                            oarea: Some([-1, -1, 4, 4]),
                            extend: Extend::White,
                            ..AffineOptions::default()
                        },
                    )
                    .unwrap();
                assert_eq!(out.getpoint(0, 0), vec![want], "{fmt:?} tagged {tag:?}");
                assert_eq!(
                    out.getpoint(1, 1),
                    vec![7.0],
                    "the image itself resamples unchanged"
                );
            }
        }
    }

    /// Issue #692, the border ink on an alpha raster, and the reason this
    /// module does not follow vips there.
    ///
    /// The cause is **not** the paint order. `vips_affine_build` embeds before
    /// it premultiplies (`affine.c:529` then `affine.c:551`), so the ink is
    /// memset into the raster's own domain either way and `vips_embed` on its
    /// own gives the same byte pattern. What moves it is that the premultiply
    /// pair does not cancel on that pixel: `vips_premultiply` takes a
    /// **clipped** alpha into its multiplier, `nalpha = clip(a, 0, M) / M`,
    /// and `vips_unpremultiply` takes the **raw** one into its reciprocal,
    /// `factor = M / a`. Every band of a border pixel holds the same ink `E`,
    /// so the round trip is `E * clip(E, 0, M) / M * M / E`, which is
    /// `clip(E, 0, M)`. This module runs the same arithmetic against its own
    /// ceiling, the depth's on an unsigned carrier (issue #664), and the white
    /// ink never exceeds that, so `clip(E, 0, D)` is `E`.
    ///
    /// Measured on 8.18.6 with `--interpolate nearest`, whose window is one
    /// pixel, so an output shifted one step off the input reads the **pure**
    /// ink and nothing has to be solved back out of a blend. Both columns are
    /// `vips affine in.v out.v "1 0 0 1" --interpolate nearest --idx 1 --idy 1
    /// --extend white`, read at pixel `(0, 0)`.
    ///
    /// ```text
    /// carrier  bands  tag      vips affine white   libviprs
    /// uchar    4      srgb     255                 255      agree
    /// uchar    4      scrgb    1                   1        agree
    /// ushort   4      rgb16    65535               65535    agree
    /// ushort   4      grey16   65535               65535    agree
    /// ushort   4      srgb     255                 65535    DIFFER
    /// ushort   4      scrgb    1                   257      DIFFER
    /// ushort   4      b-w      255                 65535    DIFFER
    /// float    4      srgb     255                 255      agree
    /// float    4      scrgb    1                   1        agree
    /// float    4      rgb16    65535               65535    agree
    /// ```
    ///
    /// Every differing cell is a 16-bit raster wearing an 8-bit tag, which is
    /// the same condition #664 is about, and the agreeing cells are here as the
    /// positive control: the comparison would report a difference if there were
    /// one, so a divergence confined to three rows is a measurement rather than
    /// a blind spot.
    ///
    /// [`affine_alpha_bracket_holds_the_image_where_vips_collapses_it`] carries
    /// the reason not to follow.
    ///
    /// [`affine_alpha_bracket_holds_the_image_where_vips_collapses_it`]: self::affine_alpha_bracket_holds_the_image_where_vips_collapses_it
    #[test]
    fn affine_white_on_an_alpha_raster_inks_the_depth_not_the_interpretation() {
        use Interpretation as I;
        // (format, tag, what libviprs inks, what vips 8.18.6 inks)
        let cases = [
            (PixelFormat::Rgba8, I::Srgb, 255.0, 255.0),
            (PixelFormat::Rgba8, I::ScRgb, 1.0, 1.0),
            (PixelFormat::Rgba8, I::Rgb16, 255.0, 255.0),
            (PixelFormat::Rgba16, I::Rgb16, 65535.0, 65535.0),
            (PixelFormat::Rgba16, I::Grey16, 65535.0, 65535.0),
            (PixelFormat::Rgba16, I::Srgb, 65535.0, 255.0),
            (PixelFormat::Rgba16, I::ScRgb, 257.0, 1.0),
            (PixelFormat::Rgba16, I::Bw, 65535.0, 255.0),
            (PixelFormat::RgbaF32, I::Srgb, 255.0, 255.0),
            (PixelFormat::RgbaF32, I::ScRgb, 1.0, 1.0),
            (PixelFormat::RgbaF32, I::Rgb16, 65535.0, 65535.0),
        ];
        let mut differing = 0usize;
        for (format, tag, libviprs, vips) in cases {
            let value = if format == PixelFormat::RgbaF32 && tag == I::ScRgb {
                0.5
            } else {
                7.0
            };
            let bytes: Vec<u8> = (0..2 * 2 * 4)
                .flat_map(|_| match format {
                    PixelFormat::Rgba8 => vec![value as u8],
                    PixelFormat::Rgba16 => (value as u16).to_ne_bytes().to_vec(),
                    _ => (value as f32).to_ne_bytes().to_vec(),
                })
                .collect();
            let im = Raster::new(2, 2, format, bytes)
                .unwrap()
                .copy()
                .interpretation(tag)
                .build();
            let out = im
                .try_affine_with(
                    [1.0, 0.0, 0.0, 1.0],
                    Interpolator::Nearest,
                    AffineOptions {
                        oarea: Some([-1, -1, 4, 4]),
                        extend: Extend::White,
                        ..AffineOptions::default()
                    },
                )
                .unwrap();
            assert_eq!(
                out.getpoint(0, 0),
                vec![libviprs; 4],
                "{format:?} tagged {tag:?}: the ink follows the depth ceiling; \
                 vips 8.18.6 gives {vips} because its ceiling is the tag's (#692)"
            );
            assert_eq!(
                out.getpoint(1, 1),
                vec![value; 4],
                "{format:?} tagged {tag:?}: the image itself resamples unchanged"
            );
            differing += usize::from(libviprs != vips);
        }
        assert_eq!(
            differing, 3,
            "exactly three of these eleven cells diverge, and all three are a \
             16-bit carrier under an 8-bit tag"
        );
    }

    /// Issue #692, the reason the cells above are left alone.
    ///
    /// Following vips on the border means following it on the ceiling, because
    /// the border comes out at `clip(ink, 0, ceiling)` and nothing else sets
    /// it. Pre-clipping only the fill would fix the pure-ink pixel and leave
    /// every blended one wrong, since the two premultiplied spaces are scaled
    /// differently: on a `ushort` `srgb` raster with alpha 200, a colour of
    /// 25000 sits at 19608 in vips' premultiplied image and at 76.3 in this
    /// module's. So the real choice is whether `bracket_max_alpha` follows the
    /// tag on an unsigned carrier, which is #664's question, and its price is
    /// measured rather than argued.
    ///
    /// `vips affine in.v out.v "1 0 0 1" --interpolate nearest --idx 1 --idy 1`
    /// on 8.18.6 over a constant `ushort` RGBA raster, read at an interior
    /// pixel that never touches the border:
    ///
    /// ```text
    /// tag     colour  alpha   vips interior      libviprs interior
    /// srgb    25000   200     25000, alpha 200   25000, alpha 200
    /// srgb    25000   25000   255,   alpha 255   25000, alpha 25000
    /// srgb    25000   65535   97,    alpha 255   25000, alpha 65535
    /// scrgb   25000   25000   1,     alpha 1     25000, alpha 25000
    /// rgb16   25000   25000   25000, alpha 25000 25000, alpha 25000
    /// ```
    ///
    /// The `srgb` and `scrgb` rows with a real alpha are the cost: vips reads a
    /// 16-bit raster tagged `srgb` as having alpha in `0..=255`, so an ordinary
    /// alpha of 25000 clips and the whole image collapses to the tag's ceiling.
    /// It is not a border effect and it is not confined to unusual data. The
    /// `rgb16` row is the positive control, the one tag whose ceiling matches
    /// the depth, and there the two agree everywhere including the border.
    ///
    /// This test pins the libviprs column. The vips column is in the doc
    /// comment because reproducing it would mean adopting the behaviour.
    #[test]
    fn affine_alpha_bracket_holds_the_image_where_vips_collapses_it() {
        use Interpretation as I;
        for (tag, alpha) in [
            (I::Srgb, 200u16),
            (I::Srgb, 25000),
            (I::Srgb, 65535),
            (I::ScRgb, 25000),
            (I::Rgb16, 25000),
        ] {
            let bytes: Vec<u8> = (0..2 * 2)
                .flat_map(|_| {
                    let mut px = Vec::new();
                    for _ in 0..3 {
                        px.extend_from_slice(&25000u16.to_ne_bytes());
                    }
                    px.extend_from_slice(&alpha.to_ne_bytes());
                    px
                })
                .collect();
            let out = Raster::new(2, 2, PixelFormat::Rgba16, bytes)
                .unwrap()
                .copy()
                .interpretation(tag)
                .build()
                .try_affine_with(
                    [1.0, 0.0, 0.0, 1.0],
                    Interpolator::Nearest,
                    AffineOptions {
                        oarea: Some([-1, -1, 4, 4]),
                        extend: Extend::White,
                        ..AffineOptions::default()
                    },
                )
                .unwrap();
            assert_eq!(
                out.getpoint(1, 1),
                vec![25000.0, 25000.0, 25000.0, f64::from(alpha)],
                "Rgba16 tagged {tag:?} with alpha {alpha}: the premultiply \
                 bracket must round-trip the interior; vips collapses it to \
                 the tag's ceiling (#692, #664)"
            );
        }
    }

    // -----------------------------------------------------------------
    // Issue #668: the sub-pixel offset quantisation
    // -----------------------------------------------------------------

    /// A single-band float raster whose samples are `(i * 37 + 11) % 251`, a
    /// cheap deterministic sequence with enough high-frequency content that a
    /// fraction-of-a-pixel shift in the resampling offset shows up as whole
    /// units in the output. Every #668 fixture below is built from it, so the
    /// pinned numbers can be regenerated from the shape alone.
    fn offset_ramp(w: u32, h: u32) -> Raster {
        let n = (w * h) as usize;
        let data: Vec<u8> = (0..n)
            .flat_map(|i| (((i * 37 + 11) % 251) as f32).to_ne_bytes())
            .collect();
        Raster::new(
            w,
            h,
            PixelFormat::FloatF32(core::num::NonZeroU16::new(1).unwrap()),
            data,
        )
        .unwrap()
    }

    /// Read a float raster back as `f32` samples in row-major order.
    fn float_samples(r: &Raster) -> Vec<f32> {
        r.data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect()
    }

    /// Compare against a pinned oracle row with an absolute tolerance. The
    /// tolerance is 1e-3 on data spanning 0..251, which is four orders of
    /// magnitude below the divergences #668 is about (0.12 to 1.4) and two
    /// above the f32 accumulation noise between two orderings of the same
    /// convolution (up to 4.6e-5 measured).
    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: sample count");
        let mut worst = (0usize, 0.0f64);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (f64::from(*g) - f64::from(*w)).abs();
            if d > worst.1 {
                worst = (i, d);
            }
        }
        assert!(
            worst.1 <= 1e-3,
            "{what}: sample {} is {} where vips 8.18.6 gives {}, off by {}",
            worst.0,
            got[worst.0],
            want[worst.0],
            worst.1
        );
    }

    /// Issue #668. `vips_reduceh` and `vips_reducev` never evaluate the kernel
    /// at the true sub-pixel offset. They round it onto a 65-entry table and
    /// look the mask up (`reduceh.cpp:270-276`, `reducev.cpp` the same five
    /// lines):
    ///
    /// ```c
    /// const int sx = X * VIPS_TRANSFORM_SCALE * 2;
    /// const int six = sx & (VIPS_TRANSFORM_SCALE * 2 - 1);
    /// const int tx = (six + 1) >> 1;
    /// const double *cxf = reduceh->matrixf[tx];
    /// ```
    ///
    /// `VIPS_TRANSFORM_SCALE` is 64 (`interpolate.h:109`) and the tables come
    /// from `vips_reduce_make_mask(..., (float) x / VIPS_TRANSFORM_SCALE)`, so
    /// the offset is a multiple of 1/64 before a single coefficient exists.
    /// Whenever the offsets land on that grid the quantisation is invisible,
    /// which is why every dyadic factor and 2.5 agree without it and why this
    /// went unnoticed. At 4/3 the offsets are thirds and they do not.
    ///
    /// Measured on 8.18.6, `vips reduceh in.v out.v 1.3333333333333333 --gap 0`
    /// and the `reducev` twin on the transpose, both axes giving the same 12
    /// samples.
    #[test]
    fn reduce_quantises_the_sub_pixel_offset_onto_the_vips_table_grid() {
        #[rustfmt::skip]
        let want: [f32; 12] = [
            17.01585, 66.51737, 115.9886, 160.9282, 233.3695, 72.89664,
            48.0719, 114.4463, 155.2341, 233.6704, 133.9826, 34.32521,
        ];
        let shrink = 4.0 / 3.0;
        let h = offset_ramp(16, 1).reduceh(shrink, "lanczos3");
        assert_eq!((h.width(), h.height()), (12, 1), "reduceh output size");
        assert_close(&float_samples(&h), &want, "reduceh 4/3");

        let v = offset_ramp(1, 16).reducev(shrink, "lanczos3");
        assert_eq!((v.width(), v.height()), (1, 12), "reducev output size");
        assert_close(&float_samples(&v), &want, "reducev 4/3");
    }

    /// Issue #668. The same quantisation seen through `resize`, which is where
    /// it was reported. `vips_resize` splits a downscale into an integer
    /// `vips_shrink` and a residual `vips_reduce` (`resize.c:211-231` with
    /// `gap` 2.0), and the shrink half is exact, so a resize survives exactly
    /// when its residual reduce factor lands on the 1/64 grid.
    ///
    /// 0.75 of a 64-wide raster leaves a residual of 4/3 and 0.37 leaves
    /// 2.7027, both off the grid. Measured on 8.18.6 as
    /// `vips resize in.v out.v 0.75` on an 8x8 and `... 0.37` on a 16x16,
    /// giving 6x6 either way.
    #[test]
    fn resize_matches_the_oracle_at_non_dyadic_downscales() {
        #[rustfmt::skip]
        let want_075: [f32; 36] = [
            24.30509, 74.25901, 116.5843, 179.362, 220.1431, 57.59824,
            83.55322, 130.8575, 203.9105, 231.5954, 71.85882, 64.76923,
            142.7602, 211.4884, 99.14769, 69.8261, 74.73444, 139.4226,
            224.9046, 97.87094, 27.30505, 91.89622, 164.8316, 225.7807,
            103.0888, 41.04671, 113.7598, 181.8064, 158.7755, 121.0924,
            57.02991, 125.3291, 180.6986, 184.0041, 23.96046, 41.27575,
        ];
        let out = offset_ramp(8, 8).resize(0.75);
        assert_eq!((out.width(), out.height()), (6, 6), "resize 0.75 of an 8x8");
        assert_close(&float_samples(&out), &want_075, "resize 0.75");

        #[rustfmt::skip]
        let want_037: [f32; 36] = [
            93.11782, 150.1574, 128.6156, 115.19, 128.7295, 129.0424,
            135.7788, 119.4614, 106.3704, 131.3192, 118.9644, 123.6912,
            135.6535, 142.3375, 125.4933, 108.8913, 136.3413, 122.529,
            132.3722, 119.7645, 115.8759, 133.9125, 113.9134, 131.4173,
            133.7992, 120.6193, 119.2597, 118.8885, 137.8598, 122.573,
            107.2291, 117.8667, 117.5502, 127.2929, 121.129, 133.1677,
        ];
        let out = offset_ramp(16, 16).resize(0.37);
        assert_eq!(
            (out.width(), out.height()),
            (6, 6),
            "resize 0.37 of a 16x16"
        );
        assert_close(&float_samples(&out), &want_037, "resize 0.37");
    }

    /// Issue #668, the upscale half. `vips_resize` enlarges with
    /// `vips_affine` and the bicubic interpolator (`resize.c:233-305`), and
    /// `vips_interpolate_bicubic_interpolate` rounds its offset onto the same
    /// 65-entry grid with the same five lines (`bicubic.cpp:496-519`), so a
    /// scale of 2 lands on halves and agrees while 1.5 lands on thirds and
    /// does not.
    ///
    /// Measured on 8.18.6 as `vips resize in.v out.v 1.5` on a 4x4, and the
    /// numbers are identical to `vips affine in.v out.v "1.5 0 0 1.5"
    /// --interpolate bicubic --idx 0.5 --idy 0.5 --extend copy --premultiplied`,
    /// which is what pins this on the interpolator rather than on anything
    /// resize wraps around it.
    #[test]
    fn resize_matches_the_oracle_at_a_non_dyadic_upscale() {
        #[rustfmt::skip]
        let want: [f32; 36] = [
            -0.5625, 5.928774, 31.93805, 56.26953, 84.52558, 122.2951,
            28.47278, 34.96406, 60.97334, 88.4408, 107.2803, 107.4352,
            144.2324, 150.7237, 176.733, 216.6933, 198.0198, 48.32856,
            105.1875, 111.4869, 136.7635, 179.668, 187.2513, 106.6858,
            66.14259, 73.01816, 100.4948, 107.3215, 117.8489, 156.415,
            181.9022, 191.0794, 227.3439, 129.7319, 32.88796, 71.45403,
        ];
        let out = offset_ramp(4, 4).resize(1.5);
        assert_eq!((out.width(), out.height()), (6, 6), "resize 1.5 of a 4x4");
        assert_close(&float_samples(&out), &want, "resize 1.5");
    }

    /// Issue #668, the sign of the truncation. Porting the vips lines
    /// literally is not enough, because `(int)(X * 128)` truncates toward zero
    /// and `& 127` reads two's complement, so a negative coordinate picks the
    /// bucket one above the one `floor` would pick. vips never meets that case
    /// in `vips_affine_gen`: it works in the embedded space shifted by
    /// `window_offset` (`affine.c:361-362`), where the coordinate is always
    /// at least 1. libviprs keeps the unshifted coordinate, which goes
    /// negative on the first output column whenever `1/scale < 0.5`, so the
    /// quantiser has to floor rather than truncate to agree on both signs.
    ///
    /// This fixture is the smallest one that separates all three answers.
    /// Measured on 8.18.6 as `vips resize in.v out.v 3 --vscale 1` on a 6x1,
    /// against the three candidate implementations:
    ///
    /// ```text
    /// exact offset, no quantisation   12 of 18 wrong, max 0.217   <- the bug
    /// quantised with trunc and mask    1 of 18 wrong, max 0.123   <- column 1 only
    /// quantised with floor and rem     0 of 18 wrong, max 1.9e-06
    /// ```
    #[test]
    fn resize_quantises_a_negative_source_coordinate_the_way_vips_shifted_space_does() {
        #[rustfmt::skip]
        let want: [f32; 18] = [
            8.6875, 8.819399, 15.17877, 27.1875, 41.18805, 54.35938,
            66.5, 78.64062, 91.35938, 103.5, 115.6406, 128.3594,
            140.5, 152.6406, 165.812, 179.8125, 191.8212, 198.1806,
        ];
        let out = offset_ramp(6, 1).resize_with(
            3.0,
            ResizeOptions {
                vscale: Some(1.0),
                ..ResizeOptions::default()
            },
        );
        assert_eq!(
            (out.width(), out.height()),
            (18, 1),
            "resize 3 horizontally of a 6x1"
        );
        assert_close(&float_samples(&out), &want, "resize 3.0, vscale 1");
    }

    /// Issue #668, the guard on the other side. Adding the quantisation must
    /// not move a scale that was already exact, and the grid-aligned scales
    /// are most of the ones anything else in this suite pins. `resize 0.5` of
    /// a 64-wide raster reduces by exactly 2, whose offset is 0 at every
    /// output position, so the table lookup and the exact evaluation are the
    /// same mask and nothing may move. Measured on 8.18.6.
    #[test]
    fn resize_holds_the_grid_aligned_scales_still() {
        #[rustfmt::skip]
        let want: [f32; 16] = [
            45.4882, 138.7604, 208.2614, 87.58178,
            158.6375, 160.3354, 93.09722, 101.5202,
            158.9667, 45.5028, 123.4247, 197.9885,
            64.97791, 155.1446, 155.4203, 52.43541,
        ];
        let out = offset_ramp(8, 8).resize(0.5);
        assert_eq!((out.width(), out.height()), (4, 4), "resize 0.5 of an 8x8");
        assert_close(&float_samples(&out), &want, "resize 0.5");
    }

    /// Issue #668, the guard against over-reach. Only the reduce masks and the
    /// bicubic interpolator read from a table.
    /// `vips_interpolate_bilinear_interpolate` computes `X = x - ix` straight
    /// from the coordinate with no table at all (`interpolate.c:538` and the
    /// `BILINEAR_FLOAT` macro), and nohalo and lbb are nonlinear and have no
    /// tables either. So bilinear must keep the exact offset, at a scale whose
    /// offsets are thirds and would visibly move if it did not.
    ///
    /// Measured on 8.18.6 as `vips affine in.v out.v "1.5 0 0 1.5"
    /// --interpolate bilinear --idx 0.5 --idy 0.5 --extend copy
    /// --premultiplied` on a 4x4.
    #[test]
    fn affine_bilinear_keeps_the_exact_sub_pixel_offset() {
        #[rustfmt::skip]
        let want: [f32; 36] = [
            11.0, 17.16667, 41.83333, 66.5, 91.16666, 115.8333,
            35.66667, 41.83333, 66.5, 91.16666, 108.8611, 105.6389,
            134.3333, 140.5, 165.1667, 189.8333, 179.6389, 64.86111,
            107.5, 113.6667, 138.3333, 163.0, 166.75, 107.75,
            80.66666, 86.83334, 111.5, 115.25, 119.0, 143.6667,
            179.3333, 185.5, 210.1667, 130.25, 50.33333, 75.0,
        ];
        let out = offset_ramp(4, 4)
            .try_affine_with(
                [1.5, 0.0, 0.0, 1.5],
                Interpolator::Bilinear,
                AffineOptions {
                    idx: 0.5,
                    idy: 0.5,
                    extend: Extend::Copy,
                    premultiplied: true,
                    ..AffineOptions::default()
                },
            )
            .unwrap();
        assert_eq!((out.width(), out.height()), (6, 6), "affine 1.5 of a 4x4");
        assert_close(&float_samples(&out), &want, "affine 1.5 bilinear");
    }
}
