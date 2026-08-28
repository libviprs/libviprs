//! Convolution and correlation operations ported from libvips.
//!
//! This module is the next batch of the libvips operation surface required
//! by the ported integration tests (after [`crate::bands`],
//! [`crate::arithmetic`], [`crate::extract`], [`crate::conversion`],
//! [`crate::draw`], [`crate::histogram`], [`crate::imageio`],
//! [`crate::composite`], [`crate::colour`], [`crate::morphology`],
//! [`crate::mosaicing`], and [`crate::create`]): 2D convolution with a
//! mask at integer or float precision, separable convolution, rotating
//! compass convolution, Gaussian blur, unsharp-mask sharpening, the two
//! template correlations, the three named edge detectors, and the Canny
//! edge detector. Operations that can fail on caller input exist in two
//! forms, following the established convention:
//!
//! * a fallible `try_*` method returning `Result<_, ConvolutionError>`
//!   with typed errors for bad kernels and unsupported shapes; and
//! * a panicking convenience method matching the ported-test call surface
//!   (`conv`, `convsep`, `compass`, `gaussblur`, `sharpen`, `spcor`,
//!   `fastcor`) exactly, delegating to the `try_*` form. The edge
//!   detectors (`sobel`, `scharr`, `prewitt`) and `canny` keep the same
//!   pair even though no ported test reaches them.
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::conv`] | `vips_conv` | convolved image |
//! | [`Raster::convsep`] | `vips_convsep` | separably convolved image |
//! | [`Raster::compass`] | `vips_compass` | combined rotating convolutions |
//! | [`Raster::gaussblur`] | `vips_gaussblur` | Gaussian-blurred image |
//! | [`Raster::sharpen`] | `vips_sharpen` | unsharp-masked image |
//! | [`Raster::spcor`] | `vips_spcor` | normalised cross-correlation surface |
//! | [`Raster::fastcor`] | `vips_fastcor` | sum-of-squared-differences surface |
//! | [`Raster::sobel`] | `vips_sobel` | Sobel edge map, always uchar |
//! | [`Raster::scharr`] | `vips_scharr` | Scharr edge map, always uchar |
//! | [`Raster::prewitt`] | `vips_prewitt` | Prewitt edge map, always uchar |
//! | [`Raster::canny`] | `vips_canny` | suppressed gradient magnitude |
//! | [`Kernel::gaussmat`] | `vips_gaussmat` | Gaussian mask |
//! | [`Kernel::logmat`] | `vips_logmat` | Laplacian-of-Gaussian mask |
//!
//! # Semantics shared with libvips
//!
//! * **Masks.** A [`Kernel`] is the double matrix plus its `scale`
//!   divisor, exactly the pair a libvips matrix image carries. Each output
//!   pixel of [`Raster::conv`] is `sum(mask[i] * pixel[i]) / scale`
//!   (`convolution/conv.c`). The scale must be finite and non-zero.
//!   libvips matrix images also carry an `offset` summand, added once per
//!   output sample after the division; the engine honours it and carries
//!   it alongside the scale on its internal mask, but the ported surface
//!   builds `Kernel` as a two-field struct literal, so the offset is `0`
//!   on every public entry point, the value every mask in the ported
//!   suites has.
//! * **Precision.** [`Precision::Float`] is `vips_convf`: coefficients are
//!   baked as `mask / scale`, accumulation is `f64`, and the result is a
//!   32-bit float image regardless of the input depth (libvips promotes
//!   int inputs to `float`; its `double` case has no libviprs depth).
//!   [`Precision::Integer`] is the `vips_convi` C path: every coefficient
//!   is `rint()`-ed (`vips__image_intize`), the sum is accumulated in
//!   `i64`, and `(sum + scale / 2) / scale` is written back in the input
//!   format, clipped to its range. That divisor is `rint()` of the mask's
//!   **own** scale, not the brightness-corrected one `vips__image_intize`
//!   computes alongside the coefficients: `vips_convi_gen` reads the
//!   scale and the offset straight off the original mask
//!   (`convolution/convi.c:757-760`) and never sees the intized copy's
//!   metadata. Dividing by the corrected scale instead was issue #547.
//!   A float input under integer precision keeps the float path of
//!   `vips_convi_gen`: `f64` accumulation with the integer mask and no
//!   clipping.
//! * **Edges.** The input is notionally extended by replicating its edge
//!   pixels (`vips_embed` with `VIPS_EXTEND_COPY`, exactly as
//!   `vips_convi_build` / `vips_convf_build` / `vips_correlation_build`
//!   do), so every output has the same dimensions as the input and the
//!   output pixel at `(x, y)` sees the input window whose top-left corner
//!   is `(x - mask_w / 2, y - mask_h / 2)`.
//! * **Separable convolution.** `convsep` requires a `1xN` or `Nx1` mask
//!   and convolves twice, the second pass with the mask rotated 90
//!   degrees, each pass applying the mask scale (`convolution/convsep.c`).
//! * **Compass.** `compass` convolves `times` times, rotating the mask by
//!   `angle` between rounds with the exact [`Raster::rot45`] ring
//!   permutation, then combines the absolute results: [`Combine::Max`] is
//!   the pixelwise maximum (`vips_bandrank`), [`Combine::Sum`] the
//!   pixelwise sum (`vips_sum`). Summed unsigned results are carried one
//!   depth wider (`Gray8 -> Gray16` and so on, saturating at 16 bits:
//!   libviprs has no 32-bit unsigned depth).
//! * **Correlation.** `fastcor` writes the per-band sum of squared
//!   differences between the template and the input window centred on the
//!   output pixel; `spcor` writes the normalised cross-correlation in
//!   `-1..1` with the libvips "constant reference is uncorrelated" rule.
//!   Both accept any mix of unsigned and float inputs with equal band
//!   counts. libvips stores `fastcor` of int images as 32-bit unsigned
//!   sums (wrapping); libviprs has no u32 depth, so both correlations
//!   return 32-bit float images. The wrap-around of the u32 accumulator is
//!   reproduced before the float store.
//! * **Sharpen.** `vips_sharpen` transforms to LabS, blurs the L channel
//!   with a separable integer Gaussian (`sigma`, min_ampl `0.1`), maps the
//!   difference through the m1/m2 lookup curve (x1 `2.0`, brightening cap
//!   `10`, darkening cap `20` L* units), and converts back to the original
//!   interpretation. With `m1 == 0` and `m2 == 0` the curve is identically
//!   zero and the operation is an exact identity for 8-bit sources: the
//!   [`Raster::colourspace`] LabS round trip is byte-exact for every 8-bit
//!   sRGB and mono value (verified exhaustively over all 256^3 sRGB
//!   triples). 16-bit sources round-trip within LabS quantisation, as in
//!   libvips.
//! * **Edge detectors.** [`Raster::sobel`], [`Raster::scharr`] and
//!   [`Raster::prewitt`] are one abstract op in libvips
//!   (`convolution/edge.c:49-63`) differing only in a 3x3 mask, and they
//!   take no arguments at all. Each convolves with its mask and with the
//!   mask rotated 90 degrees, then combines the two gradients, and the
//!   combine rule depends on the input format (`edge.c:186-200`). A uchar
//!   input takes the fast arm: the mask is stamped `scale = 2,
//!   offset = 128`, both convolutions run at [`Precision::Integer`], and
//!   the responses combine as `|Gx| + |Gy|` clipped at 255
//!   (`edge.c:97-103`). Every other format takes the accurate arm: two
//!   [`Precision::Float`] convolutions with the raw mask, then
//!   `sqrt(Gx^2 + Gy^2)`, then a **truncating** cast to uchar
//!   (`edge.c:158-182`, `conversion/cast.c:568`). The two arms are not two
//!   spellings of one formula: on a corner where `Gx == Gy` the abs sum is
//!   `2 * g` where the magnitude is `sqrt(2) * g`. The output is uchar for
//!   every input format, keeping the band count, the dimensions and the
//!   metadata. Saturation on the uchar arm happens twice, once inside each
//!   convolution (which bounds the recovered gradient to `-256..=254`, an
//!   asymmetric range) and once on the abs sum at 255.
//! * **The edge float arm rounds to `f32` twice, and both roundings move
//!   output bytes.** libvips builds that arm out of ordinary image
//!   operations, and every one of them writes a 32-bit float image, so
//!   promoting the chain to `f64` is a parity break rather than a
//!   cleanup. `vips_multiply` and `vips_add` round `Gx^2 + Gy^2` to
//!   `f32`; then `vips_pow_const1(0.5)` special-cases the exponent to a
//!   `double` `sqrt()` and stores the root as `f32` again
//!   (`arithmetic/math2.c:147-162`). The truncating cast that follows
//!   turns either rounding into a whole output value wherever the
//!   magnitude lands just under an integer. The pinned tie fixture is
//!   driven by the **second** one: `Gx = 1.91181, Gy = 148.98773` gives
//!   ~148.99999 under both the `f32` and the `f64` square sum, and it is
//!   rounding *that* to `f32` which reaches exactly 149.0, so vips writes
//!   149 where an all-`f64` chain writes 148. Over 5M random gradient
//!   pairs in `-260..=260` each rounding moves bytes the other does not:
//!   dropping the square-sum rounding alone changed 7-14 results,
//!   dropping the post-`sqrt` store alone 2-5, and dropping both 14-20.
//!   `vips_canny` computes `gx*gx + gy*gy` in the image's own float type
//!   (`POLAR(TYPE)`) for the same reason, so the rule carries forward.
//! * **Canny.** [`Raster::canny`] is `vips_canny`
//!   (`convolution/canny.c:381-428`) and it is **Canny up to and
//!   including non-maximum suppression, and no further**: blur, a 2x2
//!   `[-1 1; -1 1]` gradient pair, `(G, theta)`, thin, stop. libvips
//!   ships no double-thresholding and no edge tracking by connectivity,
//!   which is why the operation takes no hysteresis thresholds at all,
//!   only `sigma` and `precision`. The result is a suppressed gradient
//!   magnitude rather than a binary edge map. Three details decide
//!   whether a port matches the binary. `precision` reaches **only** the
//!   blur, and the gradient stage then picks its own arm from the format
//!   of the *blurred* image (`canny.c:81`), so a uchar input comes back
//!   uchar only when the blur left it uchar. `theta` comes from
//!   `atan2(gx, gy)` with the arguments **swapped**, measured from `+y`,
//!   so a white disc reads 0 at the top, 64 on the left, 128 at the
//!   bottom and 192 on the right; the `canny.c:228` comment naming the
//!   right twice is wrong. And suppression tests `G <= low || G < high`,
//!   asymmetric on purpose: where two adjacent pixels share both `G` and
//!   `theta`, the survivor is the one on the strict `<` side, and a
//!   symmetric comparison either erases the edge or widens it to two
//!   pixels. `G` skips the sqrt on both arms and is bounded at 64 on the
//!   uchar one only; the float arm reaches 508.5 on a hard step and
//!   reads 0.5, not 0, on a flat field.
//! * **Mask precision defaults.** `gaussmat` and `logmat` default to
//!   integer precision in libvips (`create/gaussmat.c`, `create/logmat.c`
//!   both init `precision = VIPS_PRECISION_INTEGER`); the ported
//!   convolution cell passes the precision explicitly, so
//!   [`Kernel::gaussmat`] takes it as an argument, while
//!   [`Kernel::logmat`] keeps the libvips default (the ported call sites
//!   pass three arguments) and [`Kernel::logmat_with_precision`] exposes
//!   the float form the libvips originals use.
//!
//! # Divergence from stock libvips
//!
//! Three gaps are open between this module and a stock libvips. The first
//! two are integer-precision arithmetic and neither is closable here. The
//! third is not arithmetic at all: it is an argument vips accepts and this
//! module deliberately refuses, and it applies at either precision.
//!
//! The first reaches every operation that runs an integer convolution:
//! [`Raster::conv`] and [`Raster::convsep`] at [`Precision::Integer`],
//! [`Raster::compass`], [`Raster::gaussblur`], [`Raster::sharpen`],
//! [`Raster::canny`], and the uchar arm of [`Raster::sobel`],
//! [`Raster::scharr`] and [`Raster::prewitt`]. The second reaches only
//! the three that convolve with a mask the caller handed in,
//! [`Raster::conv`], [`Raster::convsep`] and [`Raster::compass`], because
//! it is about a scale libvips cannot hold and every mask this module
//! builds for itself carries an integer one. The third reaches
//! [`Raster::compass`] alone, and reaches it before `precision` is ever
//! read.
//!
//! * **The two integer-convolution kernels, issue #558.** libviprs ports
//!   `vips_convi_gen`, the portable C loop, which divides with C's `/`
//!   and so rounds towards zero (`convolution/convi.c:710`). libvips's
//!   own documentation names that loop as the specification and flags
//!   its alternative as a deviation from it (`convi.c:1276-1284`, quoted
//!   in full on [`Precision::Integer`]), and it is what libvips falls
//!   back to whenever `vips_convi_intize` declines a mask. On uchar
//!   images a Highway-enabled libvips otherwise runs a fixed-point vector
//!   path, and **the two paths convolve with different coefficients**:
//!   `intize` rebuilds the mask over a power-of-two denominator, so a
//!   3x3 box blur of scale 9 is applied as `57/512 = 0.111328` rather
//!   than `1/9 = 0.111111`.
//!
//!   That is the mechanism, and it is worth being precise about, because
//!   two plausible-sounding descriptions of it are **false**:
//!
//!   * It is not the rounding mode. On a window summing to 1147 the C
//!     path gives `(1147 + 4) / 9 = 127`, flooring gives 127, and the
//!     vector path gives `(57 * 1147 + 256) >> 9 = 128`. Switching this
//!     module from truncation to floor would move zero bytes for
//!     `gaussblur`, for `conv` with a non-negative mask, and for
//!     `canny`'s first stage, precisely the cases with the largest
//!     measured deltas.
//!   * There is no "window sum negative and even reads one lower" rule,
//!     and no bound of 2. `vips_convi_intize`'s accuracy check
//!     (`convi.c:1096-1113`) is a DC-gain test against exact real
//!     arithmetic at one grey level on a flat field; `vips_convi_gen`
//!     appears nowhere in it. It constrains `sum(w_hat - w)` and says
//!     nothing about per-pixel error, `sum((w_hat - w) * p)`. A mask it
//!     accepts has been measured **128 of 255** apart.
//!
//!   This is a property of the **library**, not of the `vips` command:
//!   pyvips, sharp, ruby-vips and anything linking a distro libvips hit
//!   the identical gap. `VIPS_NOVECTOR=1` in the environment disables the
//!   vector path and makes libvips agree with libviprs exactly. It is
//!   read once at library init, though, so it works for a CLI comparison
//!   and not for a caller who already holds an `Image`.
//!
//!   The edge detectors inherit it **quadrupled, not doubled**. The uchar
//!   arm recovers each response as `2 * (p - 128)`, which doubles a
//!   one-unit gap, and `Gx` and `Gy` can both be off at once. Measured on
//!   an 8x3 `Gray8` image, `prewitt` at `(4, 0)` reads 106 from libviprs
//!   and from `VIPS_NOVECTOR=1 vips`, and 110 from the same binary with
//!   the vector path live, because the two inner convolutions read 123
//!   and 80 here against 122 and 79 there. `vips sobel` over the
//!   `oracle-captures/convolution` `sample_mono` fixture differs on
//!   44177 of 128180 samples, by at most 4. **Compare against an
//!   HWY-enabled libvips with a tolerance of 4 on the edge detectors, not
//!   2.** The float arm has no such gap and is bit-exact either way.
//!
//!   [`Raster::canny`] inherits it **unbounded**, because non-maximum
//!   suppression turns a one-unit blur difference into a keep-or-zero
//!   decision. Measured over twelve sigmas on a 64x64 noise field, the
//!   two libvips paths disagree at nine of them, by as much as 28 on a
//!   byte at sigma 0.8. Sigma 1.4, canny's default, is one of the three
//!   that agree: its separable gaussmat has scale 64, a power of two, so
//!   the requantisation is exact. A canny suite pinned only at the
//!   default therefore passes against either implementation and proves
//!   nothing, which is why the pins here run at 0.8 and 1.6 as well.
//!   `oracle-captures/convolution/canny/` has the sweep.
//!
//!   The full contract, including the regimes where the two paths cannot
//!   differ at all, is on [`Precision::Integer`]. The dual-path evidence
//!   is captured in `oracle-captures/convolution/`, which records every
//!   integer-conv record on both libvips paths with a `paths_agree` flag
//!   and asserts it.
//!
//! * **A mask scale that rounds to zero, issue #547.**
//!   `vips_convi_gen` holds the divisor in an `int`
//!   (`convolution/convi.c:757-760`), so any `|scale| < 0.5` leaves it at
//!   `0` and C divides by it. Measured on 8.18.4 at scale 0.4, the two
//!   integer arms answer `0` (aarch64 `sdiv` returns zero rather than
//!   trapping, which is not a defined result, and x86 would trap) and the
//!   float-input arm prints `inf`. libviprs nudges a zero divisor to `1`
//!   instead, the guard `vips__image_intize` writes for its own copy at
//!   `convi.c:895-897` and the only total answer on offer, so the sums
//!   are written back undivided. [`Precision::Integer`] carries the
//!   contract and what to reach for instead. What #547 reported, a
//!   division by the brightness-corrected scale `vips__image_intize`
//!   derives instead of by the one `vips_convi_gen` reads off the mask,
//!   is fixed and is not a divergence any more; the `intize` helper
//!   documents the divisor that replaced it.
//!
//! * **An out-of-range `compass` `times`.** vips declares the bound on
//!   the GObject property (`VIPS_ARG_INT(class, "times", 101, ..., 1,
//!   1000, 2)` in `convolution/compass.c`), and GObject does not refuse
//!   the call when you miss it. It writes
//!   `value "N" of type 'gint' is invalid or out of range for property
//!   'times'` to stderr, leaves the property at its default of `2`, and
//!   runs. Measured on 8.18.4 with a 3x3 ones mask over a 4x4 black
//!   image, `--times 0`, `--times 1001` and `--times 100000` all exit
//!   `0` and write output byte-identical to `--times 2`.
//!
//!   [`Raster::try_compass`] returns [`ConvolutionError::TimesOutOfRange`]
//!   instead, and [`Raster::compass`] panics. So a caller porting
//!   `vips compass --times 1001` gets an error here where vips hands
//!   back an image, and that is deliberate: silently convolving twice
//!   when you asked for a thousand rounds is a wrong answer wearing a
//!   warning, and the warning goes to stderr where a library caller
//!   never sees it. The accepted range is identical to vips's, so
//!   anything vips actually honours is honoured here too.

use crate::colour::ColourError;
use crate::conversion::{Angle45, ConversionError, Interpretation, cast_float_sample};
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError, alloc_op_output};
use thiserror::Error;

#[cfg(test)]
use std::cell::Cell;

/// Don't allow the gaussmat/logmat mask radius to go over this
/// (`MASK_SANITY` in `create/gaussmat.c`).
const MASK_SANITY: u32 = 5000;

/// The fixed sharpen curve parameters libvips defaults to and the ported
/// `sharpen(sigma, m1, m2)` surface does not expose: flat/jaggy threshold
/// `x1`, maximum brightening `y2`, maximum darkening `y3` (L* units).
const SHARPEN_X1: f64 = 2.0;
const SHARPEN_Y2: f64 = 10.0;
const SHARPEN_Y3: f64 = 20.0;

/// The range [`Raster::compass`] accepts for `times`, the bound libvips
/// declares on the property in `convolution/compass.c`:
/// `VIPS_ARG_INT(class, "times", 101, ..., 1, 1000, 2)`. GObject refuses
/// anything outside it before the operation is built, so it never reaches
/// a convolution there either. Measured on 8.18.4 with a 3x3 ones mask
/// over a 4x4 black image: `--times 1` and `--times 1000` run, while
/// `--times 0`, `--times 1001` and `--times 100000` each draw
/// `value "N" of type 'gint' is invalid or out of range for property
/// 'times' of type 'gint'` out of GObject and then run at the property's
/// default of 2, so the number asked for never reaches a convolution.
///
/// libviprs used to check the low end only, so `times` was effectively
/// unbounded: `u32::MAX` reserved a `Vec` of 4.29 billion rasters, some
/// 400 GB of address space, and then started that many convolutions.
const COMPASS_TIMES_MIN: u32 = 1;
/// Upper end of the [`COMPASS_TIMES_MIN`] range.
const COMPASS_TIMES_MAX: u32 = 1000;

/// Calculation accuracy for the convolution operations (libvips
/// `VipsPrecision`; the `APPROXIMATE` variant is not ported).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Precision {
    /// Integer arithmetic: the mask is converted to integers with
    /// `rint()`, each sum is divided by `rint()` of the mask's own scale,
    /// and the result stays in the input format.
    ///
    /// # Parity contract (issue #558)
    ///
    /// At [`Precision::Float`], libviprs reproduces libvips 8.18.4 byte for
    /// byte.
    ///
    /// At `Precision::Integer` **on uchar images**, libvips has two
    /// implementations that disagree, and it does not bound the
    /// disagreement. libviprs implements the portable C one,
    /// `vips_convi_gen`. That is the formula libvips documents, and it
    /// says so in its own words at `convolution/convi.c:1276-1284`:
    ///
    /// > `@mask` is converted to an integer mask with `rint()` of each
    /// > element ... For `UCHAR` images, `vips_convi` uses a fast vector
    /// > path based on half-float arithmetic. **This can produce slightly
    /// > different results.** Disable the vector path with
    /// > `--vips-novector` or `VIPS_NOVECTOR` or
    /// > `vips_vector_set_enabled()`.
    ///
    /// It is also the path libvips itself falls back to whenever
    /// `vips_convi_intize` declines a mask, which it does on ordinary
    /// input, so it is the floor rather than one of two options.
    ///
    /// **`VIPS_NOVECTOR=1 vips` reproduces libviprs byte for byte.** A
    /// SIMD-enabled `vips` runs a fixed-point approximation that
    /// convolves with **requantised coefficients**, not merely a
    /// different rounding: a 3x3 box blur, scale 9, over a window summing
    /// to 1147 gives `(1147 + 4) / 9 = 127` here and
    /// `(57 * 1147 + 256) >> 9 = 128` there, because that path is
    /// filtering with `57/512`, not `1/9`. Flooring instead of
    /// truncating also gives 127, so the rounding mode is not the
    /// mechanism.
    ///
    /// Measured divergence against a vectorised 8.18.4: up to **4** for
    /// [`Raster::gaussblur`] and the uchar edge detectors, and **128 of
    /// 255**, half of full scale, for a hostile mask that libvips's own
    /// accuracy gate still accepts (`[45 -17 -25 / -33 -15 -34 /
    /// 55 53 -26]`, scale 3, over a near-binary noise field; the same
    /// mask reaches 73 over smoother noise and 2 over a zone-plate, so
    /// even that is a fixture's number rather than a bound). Downstream
    /// of a non-linear consumer such as `canny --precision integer` it is
    /// unbounded outright, because non-maximum suppression turns a
    /// one-unit blur difference into a keep-or-zero decision. Which path
    /// a given `vips` runs depends on its build, its CPU, `VIPS_NOVECTOR`
    /// and the mask.
    ///
    /// There is no honest tolerance to quote. `vips_convi_intize`'s check
    /// (`convi.c:1096-1113`) is often read as bounding the two paths
    /// within 2; it does not. It compares the requantised mask against
    /// exact real arithmetic, at one grey level, on a flat field, and
    /// `vips_convi_gen` appears nowhere in it. It constrains
    /// `sum(w_hat - w)`, a DC-gain term, and says nothing about
    /// per-pixel error, which is `sum((w_hat - w) * p)`.
    ///
    /// Three regimes exist, not two, and nothing on this API surface
    /// tells you which one a mask is in: the vector path can run and
    /// disagree; it can run and agree (any scale whose requantisation is
    /// exact, including every power of two, and every scale-1 mask); or
    /// libvips can decline the mask and run the C path itself. Sigma 1.4,
    /// the [`Raster::gaussblur`] default, is lucky only for the
    /// *separable* gaussmat, whose scale is 64. The 2D gaussmat at the
    /// same sigma has scale 216 and is not.
    ///
    /// [`Raster::sharpen`], `morph`, `rank`, every ushort or float input,
    /// and every [`Precision::Float`] path are unaffected: `sharpen`
    /// convolves the `L` of `LabS`, which is 16-bit, and the vector path
    /// is gated on `BandFmt == VIPS_FORMAT_UCHAR` (`convi.c:1151`).
    ///
    /// # A mask scale that rounds to zero (issue #547)
    ///
    /// This is the second divergence, it is unrelated to the vector path,
    /// and it is the only one libviprs chose rather than inherited.
    ///
    /// `vips_convi_gen` takes its divisor off the mask the caller handed
    /// in and holds it in an `int` (`convolution/convi.c:757-760`):
    ///
    /// ```c
    /// VipsImage *M = convolution->M;
    /// int scale = rint(vips_image_get_scale(M));
    /// int rounding = scale / 2;
    /// ```
    ///
    /// libviprs divides by that same quantity, which is what #547 fixed.
    /// So any `|scale| < 0.5` leaves libvips holding `0` and dividing by
    /// it, and there is no libvips answer left to reproduce. Measured on
    /// 8.18.4 with `Kernel { data: [[1.0, 1.0]], scale: 0.4 }` over a flat
    /// field: the uchar and ushort arms print `0`, because aarch64 `sdiv`
    /// answers zero on a zero divisor rather than trapping, which is not a
    /// defined result and would trap on x86; the float-input arm of the
    /// same generator prints `inf`.
    ///
    /// **libviprs nudges a divisor of zero to `1`.** That is the guard
    /// `vips__image_intize` writes for its own copy at
    /// `convi.c:895-897`, and it is the only total answer on offer: the
    /// window sums are written back undivided, clipped to the input
    /// format as usual. Nothing else about the mask changes, so the
    /// coefficients are still `rint()`-ed and the offset still applies.
    ///
    /// It costs nothing on any mask this module builds at this precision.
    /// [`Kernel::gaussmat`] rounds every coefficient to `rint(20 * v)` and
    /// keeps a centre tap of 20, [`Kernel::logmat`] sums integers so its
    /// scale is either an integer of magnitude 1 or more or exactly 0
    /// (already [`ConvolutionError::ZeroScale`]), and the fixed masks
    /// behind [`Raster::sobel`], [`Raster::scharr`], [`Raster::prewitt`]
    /// and [`Raster::canny`] are integers over an integer scale. The nudge
    /// therefore needs a scale that did not come from a generator at this
    /// precision: one a caller wrote into [`Kernel`] by hand, or a
    /// [`Kernel::logmat_with_precision`] float mask handed to
    /// [`Raster::conv`], [`Raster::convsep`] or [`Raster::compass`] at
    /// [`Precision::Integer`].
    ///
    /// A caller who wants a sub-unit scale has two ways to keep libvips
    /// parity: scale the coefficients up instead and leave the divisor at
    /// or above 1, or use [`Precision::Float`], which has no `int`
    /// anywhere in the path and divides by the scale exactly as written.
    Integer,
    /// Floating-point arithmetic: the result is a 32-bit float image.
    ///
    /// `vips_convf` has one implementation, so this precision is
    /// identical on every libvips build: `VIPS_NOVECTOR=1` changes
    /// nothing, and the two-path divergence documented on
    /// [`Precision::Integer`] does not reach it.
    Float,
}

/// How [`Raster::compass`] combines the absolute convolution results
/// (libvips `VipsCombine`; the `MIN` variant is not ported).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Combine {
    /// Take the pixelwise maximum of the results.
    Max,
    /// Take the pixelwise sum of the results.
    Sum,
}

/// A convolution mask: the coefficient matrix and its scale divisor
/// (libvips matrix image plus its `scale` metadata).
///
/// The ported tests build this directly as a struct literal
/// (`Kernel { data, scale }`), so the fields are public and the struct is
/// deliberately not `#[non_exhaustive]`. `data` is row-major:
/// `data[row][column]`. libvips masks also carry an `offset` summand; this
/// surface fixes it at `0` (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct Kernel {
    /// Mask coefficients, one inner `Vec` per row.
    pub data: Vec<Vec<f64>>,
    /// The divisor applied to each convolution sum.
    pub scale: f64,
}

/// Typed errors for the convolution operations in [`crate::convolution`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConvolutionError {
    /// The kernel has no rows or an empty row.
    #[error("empty kernel")]
    EmptyKernel,
    /// The kernel rows have differing lengths.
    #[error("ragged kernel: row {row} has {got} elements, expected {expected}")]
    RaggedKernel {
        row: usize,
        got: usize,
        expected: usize,
    },
    /// The kernel scale is zero, which would divide every sum by zero.
    #[error("kernel scale must be non-zero")]
    ZeroScale,
    /// A mask scalar is `NaN` or infinite. Neither survives the integer
    /// path: a non-finite scale rounds to an integer scale of `0` and
    /// divides by it, and a non-finite offset saturates to `i64::MAX` and
    /// overflows the summand add.
    #[error("mask {param} must be finite, got {value}")]
    NonFiniteMaskParameter {
        /// Which mask scalar was rejected, `"scale"` or `"offset"`.
        param: &'static str,
        /// The offending value, as supplied.
        value: f64,
    },
    /// `convsep` needs a `1xN` or `Nx1` kernel (libvips
    /// `vips_check_separable`).
    #[error("separable convolution needs a 1xN or Nx1 kernel, got {width}x{height}")]
    NotSeparable { width: u32, height: u32 },
    /// `compass` rotates its kernel with `rot45`, which is defined on
    /// odd-sided square kernels only.
    #[error("compass needs an odd-sided square kernel, got {width}x{height}")]
    NotOddSquareKernel { width: u32, height: u32 },
    /// A [`Raster::compass`] `times` outside the range libvips declares
    /// on the property, `VIPS_ARG_INT(class, "times", 101, ..., 1, 1000,
    /// 2)` in `convolution/compass.c`. GObject refuses both ends before
    /// the operation is built, so neither reaches a convolution in vips
    /// and neither does here. `times` used to be checked against zero
    /// only, which left the top open: `u32::MAX` reserved a result vector
    /// of 4.29 billion rasters and then ran that many convolutions.
    #[error("compass times must be between {min} and {max}, got {times}")]
    TimesOutOfRange {
        /// The `times` that was asked for.
        times: u32,
        /// The smallest `times` libvips accepts, `1`.
        min: u32,
        /// The largest `times` libvips accepts, `1000`.
        max: u32,
    },
    /// The correlation template must have the same band count as the
    /// image.
    #[error("correlation band count mismatch: image has {image} bands, template {template}")]
    BandCountMismatch { image: usize, template: usize },
    /// A mask generator argument is outside the libvips argument range.
    #[error("{op}: {param} must be a finite value in {min}..={max}, got {value}")]
    InvalidMaskParameter {
        op: &'static str,
        param: &'static str,
        min: f64,
        max: f64,
        value: f64,
    },
    /// The generated mask would exceed the libvips `MASK_SANITY` radius.
    #[error("{op}: mask too large")]
    MaskTooLarge { op: &'static str },
    /// A colourspace conversion inside `sharpen` failed (unsupported
    /// source interpretation or too few bands).
    #[error(transparent)]
    Colour(#[from] ColourError),
    /// The `vips_cast` that closes the edge detectors' float arm failed
    /// (`edge.c:174`). It casts a float magnitude image to uchar without
    /// changing the band count, so in practice only the allocation inside
    /// [`Raster::try_cast`] can reach this.
    #[error(transparent)]
    Conversion(#[from] ConversionError),
    /// Constructing a result raster failed (allocation budget, size
    /// overflow).
    #[error(transparent)]
    Raster(#[from] RasterError),
}

#[track_caller]
fn expect_conv<T>(op: &str, r: Result<T, ConvolutionError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

/// C `rint()` under the default rounding mode: round half to even.
#[inline]
fn rint(v: f64) -> f64 {
    v.round_ties_even()
}

/// A validated dense view of a [`Kernel`]: dimensions, the row-major
/// coefficients, and the two scalars the convolution arithmetic needs.
///
/// libvips keeps `scale` and `offset` on the mask image itself
/// (`vips_image_get_scale` / `vips_image_get_offset`) rather than at the
/// call site, which is what lets `convolution/convsep.c:89-94` express
/// "second pass, same mask, offset zero" by copying the mask and stamping
/// one field. Carrying them here does the same job: a mask and its
/// scalars cannot be separated on the way into [`conv_raster`], and they
/// survive a rotation instead of being rebuilt by hand on the far side.
struct DenseKernel {
    w: usize,
    h: usize,
    coeff: Vec<f64>,
    /// The divisor applied to each convolution sum, from [`Kernel::scale`].
    scale: f64,
    /// The summand applied once per output sample. [`Kernel`] has no
    /// offset field, so everything on the ported surface leaves this at
    /// `0.0`; [`DenseKernel::with_offset`] is how an internal caller
    /// stamps the `vips_image_set_double(mask, "offset", ...)` its C
    /// original does.
    offset: f64,
}

impl DenseKernel {
    fn new(kernel: &Kernel) -> Result<Self, ConvolutionError> {
        if kernel.data.is_empty() || kernel.data[0].is_empty() {
            return Err(ConvolutionError::EmptyKernel);
        }
        let w = kernel.data[0].len();
        let mut coeff = Vec::with_capacity(w * kernel.data.len());
        for (row, r) in kernel.data.iter().enumerate() {
            if r.len() != w {
                return Err(ConvolutionError::RaggedKernel {
                    row,
                    got: r.len(),
                    expected: w,
                });
            }
            coeff.extend_from_slice(r);
        }
        Ok(DenseKernel {
            w,
            h: kernel.data.len(),
            coeff,
            scale: kernel.scale,
            offset: 0.0,
        })
    }

    /// Stamp the mask offset summand, as `convolution/edge.c` and
    /// `convolution/canny.c` do with
    /// `vips_image_set_double(mask, "offset", 128.0)`.
    fn with_offset(mut self, offset: f64) -> Self {
        self.offset = offset;
        self
    }
}

impl Kernel {
    /// Mask width in coefficients (libvips matrix `Xsize`).
    pub fn width(&self) -> u32 {
        self.data.first().map_or(0, |r| r.len()) as u32
    }

    /// Mask height in coefficients (libvips matrix `Ysize`).
    pub fn height(&self) -> u32 {
        self.data.len() as u32
    }

    /// Largest coefficient (libvips `max` on the mask image). Negative
    /// infinity for an empty kernel.
    pub fn max(&self) -> f64 {
        self.data
            .iter()
            .flatten()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Fallible form of [`Kernel::gaussmat`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::InvalidMaskParameter`] when `sigma` or
    /// `min_ampl` is outside the libvips argument range
    /// `0.000001..=10000`, or [`ConvolutionError::MaskTooLarge`] when the
    /// mask radius would exceed the libvips sanity bound.
    pub fn try_gaussmat(
        sigma: f64,
        min_ampl: f64,
        separable: bool,
        precision: Precision,
    ) -> Result<Kernel, ConvolutionError> {
        check_mask_param("gaussmat", "sigma", sigma)?;
        check_mask_param("gaussmat", "min_ampl", min_ampl)?;

        let sig2 = 2.0 * sigma * sigma;
        // `int max_x = VIPS_CLIP(0, 8 * sigma, MASK_SANITY)`: the C double
        // to int conversion truncates.
        let max_x = (8.0 * sigma).clamp(0.0, MASK_SANITY as f64) as u32;

        // Find the size of the mask: the first x whose amplitude drops
        // below min_ampl ends the loop (gaussmat.c allows x == 0, a 1x1
        // mask).
        let mut x = 0;
        while x < max_x {
            let v = (-((x * x) as f64) / sig2).exp();
            if v < min_ampl {
                break;
            }
            x += 1;
        }
        if x >= MASK_SANITY {
            return Err(ConvolutionError::MaskTooLarge { op: "gaussmat" });
        }
        let width = 2 * x.saturating_sub(1) as usize + 1;
        let height = if separable { 1 } else { width };

        let mut sum = 0.0;
        let mut data = Vec::with_capacity(height);
        for y in 0..height {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                let xo = x as f64 - (width / 2) as f64;
                let yo = y as f64 - (height / 2) as f64;
                let distance = xo * xo + yo * yo;
                let mut v = (-distance / sig2).exp();
                if precision != Precision::Float {
                    v = rint(20.0 * v);
                }
                sum += v;
                row.push(v);
            }
            data.push(row);
        }
        // Make sure we can't make sum == 0: it'd certainly cause /0 later
        // (gaussmat.c).
        if sum == 0.0 {
            sum = 1.0;
        }

        Ok(Kernel { data, scale: sum })
    }

    /// Create a Gaussian mask of standard deviation `sigma`, clipped where
    /// the amplitude falls below `min_ampl` (libvips `vips_gaussmat`).
    ///
    /// The mask has odd size. With [`Precision::Float`] the maximum value
    /// is normalised to `1.0`; otherwise each element is `rint(20 * v)`,
    /// the integer mask `vips_conv` uses. With `separable` only the centre
    /// row is generated (`1xN`), for use with [`Raster::convsep`]. `scale`
    /// is the sum of all elements.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Kernel::try_gaussmat`].
    #[track_caller]
    pub fn gaussmat(sigma: f64, min_ampl: f64, separable: bool, precision: Precision) -> Kernel {
        expect_conv(
            "gaussmat",
            Self::try_gaussmat(sigma, min_ampl, separable, precision),
        )
    }

    /// Fallible form of [`Kernel::logmat_with_precision`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::InvalidMaskParameter`] when `sigma` or
    /// `min_ampl` is outside the libvips argument range
    /// `0.000001..=10000`, or [`ConvolutionError::MaskTooLarge`] when the
    /// mask never flattens out within the libvips sanity bound.
    pub fn try_logmat(
        sigma: f64,
        min_ampl: f64,
        separable: bool,
        precision: Precision,
    ) -> Result<Kernel, ConvolutionError> {
        check_mask_param("logmat", "sigma", sigma)?;
        check_mask_param("logmat", "min_ampl", min_ampl)?;

        let sig2 = sigma * sigma;

        // Find the size of the mask: eval out beyond the minimum to where
        // the curve comes back up towards zero, i.e. stop when the change
        // from the previous point is non-negative and the absolute value
        // is below min_ampl (logmat.c).
        let mut last = 0.0;
        let mut x = 0;
        while x < MASK_SANITY {
            let distance = (x * x) as f64;
            let val = 0.5 * (2.0 - distance / sig2) * (-distance / (2.0 * sig2)).exp();
            if val - last >= 0.0 && val.abs() < min_ampl {
                break;
            }
            last = val;
            x += 1;
        }
        if x == MASK_SANITY {
            return Err(ConvolutionError::MaskTooLarge { op: "logmat" });
        }

        let width = 2 * x as usize + 1;
        let height = if separable { 1 } else { width };

        let mut sum = 0.0;
        let mut data = Vec::with_capacity(height);
        for y in 0..height {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                let xo = x as f64 - (width / 2) as f64;
                let yo = y as f64 - (height / 2) as f64;
                let distance = xo * xo + yo * yo;
                let mut v = 0.5 * (2.0 - distance / sig2) * (-distance / (2.0 * sig2)).exp();
                if precision == Precision::Integer {
                    v = rint(20.0 * v);
                }
                sum += v;
                row.push(v);
            }
            data.push(row);
        }
        // Unlike gaussmat, logmat.c stores the raw sum with no zero guard:
        // an integer Laplacian-of-Gaussian mask can legitimately sum (and
        // therefore scale) to exactly zero. Convolving with such a mask is
        // a typed ZeroScale error rather than the C path's division by an
        // unadjusted zero.
        Ok(Kernel { data, scale: sum })
    }

    /// Create a Laplacian-of-Gaussian mask of standard deviation `sigma`
    /// at integer precision, the libvips default (libvips `vips_logmat`;
    /// `create/logmat.c` inits `precision = VIPS_PRECISION_INTEGER`). The
    /// ported call sites pass exactly these three arguments; use
    /// [`Kernel::logmat_with_precision`] for the float form.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Kernel::try_logmat`].
    #[track_caller]
    pub fn logmat(sigma: f64, min_ampl: f64, separable: bool) -> Kernel {
        expect_conv(
            "logmat",
            Self::try_logmat(sigma, min_ampl, separable, Precision::Integer),
        )
    }

    /// [`Kernel::logmat`] with an explicit precision, matching the
    /// `precision=` optional argument of `vips_logmat`. With
    /// [`Precision::Float`] the maximum value is normalised to `1.0`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Kernel::try_logmat`].
    #[track_caller]
    pub fn logmat_with_precision(
        sigma: f64,
        min_ampl: f64,
        separable: bool,
        precision: Precision,
    ) -> Kernel {
        expect_conv(
            "logmat",
            Self::try_logmat(sigma, min_ampl, separable, precision),
        )
    }
}

/// Enforce the libvips `0.000001..=10000` argument range shared by the
/// gaussmat/logmat sigma and min_ampl parameters.
fn check_mask_param(
    op: &'static str,
    param: &'static str,
    value: f64,
) -> Result<(), ConvolutionError> {
    const MIN: f64 = 0.000001;
    const MAX: f64 = 10000.0;
    if !value.is_finite() || !(MIN..=MAX).contains(&value) {
        return Err(ConvolutionError::InvalidMaskParameter {
            op,
            param,
            min: MIN,
            max: MAX,
            value,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared pixel plumbing
// ---------------------------------------------------------------------------

/// An empty `Vec<T>` with room for `len` items, reserved fallibly.
///
/// `width` and `height` only name the raster in the error; `bytes` is the
/// real size of the request, which for these intermediates is several
/// times the raster's own byte length.
///
/// This is [`alloc_op_output`]'s contract for a buffer that is not an
/// op's output: reserve with [`Vec::try_reserve_exact`] and report
/// [`RasterError::AllocationFailed`], never reach `handle_alloc_error`
/// and abort. Callers fill the returned `Vec` with `extend` or `resize`,
/// neither of which reallocates while the reserved capacity holds.
fn try_buffer<T>(width: u32, height: u32, len: usize) -> Result<Vec<T>, RasterError> {
    let bytes = len.saturating_mul(size_of::<T>());
    // Test-only: count this reservation and honour a lowered per-thread
    // ceiling, so a test can both see that an intermediate goes through here at
    // all and drive the fallible branch at a raster it can build. This and the
    // thread-local it reads compile only under `cfg(test)`, so a production
    // reservation is bounded solely by the allocator.
    #[cfg(test)]
    let cap = CONV_BUFFER_PROBE.with(|c| {
        let (cap, calls) = c.get();
        c.set((cap, calls + 1));
        cap
    });
    #[cfg(test)]
    if bytes as u64 > cap {
        return Err(RasterError::AllocationFailed {
            width,
            height,
            bytes,
        });
    }
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| RasterError::AllocationFailed {
            width,
            height,
            bytes,
        })?;
    Ok(v)
}

#[cfg(test)]
thread_local! {
    /// Per-thread [`try_buffer`] probe: a ceiling in bytes on a single
    /// reservation, and how many reservations have been made under the current
    /// [`with_conv_buffer_probe`] call.
    ///
    /// The ceiling defaults to `u64::MAX`, so an ordinary run bounds an
    /// intermediate only by what the allocator will serve. The count is what
    /// makes a `vec![0i32; n]` put back in place of a [`try_buffer`] call
    /// visible to a test: the two spellings differ only in what they do when
    /// the allocation fails, which for a plane a test can actually build is
    /// never, so nothing else distinguishes them (issue #627).
    static CONV_BUFFER_PROBE: Cell<(u64, usize)> = const { Cell::new((u64::MAX, 0)) };
}

/// Test-only hook: run `f` with the calling thread's [`try_buffer`] ceiling
/// lowered to `max_bytes`, returning its value alongside the number of
/// reservations it made, and restoring the previous probe afterwards including
/// on unwind.
///
/// The probe is thread-local, so tests running in parallel do not perturb one
/// another, and it compiles only under `cfg(test)`.
#[cfg(test)]
fn with_conv_buffer_probe<R>(max_bytes: u64, f: impl FnOnce() -> R) -> (R, usize) {
    struct Restore((u64, usize));
    impl Drop for Restore {
        fn drop(&mut self) {
            CONV_BUFFER_PROBE.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(CONV_BUFFER_PROBE.with(|c| c.replace((max_bytes, 0))));
    let value = f();
    let calls = CONV_BUFFER_PROBE.with(|c| c.get().1);
    (value, calls)
}

/// Read every sample of `r` as `f64`, row-major with bands interleaved.
/// Unsigned samples convert exactly; float samples widen exactly.
///
/// Eight bytes per sample where the source carries one or two makes this
/// the largest single allocation on the convolution path: measured
/// post-#598 on a 4000x4000 `Rgb8` integer conv, 384 MB of a 486 MB peak
/// for 48 MB of input. It is also the allocation every fallible entry
/// point in this module sits on, so it is reserved through
/// [`try_buffer`] and reports [`RasterError::AllocationFailed`]. It used
/// to be a plain `.collect()`, which on failure reaches
/// `handle_alloc_error` and **aborts the process**. A `try_` API that
/// aborts is worse than an infallible one, because a caller reasonably
/// reads the `Result` as covering allocation (issue #575).
fn samples_f64(r: &Raster) -> Result<Vec<f64>, RasterError> {
    let fmt = r.format();
    let n = r.width() as usize * r.height() as usize * fmt.channels();
    let data = r.data();
    let mut out = try_buffer(r.width(), r.height(), n)?;
    match fmt.bytes_per_channel() {
        1 => out.extend(data.iter().map(|&b| b as f64)),
        2 => out.extend((0..n).map(|i| u16::from_ne_bytes([data[i * 2], data[i * 2 + 1]]) as f64)),
        _ => out.extend((0..n).map(|i| {
            f32::from_ne_bytes([
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ]) as f64
        })),
    }
    Ok(out)
}

/// The 32-bit float format with `channels` bands (the `vips_convf` output
/// depth).
fn float_format(channels: usize) -> PixelFormat {
    PixelFormat::with_channels(channels, 4).expect("validated channel count has a float format")
}

/// Build a float raster from `f64` samples (stored as `f32`, the crate's
/// float depth).
///
/// The buffer comes from [`alloc_op_output`] and the raster from
/// [`Raster::from_op_output`], the same pair `arithmetic.rs` uses, so the
/// allocation is fallible rather than aborting and the byte budget is not
/// re-applied to an output that is legitimately wider than its input. Going
/// through the budgeted [`Raster::new`] instead is what made the panicking
/// `sobel()` panic on a legal 16-bit input above ~4 GiB, because the float
/// intermediate is four times the source (issue #575).
fn raster_from_f64(
    w: u32,
    h: u32,
    channels: usize,
    samples: &[f64],
) -> Result<Raster, RasterError> {
    debug_assert_eq!(samples.len(), w as usize * h as usize * channels);
    let fmt = float_format(channels);
    let mut data = alloc_op_output(w, h, fmt)?;
    for (out, &v) in data.as_chunks_mut::<4>().0.iter_mut().zip(samples) {
        *out = (v as f32).to_ne_bytes();
    }
    Raster::from_op_output(w, h, fmt, data)
}

/// Build an unsigned raster in `fmt` from already-clipped integer samples.
///
/// Same [`alloc_op_output`] / [`Raster::from_op_output`] pair as
/// [`raster_from_f64`], for the same reason (issue #575).
fn raster_from_i64(
    w: u32,
    h: u32,
    fmt: PixelFormat,
    samples: &[i64],
) -> Result<Raster, RasterError> {
    debug_assert_eq!(samples.len(), w as usize * h as usize * fmt.channels());
    let mut data = alloc_op_output(w, h, fmt)?;
    if fmt.bytes_per_channel() == 1 {
        for (out, &v) in data.iter_mut().zip(samples) {
            *out = v as u8;
        }
    } else {
        for (out, &v) in data.as_chunks_mut::<2>().0.iter_mut().zip(samples) {
            *out = (v as u16).to_ne_bytes();
        }
    }
    Raster::from_op_output(w, h, fmt, data)
}

/// Clamp a mask-relative coordinate to the image, replicating edge pixels
/// (`VIPS_EXTEND_COPY`).
#[inline]
fn clamp_coord(v: i64, size: u32) -> usize {
    v.clamp(0, size as i64 - 1) as usize
}

// ---------------------------------------------------------------------------
// conv
// ---------------------------------------------------------------------------

/// The three things `vips_convi_gen` reads off a mask: the rounded
/// coefficients, the divisor, and the rounded offset.
///
/// Named fields rather than a tuple because the scale and the offset are
/// both `i64` and mean entirely different things; transposing them in a
/// destructuring pattern would compile and quietly produce wrong pixels.
struct IntKernel {
    coeff: Vec<i64>,
    scale: i64,
    offset: i64,
}

/// Build the integer mask the `vips_convi` generator convolves with.
///
/// The coefficients are `vips__image_intize`'s half of the job: `rint()`
/// every element (`convolution/convi.c:890-893`).
///
/// The scale and the offset are not. `vips_convi_gen` reads both off
/// `convolution->M`, which is the mask the caller handed in
/// (`convi.c:757-760`):
///
/// ```c
/// VipsImage *M = convolution->M;
/// int scale = rint(vips_image_get_scale(M));
/// int rounding = scale / 2;
/// int offset = rint(vips_image_get_offset(M));
/// ```
///
/// `vips_convi_build` does shadow `M` with the intized copy, but only for
/// as long as it takes to harvest the coefficients (`convi.c:1179-1181`),
/// and it never writes that copy back onto the object. So the brightness
/// nudge `vips__image_intize` computes into `out_scale`
/// (`convi.c:911-913`) is dead code as far as `convi` is concerned. It is
/// live only for the approximate paths, `conva` and `convasep`
/// (`conva.c:1269`, `convasep.c:862`), and libviprs implements neither.
///
/// Dividing by the nudge instead was issue #547, and it was not a
/// rounding nit: on `[[3.0, 0.4, 0.4, 0.4, 0.4]]` at scale `1.0` the
/// nudge is `-1`, so a flat grey field came out black where vips answers
/// white.
///
/// The offset needs no such care. `rint()` is idempotent, so rounding it
/// here rather than off the original mask cannot move it.
///
/// A scale that rounds to zero is nudged to `1` here, the guard
/// `vips__image_intize` writes for its own copy at `convi.c:895-897`. It
/// is the one place libviprs cannot follow libvips, because there is
/// nothing there to follow: `int scale` holds `0` and C divides by it.
/// The measurements and the contract are on [`Precision::Integer`], where
/// a caller can find them.
///
/// The offset clamp keeps the rounded offset inside the `int` that
/// `vips_convi_gen` reads it into, which also keeps the `i64` add on the
/// unsigned arm from overflowing on a large finite offset. [`Scan::new`]
/// has already rejected a non-finite one.
fn intize(dense: &DenseKernel) -> IntKernel {
    let mut scale = rint(dense.scale);
    if scale == 0.0 {
        scale = 1.0;
    }

    IntKernel {
        coeff: dense.coeff.iter().map(|&v| rint(v) as i64).collect(),
        scale: scale as i64,
        offset: rint(dense.offset).clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i64,
    }
}

/// The clip ceiling for an unsigned format depth.
#[inline]
fn depth_max(fmt: PixelFormat) -> i64 {
    if fmt.bytes_per_channel() == 1 {
        255
    } else {
        65535
    }
}

// ---------------------------------------------------------------------------
// The shared traversal
// ---------------------------------------------------------------------------

/// One surviving mask tap: where it reads, and what it multiplies by.
///
/// `ty` and `tx` are already offset by the traversal's common mask
/// origin, so the source sample for output `(y, x)` is
/// `ytab[y + ty] + xtab[x + tx] + band` and there is no clamping left to
/// do per tap.
struct Tap<C> {
    ty: usize,
    tx: usize,
    c: C,
}

/// Compact a row-major `kw` by `kh` mask to its non-zero taps, positioned
/// against a traversal origin of `(ay, ax)` half-extents.
///
/// libvips squeezes zeros out of both convolution cores:
/// `vips_convf_build` does it to the scaled double mask
/// (`convolution/convf.c:314-321`) and `vips_convi_build` to the intized
/// integer one (`convolution/convi.c:1189-1197`), and both inner loops
/// then run over `nnz` instead of the whole mask.
///
/// This is not only cheaper on a sparse mask, it is the answer libvips
/// gives. `0.0 * inf` is `NaN`, so a structural zero sitting over a
/// non-finite sample used to poison the entire response and drive the
/// result to 0 where vips reads 255 (issue #574). All three edge
/// detectors have structural zeros, so it was reachable straight off the
/// ported surface. Dropping `+ 0.0 * x` cannot move a finite answer: it
/// can only change the sign of a zero, and a signed zero does not survive
/// `a * a`.
///
/// A mask that is entirely zero keeps **one** tap, at mask index 0 and
/// with coefficient zero, because that is what libvips keeps: both cores
/// force `nnz` back up to 1 rather than leave the inner loop with nothing
/// to do (`convf.c:325-333`, `convi.c:1199-1206`). It stays observable,
/// since the surviving tap still multiplies a sample and so still answers
/// `NaN` over a non-finite one.
///
/// The zero test is `c != C::default()`, which is C's `if (coeff[i])`:
/// `-0.0` counts as zero on both sides.
fn compact_taps<C: Copy + Default + PartialEq>(
    kw: usize,
    kh: usize,
    coeff: impl Iterator<Item = C>,
    origin: (usize, usize),
) -> Vec<Tap<C>> {
    let (oy, ox) = origin;
    let (ay, ax) = (kh / 2, kw / 2);
    let zero = C::default();
    let mut taps: Vec<Tap<C>> = coeff
        .enumerate()
        .filter(|&(_, c)| c != zero)
        .map(|(k, c)| Tap {
            ty: oy + k / kw - ay,
            tx: ox + k % kw - ax,
            c,
        })
        .collect();
    if taps.is_empty() {
        taps.push(Tap {
            ty: oy - ay,
            tx: ox - ax,
            c: zero,
        });
    }
    taps
}

/// Everything one traversal shares across the masks it carries: the
/// source widened to `f64` **once**, the output geometry, and the two
/// clamped index tables that replace per-tap edge arithmetic.
///
/// The tables are `vips_embed(..., VIPS_EXTEND_COPY)` expressed as
/// indices rather than as pixels: `convf.c:335-341` embeds the input into
/// a border-replicated copy and then reads at fixed offsets, and
/// `ytab`/`xtab` get the same effect without the copy. `ytab[y + ty]` is a
/// tap's row base and `xtab[x + tx]` its column offset, so the inner loop
/// carries no clamp and the border costs exactly what the interior does.
///
/// Tap positions are relative to `origin`, the largest `kh / 2` and
/// `kw / 2` over the masks in the traversal, so masks of different sizes
/// (the 90-degree rotation of a non-square mask, for one) share one pair
/// of tables.
struct Scan {
    samples: Vec<f64>,
    w: usize,
    h: usize,
    channels: usize,
    origin: (usize, usize),
    ytab: Vec<usize>,
    xtab: Vec<usize>,
}

impl Scan {
    /// Validate every mask's scale and offset, then decode `src` and build
    /// the tables.
    ///
    /// A `NaN` scale slips straight past the zero test and reaches
    /// [`intize`], where `rint(NaN) as i64` is `0` and the integer arm then
    /// divides by it; a non-finite offset saturates `rint(offset) as i64`
    /// to `i64::MAX` and overflows the add, which panics in debug and
    /// silently wraps to black in release. Both are rejected here, at the
    /// one boundary every caller passes through.
    fn new<const M: usize>(
        src: &Raster,
        masks: &[&DenseKernel; M],
    ) -> Result<Scan, ConvolutionError> {
        for mask in masks {
            if mask.scale == 0.0 {
                return Err(ConvolutionError::ZeroScale);
            }
            for (param, value) in [("scale", mask.scale), ("offset", mask.offset)] {
                if !value.is_finite() {
                    return Err(ConvolutionError::NonFiniteMaskParameter { param, value });
                }
            }
        }
        let (w, h) = (src.width() as usize, src.height() as usize);
        let channels = src.format().channels();
        let row_stride = w * channels;
        let oy = masks.iter().map(|m| m.h / 2).max().unwrap_or(0);
        let ox = masks.iter().map(|m| m.w / 2).max().unwrap_or(0);
        let ty = masks.iter().map(|m| m.h - 1 - m.h / 2).max().unwrap_or(0);
        let tx = masks.iter().map(|m| m.w - 1 - m.w / 2).max().unwrap_or(0);
        let ytab = (0..h + oy + ty)
            .map(|t| clamp_coord(t as i64 - oy as i64, src.height()) * row_stride)
            .collect();
        let xtab = (0..w + ox + tx)
            .map(|t| clamp_coord(t as i64 - ox as i64, src.width()) * channels)
            .collect();
        Ok(Scan {
            samples: samples_f64(src)?,
            w,
            h,
            channels,
            origin: (oy, ox),
            ytab,
            xtab,
        })
    }

    /// The half-open span of output columns for which tap column `tx`
    /// reads inside the image without the edge clamp doing anything, and
    /// the source column that span starts at.
    ///
    /// `xtab` is the identity-plus-shift over that span, so the taps there
    /// walk a contiguous run of samples and the whole row can be added
    /// with one slice loop. Outside it the clamp is replicating a border
    /// pixel and the general table lookup earns its keep. Splitting the
    /// two is what lets the interior vectorise; it cannot change a value,
    /// because on that span the table and the arithmetic agree by
    /// construction.
    #[inline]
    fn interior(&self, tx: usize) -> (usize, usize, usize) {
        let ox = self.origin.1;
        let lo = ox.saturating_sub(tx).min(self.w);
        let hi = (self.w + ox).saturating_sub(tx).min(self.w).max(lo);
        (lo, hi, (lo + tx).saturating_sub(ox) * self.channels)
    }

    /// Add one tap's contribution across a whole output row.
    ///
    /// Taps are applied to an output row in mask order, exactly the order
    /// the per-sample accumulator used to add them in, so the float sums
    /// keep their rounding.
    #[inline]
    fn add_tap<C: Copy, A: Copy>(
        &self,
        acc: &mut [A],
        row: usize,
        tx: usize,
        c: C,
        mut fma: impl FnMut(&mut A, C, f64),
    ) {
        let (lo, hi, start) = self.interior(tx);
        let (samples, xtab) = (&self.samples[..], &self.xtab[..]);
        for x in 0..lo {
            let col = xtab[x + tx];
            for band in 0..self.channels {
                fma(
                    &mut acc[x * self.channels + band],
                    c,
                    samples[row + col + band],
                );
            }
        }
        if hi > lo {
            // Guarded rather than left to an empty slice: a mask wider
            // than the image can push `start` past the end of `samples`
            // even when the span itself is empty, and `&v[n..n]` still
            // demands `n <= v.len()`.
            let (a, b) = (lo * self.channels, hi * self.channels);
            let src = &samples[row + start..row + start + (b - a)];
            for (slot, &s) in acc[a..b].iter_mut().zip(src) {
                fma(slot, c, s);
            }
        }
        for x in hi..self.w {
            let col = xtab[x + tx];
            for band in 0..self.channels {
                fma(
                    &mut acc[x * self.channels + band],
                    c,
                    samples[row + col + band],
                );
            }
        }
    }

    /// Walk every output sample once, accumulating all `M` responses in
    /// `f64` off the same window, and hand each sample's responses to
    /// `emit` with its flat index.
    ///
    /// `init` seeds each accumulator, which is where the float arm's
    /// offset summand goes (`convf.c:172`).
    #[inline]
    fn float<const M: usize>(
        &self,
        taps: &[Vec<Tap<f64>>; M],
        init: [f64; M],
        mut emit: impl FnMut(usize, [f64; M]),
    ) {
        let stride = self.w * self.channels;
        let mut acc: [Vec<f64>; M] = std::array::from_fn(|_| vec![0.0f64; stride]);
        let mut idx = 0;
        for y in 0..self.h {
            for ((row, mask), &seed) in acc.iter_mut().zip(taps).zip(&init) {
                row.fill(seed);
                for t in mask {
                    let base = self.ytab[y + t.ty];
                    self.add_tap(row, base, t.tx, t.c, |slot, c, s| *slot += c * s);
                }
            }
            // A transpose of `M` parallel accumulator rows into one array
            // per sample, which `needless_range_loop` has no spelling for:
            // the index walks every plane at once, not one of them.
            #[allow(clippy::needless_range_loop)]
            for k in 0..stride {
                emit(idx, std::array::from_fn(|m| acc[m][k]));
                idx += 1;
            }
        }
    }

    /// [`Scan::float`] with `i64` accumulators: the `CONV_INT` inner loop
    /// (`convi.c:700-720`), which every unsigned input takes at
    /// [`Precision::Integer`].
    #[inline]
    fn int<const M: usize>(
        &self,
        taps: &[Vec<Tap<i64>>; M],
        mut emit: impl FnMut(usize, [i64; M]),
    ) {
        let stride = self.w * self.channels;
        let mut acc: [Vec<i64>; M] = std::array::from_fn(|_| vec![0i64; stride]);
        let mut idx = 0;
        for y in 0..self.h {
            for (row, mask) in acc.iter_mut().zip(taps) {
                row.fill(0);
                for t in mask {
                    let base = self.ytab[y + t.ty];
                    self.add_tap(row, base, t.tx, t.c, |slot, c, s| *slot += c * s as i64);
                }
            }
            // A transpose of `M` parallel accumulator rows into one array
            // per sample, which `needless_range_loop` has no spelling for:
            // the index walks every plane at once, not one of them.
            #[allow(clippy::needless_range_loop)]
            for k in 0..stride {
                emit(idx, std::array::from_fn(|m| acc[m][k]));
                idx += 1;
            }
        }
    }
}

/// `M` output buffers of one raster each, from the fallible
/// [`alloc_op_output`].
fn out_buffers<const M: usize>(
    w: u32,
    h: u32,
    fmt: PixelFormat,
) -> Result<[Vec<u8>; M], RasterError> {
    let mut out: [Vec<u8>; M] = std::array::from_fn(|_| Vec::new());
    for buf in &mut out {
        *buf = alloc_op_output(w, h, fmt)?;
    }
    Ok(out)
}

/// Wrap `M` finished output buffers as rasters.
fn rasters_from<const M: usize>(
    w: u32,
    h: u32,
    fmt: PixelFormat,
    buffers: [Vec<u8>; M],
) -> Result<[Raster; M], RasterError> {
    let mut built: [Option<Raster>; M] = std::array::from_fn(|_| None);
    for (slot, data) in built.iter_mut().zip(buffers) {
        *slot = Some(Raster::from_op_output(w, h, fmt, data)?);
    }
    Ok(built.map(|r| r.expect("every slot was just filled")))
}

/// `M` full 2D convolutions of one source in **one traversal** (both
/// precisions, all formats), the shared engine behind `conv`, `convsep`,
/// `compass` and the edge detectors.
///
/// Each `DenseKernel` carries the mask, its `scale` divisor and its
/// `offset` summand, the same three things a libvips matrix image
/// carries. The public [`Kernel`] has no offset field, so everything on
/// the ported surface convolves with the zero summand.
///
/// Every output sample's `M` responses come off the same window, so the
/// masks share one source decode, one set of clamped index tables and one
/// pass over the image instead of `M` of each (issue #562). `M = 1`
/// monomorphises back to a single convolution: [`conv_raster`] is that
/// wrapper, and every pin in this module holds it to the byte.
///
/// Each arm adds the summand exactly where its C counterpart does: before
/// the clip on the integer/unsigned path (`convi.c:710`), after the
/// division and with no clip on the integer/float-input path
/// (`convi.c:733`), and as the starting value of the accumulator at float
/// precision, where the scale is already baked into the coefficients
/// (`convf.c:172`). Both integer arms use the `rint()`-ed offset
/// [`intize`] hands back, matching the `int offset = rint(...)`
/// `vips_convi_gen` reads off the mask; the float arm keeps it as an
/// unrounded `f64`, as `vips_convf_gen` does. It is applied once per
/// output *sample*, not once per tap, so it is one add per sample however
/// large the mask is, and there is nothing here worth specialising away.
///
/// The edge detectors are what need this: `convolution/edge.c` stamps
/// `offset = 128.0, scale = 2.0` on its mask for the uchar path, and
/// `convolution/canny.c` stamps `offset = 128.0` on its gradient mask, so
/// a signed response lands centred in the unsigned output range instead of
/// clipping away at zero. Two libvips rules come with the summand, and a
/// consumer that ignores either gets wrong pixels:
///
/// * `vips_convsep` applies it on the **first pass only**. The second
///   pass runs against a copy of the mask with the offset stamped back to
///   zero (`convolution/convsep.c:89-94`), because the summand is in
///   output units and a two-pass mask would otherwise add it twice.
/// * `vips_compass` takes the absolute value of every rotation before
///   combining them (`convolution/compass.c`), and an offset makes that
///   meaningless: 128 moves the zero point of the response, so `.abs()`
///   folds it about the wrong value. A compass mask carries offset zero.
fn conv_raster_n<const M: usize>(
    src: &Raster,
    masks: [&DenseKernel; M],
    precision: Precision,
) -> Result<[Raster; M], ConvolutionError> {
    let scan = Scan::new(src, &masks)?;
    let (w, h) = (src.width(), src.height());
    let channels = src.format().channels();

    match precision {
        Precision::Float => {
            // vips_convf: bake the scale into the coefficients, squeeze
            // out the zeros, seed the accumulator with the offset,
            // accumulate in f64, store 32-bit float.
            let taps = masks
                .map(|k| compact_taps(k.w, k.h, k.coeff.iter().map(|&v| v / k.scale), scan.origin));
            let fmt = float_format(channels);
            let mut out = out_buffers::<M>(w, h, fmt)?;
            scan.float(&taps, masks.map(|k| k.offset), |i, sums| {
                for (buf, &v) in out.iter_mut().zip(&sums) {
                    buf[i * 4..i * 4 + 4].copy_from_slice(&(v as f32).to_ne_bytes());
                }
            });
            Ok(rasters_from(w, h, fmt, out)?)
        }
        Precision::Integer => {
            let ints: [IntKernel; M] = std::array::from_fn(|m| intize(masks[m]));
            if src.format().is_float() {
                // vips_convi_gen keeps a double path for float inputs: the
                // integer mask, real division, the rounded offset added
                // after it, no rounding of the result and no clip.
                let taps: [_; M] = std::array::from_fn(|m| {
                    compact_taps(
                        masks[m].w,
                        masks[m].h,
                        ints[m].coeff.iter().map(|&v| v as f64),
                        scan.origin,
                    )
                });
                let params: [(f64, f64); M] =
                    std::array::from_fn(|m| (ints[m].scale as f64, ints[m].offset as f64));
                let fmt = float_format(channels);
                let mut out = out_buffers::<M>(w, h, fmt)?;
                scan.float(&taps, [0.0; M], |i, sums| {
                    for ((buf, &sum), &(iscale, ioffset)) in out.iter_mut().zip(&sums).zip(&params)
                    {
                        let v = (sum / iscale + ioffset) as f32;
                        buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
                    }
                });
                Ok(rasters_from(w, h, fmt, out)?)
            } else {
                // CONV_INT: i64 accumulation, (sum + scale/2) / scale with
                // C truncating division, then the offset, then the clip
                // into the input format. The offset lands before the clip,
                // so it recentres the response rather than shifting an
                // already-saturated one.
                let taps: [_; M] = std::array::from_fn(|m| {
                    compact_taps(
                        masks[m].w,
                        masks[m].h,
                        ints[m].coeff.iter().copied(),
                        scan.origin,
                    )
                });
                let params: [(i64, i64, i64); M] =
                    std::array::from_fn(|m| (ints[m].scale / 2, ints[m].scale, ints[m].offset));
                let fmt = src.format();
                let max = depth_max(fmt);
                let mut out = out_buffers::<M>(w, h, fmt)?;
                if fmt.bytes_per_channel() == 1 {
                    scan.int(&taps, |i, sums| {
                        for ((buf, &sum), &(rounding, iscale, ioffset)) in
                            out.iter_mut().zip(&sums).zip(&params)
                        {
                            buf[i] = ((sum + rounding) / iscale + ioffset).clamp(0, max) as u8;
                        }
                    });
                } else {
                    scan.int(&taps, |i, sums| {
                        for ((buf, &sum), &(rounding, iscale, ioffset)) in
                            out.iter_mut().zip(&sums).zip(&params)
                        {
                            let v = ((sum + rounding) / iscale + ioffset).clamp(0, max) as u16;
                            buf[i * 2..i * 2 + 2].copy_from_slice(&v.to_ne_bytes());
                        }
                    });
                }
                Ok(rasters_from(w, h, fmt, out)?)
            }
        }
    }
}

/// One full 2D convolution pass: [`conv_raster_n`] with a single mask.
fn conv_raster(
    src: &Raster,
    dense: &DenseKernel,
    precision: Precision,
) -> Result<Raster, ConvolutionError> {
    let [out] = conv_raster_n(src, [dense], precision)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Kernel rotation helpers
// ---------------------------------------------------------------------------

/// Rotate a kernel matrix 90 degrees clockwise (`vips_rot` on the mask,
/// as `vips_convsep` does for its second pass).
fn rot90_kernel(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let in_h = data.len();
    let in_w = data[0].len();
    (0..in_w)
        .map(|j| (0..in_h).map(|i| data[in_h - 1 - i][j]).collect())
        .collect()
}

/// Rotate an odd-square kernel matrix by a multiple of 45 degrees using
/// the exact [`Raster::rot45`] ring permutation. The permutation is
/// computed on an index raster so the `f64` coefficients survive without
/// any float quantisation.
fn rot45_kernel(data: &[Vec<f64>], angle: Angle45) -> Vec<Vec<f64>> {
    let n = data.len();
    let indices: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
    let idx = Raster::from_f32_samples(n as u32, n as u32, float_format(1), &indices)
        .expect("index raster for an odd-square kernel is well-formed");
    let rotated = idx.rot45(angle);
    let perm = rotated
        .f32_samples()
        .expect("rot45 preserves the float format");
    (0..n)
        .map(|j| {
            (0..n)
                .map(|i| {
                    let s = perm[j * n + i] as usize;
                    data[s / n][s % n]
                })
                .collect()
        })
        .collect()
}

impl DenseKernel {
    /// The coefficients back as rows, for the matrix rotation helpers.
    fn rows(&self) -> Vec<Vec<f64>> {
        self.coeff.chunks(self.w).map(<[f64]>::to_vec).collect()
    }

    /// This mask rotated 90 degrees clockwise, keeping its scale and its
    /// offset. `vips_rot` copies the mask metadata across, which is
    /// exactly why `convsep.c:94` has to stamp the offset back to zero by
    /// hand rather than rely on the rotation dropping it.
    fn rot90(&self) -> Self {
        self.respun(rot90_kernel(&self.rows()))
    }

    /// This mask rotated by a multiple of 45 degrees, keeping its scale
    /// and its offset (`vips_compass` rotates its mask with `vips_rot45`).
    fn rot45(&self, angle: Angle45) -> Self {
        self.respun(rot45_kernel(&self.rows(), angle))
    }

    /// Rebuild from rotated rows, carrying the two scalars across. Going
    /// back through a [`Kernel`] literal here would silently drop the
    /// offset, since `Kernel` has nowhere to put it.
    fn respun(&self, data: Vec<Vec<f64>>) -> Self {
        DenseKernel {
            w: data[0].len(),
            h: data.len(),
            coeff: data.into_iter().flatten().collect(),
            scale: self.scale,
            offset: self.offset,
        }
    }
}

// ---------------------------------------------------------------------------
// Raster methods
// ---------------------------------------------------------------------------

impl Raster {
    /// Fallible form of [`Raster::conv`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::EmptyKernel`] / [`ConvolutionError::RaggedKernel`]
    /// for a malformed mask, [`ConvolutionError::ZeroScale`] for a zero
    /// scale, [`ConvolutionError::NonFiniteMaskParameter`] for a `NaN` or
    /// infinite one, or [`ConvolutionError::Raster`] on allocation
    /// failure.
    pub fn try_conv(
        &self,
        kernel: &Kernel,
        precision: Precision,
    ) -> Result<Raster, ConvolutionError> {
        let dense = DenseKernel::new(kernel)?;
        conv_raster(self, &dense, precision)
    }

    /// Convolve the image with `kernel` (libvips `vips_conv`).
    ///
    /// Each output pixel is `sum(mask[i] * pixel[i]) / scale`, evaluated
    /// over the input window centred on the pixel with edges replicated.
    /// See the [module docs](crate::convolution) for the integer/float
    /// precision semantics and output formats.
    ///
    /// At [`Precision::Integer`] on a uchar image the result diverges
    /// from an HWY-enabled libvips, by an amount nobody has bounded
    /// (issues #558 and #547); `VIPS_NOVECTOR=1 vips` reproduces this
    /// exactly. See [`Precision::Integer`] for the contract and
    /// [Divergence from stock libvips](crate::convolution#divergence-from-stock-libvips)
    /// for the mechanism.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_conv`].
    #[track_caller]
    pub fn conv(&self, kernel: &Kernel, precision: Precision) -> Raster {
        expect_conv("conv", self.try_conv(kernel, precision))
    }

    /// Fallible form of [`Raster::convsep`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::NotSeparable`] unless the kernel is `1xN` or
    /// `Nx1`, plus the [`Raster::try_conv`] errors.
    pub fn try_convsep(
        &self,
        kernel: &Kernel,
        precision: Precision,
    ) -> Result<Raster, ConvolutionError> {
        let dense = DenseKernel::new(kernel)?;
        if dense.w != 1 && dense.h != 1 {
            return Err(ConvolutionError::NotSeparable {
                width: dense.w as u32,
                height: dense.h as u32,
            });
        }
        // vips_convsep: convolve with the mask, then with the mask
        // rotated 90 degrees. The scale divides in both passes, the offset
        // applies to the first only (`convsep.c:89-94`), which is what the
        // explicit zeroing below says. Both ride on the mask, so neither
        // can drift away from the coefficients it belongs to.
        let first = conv_raster(self, &dense, precision)?;
        let second = dense.rot90().with_offset(0.0);
        conv_raster(&first, &second, precision)
    }

    /// Separable convolution with a 1D kernel (libvips `vips_convsep`):
    /// the image is convolved twice, once with the `1xN` (or `Nx1`) mask
    /// and once with the mask rotated 90 degrees. For a separable mask
    /// such as [`Kernel::gaussmat`] with `separable = true` this matches
    /// the full 2D convolution at a fraction of the cost.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_convsep`].
    #[track_caller]
    pub fn convsep(&self, kernel: &Kernel, precision: Precision) -> Raster {
        expect_conv("convsep", self.try_convsep(kernel, precision))
    }

    /// Fallible form of [`Raster::compass`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::NotOddSquareKernel`] unless the kernel is an
    /// odd-sided square, [`ConvolutionError::TimesOutOfRange`] for a
    /// `times` outside the `1..=1000` bound libvips declares on its
    /// property, plus the [`Raster::try_conv`] errors.
    pub fn try_compass(
        &self,
        kernel: &Kernel,
        times: u32,
        angle: Angle45,
        combine: Combine,
        precision: Precision,
    ) -> Result<Raster, ConvolutionError> {
        let dense = DenseKernel::new(kernel)?;
        if dense.w != dense.h || dense.w % 2 == 0 {
            return Err(ConvolutionError::NotOddSquareKernel {
                width: dense.w as u32,
                height: dense.h as u32,
            });
        }
        // vips bounds `times` at the GObject property, so an out-of-range
        // value never builds an operation there; see `COMPASS_TIMES_MIN`.
        // The upper end matters as much as the lower: the loop below
        // reserves one result raster per round and convolves the whole
        // image into each.
        if !(COMPASS_TIMES_MIN..=COMPASS_TIMES_MAX).contains(&times) {
            return Err(ConvolutionError::TimesOutOfRange {
                times,
                min: COMPASS_TIMES_MIN,
                max: COMPASS_TIMES_MAX,
            });
        }

        // vips_compass: convolve, rotate the mask by `angle`, repeat.
        // The rotation carries the scale and the offset, so the mask stays
        // one object all the way round the loop.
        let mut results = Vec::with_capacity(times as usize);
        let mut mask = dense;
        for _ in 0..times {
            results.push(conv_raster(self, &mask, precision)?);
            mask = mask.rot45(angle);
        }

        // Take the absolute value of every result, then combine
        // (vips_abs + vips_bandrank / vips_sum). All results share one
        // format because they come from the same input and precision.
        let (w, h) = (self.width(), self.height());
        let channels = results[0].format().channels();
        let planes: Vec<Vec<f64>> = results
            .iter()
            .map(samples_f64)
            .collect::<Result<_, RasterError>>()?;
        let n = planes[0].len();
        let mut combined = try_buffer::<f64>(w, h, n)?;
        combined.resize(n, 0.0);
        for i in 0..n {
            let mut acc: f64 = planes[0][i].abs();
            for plane in &planes[1..] {
                let v = plane[i].abs();
                acc = match combine {
                    Combine::Max => acc.max(v),
                    Combine::Sum => acc + v,
                };
            }
            combined[i] = acc;
        }

        let fmt = results[0].format();
        if fmt.is_float() {
            Ok(raster_from_f64(w, h, channels, &combined)?)
        } else {
            // Unsigned inputs stay unsigned for Max; Sum promotes one
            // depth (vips_sum promotes uchar sums; libviprs tops out at 16
            // bits and saturates, as the arithmetic batch does).
            let out_fmt = match combine {
                Combine::Max => fmt,
                Combine::Sum => PixelFormat::with_channels(channels, 2)
                    .expect("validated channel count has a 16-bit format"),
            };
            let max = depth_max(out_fmt);
            let mut vals = try_buffer::<i64>(w, h, n)?;
            vals.extend(combined.iter().map(|&v| (v as i64).min(max)));
            Ok(raster_from_i64(w, h, out_fmt, &vals)?)
        }
    }

    /// Compass-direction convolution (libvips `vips_compass`): convolve
    /// `times` times, rotating `kernel` by `angle` between rounds, and
    /// combine the absolute results with `combine`.
    ///
    /// `times` must be in `1..=1000`, the range libvips declares on the
    /// property (`VIPS_ARG_INT(class, "times", 101, ..., 1, 1000, 2)` in
    /// `convolution/compass.c`); anything outside it is
    /// [`ConvolutionError::TimesOutOfRange`]. The upper end is not
    /// decoration: every round convolves the whole image again and keeps
    /// the result, so an unbounded `times` is an unbounded amount of work
    /// and an unbounded amount of memory.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_compass`].
    #[track_caller]
    pub fn compass(
        &self,
        kernel: &Kernel,
        times: u32,
        angle: Angle45,
        combine: Combine,
        precision: Precision,
    ) -> Raster {
        expect_conv(
            "compass",
            self.try_compass(kernel, times, angle, combine, precision),
        )
    }

    /// Fallible form of [`Raster::gaussblur`].
    ///
    /// # Errors
    ///
    /// The [`Kernel::try_gaussmat`] and [`Raster::try_convsep`] errors,
    /// plus [`ConvolutionError::Raster`] carrying
    /// [`RasterError::AllocationFailed`] if the `sigma < 0.2` copy below
    /// cannot be allocated.
    pub fn try_gaussblur(
        &self,
        sigma: f64,
        min_ampl: f64,
        precision: Precision,
    ) -> Result<Raster, ConvolutionError> {
        // vips_gaussblur: gaussmat would make a 1x1 mask for anything
        // smaller than this, so just copy.
        //
        // Through `try_clone`, not `clone`: a plain clone is an
        // image-sized allocation that aborts the process on failure, and
        // this is the one path in the operation that does not otherwise
        // touch a fallible allocator, so it was the whole of what stopped
        // `try_gaussblur` being abort-free (issue #575).
        if sigma < 0.2 {
            return Ok(self.try_clone()?);
        }
        let mask = Kernel::try_gaussmat(sigma, min_ampl, true, precision)?;
        self.try_convsep(&mask, precision)
    }

    /// Gaussian blur (libvips `vips_gaussblur`): builds the separable
    /// Gaussian mask for `sigma` / `min_ampl` and runs [`Raster::convsep`]
    /// at the given precision. A `sigma` below `0.2` returns a copy, as in
    /// libvips.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_gaussblur`].
    #[track_caller]
    pub fn gaussblur(&self, sigma: f64, min_ampl: f64, precision: Precision) -> Raster {
        expect_conv("gaussblur", self.try_gaussblur(sigma, min_ampl, precision))
    }

    /// Fallible form of [`Raster::sharpen`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::Colour`] when the image has no LabS colourspace
    /// route (for example a 2-band multiband image), plus the
    /// [`Kernel::try_gaussmat`] errors. The LabS round trip this opens and
    /// closes reserves its output buffers fallibly at both ends now, so a
    /// conversion the host cannot allocate arrives here as
    /// [`ConvolutionError::Colour`] carrying [`ColourError::Raster`]
    /// instead of ending the process (issue #672).
    ///
    /// Every allocation this function makes for itself is fallible now: the
    /// widening goes through [`Raster::try_f32_samples`], the clamped L plane
    /// and the two separable blur passes through the module's fallible
    /// reservation helper, and the result is the LabS raster moved rather than
    /// cloned. So an allocation failure in any of them arrives here as
    /// [`ConvolutionError::Raster`] instead of reaching `handle_alloc_error`
    /// and ending the process (issue #627).
    ///
    /// Together with #672 and #685 having done the same for the round trip's
    /// own buffers, that leaves **no image-sized infallible allocation on this
    /// path at all**, which is what the doc here used to point at in both
    /// directions and no longer can.
    ///
    /// What is left is not image-sized. `colour.rs` carries the input's
    /// attachments onto each end of the round trip with `fields.clone()`, an
    /// embedded ICC profile among them, and that copy allocates infallibly, as
    /// does the Gaussian mask row `mask1d` collects. One is bounded by an
    /// attachment and the other by the mask sanity radius rather than by the
    /// pixel count, and the first is the same residue `Raster::try_clone`
    /// names for itself (it is crate-private, so no link). That is the only
    /// sense in which this is still not abort-free.
    pub fn try_sharpen(&self, sigma: f64, m1: f64, m2: f64) -> Result<Raster, ConvolutionError> {
        // vips_sharpen: remember the interpretation, work in LabS.
        let old_interpretation = self.interpretation();
        let labs = self.try_colourspace(Interpretation::Labs)?;
        let channels = labs.format().channels();
        let (rw, rh) = (labs.width(), labs.height());
        let (w, h) = (rw as usize, rh as usize);

        // "We always sharpen a short, so there's no point using a float
        // mask": a separable integer Gaussian at 10% amplitude.
        let mask = Kernel::try_gaussmat(sigma, 0.1, true, Precision::Integer)?;
        let mask1d: Vec<i64> = mask.data[0].iter().map(|&v| rint(v) as i64).collect();
        // The same divisor `vips_convi_gen` takes off the mask, which is
        // what [`intize`] hands the 2D path (`convi.c:758-759`). A
        // gaussmat integer mask has an integer scale anyway, so `rint`
        // here is the sum of the elements unchanged.
        let iscale = rint(mask.scale) as i64;

        // vips_cast_short on the L band: the LabS codes from colourspace
        // are already rounded; clamp into the signed 16-bit range.
        //
        // Through the fallible widening, not `f32_samples`: that one collects,
        // and a `.collect()` allocates through `handle_alloc_error`, which
        // aborts rather than returning. It was the largest allocation on this
        // path and the whole of what kept `try_sharpen` off the abort-free list
        // (issue #627).
        let mut samples = labs.try_f32_samples()?;
        let mut l = try_buffer::<i32>(rw, rh, w * h)?;
        l.extend(
            (0..w * h).map(|p| (samples[p * channels] as f64).clamp(-32768.0, 32767.0) as i32),
        );

        // Separable integer blur of L with the short clip range, exactly
        // vips_convsep at integer precision on a short image.
        let blur_h = convsep_short_pass(&l, w, h, &mask1d, iscale, true)?;
        let blurred = convsep_short_pass(&blur_h, w, h, &mask1d, iscale, false)?;

        // The vips_sharpen LUT, evaluated directly: index i = diff + 32768
        // rescales to +/- 100 as (i - 32767) / 327.67, runs the m1/m2
        // curve capped at y2/-y3, and rounds back to LabS code units.
        //
        // Written back over the widened samples in place. This used to clone
        // them first, which is a second image-sized allocation for a buffer
        // whose a/b (and extra) bands already hold exactly the values that have
        // to survive; only the L band is ever overwritten.
        for p in 0..w * h {
            let v1 = l[p];
            let v2 = blurred[p];
            let diff = (v1 & 0x7fff) - (v2 & 0x7fff);
            let v = (diff + 1) as f64 / 327.67;
            let y = if v < -SHARPEN_X1 {
                (v + SHARPEN_X1) * m2 + -SHARPEN_X1 * m1
            } else if v < SHARPEN_X1 {
                v * m1
            } else {
                (v - SHARPEN_X1) * m2 + SHARPEN_X1 * m1
            };
            let y = y.clamp(-SHARPEN_Y3, SHARPEN_Y2);
            let adjusted = (v1 + rint(y * 327.67) as i32).clamp(0, 32767);
            samples[p * channels] = adjusted as f32;
        }

        // Reattach a/b (and any extra bands, untouched in `samples`) and
        // convert back to the original interpretation. `labs` is moved into the
        // result rather than cloned: it is dead after the widening, and
        // `Clone::clone` on a raster is another whole image copy that aborts on
        // failure instead of returning.
        let mut sharpened = labs;
        for (dst, s) in sharpened
            .data_mut()
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(samples.iter())
        {
            *dst = s.to_ne_bytes();
        }
        Ok(sharpened.try_colourspace(old_interpretation)?)
    }

    /// Unsharp masking for print (libvips `vips_sharpen`): blur the LabS
    /// L channel with a separable integer Gaussian of the given `sigma`,
    /// pass the difference through the two-slope curve (`m1` in flat
    /// areas, `m2` in jaggy areas, thresholds fixed at the libvips
    /// defaults), and add it back. With `m1 == 0` and `m2 == 0` this is an
    /// exact identity for 8-bit sources (see the module docs).
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_sharpen`].
    #[track_caller]
    pub fn sharpen(&self, sigma: f64, m1: f64, m2: f64) -> Raster {
        expect_conv("sharpen", self.try_sharpen(sigma, m1, m2))
    }

    /// Fallible form of [`Raster::spcor`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::BandCountMismatch`] unless the template has the
    /// image's band count, or [`ConvolutionError::Raster`] on allocation
    /// failure.
    pub fn try_spcor(&self, template: &Raster) -> Result<Raster, ConvolutionError> {
        let channels = check_correlation_bands(self, template)?;
        let (w, h) = (self.width(), self.height());
        let (tw, th) = (template.width() as usize, template.height() as usize);
        let n_pels = (tw * th) as f64;
        let input = samples_f64(self)?;
        let refs = samples_f64(template)?;
        let row_stride = w as usize * channels;

        // Pre-generate: per-band template mean and
        // sqrt(sum((ref - mean)^2)) (vips_spcor_pre_generate).
        let mut rmean = vec![0.0f64; channels];
        let mut c1 = vec![0.0f64; channels];
        for b in 0..channels {
            let mut sum = 0.0;
            for p in 0..tw * th {
                sum += refs[p * channels + b];
            }
            rmean[b] = sum / n_pels;
            let mut sum2 = 0.0;
            for p in 0..tw * th {
                let d = refs[p * channels + b] - rmean[b];
                sum2 += d * d;
            }
            c1[b] = sum2.sqrt();
        }

        let (ax, ay) = ((tw / 2) as i64, (th / 2) as i64);
        let mut out = try_buffer::<f64>(w, h, input.len())?;
        out.resize(input.len(), 0.0);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                for b in 0..channels {
                    // Mean of the input window under the template.
                    let mut sum1 = 0.0;
                    for j in 0..th as i64 {
                        let sy = clamp_coord(y + j - ay, h);
                        for i in 0..tw as i64 {
                            let sx = clamp_coord(x + i - ax, w);
                            sum1 += input[sy * row_stride + sx * channels + b];
                        }
                    }
                    let imean = sum1 / n_pels;

                    // Sum-of-squares of the window and the
                    // product-of-differences against the template.
                    let mut sum2 = 0.0;
                    let mut sum3 = 0.0;
                    for j in 0..th as i64 {
                        let sy = clamp_coord(y + j - ay, h);
                        for i in 0..tw as i64 {
                            let sx = clamp_coord(x + i - ax, w);
                            let ip = input[sy * row_stride + sx * channels + b];
                            let rp = refs[(j as usize * tw + i as usize) * channels + b];
                            let t = ip - imean;
                            sum2 += t * t;
                            sum3 += (rp - rmean[b]) * t;
                        }
                    }

                    let c2 = c1[b] * sum2.sqrt();
                    // A constant reference (or window) is regarded as
                    // uncorrelated.
                    let cc = if c2 == 0.0 { 0.0 } else { sum3 / c2 };
                    out[y as usize * row_stride + x as usize * channels + b] = cc;
                }
            }
        }
        Ok(raster_from_f64(w, h, channels, &out)?)
    }

    /// Spatial correlation (libvips `vips_spcor`): each output pixel is
    /// the normalised cross-correlation (`-1..1`) between `template` and
    /// the input window centred on it, per band. The maximum marks the
    /// best match; a perfect match scores exactly `1.0`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_spcor`].
    #[track_caller]
    pub fn spcor(&self, template: &Raster) -> Raster {
        expect_conv("spcor", self.try_spcor(template))
    }

    /// Fallible form of [`Raster::fastcor`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::BandCountMismatch`] unless the template has the
    /// image's band count, or [`ConvolutionError::Raster`] on allocation
    /// failure.
    pub fn try_fastcor(&self, template: &Raster) -> Result<Raster, ConvolutionError> {
        let channels = check_correlation_bands(self, template)?;
        let (w, h) = (self.width(), self.height());
        let (tw, th) = (template.width() as usize, template.height() as usize);
        let input = samples_f64(self)?;
        let refs = samples_f64(template)?;
        let row_stride = w as usize * channels;
        let (ax, ay) = ((tw / 2) as i64, (th / 2) as i64);

        // vips__formatalike: any float input switches both sides to the
        // float path (f32 accumulation, CORR_FLOAT); two unsigned inputs
        // keep the integer path (u32 accumulation with C wrap-around,
        // CORR_INT).
        let float_path = self.format().is_float() || template.format().is_float();

        let mut out = try_buffer::<f64>(w, h, input.len())?;
        out.resize(input.len(), 0.0);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                for b in 0..channels {
                    let o = y as usize * row_stride + x as usize * channels + b;
                    if float_path {
                        let mut sum = 0.0f32;
                        for j in 0..th as i64 {
                            let sy = clamp_coord(y + j - ay, h);
                            for i in 0..tw as i64 {
                                let sx = clamp_coord(x + i - ax, w);
                                let dif = refs[(j as usize * tw + i as usize) * channels + b]
                                    as f32
                                    - input[sy * row_stride + sx * channels + b] as f32;
                                sum += dif * dif;
                            }
                        }
                        out[o] = sum as f64;
                    } else {
                        let mut sum = 0u32;
                        for j in 0..th as i64 {
                            let sy = clamp_coord(y + j - ay, h);
                            for i in 0..tw as i64 {
                                let sx = clamp_coord(x + i - ax, w);
                                let t = refs[(j as usize * tw + i as usize) * channels + b] as i64
                                    - input[sy * row_stride + sx * channels + b] as i64;
                                sum = sum.wrapping_add((t * t) as u32);
                            }
                        }
                        out[o] = sum as f64;
                    }
                }
            }
        }
        Ok(raster_from_f64(w, h, channels, &out)?)
    }

    /// Fast correlation (libvips `vips_fastcor`): each output pixel is the
    /// per-band sum of squared differences between `template` and the
    /// input window centred on it. The minimum marks the best match; a
    /// perfect match scores exactly `0`.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_fastcor`].
    #[track_caller]
    pub fn fastcor(&self, template: &Raster) -> Raster {
        expect_conv("fastcor", self.try_fastcor(template))
    }
}

// ---------------------------------------------------------------------------
// Edge detectors (the abstract VipsEdge op)
// ---------------------------------------------------------------------------

/// `vips_sobel`'s mask (`convolution/edge.c:244-247`). This is the
/// **vertical** derivative; the horizontal one is its
/// [`DenseKernel::rot90`].
const SOBEL_MASK: [[f64; 3]; 3] = [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]];

/// `vips_scharr`'s mask (`convolution/edge.c:277-280`). This one is the
/// **horizontal** derivative where sobel's is the vertical one, which is
/// why the two impulse responses are each other's vertical mirror.
const SCHARR_MASK: [[f64; 3]; 3] = [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]];

/// `vips_prewitt`'s mask (`convolution/edge.c:310-313`), the horizontal
/// derivative like scharr's. All three masks are rank 1, so all three are
/// separable, not just this one: sobel is `[1,0,-1]^T * [1,2,1]`, scharr
/// is `[3,10,3]^T * [-1,0,1]` and prewitt is `[1,1,1]^T * [-1,0,1]`.
/// libvips exploits it on none of them and neither does this port: the
/// three ops all run the same pair of full 2D responses, and a separable
/// pass would round differently on the integer arm. That is the reason to
/// leave them alone, not the false premise that only prewitt factors. The
/// pair is one traversal rather than two passes (issue #562), which is a
/// different saving entirely and costs no rounding.
const PREWITT_MASK: [[f64; 3]; 3] = [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];

/// The scale divisor `vips_edge_build_uchar` stamps on its mask copy
/// (`convolution/edge.c:126`): halving the response keeps a full-swing
/// gradient inside the 8 bits the recentred convolution has to fit into.
const EDGE_UCHAR_SCALE: f64 = 2.0;

/// The offset summand `vips_edge_build_uchar` stamps on its mask copy
/// (`convolution/edge.c:125`), and the zero point the combine step
/// subtracts back out. `convolution/canny.c:83` stamps the same 128 for
/// the same reason.
const EDGE_UCHAR_OFFSET: f64 = 128.0;

/// Edge detectors: the three named 3x3 gradient operators, `vips_sobel`,
/// `vips_scharr` and `vips_prewitt`. They keep their own block because
/// they do not go through the `conv_raster_n` entry point at all: they
/// drive the shared traversal directly, so the combine happens inside it
/// and neither gradient plane is ever built (issue #562).
impl Raster {
    /// The whole abstract `VipsEdge` op (`convolution/edge.c`) for one 3x3
    /// gradient mask: the shared engine behind [`Raster::sobel`],
    /// [`Raster::scharr`] and [`Raster::prewitt`], which differ only in
    /// the matrix they hand it.
    ///
    /// `vips_edge_build` dispatches purely on the input format
    /// (`edge.c:186-200`), and the two arms are not two spellings of one
    /// formula:
    ///
    /// * **uchar** takes the fast arm (`edge.c:113-155`). The mask is
    ///   stamped with [`EDGE_UCHAR_SCALE`] and [`EDGE_UCHAR_OFFSET`] so a
    ///   signed gradient lands centred in the unsigned output range, both
    ///   responses are computed at [`Precision::Integer`], and they are
    ///   recovered as `2 * (p - 128)` and combined as an **abs sum**
    ///   clipped at 255 (`edge.c:97-103`). libvips comments the choice as
    ///   "avoid the sqrt() for uchar", and it is not an approximation of
    ///   the other arm: on a corner where `Gx == Gy` the abs sum is
    ///   `2 * g` where the magnitude is `sqrt(2) * g`, which the measured
    ///   7x7 corner shows directly (sobel reads 58 here and 42 through
    ///   the float arm).
    /// * **every other format** takes the accurate arm
    ///   (`edge.c:158-182`): two [`Precision::Float`] responses to the raw
    ///   mask, then `sqrt(Gx^2 + Gy^2)`, then the `vips_cast_uchar` of
    ///   `edge.c:174`, which clips into `0..=255` and truncates towards
    ///   zero exactly as `conversion/cast.c:566-568` states. That cast is
    ///   [`cast_float_sample`], the same scalar [`Raster::try_cast`] runs.
    ///
    /// The output is uchar either way, keeping the band count, the
    /// dimensions and the metadata of the input.
    ///
    /// Saturation on the uchar arm happens **twice**, and both are load
    /// bearing. The convolution clips its own output into `0..=255` around
    /// the 128 zero point, so the recovered `2 * (p - 128)` spans
    /// `-256..=254`, and the abs sum then clips again at 255. The
    /// asymmetric bound is why the impulse response reads 254 in some
    /// cells of the ring and 255 in others. Both clips survive the fusion
    /// unchanged: the intermediate `Raster` the first one used to be
    /// written into is gone, the clip that filled it is not.
    ///
    /// The float arm keeps libvips' 32-bit intermediates rather than
    /// promoting to `f64`. Each response is rounded to `f32` where
    /// `vips_convf` stores its float image, and the combine then rounds
    /// twice more, once on `Gx^2 + Gy^2` and once on the stored root.
    /// Every one of them moves output bytes, so this is not a chain to
    /// "simplify" to `f64`. The response rounding now lands on the
    /// accumulator instead of on the way into a raster, which is the same
    /// rounding in a cheaper place; the [module docs](crate::convolution)
    /// carry the rule and weigh the other two against each other.
    fn edge_detect(&self, mask: &[[f64; 3]; 3]) -> Result<Raster, ConvolutionError> {
        let rows: Vec<Vec<f64>> = mask.iter().map(|row| row.to_vec()).collect();
        let channels = self.format().channels();
        let (w, h) = (self.width(), self.height());
        let fmt = PixelFormat::with_channels(channels, 1)
            .expect("an existing raster's band count has an 8-bit format");

        // A 1-byte channel is exactly libvips' VIPS_FORMAT_UCHAR: the
        // 16-bit and float carriers are 2 and 4 bytes wide.
        let uchar = self.format().bytes_per_channel() == 1;
        let dense = if uchar {
            DenseKernel::new(&Kernel {
                data: rows,
                scale: EDGE_UCHAR_SCALE,
            })?
            .with_offset(EDGE_UCHAR_OFFSET)
        } else {
            DenseKernel::new(&Kernel {
                data: rows,
                scale: 1.0,
            })?
        };
        let spun = dense.rot90();
        let masks = [&dense, &spun];
        let scan = Scan::new(self, &masks)?;
        let mut data = alloc_op_output(w, h, fmt)?;

        if uchar {
            let ints = [intize(&dense), intize(&spun)];
            let taps: [_; 2] = std::array::from_fn(|m| {
                compact_taps(
                    masks[m].w,
                    masks[m].h,
                    ints[m].coeff.iter().copied(),
                    scan.origin,
                )
            });
            let params: [(i64, i64, i64); 2] =
                std::array::from_fn(|m| (ints[m].scale / 2, ints[m].scale, ints[m].offset));
            let max = depth_max(self.format());
            scan.int(&taps, |i, sums| {
                let mut acc = 0i32;
                for (&sum, &(rounding, iscale, ioffset)) in sums.iter().zip(&params) {
                    // Saturation number one, and it is load bearing: the
                    // convolution clips its own output into `0..=255`
                    // around the 128 zero point, so the recovered
                    // `2 * (p - 128)` spans an asymmetric `-256..=254`.
                    // Reading it out of an accumulator instead of out of
                    // a materialised `Raster` drops the buffer, not the
                    // clip.
                    let p = ((sum + rounding) / iscale + ioffset).clamp(0, max) as i32;
                    acc += (2 * (p - 128)).abs();
                }
                // Saturation number two.
                data[i] = acc.min(255) as u8;
            });
        } else {
            let taps = masks
                .map(|k| compact_taps(k.w, k.h, k.coeff.iter().map(|&v| v / k.scale), scan.origin));
            scan.float(&taps, masks.map(|k| k.offset), |i, sums| {
                // The 32-bit rounding of each response is what
                // `vips_convf` writing a float image does, so it happens
                // here, before the square, exactly as it did when the two
                // gradients were rasters.
                let a = sums[0] as f32;
                let b = sums[1] as f32;
                let square_sum = a * a + b * b;
                let magnitude = f64::from(square_sum).sqrt() as f32;
                // `edge.c:174` is a `vips_cast` call on the whole
                // magnitude image, and [`cast_float_sample`] is one
                // sample of that cast: clip into range, truncate towards
                // zero, `NaN` to `0`. Calling the same scalar
                // `Raster::try_cast` calls is what keeps the two
                // spellings from drifting apart, now that there is no
                // float raster left to hand it.
                data[i] = cast_float_sample(f64::from(magnitude), 1) as u8;
            });
        }

        let mut out = Raster::from_op_output(w, h, fmt, data)?;
        // vips builds the result inside the input's pipeline, so the
        // interpretation and the resolution survive the format change,
        // and so do the attachments: `vips sobel` on a jpeg carrying 186
        // bytes of `exif-data` and a 564-byte ICC profile hands both
        // through unchanged, on either arm. `conv`, `convsep`, `compass`,
        // `gaussblur`, `spcor` and `fastcor` still return
        // `RasterMeta::default()` and drop the fields, where vips carries
        // both through all six; that is issue #719 and it is not fixed
        // here.
        out.carry_meta_from(self);
        Ok(out)
    }
    /// Fallible form of [`Raster::sobel`], which carries the contract:
    /// the output is always uchar, and the combine rule changes with the
    /// input format.
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::Raster`] if the result raster cannot be
    /// allocated. The mask is a compile-time constant with a non-zero
    /// finite scale, so no other variant is reachable today.
    pub fn try_sobel(&self) -> Result<Raster, ConvolutionError> {
        self.edge_detect(&SOBEL_MASK)
    }

    /// Sobel edge detector (libvips `vips_sobel`), which takes no
    /// arguments.
    ///
    /// Answers the 3x3 Sobel mask and the same mask rotated 90 degrees,
    /// then combines the two gradients into an edge map. Both responses
    /// and the combine come out of a single traversal, so nothing between
    /// the input and the output is ever materialised. What follows is the
    /// contract for all three detectors:
    /// [`Raster::scharr`] and [`Raster::prewitt`] are this op with a
    /// different 3x3 mask and nothing else changed.
    ///
    /// **The output is always uchar**, whatever went in. That is a
    /// narrowing step in the middle of a pipeline rather than a neutral
    /// one: `Gray16` comes back `Gray8` and `RgbaF32` comes back
    /// `Rgba8`, four bytes per sample down to one. Width, height, band
    /// count, interpretation, resolution and the attached metadata all
    /// survive.
    ///
    /// **The combine rule changes with the input format**, and the two
    /// rules are different functions rather than two precisions of one
    /// (`edge.c:186-200`):
    ///
    /// * a uchar input takes the fast arm, `|Gx| + |Gy|` **clipped at
    ///   255**, through two integer convolutions;
    /// * every other format takes the accurate arm,
    ///   `sqrt(Gx^2 + Gy^2)` through two float convolutions and then a
    ///   **truncating** cast down to uchar.
    ///
    /// So casting to float first "for accuracy" does not refine the
    /// answer, it swaps the formula: the same 7x7 corner reads 58 through
    /// the uchar arm and 42 through the float one.
    ///
    /// **Alpha is convolved as an ordinary band.** `rgba.sobel()` gives
    /// back an image whose alpha channel is itself an edge map, so a
    /// fully opaque input comes out fully transparent except along its
    /// edges. That is faithful to `vips sobel`, which runs the combine
    /// over `width * Bands` with no alpha case (`edge.c:76-105`), and it
    /// is rarely what a caller wants: split the colour bands off first if
    /// it is not.
    ///
    /// See also [`Raster::scharr`], which saturates far sooner, and
    /// [`Raster::prewitt`], which responds the most weakly, plus
    /// [Divergence from stock libvips](crate::convolution#divergence-from-stock-libvips)
    /// for the uchar arm's gap against an HWY-enabled libvips.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_sobel`].
    #[track_caller]
    pub fn sobel(&self) -> Raster {
        expect_conv("sobel", self.try_sobel())
    }

    /// Fallible form of [`Raster::scharr`]. The output is always uchar
    /// and the combine rule changes with the input format; the contract
    /// is on [`Raster::sobel`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::Raster`] if the result raster cannot be
    /// allocated; see [`Raster::try_sobel`].
    pub fn try_scharr(&self) -> Result<Raster, ConvolutionError> {
        self.edge_detect(&SCHARR_MASK)
    }

    /// Scharr edge detector (libvips `vips_scharr`), which takes no
    /// arguments.
    ///
    /// The same op as [`Raster::sobel`] with the Scharr mask, and the
    /// same contract: always-uchar output, a combine rule that changes
    /// with the input format, and alpha edge-detected as an ordinary
    /// band. [`Raster::sobel`] spells all three out.
    ///
    /// Scharr's taps sum to four times sobel's (`3 + 10 + 3` against
    /// `1 + 2 + 1`; it is the centre tap alone that is five times as
    /// heavy), so on 8-bit input it reads closer to a threshold than to a
    /// gradient. A plain 10 -> 20 step already answers 160, and a corner
    /// of that same ten-level step saturates outright at 255. Reach for
    /// it when you want edges marked rather than measured.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_scharr`].
    #[track_caller]
    pub fn scharr(&self) -> Raster {
        expect_conv("scharr", self.try_scharr())
    }

    /// Fallible form of [`Raster::prewitt`]. The output is always uchar
    /// and the combine rule changes with the input format; the contract
    /// is on [`Raster::sobel`].
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::Raster`] if the result raster cannot be
    /// allocated; see [`Raster::try_sobel`].
    pub fn try_prewitt(&self) -> Result<Raster, ConvolutionError> {
        self.edge_detect(&PREWITT_MASK)
    }

    /// Prewitt edge detector (libvips `vips_prewitt`), which takes no
    /// arguments.
    ///
    /// The same op as [`Raster::sobel`] with the Prewitt mask, and the
    /// same contract: always-uchar output, a combine rule that changes
    /// with the input format, and alpha edge-detected as an ordinary
    /// band. [`Raster::sobel`] spells all three out.
    ///
    /// Prewitt weights its three taps equally instead of favouring the
    /// centre row, so it responds the most weakly of the three and keeps
    /// its headroom the longest: a 10 -> 20 step answers 30 where sobel
    /// answers 40 and scharr 160.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_prewitt`].
    #[track_caller]
    pub fn prewitt(&self) -> Raster {
        expect_conv("prewitt", self.try_prewitt())
    }
}

// ---------------------------------------------------------------------------
// Canny edge detector
// ---------------------------------------------------------------------------

/// The 2x2 `-1/+1` difference `vips_canny_gradient` builds
/// (`convolution/canny.c:77-80`). `Gy` is `vips_rot90` of it, which is
/// `[[-1, -1], [1, 1]]`, and the rotation carries the mask metadata
/// across, so the uchar arm's offset rides along without being restamped.
const CANNY_GRADIENT_MASK: [[f64; 2]; 2] = [[-1.0, 1.0], [-1.0, 1.0]];

/// The `min_ampl` canny's blur runs at. `canny.c:393` passes `sigma` and
/// `precision` and nothing else, so `vips_gaussblur`'s own default of
/// `0.2` stands.
const CANNY_MIN_AMPL: f64 = 0.2;

/// The eight neighbours `vips_canny_thin_generate` steps to
/// (`canny.c:322-329`), as `(dx, dy)` from the **centre** of the 3x3.
///
/// The C writes them as offsets from the top-left, in typed units built
/// out of `lsk` and `psk`, with the centre at `tp[lsk + psk]`; subtracting
/// that centre is what turns them into these deltas. The order runs
/// **counter-clockwise from top-middle**, which is not the numbering most
/// implementations use, and a table rotated by one step still produces a
/// plausible-looking image:
///
/// ```text
///  1 | 0 | 7
/// ---+---+---
///  2 | X | 6
/// ---+---+---
///  3 | 4 | 5
/// ```
const CANNY_THIN_DIRECTIONS: [(i32, i32); 8] = [
    (0, -1),  // 0: top middle
    (-1, -1), // 1: top left
    (-1, 0),  // 2: middle left
    (-1, 1),  // 3: bottom left
    (0, 1),   // 4: bottom middle
    (1, 1),   // 5: bottom right
    (1, 0),   // 6: middle right
    (1, -1),  // 7: top right
];

/// `VIPS_DEG` (`include/vips/util.h:51`), which is **not** a multiply by
/// `180 / pi`: it divides by `2 * pi` and then multiplies by 360, two
/// roundings in that order. Spelling it the short way moves the last bit
/// of some angles, and canny truncates the result twice, so the spelling
/// is part of the contract rather than a style choice.
#[inline]
fn vips_deg(radians: f64) -> f64 {
    (radians / (2.0 * std::f64::consts::PI)) * 360.0
}

/// `vips_canny_polar_atan2`, the 256-entry table `vips_atan2_init` fills
/// in once at first use (`canny.c:199-222`).
///
/// The index packs a sign-extended 4-bit `gx` into the low nibble and the
/// raw bits 4..=7 of `gy` into the high one, so the table is `atan2` with
/// four bits of precision per axis: each nibble is read back as a signed
/// `-8..=7`, and the angle is coded `0..256` for `0..360` degrees by a
/// **truncating** `256 * theta / 360` with the wraparound coming from the
/// `& 0xFF` rather than from the arithmetic.
///
/// Kept as a literal rather than built lazily, because it is a fixed
/// property of the C and belongs where it can be read. The unit test
/// recomputes every entry in `f64` from [`vips_deg`] and `atan2`, so a
/// typo here fails rather than silently rotating an image. That
/// recomputation is host independent: sixty entries land on exact angles
/// that survive the chain exactly, and the closest of the other 196 sits
/// 0.019 away from a truncation boundary.
#[rustfmt::skip]
const CANNY_ATAN2_LUT: [u8; 256] = [
      0,  64,  64,  64,  64,  64,  64,  64, 192, 192, 192, 192, 192, 192, 192, 192,
      0,  32,  45,  50,  54,  55,  57,  58, 197, 197, 198, 200, 201, 205, 210, 224,
      0,  18,  32,  40,  45,  48,  50,  52, 201, 203, 205, 207, 210, 215, 224, 237,
      0,  13,  23,  32,  37,  41,  45,  47, 206, 208, 210, 214, 218, 224, 232, 242,
      0,   9,  18,  26,  32,  36,  40,  42, 210, 213, 215, 219, 224, 229, 237, 246,
      0,   8,  15,  22,  27,  32,  35,  38, 214, 217, 220, 224, 228, 233, 240, 247,
      0,   6,  13,  18,  23,  28,  32,  35, 218, 220, 224, 227, 232, 237, 242, 249,
      0,   5,  11,  16,  21,  25,  28,  32, 221, 224, 227, 230, 234, 239, 244, 250,
    128, 122, 118, 113, 109, 105, 101,  98, 160, 157, 154, 150, 146, 142, 137, 133,
    128, 122, 116, 111, 106, 102,  99,  96, 162, 160, 156, 153, 149, 144, 139, 133,
    128, 121, 114, 109, 104,  99,  96,  92, 165, 163, 160, 156, 151, 146, 141, 134,
    128, 119, 112, 105, 100,  96,  92,  89, 169, 166, 163, 160, 155, 150, 143, 136,
    128, 118, 109, 101,  96,  91,  87,  85, 173, 170, 168, 164, 160, 154, 146, 137,
    128, 114, 104,  96,  90,  86,  82,  80, 177, 175, 173, 169, 165, 160, 151, 141,
    128, 109,  96,  87,  82,  79,  77,  75, 182, 180, 178, 176, 173, 168, 160, 146,
    128,  96,  82,  77,  73,  72,  70,  69, 186, 186, 185, 183, 182, 178, 173, 160,
];

/// One sample of `POLAR_UCHAR` (`canny.c:111-127`): `(G, theta)` from a
/// pair of gradients already recentred off the mask's 128 offset, so both
/// are in `-128..=127`.
///
/// `G` deliberately **skips the sqrt**, since only relative magnitude
/// matters to the suppression that follows, and it is shifted down to fit
/// a byte. It lands in `0..=64`, never the full byte range: the maximum is
/// `(16384 + 16384 + 256) >> 9`. A test that only checks "it fits in a
/// byte" does not catch a wrong shift.
///
/// The LUT index leans on two's complement and on `>>` being arithmetic,
/// which is why the shift happens on `i32` and the mask afterwards. The
/// index cannot leave `0..=255` whatever it is handed, because
/// `gy & 0xf0` keeps four bits and `(gx >> 4) & 0xf` keeps four more.
#[inline]
fn canny_polar_uchar(gx: i32, gy: i32) -> (u8, u8) {
    debug_assert!((-128..=127).contains(&gx) && (-128..=127).contains(&gy));
    let index = ((gx >> 4) & 0xf) | (gy & 0xf0);
    (
        ((gx * gx + gy * gy + 256) >> 9) as u8,
        CANNY_ATAN2_LUT[index as usize],
    )
}

/// One sample of `POLAR(TYPE)` (`canny.c:134-152`), the arm every format
/// other than uchar takes.
///
/// The C reads both gradients into `double`, does all the arithmetic
/// there and stores the result in the pixel type, so the only narrowing is
/// the one on the way out. Two things this arm does **not** share with the
/// uchar one: `G` has no ceiling at all (a hard 0/255 step reaches 508.5),
/// and a flat region gives `0.5` rather than `0`, because of the `+ 256.0`
/// in the numerator.
///
/// `atan2(gx, gy)` has its arguments swapped relative to the usual
/// convention, so theta is measured from `+y`. Writing the conventional
/// order gives a plausible-looking image rotated by 90 degrees.
#[inline]
fn canny_polar_float(gx: f64, gy: f64) -> (f32, f32) {
    let theta = vips_deg(gx.atan2(gy));
    (
        ((gx * gx + gy * gy + 256.0) / 512.0) as f32,
        (256.0 * ((theta + 360.0) % 360.0) / 360.0) as f32,
    )
}

/// The neighbour of `(x, y)` in direction `k`, clamped into the image.
///
/// `canny.c:414` embeds the polar image by one pixel all round with
/// `VIPS_EXTEND_COPY` before thinning, and clamping the read is what that
/// embed does: an edge lying on the frame compares against duplicates of
/// itself and survives, where supplying zeros outside the image would
/// suppress it.
#[inline]
fn canny_neighbour(
    x: usize,
    y: usize,
    k: i32,
    w: usize,
    h: usize,
    bands: usize,
    band: usize,
) -> usize {
    let (dx, dy) = CANNY_THIN_DIRECTIONS[k as usize];
    let nx = clamp_coord(x as i64 + i64::from(dx), w as u32);
    let ny = clamp_coord(y as i64 + i64::from(dy), h as u32);
    (ny * w + nx) * bands + band
}

/// `THIN(unsigned char)` (`canny.c:252-282`) over the whole plane.
///
/// `theta` picks a direction pair and the residual interpolates linearly
/// between the two neighbours in it, then again between the two opposite
/// ones, and `G` survives only if it beats both. Two things have to be
/// spelled out:
///
/// * The interpolation **widens**. In C `TYPE * int` promotes to `int`, so
///   `lowa * (32 - residual)` is computed at 32 bits and only the result
///   narrows back to a byte. `G` reaches 64 and the weight reaches 32, so
///   the product reaches 2048: `u8` arithmetic here overflows and panics
///   in debug.
/// * The test is `G <= low || G < high`, `<=` against one side and `<`
///   against the other. It reads like a typo and it is not. Where two
///   adjacent pixels share both `G` and `theta` the survivor is always the
///   one on the strict `<` side, and making the comparison symmetric
///   either erases the edge or widens it to two pixels.
fn canny_thin_uchar(polar: &[(u8, u8)], w: usize, h: usize, bands: usize, out: &mut [u8]) {
    for y in 0..h {
        for x in 0..w {
            for band in 0..bands {
                let centre = (y * w + x) * bands + band;
                let (g, theta) = polar[centre];
                let theta = i32::from(theta);
                let low_theta = (theta / 32) & 0x7;
                let high_theta = (low_theta + 1) & 0x7;
                let residual = theta - low_theta * 32;
                let at = |k: i32| i32::from(polar[canny_neighbour(x, y, k, w, h, bands, band)].0);
                // The narrowing back to a byte is the C's assignment to
                // `TYPE`; it never actually truncates, because both
                // weights sum to 32 and `G` is bounded at 64.
                let blend = |a: i32, b: i32| ((a * (32 - residual) + b * residual) / 32) as u8;
                let low = blend(at(low_theta), at(high_theta));
                let high = blend(at((low_theta + 4) & 0x7), at((high_theta + 4) & 0x7));
                out[centre] = if g <= low || g < high { 0 } else { g };
            }
        }
    }
}

/// `THIN(float)` (`canny.c:252-282`), the arm every format other than
/// uchar takes. Same shape as [`canny_thin_uchar`], with the arithmetic
/// kept in the pixel type as the C does: `theta / 32` is a float divide
/// before the truncation, so the bucket edges differ subtly from an
/// integer divide, and every product, sum and division rounds to `f32`.
fn canny_thin_float(polar: &[(f32, f32)], w: usize, h: usize, bands: usize, out: &mut [u8]) {
    let cells = out.as_chunks_mut::<4>().0;
    for y in 0..h {
        for x in 0..w {
            for band in 0..bands {
                let centre = (y * w + x) * bands + band;
                let (g, theta) = polar[centre];
                let low_theta = ((theta / 32.0) as i32) & 0x7;
                let high_theta = (low_theta + 1) & 0x7;
                let residual = theta - (low_theta * 32) as f32;
                let at = |k: i32| polar[canny_neighbour(x, y, k, w, h, bands, band)].0;
                let blend = |a: f32, b: f32| (a * (32.0 - residual) + b * residual) / 32.0;
                let low = blend(at(low_theta), at(high_theta));
                let high = blend(at((low_theta + 4) & 0x7), at((high_theta + 4) & 0x7));
                let kept = if g <= low || g < high { 0.0 } else { g };
                cells[centre] = kept.to_ne_bytes();
            }
        }
    }
}

/// Canny edge detection, `vips_canny` (`convolution/canny.c`).
impl Raster {
    /// Stages 1 and 2 of `vips_canny_build` (`canny.c:393-400`): the
    /// Gaussian blur, then the two 2x2 gradient responses, in that order
    /// and in **one** traversal.
    ///
    /// The arm the gradient runs on is decided by the format of the
    /// **blurred** image, not of the input (`canny.c:81`), and that is the
    /// single most misleading line in the operation. On the float arm
    /// gaussblur has already promoted a uchar input by the time the
    /// gradient stage looks, so the uchar branch cannot fire; the only two
    /// ways into it are a `sigma` below `0.2`, where
    /// [`Raster::try_gaussblur`] short-circuits to a copy, and integer
    /// precision, where the separable convolution keeps the input format.
    /// Since canny's own default is float precision, the uchar arm is off
    /// the default path entirely.
    ///
    /// Both responses come off one pass over one source decode
    /// ([`conv_raster_n`], issue #562), and the order matters here in a
    /// way it does not for the edge detectors: they combine symmetrically,
    /// where canny takes `atan2` off the pair and a swap rotates every
    /// angle by 90 degrees.
    fn canny_gradient(
        &self,
        sigma: f64,
        precision: Precision,
    ) -> Result<[Raster; 2], ConvolutionError> {
        let blurred = self.try_gaussblur(sigma, CANNY_MIN_AMPL, precision)?;
        let rows: Vec<Vec<f64>> = CANNY_GRADIENT_MASK.iter().map(|row| row.to_vec()).collect();
        let mask = DenseKernel::new(&Kernel {
            data: rows,
            scale: 1.0,
        })?;
        // canny.c:81-87. A 1-byte channel is libvips' VIPS_FORMAT_UCHAR;
        // the 16-bit and float carriers are 2 and 4 bytes wide.
        let (mask, gradient_precision) = if blurred.format().bytes_per_channel() == 1 {
            (mask.with_offset(EDGE_UCHAR_OFFSET), Precision::Integer)
        } else {
            (mask, Precision::Float)
        };
        let spun = mask.rot90();
        conv_raster_n(&blurred, [&mask, &spun], gradient_precision)
    }

    /// Fallible form of [`Raster::canny`], which carries the contract.
    ///
    /// # Errors
    ///
    /// [`ConvolutionError::InvalidMaskParameter`] when `sigma` is not a
    /// finite value the Gaussian mask generator accepts (see
    /// [`Kernel::try_gaussmat`]), [`ConvolutionError::MaskTooLarge`] when
    /// the blur mask would exceed the libvips sanity radius, and
    /// [`ConvolutionError::Raster`] if a result raster, the polar scratch or
    /// the float arm's widening cannot be allocated. The gradient mask is a
    /// compile-time constant with a non-zero finite scale, so no kernel-shape
    /// variant is reachable.
    ///
    /// Both arms are abort-free in the sense #575 set: every **image-sized**
    /// allocation on the path is reserved fallibly, the widening the float arm
    /// reads its two gradient rasters back through included, which used to
    /// `.collect()` and so end the process on failure (issue #627).
    ///
    /// Smaller allocations on the path are still infallible and deliberately
    /// out of that scope: the convolution scan's per-row accumulator and its
    /// two clamp tables, the 2x2 gradient mask, and the `fields.clone()` that
    /// carries the input's attachments onto the result. None of them scales
    /// with the pixel count.
    pub fn try_canny(&self, sigma: f64, precision: Precision) -> Result<Raster, ConvolutionError> {
        let [gx, gy] = self.canny_gradient(sigma, precision)?;
        let fmt = gx.format();
        let (w, h) = (gx.width(), gx.height());
        let bands = fmt.channels();
        let (uw, uh) = (w as usize, h as usize);
        let mut data = alloc_op_output(w, h, fmt)?;

        if fmt.bytes_per_channel() == 1 {
            // The polar image vips materialises is one raster of 2 * bands
            // interleaving (G, theta); a pair per sample is the same
            // layout without the doubled band count, and it is what the
            // thin stage reads back.
            let mut polar = try_buffer::<(u8, u8)>(w, h, gx.data().len())?;
            polar.extend(
                gx.data()
                    .iter()
                    .zip(gy.data())
                    .map(|(&a, &b)| canny_polar_uchar(i32::from(a) - 128, i32::from(b) - 128)),
            );
            canny_thin_uchar(&polar, uw, uh, bands, &mut data);
        } else {
            // The fallible widening, not `f32_samples`: that one collects, and
            // a `.collect()` aborts the process on an allocation failure rather
            // than returning, which is what stopped this arm being abort-free
            // when the rest of the module went (issue #627). Both rasters come
            // out of the float gradient stage, so `NotFloatFormat` is not
            // reachable here; `?` covers it anyway rather than asserting it.
            let sx = gx.try_f32_samples()?;
            let sy = gy.try_f32_samples()?;
            let mut polar = try_buffer::<(f32, f32)>(w, h, sx.len())?;
            polar.extend(
                sx.iter()
                    .zip(&sy)
                    .map(|(&a, &b)| canny_polar_float(f64::from(a), f64::from(b))),
            );
            canny_thin_float(&polar, uw, uh, bands, &mut data);
        }

        let mut out = Raster::from_op_output(w, h, fmt, data)?;
        // vips builds the result inside the input's pipeline, so the
        // interpretation, the resolution and the attachments all survive,
        // the same as the edge detectors.
        out.carry_meta_from(self);
        Ok(out)
    }

    /// Canny edge detector (libvips `vips_canny`).
    ///
    /// **This is Canny up to and including non-maximum suppression, and
    /// no further.** `vips_canny_build` blurs, takes a 2x2 gradient,
    /// converts to `(G, theta)`, thins, and stops: there is no
    /// double-thresholding and no edge tracking by connectivity, which is
    /// why the operation takes no hysteresis thresholds. Expect a
    /// suppressed gradient magnitude rather than a binary edge map, so
    /// thinner and greyer than a textbook Canny.
    ///
    /// The four stages, in order (`canny.c:381-428`):
    ///
    /// 1. [`Raster::gaussblur`] at `sigma` and `precision`, with
    ///    `min_ampl` left at its `0.2` default. **This is the only stage
    ///    `precision` reaches.**
    /// 2. A 2x2 `[-1 1; -1 1]` difference and the same mask rotated 90
    ///    degrees, one traversal, at a precision the stage picks for
    ///    itself.
    /// 3. `(G, theta)`, where `G` skips the sqrt because only relative
    ///    magnitude matters downstream, and `theta` is coded `0..256` for
    ///    `0..360` degrees.
    /// 4. Non-maximum suppression along `theta`, against neighbours
    ///    interpolated between the two nearest of eight directions.
    ///
    /// Width, height, band count, interpretation, resolution and the
    /// attached metadata all round-trip.
    ///
    /// # The output format is not the input format
    ///
    /// The gradient stage keys off the format of the **blurred** image
    /// (`canny.c:81`), so `precision` decides the output depth for a uchar
    /// input, indirectly and only through the blur:
    ///
    /// | input | precision | sigma | output |
    /// |---|---|---|---|
    /// | uchar | integer | any | uchar |
    /// | uchar | float | `< 0.2` | uchar |
    /// | uchar | float | `>= 0.2` | float |
    /// | 16-bit or float | any | any | float |
    ///
    /// Canny defaults to float precision in libvips, so the uchar arm is
    /// off the default path. The two arms differ in range as well as in
    /// depth: `G` is bounded at **64** on the uchar arm and unbounded on
    /// the float one, where the same hard step reads 508.5, and a flat
    /// region reads `0.5` rather than `0`.
    ///
    /// # Divergence from the vips CLI on an out-of-range sigma
    ///
    /// `vips canny --sigma 0` does not fail. GObject refuses any value
    /// outside `0.01..1000`, leaves `sigma` at its `1.4` default and still
    /// exits 0, so the CLI silently substitutes a different blur.
    /// `try_canny` honours whatever it is given, exactly as
    /// [`Raster::try_gaussblur`] already does, so a `sigma` below `0.2` is
    /// a no-blur request rather than a quiet 1.4.
    ///
    /// # Panics
    ///
    /// Panics on any [`ConvolutionError`]; see [`Raster::try_canny`].
    #[track_caller]
    pub fn canny(&self, sigma: f64, precision: Precision) -> Raster {
        expect_conv("canny", self.try_canny(sigma, precision))
    }
}

/// Shared correlation validation: equal band counts.
fn check_correlation_bands(image: &Raster, template: &Raster) -> Result<usize, ConvolutionError> {
    let channels = image.format().channels();
    if channels != template.format().channels() {
        return Err(ConvolutionError::BandCountMismatch {
            image: channels,
            template: template.format().channels(),
        });
    }
    Ok(channels)
}

/// One 1D integer convolution pass over an L-code plane in the signed
/// 16-bit domain, horizontal or vertical: the `CONV_INT` inner loop of
/// `vips_convi_gen` with `CLIP_SHORT`, which is how `vips_sharpen` blurs
/// the L channel (`vips_convsep` at integer precision on a short image).
///
/// The output plane is image-sized and `try_sharpen` calls this twice, so it
/// comes from [`try_buffer`] rather than `vec![0i32; n]`: the macro form
/// allocates through `handle_alloc_error` and aborts the process, which a
/// fallible entry point cannot afford (issue #627).
///
/// # Errors
///
/// [`RasterError::AllocationFailed`] if the output plane cannot be reserved.
fn convsep_short_pass(
    src: &[i32],
    w: usize,
    h: usize,
    mask1d: &[i64],
    iscale: i64,
    horizontal: bool,
) -> Result<Vec<i32>, RasterError> {
    let rounding = iscale / 2;
    let half = (mask1d.len() / 2) as i64;
    let mut out = try_buffer::<i32>(w as u32, h as u32, src.len())?;
    out.resize(src.len(), 0);
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let mut sum = 0i64;
            for (k, &m) in mask1d.iter().enumerate() {
                let off = k as i64 - half;
                let (sx, sy) = if horizontal {
                    (x + off, y)
                } else {
                    (x, y + off)
                };
                let sx = clamp_coord(sx, w as u32);
                let sy = clamp_coord(sy, h as u32);
                sum += m * src[sy * w + sx] as i64;
            }
            let v = (sum + rounding) / iscale;
            out[y as usize * w + x as usize] = v.clamp(-32768, 32767) as i32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imageio::MetadataValue;
    use crate::raster::with_f32_samples_alloc_cap;

    /// Deterministic pseudo-random byte stream for synthetic images.
    fn lcg(seed: u32) -> impl FnMut() -> u8 {
        let mut state = seed;
        move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        }
    }

    /// A `w x h` Gray8 raster of deterministic noise.
    fn noise_gray(w: u32, h: u32, seed: u32) -> Raster {
        let mut next = lcg(seed);
        let data: Vec<u8> = (0..w as usize * h as usize).map(|_| next()).collect();
        Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
    }

    /// A `w x h` Rgb8 raster of deterministic noise.
    fn noise_rgb(w: u32, h: u32, seed: u32) -> Raster {
        let mut next = lcg(seed);
        let data: Vec<u8> = (0..w as usize * h as usize * 3).map(|_| next()).collect();
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /// Scalar reference convolution at `(x, y)` with edge replication,
    /// unclipped, in `f64`.
    fn ref_conv(im: &Raster, kernel: &Kernel, x: i64, y: i64) -> Vec<f64> {
        let channels = im.format().channels();
        let kh = kernel.data.len() as i64;
        let kw = kernel.data[0].len() as i64;
        let (ax, ay) = (kw / 2, kh / 2);
        let mut sums = vec![0.0; channels];
        for j in 0..kh {
            for i in 0..kw {
                let sx = (x + i - ax).clamp(0, im.width() as i64 - 1) as u32;
                let sy = (y + j - ay).clamp(0, im.height() as i64 - 1) as u32;
                let px = im.getpoint(sx, sy);
                for (s, &p) in sums.iter_mut().zip(px.iter()) {
                    *s += kernel.data[j as usize][i as usize] * p;
                }
            }
        }
        sums.iter().map(|&s| s / kernel.scale).collect()
    }

    /// The four masks of test_convolution.py: sharp, blur, line, sobel.
    fn ported_masks() -> Vec<Kernel> {
        vec![
            Kernel {
                data: vec![
                    vec![-1.0, -1.0, -1.0],
                    vec![-1.0, 16.0, -1.0],
                    vec![-1.0, -1.0, -1.0],
                ],
                scale: 8.0,
            },
            Kernel {
                data: vec![
                    vec![1.0, 1.0, 1.0],
                    vec![1.0, 1.0, 1.0],
                    vec![1.0, 1.0, 1.0],
                ],
                scale: 9.0,
            },
            Kernel {
                data: vec![
                    vec![1.0, 1.0, 1.0],
                    vec![-2.0, -2.0, -2.0],
                    vec![1.0, 1.0, 1.0],
                ],
                scale: 1.0,
            },
            Kernel {
                data: vec![
                    vec![1.0, 2.0, 1.0],
                    vec![0.0, 0.0, 0.0],
                    vec![-1.0, -2.0, -1.0],
                ],
                scale: 1.0,
            },
        ]
    }

    /// gaussmat integer, sigma 1, min_ampl 0.1: the exact 5x5 libvips
    /// matrix (`rint(20 * exp(-d2 / 2))`) with scale 124, the values
    /// test_create.py::test_gaussmat pins (max 20, centre 20).
    #[test]
    fn gaussmat_integer_exact_matrix() {
        let k = Kernel::gaussmat(1.0, 0.1, false, Precision::Integer);
        assert_eq!(k.width(), 5);
        assert_eq!(k.height(), 5);
        let expected = [
            [0.0, 2.0, 3.0, 2.0, 0.0],
            [2.0, 7.0, 12.0, 7.0, 2.0],
            [3.0, 12.0, 20.0, 12.0, 3.0],
            [2.0, 7.0, 12.0, 7.0, 2.0],
            [0.0, 2.0, 3.0, 2.0, 0.0],
        ];
        for (row, exp) in k.data.iter().zip(expected.iter()) {
            assert_eq!(row.as_slice(), exp.as_slice());
        }
        assert_eq!(k.scale, 124.0);
        assert_eq!(k.max(), 20.0);
        assert_eq!(k.data[2][2], 20.0);
    }

    /// gaussmat float: centre normalised to 1.0, elements are the raw
    /// exponentials, scale is their sum.
    #[test]
    fn gaussmat_float_values() {
        let k = Kernel::gaussmat(1.0, 0.1, false, Precision::Float);
        assert_eq!(k.width(), 5);
        assert_eq!(k.height(), 5);
        assert_eq!(k.data[2][2], 1.0);
        assert!((k.max() - 1.0).abs() < 1e-12);
        let mut sum = 0.0;
        for j in 0..5i64 {
            for i in 0..5i64 {
                let d = ((i - 2) * (i - 2) + (j - 2) * (j - 2)) as f64;
                let v = (-d / 2.0).exp();
                assert!((k.data[j as usize][i as usize] - v).abs() < 1e-12);
                sum += v;
            }
        }
        assert!((k.scale - sum).abs() < 1e-12);
    }

    /// Separable gaussmat is the centre row: 1xN with the same width as
    /// the 2D form, and the row sum as scale.
    #[test]
    fn gaussmat_separable_shapes() {
        let k2 = Kernel::gaussmat(2.0, 0.1, false, Precision::Float);
        let k1 = Kernel::gaussmat(2.0, 0.1, true, Precision::Float);
        assert_eq!(k2.width(), 9);
        assert_eq!(k2.height(), 9);
        assert_eq!(k1.width(), 9);
        assert_eq!(k1.height(), 1);
        assert_eq!(k1.data[0], k2.data[4]);
        let row_sum: f64 = k1.data[0].iter().sum();
        assert!((k1.scale - row_sum).abs() < 1e-12);
        assert_eq!(k1.data[0][4], 1.0);
    }

    /// logmat integer, sigma 1, min_ampl 0.1: 7x7 with centre and max 20
    /// (test_create.py::test_logmat), and the faithful scale of exactly
    /// zero that the C code stores for this mask (logmat.c has no zero
    /// guard, unlike gaussmat.c).
    #[test]
    fn logmat_integer_shape_and_zero_scale() {
        let k = Kernel::logmat(1.0, 0.1, false);
        assert_eq!(k.width(), 7);
        assert_eq!(k.height(), 7);
        assert_eq!(k.max(), 20.0);
        assert_eq!(k.data[3][3], 20.0);
        let sum: f64 = k.data.iter().flatten().sum();
        assert_eq!(sum, 0.0);
        assert_eq!(k.scale, 0.0);
    }

    /// logmat float separable: 7x1 with maximum 1.0, the values the
    /// libvips original pins with precision="float".
    #[test]
    fn logmat_float_separable() {
        let k = Kernel::logmat_with_precision(1.0, 0.1, true, Precision::Float);
        assert_eq!(k.width(), 7);
        assert_eq!(k.height(), 1);
        assert!((k.max() - 1.0).abs() < 1e-12);
        assert_eq!(k.data[0][3], 1.0);
        // Known LoG values at distance 1 and 2.
        assert!((k.data[0][2] - 0.5 * (-0.5f64).exp()).abs() < 1e-12);
        assert!((k.data[0][1] - (-(2.0f64).exp().recip())).abs() < 1e-12);
    }

    /// Convolving with a zero-scale mask (the sigma-1 integer logmat) is a
    /// typed error on both precision paths.
    #[test]
    fn conv_zero_scale_is_typed_error() {
        let im = noise_gray(8, 8, 7);
        let k = Kernel::logmat(1.0, 0.1, false);
        assert!(matches!(
            im.try_conv(&k, Precision::Integer),
            Err(ConvolutionError::ZeroScale)
        ));
        assert!(matches!(
            im.try_conv(&k, Precision::Float),
            Err(ConvolutionError::ZeroScale)
        ));
    }

    /// A 1x1 identity kernel copies the image exactly at integer
    /// precision and reproduces the values as float at float precision.
    #[test]
    fn conv_identity_kernel() {
        let id = Kernel {
            data: vec![vec![1.0]],
            scale: 1.0,
        };
        for im in [noise_gray(9, 7, 1), noise_rgb(9, 7, 2)] {
            let out = im.conv(&id, Precision::Integer);
            assert_eq!(out.format(), im.format());
            assert_eq!(out.data(), im.data());

            let outf = im.conv(&id, Precision::Float);
            assert!(outf.format().is_float());
            assert_eq!(outf.format().channels(), im.format().channels());
            for y in 0..im.height() {
                for x in 0..im.width() {
                    assert_eq!(outf.getpoint(x, y), im.getpoint(x, y));
                }
            }
        }
    }

    /// Box blur on a hand-built 3x3 gray image: interior pixel and the
    /// edge-replicated corner match hand-computed values, integer
    /// truncation and float exactness included.
    #[test]
    fn conv_box_blur_hand_values() {
        let im = Raster::new(
            3,
            3,
            PixelFormat::Gray8,
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90],
        )
        .unwrap();
        let blur = Kernel {
            data: vec![
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0],
            ],
            scale: 9.0,
        };

        let out = im.conv(&blur, Precision::Integer);
        // Interior: (450 + 4) / 9 = 50 (exact).
        assert_eq!(out.getpoint(1, 1), vec![50.0]);
        // Corner with edge replication: 4*10 + 2*20 + 2*40 + 50 = 210;
        // (210 + 4) / 9 truncates to 23.
        assert_eq!(out.getpoint(0, 0), vec![23.0]);

        let outf = im.conv(&blur, Precision::Float);
        assert_eq!(outf.getpoint(1, 1), vec![50.0]);
        let corner = outf.getpoint(0, 0)[0];
        assert!((corner - 210.0 / 9.0).abs() < 1e-5);
    }

    /// conv matches the scalar reference for all four ported masks on
    /// noise, mono and colour: float precision to 1e-3 against the exact
    /// reference, integer precision within 1.0 of the clipped reference.
    #[test]
    fn conv_matches_reference_on_noise() {
        for im in [noise_gray(20, 20, 3), noise_rgb(20, 20, 4)] {
            for kernel in ported_masks() {
                let outf = im.conv(&kernel, Precision::Float);
                let outi = im.conv(&kernel, Precision::Integer);
                assert_eq!(outi.format(), im.format());
                for (x, y) in [(5, 5), (10, 7), (18, 19), (0, 0)] {
                    let expected = ref_conv(&im, &kernel, x as i64, y as i64);
                    let gotf = outf.getpoint(x, y);
                    let goti = outi.getpoint(x, y);
                    for c in 0..expected.len() {
                        assert!(
                            (gotf[c] - expected[c]).abs() < 1e-3,
                            "float conv at ({x},{y}) band {c}: got {}, expected {}",
                            gotf[c],
                            expected[c]
                        );
                        let clipped = expected[c].clamp(0.0, 255.0);
                        assert!(
                            (goti[c] - clipped).abs() <= 1.0,
                            "integer conv at ({x},{y}) band {c}: got {}, expected {clipped}",
                            goti[c]
                        );
                    }
                }
            }
        }
    }

    /// The three fields `vips_convi_gen` reads off a mask
    /// (`convolution/convi.c:757-760`): `rint()`-ed coefficients from
    /// `vips__image_intize`, `rint()` of the mask's **own** scale, and
    /// the rounded offset.
    ///
    /// The scale is the half that moved in #547. Each mask below is one
    /// whose `vips__image_intize` brightness nudge lands somewhere else,
    /// so these assertions fail on the pre-#547 spelling rather than
    /// holding either way; the nudge each one used to produce is named in
    /// the comment beside it.
    #[test]
    fn intize_matches_what_vips_convi_gen_reads() {
        let fractional = Kernel {
            data: vec![vec![0.4, 0.4]],
            scale: 0.8,
        };
        let int = intize(&DenseKernel::new(&fractional).unwrap());
        assert_eq!(int.coeff, vec![0, 0]);
        assert_eq!(int.scale, 1);
        assert_eq!(int.offset, 0);

        // rint(0.8) is 1, and the offset does not follow the rounding:
        // 127.6 rounds to 128 and stays there.
        let int = intize(&DenseKernel::new(&fractional).unwrap().with_offset(127.6));
        assert_eq!(int.scale, 1);
        assert_eq!(int.offset, 128);

        // The issue's own mask: sum 4.6 over scale 1, integer sum 3, so
        // the nudge is rint(1 + (3 - 4.6)) = -1. Dividing by that is what
        // turned a flat grey field black where vips answers white.
        let inverted = Kernel {
            data: vec![vec![3.0, 0.4, 0.4, 0.4, 0.4]],
            scale: 1.0,
        };
        let int = intize(&DenseKernel::new(&inverted).unwrap());
        assert_eq!(int.coeff, vec![3, 0, 0, 0, 0]);
        assert_eq!(int.scale, 1);

        // Rounding the coefficients up rather than away: nudge 2, not 1.
        let up = Kernel {
            data: vec![vec![2.0, 0.6, 0.6]],
            scale: 1.0,
        };
        let int = intize(&DenseKernel::new(&up).unwrap());
        assert_eq!(int.coeff, vec![2, 1, 1]);
        assert_eq!(int.scale, 1);

        // Negative coefficients reach it too: nudge 2, not 1.
        let signed = Kernel {
            data: vec![vec![-1.4, 3.6, -1.4]],
            scale: 1.0,
        };
        let int = intize(&DenseKernel::new(&signed).unwrap());
        assert_eq!(int.coeff, vec![-1, 4, -1]);
        assert_eq!(int.scale, 1);

        // A non-unit scale: nudge 3, not 2.
        let box06 = Kernel {
            data: vec![vec![0.6; 3]; 3],
            scale: 2.0,
        };
        let int = intize(&DenseKernel::new(&box06).unwrap());
        assert_eq!(int.coeff, vec![1; 9]);
        assert_eq!(int.scale, 2);

        // rint() is round-half-to-even on the scale as well: 2.5 is 2.
        let ones = Kernel {
            data: vec![vec![1.0; 3]; 3],
            scale: 2.5,
        };
        assert_eq!(intize(&DenseKernel::new(&ones).unwrap()).scale, 2);

        // A scale that rounds to zero is nudged to 1, the guard
        // `vips__image_intize` writes for its own copy. Both signs of
        // zero take it, and the pre-#547 nudge here was -2.
        for scale in [0.4, -0.4, 0.49] {
            let tiny = Kernel {
                data: vec![vec![1.0, 1.0]],
                scale,
            };
            assert_eq!(
                intize(&DenseKernel::new(&tiny).unwrap()).scale,
                1,
                "a scale of {scale} rounds to zero and must be nudged to 1"
            );
        }

        // The ported blur mask stays untouched: ints in, scale out.
        let int = intize(
            &DenseKernel::new(&ported_masks()[1])
                .unwrap()
                .with_offset(128.0),
        );
        assert_eq!(int.coeff, vec![1; 9]);
        assert_eq!(int.scale, 9);
        assert_eq!(int.offset, 128);

        // rint() is round-half-to-even, like C under the default mode, for
        // the coefficients and for the offset alike.
        let halves = Kernel {
            data: vec![vec![0.5, 1.5, 2.5]],
            scale: 4.5,
        };
        let int = intize(&DenseKernel::new(&halves).unwrap().with_offset(0.5));
        assert_eq!(int.coeff, vec![0, 2, 2]);
        assert_eq!(int.offset, 0);
        let int = intize(&DenseKernel::new(&halves).unwrap().with_offset(1.5));
        assert_eq!(int.offset, 2);

        // A finite offset too large for a C `int` is clamped rather than
        // saturated to `i64::MAX`, so the summand add on the unsigned arm
        // cannot overflow. libvips reads the offset as `int` and gets the
        // same bound.
        let int = DenseKernel::new(&halves).unwrap();
        assert_eq!(intize(&int.with_offset(9.3e18)).offset, i64::from(i32::MAX));
        let int = DenseKernel::new(&halves).unwrap();
        assert_eq!(
            intize(&int.with_offset(-9.3e18)).offset,
            i64::from(i32::MIN)
        );
    }

    /// A 5x1 flat field on each of the three carriers the integer
    /// convolution arm has: clipped uchar, clipped ushort, and the
    /// unclipped float-input path of `vips_convi_gen`.
    fn flat_5x1() -> [Raster; 3] {
        [
            Raster::new(5, 1, PixelFormat::Gray8, vec![100u8; 5]).unwrap(),
            Raster::new(
                5,
                1,
                PixelFormat::Gray16,
                (0..5).flat_map(|_| 1000u16.to_ne_bytes()).collect(),
            )
            .unwrap(),
            Raster::from_f32_samples(5, 1, float_format(1), &[100.0f32; 5]).unwrap(),
        ]
    }

    /// #547: the integer arm divides by `rint()` of the mask's own scale,
    /// the way `vips_convi_gen` does, and not by the brightness-corrected
    /// scale `vips__image_intize` computes and libvips never reads.
    ///
    /// Every expectation below was measured on vips 8.18.4 under
    /// `VIPS_NOVECTOR=1`, the scalar `vips_convi_gen` arm this module
    /// ports, and re-measured on the default HWY vector path with
    /// `env -u VIPS_NOVECTOR`, which is the only way to unset it: an
    /// empty `VIPS_NOVECTOR=` still counts as set, because
    /// `iofuncs/vector.cpp:89` is a bare `g_getenv` and the empty string
    /// is a non-`NULL` pointer. Six of the seven print the same bytes on
    /// both paths. The seventh, `zeroed`, does not and is ordinary #558
    /// territory: its coefficients round away to a zero mask on the
    /// scalar arm, while the vector path requantises `0.4 / 0.8`
    /// exactly and answers 100 on the uchar fixture. As everywhere else
    /// in this module, the pin follows the scalar arm.
    ///
    /// The fixtures are flat, so one number describes the whole output.
    /// The comment beside each row is what libviprs answered before the
    /// fix, which is what makes the row worth pinning at all: the first
    /// four move on at least one carrier, and the first one has the
    /// divisor coming out negative, so it is black where vips is white.
    #[test]
    fn conv_integer_divides_by_the_original_mask_scale() {
        // mask, scale, [uchar, ushort, float] from vips; `was` in the
        // comment is the pre-#547 libviprs answer.
        let cases = [
            // nudge -1: was [0, 0, -300.0], black where vips is white.
            (
                "issue",
                vec![vec![3.0, 0.4, 0.4, 0.4, 0.4]],
                1.0,
                [255.0, 3000.0, 300.0],
            ),
            // nudge 2: was [200, 2000, 200.0].
            ("up", vec![vec![2.0, 0.6, 0.6]], 1.0, [255.0, 4000.0, 400.0]),
            // nudge 2: was [100, 1000, 100.0].
            (
                "signed",
                vec![vec![-1.4, 3.6, -1.4]],
                1.0,
                [200.0, 2000.0, 200.0],
            ),
            // nudge 3: was [255, 3000, 300.0]. The uchar cell clips
            // either way, which is exactly how a mask like this hides on
            // an 8-bit fixture.
            ("box06", vec![vec![0.6; 3]; 3], 2.0, [255.0, 4500.0, 450.0]),
            // Controls, where the nudge already agreed with rint(scale)
            // and nothing moves.
            ("ones", vec![vec![1.0; 3]; 3], 2.5, [255.0, 4500.0, 450.0]),
            ("blur", vec![vec![1.0; 3]; 3], 9.0, [100.0, 1000.0, 100.0]),
            // Every coefficient rounds away, so the mask is all zeros and
            // the divisor cannot show.
            ("zeroed", vec![vec![0.4, 0.4]], 0.8, [0.0, 0.0, 0.0]),
        ];

        for (name, data, scale, want) in cases {
            let kernel = Kernel { data, scale };
            for (im, &expected) in flat_5x1().iter().zip(&want) {
                let out = im.try_conv(&kernel, Precision::Integer).unwrap();
                for x in 0..5 {
                    let got = out.getpoint(x, 0)[0];
                    assert!(
                        (got - expected).abs() < 1e-6,
                        "{name} at ({x}, 0) on {:?}: got {got}, vips says {expected}",
                        im.format()
                    );
                }
            }
        }
    }

    /// A mask scale that rounds to zero has no libvips answer to match,
    /// and libviprs does not pretend otherwise.
    ///
    /// `vips_convi_gen` reads `int scale = rint(...)`, so a scale under
    /// 0.5 leaves it holding `0` and the generator divides by it.
    /// Measured on 8.18.4 with `[[1.0, 1.0]]` at scale 0.4: the two
    /// integer arms answer `0` (aarch64 `sdiv` returns zero rather than
    /// trapping, which is not a defined result, and x86 would trap) and
    /// the float-input arm prints `inf`. libviprs nudges the divisor to
    /// `1` instead, which is the guard `vips__image_intize` writes for
    /// its own copy at `convi.c:895-897`, and is the only total answer on
    /// offer. That makes these three the deliberate divergence in this
    /// change, so they are pinned rather than left to drift.
    #[test]
    fn conv_integer_scale_rounding_to_zero_divides_by_one() {
        let kernel = Kernel {
            data: vec![vec![1.0, 1.0]],
            scale: 0.4,
        };
        // 2 * the flat value, undivided, rather than vips' 0 / 0 / inf.
        for (im, expected) in flat_5x1().iter().zip([200.0, 2000.0, 200.0]) {
            let out = im.try_conv(&kernel, Precision::Integer).unwrap();
            let got = out.getpoint(0, 0)[0];
            assert!(
                (got - expected).abs() < 1e-6,
                "zero-rounding scale on {:?}: got {got}, want {expected}",
                im.format()
            );
        }
    }

    /// #575: the `f64` widening buffer is reserved fallibly, so a request
    /// the host cannot serve surfaces as
    /// [`RasterError::AllocationFailed`] instead of reaching
    /// `handle_alloc_error` and aborting the process.
    ///
    /// `samples_f64` is the biggest allocation on the convolution path,
    /// eight bytes per sample where the source carries one, and every
    /// `try_` entry point in this module goes through it, so the plain
    /// `.collect()` it used to be made the whole fallible surface
    /// abortable. Driving the widening itself to failure would need a
    /// raster the machine cannot hold, so the reservation is exercised
    /// directly at a length no allocator can serve; the widening is
    /// covered by every other convolution test in this file.
    #[test]
    fn samples_f64_reserves_fallibly_rather_than_aborting() {
        assert!(matches!(
            try_buffer::<f64>(1, 1, usize::MAX / 4),
            Err(RasterError::AllocationFailed { .. })
        ));
        // The byte count in the error is the size of the request, which
        // is eight times the sample count, not the raster's own length.
        assert!(matches!(
            try_buffer::<f64>(3, 2, usize::MAX),
            Err(RasterError::AllocationFailed {
                width: 3,
                height: 2,
                bytes: usize::MAX
            })
        ));
        // The ordinary path still widens every sample exactly.
        let im = Raster::new(2, 1, PixelFormat::Gray8, vec![7, 9]).unwrap();
        assert_eq!(samples_f64(&im).unwrap(), vec![7.0, 9.0]);
    }

    /// #627: the `f32` widening `try_sharpen` sits on is fallible, so an
    /// allocation the host cannot serve arrives as
    /// [`ConvolutionError::Raster`] instead of reaching `handle_alloc_error`
    /// and ending the process.
    ///
    /// #575 took nine of the eleven entry points here abort-free and
    /// `try_sharpen` could not follow, because the abort was not in this file:
    /// it was the `.collect()` inside [`Raster::f32_samples`]. A `try_` API
    /// that aborts is worse than an infallible one, since a caller reasonably
    /// reads the `Result` as covering allocation.
    ///
    /// The widening is reached at a buildable input through the `cfg(test)`
    /// ceiling in `raster.rs`; a LabS raster whose samples genuinely exhaust
    /// the allocator is far past the construction budget. The uncapped call
    /// alongside it is what stops the ceiling passing for a guard on its own.
    #[test]
    fn sharpen_widening_returns_typed_error_not_abort() {
        let im = noise_rgb(6, 4, 31);
        let capped = with_f32_samples_alloc_cap(16, || im.try_sharpen(1.0, 1.0, 2.0));
        // Summarised rather than `{capped:?}`, which prints the whole pixel
        // buffer of the raster the failing case wrongly returns.
        let got = capped.as_ref().map(|r| (r.width(), r.height(), r.format()));
        assert!(
            matches!(
                capped,
                Err(ConvolutionError::Raster(
                    RasterError::AllocationFailed { .. }
                ))
            ),
            "an unservable sharpen widening must be a typed error, got {got:?}"
        );
        assert!(im.try_sharpen(1.0, 1.0, 2.0).is_ok());
    }

    /// #627: the same widening on canny's float arm, which reads the two
    /// gradient rasters back as `f32` before the non-maximum suppression.
    ///
    /// The uchar arm never widens, so it stays green under the same ceiling.
    /// That split is what pins the guard to the float arm rather than to
    /// canny in general: canny's own default is float precision, so the arm
    /// that used to abort is the one on the default path.
    #[test]
    fn canny_float_arm_widening_returns_typed_error_not_abort() {
        let im = noise_gray(8, 8, 32);
        let capped = with_f32_samples_alloc_cap(16, || im.try_canny(1.4, Precision::Float));
        let got = capped.as_ref().map(|r| (r.width(), r.height(), r.format()));
        assert!(
            matches!(
                capped,
                Err(ConvolutionError::Raster(
                    RasterError::AllocationFailed { .. }
                ))
            ),
            "an unservable canny widening must be a typed error, got {got:?}"
        );
        assert!(im.try_canny(1.4, Precision::Float).is_ok());
        // sigma < 0.2 short-circuits the blur to a copy, so the gradient runs
        // on a uchar image and the float widening is never reached.
        assert!(
            with_f32_samples_alloc_cap(16, || im.try_canny(0.1, Precision::Integer)).is_ok(),
            "the uchar arm does not widen, so the ceiling must not reach it"
        );
    }

    /// #627: the three image-sized `i32` planes `try_sharpen` builds for
    /// itself, the clamped L band and the two separable blur passes, are
    /// reserved through [`try_buffer`] rather than `vec![0i32; n]`.
    ///
    /// The count is the load-bearing half. `vec![0i32; n]` and `try_buffer`
    /// behave identically at every size a test can build, and differ only in
    /// what they do when the allocation fails, which is that one aborts the
    /// process and the other returns; so putting the macro back in any one of
    /// the three places is invisible to an assertion on the result alone. All
    /// three planes are `w * h` `i32`s, so a byte ceiling cannot tell them
    /// apart either, and only the count moves when one of them regresses.
    ///
    /// The capped call alongside it pins the other half: the failure that
    /// reaches a caller is [`ConvolutionError::Raster`] and not an abort.
    #[test]
    fn sharpen_scratch_planes_are_fallible_not_aborting() {
        let im = noise_rgb(6, 4, 33);

        let (ok, calls) = with_conv_buffer_probe(u64::MAX, || im.try_sharpen(1.0, 1.0, 2.0));
        assert!(ok.is_ok());
        assert_eq!(
            calls, 3,
            "the L plane and both blur passes must each reserve fallibly"
        );

        let (capped, _) = with_conv_buffer_probe(16, || im.try_sharpen(1.0, 1.0, 2.0));
        let got = capped.as_ref().map(|r| (r.width(), r.height(), r.format()));
        assert!(
            matches!(
                capped,
                Err(ConvolutionError::Raster(
                    RasterError::AllocationFailed { .. }
                ))
            ),
            "an unservable sharpen plane must be a typed error, got {got:?}"
        );
    }

    /// FNV-1a over a whole buffer, so a full `data()` comparison fits in
    /// one pinned constant.
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    /// A `w x h` single-band float raster of deterministic noise spanning
    /// negatives, for the two arms that never clip.
    fn noise_float(w: u32, h: u32, seed: u32) -> Raster {
        let mut next = lcg(seed);
        let samples: Vec<f32> = (0..w as usize * h as usize)
            .map(|_| f32::from(next()) - 128.0)
            .collect();
        Raster::from_f32_samples(w, h, float_format(1), &samples).unwrap()
    }

    /// The regression guard for the mask offset: at `offset` 0
    /// `conv_raster` reproduces the pre-offset output byte for byte. The
    /// digests are FNV-1a over the whole `data()` buffer, captured from
    /// the implementation before the summand existed, for all four
    /// `ported_masks()` at both precisions on the uchar, colour, and float
    /// input arms. The public `conv` has to agree with them as well.
    ///
    /// One case at the bottom deliberately does *not* reproduce the base
    /// bytes; see the comment there.
    #[test]
    fn conv_raster_at_offset_zero_reproduces_the_pre_offset_bytes() {
        // Per input, per mask: [integer-precision digest, float-precision
        // digest].
        let cases: [(Raster, [[u64; 2]; 4]); 3] = [
            (
                noise_gray(20, 20, 3),
                [
                    [0x174d_88ee_bcd8_e438, 0x3bd6_7d6f_0acf_a5e1],
                    [0x9fc6_4404_fea1_81aa, 0x846e_0e05_d2fc_f4a8],
                    [0xe3d0_476d_d341_4c0c, 0x2040_8791_364f_a5fa],
                    [0xc498_5b9d_52c5_d19d, 0x2161_3ea2_a21a_d2a2],
                ],
            ),
            (
                noise_rgb(20, 20, 4),
                [
                    [0xafe4_ffc1_6589_4821, 0x39af_ddc5_5d7b_224a],
                    [0x394a_993d_89dc_9a48, 0x384b_b952_bc40_fbd2],
                    [0xac22_fd7d_f031_f7ee, 0xc48f_d1e2_df84_e123],
                    [0x3fbb_9591_64dc_34f1, 0x9fa3_91a6_b1da_13dc],
                ],
            ),
            (
                noise_float(20, 20, 7),
                [
                    [0x3ac8_42a9_30db_a65a, 0x3ac8_42a9_30db_a65a],
                    [0x8c9f_1585_edd2_37cc, 0x9349_0ad8_c550_c5bd],
                    [0xa98a_afa4_f29e_4792, 0xa98a_afa4_f29e_4792],
                    [0xbed3_519d_d204_b58a, 0xbed3_519d_d204_b58a],
                ],
            ),
        ];

        for (im, per_mask) in &cases {
            for (mask, digests) in ported_masks().iter().zip(per_mask) {
                let dense = DenseKernel::new(mask).unwrap();
                for (precision, expected) in [
                    (Precision::Integer, digests[0]),
                    (Precision::Float, digests[1]),
                ] {
                    let out = conv_raster(im, &dense, precision).unwrap();
                    let shim = im.conv(mask, precision);
                    assert_eq!(
                        fnv1a(out.data()),
                        expected,
                        "offset 0 changed the {precision:?} output for {:?} input, mask {mask:?}",
                        im.format()
                    );
                    assert_eq!(out.format(), shim.format());
                    assert_eq!(
                        out.data(),
                        shim.data(),
                        "conv no longer agrees with the engine it delegates to"
                    );
                }
            }
        }

        // The one deliberate deviation from base, and the reason for the
        // `### Fixed` CHANGELOG entry. On the integer-precision float-input
        // arm the result went from `sum / iscale` to
        // `sum / iscale + ioffset as f64`, which is bit-identical for every
        // f64 except `-0.0`: adding `+0.0` promotes it to `+0.0`, and the
        // sign bit reaches `data()`. It takes a negative integer scale to
        // reach, which no `ported_masks()` entry has, and which the mask
        // below does. vips 8.18.4 writes `+0.0` here, C's
        // `(sum / scale) + offset` promoting the `int 0` the same way, so
        // base libviprs was the one diverging. libviprs' own float-precision
        // arm already wrote `+0.0`, and now both arms agree: the two
        // digests below are the same number.
        //
        // Row 0 is all zeros (the `-0.0` sites), row 1 all 5.0, so the rest
        // of the buffer is pinned alongside the sign bit.
        const BASE_MINUS_ZERO: u64 = 0x5e65_8c80_2e0c_dfa5;
        const VIPS_PLUS_ZERO: u64 = 0x453c_3d6d_5edc_8fa5;
        let samples: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0];
        let im = Raster::from_f32_samples(4, 2, float_format(1), &samples).unwrap();
        let negative = Kernel {
            data: vec![vec![1.0, 1.0, 1.0]],
            scale: -3.0,
        };
        let dense = DenseKernel::new(&negative).unwrap();
        for precision in [Precision::Integer, Precision::Float] {
            let out = conv_raster(&im, &dense, precision).unwrap();
            assert_eq!(
                fnv1a(out.data()),
                VIPS_PLUS_ZERO,
                "negative-scale float input at {precision:?} precision no longer matches vips"
            );
        }
        assert_ne!(
            VIPS_PLUS_ZERO, BASE_MINUS_ZERO,
            "the base digest and the vips-matching one must differ, or this case pins nothing"
        );
    }

    /// The uchar recipe of `convolution/edge.c` (`offset` 128, `scale` 2 on
    /// a sobel mask): the offset lands before the clip, so a flat region
    /// reads as the 128 zero point, a falling edge saturates at 0 and a
    /// rising edge at 255. Without the offset the same input never reaches
    /// the top of the range.
    #[test]
    fn conv_raster_offset_clips_symmetrically_on_the_uchar_arm() {
        // Three columns wide so the horizontal edge replication is a
        // no-op; rows 2..=4 are the bright band.
        let rows: [u8; 7] = [0, 0, 255, 255, 255, 0, 0];
        let data: Vec<u8> = rows.iter().flat_map(|&v| [v, v, v]).collect();
        let im = Raster::new(3, 7, PixelFormat::Gray8, data).unwrap();
        let mask = Kernel {
            data: vec![
                vec![1.0, 2.0, 1.0],
                vec![0.0, 0.0, 0.0],
                vec![-1.0, -2.0, -1.0],
            ],
            scale: 2.0,
        };
        let dense = DenseKernel::new(&mask).unwrap().with_offset(128.0);

        let out = conv_raster(&im, &dense, Precision::Integer).unwrap();
        assert_eq!(out.format(), PixelFormat::Gray8);
        // sum 0 -> (0 + 1) / 2 + 128; sum -1020 -> -509 + 128 -> clipped to
        // 0; sum +1020 -> 510 + 128 -> clipped to 255.
        let got: Vec<f64> = (0..7).map(|y| out.getpoint(1, y)[0]).collect();
        assert_eq!(got, vec![128.0, 0.0, 0.0, 128.0, 255.0, 255.0, 128.0]);

        // The same mask with no offset: the low end still clips at 0, but
        // nothing reaches 255 and the flat region is black.
        let plain =
            conv_raster(&im, &DenseKernel::new(&mask).unwrap(), Precision::Integer).unwrap();
        let got: Vec<f64> = (0..7).map(|y| plain.getpoint(1, y)[0]).collect();
        assert_eq!(got, vec![0.0, 0.0, 0.0, 0.0, 255.0, 255.0, 0.0]);
    }

    /// A float input never clips, on either arm. Under integer precision
    /// the offset is the rounded one `vips_convi_gen` reads
    /// (`rint()`, half to even); under float precision `vips_convf_gen`
    /// keeps the raw double.
    #[test]
    fn conv_raster_offset_is_unclipped_on_the_float_input_arms() {
        let im = Raster::from_f32_samples(2, 2, float_format(1), &[10.0; 4]).unwrap();
        let mask = ported_masks().remove(1); // the 3x3 all-ones blur, scale 9

        // Integer precision, float input: intize gives an all-ones mask and
        // scale 9, so the sum is exactly 10 before the offset.
        for (offset, want) in [
            (1000.0, 1010.0),
            (-1000.0, -990.0),
            (0.5, 10.0),
            (1.5, 12.0),
        ] {
            let dense = DenseKernel::new(&mask).unwrap().with_offset(offset);
            let out = conv_raster(&im, &dense, Precision::Integer).unwrap();
            assert_eq!(out.format(), float_format(1));
            let got = out.getpoint(0, 0)[0];
            assert!(
                (got - want).abs() < 1e-4,
                "integer precision, float input, offset {offset}: got {got}, want {want}"
            );
        }

        // Float precision keeps the offset unrounded and unclipped.
        for (offset, want) in [(1000.0, 1010.0), (-1000.0, -990.0), (0.5, 10.5)] {
            let dense = DenseKernel::new(&mask).unwrap().with_offset(offset);
            let out = conv_raster(&im, &dense, Precision::Float).unwrap();
            let got = out.getpoint(1, 1)[0];
            assert!(
                (got - want).abs() < 1e-4,
                "float precision, offset {offset}: got {got}, want {want}"
            );
        }
    }

    /// A non-finite mask scalar is a typed error, not a panic and not a
    /// silent wrong answer. Before the guard, `f64::INFINITY` as an offset
    /// saturated `rint(offset) as i64` to `i64::MAX` and overflowed the
    /// summand add (a panic in debug, black pixels in release), `NaN` as an
    /// offset was silently dropped to 0 on the integer arms while the float
    /// arm returned `NaN`, a `NaN` scale slipped past the `scale == 0.0`
    /// test into an integer divide by zero, and an infinite scale past it
    /// into an all-zero image.
    #[test]
    fn conv_raster_rejects_a_non_finite_scale_or_offset() {
        let im = noise_gray(4, 4, 11);
        let ones = Kernel {
            data: vec![vec![1.0, 1.0, 1.0]],
            scale: 3.0,
        };
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for precision in [Precision::Integer, Precision::Float] {
                let dense = DenseKernel::new(&ones).unwrap().with_offset(bad);
                assert!(matches!(
                    conv_raster(&im, &dense, precision),
                    Err(ConvolutionError::NonFiniteMaskParameter {
                        param: "offset",
                        value
                    }) if value.is_nan() == bad.is_nan()
                ));

                let scaled = Kernel {
                    data: ones.data.clone(),
                    scale: bad,
                };
                assert!(matches!(
                    im.try_conv(&scaled, precision),
                    Err(ConvolutionError::NonFiniteMaskParameter { param: "scale", .. })
                ));
            }
        }

        // A zero scale keeps its own more specific error.
        let zero = Kernel {
            data: ones.data.clone(),
            scale: 0.0,
        };
        assert!(matches!(
            im.try_conv(&zero, Precision::Integer),
            Err(ConvolutionError::ZeroScale)
        ));
    }

    /// A rotated mask keeps its scale and its offset. Going back through a
    /// `Kernel` literal would drop the offset silently, which is the whole
    /// reason the rotation lives on `DenseKernel`.
    #[test]
    fn rotating_a_dense_kernel_carries_the_scale_and_offset() {
        let sobel = Kernel {
            data: vec![
                vec![1.0, 2.0, 1.0],
                vec![0.0, 0.0, 0.0],
                vec![-1.0, -2.0, -1.0],
            ],
            scale: 2.0,
        };
        let dense = DenseKernel::new(&sobel).unwrap().with_offset(128.0);
        for spun in [dense.rot90(), dense.rot45(Angle45::D45)] {
            assert!(
                (spun.scale - 2.0).abs() < f64::EPSILON,
                "rotation dropped the scale: got {}",
                spun.scale
            );
            assert!(
                (spun.offset - 128.0).abs() < f64::EPSILON,
                "rotation dropped the offset: got {}",
                spun.offset
            );
            assert_eq!((spun.w, spun.h), (3, 3));
        }

        // rot90 of a 1xN is an Nx1, coefficients in order.
        let row = Kernel {
            data: vec![vec![1.0, 2.0, 3.0]],
            scale: 6.0,
        };
        let spun = DenseKernel::new(&row).unwrap().rot90();
        assert_eq!((spun.w, spun.h), (1, 3));
        assert_eq!(spun.coeff, vec![1.0, 2.0, 3.0]);
    }

    /// convsep equals the full 2D convolution with the outer-product mask
    /// at float precision (the Gaussian factorises exactly).
    #[test]
    fn convsep_equals_conv_2d_float() {
        let sep = Kernel::gaussmat(1.5, 0.1, true, Precision::Float);
        let full = Kernel::gaussmat(1.5, 0.1, false, Precision::Float);
        for im in [noise_gray(16, 12, 5), noise_rgb(16, 12, 6)] {
            let a = im.conv(&full, Precision::Float);
            let b = im.convsep(&sep, Precision::Float);
            for y in 0..im.height() {
                for x in 0..im.width() {
                    let pa = a.getpoint(x, y);
                    let pb = b.getpoint(x, y);
                    for c in 0..pa.len() {
                        assert!(
                            (pa[c] - pb[c]).abs() < 1e-2,
                            "sep mismatch at ({x},{y}) band {c}: {} vs {}",
                            pa[c],
                            pb[c]
                        );
                    }
                }
            }
        }
    }

    /// convsep accepts Nx1 masks too, and matches two manual conv passes
    /// (second pass with the 90-degree-rotated mask).
    #[test]
    fn convsep_vertical_mask_matches_manual_passes() {
        // Asymmetric 1D mask: rotation direction matters.
        let row = Kernel {
            data: vec![vec![1.0, 2.0, 4.0]],
            scale: 7.0,
        };
        let col = Kernel {
            data: vec![vec![1.0], vec![2.0], vec![4.0]],
            scale: 7.0,
        };
        let im = noise_gray(10, 10, 9);

        // Horizontal mask: pass 1 with the row, pass 2 with the column in
        // the same order (rot90 clockwise of a 1xN row keeps order).
        let manual = im
            .conv(&row, Precision::Integer)
            .conv(&col, Precision::Integer);
        assert_eq!(im.convsep(&row, Precision::Integer).data(), manual.data());

        // Vertical mask: rot90 clockwise of an Nx1 column reverses it.
        let row_rev = Kernel {
            data: vec![vec![4.0, 2.0, 1.0]],
            scale: 7.0,
        };
        let manual = im
            .conv(&col, Precision::Integer)
            .conv(&row_rev, Precision::Integer);
        assert_eq!(im.convsep(&col, Precision::Integer).data(), manual.data());
    }

    /// The internal 45-degree kernel rotation matches the classic sobel
    /// rotation (one clockwise ring step), and eight steps are an
    /// identity.
    #[test]
    fn rot45_kernel_sobel() {
        let sobel = vec![
            vec![1.0, 2.0, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![-1.0, -2.0, -1.0],
        ];
        let once = rot45_kernel(&sobel, Angle45::D45);
        assert_eq!(
            once,
            vec![
                vec![0.0, 1.0, 2.0],
                vec![-1.0, 0.0, 1.0],
                vec![-2.0, -1.0, 0.0],
            ]
        );
        let full = rot45_kernel(&sobel, Angle45::D0);
        assert_eq!(full, sobel);
        let mut spun = sobel.clone();
        for _ in 0..8 {
            spun = rot45_kernel(&spun, Angle45::D45);
        }
        assert_eq!(spun, sobel);
    }

    /// compass with times 1 is |conv|; with times 2 at D45 it matches the
    /// manual max/sum of the two rotated convolutions.
    #[test]
    fn compass_matches_manual_combination() {
        let sobel = Kernel {
            data: vec![
                vec![1.0, 2.0, 1.0],
                vec![0.0, 0.0, 0.0],
                vec![-1.0, -2.0, -1.0],
            ],
            scale: 1.0,
        };
        let sobel45 = Kernel {
            data: rot45_kernel(&sobel.data, Angle45::D45),
            scale: 1.0,
        };
        let im = noise_gray(12, 12, 11);

        let one = im.compass(&sobel, 1, Angle45::D45, Combine::Max, Precision::Float);
        let conv = im.conv(&sobel, Precision::Float);
        for y in 0..im.height() {
            for x in 0..im.width() {
                assert_eq!(one.getpoint(x, y)[0], conv.getpoint(x, y)[0].abs());
            }
        }

        let two_max = im.compass(&sobel, 2, Angle45::D45, Combine::Max, Precision::Float);
        let two_sum = im.compass(&sobel, 2, Angle45::D45, Combine::Sum, Precision::Float);
        let conv45 = im.conv(&sobel45, Precision::Float);
        for y in 0..im.height() {
            for x in 0..im.width() {
                let a = conv.getpoint(x, y)[0].abs();
                let b = conv45.getpoint(x, y)[0].abs();
                assert_eq!(two_max.getpoint(x, y)[0], a.max(b));
                let sum = two_sum.getpoint(x, y)[0];
                assert!((sum - (a + b)).abs() < 1e-3);
            }
        }
    }

    /// compass at integer precision keeps the unsigned format for Max and
    /// widens for Sum.
    #[test]
    fn compass_integer_formats() {
        let sharp = ported_masks().remove(0);
        let im = noise_rgb(10, 10, 13);
        let max = im.compass(&sharp, 3, Angle45::D45, Combine::Max, Precision::Integer);
        assert_eq!(max.format(), PixelFormat::Rgb8);
        let sum = im.compass(&sharp, 3, Angle45::D45, Combine::Sum, Precision::Integer);
        assert_eq!(sum.format(), PixelFormat::Rgb16);
        assert_eq!(max.width(), im.width());
        assert_eq!(sum.height(), im.height());
    }

    /// gaussblur is exactly gaussmat + convsep at both precisions, and a
    /// sigma below 0.2 returns a byte-identical copy.
    #[test]
    fn gaussblur_is_separable_gaussian() {
        let im = noise_rgb(14, 10, 15);
        for precision in [Precision::Integer, Precision::Float] {
            let mask = Kernel::gaussmat(1.4, 0.2, true, precision);
            let direct = im.convsep(&mask, precision);
            let blurred = im.gaussblur(1.4, 0.2, precision);
            assert_eq!(direct.data(), blurred.data());
        }
        let copy = im.gaussblur(0.1, 0.2, Precision::Integer);
        assert_eq!(copy.data(), im.data());
        assert_eq!(copy.format(), im.format());
    }

    /// #575: the `sigma < 0.2` copy goes through
    /// [`Raster::try_clone`], so it is fallible like the rest of the
    /// operation, and it still carries the metadata a plain `.clone()`
    /// carried.
    ///
    /// The copy is an image-sized allocation and it used to be a bare
    /// `self.clone()`, which reaches `handle_alloc_error` and ends the
    /// process rather than returning. It was the only such allocation left
    /// on `try_gaussblur`, so it was the whole of what kept the operation
    /// off the abort-free list. Reconstructing through `Raster::new`
    /// instead would have compiled and silently dropped the header and the
    /// attached fields, which is what this pins.
    #[test]
    fn gaussblur_short_circuit_copy_keeps_metadata() {
        let mut im = noise_rgb(6, 4, 21);
        im.set_field("hello", MetadataValue::Int(7));
        let im = im.copy().xres(42.0).build();

        let copy = im.try_gaussblur(0.1, 0.2, Precision::Integer).unwrap();
        assert_eq!(copy.data(), im.data());
        assert_eq!(copy.format(), im.format());
        assert!(
            (copy.xres() - 42.0).abs() < 1e-9,
            "xres must survive the short-circuit copy, got {}",
            copy.xres()
        );
        assert_eq!(copy.interpretation(), im.interpretation());
        assert_eq!(copy.get_field("hello"), Some(MetadataValue::Int(7)));
    }

    /// fastcor hand values: a 1x1 template subtracts and squares every
    /// pixel; unsigned inputs use the integer path.
    #[test]
    fn fastcor_hand_values() {
        let im = Raster::new(2, 1, PixelFormat::Gray8, vec![0, 100]).unwrap();
        let t = Raster::new(1, 1, PixelFormat::Gray8, vec![200]).unwrap();
        let cor = im.fastcor(&t);
        assert!(cor.format().is_float());
        assert_eq!(cor.getpoint(0, 0), vec![40000.0]);
        assert_eq!(cor.getpoint(1, 0), vec![10000.0]);
    }

    /// fastcor finds an extracted patch: SSD is exactly zero at the patch
    /// centre and positive elsewhere; spcor scores exactly 1.0 there.
    #[test]
    fn correlation_peaks_at_match() {
        for im in [noise_gray(40, 40, 21), noise_rgb(40, 40, 22)] {
            let patch = im.extract(12, 17, 8, 8).unwrap();

            let cor = im.fastcor(&patch);
            assert_eq!(cor.width(), im.width());
            assert_eq!(cor.height(), im.height());
            let (v, x, y) = cor.minpos();
            assert_eq!(v, 0.0);
            assert_eq!((x, y), (16, 21));

            let ncc = im.spcor(&patch);
            let (v, x, y) = ncc.maxpos();
            assert!((v - 1.0).abs() < 1e-6, "NCC at match should be 1, got {v}");
            assert_eq!((x, y), (16, 21));
        }
    }

    /// A constant template is regarded as uncorrelated: spcor returns
    /// zero everywhere (the c2 == 0 arm).
    #[test]
    fn spcor_constant_template_is_uncorrelated() {
        let im = noise_gray(10, 10, 31);
        let t = Raster::new(3, 3, PixelFormat::Gray8, vec![7; 9]).unwrap();
        let ncc = im.spcor(&t);
        assert_eq!(ncc.min(), 0.0);
        assert_eq!(ncc.max(), 0.0);
    }

    /// Correlation templates must have the image's band count.
    #[test]
    fn correlation_band_mismatch_is_typed_error() {
        let im = noise_rgb(8, 8, 33);
        let t = noise_gray(3, 3, 34);
        assert!(matches!(
            im.try_fastcor(&t),
            Err(ConvolutionError::BandCountMismatch {
                image: 3,
                template: 1
            })
        ));
        assert!(matches!(
            im.try_spcor(&t),
            Err(ConvolutionError::BandCountMismatch { .. })
        ));
    }

    /// sharpen with m1 = m2 = 0 is a byte-exact identity for 8-bit mono
    /// and colour sources, across the ported sigma sweep.
    #[test]
    fn sharpen_zero_slopes_is_identity() {
        for im in [noise_gray(24, 18, 41), noise_rgb(24, 18, 42)] {
            for sigma in [0.5, 1.0, 1.5, 2.0] {
                let out = im.sharpen(sigma, 0.0, 0.0);
                assert_eq!(out.format(), im.format());
                assert_eq!(out.data(), im.data(), "sigma {sigma}");
            }
        }
    }

    /// sharpen keeps dimensions and format and reproduces vips's own
    /// response to a hard vertical edge, byte for byte.
    ///
    /// This used to assert instead that the a/b chroma codes survive
    /// sharpening within a count, on the reasoning that the unsharp mask
    /// only touches L. That premise is false, and libvips does not hold
    /// to it either: the mask lifts the bright side of the edge to
    /// scRGB values that the per-channel `Y2v` lookup
    /// (`colour/LabQ2sRGB.c:282-353`, issue #581) quantises to
    /// [249, 249, 248] rather than a flat grey, which reads back as
    /// LabS chroma [-43, 119]. vips 8.18.4 does exactly the same, so the
    /// honest pin is the measurement, not a tolerance.
    ///
    /// Measured with `vips rawload edge.raw edge.v 20 10 3 --format
    /// uchar --interpretation srgb`, then `vips sharpen edge.v out.v
    /// --sigma 1 --m1 1 --m2 2` and `vips rawsave`. All ten rows of the
    /// result are identical, so one row pins the whole image.
    #[test]
    fn sharpen_sharpens_an_edge_like_vips() {
        // A hard vertical edge: flat halves at 40 and 220.
        let mut data = vec![0u8; 20 * 10 * 3];
        for y in 0..10 {
            for x in 0..20 {
                let v = if x < 10 { 40 } else { 220 };
                let o = (y * 20 + x) * 3;
                data[o] = v;
                data[o + 1] = v;
                data[o + 2] = v;
            }
        }
        let im = Raster::new(20, 10, PixelFormat::Rgb8, data).unwrap();
        let sharp = im.sharpen(1.0, 1.0, 2.0);
        assert_eq!(sharp.width(), im.width());
        assert_eq!(sharp.height(), im.height());
        assert_eq!(sharp.format(), im.format());
        assert_ne!(sharp.data(), im.data(), "sharpening an edge must change it");

        // vips's row, repeated down the image.
        let mut row = Vec::with_capacity(20 * 3);
        for x in 0..20 {
            let px: [u8; 3] = match x {
                8 => [26, 26, 26],
                9 => [0, 0, 0],
                10 => [249, 249, 248],
                11 => [239, 239, 239],
                x if x < 10 => [40, 40, 40],
                _ => [220, 220, 220],
            };
            row.extend_from_slice(&px);
        }
        let want: Vec<u8> = row.repeat(10);
        assert_eq!(sharp.data(), &want[..], "sharpen must match vips 8.18.4");

        // The chroma break at the edge pixel is real and vips shares it:
        // `vips colourspace out.v labs` reads [32079, -43, 119] there.
        let labs = sharp.colourspace(Interpretation::Labs);
        let px = labs.f32_samples().unwrap();
        for (c, want) in [32079.0f32, -43.0, 119.0].into_iter().enumerate() {
            assert!(
                (px[10 * 3 + c] - want).abs() < 1e-6,
                "labs band {c} at the edge pixel: vips says {want}, got {}",
                px[10 * 3 + c]
            );
        }
    }

    /// sharpen on an image with no LabS route is a typed colour error.
    #[test]
    fn sharpen_unsupported_source_is_typed_error() {
        let two = PixelFormat::with_channels(2, 1).unwrap();
        let im = Raster::zeroed(4, 4, two).unwrap();
        assert!(matches!(
            im.try_sharpen(1.0, 0.0, 0.0),
            Err(ConvolutionError::Colour(_))
        ));
    }

    /// Kernel shape errors are typed: empty, ragged, non-separable for
    /// convsep, and non-odd-square for compass. The `times` bound has its
    /// own test below.
    #[test]
    fn kernel_shape_errors() {
        let im = noise_gray(6, 6, 51);
        let empty = Kernel {
            data: vec![],
            scale: 1.0,
        };
        assert!(matches!(
            im.try_conv(&empty, Precision::Float),
            Err(ConvolutionError::EmptyKernel)
        ));
        let ragged = Kernel {
            data: vec![vec![1.0, 2.0], vec![3.0]],
            scale: 1.0,
        };
        assert!(matches!(
            im.try_conv(&ragged, Precision::Float),
            Err(ConvolutionError::RaggedKernel {
                row: 1,
                got: 1,
                expected: 2
            })
        ));
        let square = Kernel {
            data: vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            scale: 4.0,
        };
        assert!(matches!(
            im.try_convsep(&square, Precision::Float),
            Err(ConvolutionError::NotSeparable {
                width: 2,
                height: 2
            })
        ));
        assert!(matches!(
            im.try_compass(&square, 1, Angle45::D45, Combine::Max, Precision::Float),
            Err(ConvolutionError::NotOddSquareKernel { .. })
        ));
    }

    /// The libvips bound on `compass`'s `times`, the
    /// `VIPS_ARG_INT(class, "times", 101, ..., 1, 1000, 2)` range
    /// `convolution/compass.c:162-167` declares at v8.18.4. GObject
    /// refuses both ends before the operation is built, measured on a 3x3
    /// ones mask over a 4x4 black image: `vips compass a.v o.v m.mat
    /// --times 1` and `--times 1000` run, while `--times 0`,
    /// `--times 1001` and `--times 100000` each draw
    /// `value "N" of type 'gint' is invalid or out of range for property
    /// 'times' of type 'gint'` out of GObject and fall back to the
    /// property's default of 2, so the number asked for never reaches a
    /// convolution.
    ///
    /// libviprs used to check the low end only, so the high end was
    /// unbounded: `u32::MAX` reserved a result vector of 4.29 billion
    /// rasters, roughly 400 GB of address space, and then started that
    /// many whole-image convolutions.
    #[test]
    fn compass_times_outside_the_vips_property_range_is_rejected() {
        let im = noise_gray(4, 4, 51);
        let odd = Kernel {
            data: vec![vec![1.0]],
            scale: 1.0,
        };
        for times in [0u32, 1001, 100_000, u32::MAX] {
            assert!(
                matches!(
                    im.try_compass(&odd, times, Angle45::D45, Combine::Max, Precision::Integer),
                    Err(ConvolutionError::TimesOutOfRange {
                        times: got,
                        min: 1,
                        max: 1000
                    }) if got == times
                ),
                "times = {times} is outside 1..=1000 and must be refused"
            );
        }
        // Both ends of the range are accepted. A 1x1 identity mask is its
        // own rot45, so every round answers the input and the `Max`
        // combine hands it straight back however many rounds run.
        for times in [1u32, 1000] {
            let out = im
                .try_compass(&odd, times, Angle45::D45, Combine::Max, Precision::Integer)
                .unwrap_or_else(|e| panic!("times = {times} is inside 1..=1000: {e}"));
            assert_eq!(out.data(), im.data(), "times = {times} should be accepted");
        }
    }

    /// gaussmat/logmat argument validation matches the libvips bounds.
    #[test]
    fn mask_parameter_errors() {
        assert!(matches!(
            Kernel::try_gaussmat(0.0, 0.1, false, Precision::Integer),
            Err(ConvolutionError::InvalidMaskParameter { param: "sigma", .. })
        ));
        assert!(matches!(
            Kernel::try_gaussmat(1.0, f64::NAN, false, Precision::Integer),
            Err(ConvolutionError::InvalidMaskParameter {
                param: "min_ampl",
                ..
            })
        ));
        assert!(matches!(
            Kernel::try_logmat(-1.0, 0.1, false, Precision::Integer),
            Err(ConvolutionError::InvalidMaskParameter { param: "sigma", .. })
        ));
    }

    /// conv at integer precision on a float raster keeps the float path:
    /// integer mask, real division, no clipping.
    #[test]
    fn conv_integer_on_float_raster() {
        let fmt = PixelFormat::with_channels(1, 4).unwrap();
        let im = Raster::from_f32_samples(2, 1, fmt, &[-10.0, 350.5]).unwrap();
        let id = Kernel {
            data: vec![vec![1.0]],
            scale: 1.0,
        };
        let out = im.conv(&id, Precision::Integer);
        assert!(out.format().is_float());
        assert_eq!(out.getpoint(0, 0), vec![-10.0]);
        assert_eq!(out.getpoint(1, 0), vec![350.5]);
    }

    /// conv at integer precision on 16-bit rasters clips into the 16-bit
    /// range rather than the 8-bit one.
    #[test]
    fn conv_integer_16bit_clip() {
        let hi = 60000u16.to_ne_bytes();
        let im = Raster::new(1, 1, PixelFormat::Gray16, vec![hi[0], hi[1]]).unwrap();
        let double = Kernel {
            data: vec![vec![2.0]],
            scale: 1.0,
        };
        let out = im.conv(&double, Precision::Integer);
        assert_eq!(out.format(), PixelFormat::Gray16);
        assert_eq!(out.getpoint(0, 0), vec![65535.0]);
    }

    // -----------------------------------------------------------------
    // Edge detectors: sobel / scharr / prewitt
    // -----------------------------------------------------------------

    /// The `oracle-captures/convolution` `impulse_mono.v` fixture: a
    /// 21x21 uchar black canvas carrying a single 255 impulse at
    /// (10, 10). Synthetic and lossless, so the recorded vips output is
    /// exactly pinnable.
    fn impulse_mono() -> Raster {
        let mut data = vec![0u8; 21 * 21];
        data[10 * 21 + 10] = 255;
        Raster::new(21, 21, PixelFormat::Gray8, data).unwrap()
    }

    /// A single-band uchar raster built from a per-pixel closure.
    fn gray_from(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Raster {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                data.push(f(x, y));
            }
        }
        Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
    }

    /// A single-band 16-bit raster built from a per-pixel closure.
    fn gray16_from(w: u32, h: u32, f: impl Fn(u32, u32) -> u16) -> Raster {
        let mut data = Vec::with_capacity((w * h * 2) as usize);
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&f(x, y).to_ne_bytes());
            }
        }
        Raster::new(w, h, PixelFormat::Gray16, data).unwrap()
    }

    /// A single-band float raster built from a per-pixel closure.
    fn float_from(w: u32, h: u32, f: impl Fn(u32, u32) -> f32) -> Raster {
        let mut samples = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                samples.push(f(x, y));
            }
        }
        Raster::from_f32_samples(w, h, float_format(1), &samples).unwrap()
    }

    /// One named edge detector: the method name for assertion messages
    /// and the method itself.
    type EdgeOp = (&'static str, fn(&Raster) -> Raster);

    /// The three detectors as `(name, method)` pairs, so one captured
    /// table can be replayed against all of them.
    fn edge_ops() -> [EdgeOp; 3] {
        [
            ("sobel", Raster::sobel),
            ("scharr", Raster::scharr),
            ("prewitt", Raster::prewitt),
        ]
    }

    /// Band 0 of a uchar raster at `(x, y)`.
    fn u8_at(im: &Raster, x: u32, y: u32) -> u8 {
        let channels = im.format().channels() as u32;
        im.data()[((y * im.width() + x) * channels) as usize]
    }

    /// `vips sobel` / `vips scharr` / `vips prewitt` on the 21x21
    /// impulse, replayed point for point. The sobel numbers are the 27
    /// probes of the `sobel_impulse` record in
    /// `oracle-captures/convolution/oracle.json`; the scharr and prewitt
    /// grids were captured the same way from vips 8.18.4.
    ///
    /// The three responses differ only in which neighbours read 254
    /// rather than 255, and that is the double-saturation signature the
    /// uchar arm has to reproduce: the inner integer conv clips the
    /// recovered gradient into `-128..=127` around the 128 offset, so
    /// `2 * (p - 128)` reaches -256 but only +254, and the abs-sum then
    /// clips a second time at 255. An implementation that saturated only
    /// once would write 255 everywhere in the ring.
    ///
    /// The trailing number is the sum over the whole 21x21 output, which
    /// pins `avg` from the same records (2038 / 441 = 4.621315,
    /// 2036 / 441 = 4.616780) and proves everything outside the probe
    /// block is zero.
    #[test]
    fn edge_detectors_match_the_vips_impulse_response() {
        // Rows are y = 8..=12, columns x = 8..=12: the oracle probe block.
        let expected: [[[u8; 5]; 5]; 3] = [
            [
                [0, 0, 0, 0, 0],
                [0, 255, 255, 255, 0],
                [0, 254, 0, 255, 0],
                [0, 255, 254, 255, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 255, 254, 255, 0],
                [0, 254, 0, 255, 0],
                [0, 255, 255, 255, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 255, 254, 255, 0],
                [0, 254, 0, 254, 0],
                [0, 255, 254, 255, 0],
                [0, 0, 0, 0, 0],
            ],
        ];
        let totals = [2038u32, 2038, 2036];

        let im = impulse_mono();
        for (((name, op), block), total) in edge_ops().into_iter().zip(expected).zip(totals) {
            let out = op(&im);
            assert_eq!(out.format(), PixelFormat::Gray8, "{name} output format");
            assert_eq!((out.width(), out.height()), (21, 21), "{name} output size");
            for (row, wanted) in block.iter().enumerate() {
                for (col, &want) in wanted.iter().enumerate() {
                    let (x, y) = (8 + col as u32, 8 + row as u32);
                    assert_eq!(u8_at(&out, x, y), want, "{name} at ({x},{y})");
                }
            }
            let sum: u32 = out.data().iter().map(|&b| u32::from(b)).sum();
            assert_eq!(sum, total, "{name} whole-image sum");
        }
    }

    /// The measured vertical-step row of the vips 8.18.4 table: a 7x7
    /// uchar image, background 10 stepping to 20 at x >= 4. A pure
    /// vertical step is a pure Gx, so the answer is `|Gx|` alone:
    /// `10 * (1 + 2 + 1)` for sobel, `10 * (3 + 10 + 3)` for scharr,
    /// `10 * (1 + 1 + 1)` for prewitt, on the two columns straddling the
    /// step and zero everywhere else.
    #[test]
    fn edge_detectors_match_the_measured_vertical_step() {
        let im = gray_from(7, 7, |x, _| if x >= 4 { 20 } else { 10 });
        for ((name, op), want) in edge_ops().into_iter().zip([40u8, 160, 30]) {
            let out = op(&im);
            assert_eq!(out.format(), PixelFormat::Gray8, "{name} output format");
            for y in 0..7 {
                for x in 0..7 {
                    let expect = if x == 3 || x == 4 { want } else { 0 };
                    assert_eq!(u8_at(&out, x, y), expect, "{name} at ({x},{y})");
                }
            }
        }
    }

    /// The uchar arm combines `|Gx| + |Gy|` (`edge.c:97-103`) and every
    /// other format combines `sqrt(Gx^2 + Gy^2)` (`edge.c:158-182`), so
    /// the same picture reads differently on the two arms. The fixture is
    /// a 7x7 corner, background 10 with a 20 quadrant at x >= 4 && y >= 4,
    /// where Gx and Gy are equal by construction and the two rules are
    /// furthest apart: `2 * g` against `sqrt(2) * g`.
    ///
    /// scharr is the sharpest witness. Its corner gradients are 130 each,
    /// so the abs-sum is 260 and saturates to 255 while the magnitude is
    /// 183.847, and a magnitude-based uchar arm could not reach 255 here.
    ///
    /// The uchar sobel expectation is 58, not the 60 the vips binary
    /// prints by default; see
    /// `edge_uchar_negative_gradient_follows_the_scalar_convi_rounding`.
    #[test]
    fn edge_uchar_combines_the_abs_sum_and_float_the_magnitude() {
        let corner = |x: u32, y: u32| x >= 4 && y >= 4;
        let uchar = gray_from(7, 7, |x, y| if corner(x, y) { 20 } else { 10 });
        let float = float_from(7, 7, |x, y| if corner(x, y) { 20.0 } else { 10.0 });

        for ((name, op), (want_uchar, want_float)) in
            edge_ops()
                .into_iter()
                .zip([(58u8, 42u8), (255, 183), (40, 28)])
        {
            assert_eq!(u8_at(&op(&uchar), 4, 4), want_uchar, "{name} uchar corner");
            assert_eq!(u8_at(&op(&float), 4, 4), want_float, "{name} float corner");
        }
    }

    /// The float arm ends in `vips_cast_uchar`, which **truncates**
    /// (`conversion/cast.c:568`, "Floats are truncated (not rounded)").
    /// [`Raster::cast`] rounds, so the edge detectors must not use it.
    ///
    /// Two witnesses. The scharr corner magnitude is `sqrt(2) * 130 =
    /// 183.847`: truncation gives 183, rounding 184, and vips 8.18.4
    /// prints 183. The 5x5 float fixture below drives a prewitt response
    /// whose magnitude lands just under an integer, captured whole from
    /// the binary.
    ///
    /// That second fixture also pins the **`f32`** intermediates, which
    /// is a separate rule from truncation and easy to mistake for it. The
    /// magnitude there is ~148.99999 under both an `f32` and an `f64`
    /// square sum; what reaches 149 is storing the root as `f32`
    /// (`arithmetic/math2.c:147-162`). An implementation that truncates
    /// correctly but computes the whole chain in `f64` answers 148 and
    /// fails here.
    #[test]
    fn edge_float_path_truncates_the_cast_to_uchar() {
        let corner = float_from(7, 7, |x, y| if x >= 4 && y >= 4 { 20.0 } else { 10.0 });
        assert_eq!(u8_at(&corner.scharr(), 4, 4), 183);

        let mut samples = vec![0.0f32; 25];
        samples[2 * 5 + 3] = 148.98773;
        samples[3 * 5 + 2] = 1.91181;
        let im = Raster::from_f32_samples(5, 5, float_format(1), &samples).unwrap();
        #[rustfmt::skip]
        let want: [u8; 25] = [
            0, 0,   0,   0,   0,
            0, 0,   210, 148, 210,
            0, 2,   149, 2,   148,
            0, 1,   210, 149, 210,
            0, 2,   1,   2,   0,
        ];
        assert_eq!(
            im.prewitt().data(),
            &want[..],
            "vips prewitt on the tie fixture: this arm needs f32 \
             intermediates, not f64. Reading 148 where 149 is expected means \
             the chain has been promoted to f64 - the sqrt result is stored \
             back as f32 (math2.c:147-162), and that is what lifts ~148.99999 \
             to exactly 149.0 before the truncating cast"
        );
    }

    /// A mask larger than the image it convolves still walks the whole
    /// window, with every tap clamped onto the one or two rows and columns
    /// there are.
    ///
    /// This is the case the traversal's index tables have to get right at
    /// both ends. A tap far enough left of a narrow image has an empty
    /// unclamped span, so the fast interior path covers nothing and every
    /// sample comes off the replicated border; a tap far enough right has
    /// the same property from the other side. Both are perfectly legal
    /// input: `vips_embed` extends by `M->Xsize - 1` regardless of how
    /// wide the image is.
    ///
    /// The reference is the straightforward per-window sum with an
    /// explicit clamp, so it agrees with the engine only if the tables and
    /// the clamp agree.
    #[test]
    fn a_mask_wider_than_the_image_clamps_every_tap() {
        let kernel = Kernel {
            data: (0..7)
                .map(|j| (0..9).map(|i| f64::from(i * 3 + j) - 12.0).collect())
                .collect(),
            scale: 5.0,
        };
        for im in [
            noise_gray(1, 1, 11),
            noise_gray(1, 6, 12),
            noise_gray(6, 1, 13),
            noise_rgb(2, 3, 14),
        ] {
            let out = im.conv(&kernel, Precision::Float);
            assert_eq!(out.width(), im.width());
            assert_eq!(out.height(), im.height());
            for y in 0..im.height() {
                for x in 0..im.width() {
                    let want = ref_conv(&im, &kernel, i64::from(x), i64::from(y));
                    let got = out.getpoint(x, y);
                    for (c, (&g, &w)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (g - w).abs() < 1e-3,
                            "{}x{} at ({x},{y}) band {c}: got {g}, expected {w}",
                            im.width(),
                            im.height()
                        );
                    }
                }
            }
        }
    }

    /// A structural zero in the mask must not read the sample under it.
    ///
    /// libvips squeezes zero coefficients out before it convolves
    /// (`convf.c:314-321`, `convi.c:1189-1197`), so a zero tap sitting
    /// over an infinity contributes nothing. Multiplying anyway gives
    /// `0.0 * inf = NaN`, which poisons the whole response, survives the
    /// square and the root, and then clips to 0 -- an inverted answer,
    /// delivered silently (issue #574).
    ///
    /// The fixture is a 5x5 float image that is all zero except for a
    /// single `f32::INFINITY` at its centre. All three masks have
    /// structural zeros, and all three read the same ring. Captured from
    /// vips 8.18.4:
    ///
    /// ```text
    /// vips rawload inf5.raw inf5.v 5 5 1 --format float
    /// vips sobel inf5.v s.v && vips getpoint s.v 2 1   -> 255
    /// ```
    ///
    /// Before the fix the four ring cells at indices 7, 11, 13 and 17
    /// read 0 where vips reads 255.
    #[test]
    fn a_zero_mask_tap_does_not_poison_a_non_finite_sample() {
        let im = float_from(
            5,
            5,
            |x, y| {
                if (x, y) == (2, 2) { f32::INFINITY } else { 0.0 }
            },
        );
        #[rustfmt::skip]
        let want: [u8; 25] = [
            0,   0,   0,   0,   0,
            0, 255, 255, 255,   0,
            0, 255,   0, 255,   0,
            0, 255, 255, 255,   0,
            0,   0,   0,   0,   0,
        ];
        for (name, op) in edge_ops() {
            assert_eq!(
                op(&im).data(),
                &want[..],
                "{name} over an infinity: a zero tap must be squeezed out, \
                 not multiplied. Reading 0 at indices 7, 11, 13 and 17 means \
                 `0.0 * inf` produced a NaN and drove the magnitude to zero"
            );
        }
    }

    /// An all-zero mask keeps exactly one tap, at mask index 0.
    ///
    /// Both libvips cores force `nnz` back up to 1 when every coefficient
    /// squeezed out (`convf.c:325-333`, `convi.c:1199-1206`), so the
    /// surviving tap still multiplies a sample and an all-zero mask still
    /// answers `NaN` over a non-finite one -- but only where the window's
    /// top-left corner is the non-finite sample, not everywhere in its
    /// neighbourhood. Captured from vips 8.18.4 on the same 5x5 infinity
    /// fixture, with a 3x3 all-zero mask at scale 1:
    ///
    /// ```text
    /// vips conv inf5.v z.v zero3.mat --precision float
    /// vips getpoint z.v 3 3   -> nan     (every other sample: 0)
    /// ```
    ///
    /// Both precisions give the same answer, since the integer arm takes
    /// the double inner loop on a float input.
    #[test]
    fn an_all_zero_mask_keeps_the_single_tap_libvips_keeps() {
        let im = float_from(
            5,
            5,
            |x, y| {
                if (x, y) == (2, 2) { f32::INFINITY } else { 0.0 }
            },
        );
        let zeros = Kernel {
            data: vec![vec![0.0; 3]; 3],
            scale: 1.0,
        };
        for precision in [Precision::Float, Precision::Integer] {
            let out = im.conv(&zeros, precision);
            let got = out.f32_samples().expect("conv of a float input is float");
            for (i, v) in got.iter().enumerate() {
                let (x, y) = (i % 5, i / 5);
                if (x, y) == (3, 3) {
                    assert!(
                        v.is_nan(),
                        "{precision:?}: the one surviving tap reads (2,2), so \
                         (3,3) has to be NaN; got {v}"
                    );
                } else {
                    assert_eq!(*v, 0.0, "{precision:?}: sample {i} at ({x},{y})");
                }
            }
        }
    }

    /// The whole float arm, replayed against vips 8.18.4 output captured
    /// on two fixtures: a 7x7 float image whose samples are exact
    /// quarters spanning negatives, and a 5x5 16-bit image (any format
    /// other than uchar takes the float arm, `edge.c:186-200`).
    ///
    /// The float arm is bit-stable: `VIPS_NOVECTOR=1` reproduces both
    /// captures byte for byte, unlike the uchar arm.
    #[test]
    fn edge_float_arm_matches_the_vips_capture() {
        #[rustfmt::skip]
        let float_expected: [[u8; 49]; 3] = [
            [
                17, 11, 13, 11, 9, 11, 9, 21, 8, 3, 3, 3, 3, 8, 8, 8, 13, 8, 3, 3, 8, 8, 3, 3, 8,
                13, 8, 8, 8, 3, 3, 3, 3, 8, 21, 8, 3, 3, 3, 3, 3, 9, 17, 11, 13, 11, 9, 11, 20,
            ],
            [
                69, 58, 66, 58, 49, 58, 48, 95, 37, 15, 25, 25, 15, 48, 48, 37, 74, 37, 15, 25, 18,
                18, 25, 15, 37, 74, 37, 48, 48, 15, 25, 25, 15, 37, 95, 18, 25, 15, 15, 25, 25, 37,
                79, 58, 66, 58, 49, 58, 80,
            ],
            [
                13, 6, 7, 6, 4, 6, 5, 14, 6, 5, 3, 3, 5, 2, 2, 6, 6, 6, 5, 3, 10, 10, 3, 5, 6, 6,
                6, 2, 2, 5, 3, 3, 5, 6, 14, 10, 3, 5, 5, 3, 3, 7, 10, 6, 7, 6, 4, 6, 15,
            ],
        ];
        #[rustfmt::skip]
        let u16_expected: [[u8; 25]; 3] = [
            [69, 47, 55, 47, 68, 87, 34, 13, 13, 32, 32, 34, 54, 34, 32, 32, 13, 13, 34, 87, 68,
             47, 55, 47, 69],
            [255, 235, 255, 235, 255, 255, 149, 63, 102, 72, 192, 149, 255, 149, 192, 72, 102, 63,
             149, 255, 255, 235, 255, 235, 255],
            [52, 24, 29, 24, 40, 57, 26, 22, 15, 40, 10, 26, 26, 26, 10, 40, 15, 22, 26, 57, 40,
             24, 29, 24, 52],
        ];

        let floats = float_from(7, 7, |x, y| ((x * 3 + y * 5) % 11) as f32 * 0.75 - 4.0);
        let shorts = gray16_from(5, 5, |x, y| (((x * 3 + y * 5) % 11) * 3 + 300) as u16);
        for (((name, op), want_float), want_u16) in
            edge_ops().into_iter().zip(float_expected).zip(u16_expected)
        {
            let out = op(&floats);
            assert_eq!(
                out.format(),
                PixelFormat::Gray8,
                "{name} float input format"
            );
            assert_eq!(out.data(), &want_float[..], "{name} on the float fixture");

            let out = op(&shorts);
            assert_eq!(
                out.format(),
                PixelFormat::Gray8,
                "{name} 16-bit input format"
            );
            assert_eq!(out.data(), &want_u16[..], "{name} on the 16-bit fixture");
        }
    }

    /// Output is always uchar (`edge.c` ends the non-uchar arm in
    /// `vips_cast_uchar` and the uchar arm never leaves 8 bits), the band
    /// count and the dimensions are preserved, and the 16-bit and float
    /// carriers all narrow to their 8-bit sibling.
    #[test]
    fn edge_output_is_always_uchar_with_the_input_bands() {
        let two8 = PixelFormat::with_channels(2, 1).unwrap();
        let five16 = PixelFormat::with_channels(5, 2).unwrap();
        let five8 = PixelFormat::with_channels(5, 1).unwrap();
        let cases = [
            (PixelFormat::Gray8, PixelFormat::Gray8),
            (PixelFormat::Gray16, PixelFormat::Gray8),
            (PixelFormat::Rgb8, PixelFormat::Rgb8),
            (PixelFormat::Rgb16, PixelFormat::Rgb8),
            (PixelFormat::Rgba8, PixelFormat::Rgba8),
            (PixelFormat::Rgba16, PixelFormat::Rgba8),
            (PixelFormat::RgbaF32, PixelFormat::Rgba8),
            (two8, two8),
            (five16, five8),
        ];
        for (src, want) in cases {
            let im = Raster::zeroed(4, 3, src).unwrap();
            for (name, op) in edge_ops() {
                let out = op(&im);
                assert_eq!(out.format(), want, "{name} of {src:?}");
                assert_eq!(
                    (out.width(), out.height()),
                    (4, 3),
                    "{name} of {src:?} size"
                );
            }
        }
    }

    /// Every band is convolved and combined on its own, exactly as
    /// `vips conv` is per-band: a 7x7 RGB fixture with a 10 -> 20 step at
    /// x >= 4 in band 0, a flat 77 in band 1, and a 30 -> 60 step at
    /// x >= 2 in band 2 answers 40 / 0 / 120 for sobel on the columns
    /// straddling each step, and a flat band contributes nothing anywhere.
    /// Captured from vips 8.18.4.
    #[test]
    fn edge_treats_every_band_independently() {
        let mut data = Vec::with_capacity(7 * 7 * 3);
        for _ in 0..7 {
            for x in 0..7u32 {
                data.push(if x >= 4 { 20 } else { 10 });
                data.push(77);
                data.push(if x >= 2 { 60 } else { 30 });
            }
        }
        let im = Raster::new(7, 7, PixelFormat::Rgb8, data).unwrap();

        for ((name, op), want) in
            edge_ops()
                .into_iter()
                .zip([[40u8, 0, 120], [160, 0, 254], [30, 0, 90]])
        {
            let out = op(&im);
            assert_eq!(out.format(), PixelFormat::Rgb8, "{name} output format");
            let row = &out.data()[3 * 7 * 3..4 * 7 * 3];
            let band0: Vec<u8> = row.iter().step_by(3).copied().collect();
            let band1: Vec<u8> = row.iter().skip(1).step_by(3).copied().collect();
            let band2: Vec<u8> = row.iter().skip(2).step_by(3).copied().collect();
            assert_eq!(
                band0,
                vec![0, 0, 0, want[0], want[0], 0, 0],
                "{name} band 0"
            );
            assert_eq!(band1, vec![0; 7], "{name} band 1 is flat");
            assert_eq!(
                band2,
                vec![0, want[2], want[2], 0, 0, 0, 0],
                "{name} band 2"
            );
        }
    }

    /// A negative uchar gradient reads low against an HWY-enabled
    /// libvips, and this pins the gap so it cannot drift unnoticed.
    /// **The bound is 4, not 2** (issue #558).
    ///
    /// `vips_convi_gen` divides with C's truncating `/`
    /// (`convolution/convi.c:710`, `((sum + rounding) / scale) + offset`),
    /// which for a negative sum rounds towards zero. Any libvips built
    /// with HWY takes a vector path for uchar integer convolutions that
    /// finishes with an arithmetic shift instead, which floors, and
    /// `vips_convi_intize` only requires the two to agree within 2
    /// (`convi.c:1107-1112`). That is a property of the library, not of
    /// the `vips` command, so pyvips and every other binding sees it too.
    /// libviprs ports the scalar C path, so an inner conv whose window
    /// sum is negative and even reads one lower here, and the uchar arm's
    /// `2 * (p - 128)` recovery doubles that to two per gradient.
    ///
    /// Two fixtures, both measured against vips 8.18.4.
    ///
    /// One gradient affected, gap 2: a horizontal 10 -> 20 step at
    /// y >= 4 gives sobel's base mask a sum of -40 straddling the step.
    /// `VIPS_NOVECTOR=1 vips sobel` prints 38 here, matching libviprs;
    /// the default `vips sobel` prints 40. The vertical step of the same
    /// size stays positive and both agree on 40, the control in the same
    /// assertion.
    ///
    /// **Both gradients affected, gap 4**, which is the bound a caller
    /// actually has to allow for. On the 8x3 fixture below, `prewitt` at
    /// (4,0) puts both inner convolutions on the negative-and-even case:
    /// libviprs and `VIPS_NOVECTOR=1 vips` read 123 and 80 and answer
    /// 106, while the same binary with the vector path live reads 122 and
    /// 79 and answers 110.
    #[test]
    fn edge_uchar_negative_gradient_follows_the_scalar_convi_rounding() {
        let horizontal = gray_from(7, 7, |_, y| if y >= 4 { 20 } else { 10 });
        assert_eq!(u8_at(&horizontal.sobel(), 3, 3), 38);
        assert_eq!(u8_at(&horizontal.sobel(), 3, 4), 38);

        let vertical = gray_from(7, 7, |x, _| if x >= 4 { 20 } else { 10 });
        assert_eq!(u8_at(&vertical.sobel(), 3, 3), 40);
        assert_eq!(u8_at(&vertical.sobel(), 4, 3), 40);

        #[rustfmt::skip]
        let pixels: Vec<u8> = vec![
            79, 46, 165, 221, 20, 220, 238, 241,
            190, 170, 207, 147, 79, 137, 17, 42,
            243, 112, 225, 97, 123, 226, 86, 173,
        ];
        let both = Raster::new(8, 3, PixelFormat::Gray8, pixels).unwrap();
        assert_eq!(u8_at(&both.prewitt(), 4, 0), 106, "gap-4 fixture");
    }

    /// A fused traversal answers with each mask's own response in the
    /// order it was handed them, matches what the same masks give one at a
    /// time, and serves a mask carrying an offset and no scale. All three
    /// are contract rather than accident. The edge detectors combine
    /// symmetrically so they cannot tell the order apart, but `vips_canny`
    /// takes `atan2` off the pair, where a swap silently rotates every
    /// angle by 90 degrees.
    ///
    /// The 3x3 mask here is deliberately asymmetric under rotation, so the
    /// two responses really are different images and the ordering
    /// assertion has teeth. The second half is canny's own call shape
    /// (`canny.c:68-92`): a 2x2 `-1/+1` difference stamped
    /// `offset = 128` with no scale, which is the only shape the core has
    /// to serve that is neither 3x3 nor 1xN, and the only one where the
    /// even-sized anchor question arises. That anchor rides through
    /// [`Scan`]'s index tables now, so a 2x2 is also the case that proves
    /// a mask with no trailing tap below or right of its centre still
    /// indexes them correctly.
    #[test]
    fn conv_raster_n_answers_the_masks_in_order_and_serves_canny_s_2x2() {
        let im = noise_gray(9, 7, 557);

        let asymmetric = Kernel {
            data: vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ],
            scale: 3.0,
        };
        let dense = DenseKernel::new(&asymmetric).unwrap();
        let rot = dense.rot90();
        let [first, second] = conv_raster_n(&im, [&dense, &rot], Precision::Float).unwrap();
        assert_eq!(
            first.data(),
            conv_raster(&im, &dense, Precision::Float).unwrap().data(),
            "`.0` must be the response to the mask itself"
        );
        assert_eq!(
            second.data(),
            conv_raster(&im, &dense.rot90(), Precision::Float)
                .unwrap()
                .data(),
            "`.1` must be the response to the rotated mask"
        );
        assert_ne!(
            first.data(),
            second.data(),
            "the mask has to be asymmetric or the ordering claim is untestable"
        );

        let canny = DenseKernel::new(&Kernel {
            data: vec![vec![-1.0, 1.0], vec![-1.0, 1.0]],
            scale: 1.0,
        })
        .unwrap()
        .with_offset(EDGE_UCHAR_OFFSET);
        let spun = canny.rot90();
        assert_eq!((spun.w, spun.h), (2, 2), "rot90 of a 2x2 is a 2x2");
        assert_eq!(
            spun.coeff,
            vec![-1.0, -1.0, 1.0, 1.0],
            "rot90 of [[-1,1],[-1,1]] is [[-1,-1],[1,1]]"
        );
        assert!(
            (spun.offset - EDGE_UCHAR_OFFSET).abs() < f64::EPSILON,
            "the 2x2 rotation dropped the offset: got {}",
            spun.offset
        );
        assert!(
            (spun.scale - 1.0).abs() < f64::EPSILON,
            "the 2x2 rotation dropped the scale: got {}",
            spun.scale
        );

        let [gx, gy] = conv_raster_n(&im, [&canny, &spun], Precision::Integer).unwrap();
        assert_eq!(
            gx.data(),
            conv_raster(&im, &canny, Precision::Integer).unwrap().data(),
            "2x2 `.0` is the mask response"
        );
        assert_eq!(
            gy.data(),
            conv_raster(&im, &spun, Precision::Integer).unwrap().data(),
            "2x2 `.1` is the rot90 response"
        );

        // The offset rides along both halves: a flat image differences to
        // zero everywhere, so every sample recentres on exactly 128.
        let flat = gray_from(4, 4, |_, _| 90);
        let [fx, fy] = conv_raster_n(&flat, [&canny, &spun], Precision::Integer).unwrap();
        assert_eq!(fx.data(), &[128u8; 16][..], "flat Gx recentres on 128");
        assert_eq!(fy.data(), &[128u8; 16][..], "flat Gy recentres on 128");
    }

    /// The result inherits the source metadata, as a vips pipeline does:
    /// the interpretation survives even though the format changed, and so
    /// do the resolution and offset fields.
    ///
    /// The **attached** fields survive too, which is what `vips sobel`
    /// does and is easy to drop by accident: a jpeg carrying 186 bytes of
    /// `exif-data` and a 564-byte ICC profile comes back out of the
    /// binary carrying both, on either arm. `out.meta` alone would leave
    /// them behind.
    #[test]
    fn edge_inherits_the_source_metadata() {
        let mut im = gray16_from(4, 4, |x, y| u16::try_from(x * 900 + y * 70).unwrap())
            .copy()
            .interpretation(Interpretation::Grey16)
            .xres(42.0)
            .build();
        im.set_field("exif-data", MetadataValue::Blob(vec![7, 8, 9]));
        im.set_field("icc-profile-data", MetadataValue::Blob(vec![1, 2]));
        for (name, op) in edge_ops() {
            let out = op(&im);
            assert_eq!(
                out.interpretation(),
                Interpretation::Grey16,
                "{name} interpretation"
            );
            assert!((out.xres() - 42.0).abs() < 1e-12, "{name} xres");
            assert_eq!(
                out.get_field("exif-data"),
                Some(MetadataValue::Blob(vec![7, 8, 9])),
                "{name} dropped the EXIF blob"
            );
            assert_eq!(
                out.get_field("icc-profile-data"),
                Some(MetadataValue::Blob(vec![1, 2])),
                "{name} dropped the ICC profile"
            );
        }
    }

    /// The `try_*` and panicking forms are the same call, and the three
    /// masks really are different matrices: on the 7x7 vertical step the
    /// detectors answer 40, 160 and 30, so no two of them agree.
    #[test]
    fn edge_try_and_panicking_forms_agree() {
        let im = gray_from(7, 7, |x, _| if x >= 4 { 20 } else { 10 });
        let got: Vec<Vec<u8>> = edge_ops()
            .into_iter()
            .map(|(_, op)| op(&im).data().to_vec())
            .collect();
        assert_ne!(got[0], got[1], "sobel and scharr");
        assert_ne!(got[1], got[2], "scharr and prewitt");
        assert_ne!(got[0], got[2], "sobel and prewitt");

        assert_eq!(im.try_sobel().unwrap().data(), got[0].as_slice());
        assert_eq!(im.try_scharr().unwrap().data(), got[1].as_slice());
        assert_eq!(im.try_prewitt().unwrap().data(), got[2].as_slice());
    }

    // -----------------------------------------------------------------
    // canny (issues #511, #559, #560)
    //
    // Every expected value below comes from
    // `oracle-captures/convolution/canny/oracle.json`, captured from vips
    // 8.18.4 on **both** libvips paths. Where the two disagree the pin is
    // the `VIPS_NOVECTOR=1` arm, which is the portable C libviprs targets
    // (issue #558).
    // -----------------------------------------------------------------

    /// The LCG `oracle-captures/convolution/canny/capture.py` builds its
    /// noise fixtures with, reproduced so the digests below are of the
    /// same bytes vips measured. Deliberately not the module's own
    /// [`lcg`] helper: that is a different generator with a different
    /// stream, and the captured digests are of this one.
    fn oracle_lcg(n: usize, seed: u32) -> Vec<u8> {
        let mut state = u64::from(seed & 0x7fff_ffff);
        (0..n)
            .map(|_| {
                state = (1_103_515_245 * state + 12_345) & 0x7fff_ffff;
                ((state >> 16) & 0xff) as u8
            })
            .collect()
    }

    /// `fixtures/step9.pgm`: 9x9 uchar, columns 0-3 black and 4-8 white.
    /// A pure `Gx` edge and the simplest non-trivial case.
    fn canny_step9() -> Raster {
        gray_from(9, 9, |x, _| if x < 4 { 0 } else { 255 })
    }

    /// `fixtures/square9.pgm`: 9x9 uchar with a 4x4 white block in the
    /// top-left. Its bottom-right corner drives both gradient
    /// convolutions into their negative clip, which is the only way to
    /// reach the uchar ceiling of `G == 64`.
    fn canny_square9() -> Raster {
        gray_from(9, 9, |x, y| if x < 4 && y < 4 { 255 } else { 0 })
    }

    /// The half-step ramp behind the four `fixtures/plateau_*` images.
    /// The 128 in the middle is what gives two adjacent pixels the same
    /// `G` and the same `theta`.
    const CANNY_RAMP: [u8; 9] = [0, 0, 0, 0, 128, 255, 255, 255, 255];

    /// One of `fixtures/plateau_h`, `plateau_h_rev`, `plateau_v`,
    /// `plateau_v_rev`: the ramp laid along x (9x5) or along y (5x9),
    /// forwards or mirrored.
    fn canny_plateau(vertical: bool, reversed: bool) -> Raster {
        let pick = move |i: u32| CANNY_RAMP[if reversed { 8 - i } else { i } as usize];
        if vertical {
            gray_from(5, 9, move |_, y| pick(y))
        } else {
            gray_from(9, 5, move |x, _| pick(x))
        }
    }

    /// `fixtures/disc33.pgm`: the "white disc on a black background" the
    /// `canny.c:228` comment describes, radius 12 on 33x33.
    fn canny_disc33() -> Raster {
        gray_from(33, 33, |x, y| {
            let (dx, dy) = (x as i32 - 16, y as i32 - 16);
            if dx * dx + dy * dy <= 144 { 255 } else { 0 }
        })
    }

    /// `fixtures/border7.pgm`: a white column on the left frame edge and
    /// a white row on the bottom one, so real edges sit in the outer ring
    /// where the `Extend::Copy` embed duplicates neighbours.
    fn canny_border7() -> Raster {
        gray_from(7, 7, |x, y| if x == 0 || y == 6 { 255 } else { 0 })
    }

    /// The twenty `(gx, gy)` pairs `fixtures/octants26.pgm` is engineered
    /// to produce: all eight octants, the four axes, the four diagonals,
    /// `gx == gy == 0`, and three gradients below the LUT's 4-bit
    /// resolution.
    const CANNY_OCTANT_TARGETS: [(i32, i32); 20] = [
        (0, 0),
        (64, 0),
        (0, 64),
        (-64, 0),
        (0, -64),
        (64, 64),
        (-64, 64),
        (64, -64),
        (-64, -64),
        (96, 32),
        (32, 96),
        (-96, 32),
        (32, -96),
        (96, -32),
        (-32, 96),
        (-96, -32),
        (8, 0),
        (0, 8),
        (-8, -8),
        (120, 120),
    ];

    /// `fixtures/octants26.pgm`. For a 2x2 window `a b / c d` the two
    /// convolution sums are `sx = b + d - a - c` and `sy = c + d - a - b`,
    /// so `a = 128`, `b = 128 + (sx - sy) / 4`, `c = 128 - (sx - sy) / 4`
    /// and `d = 128 + (sx + sy) / 2` puts any wanted `(gx, gy)` at the
    /// window's bottom-right pixel, on a flat 128 background.
    fn canny_octants26() -> Raster {
        let mut data = vec![128u8; 26 * 26];
        for (n, &(sx, sy)) in CANNY_OCTANT_TARGETS.iter().enumerate() {
            let (bx, by) = (2 + (n % 5) * 5, 2 + (n / 5) * 5);
            let k = (sx - sy) / 4;
            data[by * 26 + bx + 1] = (128 + k) as u8;
            data[(by + 1) * 26 + bx] = (128 - k) as u8;
            data[(by + 1) * 26 + bx + 1] = (128 + (sx + sy) / 2) as u8;
        }
        Raster::new(26, 26, PixelFormat::Gray8, data).unwrap()
    }

    /// Where the `n`th octant probe reads: the bottom-right of its 2x2.
    fn canny_octant_probe(n: usize) -> (u32, u32) {
        ((2 + (n % 5) * 5 + 1) as u32, (2 + (n / 5) * 5 + 1) as u32)
    }

    /// `fixtures/noise64.pgm`: 64x64 uchar LCG noise. At sigma 0.01 the
    /// blur is an exact copy, so this drives the polar stage directly and
    /// reaches all 256 atan2 LUT indices.
    fn canny_noise64() -> Raster {
        Raster::new(64, 64, PixelFormat::Gray8, oracle_lcg(64 * 64, 20_260_825)).unwrap()
    }

    /// `fixtures/noise16rgb.ppm`: 16x16x3 uchar LCG noise, for the
    /// `(w, h, b)` round trip and per-band independence.
    fn canny_noise16rgb() -> Raster {
        Raster::new(16, 16, PixelFormat::Rgb8, oracle_lcg(16 * 16 * 3, 4242)).unwrap()
    }

    /// The digest `oracle.json` records as `raw_sha256`: sha256 of
    /// `vips rawsave` output, which is the samples alone with no header.
    /// Float samples are re-serialised little-endian rather than natively
    /// so the pin means the same thing wherever the tests run.
    fn oracle_raw_sha256(r: &Raster) -> String {
        let bytes: Vec<u8> = if r.format().is_float() {
            r.f32_samples()
                .expect("a float raster has f32 samples")
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect()
        } else {
            r.data().to_vec()
        };
        crate::checksum::hash_tile(&bytes, crate::checksum::ChecksumAlgo::Sha256)
    }

    /// Every row of a uchar raster, bands interleaved.
    fn u8_rows(r: &Raster) -> Vec<Vec<u8>> {
        let stride = r.width() as usize * r.format().channels();
        r.data().chunks(stride).map(<[u8]>::to_vec).collect()
    }

    /// Every row of a float raster, bands interleaved.
    fn f32_rows(r: &Raster) -> Vec<Vec<f32>> {
        let stride = r.width() as usize * r.format().channels();
        r.f32_samples()
            .expect("a float raster has f32 samples")
            .chunks(stride)
            .map(<[f32]>::to_vec)
            .collect()
    }

    /// Assert a float raster matches a measured grid, to a tolerance and
    /// with the offending cell named. The byte-exact half of these pins
    /// is the `raw_sha256` next to each call.
    fn assert_f32_grid(got: &Raster, want: &[[f32; 9]; 9], what: &str) {
        for (y, (row, wrow)) in f32_rows(got).into_iter().zip(want).enumerate() {
            for (x, (v, w)) in row.into_iter().zip(wrow).enumerate() {
                assert!(
                    (v - w).abs() < 1e-3,
                    "{what} at ({x}, {y}): read {v}, want {w}"
                );
            }
        }
    }

    /// `vips canny --sigma 1.4 --precision float`, the default call, on
    /// the 9x9 step.
    ///
    /// The answer is a **float** image, not a byte one, and that is the
    /// first thing a port gets wrong. `canny.c:81` tests the format of
    /// the *blurred* image, not of the input, and on the float arm
    /// gaussblur has already promoted the uchar step to float by then, so
    /// the uchar gradient branch never fires. Nothing here fits in a
    /// byte: the surviving column carries 47.99, and a hard 0/255 step
    /// with no blur in front of it reaches 508.5.
    #[test]
    fn canny_reproduces_vips_on_the_default_float_arm() {
        let out = canny_step9().canny(1.4, Precision::Float);
        assert_eq!(out.format(), float_format(1), "float arm output format");
        assert_eq!((out.width(), out.height()), (9, 9), "size round-trips");
        assert_eq!(
            oracle_raw_sha256(&out),
            "2f6ab0a309442a20357f1314576f8f81411e6fc3dca23533ada523a9262eaee1",
            "record default_step9_float"
        );
        let mut want = [[0.0f32; 9]; 9];
        for row in &mut want {
            row[4] = 47.992_317;
        }
        assert_f32_grid(&out, &want, "step9 float");

        // The same op on the corner fixture, where the two gradients are
        // both live and the diagonal edge is what survives.
        let square = canny_square9().canny(1.4, Precision::Float);
        assert_eq!(
            oracle_raw_sha256(&square),
            "d614673426996af46331ab84e22cc465b63089c40b12ec6c95d98decd8728c81",
            "record default_square9_float"
        );
    }

    /// `vips canny --sigma 1.4 --precision integer` on the same step, and
    /// on the corner fixture. Integer precision keeps the blur uchar, so
    /// the gradient stage takes its `offset = 128` integer arm and the
    /// whole operation stays in a byte, where `G` is bounded at 64.
    #[test]
    fn canny_reproduces_vips_on_the_uchar_integer_arm() {
        let out = canny_step9().canny(1.4, Precision::Integer);
        assert_eq!(out.format(), PixelFormat::Gray8, "integer arm stays uchar");
        assert_eq!(u8_rows(&out), vec![vec![0, 0, 0, 0, 32, 0, 0, 0, 0]; 9]);

        let square = canny_square9().canny(1.4, Precision::Integer);
        assert_eq!(
            u8_rows(&square),
            vec![
                vec![0, 0, 0, 0, 32, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 32, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 34, 0, 0, 0, 0],
                vec![0, 0, 0, 33, 36, 0, 0, 0, 0],
                vec![32, 32, 34, 37, 0, 0, 0, 0, 0],
                vec![0; 9],
                vec![0; 9],
                vec![0; 9],
                vec![0; 9],
            ],
            "record default_square9_integer"
        );
    }

    /// The two arms do not merely round differently, they have different
    /// ranges. `square9` at sigma 0.01 (the blur is an exact copy) drives
    /// both convolutions into their negative clip at (4, 4), which is the
    /// only way to reach the uchar ceiling of 64; the same fixture on the
    /// float arm answers 508.5078125 on the straight edges, eight times
    /// what a byte holds.
    ///
    /// The interpolation inside suppression is what makes 64 dangerous:
    /// `G * (32 - residual)` is `64 * 32 = 2048` there, so a port that
    /// blends in `u8` overflows and panics in debug. C promotes to `int`.
    #[test]
    fn canny_uchar_g_tops_out_at_64_where_the_float_arm_reaches_508() {
        let uchar = canny_square9().canny(0.01, Precision::Integer);
        assert_eq!(
            u8_rows(&uchar),
            vec![
                vec![0, 0, 0, 0, 32, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 32, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 32, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 32, 0, 0, 0, 0],
                vec![32, 32, 32, 32, 64, 0, 0, 0, 0],
                vec![0; 9],
                vec![0; 9],
                vec![0; 9],
                vec![0; 9],
            ],
            "record gmax_square9_uchar"
        );

        let float = canny_square9()
            .cast(float_format(1))
            .canny(0.01, Precision::Float);
        let mut want = [[0.0f32; 9]; 9];
        for row in want.iter_mut().take(4) {
            row[4] = 508.507_8;
        }
        want[4] = [
            508.507_8, 508.507_8, 508.507_8, 508.507_8, 254.503_9, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_f32_grid(&float, &want, "record gmax_square9_float");
    }

    /// The 256-entry atan2 LUT of `canny.c:200-222`, recomputed in `f64`
    /// from the C exactly as written: each nibble sign-extended to
    /// `-8..=7`, `VIPS_DEG(atan2(x, y)) + 360`, then a **truncating**
    /// `256 * theta / 360` and `& 0xFF`.
    ///
    /// `VIPS_DEG` is `(a / (2 * pi)) * 360`, not `a * (180 / pi)`: two
    /// roundings in that order. The sixty entries that land on an exact
    /// angle are exactly representable through that chain, and the
    /// closest of the other 196 sits 0.019 away from a truncation
    /// boundary, so this recomputation does not depend on the host's
    /// `atan2` being bit-identical to the one the table was built with.
    #[test]
    fn canny_atan2_lut_is_the_canny_c_table() {
        let sign_extend = |v: i32| if v & 0x8 != 0 { v - 0x10 } else { v };
        for (i, &entry) in CANNY_ATAN2_LUT.iter().enumerate() {
            let x = sign_extend(i as i32 & 0xF);
            let y = sign_extend((i as i32 >> 4) & 0xF);
            let theta = vips_deg(f64::from(x).atan2(f64::from(y))) + 360.0;
            let want = ((256.0 * theta / 360.0) as i32 & 0xFF) as u8;
            assert_eq!(entry, want, "LUT[{i}] for (x, y) = ({x}, {y})");
        }
        // The cardinal directions, spelled out: theta is measured from
        // +y with the arguments swapped, so a gradient pointing along +y
        // reads 0 and one along +x reads 64.
        assert_eq!(CANNY_ATAN2_LUT[0x01], 64, "(gx, gy) = (1, 0)");
        assert_eq!(CANNY_ATAN2_LUT[0x10], 0, "(gx, gy) = (0, 1)");
        assert_eq!(CANNY_ATAN2_LUT[0x0f], 192, "(gx, gy) = (-1, 0)");
        assert_eq!(CANNY_ATAN2_LUT[0xf0], 128, "(gx, gy) = (0, -1)");
        assert_eq!(CANNY_ATAN2_LUT[0x00], 0, "atan2(0, 0) is 0");
    }

    /// The two polar arms on the twenty engineered `(gx, gy)` pairs, the
    /// values `oracle.json -> derived_polar.octants` records.
    ///
    /// The last three rows are the interesting ones. `(8, 0)` and
    /// `(0, 8)` both read theta 0 on the uchar path, because the LUT
    /// throws away the bottom four bits of each axis and a gradient
    /// smaller than 16 collapses into bucket zero; the float path reads
    /// the correct 64 and 0. That is not a porting bug to fix, it is what
    /// the binary does.
    #[test]
    fn canny_polar_matches_the_measured_octants_on_both_arms() {
        let want: [(u8, u8, f32, f32); 20] = [
            (0, 0, 0.5, 0.0),
            (8, 64, 8.5, 64.0),
            (8, 0, 8.5, 0.0),
            (8, 192, 8.5, 192.0),
            (8, 128, 8.5, 128.0),
            (16, 32, 16.5, 32.0),
            (16, 224, 16.5, 224.0),
            (16, 96, 16.5, 96.0),
            (16, 160, 16.5, 160.0),
            (20, 50, 20.5, 50.890_7),
            (20, 13, 20.5, 13.109_297),
            (20, 205, 20.5, 205.109_3),
            (20, 114, 20.5, 114.890_7),
            (20, 77, 20.5, 77.109_3),
            (20, 242, 20.5, 242.890_7),
            (20, 178, 20.5, 178.890_7),
            (0, 0, 0.625, 64.0),
            (0, 0, 0.625, 0.0),
            (0, 160, 0.75, 160.0),
            (56, 32, 56.75, 32.0),
        ];
        for (&(gx, gy), &(ug, ut, fg, ft)) in CANNY_OCTANT_TARGETS.iter().zip(&want) {
            assert_eq!(
                canny_polar_uchar(gx, gy),
                (ug, ut),
                "uchar polar of ({gx}, {gy})"
            );
            let (g, t) = canny_polar_float(f64::from(gx), f64::from(gy));
            assert!((g - fg).abs() < 1e-4, "float G of ({gx}, {gy}): {g}");
            assert!((t - ft).abs() < 1e-3, "float theta of ({gx}, {gy}): {t}");
        }
    }

    /// The gradient stage really does put those `(gx, gy)` pairs where
    /// the fixture says, which is what makes the octant pins above a test
    /// of the whole polar path rather than of arithmetic in isolation.
    ///
    /// It also pins the 2x2 anchor. A 2x2 mask has no tap below or right
    /// of its centre, so the window for output `(x, y)` is
    /// `(x - 1, y - 1)..=(x, y)`, and getting that off by one moves every
    /// probe to the wrong pixel.
    #[test]
    fn canny_gradient_recovers_the_engineered_pairs() {
        let [gx, gy] = canny_octants26()
            .canny_gradient(0.01, Precision::Integer)
            .unwrap();
        assert_eq!(gx.format(), PixelFormat::Gray8, "integer arm keeps uchar");
        for (n, &(sx, sy)) in CANNY_OCTANT_TARGETS.iter().enumerate() {
            let (x, y) = canny_octant_probe(n);
            assert_eq!(
                (
                    i32::from(u8_at(&gx, x, y)) - 128,
                    i32::from(u8_at(&gy, x, y)) - 128
                ),
                (sx, sy),
                "probe {n} at ({x}, {y})"
            );
        }

        // And the whole operation over the same fixture, on both arms.
        assert_eq!(
            oracle_raw_sha256(&canny_octants26().canny(0.01, Precision::Integer)),
            "6f3bb853b2e2a617b99ac26c8c9463db2eaba9632c9b1b9a28b105086879f9a4",
            "record octants_uchar"
        );
        assert_eq!(
            oracle_raw_sha256(
                &canny_octants26()
                    .cast(float_format(1))
                    .canny(0.01, Precision::Float)
            ),
            "84023df0c31b34bc1e2d006d31d639dc168187284ba7f354e64ac297170a4299",
            "record octants_float"
        );
    }

    /// The orientation the `canny.c:228` comment gets wrong. It says
    /// "0 at the top, 64 on the left, 128 on the right and 192 on the
    /// right edge", naming the right twice and dropping the bottom.
    ///
    /// Measured on the disc, uchar arm: **0 at the top, 64 on the left,
    /// 128 at the bottom, 192 on the right**. Both arms call
    /// `atan2(gx, gy)` with the arguments swapped relative to the usual
    /// convention, which is what puts 0 at the top rather than on the
    /// right. The float arm reads 2.65 / 61.35 / 125.35 / 194.65 at the
    /// same four points: the 2x2 mask measures the gradient half a pixel
    /// off centre, and the LUT's 4-bit quantisation is what hides that on
    /// the uchar arm.
    #[test]
    fn canny_theta_reads_zero_at_the_top_of_a_white_disc() {
        let [gx, gy] = canny_disc33()
            .canny_gradient(1.4, Precision::Integer)
            .unwrap();
        let uchar_at = |x: u32, y: u32| {
            canny_polar_uchar(
                i32::from(u8_at(&gx, x, y)) - 128,
                i32::from(u8_at(&gy, x, y)) - 128,
            )
        };
        for (name, x, y, theta) in [
            ("top", 16, 4, 0u8),
            ("left", 4, 16, 64),
            ("bottom", 16, 28, 128),
            ("right", 28, 16, 192),
        ] {
            assert_eq!(uchar_at(x, y), (32, theta), "uchar disc {name}");
        }

        let [fx, fy] = canny_disc33()
            .canny_gradient(1.4, Precision::Float)
            .unwrap();
        let (sx, sy) = (fx.f32_samples().unwrap(), fy.f32_samples().unwrap());
        for (name, x, y, theta) in [
            ("top", 16usize, 5usize, 2.647_448_f32),
            ("left", 5, 16, 61.352_55),
            ("bottom", 16, 28, 125.352_554),
            ("right", 28, 16, 194.647_45),
        ] {
            let i = y * 33 + x;
            let (g, t) = canny_polar_float(f64::from(sx[i]), f64::from(sy[i]));
            assert!((g - 42.543_835).abs() < 1e-3, "float disc {name} G: {g}");
            assert!((t - theta).abs() < 1e-3, "float disc {name} theta: {t}");
        }

        assert_eq!(
            oracle_raw_sha256(&canny_disc33().canny(1.4, Precision::Integer)),
            "816037cbd20a5101d471c898f5d264fc03cf8f1f82e82e3fe618b85eb0cf22de",
            "record default_disc33_integer"
        );
        assert_eq!(
            oracle_raw_sha256(&canny_disc33().canny(1.4, Precision::Float)),
            "4cff279b981f71b378fcc6a5041b12baea2347be4b5c6433c5a9440532962963",
            "record default_disc33_float"
        );
    }

    /// `G` on the uchar arm can never leave `0..=64`, and it does reach
    /// 64. `(gx * gx + gy * gy + 256) >> 9` with both terms clipped to
    /// `-128..=127` tops out at `(16384 + 16384 + 256) >> 9`, so a wrong
    /// shift is not caught by "it fits in a byte".
    #[test]
    fn canny_polar_uchar_g_stays_inside_0_to_64() {
        let mut highest = 0u8;
        for gx in -128..=127i32 {
            for gy in -128..=127i32 {
                let (g, _) = canny_polar_uchar(gx, gy);
                assert!(g <= 64, "G {g} at ({gx}, {gy})");
                highest = highest.max(g);
            }
        }
        assert_eq!(highest, 64, "the ceiling is reached, not merely respected");
        // The float arm has no such ceiling and no zero at the bottom:
        // the `+ 256.0` makes a flat region 0.5 rather than 0.
        assert!(
            (canny_polar_float(0.0, 0.0).0 - 0.5).abs() < f32::EPSILON,
            "flat float G is 0.5, not 0"
        );
        assert!(
            (canny_polar_float(-510.0, 0.0).0 - 508.507_8).abs() < 1e-3,
            "float G is not bounded to a byte"
        );
    }

    /// The suppression test is `G <= low || G < high`, with `<=` on one
    /// side and `<` on the other, and it is not a typo. The plateau
    /// fixture gives x=4 and x=5 the same `G` (32) and the same `theta`
    /// (64), so exactly one of the two can survive, and which one is
    /// decided entirely by that asymmetry.
    ///
    /// The mirrored fixture puts the same plateau at theta 192 and the
    /// survivor moves to the other side. Between them the pair rules out
    /// every "tidied" variant: both comparisons written `<=` erases the
    /// edge, both written `<` keeps a 2-pixel-wide edge, and swapping
    /// them keeps the wrong pixel. The survivor is always the one on the
    /// strict `<` side.
    #[test]
    fn canny_suppression_keeps_the_strict_less_than_side_of_a_plateau() {
        for (reversed, survivor) in [(false, 4usize), (true, 5)] {
            let im = canny_plateau(false, reversed);
            let out = im.canny(0.01, Precision::Integer);
            let mut want = vec![0u8; 9];
            want[survivor] = 32;
            assert_eq!(
                u8_rows(&out),
                vec![want; 5],
                "plateau_h{} survivor",
                if reversed { "_rev" } else { "" }
            );

            // The plateau really is a plateau: both candidates carry the
            // same G and the same theta going in, so nothing but the
            // comparison can be choosing between them.
            let [gx, gy] = im.canny_gradient(0.01, Precision::Integer).unwrap();
            let polar_at = |x: u32| {
                canny_polar_uchar(
                    i32::from(u8_at(&gx, x, 2)) - 128,
                    i32::from(u8_at(&gy, x, 2)) - 128,
                )
            };
            assert_eq!(polar_at(4), polar_at(5), "the two candidates must tie");
            assert_eq!(polar_at(4).0, 32, "plateau G");
            assert_eq!(
                polar_at(4).1,
                if reversed { 192 } else { 64 },
                "plateau theta"
            );
        }
    }

    /// The same asymmetry on the other axis, where theta is 0 and 128
    /// rather than 64 and 192. This is the pair that catches a direction
    /// table rotated by one step: the offsets run **counter-clockwise
    /// from top-middle**, which is not the order most implementations
    /// number their neighbours in.
    #[test]
    fn canny_suppression_asymmetry_holds_on_the_vertical_axis() {
        for (reversed, survivor) in [(false, 4usize), (true, 5)] {
            let out = canny_plateau(true, reversed).canny(0.01, Precision::Integer);
            let mut want = vec![vec![0u8; 5]; 9];
            want[survivor] = vec![32; 5];
            assert_eq!(
                u8_rows(&out),
                want,
                "plateau_v{} survivor",
                if reversed { "_rev" } else { "" }
            );
        }
    }

    /// The outer ring is **not** zeroed. `vips_embed` with
    /// `VIPS_EXTEND_COPY` duplicates the edge pixels, so an edge lying on
    /// the frame compares against copies of itself and survives.
    ///
    /// `border7` puts real edges on the frame on purpose. Its last row
    /// comes out `0 64 32 32 32 32 32`: live data right on the boundary,
    /// which a port that supplied zeros outside the image would lose.
    #[test]
    fn canny_keeps_edges_that_lie_on_the_frame() {
        let out = canny_border7().canny(0.01, Precision::Integer);
        assert_eq!(
            u8_rows(&out),
            vec![
                vec![0, 32, 0, 0, 0, 0, 0],
                vec![0, 32, 0, 0, 0, 0, 0],
                vec![0, 32, 0, 0, 0, 0, 0],
                vec![0, 32, 0, 0, 0, 0, 0],
                vec![0, 32, 0, 0, 0, 0, 0],
                vec![0, 32, 0, 0, 0, 0, 0],
                vec![0, 64, 32, 32, 32, 32, 32],
            ],
            "record border7_uchar"
        );

        // And on the float arm at the default sigma, where the blur
        // spreads the frame edges into the interior.
        let float = canny_border7().canny(1.4, Precision::Float);
        assert_eq!(
            oracle_raw_sha256(&float),
            "667a55e2a7285d7f3d18b2648d5d8b66f3eef8bca2cf87f8405e2d7616977e3c",
            "record border7_float"
        );
    }

    /// The output format follows the format of the **blurred** image,
    /// which is not the same thing as the format of the input. On the
    /// float arm a uchar input has already been promoted by gaussblur, so
    /// the uchar gradient branch cannot fire, and the only way back into
    /// it is a sigma below 0.2, where gaussblur short-circuits to a copy.
    ///
    /// libviprs has no `double` depth and no `VipsPrecision::APPROXIMATE`,
    /// so the reachable half of `oracle.json -> format_table` is this.
    #[test]
    fn canny_output_format_follows_the_blurred_image() {
        let cases: [(PixelFormat, f64, Precision, PixelFormat); 10] = [
            (
                PixelFormat::Gray8,
                1.4,
                Precision::Integer,
                PixelFormat::Gray8,
            ),
            (PixelFormat::Gray8, 1.4, Precision::Float, float_format(1)),
            (
                PixelFormat::Gray8,
                0.19,
                Precision::Float,
                PixelFormat::Gray8,
            ),
            (PixelFormat::Gray8, 0.2, Precision::Float, float_format(1)),
            (
                PixelFormat::Gray8,
                0.1,
                Precision::Integer,
                PixelFormat::Gray8,
            ),
            (
                PixelFormat::Gray16,
                1.4,
                Precision::Integer,
                float_format(1),
            ),
            (PixelFormat::Gray16, 0.1, Precision::Float, float_format(1)),
            (
                PixelFormat::Rgb8,
                1.4,
                Precision::Integer,
                PixelFormat::Rgb8,
            ),
            (PixelFormat::Rgb8, 1.4, Precision::Float, float_format(3)),
            (
                PixelFormat::RgbaF32,
                1.4,
                Precision::Integer,
                PixelFormat::RgbaF32,
            ),
        ];
        for (src, sigma, precision, want) in cases {
            let im = Raster::zeroed(9, 9, src).unwrap();
            let out = im.canny(sigma, precision);
            assert_eq!(
                out.format(),
                want,
                "canny of {src:?} at sigma {sigma} {precision:?}"
            );
            assert_eq!((out.width(), out.height()), (9, 9), "size of {src:?}");
        }
    }

    /// Size, band count, interpretation and the attached metadata all
    /// round-trip, and the bands are independent: `vips canny` on a
    /// 3-band image is the same op run three times.
    #[test]
    fn canny_round_trips_size_bands_and_metadata() {
        let mut im = canny_noise16rgb()
            .copy()
            .interpretation(Interpretation::Srgb)
            .xres(42.0)
            .build();
        im.set_field("exif-data", MetadataValue::Blob(vec![7, 8, 9]));
        let out = im.canny(1.4, Precision::Integer);
        assert_eq!(out.format(), PixelFormat::Rgb8, "bands round-trip");
        assert_eq!((out.width(), out.height()), (16, 16), "size round-trips");
        assert_eq!(out.interpretation(), Interpretation::Srgb, "interpretation");
        assert!((out.xres() - 42.0).abs() < 1e-12, "xres");
        assert_eq!(
            out.get_field("exif-data"),
            Some(MetadataValue::Blob(vec![7, 8, 9])),
            "attached metadata"
        );
        assert_eq!(
            oracle_raw_sha256(&out),
            "d1c08b4dbdcf9ec9eb005ebd3b4112c418ed0bb94753432b9d2dcecba21a9b4c",
            "record default_noise16rgb_integer"
        );

        // Band independence: band 1 of the colour answer is the mono
        // answer for band 1 of the source.
        let band1 = gray_from(16, 16, |x, y| im.data()[((y * 16 + x) * 3 + 1) as usize]);
        let mono = band1.canny(1.4, Precision::Integer);
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    out.data()[((y * 16 + x) * 3 + 1) as usize],
                    u8_at(&mono, x, y),
                    "band 1 at ({x}, {y})"
                );
            }
        }
    }

    /// Where the two libvips implementations disagree, libviprs is the
    /// portable C one (issue #558), and this is the pin that says so.
    ///
    /// `--precision integer` diverges between vectorised and scalar
    /// libvips at nine of the twelve sigmas the capture swept, by as much
    /// as 28 on a byte through canny's non-linear stages. **Sigma 1.4,
    /// the default, is one of the three that agree**, because its
    /// separable gaussmat has scale 64 and a power of two requantises
    /// exactly. A suite pinned only at the default would pass against
    /// either implementation and prove nothing, so the sigmas here are
    /// 0.8 and 1.6, where the two answers differ in 681 and 280 of the
    /// 4096 samples.
    ///
    /// Both digests are asserted: the one libviprs must produce, and the
    /// one it must not. They are the capture's own, from
    /// `oracle.json -> vector_scalar_sweep`, which records a
    /// `vector_raw_sha256` and a `novector_raw_sha256` for every
    /// (fixture, precision, sigma) it swept.
    #[test]
    fn canny_targets_the_portable_c_libvips_where_the_two_disagree() {
        let noise = canny_noise64();
        for (sigma, novector, vector) in [
            (
                0.8,
                "49403130c8ceda8d5b6d8706bb599b1ae8d2685249f8cd88455a1e279d3ee9a3",
                "c9d53c9ed50d2174adb875662028f972a3a498ec14a1b9d1914a858d77c973f4",
            ),
            (
                1.6,
                "d51bc95aff59a56338f32f1caea39cef89443ad26973bf0083a3513322854597",
                "23a57a8192d5773ed29587430feba68f57c636a7125b6ef96ed9eea8f488d87b",
            ),
        ] {
            let got = oracle_raw_sha256(&noise.canny(sigma, Precision::Integer));
            assert_eq!(got, novector, "sigma {sigma} must match VIPS_NOVECTOR=1");
            assert_ne!(got, vector, "sigma {sigma} must not match the vector path");
        }

        // Sigma 1.4 is where the two agree, so it is a parity pin rather
        // than a discriminating one.
        assert_eq!(
            oracle_raw_sha256(&noise.canny(1.4, Precision::Integer)),
            "1969e4d9be44bf44ad2b4a548939b65688a9fdde090e0dd43f60086125f967c5",
            "sigma 1.4, where both libvips paths agree"
        );
        // The whole operation with no blur at all, which is what reaches
        // every one of the 256 atan2 LUT indices.
        assert_eq!(
            oracle_raw_sha256(&noise.canny(0.01, Precision::Integer)),
            "c01f04c1a300765b460488d8d9c305efca088256fac0bbd7ab850084a7f08662",
            "record gmax_noise64_uchar"
        );
    }

    /// A sigma below 0.2 makes the blur an exact copy
    /// (`convolution/gaussblur.c:71`), so canny reduces to gradient,
    /// polar and thin. There is no sigma *threshold* on the format
    /// question, only that copy: 0.01, 0.1 and 0.19 all give the same
    /// bytes, and 0.2 changes the format on the float arm without
    /// changing a single value, because from 0.2 to 0.55 the integer
    /// gaussmat is still a 1x1 identity.
    #[test]
    fn canny_sigma_below_the_blur_threshold_is_an_exact_no_op() {
        let step = canny_step9();
        let base = step.canny(0.01, Precision::Float);
        for sigma in [0.1, 0.19] {
            assert_eq!(
                step.canny(sigma, Precision::Float).data(),
                base.data(),
                "sigma {sigma} must be the same no-blur answer"
            );
            assert_eq!(
                step.canny(sigma, Precision::Float).format(),
                PixelFormat::Gray8,
                "sigma {sigma} keeps the blur uchar"
            );
        }
        assert_eq!(
            u8_rows(&base),
            vec![vec![0, 0, 0, 0, 32, 0, 0, 0, 0]; 9],
            "record sigma_step9_0.01_float"
        );

        // 0.2 is where gaussblur stops short-circuiting. The value does
        // not change, the format does.
        let promoted = step.canny(0.2, Precision::Float);
        assert_eq!(promoted.format(), float_format(1), "sigma 0.2 promotes");
        let mut want = [[0.0f32; 9]; 9];
        for row in &mut want {
            row[4] = 508.507_8;
        }
        assert_f32_grid(&promoted, &want, "record sigma_step9_0.2_float");
    }

    /// Images small enough that every 3x3 window is mostly border, which
    /// is where replacing the `Extend::Copy` embed with a clamped read
    /// would show up if the two were not the same thing. A 1x1 image
    /// embeds to 3x3 copies of one pixel, so every neighbour ties with
    /// the centre and the `<=` against `low` zeroes it.
    ///
    /// Measured with `VIPS_NOVECTOR=1 vips canny --sigma 0.01`, on both
    /// precisions (below 0.2 the blur is a copy, so the two arms agree).
    #[test]
    fn canny_handles_images_smaller_than_its_own_window() {
        let cases: [(u32, u32, Vec<u8>, Vec<u8>); 4] = [
            (1, 1, vec![200], vec![0]),
            (1, 3, vec![0, 128, 255], vec![0, 32, 0]),
            (3, 1, vec![0, 128, 255], vec![0, 32, 0]),
            (2, 2, vec![0, 255, 90, 10], vec![0, 32, 32, 64]),
        ];
        for (w, h, src, want) in cases {
            let im = Raster::new(w, h, PixelFormat::Gray8, src).unwrap();
            for precision in [Precision::Integer, Precision::Float] {
                let out = im.canny(0.01, precision);
                assert_eq!((out.width(), out.height()), (w, h), "{w}x{h} size");
                assert_eq!(out.data(), want.as_slice(), "{w}x{h} at {precision:?}");
            }
        }
    }

    /// The `try_*` and panicking forms are the same call, and a sigma
    /// outside the mask generator's range is a typed error rather than a
    /// panic.
    ///
    /// libviprs does **not** reproduce what the vips CLI does with an
    /// out-of-range sigma. GObject refuses anything outside `0.01..1000`
    /// with a `GLib-GObject-CRITICAL`, silently leaves sigma at its 1.4
    /// default and still exits 0, so `vips canny --sigma 0` is byte
    /// identical to `--sigma 1.4`. That is the property system talking,
    /// not the operation, and silently ignoring an argument is not a
    /// behaviour worth porting: `try_canny` honours whatever it is given,
    /// exactly as [`Raster::try_gaussblur`] already does.
    #[test]
    fn canny_try_and_panicking_forms_agree() {
        let im = canny_step9();
        assert_eq!(
            im.canny(1.4, Precision::Integer).data(),
            im.try_canny(1.4, Precision::Integer).unwrap().data()
        );
        // Below 0.2 the blur is a copy, so sigma 0 is a legal no-blur
        // request here rather than the 1.4 the CLI quietly substitutes.
        assert_eq!(
            im.try_canny(0.0, Precision::Float).unwrap().data(),
            im.try_canny(0.01, Precision::Float).unwrap().data(),
            "sigma 0 is the no-blur answer, not the 1.4 one"
        );
        assert_ne!(
            im.try_canny(0.0, Precision::Float).unwrap().format(),
            im.try_canny(1.4, Precision::Float).unwrap().format(),
            "and it is not what --sigma 0 gives the CLI"
        );
        assert!(matches!(
            im.try_canny(f64::NAN, Precision::Float),
            Err(ConvolutionError::InvalidMaskParameter {
                op: "gaussmat",
                param: "sigma",
                ..
            })
        ));
    }
}
