//! Mosaicing operations ported from libvips.
//!
//! This module is the next batch of the libvips operation surface required
//! by the ported integration tests (after [`crate::bands`],
//! [`crate::arithmetic`], [`crate::extract`], [`crate::conversion`],
//! [`crate::draw`], [`crate::histogram`], [`crate::imageio`],
//! [`crate::composite`], [`crate::colour`], and [`crate::morphology`]):
//! feathered merging of two overlapping images, automatic tie-point
//! mosaicing, and global brightness balancing of an assembled mosaic.
//! Operations that can fail on caller input exist in two forms, following
//! the established convention:
//!
//! * a fallible `try_*` method returning `Result<_, MosaicError>` with
//!   typed errors for mismatched formats, missing overlap, and failed
//!   tie-point searches; and
//! * a panicking convenience method matching the ported-test call surface
//!   (`merge`, `mosaic`, `global_balance`) exactly, delegating to the
//!   `try_*` form.
//!
//! # Operations
//!
//! | Method | libvips equivalent | Result |
//! |---|---|---|
//! | [`Raster::merge`] | `vips_merge` | feathered join of two images |
//! | [`Raster::mosaic`] | `vips_mosaic` | tie-point aligned join |
//! | [`Raster::global_balance`] | `vips_globalbalance` | brightness-balanced mosaic, float |
//!
//! # Merge semantics (from `lrmerge.c` / `tbmerge.c`)
//!
//! `self` is the reference image at `(0, 0)`; `other` is displaced to
//! `(-dx, -dy)`. The output is the union bounding box of the two placed
//! images, with uncovered corners black. Over the overlap, each scan-line
//! (horizontal merge) or column (vertical merge) is feathered with the
//! libvips raised-cosine ramp between `first` (the first non-zero `other`
//! pixel) and `last` (the last non-zero `self` pixel), the ramp width
//! clipped to the `mblend` maximum of 10 pixels (the libvips default) by
//! shrinking both ends towards the middle. Zero pixels are treated as
//! transparent on both sides, integer depths blend through the
//! `BLEND_SCALE` integer coefficient tables with truncating division, and
//! float depths blend with the double-precision coefficients, all exactly
//! as the C does. When `other` lies entirely on the wrong side (`dx > 0`
//! or beyond the reference edge), libvips falls back to `vips_insert`
//! with expansion; the same paste-without-blend fallback happens here.
//!
//! # Mosaic semantics (from `lrmosaic.c` / `im_lrcalcon.c` / `chkpair.c`)
//!
//! [`Raster::mosaic`] refines the caller's tie-point guess exactly like
//! `vips_mosaic` with the default parameters (`bandno` 0, `hwindow` 5,
//! `harea` 15, `mblend` 10): band 0 of each overlap estimate is
//! contrast-stretched to 8-bit (`vips_scale`), the reference overlap is
//! divided into three strips and the twenty highest-contrast 11x11
//! windows found in each on a 5-pixel grid, each of the sixty points is
//! located in the secondary image by normalised spatial correlation
//! (`vips_spcor`, edge-replicated) over a +-15 pixel search square, a
//! first-order linear model is fitted (`im_clinear`), outliers are
//! discarded and the model refitted until stable (`im_improve`), and the
//! surviving displacements are averaged with round-half-to-even
//! (`im_avgdxdy`). The corrected offset then drives [`Raster::merge`].
//! Failures surface as typed errors exactly where libvips errors: an
//! overlap smaller than the search geometry, fewer than twenty usable
//! high-contrast windows per strip, no tie-points surviving the fit, or a
//! singular fit matrix.
//!
//! # Global balance semantics (from `global_balance.c`)
//!
//! libvips reconstructs the mosaic tree by parsing `#LRJOIN` / `#TBJOIN`
//! image-history lines and re-opening the named source files. Rasters
//! here have no filenames, so [`Raster::merge`] and [`Raster::mosaic`]
//! attach the equivalent information structurally: a `mosaic-join-tree`
//! metadata blob carrying the join parameters and the leaf images
//! themselves. [`Raster::global_balance`] rebuilds the tree from that
//! blob, finds every leaf-pair overlap of at least 20x20 pixels, masks
//! each overlap to the pixels non-zero in both leaves, computes the
//! libvips overlap statistic (masked mean scaled by the masked pixel
//! fraction), solves the least-mean-square factor system anchored on the
//! first leaf with the default source gamma 1.6, normalises the factors
//! to a mean of one, scales every leaf by `v -> (v^(1/gamma) * factor)^gamma`
//! through a lookup table, and re-merges the tree in float. The output is
//! always float (`int_output` is not ported), one factor per leaf, same
//! geometry as the input mosaic.
//!
//! # Documented deviations from libvips
//!
//! * Both images must share one [`crate::pixel::PixelFormat`]; libvips
//!   instead casts the inputs to a common format and band count. Cast
//!   first when formats differ.
//! * The join metadata travels in the structured `mosaic-join-tree` blob
//!   (native byte order, process-internal) rather than filename-based
//!   history, so `global_balance` works on in-memory rasters; a raster
//!   without that field is a typed error, as an image without mosaic
//!   history is in libvips.
//! * Contrast-candidate ties are broken by stable scan order where the C
//!   `qsort` tie order is unspecified, and correlation-surface ties take
//!   the first maximum in row-major order where threaded `vips_max` is
//!   unspecified.
//! * `mblend` is fixed at the libvips default of 10 on the public
//!   surface; the ported tests never pass another value.

use crate::imageio::MetadataValue;
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use std::sync::OnceLock;
use thiserror::Error;

/// Join direction for [`Raster::merge`] and [`Raster::mosaic`].
///
/// `Horizontal` joins left-right with `self` on the left; `Vertical`
/// joins top-bottom with `self` on top. Mirrors `VipsDirection`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MergeDirection {
    /// Left-right join, `self` on the left.
    Horizontal,
    /// Top-bottom join, `self` on top.
    Vertical,
}

/// Typed errors for the mosaicing operations in [`crate::mosaicing`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MosaicError {
    /// The two images do not share a pixel format.
    #[error("pixel formats differ: {reference:?} vs {secondary:?}; cast to a common format first")]
    FormatMismatch {
        reference: PixelFormat,
        secondary: PixelFormat,
    },
    /// The placed images do not overlap.
    #[error("no overlap")]
    NoOverlap,
    /// The reference image extends past the secondary on the trailing
    /// edge (or starts after it), so a feathered join is impossible.
    #[error("too much overlap")]
    TooMuchOverlap,
    /// The tie-point overlap estimate is smaller than the search geometry
    /// needs.
    #[error("overlap too small for the search size")]
    OverlapTooSmall,
    /// A search strip holds fewer than the required high-contrast
    /// windows.
    #[error("found {found} tie-points, need at least {needed}")]
    TooFewPoints { found: usize, needed: usize },
    /// No tie-points survived correlation and filtering.
    #[error("no tie points")]
    NoTiePoints,
    /// A least-squares fit matrix was singular.
    #[error("singular matrix in mosaic fit")]
    SingularMatrix,
    /// The raster carries no `mosaic-join-tree` metadata, so it was not
    /// built by [`Raster::merge`] / [`Raster::mosaic`].
    #[error("no mosaic metadata: input was not built by merge or mosaic")]
    NoMosaicMetadata,
    /// The `mosaic-join-tree` metadata failed to deserialize.
    #[error("corrupt mosaic metadata")]
    CorruptMosaicMetadata,
    /// Constructing a result raster failed (allocation budget, size
    /// overflow).
    #[error(transparent)]
    Raster(#[from] RasterError),
}

#[track_caller]
fn expect_mosaic<T>(op: &str, r: Result<T, MosaicError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Shared constants (pmosaicing.h / mosaic.c defaults)
// ---------------------------------------------------------------------------

/// Number of entries in the raised-cosine blend table (`BLEND_SIZE`).
const BLEND_SHIFT: u32 = 10;
const BLEND_SIZE: usize = 1 << BLEND_SHIFT;
/// Integer blend coefficient scale (`BLEND_SCALE`).
const BLEND_SCALE: i64 = 4096;
/// Default maximum blend width (`mblend`).
const DEFAULT_MBLEND: i64 = 10;
/// Half correlation window size (`hwindow`).
const HWINDOW: i64 = 5;
/// Half search area size (`harea`).
const HAREA: i64 = 15;
/// Total tie-points searched (`VIPS_MAXPOINTS`).
const MAXPOINTS: usize = 60;
/// Number of overlap strips (`AREAS`).
const AREAS: usize = 3;
/// Minimum significant overlap in pixels (`TRIVIAL`, 20 * 20).
const TRIVIAL: i64 = 400;
/// Default source gamma for global balancing.
const BALANCE_GAMMA: f64 = 1.6;
/// Stack size of the worker thread [`Raster::try_global_balance`] runs its tree
/// walk on.
///
/// `merge` builds left-deep join trees, so a linear stitch of `N` tiles nests
/// `N - 1` joins and the `collect_leaves` / `rebuild` walks recurse once per
/// tile. `deserialize` no longer caps depth (it parses iteratively, matching
/// vips's handling of deep mosaics), so the walk is run on a thread with this
/// much stack — far beyond the ~16k-frame ceiling of the default 8 MiB stack —
/// to keep a deep long-strip panorama, or a crafted deep blob, from overflowing.
const GLOBAL_BALANCE_STACK_BYTES: usize = 256 * 1024 * 1024;
/// Metadata field carrying the serialized join tree.
const JOIN_TREE_FIELD: &str = "mosaic-join-tree";

// ---------------------------------------------------------------------------
// Rect helpers (vips_rect_*)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rect {
    left: i64,
    top: i64,
    width: i64,
    height: i64,
}

impl Rect {
    fn right(&self) -> i64 {
        self.left + self.width
    }

    fn bottom(&self) -> i64 {
        self.top + self.height
    }

    fn is_empty(&self) -> bool {
        self.width <= 0 || self.height <= 0
    }

    fn intersect(&self, other: &Rect) -> Rect {
        let left = self.left.max(other.left);
        let top = self.top.max(other.top);
        Rect {
            left,
            top,
            width: (self.right().min(other.right()) - left).max(0),
            height: (self.bottom().min(other.bottom()) - top).max(0),
        }
    }

    fn union(&self, other: &Rect) -> Rect {
        let left = self.left.min(other.left);
        let top = self.top.min(other.top);
        Rect {
            left,
            top,
            width: self.right().max(other.right()) - left,
            height: self.bottom().max(other.bottom()) - top,
        }
    }
}

// ---------------------------------------------------------------------------
// Blend tables (vips__make_blend_luts)
// ---------------------------------------------------------------------------

struct BlendLuts {
    coef1: Vec<f64>,
    coef2: Vec<f64>,
    icoef1: Vec<i64>,
    icoef2: Vec<i64>,
}

fn blend_luts() -> &'static BlendLuts {
    static LUTS: OnceLock<BlendLuts> = OnceLock::new();
    LUTS.get_or_init(|| {
        let mut luts = BlendLuts {
            coef1: Vec::with_capacity(BLEND_SIZE),
            coef2: Vec::with_capacity(BLEND_SIZE),
            icoef1: Vec::with_capacity(BLEND_SIZE),
            icoef2: Vec::with_capacity(BLEND_SIZE),
        };
        for x in 0..BLEND_SIZE {
            let a = std::f64::consts::PI * x as f64 / (BLEND_SIZE as f64 - 1.0);
            let c1 = (a.cos() + 1.0) / 2.0;
            let c2 = 1.0 - c1;
            luts.coef1.push(c1);
            luts.coef2.push(c2);
            // The C fills the int tables with a double -> int cast, which
            // truncates toward zero.
            luts.icoef1.push((c1 * BLEND_SCALE as f64) as i64);
            luts.icoef2.push((c2 * BLEND_SCALE as f64) as i64);
        }
        luts
    })
}

// ---------------------------------------------------------------------------
// Sample plumbing: one code path per depth, like the C's iblend / fblend
// ---------------------------------------------------------------------------

trait BlendSample: Copy + PartialEq {
    fn get(data: &[u8], elem: usize) -> Self;
    fn put(data: &mut [u8], elem: usize, v: Self);
    fn is_nonzero(self) -> bool;
    /// Blend `r` (reference) with `s` (secondary) at ramp index `inx`.
    fn blend(inx: usize, r: Self, s: Self) -> Self;
}

impl BlendSample for u8 {
    fn get(data: &[u8], elem: usize) -> Self {
        data[elem]
    }

    fn put(data: &mut [u8], elem: usize, v: Self) {
        data[elem] = v;
    }

    fn is_nonzero(self) -> bool {
        self != 0
    }

    fn blend(inx: usize, r: Self, s: Self) -> Self {
        let luts = blend_luts();
        // Integer blend with truncating division on each term (iblend).
        (luts.icoef1[inx] * r as i64 / BLEND_SCALE + luts.icoef2[inx] * s as i64 / BLEND_SCALE)
            as u8
    }
}

impl BlendSample for u16 {
    fn get(data: &[u8], elem: usize) -> Self {
        u16::from_ne_bytes([data[elem * 2], data[elem * 2 + 1]])
    }

    fn put(data: &mut [u8], elem: usize, v: Self) {
        data[elem * 2..elem * 2 + 2].copy_from_slice(&v.to_ne_bytes());
    }

    fn is_nonzero(self) -> bool {
        self != 0
    }

    fn blend(inx: usize, r: Self, s: Self) -> Self {
        let luts = blend_luts();
        (luts.icoef1[inx] * r as i64 / BLEND_SCALE + luts.icoef2[inx] * s as i64 / BLEND_SCALE)
            as u16
    }
}

impl BlendSample for f32 {
    fn get(data: &[u8], elem: usize) -> Self {
        f32::from_ne_bytes([
            data[elem * 4],
            data[elem * 4 + 1],
            data[elem * 4 + 2],
            data[elem * 4 + 3],
        ])
    }

    fn put(data: &mut [u8], elem: usize, v: Self) {
        data[elem * 4..elem * 4 + 4].copy_from_slice(&v.to_ne_bytes());
    }

    fn is_nonzero(self) -> bool {
        self != 0.0
    }

    fn blend(inx: usize, r: Self, s: Self) -> Self {
        let luts = blend_luts();
        // Float blend in double precision (fblend).
        (luts.coef1[inx] * r as f64 + luts.coef2[inx] * s as f64) as f32
    }
}

/// Whole-pixel zero test (`TEST_ZERO`): every band element must be zero.
fn pixel_is_zero<S: BlendSample>(data: &[u8], elem: usize, bands: usize) -> bool {
    (0..bands).all(|b| !S::get(data, elem + b).is_nonzero())
}

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

/// Per-call merge geometry, mirroring the `Overlapping` struct: all rects
/// are in output coordinates with the union at (0, 0).
struct MergeState {
    rarea: Rect,
    sarea: Rect,
    overlap: Rect,
    oarea: Rect,
}

fn build_merge_state(
    ref_: &Raster,
    sec: &Raster,
    dx: i64,
    dy: i64,
) -> Result<MergeState, MosaicError> {
    let mut rarea = Rect {
        left: 0,
        top: 0,
        width: ref_.width() as i64,
        height: ref_.height() as i64,
    };
    let mut sarea = Rect {
        left: -dx,
        top: -dy,
        width: sec.width() as i64,
        height: sec.height() as i64,
    };
    let mut overlap = rarea.intersect(&sarea);
    if overlap.is_empty() {
        return Err(MosaicError::NoOverlap);
    }
    let oarea = rarea.union(&sarea);

    // Translate so the output is at (0, 0).
    rarea.left -= oarea.left;
    rarea.top -= oarea.top;
    sarea.left -= oarea.left;
    sarea.top -= oarea.top;
    overlap.left -= oarea.left;
    overlap.top -= oarea.top;
    Ok(MergeState {
        rarea,
        sarea,
        overlap,
        oarea: Rect {
            left: 0,
            top: 0,
            width: oarea.width,
            height: oarea.height,
        },
    })
}

/// Copy the whole of `src` into `out` with its top-left corner at
/// `(left, top)` (always in range by construction).
fn paste(out: &mut Raster, src: &Raster, left: i64, top: i64) {
    let bpp = src.format().bytes_per_pixel();
    let src_stride = src.stride();
    let out_stride = out.stride();
    let row_bytes = src.width() as usize * bpp;
    let src_data = src.data();
    let out_data = out.data_mut();
    for y in 0..src.height() as usize {
        let s = y * src_stride;
        let d = (top as usize + y) * out_stride + left as usize * bpp;
        out_data[d..d + row_bytes].copy_from_slice(&src_data[s..s + row_bytes]);
    }
}

/// `find_first`: position of the first non-zero pixel of `sec`'s span,
/// scanning band elements left to right; one past the end when all zero.
/// `find_last`: last non-zero of `ref`'s span, `start - 1` when all zero
/// (through the same truncating element/band division as the C).
fn find_first<S: BlendSample>(data: &[u8], base_elem: usize, w: i64, bands: usize) -> i64 {
    let ne = w * bands as i64;
    let mut i = 0i64;
    while i < ne && !S::get(data, base_elem + i as usize).is_nonzero() {
        i += 1;
    }
    i / bands as i64
}

fn find_last<S: BlendSample>(data: &[u8], base_elem: usize, w: i64, bands: usize) -> i64 {
    let ne = w * bands as i64;
    let mut i = ne - 1;
    while i >= 0 && !S::get(data, base_elem + i as usize).is_nonzero() {
        i -= 1;
    }
    i / bands as i64
}

/// Render one merged output, all three depths sharing this generic body.
fn render_merge<S: BlendSample>(
    out: &mut Raster,
    ref_: &Raster,
    sec: &Raster,
    st: &MergeState,
    direction: MergeDirection,
    mwidth: i64,
) {
    // Stage 1/2 (vips__merge_gen difficult case): paste ref, then sec over
    // it; stage 3 rewrites the overlap with the feathered blend.
    paste(out, ref_, st.rarea.left, st.rarea.top);
    paste(out, sec, st.sarea.left, st.sarea.top);

    let bands = ref_.format().channels();
    let ov = st.overlap;
    let ref_w = ref_.width() as i64;
    let sec_w = sec.width() as i64;
    let out_w = out.width() as i64;
    let ref_data = ref_.data();
    let sec_data = sec.data();
    let out_data = out.data_mut();

    let ref_elem =
        |x: i64, y: i64| (((y - st.rarea.top) * ref_w + (x - st.rarea.left)) as usize) * bands;
    let sec_elem =
        |x: i64, y: i64| (((y - st.sarea.top) * sec_w + (x - st.sarea.left)) as usize) * bands;
    let out_elem = |x: i64, y: i64| ((y * out_w + x) as usize) * bands;

    // first/last per overlap scan-line (lr) or column (tb), in output
    // coordinates, then clipped to the maximum blend width by shrinking
    // both ends (make_firstlast).
    let fl_len = match direction {
        MergeDirection::Horizontal => ov.height,
        MergeDirection::Vertical => ov.width,
    };
    let mut fl: Vec<(i64, i64)> = Vec::with_capacity(fl_len as usize);
    for j in 0..fl_len {
        let (mut first, mut last) = match direction {
            MergeDirection::Horizontal => {
                // find_first / find_last translated to output space.
                let f = ov.left
                    + find_first::<S>(sec_data, sec_elem(ov.left, ov.top + j), ov.width, bands);
                let l = ov.left
                    + find_last::<S>(ref_data, ref_elem(ov.left, ov.top + j), ov.width, bands);
                (f, l)
            }
            MergeDirection::Vertical => {
                let x = ov.left + j;
                // find_top / find_bot: scan the column's pixels.
                let mut i = 0i64;
                while i < ov.height {
                    if !pixel_is_zero::<S>(sec_data, sec_elem(x, ov.top + i), bands) {
                        break;
                    }
                    i += 1;
                }
                let f = ov.top + i;
                let mut i = ov.height - 1;
                while i >= 0 {
                    if !pixel_is_zero::<S>(ref_data, ref_elem(x, ov.top + i), bands) {
                        break;
                    }
                    i -= 1;
                }
                let l = ov.top + i;
                (f, l)
            }
        };
        if mwidth >= 0 && last - first > mwidth {
            let shrinkby = (last - first) - mwidth;
            first += shrinkby / 2;
            last -= shrinkby / 2;
        }
        fl.push((first, last));
    }

    match direction {
        MergeDirection::Horizontal => {
            // lr_blend / iblend: per scan-line ramp between first and last.
            for j in 0..ov.height {
                let y = ov.top + j;
                let (first, last) = fl[j as usize];
                let left = (first - ov.left).clamp(0, ov.width);
                let right = (last - ov.left).clamp(left, ov.width);
                for x in 0..ov.width {
                    let ox = ov.left + x;
                    let re = ref_elem(ox, y);
                    let se = sec_elem(ox, y);
                    let oe = out_elem(ox, y);
                    let ref_zero = pixel_is_zero::<S>(ref_data, re, bands);
                    let sec_zero = pixel_is_zero::<S>(sec_data, se, bands);
                    let use_ref = if x < left {
                        !ref_zero
                    } else if x >= right {
                        sec_zero
                    } else if !ref_zero && !sec_zero {
                        let inx = (((ox - first) << BLEND_SHIFT) / (last - first)) as usize;
                        for b in 0..bands {
                            let v =
                                S::blend(inx, S::get(ref_data, re + b), S::get(sec_data, se + b));
                            S::put(out_data, oe + b, v);
                        }
                        continue;
                    } else {
                        !ref_zero
                    };
                    let (src, e) = if use_ref {
                        (ref_data, re)
                    } else {
                        (sec_data, se)
                    };
                    for b in 0..bands {
                        S::put(out_data, oe + b, S::get(src, e + b));
                    }
                }
            }
        }
        MergeDirection::Vertical => {
            // tb_blend: per-column ramp between first and last.
            for j in 0..ov.height {
                let y = ov.top + j;
                for x in 0..ov.width {
                    let ox = ov.left + x;
                    let (first, last) = fl[x as usize];
                    let re = ref_elem(ox, y);
                    let se = sec_elem(ox, y);
                    let oe = out_elem(ox, y);
                    let ref_zero = pixel_is_zero::<S>(ref_data, re, bands);
                    let sec_zero = pixel_is_zero::<S>(sec_data, se, bands);
                    let use_ref = if y < first {
                        !ref_zero
                    } else if y >= last {
                        sec_zero
                    } else if !ref_zero && !sec_zero {
                        let inx = (((y - first) << BLEND_SHIFT) / (last - first)) as usize;
                        for b in 0..bands {
                            let v =
                                S::blend(inx, S::get(ref_data, re + b), S::get(sec_data, se + b));
                            S::put(out_data, oe + b, v);
                        }
                        continue;
                    } else {
                        !ref_zero
                    };
                    let (src, e) = if use_ref {
                        (ref_data, re)
                    } else {
                        (sec_data, se)
                    };
                    for b in 0..bands {
                        S::put(out_data, oe + b, S::get(src, e + b));
                    }
                }
            }
        }
    }
}

/// Shared merge implementation; `mwidth` is the maximum blend width
/// (`-1` = unlimited).
fn merge_impl(
    ref_: &Raster,
    sec: &Raster,
    direction: MergeDirection,
    dx: i64,
    dy: i64,
    mwidth: i64,
) -> Result<Raster, MosaicError> {
    if ref_.format() != sec.format() {
        return Err(MosaicError::FormatMismatch {
            reference: ref_.format(),
            secondary: sec.format(),
        });
    }

    // Wrong-side / disjoint displacement: libvips falls back to
    // vips_insert with expansion (paste both, no blend).
    let insert_fallback = match direction {
        MergeDirection::Horizontal => dx > 0 || dx < 1 - ref_.width() as i64,
        MergeDirection::Vertical => dy > 0 || dy < 1 - ref_.height() as i64,
    };

    let mut out = if insert_fallback {
        let rarea = Rect {
            left: 0,
            top: 0,
            width: ref_.width() as i64,
            height: ref_.height() as i64,
        };
        let sarea = Rect {
            left: -dx,
            top: -dy,
            width: sec.width() as i64,
            height: sec.height() as i64,
        };
        let u = rarea.union(&sarea);
        let mut out = Raster::zeroed(u.width as u32, u.height as u32, ref_.format())?;
        paste(&mut out, ref_, -u.left, -u.top);
        paste(&mut out, sec, -dx - u.left, -dy - u.top);
        out
    } else {
        let st = build_merge_state(ref_, sec, dx, dy)?;
        match direction {
            MergeDirection::Horizontal => {
                if st.rarea.right() > st.sarea.right() || st.rarea.left > st.sarea.left {
                    return Err(MosaicError::TooMuchOverlap);
                }
            }
            MergeDirection::Vertical => {
                if st.rarea.bottom() > st.sarea.bottom() || st.rarea.top > st.sarea.top {
                    return Err(MosaicError::TooMuchOverlap);
                }
            }
        }
        let mut out = Raster::zeroed(st.oarea.width as u32, st.oarea.height as u32, ref_.format())?;
        match ref_.format().bytes_per_channel() {
            1 => render_merge::<u8>(&mut out, ref_, sec, &st, direction, mwidth),
            2 => render_merge::<u16>(&mut out, ref_, sec, &st, direction, mwidth),
            _ => render_merge::<f32>(&mut out, ref_, sec, &st, direction, mwidth),
        }
        out
    };

    let tree = JoinTree::Join {
        direction,
        dx: dx as i32,
        dy: dy as i32,
        mblend: mwidth as i32,
        ref_node: Box::new(tree_of(ref_)?),
        sec_node: Box::new(tree_of(sec)?),
    };
    out.set_field(JOIN_TREE_FIELD, MetadataValue::Blob(tree.serialize()));
    Ok(out)
}

// ---------------------------------------------------------------------------
// Join-tree metadata (the structured stand-in for #LRJOIN history lines)
// ---------------------------------------------------------------------------

enum JoinTree {
    Leaf(Raster),
    Join {
        direction: MergeDirection,
        dx: i32,
        dy: i32,
        mblend: i32,
        ref_node: Box<JoinTree>,
        sec_node: Box<JoinTree>,
    },
}

impl Drop for JoinTree {
    fn drop(&mut self) {
        // Dismantle iteratively: a left-deep tree nests one join per stitched
        // tile, so the default recursive `Box<JoinTree>` drop would overflow the
        // stack on a deep mosaic (exactly the input `deserialize` now accepts
        // without a depth cap). Each join's children are swapped out for a
        // trivial childless leaf and pushed onto a heap worklist, so every node
        // that actually drops has no children left to recurse into.
        fn placeholder() -> JoinTree {
            JoinTree::Leaf(
                Raster::new(
                    1,
                    1,
                    PixelFormat::with_channels(1, 1).expect("1x1 gray"),
                    vec![0],
                )
                .expect("1x1 leaf raster"),
            )
        }
        fn take_children(node: &mut JoinTree, pending: &mut Vec<JoinTree>) {
            if let JoinTree::Join {
                ref_node, sec_node, ..
            } = node
            {
                pending.push(std::mem::replace(&mut **ref_node, placeholder()));
                pending.push(std::mem::replace(&mut **sec_node, placeholder()));
            }
        }

        let mut pending: Vec<JoinTree> = Vec::new();
        take_children(self, &mut pending);
        while let Some(mut node) = pending.pop() {
            take_children(&mut node, &mut pending);
            // `node` now has only placeholder children and drops without
            // recursing when it goes out of scope here.
        }
    }
}

impl JoinTree {
    fn serialize(&self) -> Vec<u8> {
        let mut out = b"VMJ1".to_vec();
        self.write_node(&mut out);
        out
    }

    fn write_node(&self, out: &mut Vec<u8>) {
        match self {
            JoinTree::Leaf(r) => {
                out.push(0);
                out.extend_from_slice(&r.width().to_le_bytes());
                out.extend_from_slice(&r.height().to_le_bytes());
                out.extend_from_slice(&(r.format().channels() as u16).to_le_bytes());
                out.push(r.format().bytes_per_channel() as u8);
                out.extend_from_slice(r.data());
            }
            JoinTree::Join {
                direction,
                dx,
                dy,
                mblend,
                ref_node,
                sec_node,
            } => {
                out.push(1);
                out.push(match direction {
                    MergeDirection::Horizontal => 0,
                    MergeDirection::Vertical => 1,
                });
                out.extend_from_slice(&dx.to_le_bytes());
                out.extend_from_slice(&dy.to_le_bytes());
                out.extend_from_slice(&mblend.to_le_bytes());
                ref_node.write_node(out);
                sec_node.write_node(out);
            }
        }
    }

    fn deserialize(blob: &[u8]) -> Result<JoinTree, MosaicError> {
        let corrupt = || MosaicError::CorruptMosaicMetadata;
        let rest = blob.strip_prefix(b"VMJ1".as_slice()).ok_or_else(corrupt)?;
        let mut cursor = rest;
        let tree = Self::read_node(&mut cursor)?;
        if !cursor.is_empty() {
            return Err(corrupt());
        }
        Ok(tree)
    }

    /// Parse one join tree from `cursor`, consuming exactly its preorder bytes.
    ///
    /// Iterative with an explicit heap `Vec` stack rather than recursive.
    /// `merge` builds left-deep trees, so a linear stitch of `N` tiles nests
    /// `N - 1` joins along the reference spine; whole-slide / long-strip
    /// panoramas run to hundreds or thousands of tiles, and vips balances
    /// arbitrarily deep mosaics. A recursive parse overflows the stack on such
    /// input, and a depth cap would reject legitimate deep panoramas that vips
    /// accepts — so the preorder walk is driven off the heap instead. Work and
    /// peak memory stay O(blob length), which already bounds a crafted blob.
    fn read_node(cursor: &mut &[u8]) -> Result<JoinTree, MosaicError> {
        let corrupt = || MosaicError::CorruptMosaicMetadata;
        let take = |cursor: &mut &[u8], n: usize| -> Result<Vec<u8>, MosaicError> {
            if cursor.len() < n {
                return Err(MosaicError::CorruptMosaicMetadata);
            }
            let (head, tail) = cursor.split_at(n);
            *cursor = tail;
            Ok(head.to_vec())
        };
        let take_u32 = |cursor: &mut &[u8]| -> Result<u32, MosaicError> {
            let b = take(cursor, 4)?;
            Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let take_i32 = |cursor: &mut &[u8]| -> Result<i32, MosaicError> {
            let b = take(cursor, 4)?;
            Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };

        // A join whose header has been read but whose children are still being
        // assembled. Its reference child fills first, then its secondary child;
        // once both are in hand the frame pops and folds into a `Join`.
        struct Frame {
            direction: MergeDirection,
            dx: i32,
            dy: i32,
            mblend: i32,
            ref_node: Option<JoinTree>,
        }

        let mut stack: Vec<Frame> = Vec::new();
        loop {
            // Read the next node header in preorder. A leaf is a complete
            // subtree; a join header defers to its children, which follow it.
            let mut done = match take(cursor, 1)?[0] {
                0 => {
                    let width = take_u32(cursor)?;
                    let height = take_u32(cursor)?;
                    let ch = take(cursor, 2)?;
                    let channels = u16::from_le_bytes([ch[0], ch[1]]) as usize;
                    let bpc = take(cursor, 1)?[0] as usize;
                    let format = PixelFormat::with_channels(channels, bpc).ok_or_else(corrupt)?;
                    let len = (width as usize)
                        .checked_mul(height as usize)
                        .and_then(|p| p.checked_mul(format.bytes_per_pixel()))
                        .ok_or_else(corrupt)?;
                    let data = take(cursor, len)?;
                    let raster = Raster::new(width, height, format, data).map_err(|_| corrupt())?;
                    JoinTree::Leaf(raster)
                }
                1 => {
                    let direction = match take(cursor, 1)?[0] {
                        0 => MergeDirection::Horizontal,
                        1 => MergeDirection::Vertical,
                        _ => return Err(corrupt()),
                    };
                    let dx = take_i32(cursor)?;
                    let dy = take_i32(cursor)?;
                    let mblend = take_i32(cursor)?;
                    stack.push(Frame {
                        direction,
                        dx,
                        dy,
                        mblend,
                        ref_node: None,
                    });
                    // The reference child is the next node in the stream.
                    continue;
                }
                _ => return Err(corrupt()),
            };

            // `done` is a completed subtree. Slot it into the innermost pending
            // join, folding joins whose second child has now arrived, until it
            // lands as a reference child (more stream to read) or, with the
            // stack empty, becomes the root.
            loop {
                let Some(frame) = stack.last_mut() else {
                    return Ok(done);
                };
                if frame.ref_node.is_none() {
                    frame.ref_node = Some(done);
                    break; // its secondary child is read next
                }
                let frame = stack.pop().expect("last_mut just returned Some");
                done = JoinTree::Join {
                    direction: frame.direction,
                    dx: frame.dx,
                    dy: frame.dy,
                    mblend: frame.mblend,
                    ref_node: Box::new(frame.ref_node.expect("reference filled before secondary")),
                    sec_node: Box::new(done),
                };
            }
        }
    }
}

/// The join tree a raster contributes when used as a merge input: its own
/// tree when it was itself built by merge/mosaic, otherwise a new leaf.
fn tree_of(r: &Raster) -> Result<JoinTree, MosaicError> {
    match r.get_field(JOIN_TREE_FIELD) {
        Some(MetadataValue::Blob(blob)) => JoinTree::deserialize(&blob),
        Some(_) => Err(MosaicError::CorruptMosaicMetadata),
        None => Ok(JoinTree::Leaf(r.clone())),
    }
}

// ---------------------------------------------------------------------------
// Mosaic: tie-point search (im_lrcalcon.c / chkpair.c / spcor.c)
// ---------------------------------------------------------------------------

/// A one-band 8-bit search image (the contrast-stretched overlap).
struct GrayU8 {
    w: i64,
    h: i64,
    data: Vec<u8>,
}

impl GrayU8 {
    fn at(&self, x: i64, y: i64) -> u8 {
        self.data[(y * self.w + x) as usize]
    }
}

/// Extract band 0 of `area` and contrast-stretch it to 0..255 exactly as
/// `vips_scale` does: `clip(0, 255, trunc(f * (v - min) + 0.5))` with
/// `f = 255 / (max - min)`, or all black when the range is zero.
fn scale_band0(src: &Raster, area: Rect) -> Result<GrayU8, MosaicError> {
    let ex = src.extract(
        area.left as u32,
        area.top as u32,
        area.width as u32,
        area.height as u32,
    )?;
    let bands = ex.format().channels();
    let bpc = ex.format().bytes_per_channel();
    let data = ex.data();
    let n = (area.width * area.height) as usize;
    let sample = |i: usize| -> f64 {
        let elem = i * bands;
        match bpc {
            1 => data[elem] as f64,
            2 => u16::from_ne_bytes([data[elem * 2], data[elem * 2 + 1]]) as f64,
            _ => f32::from_ne_bytes([
                data[elem * 4],
                data[elem * 4 + 1],
                data[elem * 4 + 2],
                data[elem * 4 + 3],
            ]) as f64,
        }
    };
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for i in 0..n {
        let v = sample(i);
        mn = mn.min(v);
        mx = mx.max(v);
    }
    let mut out = vec![0u8; n];
    if mx > mn {
        let f = 255.0 / (mx - mn);
        let a = -(mn * f) + 0.5;
        for (i, o) in out.iter_mut().enumerate() {
            *o = (f * sample(i) + a).clamp(0.0, 255.0) as u8;
        }
    }
    Ok(GrayU8 {
        w: area.width,
        h: area.height,
        data: out,
    })
}

/// One tie-point set (`TiePoints`).
#[derive(Clone, Default)]
struct TiePoints {
    xref: Vec<i64>,
    yref: Vec<i64>,
    xsec: Vec<i64>,
    ysec: Vec<i64>,
    correlation: Vec<f64>,
    dx: Vec<f64>,
    dy: Vec<f64>,
    deviation: Vec<f64>,
    l_scale: f64,
    l_angle: f64,
    l_deltax: f64,
    l_deltay: f64,
}

impl TiePoints {
    fn len(&self) -> usize {
        self.xref.len()
    }
}

/// `all_black`: true when the `winsize` window centred at `(x, y)` holds
/// only zeros.
fn all_black(im: &GrayU8, x: i64, y: i64, winsize: i64) -> bool {
    let h = (winsize - 1) / 2;
    for j in 0..winsize {
        for i in 0..winsize {
            if im.at(x - h + i, y - h + j) != 0 {
                return false;
            }
        }
    }
    true
}

/// `calculate_contrast`: sum of absolute horizontal and vertical
/// neighbour differences over the window.
fn window_contrast(im: &GrayU8, x: i64, y: i64, winsize: i64) -> i64 {
    let h = (winsize - 1) / 2;
    let mut total = 0i64;
    for j in 0..winsize - 1 {
        for i in 0..winsize - 1 {
            let p = im.at(x - h + i, y - h + j) as i64;
            let lrd = p - im.at(x - h + i + 1, y - h + j) as i64;
            let tbd = p - im.at(x - h + i, y - h + j + 1) as i64;
            total += lrd.abs() + tbd.abs();
        }
    }
    total
}

/// `vips__find_best_contrast`: the `nbest` highest-contrast window
/// centres on a `hcorsize` grid over the given area.
fn find_best_contrast(
    im: &GrayU8,
    xpos: i64,
    ypos: i64,
    xsize: i64,
    ysize: i64,
    nbest: usize,
    hcorsize: i64,
) -> Result<Vec<(i64, i64)>, MosaicError> {
    let windowsize = 2 * hcorsize + 1;
    let nacross = (xsize - windowsize + hcorsize) / hcorsize;
    let ndown = (ysize - windowsize + hcorsize) / hcorsize;
    if nacross <= 0 || ndown <= 0 {
        return Err(MosaicError::OverlapTooSmall);
    }
    let mut candidates: Vec<(i64, i64, i64)> = Vec::new();
    for y in 0..ndown {
        for x in 0..nacross {
            let left = xpos + x * hcorsize;
            let top = ypos + y * hcorsize;
            if all_black(im, left, top, windowsize) {
                continue;
            }
            candidates.push((left, top, window_contrast(im, left, top, windowsize)));
        }
    }
    if candidates.len() < nbest {
        return Err(MosaicError::TooFewPoints {
            found: candidates.len(),
            needed: nbest,
        });
    }
    // Stable sort: contrast ties keep scan order (the C qsort's tie order
    // is unspecified).
    candidates.sort_by(|l, r| r.2.cmp(&l.2));
    Ok(candidates[..nbest].iter().map(|c| (c.0, c.1)).collect())
}

/// `vips__lrcalcon` / `vips__tbcalcon`: high-contrast reference points in
/// three strips of the overlap.
fn calcon(im: &GrayU8, direction: MergeDirection) -> Result<Vec<(i64, i64)>, MosaicError> {
    let border = HAREA;
    let len = MAXPOINTS / AREAS;
    let mut points = Vec::with_capacity(MAXPOINTS);
    match direction {
        MergeDirection::Horizontal => {
            let aheight = im.h / AREAS as i64;
            let (left, width) = (border, im.w - 2 * border - 1);
            let height = aheight - 2 * border - 1;
            let mut top = border;
            let mut i = 0;
            while top < im.h && i < AREAS {
                points.extend(find_best_contrast(
                    im, left, top, width, height, len, HWINDOW,
                )?);
                top += aheight;
                i += 1;
            }
        }
        MergeDirection::Vertical => {
            let awidth = im.w / AREAS as i64;
            let (top, height) = (border, im.h - 2 * border - 1);
            let width = awidth - 2 * border - 1;
            if width < 0 || height < 0 {
                return Err(MosaicError::OverlapTooSmall);
            }
            let mut left = border;
            let mut i = 0;
            while left < im.w && i < AREAS {
                points.extend(find_best_contrast(
                    im, left, top, width, height, len, HWINDOW,
                )?);
                left += awidth;
                i += 1;
            }
        }
    }
    Ok(points)
}

/// `vips__correl`: search `sec` around `(xsec, ysec)` for the best
/// normalised spatial correlation with the window around `(xref, yref)`
/// in `ref`. Returns `(correlation, x, y)`.
#[allow(clippy::too_many_arguments)]
fn correl(
    ref_im: &GrayU8,
    sec_im: &GrayU8,
    xref: i64,
    yref: i64,
    xsec: i64,
    ysec: i64,
    hwindowsize: i64,
    hsearchsize: i64,
) -> (f64, i64, i64) {
    // Window in ref, clipped to the image.
    let wl = (xref - hwindowsize).max(0);
    let wt = (yref - hwindowsize).max(0);
    let wr = (xref + hwindowsize + 1).min(ref_im.w);
    let wb = (yref + hwindowsize + 1).min(ref_im.h);
    let (ww, wh) = (wr - wl, wb - wt);

    // Search area in sec, clipped to the image.
    let sl = (xsec - hsearchsize).max(0);
    let st = (ysec - hsearchsize).max(0);
    let sr = (xsec + hsearchsize + 1).min(sec_im.w);
    let sb = (ysec + hsearchsize + 1).min(sec_im.h);
    let (sw, sh) = (sr - sl, sb - st);

    // Reference window statistics (spcor pre_generate).
    let n = (ww * wh) as f64;
    let mut sum = 0.0;
    for j in 0..wh {
        for i in 0..ww {
            sum += ref_im.at(wl + i, wt + j) as f64;
        }
    }
    let rmean = sum / n;
    let mut ss = 0.0;
    for j in 0..wh {
        for i in 0..ww {
            let d = ref_im.at(wl + i, wt + j) as f64 - rmean;
            ss += d * d;
        }
    }
    let c1 = ss.sqrt();

    // Correlation surface over the search area, the window centred at
    // each position with edge replication (EXTEND_COPY on the extracted
    // search area).
    let sec_at = |x: i64, y: i64| sec_im.at(sl + x.clamp(0, sw - 1), st + y.clamp(0, sh - 1));
    let mut best = f64::NEG_INFINITY;
    let (mut bx, mut by) = (0i64, 0i64);
    for py in 0..sh {
        for px in 0..sw {
            let (ox, oy) = (px - ww / 2, py - wh / 2);
            let mut sum1 = 0.0;
            for j in 0..wh {
                for i in 0..ww {
                    sum1 += sec_at(ox + i, oy + j) as f64;
                }
            }
            let imean = sum1 / n;
            let mut sum2 = 0.0;
            let mut sum3 = 0.0;
            for j in 0..wh {
                for i in 0..ww {
                    let ip = sec_at(ox + i, oy + j) as f64;
                    let rp = ref_im.at(wl + i, wt + j) as f64;
                    let t = ip - imean;
                    sum2 += t * t;
                    sum3 += (rp - rmean) * t;
                }
            }
            let c2 = c1 * sum2.sqrt();
            let cc = if c2 == 0.0 { 0.0 } else { sum3 / c2 };
            if cc > best {
                best = cc;
                bx = px;
                by = py;
            }
        }
    }
    (best, bx + sl, by + st)
}

/// `vips__chkpair`: correlate every reference point into the secondary.
fn chkpair(ref_im: &GrayU8, sec_im: &GrayU8, refs: &[(i64, i64)]) -> TiePoints {
    let mut p = TiePoints::default();
    for &(x, y) in refs {
        let (correlation, xs, ys) = correl(ref_im, sec_im, x, y, x, y, HWINDOW, HAREA);
        p.xref.push(x);
        p.yref.push(y);
        p.xsec.push(xs);
        p.ysec.push(ys);
        p.correlation.push(correlation);
        p.dx.push((xs - x) as f64);
        p.dy.push((ys - y) as f64);
        p.deviation.push(0.0);
    }
    p
}

/// Invert an `n` x `n` matrix (row-major) by Gauss-Jordan elimination with
/// partial pivoting. `None` when singular.
fn mat_invert(mut a: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    let mut inv = vec![0.0; n * n];
    for d in 0..n {
        inv[d * n + d] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if a[row * n + col].abs() > a[pivot * n + col].abs() {
                pivot = row;
            }
        }
        if a[pivot * n + col] == 0.0 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
                inv.swap(col * n + k, pivot * n + k);
            }
        }
        let p = a[col * n + col];
        for k in 0..n {
            a[col * n + k] /= p;
            inv[col * n + k] /= p;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = a[row * n + col];
            if f == 0.0 {
                continue;
            }
            for k in 0..n {
                a[row * n + k] -= f * a[col * n + k];
                inv[row * n + k] -= f * inv[col * n + k];
            }
        }
    }
    Some(inv)
}

/// `vips__clinear`: fit the first-order model and fill dx/dy/deviation.
/// `false` when the fit matrix is singular (the caller falls back).
fn clinear(p: &mut TiePoints) -> bool {
    let elms = p.len() as f64;
    let (mut sx1, mut sx1x1, mut sy1, mut sy1y1) = (0.0, 0.0, 0.0, 0.0);
    let (mut sx2x1, mut sx2y1, mut sy2y1, mut sy2x1, mut sx2, mut sy2) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..p.len() {
        let (xr, yr) = (p.xref[i] as f64, p.yref[i] as f64);
        let (xs, ys) = (p.xsec[i] as f64, p.ysec[i] as f64);
        sx1 += xr;
        sx1x1 += xr * xr;
        sy1 += yr;
        sy1y1 += yr * yr;
        sx2x1 += xs * xr;
        sx2y1 += xs * yr;
        sy2y1 += ys * yr;
        sy2x1 += ys * xr;
        sx2 += xs;
        sy2 += ys;
    }
    #[rustfmt::skip]
    let mat = vec![
        sx1x1 + sy1y1, 0.0,           sx1,  sy1,
        0.0,           sx1x1 + sy1y1, -sy1, sx1,
        sx1,           -sy1,          elms, 0.0,
        sy1,           sx1,           0.0,  elms,
    ];
    let g = [sx2x1 + sy2y1, -sx2y1 + sy2x1, sx2, sy2];
    let Some(inv) = mat_invert(mat, 4) else {
        return false;
    };
    let mut sol = [0.0f64; 4];
    for (r, s) in sol.iter_mut().enumerate() {
        // The fit matrix is symmetric, so reading the inverse column-wise
        // as the C does equals the usual row-wise product.
        for j in 0..4 {
            *s += inv[j * 4 + r] * g[j];
        }
    }
    let (scale, angle, xdelta, ydelta) = (sol[0], sol[1], sol[2], sol[3]);
    for i in 0..p.len() {
        let (xr, yr) = (p.xref[i] as f64, p.yref[i] as f64);
        p.dx[i] = p.xsec[i] as f64 - ((scale * xr) - (angle * yr) + xdelta);
        p.dy[i] = p.ysec[i] as f64 - ((angle * xr) + (scale * yr) + ydelta);
        p.deviation[i] = (p.dx[i] * p.dx[i] + p.dy[i] * p.dy[i]).sqrt();
    }
    p.l_scale = scale;
    p.l_angle = angle;
    p.l_deltax = xdelta;
    p.l_deltay = ydelta;
    true
}

/// `vips__initialize`: clinear, or the correlation-weighted average
/// fallback when the fit matrix is singular.
fn initialize(p: &mut TiePoints) -> Result<(), MosaicError> {
    if clinear(p) {
        return Ok(());
    }
    let max_cor = p.correlation.iter().cloned().fold(0.0f64, f64::max) - 0.04;
    let mut xdelta = 0.0;
    let mut ydelta = 0.0;
    let mut j = 0usize;
    for i in 0..p.len() {
        if p.correlation[i] >= max_cor {
            xdelta += (p.xsec[i] - p.xref[i]) as f64;
            ydelta += (p.ysec[i] - p.yref[i]) as f64;
            j += 1;
        }
    }
    if j == 0 {
        return Err(MosaicError::NoTiePoints);
    }
    xdelta /= j as f64;
    ydelta /= j as f64;
    for i in 0..p.len() {
        p.dx[i] = (p.xsec[i] - p.xref[i]) as f64 - xdelta;
        p.dy[i] = (p.ysec[i] - p.yref[i]) as f64 - ydelta;
        p.deviation[i] = (p.dx[i] * p.dx[i] + p.dy[i] * p.dy[i]).sqrt();
    }
    p.l_scale = 1.0;
    p.l_angle = 0.0;
    p.l_deltax = xdelta;
    p.l_deltay = ydelta;
    Ok(())
}

/// `copydevpoints`: keep points whose deviation/correlation ratio is
/// within the adaptive threshold; correlation must exceed 0.01.
fn copydevpoints(p: &TiePoints) -> TiePoints {
    let mut min_dev = 9999.0f64;
    let mut max_dev = 0.0f64;
    for i in 0..p.len() {
        if p.correlation[i] > 0.01 {
            let r = p.deviation[i] / p.correlation[i];
            min_dev = min_dev.min(r);
            max_dev = max_dev.max(r);
        }
    }
    let thresh_dev = (min_dev + (max_dev - min_dev) * 0.3).max(1.0);
    let mut q = TiePoints {
        l_scale: p.l_scale,
        l_angle: p.l_angle,
        l_deltax: p.l_deltax,
        l_deltay: p.l_deltay,
        ..TiePoints::default()
    };
    for i in 0..p.len() {
        if p.correlation[i] > 0.01 && p.deviation[i] / p.correlation[i] <= thresh_dev {
            q.xref.push(p.xref[i]);
            q.yref.push(p.yref[i]);
            q.xsec.push(p.xsec[i]);
            q.ysec.push(p.ysec[i]);
            q.correlation.push(p.correlation[i]);
            q.dx.push(p.dx[i]);
            q.dy.push(p.dy[i]);
            q.deviation.push(p.deviation[i]);
        }
    }
    q
}

/// `vips__improve`: repeatedly discard outliers and refit.
fn improve(points: TiePoints) -> Result<TiePoints, MosaicError> {
    let mut p = points;
    loop {
        let mut q = copydevpoints(&p);
        if q.len() == p.len() {
            return Ok(q);
        }
        if q.len() < 2 {
            return Ok(q);
        }
        if !clinear(&mut q) {
            return Err(MosaicError::SingularMatrix);
        }
        p = q;
    }
}

/// `vips__avgdxdy`: round-half-to-even average of the surviving
/// displacements.
fn avgdxdy(p: &TiePoints) -> Result<(i64, i64), MosaicError> {
    if p.xref.is_empty() {
        return Err(MosaicError::NoTiePoints);
    }
    let mut sumdx = 0i64;
    let mut sumdy = 0i64;
    for i in 0..p.len() {
        sumdx += p.xsec[i] - p.xref[i];
        sumdy += p.ysec[i] - p.yref[i];
    }
    let n = p.len() as f64;
    Ok((
        (sumdx as f64 / n).round_ties_even() as i64,
        (sumdy as f64 / n).round_ties_even() as i64,
    ))
}

/// `vips__find_lroverlap` / `vips__find_tboverlap`: refine the tie-point
/// guess into the merge displacement.
#[allow(clippy::too_many_arguments)]
fn find_overlap_offsets(
    ref_: &Raster,
    sec: &Raster,
    direction: MergeDirection,
    xref: i64,
    yref: i64,
    xsec: i64,
    ysec: i64,
) -> Result<(i64, i64), MosaicError> {
    let left = Rect {
        left: 0,
        top: 0,
        width: ref_.width() as i64,
        height: ref_.height() as i64,
    };
    let right = Rect {
        left: xref - xsec,
        top: yref - ysec,
        width: sec.width() as i64,
        height: sec.height() as i64,
    };
    let overlap = left.intersect(&right);
    if overlap.width < 2 * HAREA + 1 || overlap.height < 2 * HAREA + 1 {
        return Err(MosaicError::OverlapTooSmall);
    }

    let ref_ov = scale_band0(ref_, overlap)?;
    let sec_ov = scale_band0(
        sec,
        Rect {
            left: overlap.left - right.left,
            top: overlap.top - right.top,
            width: overlap.width,
            height: overlap.height,
        },
    )?;

    let refs = calcon(&ref_ov, direction)?;
    let mut points = chkpair(&ref_ov, &sec_ov, &refs);
    initialize(&mut points)?;
    let survivors = improve(points)?;
    let (dx, dy) = avgdxdy(&survivors)?;

    Ok((-right.left + dx, -right.top + dy))
}

// ---------------------------------------------------------------------------
// Global balance (global_balance.c)
// ---------------------------------------------------------------------------

/// A leaf image with its rectangle in final mosaic coordinates.
struct PlacedLeaf {
    raster: Raster,
    rect: Rect,
}

/// Size of the image a tree node renders to (calc_geometry).
fn tree_size(node: &JoinTree) -> (i64, i64) {
    match node {
        JoinTree::Leaf(r) => (r.width() as i64, r.height() as i64),
        JoinTree::Join {
            dx,
            dy,
            ref_node,
            sec_node,
            ..
        } => {
            let (rw, rh) = tree_size(ref_node);
            let (sw, sh) = tree_size(sec_node);
            let rarea = Rect {
                left: 0,
                top: 0,
                width: rw,
                height: rh,
            };
            let sarea = Rect {
                left: -(*dx as i64),
                top: -(*dy as i64),
                width: sw,
                height: sh,
            };
            let u = rarea.union(&sarea);
            (u.width, u.height)
        }
    }
}

/// Collect leaves depth-first (reference before secondary) with their
/// final-output rectangles.
fn collect_leaves(node: &JoinTree, ox: i64, oy: i64, out: &mut Vec<PlacedLeaf>) {
    match node {
        JoinTree::Leaf(r) => out.push(PlacedLeaf {
            raster: r.clone(),
            rect: Rect {
                left: ox,
                top: oy,
                width: r.width() as i64,
                height: r.height() as i64,
            },
        }),
        JoinTree::Join {
            dx,
            dy,
            ref_node,
            sec_node,
            ..
        } => {
            let (rw, rh) = tree_size(ref_node);
            let (sw, sh) = tree_size(sec_node);
            let rarea = Rect {
                left: 0,
                top: 0,
                width: rw,
                height: rh,
            };
            let sarea = Rect {
                left: -(*dx as i64),
                top: -(*dy as i64),
                width: sw,
                height: sh,
            };
            let u = rarea.union(&sarea);
            collect_leaves(ref_node, ox - u.left, oy - u.top, out);
            collect_leaves(
                sec_node,
                ox - (*dx as i64) - u.left,
                oy - (*dy as i64) - u.top,
                out,
            );
        }
    }
}

/// Read the flat element `i` of a raster as `f64` (any depth).
fn elem_f64(r: &Raster, i: usize) -> f64 {
    let data = r.data();
    match r.format().bytes_per_channel() {
        1 => data[i] as f64,
        2 => u16::from_ne_bytes([data[i * 2], data[i * 2 + 1]]) as f64,
        _ => f32::from_ne_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
        ]) as f64,
    }
}

/// One overlap row of the balance system: the masked-mean statistics of
/// the two leaves over their shared area (find_overlap_stats).
struct OverlapStat {
    node: usize,
    other: usize,
    nstat: f64,
    ostat: f64,
}

fn overlap_stats(a: &PlacedLeaf, b: &PlacedLeaf, overlap: &Rect) -> Option<(f64, f64)> {
    let bands = a.raster.format().channels();
    let n_pels = (overlap.width * overlap.height) as f64;
    let mut nnz = 0u64;
    let mut sum_n = 0.0f64;
    let mut sum_o = 0.0f64;
    for y in overlap.top..overlap.bottom() {
        for x in overlap.left..overlap.right() {
            let ae = (((y - a.rect.top) * a.rect.width + (x - a.rect.left)) as usize) * bands;
            let be = (((y - b.rect.top) * b.rect.width + (x - b.rect.left)) as usize) * bands;
            for c in 0..bands {
                let nv = elem_f64(&a.raster, ae + c);
                let ov = elem_f64(&b.raster, be + c);
                if nv != 0.0 && ov != 0.0 {
                    nnz += 1;
                    sum_n += nv;
                    sum_o += ov;
                }
            }
        }
    }
    // count_nonzero mirrors the C double round-trip through the 0/255
    // mask average, including its truncation to integer.
    let avg_mask = 255.0 * nnz as f64 / (n_pels * bands as f64);
    let count = ((avg_mask * n_pels) / 255.0) as i64;
    if (count as f64) < TRIVIAL as f64 {
        return None;
    }
    let scale = count as f64 / n_pels;
    let nstat = (sum_n / (n_pels * bands as f64)) * scale;
    let ostat = (sum_o / (n_pels * bands as f64)) * scale;
    Some((nstat, ostat))
}

/// `find_factors`: least-mean-square balance factors, anchored on leaf 0
/// and normalised to a mean of 1.0.
fn find_factors(stats: &[OverlapStat], nim: usize, gamma: f64) -> Result<Vec<f64>, MosaicError> {
    let novl = stats.len();
    let cols = nim - 1;
    if novl == 0 || cols == 0 {
        return Err(MosaicError::SingularMatrix);
    }
    let mut k = vec![0.0f64; novl];
    let mut m = vec![0.0f64; novl * cols];
    for (row, s) in stats.iter().enumerate() {
        let ns = s.nstat.powf(1.0 / gamma);
        let os = s.ostat.powf(1.0 / gamma);
        if s.node == 0 {
            k[row] = ns;
            m[row * cols + (s.other - 1)] = os;
        } else {
            m[row * cols + (s.node - 1)] = -ns;
            m[row * cols + (s.other - 1)] = os;
        }
    }
    // fac[1..] = (M^T M)^-1 M^T K.
    let mut mtm = vec![0.0f64; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0;
            for (r, _) in stats.iter().enumerate() {
                sum += m[r * cols + i] * m[r * cols + j];
            }
            mtm[i * cols + j] = sum;
        }
    }
    let inv = mat_invert(mtm, cols).ok_or(MosaicError::SingularMatrix)?;
    let mut mtk = vec![0.0f64; cols];
    for (i, v) in mtk.iter_mut().enumerate() {
        for (r, kv) in k.iter().enumerate() {
            *v += m[r * cols + i] * kv;
        }
    }
    let mut fac = vec![1.0f64; nim];
    for i in 0..cols {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += inv[i * cols + j] * mtk[j];
        }
        fac[i + 1] = sum;
    }
    let avg = fac.iter().sum::<f64>() / nim as f64;
    for f in &mut fac {
        *f /= avg;
    }
    Ok(fac)
}

/// `transformf`: scale a leaf by its factor into a float raster. 8/16-bit
/// depths go through the gamma lookup table
/// `v -> (v^(1/gamma) * fac)^gamma`; float depths scale linearly; a
/// factor of exactly 1.0 is a plain float cast.
fn transform_leaf(leaf: &Raster, fac: f64, gamma: f64) -> Result<Raster, MosaicError> {
    let bands = leaf.format().channels();
    let out_format =
        PixelFormat::with_channels(bands, 4).ok_or(MosaicError::CorruptMosaicMetadata)?;
    let mut out = Raster::zeroed(leaf.width(), leaf.height(), out_format)?;
    let n = leaf.width() as usize * leaf.height() as usize * bands;
    let bpc = leaf.format().bytes_per_channel();
    let lut: Option<Vec<f32>> = if fac == 1.0 || bpc == 4 {
        None
    } else {
        let entries = if bpc == 1 { 256 } else { 65536 };
        Some(
            (0..entries)
                .map(|v| ((v as f64).powf(1.0 / gamma) * fac).powf(gamma) as f32)
                .collect(),
        )
    };
    let out_data = out.data_mut();
    for i in 0..n {
        let v = elem_f64(leaf, i);
        let t: f32 = if fac == 1.0 {
            v as f32
        } else if let Some(lut) = &lut {
            lut[v as usize]
        } else {
            (v * fac) as f32
        };
        out_data[i * 4..i * 4 + 4].copy_from_slice(&t.to_ne_bytes());
    }
    Ok(out)
}

/// `make_mos_image` / `vips__build_mosaic`: re-merge the tree over the
/// transformed float leaves.
fn rebuild(
    node: &JoinTree,
    facs: &[f64],
    gamma: f64,
    next_leaf: &mut usize,
) -> Result<Raster, MosaicError> {
    match node {
        JoinTree::Leaf(r) => {
            let fac = facs[*next_leaf];
            *next_leaf += 1;
            transform_leaf(r, fac, gamma)
        }
        JoinTree::Join {
            direction,
            dx,
            dy,
            mblend,
            ref_node,
            sec_node,
        } => {
            let im1 = rebuild(ref_node, facs, gamma, next_leaf)?;
            let im2 = rebuild(sec_node, facs, gamma, next_leaf)?;
            merge_impl(
                &im1,
                &im2,
                *direction,
                *dx as i64,
                *dy as i64,
                *mblend as i64,
            )
        }
    }
}

/// Solve the least-mean-square balance for the join tree serialized in `blob`
/// and re-render the mosaic with the resulting per-leaf factors.
///
/// Split out of [`Raster::try_global_balance`] so the recursive tree walk can
/// run on a worker thread with a large stack (see `GLOBAL_BALANCE_STACK_BYTES`).
fn balance_join_tree(blob: &[u8]) -> Result<Raster, MosaicError> {
    let tree = JoinTree::deserialize(blob)?;

    let mut leaves = Vec::new();
    collect_leaves(&tree, 0, 0, &mut leaves);
    let nim = leaves.len();

    // Every leaf pair with a significant overlap contributes one equation
    // (test_overlap).
    let mut stats = Vec::new();
    for i in 0..nim {
        for j in i + 1..nim {
            let overlap = leaves[i].rect.intersect(&leaves[j].rect);
            if overlap.is_empty() || overlap.width * overlap.height < TRIVIAL {
                continue;
            }
            if let Some((nstat, ostat)) = overlap_stats(&leaves[i], &leaves[j], &overlap) {
                stats.push(OverlapStat {
                    node: i,
                    other: j,
                    nstat,
                    ostat,
                });
            }
        }
    }

    let facs = find_factors(&stats, nim, BALANCE_GAMMA)?;
    let mut next_leaf = 0usize;
    rebuild(&tree, &facs, BALANCE_GAMMA, &mut next_leaf)
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

impl Raster {
    /// Merge, the fallible form of [`Raster::merge`].
    ///
    /// # Errors
    ///
    /// Returns [`MosaicError::FormatMismatch`] when the images differ in
    /// pixel format, [`MosaicError::NoOverlap`] when the displaced images
    /// share no pixels, and [`MosaicError::TooMuchOverlap`] when `self`
    /// extends past `other` on the trailing edge.
    pub fn try_merge(
        &self,
        other: &Raster,
        direction: MergeDirection,
        dx: i32,
        dy: i32,
    ) -> Result<Raster, MosaicError> {
        merge_impl(self, other, direction, dx as i64, dy as i64, DEFAULT_MBLEND)
    }

    /// Merge two overlapping images with a feathered seam (libvips
    /// `vips_merge` with the default `mblend` of 10).
    ///
    /// `self` is the reference image at `(0, 0)` (left for
    /// [`MergeDirection::Horizontal`], top for
    /// [`MergeDirection::Vertical`]); `other` is displaced to
    /// `(-dx, -dy)`, so a left-right join with a 10-pixel overlap is
    /// `dx = 10 - self.width()`. The output covers the union of the two
    /// placed images; zero pixels are transparent in the seam. The result
    /// carries the metadata [`Raster::global_balance`] needs.
    ///
    /// # Panics
    ///
    /// Panics on mismatched formats or impossible geometry; use
    /// [`Raster::try_merge`] to handle those as errors.
    pub fn merge(&self, other: &Raster, direction: MergeDirection, dx: i32, dy: i32) -> Raster {
        expect_mosaic("merge", self.try_merge(other, direction, dx, dy))
    }

    /// Mosaic, the fallible form of [`Raster::mosaic`].
    ///
    /// # Errors
    ///
    /// Returns [`MosaicError::OverlapTooSmall`] when the tie-point
    /// estimate leaves less overlap than the search geometry needs,
    /// [`MosaicError::TooFewPoints`] when a search strip has too few
    /// usable high-contrast windows, [`MosaicError::NoTiePoints`] /
    /// [`MosaicError::SingularMatrix`] when the fit fails, plus every
    /// [`Raster::try_merge`] error for the final join.
    pub fn try_mosaic(
        &self,
        other: &Raster,
        direction: MergeDirection,
        ref_x: i32,
        ref_y: i32,
        sec_x: i32,
        sec_y: i32,
    ) -> Result<Raster, MosaicError> {
        if self.format() != other.format() {
            return Err(MosaicError::FormatMismatch {
                reference: self.format(),
                secondary: other.format(),
            });
        }
        let (dx0, dy0) = find_overlap_offsets(
            self,
            other,
            direction,
            ref_x as i64,
            ref_y as i64,
            sec_x as i64,
            sec_y as i64,
        )?;
        merge_impl(self, other, direction, dx0, dy0, DEFAULT_MBLEND)
    }

    /// Join two images given an approximate tie-point (libvips
    /// `vips_mosaic` with the default `bandno` 0, `hwindow` 5, `harea`
    /// 15, `mblend` 10).
    ///
    /// `other` is positioned so that its pixel `(sec_x, sec_y)` lands on
    /// `(ref_x, ref_y)` in `self`; the exact displacement is then refined
    /// by correlating sixty high-contrast points of the estimated overlap
    /// and the images are joined with [`Raster::merge`] semantics.
    ///
    /// # Panics
    ///
    /// Panics when the overlap is too small to search, too few
    /// high-contrast tie-point candidates exist, or the fit fails; use
    /// [`Raster::try_mosaic`] to handle those as errors.
    pub fn mosaic(
        &self,
        other: &Raster,
        direction: MergeDirection,
        ref_x: i32,
        ref_y: i32,
        sec_x: i32,
        sec_y: i32,
    ) -> Raster {
        expect_mosaic(
            "mosaic",
            self.try_mosaic(other, direction, ref_x, ref_y, sec_x, sec_y),
        )
    }

    /// Global balance, the fallible form of [`Raster::global_balance`].
    ///
    /// # Errors
    ///
    /// Returns [`MosaicError::NoMosaicMetadata`] when the raster was not
    /// built by [`Raster::merge`] / [`Raster::mosaic`],
    /// [`MosaicError::CorruptMosaicMetadata`] when the join metadata does
    /// not deserialize, and [`MosaicError::SingularMatrix`] when the
    /// factor system cannot be solved (for example when every overlap is
    /// smaller than the 20x20 significance threshold).
    pub fn try_global_balance(&self) -> Result<Raster, MosaicError> {
        let blob = match self.get_field(JOIN_TREE_FIELD) {
            Some(MetadataValue::Blob(blob)) => blob,
            Some(_) => return Err(MosaicError::CorruptMosaicMetadata),
            None => return Err(MosaicError::NoMosaicMetadata),
        };
        // The tree walk (`collect_leaves` / `rebuild`) recurses once per mosaic
        // tile, so a deep long-strip mosaic would overflow the default stack now
        // that the join-tree parse is uncapped. Run it on a worker thread with a
        // large explicit stack; the computation and result are identical to
        // running inline (see `GLOBAL_BALANCE_STACK_BYTES`).
        std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(GLOBAL_BALANCE_STACK_BYTES)
                .spawn_scoped(scope, || balance_join_tree(&blob))
                .expect("spawn global-balance worker thread")
                .join()
                .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
        })
    }

    /// Remove brightness differences across an assembled mosaic (libvips
    /// `vips_globalbalance` with the default gamma of 1.6 and float
    /// output).
    ///
    /// The input must have been built by [`Raster::merge`] /
    /// [`Raster::mosaic`], whose outputs carry the join metadata this
    /// operation rebuilds from. Each source image is scaled by a
    /// least-mean-square balance factor computed from the leaf overlap
    /// statistics, and the mosaic is reassembled in float with the same
    /// geometry as the input.
    ///
    /// # Panics
    ///
    /// Panics when the raster carries no mosaic metadata or the factor
    /// system cannot be solved; use [`Raster::try_global_balance`] to
    /// handle those as errors.
    pub fn global_balance(&self) -> Raster {
        expect_mosaic("global_balance", self.try_global_balance())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(w: u32, h: u32, v: u8) -> Raster {
        Raster::new(w, h, PixelFormat::Gray8, vec![v; (w * h) as usize]).unwrap()
    }

    /// Deterministic high-contrast texture: a hash of the scene
    /// coordinates mapped into 30..=225 so no pixel is transparent.
    fn textured(w: u32, h: u32, ox: u32, oy: u32) -> Raster {
        let mut data = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let mut v = (x + ox).wrapping_mul(0x9E37_79B9) ^ (y + oy).wrapping_mul(0x85EB_CA6B);
                v ^= v >> 13;
                v = v.wrapping_mul(0xC2B2_AE35);
                v ^= v >> 16;
                data.push(30 + (v % 196) as u8);
            }
        }
        Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
    }

    /**
     * Compile-position pin for the exact ported-test call surface of the
     * mosaicing batch (ported_mosaicing.rs): merge and mosaic with the
     * MergeDirection enum and i32 offsets, and global_balance.
     * Works by invoking every call site shape the ported tests use.
     */
    #[test]
    fn ported_call_surface_compiles() {
        let left = textured(80, 60, 0, 0);
        let right = textured(80, 60, 70, 0);
        let dx = 10 - left.width() as i32;
        let join: Raster = left.merge(&right, MergeDirection::Horizontal, dx, 0);
        assert_eq!(join.width(), left.width() + right.width() - 10);
        assert_eq!(join.height(), left.height().max(right.height()));
        let balanced: Raster = join.global_balance();
        assert_eq!(balanced.format().channels(), 1);
        let _ = |a: &Raster, b: &Raster| -> Raster {
            a.mosaic(
                b,
                MergeDirection::Horizontal,
                a.width() as i32 - 30,
                0,
                30,
                0,
            )
        };
    }

    /**
     * Tests left-right merge geometry, the ported test_lrmerge contract:
     * a 10-pixel overlap yields width w1 + w2 - 10 and height
     * max(h1, h2), and the ref-only / sec-only regions carry their source
     * pixels unchanged.
     * Works on two textured images of different heights.
     */
    #[test]
    fn lrmerge_geometry_and_content() {
        let left = textured(60, 50, 0, 0);
        let right = textured(60, 40, 50, 0);
        let dx = 10 - left.width() as i32;
        let join = left.merge(&right, MergeDirection::Horizontal, dx, 0);
        assert_eq!(join.width(), 110);
        assert_eq!(join.height(), 50);
        assert_eq!(join.format(), PixelFormat::Gray8);
        // Far left: pure ref; far right: pure sec.
        assert_eq!(join.getpoint(0, 0), left.getpoint(0, 0));
        assert_eq!(join.getpoint(20, 30), left.getpoint(20, 30));
        assert_eq!(join.getpoint(109, 10), right.getpoint(59, 10));
        // Below sec's extent in sec-only columns: black.
        assert_eq!(join.getpoint(109, 45)[0], 0.0);
    }

    /**
     * Tests top-bottom merge geometry, the ported test_tbmerge contract:
     * height h1 + h2 - overlap, width max(w1, w2).
     * Works on two textured images of different widths.
     */
    #[test]
    fn tbmerge_geometry() {
        let top = textured(50, 60, 0, 0);
        let bottom = textured(45, 60, 0, 50);
        let dy = 10 - top.height() as i32;
        let join = top.merge(&bottom, MergeDirection::Vertical, 0, dy);
        assert_eq!(join.width(), 50);
        assert_eq!(join.height(), 110);
        assert_eq!(join.getpoint(10, 0), top.getpoint(10, 0));
        assert_eq!(join.getpoint(10, 109), bottom.getpoint(10, 59));
    }

    /**
     * Tests the raised-cosine seam on uniform images: the blend band is
     * the centred mblend-wide strip of the overlap, pure ref left of it,
     * pure sec right of it, and inside it each column matches the exact
     * integer-lut blend the C computes.
     * Works by merging uniform 100 and 200 images with a 30-pixel overlap
     * and recomputing every overlap column with the same formulas.
     */
    #[test]
    fn lrmerge_blend_band_exact() {
        let left = uniform(60, 20, 100);
        let right = uniform(60, 20, 200);
        let dx = 30 - left.width() as i32; // 30-pixel overlap at x 30..60
        let join = left.merge(&right, MergeDirection::Horizontal, dx, 0);
        assert_eq!(join.width(), 90);
        // first = 30, last = 59; wider than mblend 10, so the band
        // shrinks by (29 - 10) / 2 = 9 on each side: [39, 50).
        let (first, last) = (39i64, 50i64);
        for x in 30..60i64 {
            let expected = if x < first {
                100.0
            } else if x >= last {
                200.0
            } else {
                let inx = ((x - first) << BLEND_SHIFT) / (last - first);
                let luts = blend_luts();
                (luts.icoef1[inx as usize] * 100 / BLEND_SCALE
                    + luts.icoef2[inx as usize] * 200 / BLEND_SCALE) as f64
            };
            assert_eq!(
                join.getpoint(x as u32, 10)[0],
                expected,
                "overlap column {x}"
            );
        }
    }

    /**
     * Tests zero-pixel transparency in the seam: where sec is zero inside
     * the overlap the ref pixel shows through unblended, and vice versa.
     * Works by zeroing one sec row inside the overlap and checking that
     * row keeps the ref value across the whole overlap.
     */
    #[test]
    fn lrmerge_zero_is_transparent() {
        let left = uniform(40, 10, 100);
        let mut right = uniform(40, 10, 200);
        // Zero out sec's row 3 over the whole image.
        for x in 0..40 {
            right.put_pixel(x, 3, &[0]);
        }
        let dx = 20 - left.width() as i32;
        let join = left.merge(&right, MergeDirection::Horizontal, dx, 0);
        for x in 20..40u32 {
            assert_eq!(join.getpoint(x, 3)[0], 100.0, "ref shows through at {x}");
        }
        // A normal row still reaches pure sec on the right of the band.
        assert_eq!(join.getpoint(39, 5)[0], 200.0);
    }

    /**
     * Tests the insert fallback: a positive dx cannot be feathered, so
     * the images are pasted into the union unblended, exactly like the
     * vips_insert path.
     * Works with dx = 5 (sec placed left of ref) and checks both source
     * regions and the output size.
     */
    #[test]
    fn merge_insert_fallback() {
        let left = uniform(20, 10, 100);
        let right = uniform(15, 10, 200);
        let join = left.merge(&right, MergeDirection::Horizontal, 5, 0);
        // sec at (-5, 0), ref at (0, 0): union spans -5..20.
        assert_eq!(join.width(), 25);
        assert_eq!(join.height(), 10);
        assert_eq!(join.getpoint(0, 0)[0], 200.0, "sec-only far left");
        assert_eq!(join.getpoint(24, 0)[0], 100.0, "ref-only far right");
        // Overlap region: sec pasted over ref.
        assert_eq!(join.getpoint(7, 0)[0], 200.0);
    }

    /**
     * Tests merge error cases: mismatched formats, disjoint placements
     * (vertical gap), and too much overlap (ref extends past sec).
     * Works by asserting each try_merge error variant.
     */
    #[test]
    fn merge_errors() {
        let a = uniform(20, 10, 10);
        let rgb = Raster::zeroed(20, 10, PixelFormat::Rgb8).unwrap();
        assert!(matches!(
            a.try_merge(&rgb, MergeDirection::Horizontal, -10, 0),
            Err(MosaicError::FormatMismatch { .. })
        ));
        // In-range dx but a vertical displacement pushing sec fully
        // below ref: no overlap.
        assert!(matches!(
            a.try_merge(&a, MergeDirection::Horizontal, -10, -20),
            Err(MosaicError::NoOverlap)
        ));
        // sec narrower than ref at the same origin: ref's right edge
        // passes sec's.
        let narrow = uniform(10, 10, 10);
        assert!(matches!(
            a.try_merge(&narrow, MergeDirection::Horizontal, 0, 0),
            Err(MosaicError::TooMuchOverlap)
        ));
    }

    /**
     * Tests float merging: FloatF32 rasters blend through the
     * double-precision coefficient path and keep the float format, which
     * the global-balance rebuild relies on.
     * Works by merging two uniform float images and checking a blend
     * column against the double lut.
     */
    #[test]
    fn merge_float_path() {
        let n = std::num::NonZeroU16::new(1).unwrap();
        let fmt = PixelFormat::FloatF32(n);
        let mut a = Raster::zeroed(30, 5, fmt).unwrap();
        let mut b = Raster::zeroed(30, 5, fmt).unwrap();
        for v in a.data_mut().chunks_exact_mut(4) {
            v.copy_from_slice(&100.0f32.to_ne_bytes());
        }
        for v in b.data_mut().chunks_exact_mut(4) {
            v.copy_from_slice(&200.0f32.to_ne_bytes());
        }
        let dx = 10 - a.width() as i32;
        let join = a.merge(&b, MergeDirection::Horizontal, dx, 0);
        assert_eq!(join.format(), fmt);
        assert_eq!(join.width(), 50);
        // Overlap 20..30, band [24, 30... shrink (9 - 10)? width 10 - 1 =
        // 9 <= 10: no shrink; first 20, last 29.
        let luts = blend_luts();
        let x = 24i64;
        let inx = ((x - 20) << BLEND_SHIFT) / 9;
        let expected = (luts.coef1[inx as usize] * 100.0 + luts.coef2[inx as usize] * 200.0) as f32;
        assert_eq!(join.getpoint(x as u32, 2)[0], expected as f64);
    }

    /**
     * Tests the full mosaic search pipeline recovers a known true offset:
     * two crops of one textured scene are mosaiced with a tie-point guess
     * off by (+3, -2), and the correlation search must find the exact
     * placement, giving the geometry of the true overlap.
     * Works by comparing output dimensions and a sec-only pixel against
     * the scene.
     */
    #[test]
    fn lrmosaic_recovers_true_offset() {
        // Scene 300x240; ref = left 160 columns, sec = 200 columns
        // starting at (100, 3).
        let ref_ = textured(160, 240, 0, 0);
        let sec = textured(200, 234, 100, 3);
        // Truth: sec pixel (30, 117) == scene (130, 120) == ref (130, 120).
        // Guess passes (27, 119) instead: 3 columns left, 2 rows down.
        let join = ref_.mosaic(&sec, MergeDirection::Horizontal, 130, 120, 27, 119);
        assert_eq!(join.width(), 300, "true sec placement at x=100");
        assert_eq!(join.height(), 240, "sec fits inside ref rows 3..237");
        // A pixel right of the overlap comes straight from sec at the
        // true offset.
        assert_eq!(join.getpoint(299, 100), sec.getpoint(199, 97));
    }

    /**
     * Tests the vertical mosaic search: same scene-crop construction
     * top-bottom with a guess off by (-2, +4).
     * Works by checking the output dimensions equal the true union.
     */
    #[test]
    fn tbmosaic_recovers_true_offset() {
        let ref_ = textured(240, 160, 0, 0);
        let sec = textured(234, 200, 3, 100);
        // Truth: sec pixel (117, 30) == scene (120, 130) == ref (120, 130).
        let join = ref_.mosaic(&sec, MergeDirection::Vertical, 120, 130, 119, 26);
        assert_eq!(join.width(), 240);
        assert_eq!(join.height(), 300);
    }

    /**
     * Tests mosaic search failure modes: an overlap estimate smaller than
     * the 31-pixel search square is OverlapTooSmall, and an overlap
     * without contrast (uniform images) has no usable tie-points.
     * Works by asserting the try_mosaic error variants.
     */
    #[test]
    fn mosaic_errors() {
        let a = textured(100, 100, 0, 0);
        let b = textured(100, 100, 90, 0);
        // Guess placing sec with only a 10-column overlap.
        assert!(matches!(
            a.try_mosaic(&b, MergeDirection::Horizontal, 95, 50, 5, 50),
            Err(MosaicError::OverlapTooSmall)
        ));
        // A uniform overlap has no high-contrast windows at all: every
        // candidate is skipped as all-black after contrast stretching.
        let ua = uniform(200, 200, 100);
        let ub = uniform(200, 200, 100);
        assert!(matches!(
            ua.try_mosaic(&ub, MergeDirection::Horizontal, 150, 100, 50, 100),
            Err(MosaicError::TooFewPoints { .. })
        ));
    }

    /**
     * Tests global balance on a two-image mosaic of uniform brightness
     * 100 and 150: the factor system equalises the two sides, so after
     * balancing the ref-only and sec-only regions agree, at the value the
     * closed-form factors predict, and the output is float with the
     * mosaic's geometry.
     * Works by recomputing the expected factors analytically (single
     * overlap row: f2 = (100/150)^(1/gamma), normalised to mean 1) and
     * comparing balanced pixels.
     */
    #[test]
    fn global_balance_two_image_mosaic() {
        let left = uniform(60, 40, 100);
        let right = uniform(60, 40, 150);
        let dx = 10 - left.width() as i32;
        let join = left.merge(&right, MergeDirection::Horizontal, dx, 0);
        assert_eq!(join.width(), 110);

        let balanced = join.global_balance();
        assert_eq!(balanced.width(), 110);
        assert_eq!(balanced.height(), 40);
        assert_eq!(balanced.format().channels(), 1);
        assert!(balanced.format().is_float(), "output must be float");

        let gamma = BALANCE_GAMMA;
        let f2 = (100.0f64).powf(1.0 / gamma) / (150.0f64).powf(1.0 / gamma);
        let avg = (1.0 + f2) / 2.0;
        let expected_ref = 100.0 * (1.0 / avg).powf(gamma);
        let expected_sec = 150.0 * (f2 / avg).powf(gamma);

        let got_ref = balanced.getpoint(5, 20)[0];
        let got_sec = balanced.getpoint(105, 20)[0];
        assert!(
            (got_ref - expected_ref).abs() < 1e-3,
            "ref side {got_ref} vs {expected_ref}"
        );
        assert!(
            (got_sec - expected_sec).abs() < 1e-3,
            "sec side {got_sec} vs {expected_sec}"
        );
        assert!(
            (got_ref - got_sec).abs() < 1e-3,
            "balancing equalises the two uniform sides"
        );
    }

    /**
     * Tests that global balance requires mosaic metadata: a plain raster
     * is NoMosaicMetadata, and a corrupted blob is CorruptMosaicMetadata.
     * Works by asserting the try_global_balance error variants.
     */
    #[test]
    fn global_balance_metadata_errors() {
        let plain = uniform(10, 10, 50);
        assert!(matches!(
            plain.try_global_balance(),
            Err(MosaicError::NoMosaicMetadata)
        ));
        let mut bad = uniform(10, 10, 50);
        bad.set_field(JOIN_TREE_FIELD, MetadataValue::Blob(vec![1, 2, 3]));
        assert!(matches!(
            bad.try_global_balance(),
            Err(MosaicError::CorruptMosaicMetadata)
        ));
    }

    /**
     * Tests that a very deep left-deep join tree — the shape a long linear
     * mosaic produces — deserialises without a depth cap and without
     * overflowing the stack, matching vips which balances arbitrarily deep
     * mosaics. `merge` nests each new join's reference child, so a stitch of
     * `N + 1` tiles is a spine of `N` joins; `N` here is far past both the old
     * 512 cap and the recursive stack-overflow threshold (a few thousand
     * frames on a 2 MiB test-thread stack). Works by emitting the tree's
     * preorder bytes directly (N join headers then N + 1 one-pixel leaves) so
     * the blob itself is never built through the recursive `serialize`.
     */
    #[test]
    fn deep_join_tree_deserializes_without_stack_overflow() {
        const N: usize = 50_000;
        let mut blob = b"VMJ1".to_vec();
        for _ in 0..N {
            blob.push(1); // Join tag
            blob.push(0); // MergeDirection::Horizontal
            blob.extend_from_slice(&0i32.to_le_bytes()); // dx
            blob.extend_from_slice(&0i32.to_le_bytes()); // dy
            blob.extend_from_slice(&0i32.to_le_bytes()); // mblend
        }
        for _ in 0..(N + 1) {
            blob.push(0); // Leaf tag
            blob.extend_from_slice(&1u32.to_le_bytes()); // width
            blob.extend_from_slice(&1u32.to_le_bytes()); // height
            blob.extend_from_slice(&1u16.to_le_bytes()); // channels
            blob.push(1); // bytes per channel
            blob.push(0); // one pixel
        }

        let tree = JoinTree::deserialize(&blob).expect("deep linear mosaic deserialises");
        // Walk the reference spine iteratively (recursion here would overflow
        // just as the old parser did) and confirm the full depth survived.
        let mut depth = 0usize;
        let mut node = &tree;
        while let JoinTree::Join { ref_node, .. } = node {
            depth += 1;
            node = ref_node;
        }
        assert_eq!(depth, N, "every join level round-trips, no depth cap");
    }

    /**
     * Tests that a truncated deeply-nested blob still fails with a clean typed
     * error rather than overflowing the stack: an all-`Join` header run with no
     * leaves runs the cursor dry, and the iterative parse reports
     * `CorruptMosaicMetadata` both directly and through global balance's
     * metadata path (the frame stack lives on the heap, so it cannot abort).
     */
    #[test]
    fn deep_truncated_join_tree_blob_errors_cleanly() {
        let mut blob = b"VMJ1".to_vec();
        for _ in 0..50_000 {
            blob.push(1); // Join tag
            blob.push(0); // MergeDirection::Horizontal
            blob.extend_from_slice(&0i32.to_le_bytes()); // dx
            blob.extend_from_slice(&0i32.to_le_bytes()); // dy
            blob.extend_from_slice(&0i32.to_le_bytes()); // mblend
        }
        assert!(matches!(
            JoinTree::deserialize(&blob),
            Err(MosaicError::CorruptMosaicMetadata)
        ));

        let mut raster = uniform(10, 10, 50);
        raster.set_field(JOIN_TREE_FIELD, MetadataValue::Blob(blob));
        assert!(matches!(
            raster.try_global_balance(),
            Err(MosaicError::CorruptMosaicMetadata)
        ));
    }

    /**
     * Tests the join-tree metadata round trip: a merge of a merge
     * serialises a three-leaf tree that deserialises back to the same
     * geometry, sizes, and pixel data.
     * Works by building a small two-join mosaic and walking the decoded
     * tree.
     */
    #[test]
    fn join_tree_round_trip() {
        let a = textured(40, 30, 0, 0);
        let b = textured(40, 30, 25, 0);
        let c = textured(55, 30, 0, 22);
        let ab = a.merge(&b, MergeDirection::Horizontal, 15 - 40, 0);
        let abc = ab.merge(&c, MergeDirection::Vertical, 0, 8 - 30);
        let blob = match abc.get_field(JOIN_TREE_FIELD) {
            Some(MetadataValue::Blob(blob)) => blob,
            other => panic!("expected join-tree blob, got {other:?}"),
        };
        let tree = JoinTree::deserialize(&blob).unwrap();
        // Borrow rather than move: `JoinTree` implements `Drop` (iterative, to
        // stay stack-safe on deep trees), so its fields cannot be moved out.
        let JoinTree::Join {
            direction: MergeDirection::Vertical,
            dy: -22,
            ref_node,
            sec_node,
            ..
        } = &tree
        else {
            panic!("root must be the vertical join at dy -22");
        };
        let JoinTree::Leaf(leaf_c) = &**sec_node else {
            panic!("secondary of the root is leaf c");
        };
        assert_eq!(leaf_c.data(), c.data());
        assert_eq!(tree_size(ref_node), (65, 30));
        let JoinTree::Join {
            direction: MergeDirection::Horizontal,
            dx: -25,
            ..
        } = &**ref_node
        else {
            panic!("reference of the root is the horizontal join at dx -25");
        };
    }

    /**
     * Tests leaf placement in final coordinates: for a horizontal join
     * then a vertical join, collect_leaves reproduces the same rects the
     * merge geometry used, which global balance's overlap statistics
     * depend on.
     * Works by comparing against hand-computed placements.
     */
    #[test]
    fn collect_leaves_placement() {
        let a = textured(40, 30, 0, 0);
        let b = textured(40, 30, 25, 0);
        let ab = a.merge(&b, MergeDirection::Horizontal, 15 - 40, 0);
        assert_eq!((ab.width(), ab.height()), (65, 30));
        let blob = match ab.get_field(JOIN_TREE_FIELD) {
            Some(MetadataValue::Blob(blob)) => blob,
            _ => unreachable!(),
        };
        let tree = JoinTree::deserialize(&blob).unwrap();
        let mut leaves = Vec::new();
        collect_leaves(&tree, 0, 0, &mut leaves);
        assert_eq!(leaves.len(), 2);
        assert_eq!(
            leaves[0].rect,
            Rect {
                left: 0,
                top: 0,
                width: 40,
                height: 30
            }
        );
        assert_eq!(
            leaves[1].rect,
            Rect {
                left: 25,
                top: 0,
                width: 40,
                height: 30
            }
        );
    }

    /**
     * Tests the blend luts against the closed form: coef1 is the raised
     * cosine from 1 to 0, coef2 its complement, and the integer tables
     * are the truncated scaled values, with the endpoints exact.
     * Works by spot-checking the ends and middle of the tables.
     */
    #[test]
    fn blend_luts_shape() {
        let luts = blend_luts();
        assert_eq!(luts.coef1[0], 1.0);
        assert_eq!(luts.icoef1[0], BLEND_SCALE);
        assert_eq!(luts.icoef2[0], 0);
        assert!(luts.coef1[BLEND_SIZE - 1].abs() < 1e-12);
        assert_eq!(luts.icoef1[BLEND_SIZE - 1], 0);
        let mid = luts.coef1[BLEND_SIZE / 2] + luts.coef2[BLEND_SIZE / 2];
        assert!((mid - 1.0).abs() < 1e-12);
    }

    /**
     * Tests round-half-to-even displacement averaging (im_avgdxdy uses
     * rint): an average of 0.5 rounds to 0 and 1.5 rounds to 2.
     * Works by feeding two-point sets with those averages.
     */
    #[test]
    fn avgdxdy_rounds_ties_to_even() {
        let p = TiePoints {
            xref: vec![0, 0],
            yref: vec![0, 0],
            xsec: vec![0, 1], // average 0.5 -> 0
            ysec: vec![1, 2], // average 1.5 -> 2
            correlation: vec![1.0, 1.0],
            dx: vec![0.0; 2],
            dy: vec![0.0; 2],
            deviation: vec![0.0; 2],
            ..TiePoints::default()
        };
        assert_eq!(avgdxdy(&p).unwrap(), (0, 2));
    }

    /**
     * Tests the clinear fit on a pure translation: every point displaced
     * by (3, -2) fits scale 1, angle 0, delta (3, -2) with zero
     * deviations, so improve keeps all points.
     * Works by fitting four spread points and checking the model.
     */
    #[test]
    fn clinear_pure_translation() {
        let mut p = TiePoints::default();
        for (x, y) in [(10i64, 10i64), (40, 12), (12, 45), (44, 41)] {
            p.xref.push(x);
            p.yref.push(y);
            p.xsec.push(x + 3);
            p.ysec.push(y - 2);
            p.correlation.push(1.0);
            p.dx.push(0.0);
            p.dy.push(0.0);
            p.deviation.push(0.0);
        }
        assert!(clinear(&mut p));
        assert!((p.l_scale - 1.0).abs() < 1e-9);
        assert!(p.l_angle.abs() < 1e-9);
        assert!((p.l_deltax - 3.0).abs() < 1e-9);
        assert!((p.l_deltay + 2.0).abs() < 1e-9);
        assert!(p.deviation.iter().all(|&d| d < 1e-9));
        let kept = improve(p).unwrap();
        assert_eq!(kept.len(), 4);
    }

    /**
     * Tests improve discards an outlier: three consistent displacements
     * and one wild one leave the consistent trio after refitting.
     * Works by checking the survivor count and the averaged offset.
     */
    #[test]
    fn improve_discards_outlier() {
        let mut p = TiePoints::default();
        for (x, y) in [(10i64, 10i64), (60, 12), (12, 65)] {
            p.xref.push(x);
            p.yref.push(y);
            p.xsec.push(x + 5);
            p.ysec.push(y + 1);
            p.correlation.push(0.95);
            p.dx.push(0.0);
            p.dy.push(0.0);
            p.deviation.push(0.0);
        }
        // The outlier.
        p.xref.push(62);
        p.yref.push(60);
        p.xsec.push(62 + 14);
        p.ysec.push(60 - 9);
        p.correlation.push(0.2);
        p.dx.push(0.0);
        p.dy.push(0.0);
        p.deviation.push(0.0);
        initialize(&mut p).unwrap();
        let kept = improve(p).unwrap();
        assert_eq!(kept.len(), 3, "outlier dropped");
        assert_eq!(avgdxdy(&kept).unwrap(), (5, 1));
    }

    /**
     * Tests the correlation search finds an exact translated match: a
     * textured reference window is located in a secondary image shifted
     * by (4, -3), with correlation 1 at the true position.
     * Works by running correl directly on two shifted views of one
     * texture.
     */
    #[test]
    fn correl_finds_shift() {
        let scene = textured(80, 80, 0, 0);
        let view = |ox: i64, oy: i64| GrayU8 {
            w: 50,
            h: 50,
            data: {
                let mut d = Vec::with_capacity(2500);
                for y in 0..50i64 {
                    for x in 0..50i64 {
                        d.push(scene.data()[((y + oy) * 80 + (x + ox)) as usize]);
                    }
                }
                d
            },
        };
        let ref_im = view(10, 10);
        let sec_im = view(14, 7); // sec(x, y) == ref(x + 4, y - 3)... i.e.
        // ref feature at (x, y) appears in sec at (x - 4, y + 3).
        let (cc, xs, ys) = correl(&ref_im, &sec_im, 25, 25, 25, 25, HWINDOW, HAREA);
        assert_eq!((xs, ys), (21, 28));
        assert!((cc - 1.0).abs() < 1e-9, "perfect match correlates at 1");
    }

    /**
     * Tests scale_band0 matches the vips_scale contract: the band minimum
     * maps to 0, the maximum to 255, and a zero-range image scales to all
     * black.
     * Works on a two-pixel ramp and a uniform image.
     */
    #[test]
    fn scale_band0_contract() {
        let r = Raster::new(2, 1, PixelFormat::Gray8, vec![10, 210]).unwrap();
        let s = scale_band0(
            &r,
            Rect {
                left: 0,
                top: 0,
                width: 2,
                height: 1,
            },
        )
        .unwrap();
        assert_eq!(s.data, vec![0, 255]);
        let u = uniform(3, 1, 77);
        let s = scale_band0(
            &u,
            Rect {
                left: 0,
                top: 0,
                width: 3,
                height: 1,
            },
        )
        .unwrap();
        assert_eq!(s.data, vec![0, 0, 0], "zero range scales to black");
    }
}
