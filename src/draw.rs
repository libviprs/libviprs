//! Extensible in-place raster drawing.
//!
//! This module is the drawing seam for [`Raster`]. Every shape or paint
//! operation is a value that implements the [`DrawOp`] trait, whose single
//! [`apply`](DrawOp::apply) method mutates a `Raster` in place. New ops
//! (polygon, ellipse, gradient, ...) plug in by adding one `impl DrawOp` and,
//! optionally, one convenience method on `Raster`; the core [`Raster`] type
//! and the existing ops never change. This keeps the drawing surface open for
//! extension and closed for modification.
//!
//! # Why a trait, not hard-coded methods
//!
//! A `DrawOp` is a first-class, inspectable value. Callers can build a
//! `Vec<Box<dyn DrawOp>>` and replay it, wrap an op to log or clip it, or ship
//! a custom op from a downstream crate without a libviprs release. The
//! built-in ops ([`Circle`], [`Rectangle`], [`Line`], [`Flood`], [`Mask`],
//! [`Smudge`], [`Paste`]) are ordinary implementors with no privileged access,
//! so a third-party op is exactly as capable as a built-in one.
//!
//! This is deliberately *separate* from
//! [`Extensions`](crate::extensions::Extensions), which carries opaque
//! pipeline-level context (metrics recorders, tracing spans) into the pyramid
//! engine. `DrawOp` is about mutating pixel buffers; `Extensions` is about
//! threading shared context through a run. They solve different problems, so
//! they stay distinct rather than one being forced through the other.
//!
//! # Coordinates and clipping
//!
//! Op coordinates are `i32` so shapes may be positioned partly off-canvas.
//! Every op clips to the raster bounds: pixels outside `0..width` / `0..height`
//! are silently skipped, matching the clip-don't-panic convention of classic
//! raster libraries. Coordinate handling is therefore always infallible.
//!
//! The flood seed is the one place the ported libvips surface wants an error
//! instead: as a [`DrawOp`], a [`Flood`] whose seed lies off-canvas is a
//! silent no-op like every other op, while the [`Raster::draw_flood`] /
//! [`Raster::draw_flood_blob`] wrappers validate the seed first and return
//! [`DrawError::SeedOutOfBounds`].
//!
//! # Ink
//!
//! `ink` is the raw pixel value to paint, as bytes: one whole pixel wide, so
//! `&[100]` paints a `Gray8` pixel and `&[r, g, b]` an `Rgb8` one. Ink is
//! validated against the target raster's pixel width
//! ([`PixelFormat::bytes_per_pixel`](crate::pixel::PixelFormat::bytes_per_pixel))
//! before painting: a mismatch is a caller bug (for example a 3-byte RGB ink
//! on an `Rgba8` raster would paint the alpha band from the red byte). The
//! panicking `Raster::draw_*` forms panic and the fallible `try_draw_*` forms
//! return [`DrawError::InkLengthMismatch`] instead of silently corrupting
//! channels. The two flood wrappers are the documented exception to that
//! `draw_*`-panics / `try_draw_*`-returns split: [`Raster::draw_flood`] and
//! [`Raster::draw_flood_blob`] are `draw_*`-named but inherently fallible — they
//! also reject an off-canvas seed with [`DrawError::SeedOutOfBounds`] — so they
//! return an ink mismatch as `Err(DrawError::InkLengthMismatch)` rather than
//! panicking, and there is deliberately no `try_draw_flood`. The built-in ops
//! enforce the same check inside their [`apply`](DrawOp::apply), so the generic
//! [`Raster::draw`]`(&op)` entry point and the op constructors ([`Circle`],
//! [`Rectangle`], [`Line`], [`Flood`], [`Mask`]) are guarded too — a wrong-width
//! ink there panics rather than painting garbage.
//!
//! **Breaking change (from the pre-#294 drawing seam):** a shorter ink is no
//! longer broadcast by the draw ops. Earlier versions cycled a too-short ink to
//! fill the pixel, so a single-value ink such as `&[128]` painted a uniform
//! shade on a multi-band raster; that silent cycle is the channel-corruption
//! route the length check now closes, so a shorter ink is rejected (panic /
//! `Err`) on every `draw_*` / `try_draw_*` / `draw(&op)` path. The only surface
//! that still broadcasts is the raw [`Raster::put_pixel`] escape hatch below.
//!
//! The low-level [`Raster::put_pixel`] writer is the raw escape hatch: it
//! copies ink verbatim, cycling if it is shorter than the pixel's byte width
//! and truncating if longer, so custom [`DrawOp`]s that paint through it opt
//! out of the length check — and it is the deliberate opt-in for the
//! shorter-ink broadcast the draw ops no longer perform.
//!
//! # Example: a custom op
//!
//! ```
//! use libviprs::{PixelFormat, Raster};
//! use libviprs::draw::DrawOp;
//!
//! // A one-off op that paints a single horizontal scanline.
//! struct HLine<'a> { ink: &'a [u8], y: i32, x0: i32, x1: i32 }
//!
//! impl DrawOp for HLine<'_> {
//!     fn apply(&self, raster: &mut Raster) {
//!         for x in self.x0..=self.x1 {
//!             raster.put_pixel(x, self.y, self.ink);
//!         }
//!     }
//! }
//!
//! let mut im = Raster::zeroed(8, 8, PixelFormat::Gray8).unwrap();
//! im.draw(&HLine { ink: &[255], y: 3, x0: 0, x1: 7 });
//! assert_eq!(im.getpoint(0, 3), vec![255.0]);
//! ```

use thiserror::Error;

use crate::pixel::PixelFormat;
use crate::raster::Raster;

/// An in-place raster drawing operation.
///
/// Implement this for any new shape or paint effect. [`apply`](Self::apply)
/// receives exclusive access to the target [`Raster`] and mutates it directly.
/// Implementations must clip to the raster bounds and never panic on
/// out-of-range coordinates; use [`Raster::put_pixel`], which clips for you.
pub trait DrawOp {
    /// Paint this op onto `raster`, mutating it in place.
    fn apply(&self, raster: &mut Raster);
}

/// Typed errors for the fallible drawing wrappers in [`crate::draw`].
///
/// Ops themselves clip out-of-range coordinates rather than erroring; these
/// errors surface from the `Raster` convenience methods that validate their
/// inputs up front — the flood-fill seed, the ink channel width, and the
/// target's pixel format for the sample-blending ops.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DrawError {
    /// A flood-fill seed lies outside the image.
    #[error("flood seed ({x},{y}) out of bounds for {image_w}x{image_h}")]
    SeedOutOfBounds {
        x: i32,
        y: i32,
        image_w: u32,
        image_h: u32,
    },
    /// The ink byte length does not match the target raster's pixel width.
    ///
    /// Ink is painted verbatim as one whole pixel, so it must be exactly
    /// [`PixelFormat::bytes_per_pixel`](crate::pixel::PixelFormat::bytes_per_pixel)
    /// bytes (`channels * bytes_per_channel`) for the target raster. A shorter
    /// ink would silently cycle and a longer one truncate, corrupting channels
    /// — for example a 3-byte RGB ink on an `Rgba8` raster paints the alpha
    /// band from the red byte — so a mismatch is rejected instead of drawing
    /// garbage. Paint through [`Raster::put_pixel`] to opt out of the check.
    #[error(
        "ink of {ink_len} bytes does not match the {pixel_bytes}-byte pixel width of {format:?}; \
         a mismatched ink is no longer broadcast to fit (a shorter ink cycled, a longer one \
         truncated) — supply exactly {pixel_bytes} bytes, or paint through Raster::put_pixel to \
         opt into that broadcast"
    )]
    InkLengthMismatch {
        /// Length of the supplied ink slice, in bytes.
        ink_len: usize,
        /// Bytes required for one whole pixel of the target format.
        pixel_bytes: usize,
        /// The target raster's pixel format.
        format: PixelFormat,
    },
    /// A draw operation that decodes samples as unsigned integers was asked to
    /// operate on a 32-bit-float raster
    /// ([`RgbaF32`](crate::pixel::PixelFormat::RgbaF32) /
    /// [`FloatF32`](crate::pixel::PixelFormat::FloatF32)).
    ///
    /// The sample-blending ops (currently [`Mask`]) read and write channels as
    /// unsigned 8/16-bit values and cannot yet interpret `f32` samples. Rather
    /// than panicking deep in the sample decode — or, worse, misreading float
    /// bytes as `u16` pairs — the fallible wrappers reject a float target up
    /// front. Cast to an unsigned 8/16-bit format first.
    #[error(
        "draw operation does not support the 32-bit-float raster {format:?}; \
         cast to an unsigned 8/16-bit format first"
    )]
    UnsupportedFloatRaster {
        /// The target raster's (float) pixel format.
        format: PixelFormat,
    },
}

/// A circle, drawn as an outline or filled disc.
///
/// Built with [`Circle::outline`] / [`Circle::filled`], or applied through the
/// [`Raster::draw_circle`] / [`Raster::draw_circle_filled`] convenience
/// methods. The outline is a 1px midpoint circle; the fill paints the
/// horizontal spans between the same midpoint extents, so an outline and a
/// fill of identical parameters share exactly the same boundary pixels and
/// flood-filling an outlined circle yields the filled disc, as in libvips.
#[derive(Debug, Clone)]
pub struct Circle<'a> {
    /// Pixel value to paint (see the [module docs](self#ink)).
    pub ink: &'a [u8],
    /// Centre x coordinate.
    pub cx: i32,
    /// Centre y coordinate.
    pub cy: i32,
    /// Radius in pixels. A radius `< 0` draws nothing.
    pub radius: i32,
    /// Whether to fill the disc (`true`) or draw only the outline (`false`).
    pub fill: bool,
}

impl<'a> Circle<'a> {
    /// A 1px-thick circle outline.
    pub fn outline(ink: &'a [u8], cx: i32, cy: i32, radius: i32) -> Self {
        Self {
            ink,
            cx,
            cy,
            radius,
            fill: false,
        }
    }

    /// A solid filled disc.
    pub fn filled(ink: &'a [u8], cx: i32, cy: i32, radius: i32) -> Self {
        Self {
            ink,
            cx,
            cy,
            radius,
            fill: true,
        }
    }
}

impl DrawOp for Circle<'_> {
    fn apply(&self, raster: &mut Raster) {
        assert_ink_pixel_width(raster, self.ink);
        if self.radius < 0 {
            return;
        }
        // If the circle's bounding box lies wholly off-canvas, no pixel can be
        // painted, so return before walking the O(radius) midpoint octant. This
        // paints the same (zero) pixels as vips — which clips each point/scanline
        // during its walk — while a huge off-canvas radius can no longer spin for
        // seconds. (An on-canvas huge radius still walks, matching vips, but
        // `fill_circle`/`outline_circle` keep each step O(canvas), not O(radius).)
        let (cx, cy, r) = (self.cx as i64, self.cy as i64, self.radius as i64);
        let (w, h) = (raster.width() as i64, raster.height() as i64);
        if cx + r < 0 || cx - r >= w || cy + r < 0 || cy - r >= h {
            return;
        }
        if self.fill {
            fill_circle(raster, self.ink, self.cx, self.cy, self.radius);
        } else {
            outline_circle(raster, self.ink, self.cx, self.cy, self.radius);
        }
    }
}

/// A rectangle, drawn as an outline or filled.
///
/// Built with [`Rectangle::outline`] / [`Rectangle::filled`], or applied
/// through [`Raster::draw_rect`] / [`Raster::draw_rect_filled`]. `left`/`top`
/// are the top-left corner; `width`/`height` extend right/down. Non-positive
/// `width` or `height` draws nothing.
#[derive(Debug, Clone)]
pub struct Rectangle<'a> {
    /// Pixel value to paint (see the [module docs](self#ink)).
    pub ink: &'a [u8],
    /// Left edge x coordinate.
    pub left: i32,
    /// Top edge y coordinate.
    pub top: i32,
    /// Width in pixels.
    pub width: i32,
    /// Height in pixels.
    pub height: i32,
    /// Whether to fill the rectangle (`true`) or draw only the 1px border.
    pub fill: bool,
}

impl<'a> Rectangle<'a> {
    /// A 1px-thick rectangle border.
    pub fn outline(ink: &'a [u8], left: i32, top: i32, width: i32, height: i32) -> Self {
        Self {
            ink,
            left,
            top,
            width,
            height,
            fill: false,
        }
    }

    /// A solid filled rectangle.
    pub fn filled(ink: &'a [u8], left: i32, top: i32, width: i32, height: i32) -> Self {
        Self {
            ink,
            left,
            top,
            width,
            height,
            fill: true,
        }
    }
}

impl DrawOp for Rectangle<'_> {
    fn apply(&self, raster: &mut Raster) {
        assert_ink_pixel_width(raster, self.ink);
        if self.width <= 0 || self.height <= 0 {
            return;
        }
        // `right`/`bottom` are the inclusive far edges. Widen to i64 so a large
        // `left + width` cannot overflow i32.
        let right = self.left as i64 + self.width as i64 - 1;
        let bottom = self.top as i64 + self.height as i64 - 1;
        let w = raster.width() as i64;
        let h = raster.height() as i64;
        // Clip the drawn extents to the canvas before iterating. `put_pixel`
        // already no-ops off-canvas, so the painted pixels are unchanged; this
        // only skips the writes that would have been no-ops. Without it a huge
        // but in-range rect (libvips caps coordinates at 1e9) spins through up
        // to ~1e18 no-op iterations — a denial-of-service hang. libvips
        // `vips_draw_rect` clips the rect to the image the same way.
        let top = self.top as i64;
        let left = self.left as i64;
        if self.fill {
            let x0 = left.max(0);
            let y0 = top.max(0);
            let x1 = right.min(w - 1);
            let y1 = bottom.min(h - 1);
            if x0 > x1 || y0 > y1 {
                return;
            }
            for y in y0..=y1 {
                for x in x0..=x1 {
                    raster.put_pixel(x as i32, y as i32, self.ink);
                }
            }
        } else {
            // Border edges: clip the span to the canvas and skip an edge whose
            // fixed coordinate is itself off-canvas (which also avoids casting
            // an out-of-range far edge back to i32).
            let x0 = left.max(0);
            let x1 = right.min(w - 1);
            let y0 = top.max(0);
            let y1 = bottom.min(h - 1);
            if x0 <= x1 {
                if (0..h).contains(&top) {
                    for x in x0..=x1 {
                        raster.put_pixel(x as i32, top as i32, self.ink);
                    }
                }
                if (0..h).contains(&bottom) {
                    for x in x0..=x1 {
                        raster.put_pixel(x as i32, bottom as i32, self.ink);
                    }
                }
            }
            if y0 <= y1 {
                if (0..w).contains(&left) {
                    for y in y0..=y1 {
                        raster.put_pixel(left as i32, y as i32, self.ink);
                    }
                }
                if (0..w).contains(&right) {
                    for y in y0..=y1 {
                        raster.put_pixel(right as i32, y as i32, self.ink);
                    }
                }
            }
        }
    }
}

/// A 1px line segment, inclusive of both endpoints.
///
/// Built with [`Line::new`], or applied through [`Raster::draw_line`]. This is
/// the `vips_draw_line` analogue: a Bresenham walk from `(x1, y1)` to
/// `(x2, y2)` painting one pixel per step along the major axis. Off-canvas
/// portions clip pixel-by-pixel, so either endpoint may lie outside the
/// raster.
#[derive(Debug, Clone)]
pub struct Line<'a> {
    /// Pixel value to paint (see the [module docs](self#ink)).
    pub ink: &'a [u8],
    /// Start x coordinate (inclusive).
    pub x1: i32,
    /// Start y coordinate (inclusive).
    pub y1: i32,
    /// End x coordinate (inclusive).
    pub x2: i32,
    /// End y coordinate (inclusive).
    pub y2: i32,
}

impl<'a> Line<'a> {
    /// A line from `(x1, y1)` to `(x2, y2)`, both endpoints painted.
    pub fn new(ink: &'a [u8], x1: i32, y1: i32, x2: i32, y2: i32) -> Self {
        Self {
            ink,
            x1,
            y1,
            x2,
            y2,
        }
    }
}

impl DrawOp for Line<'_> {
    fn apply(&self, raster: &mut Raster) {
        assert_ink_pixel_width(raster, self.ink);
        // If the segment's bounding box is wholly off-canvas, no point can be
        // painted; return before the Bresenham walk so a max-coord off-canvas
        // line can't spin through ~1e9 no-op steps. vips (`vips__draw_line_direct`)
        // clips per point across the whole walk, so this changes no on-canvas
        // pixel — it only drops a walk that would paint nothing. A line that
        // does cross the canvas is still walked in full, exactly matching vips's
        // pointwise-clipped trajectory (endpoint clipping would diverge from it).
        let (w, h) = (raster.width() as i64, raster.height() as i64);
        let (x1, y1) = (self.x1 as i64, self.y1 as i64);
        let (bx2, by2) = (self.x2 as i64, self.y2 as i64);
        if x1.max(bx2) < 0 || x1.min(bx2) >= w || y1.max(by2) < 0 || y1.min(by2) >= h {
            return;
        }
        // Classic integer Bresenham over all octants. Runs in i64 so the
        // deltas of extreme i32 endpoints cannot overflow.
        let (mut x, mut y) = (self.x1 as i64, self.y1 as i64);
        let (x2, y2) = (self.x2 as i64, self.y2 as i64);
        let dx = (x2 - x).abs();
        let sx = if x < x2 { 1 } else { -1 };
        let dy = -(y2 - y).abs();
        let sy = if y < y2 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            // x and y stay between the i32 endpoints, so the casts are exact.
            raster.put_pixel(x as i32, y as i32, self.ink);
            if x == x2 && y == y2 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }
}

/// A scanline flood fill from a seed point, 4-connected like libvips.
///
/// Two variants, mirroring `vips_draw_flood`:
///
/// * [`Flood::bounded`] (the libvips default): paint every pixel 4-connected
///   to the seed whose value differs from `ink`. The fill is bounded by
///   pixels already equal to the ink colour, so it pours up to an ink edge;
///   a seed already sitting on ink is a no-op.
/// * [`Flood::blob`] (libvips `equal: true`, historically `im_flood_blob`):
///   paint every pixel 4-connected to the seed whose value equals the seed
///   pixel's value, recolouring the blob the seed sits in. If the blob is
///   already the ink colour there is nothing to do.
///
/// Pixels compare as whole pixels (every byte), so multi-band rasters flood
/// on exact colour matches. As a [`DrawOp`], an off-canvas seed is a silent
/// no-op like any other clipped op; the [`Raster::draw_flood`] /
/// [`Raster::draw_flood_blob`] wrappers validate the seed first and return
/// [`DrawError::SeedOutOfBounds`], which is the surface the ported libvips
/// tests call.
#[derive(Debug, Clone)]
pub struct Flood<'a> {
    /// Pixel value to paint (see the [module docs](self#ink)).
    pub ink: &'a [u8],
    /// Seed x coordinate.
    pub x: i32,
    /// Seed y coordinate.
    pub y: i32,
    /// libvips `equal`: `false` fills while pixels differ from `ink`
    /// (bounded); `true` fills while pixels equal the seed value (blob).
    pub equal: bool,
}

impl<'a> Flood<'a> {
    /// A fill bounded by pixels equal to `ink` (the `vips_draw_flood`
    /// default).
    pub fn bounded(ink: &'a [u8], x: i32, y: i32) -> Self {
        Self {
            ink,
            x,
            y,
            equal: false,
        }
    }

    /// A fill of the connected region whose pixels equal the seed pixel
    /// (`vips_draw_flood` with `equal` set).
    pub fn blob(ink: &'a [u8], x: i32, y: i32) -> Self {
        Self {
            ink,
            x,
            y,
            equal: true,
        }
    }
}

impl DrawOp for Flood<'_> {
    fn apply(&self, raster: &mut Raster) {
        assert_ink_pixel_width(raster, self.ink);
        let w = raster.width() as i64;
        let h = raster.height() as i64;
        if self.x < 0 || self.y < 0 || (self.x as i64) >= w || (self.y as i64) >= h {
            return;
        }
        let bpp = raster.format().bytes_per_pixel();
        let Some(ink_px) = ink_pixel(self.ink, bpp) else {
            return;
        };
        let stride = raster.stride();
        let (w, h) = (w as usize, h as usize);
        let (sx, sy) = (self.x as usize, self.y as usize);
        let data = raster.data_mut();
        /// The `bpp` bytes of the pixel at `(x, y)`.
        fn px(data: &[u8], x: usize, y: usize, stride: usize, bpp: usize) -> &[u8] {
            let off = y * stride + x * bpp;
            &data[off..off + bpp]
        }
        let seed = px(data, sx, sy, stride, bpp).to_vec();
        if self.equal && seed == ink_px {
            // Repainting a blob in its own colour would re-satisfy the fill
            // predicate forever; libvips returns immediately, and so do we.
            return;
        }
        // `true` when the pixel should be painted. Painting writes `ink_px`,
        // which makes the predicate false, so every pixel is painted at most
        // once and the loop terminates.
        let fills = |data: &[u8], x: usize, y: usize| {
            let p = px(data, x, y, stride, bpp);
            if self.equal {
                p == seed.as_slice()
            } else {
                p != ink_px.as_slice()
            }
        };
        // Scanline fill: paint the whole run around each popped seed, then
        // push one seed per fillable run on the rows above and below. Only
        // horizontal and vertical neighbours propagate, giving the libvips
        // 4-connectivity (a diagonal-only gap does not leak).
        let mut stack = vec![(sx, sy)];
        while let Some((x, y)) = stack.pop() {
            if !fills(data, x, y) {
                continue;
            }
            let mut x0 = x;
            while x0 > 0 && fills(data, x0 - 1, y) {
                x0 -= 1;
            }
            let mut x1 = x;
            while x1 + 1 < w && fills(data, x1 + 1, y) {
                x1 += 1;
            }
            for xx in x0..=x1 {
                let off = y * stride + xx * bpp;
                data[off..off + bpp].copy_from_slice(&ink_px);
            }
            for ny in [y.wrapping_sub(1), y + 1] {
                if ny >= h {
                    continue;
                }
                let mut xx = x0;
                while xx <= x1 {
                    if fills(data, xx, ny) {
                        stack.push((xx, ny));
                        while xx <= x1 && fills(data, xx, ny) {
                            xx += 1;
                        }
                    } else {
                        xx += 1;
                    }
                }
            }
        }
    }
}

/// Ink painted through an 8-bit stencil mask, the `vips_draw_mask` analogue.
///
/// `mask` must be a single-band 8-bit raster ([`Gray8`] or a one-band
/// [`Multi8`]); any other mask format is a documented no-op, mirroring the
/// seam's clip-don't-panic convention where libvips would error. The mask's
/// top-left corner lands at `(x, y)` on the target, clipping where it
/// overhangs. Each mask value `m` blends the ink over the existing pixel per
/// channel as `new = (ink * m + old * (255 - m)) / 255` with truncating
/// integer division, matching libvips' blend: `m == 255` writes the ink
/// verbatim, `m == 0` leaves the pixel untouched. 16-bit channels blend on
/// their decoded native-endian values.
///
/// [`Gray8`]: crate::pixel::PixelFormat::Gray8
/// [`Multi8`]: crate::pixel::PixelFormat::Multi8
#[derive(Debug, Clone)]
pub struct Mask<'a> {
    /// Pixel value to paint at full mask opacity (see the
    /// [module docs](self#ink)).
    pub ink: &'a [u8],
    /// Single-band 8-bit stencil; each pixel is the ink opacity `0..=255`.
    pub mask: &'a Raster,
    /// Target x of the mask's left edge.
    pub x: i32,
    /// Target y of the mask's top edge.
    pub y: i32,
}

impl<'a> Mask<'a> {
    /// Ink through `mask` with the mask's top-left corner at `(x, y)`.
    pub fn new(ink: &'a [u8], mask: &'a Raster, x: i32, y: i32) -> Self {
        Self { ink, mask, x, y }
    }
}

impl DrawOp for Mask<'_> {
    fn apply(&self, raster: &mut Raster) {
        assert_ink_pixel_width(raster, self.ink);
        let mfmt = self.mask.format();
        if mfmt.channels() != 1 || mfmt.bytes_per_channel() != 1 {
            return;
        }
        let fmt = raster.format();
        let bpp = fmt.bytes_per_pixel();
        let bpc = fmt.bytes_per_channel();
        let channels = fmt.channels();
        let Some(ink_px) = ink_pixel(self.ink, bpp) else {
            return;
        };
        let ink_vals: Vec<u32> = (0..channels)
            .map(|c| channel_at(&ink_px, 0, c, bpc))
            .collect();
        let w = raster.width() as i64;
        let h = raster.height() as i64;
        let stride = raster.stride();
        let mstride = self.mask.stride();
        let mdata = self.mask.data();
        let data = raster.data_mut();
        for my in 0..self.mask.height() as i64 {
            let ty = self.y as i64 + my;
            if ty < 0 || ty >= h {
                continue;
            }
            for mx in 0..self.mask.width() as i64 {
                let tx = self.x as i64 + mx;
                if tx < 0 || tx >= w {
                    continue;
                }
                let m = mdata[my as usize * mstride + mx as usize] as u32;
                if m == 0 {
                    continue;
                }
                let off = ty as usize * stride + tx as usize * bpp;
                for (c, &ink_c) in ink_vals.iter().enumerate() {
                    let old = channel_at(data, off, c, bpc);
                    // Weighted average of two in-range values stays in range,
                    // and the u32 product cannot overflow (65535 * 255 max).
                    let v = (ink_c * m + old * (255 - m)) / 255;
                    set_channel_at(data, off, c, bpc, v);
                }
            }
        }
    }
}

/// A box-blur smudge of a rectangular region, the `vips_draw_smudge`
/// analogue.
///
/// Every pixel inside the rect (clipped to the canvas) is replaced by the
/// rounded per-channel mean of its 3x3 neighbourhood. Neighbours are read
/// from a snapshot of the pre-smudge pixels, so the result is independent of
/// scan order; windows overhanging the rect read the surrounding image, and
/// windows overhanging the canvas shrink to the pixels actually present (the
/// mean divides by the sampled count). Only pixels inside the rect are
/// written, so pasting the original region back restores the original image,
/// which is exactly what the ported libvips test asserts.
#[derive(Debug, Clone)]
pub struct Smudge {
    /// Left edge x coordinate.
    pub left: i32,
    /// Top edge y coordinate.
    pub top: i32,
    /// Width in pixels.
    pub width: i32,
    /// Height in pixels.
    pub height: i32,
}

impl Smudge {
    /// Smudge the rect at `(left, top)` sized `width` x `height`.
    pub fn new(left: i32, top: i32, width: i32, height: i32) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }
}

impl DrawOp for Smudge {
    fn apply(&self, raster: &mut Raster) {
        if self.width <= 0 || self.height <= 0 {
            return;
        }
        let w = raster.width() as i64;
        let h = raster.height() as i64;
        // Clip the target rect to the canvas; i64 so left + width can't wrap.
        let rx0 = (self.left as i64).max(0);
        let ry0 = (self.top as i64).max(0);
        let rx1 = (self.left as i64 + self.width as i64 - 1).min(w - 1);
        let ry1 = (self.top as i64 + self.height as i64 - 1).min(h - 1);
        if rx0 > rx1 || ry0 > ry1 {
            return;
        }
        // Snapshot the rect grown by one pixel (clipped): every 3x3 window a
        // rect pixel can see lies inside the snapshot.
        let sx0 = (rx0 - 1).max(0);
        let sy0 = (ry0 - 1).max(0);
        let sx1 = (rx1 + 1).min(w - 1);
        let sy1 = (ry1 + 1).min(h - 1);
        let sw = (sx1 - sx0 + 1) as usize;
        let sh = (sy1 - sy0 + 1) as usize;
        let fmt = raster.format();
        let channels = fmt.channels();
        let bpc = fmt.bytes_per_channel();
        let bpp = fmt.bytes_per_pixel();
        let stride = raster.stride();
        let mut snap = vec![0f64; sw * sh * channels];
        {
            let data = raster.data();
            for row in 0..sh {
                for col in 0..sw {
                    let off = (sy0 as usize + row) * stride + (sx0 as usize + col) * bpp;
                    for c in 0..channels {
                        snap[(row * sw + col) * channels + c] =
                            channel_at(data, off, c, bpc) as f64;
                    }
                }
            }
        }
        let max = if bpc == 1 { 255.0 } else { 65535.0 };
        let data = raster.data_mut();
        for y in ry0..=ry1 {
            for x in rx0..=rx1 {
                let off = y as usize * stride + x as usize * bpp;
                for c in 0..channels {
                    let mut sum = 0f64;
                    let mut n = 0u32;
                    for wy in (y - 1)..=(y + 1) {
                        if wy < 0 || wy >= h {
                            continue;
                        }
                        for wx in (x - 1)..=(x + 1) {
                            if wx < 0 || wx >= w {
                                continue;
                            }
                            let si = ((wy - sy0) as usize * sw + (wx - sx0) as usize) * channels;
                            sum += snap[si + c];
                            n += 1;
                        }
                    }
                    let v = (sum / n as f64).round().clamp(0.0, max) as u32;
                    set_channel_at(data, off, c, bpc, v);
                }
            }
        }
    }
}

/// A sub-image pasted into the raster, the `vips_draw_image` analogue.
///
/// The overlay's pixels replace the target's (libvips "set" mode) with the
/// overlay's top-left corner at `(x, y)`. Off-canvas rows and columns clip
/// away, so the overlay may straddle any edge. The overlay must share the
/// target's [`PixelFormat`](crate::pixel::PixelFormat); pasting a mismatched
/// format is a documented no-op (libvips would cast the sub-image, a
/// conversion this seam deliberately leaves to the caller).
#[derive(Debug, Clone)]
pub struct Paste<'a> {
    /// The sub-image whose pixels replace the target's.
    pub image: &'a Raster,
    /// Target x of the overlay's left edge.
    pub x: i32,
    /// Target y of the overlay's top edge.
    pub y: i32,
}

impl<'a> Paste<'a> {
    /// Paste `image` with its top-left corner at `(x, y)`.
    pub fn new(image: &'a Raster, x: i32, y: i32) -> Self {
        Self { image, x, y }
    }
}

impl DrawOp for Paste<'_> {
    fn apply(&self, raster: &mut Raster) {
        if self.image.format() != raster.format() {
            return;
        }
        let w = raster.width() as i64;
        let h = raster.height() as i64;
        let (x, y) = (self.x as i64, self.y as i64);
        let dx0 = x.max(0);
        let dy0 = y.max(0);
        let dx1 = (x + self.image.width() as i64 - 1).min(w - 1);
        let dy1 = (y + self.image.height() as i64 - 1).min(h - 1);
        if dx0 > dx1 || dy0 > dy1 {
            return;
        }
        let bpp = raster.format().bytes_per_pixel();
        let dstride = raster.stride();
        let sstride = self.image.stride();
        let sdata = self.image.data();
        let ddata = raster.data_mut();
        let row_bytes = (dx1 - dx0 + 1) as usize * bpp;
        for dy in dy0..=dy1 {
            let s0 = (dy - y) as usize * sstride + (dx0 - x) as usize * bpp;
            let d0 = dy as usize * dstride + dx0 as usize * bpp;
            ddata[d0..d0 + row_bytes].copy_from_slice(&sdata[s0..s0 + row_bytes]);
        }
    }
}

/// Materialise `ink` as one full pixel of `bpp` bytes, cycling and truncating
/// exactly like [`Raster::put_pixel`]. `None` for empty ink (a no-op there
/// too).
fn ink_pixel(ink: &[u8], bpp: usize) -> Option<Vec<u8>> {
    if ink.is_empty() {
        return None;
    }
    Some((0..bpp).map(|i| ink[i % ink.len()]).collect())
}

/// Read channel `c` of the pixel at byte offset `off` as its unsigned value
/// (native-endian for 16-bit channels, like `Raster::getpoint`).
/// Unsigned depths only: the panic arm keeps the sample-level draw ops,
/// which predate the float formats, from misreading float bytes as `u16`
/// pairs. (`put_pixel` and the raw-ink paths copy whole `bpp`-sized pixels
/// and handle float fine.)
fn channel_at(data: &[u8], off: usize, c: usize, bpc: usize) -> u32 {
    match bpc {
        1 => data[off + c] as u32,
        2 => {
            let b = off + c * 2;
            u16::from_ne_bytes([data[b], data[b + 1]]) as u32
        }
        _ => panic!(
            "this draw operation does not support float rasters yet; \
             cast to an unsigned 8/16-bit format first"
        ),
    }
}

/// Write channel `c` of the pixel at byte offset `off`, saturating to the
/// channel's range. Unsigned depths only; see [`channel_at`].
fn set_channel_at(data: &mut [u8], off: usize, c: usize, bpc: usize, v: u32) {
    match bpc {
        1 => data[off + c] = v.min(255) as u8,
        2 => {
            let b = off + c * 2;
            let bytes = (v.min(65535) as u16).to_ne_bytes();
            data[b] = bytes[0];
            data[b + 1] = bytes[1];
        }
        _ => panic!(
            "this draw operation does not support float rasters yet; \
             cast to an unsigned 8/16-bit format first"
        ),
    }
}

/// Walk the first octant of the midpoint circle of `radius`, calling
/// `step(x, y)` once per algorithm step with `x >= y >= 0`. Both the outline
/// and the fill derive their pixels from this single walk, so they agree on
/// the circle boundary by construction: flood-filling an outlined circle
/// reproduces the filled disc exactly, as it does in libvips.
///
/// `pub(crate)` because `hough_circle` in [`crate::arithmetic`] votes along
/// this same point set: the set is symmetric under negation, so a pixel
/// lying on a drawn midpoint circle always votes for that circle's exact
/// centre.
pub(crate) fn for_each_octant_step(radius: i32, mut step: impl FnMut(i32, i32)) {
    let mut x = radius;
    let mut y = 0;
    // Decision variable for the midpoint algorithm.
    let mut err = 1 - radius;
    while x >= y {
        step(x, y);
        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

/// Midpoint-circle outline: plot the eight symmetric octant points.
fn outline_circle(raster: &mut Raster, ink: &[u8], cx: i32, cy: i32, radius: i32) {
    if radius == 0 {
        raster.put_pixel(cx, cy, ink);
        return;
    }
    for_each_octant_step(radius, |x, y| {
        for (px, py) in [
            (cx + x, cy + y),
            (cx - x, cy + y),
            (cx + x, cy - y),
            (cx - x, cy - y),
            (cx + y, cy + x),
            (cx - y, cy + x),
            (cx + y, cy - x),
            (cx - y, cy - x),
        ] {
            raster.put_pixel(px, py, ink);
        }
    });
}

/// Solid disc: paint the horizontal spans between the outline's octant
/// extents, as libvips `vips_draw_circle` does. Sharing the octant walk with
/// [`outline_circle`] keeps the fill and the outline boundary identical,
/// which the ported flood test relies on.
fn fill_circle(raster: &mut Raster, ink: &[u8], cx: i32, cy: i32, radius: i32) {
    if radius == 0 {
        raster.put_pixel(cx, cy, ink);
        return;
    }
    // Widen to i64 so a large centre plus radius cannot overflow i32, and clip
    // each span to the canvas before painting. `put_pixel` already no-ops
    // off-canvas, so the painted pixels are byte-for-byte identical to an
    // unclipped span; this only skips writes that would have been no-ops.
    // Without it every octant step iterates its full ~2*radius-wide span, so a
    // huge but in-range radius is O(radius^2) — a denial-of-service hang. This
    // clips the span (and drops off-canvas rows) exactly as `vips_draw_circle`
    // does in `vips_draw_circle_draw_scanline`, keeping the fill O(radius).
    let w = raster.width() as i64;
    let h = raster.height() as i64;
    let (cx, cy) = (cx as i64, cy as i64);
    for_each_octant_step(radius, |x, y| {
        let (x, y) = (x as i64, y as i64);
        for (x0, x1, py) in [
            (cx - x, cx + x, cy + y),
            (cx - x, cx + x, cy - y),
            (cx - y, cx + y, cy + x),
            (cx - y, cx + y, cy - x),
        ] {
            if py < 0 || py >= h {
                continue;
            }
            let py = py as i32;
            let lo = x0.max(0);
            let hi = x1.min(w - 1);
            for px in lo..=hi {
                raster.put_pixel(px as i32, py, ink);
            }
        }
    });
}

/// Unwrap a [`DrawError`] into a panic tagged with the operation name, the way
/// the panicking `draw_*` wrappers turn an ink-width bug into an immediate,
/// clear failure. Mirrors `expect_resample` in [`crate::resample`].
#[track_caller]
fn expect_draw<T>(op: &str, r: Result<T, DrawError>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{op}: {e}"),
    }
}

/// Panic unless `ink` is exactly one pixel wide for `raster`'s format.
///
/// The built-in ink-bearing ops call this at the top of their [`DrawOp::apply`]
/// so the public [`Raster::draw`]`(&op)` entry point and the op constructors are
/// guarded exactly like the `draw_*` wrappers, closing the #294 corruption
/// route: a wrong-width ink would otherwise cycle (too short) or truncate (too
/// long) through [`Raster::put_pixel`], painting channels from the wrong bytes.
///
/// The `try_draw_*` / `draw_*` wrappers validate up front and short-circuit
/// before `apply`, so for them this is a redundant re-check that always passes;
/// it fires only on the infallible `draw(&op)` path, where a mismatch is a
/// documented caller bug and panicking mirrors the `draw_*` wrappers' behaviour.
/// Custom third-party ops that paint through `put_pixel` still opt out of the
/// check by not calling this.
#[track_caller]
fn assert_ink_pixel_width(raster: &Raster, ink: &[u8]) {
    if let Err(e) = raster.check_ink(ink) {
        panic!("{e}");
    }
}

impl Raster {
    /// Apply any [`DrawOp`] to this raster in place.
    ///
    /// This is the generic entry point; the `draw_*` methods below are thin
    /// wrappers over it for the common shapes.
    pub fn draw<O: DrawOp + ?Sized>(&mut self, op: &O) {
        op.apply(self);
    }

    /// Write `ink` to the pixel at `(x, y)`, clipping if it lies off-canvas.
    ///
    /// `ink` is copied verbatim, cycling if shorter than the pixel's byte width
    /// and truncating if longer. Out-of-bounds coordinates (including negative
    /// ones) are a silent no-op, so drawing code never has to bounds-check.
    pub fn put_pixel(&mut self, x: i32, y: i32, ink: &[u8]) {
        if x < 0 || y < 0 || x >= self.width() as i32 || y >= self.height() as i32 {
            return;
        }
        if ink.is_empty() {
            return;
        }
        let bpp = self.format().bytes_per_pixel();
        let stride = self.stride();
        let start = y as usize * stride + x as usize * bpp;
        let data = self.data_mut();
        for (i, byte) in data[start..start + bpp].iter_mut().enumerate() {
            *byte = ink[i % ink.len()];
        }
    }

    /// Draw a circle outline (see [`Circle::outline`]).
    ///
    /// # Panics
    ///
    /// Panics if `ink` is not exactly one pixel wide for this raster's format;
    /// see [`Raster::try_draw_circle`] for the fallible form.
    #[track_caller]
    pub fn draw_circle(&mut self, ink: &[u8], cx: i32, cy: i32, radius: i32) {
        expect_draw("draw_circle", self.try_draw_circle(ink, cx, cy, radius));
    }

    /// Fallible form of [`Raster::draw_circle`].
    ///
    /// # Errors
    ///
    /// [`DrawError::InkLengthMismatch`] if `ink` is not exactly one pixel wide
    /// for this raster's format.
    pub fn try_draw_circle(
        &mut self,
        ink: &[u8],
        cx: i32,
        cy: i32,
        radius: i32,
    ) -> Result<(), DrawError> {
        self.check_ink(ink)?;
        self.draw(&Circle::outline(ink, cx, cy, radius));
        Ok(())
    }

    /// Draw a filled disc (see [`Circle::filled`]).
    ///
    /// # Panics
    ///
    /// Panics if `ink` is not exactly one pixel wide for this raster's format;
    /// see [`Raster::try_draw_circle_filled`] for the fallible form.
    #[track_caller]
    pub fn draw_circle_filled(&mut self, ink: &[u8], cx: i32, cy: i32, radius: i32) {
        expect_draw(
            "draw_circle_filled",
            self.try_draw_circle_filled(ink, cx, cy, radius),
        );
    }

    /// Fallible form of [`Raster::draw_circle_filled`].
    ///
    /// # Errors
    ///
    /// [`DrawError::InkLengthMismatch`] if `ink` is not exactly one pixel wide
    /// for this raster's format.
    pub fn try_draw_circle_filled(
        &mut self,
        ink: &[u8],
        cx: i32,
        cy: i32,
        radius: i32,
    ) -> Result<(), DrawError> {
        self.check_ink(ink)?;
        self.draw(&Circle::filled(ink, cx, cy, radius));
        Ok(())
    }

    /// Draw a rectangle outline (see [`Rectangle::outline`]).
    ///
    /// # Panics
    ///
    /// Panics if `ink` is not exactly one pixel wide for this raster's format;
    /// see [`Raster::try_draw_rect`] for the fallible form.
    #[track_caller]
    pub fn draw_rect(&mut self, ink: &[u8], left: i32, top: i32, width: i32, height: i32) {
        expect_draw(
            "draw_rect",
            self.try_draw_rect(ink, left, top, width, height),
        );
    }

    /// Fallible form of [`Raster::draw_rect`].
    ///
    /// # Errors
    ///
    /// [`DrawError::InkLengthMismatch`] if `ink` is not exactly one pixel wide
    /// for this raster's format.
    pub fn try_draw_rect(
        &mut self,
        ink: &[u8],
        left: i32,
        top: i32,
        width: i32,
        height: i32,
    ) -> Result<(), DrawError> {
        self.check_ink(ink)?;
        self.draw(&Rectangle::outline(ink, left, top, width, height));
        Ok(())
    }

    /// Draw a filled rectangle (see [`Rectangle::filled`]).
    ///
    /// # Panics
    ///
    /// Panics if `ink` is not exactly one pixel wide for this raster's format;
    /// see [`Raster::try_draw_rect_filled`] for the fallible form.
    #[track_caller]
    pub fn draw_rect_filled(&mut self, ink: &[u8], left: i32, top: i32, width: i32, height: i32) {
        expect_draw(
            "draw_rect_filled",
            self.try_draw_rect_filled(ink, left, top, width, height),
        );
    }

    /// Fallible form of [`Raster::draw_rect_filled`].
    ///
    /// # Errors
    ///
    /// [`DrawError::InkLengthMismatch`] if `ink` is not exactly one pixel wide
    /// for this raster's format.
    pub fn try_draw_rect_filled(
        &mut self,
        ink: &[u8],
        left: i32,
        top: i32,
        width: i32,
        height: i32,
    ) -> Result<(), DrawError> {
        self.check_ink(ink)?;
        self.draw(&Rectangle::filled(ink, left, top, width, height));
        Ok(())
    }

    /// Draw a 1px line from `(x1, y1)` to `(x2, y2)`, both endpoints
    /// inclusive (see [`Line`]).
    ///
    /// # Panics
    ///
    /// Panics if `ink` is not exactly one pixel wide for this raster's format;
    /// see [`Raster::try_draw_line`] for the fallible form.
    #[track_caller]
    pub fn draw_line(&mut self, ink: &[u8], x1: i32, y1: i32, x2: i32, y2: i32) {
        expect_draw("draw_line", self.try_draw_line(ink, x1, y1, x2, y2));
    }

    /// Fallible form of [`Raster::draw_line`].
    ///
    /// # Errors
    ///
    /// [`DrawError::InkLengthMismatch`] if `ink` is not exactly one pixel wide
    /// for this raster's format.
    pub fn try_draw_line(
        &mut self,
        ink: &[u8],
        x1: i32,
        y1: i32,
        x2: i32,
        y2: i32,
    ) -> Result<(), DrawError> {
        self.check_ink(ink)?;
        self.draw(&Line::new(ink, x1, y1, x2, y2));
        Ok(())
    }

    /// Flood-fill with `ink` from `(x, y)`, bounded by pixels already equal
    /// to the ink colour (see [`Flood::bounded`]).
    ///
    /// # Errors
    ///
    /// [`DrawError::InkLengthMismatch`] if `ink` is not exactly one pixel wide
    /// for this raster's format, or [`DrawError::SeedOutOfBounds`] when the
    /// seed lies outside the image, matching the ported libvips surface.
    pub fn draw_flood(&mut self, ink: &[u8], x: i32, y: i32) -> Result<(), DrawError> {
        self.check_ink(ink)?;
        self.flood_seed_in_bounds(x, y)?;
        self.draw(&Flood::bounded(ink, x, y));
        Ok(())
    }

    /// Flood-fill with `ink` the 4-connected region whose pixels equal the
    /// value at `(x, y)` (see [`Flood::blob`]).
    ///
    /// # Errors
    ///
    /// [`DrawError::InkLengthMismatch`] if `ink` is not exactly one pixel wide
    /// for this raster's format, or [`DrawError::SeedOutOfBounds`] when the
    /// seed lies outside the image.
    pub fn draw_flood_blob(&mut self, ink: &[u8], x: i32, y: i32) -> Result<(), DrawError> {
        self.check_ink(ink)?;
        self.flood_seed_in_bounds(x, y)?;
        self.draw(&Flood::blob(ink, x, y));
        Ok(())
    }

    /// Paint `ink` through the single-band 8-bit stencil `mask` with the
    /// mask's top-left corner at `(x, y)` (see [`Mask`]).
    ///
    /// # Panics
    ///
    /// Panics if `ink` is not exactly one pixel wide for this raster's format,
    /// or if this raster is a 32-bit-float format
    /// ([`RgbaF32`](crate::pixel::PixelFormat::RgbaF32) /
    /// [`FloatF32`](crate::pixel::PixelFormat::FloatF32)) — the mask blend
    /// decodes unsigned 8/16-bit samples and does not support float targets.
    /// See [`Raster::try_draw_mask`] for the fallible form that returns
    /// [`DrawError::InkLengthMismatch`] / [`DrawError::UnsupportedFloatRaster`]
    /// instead.
    #[track_caller]
    pub fn draw_mask(&mut self, ink: &[u8], mask: &Raster, x: i32, y: i32) {
        expect_draw("draw_mask", self.try_draw_mask(ink, mask, x, y));
    }

    /// Fallible form of [`Raster::draw_mask`].
    ///
    /// # Errors
    ///
    /// [`DrawError::InkLengthMismatch`] if `ink` is not exactly one pixel wide
    /// for this raster's format, or [`DrawError::UnsupportedFloatRaster`] if
    /// this raster is a 32-bit-float format — the mask blend decodes unsigned
    /// 8/16-bit samples and cannot yet operate on `f32` rasters, so it rejects
    /// them up front rather than panicking inside the sample decode.
    pub fn try_draw_mask(
        &mut self,
        ink: &[u8],
        mask: &Raster,
        x: i32,
        y: i32,
    ) -> Result<(), DrawError> {
        self.check_ink(ink)?;
        self.reject_float_target()?;
        self.draw(&Mask::new(ink, mask, x, y));
        Ok(())
    }

    /// Box-blur the rect at `(left, top)` sized `width` x `height` in place
    /// (see [`Smudge`]).
    pub fn draw_smudge(&mut self, left: i32, top: i32, width: i32, height: i32) {
        self.draw(&Smudge::new(left, top, width, height));
    }

    /// Paste `sub` into this raster with its top-left corner at `(x, y)`,
    /// replacing pixels (see [`Paste`]).
    pub fn draw_image(&mut self, sub: &Raster, x: i32, y: i32) {
        self.draw(&Paste::new(sub, x, y));
    }

    /// Validate that `ink` is exactly one pixel wide for this raster's format,
    /// the precondition the `draw_*` / `try_draw_*` entry points enforce before
    /// painting. Correct-length ink is required so the verbatim copy in
    /// [`Raster::put_pixel`] fills each channel from its own byte instead of
    /// silently cycling (ink too short) or truncating (ink too long).
    fn check_ink(&self, ink: &[u8]) -> Result<(), DrawError> {
        let pixel_bytes = self.format().bytes_per_pixel();
        if ink.len() != pixel_bytes {
            return Err(DrawError::InkLengthMismatch {
                ink_len: ink.len(),
                pixel_bytes,
                format: self.format(),
            });
        }
        Ok(())
    }

    /// Reject a 32-bit-float target for the sample-decoding draw ops (the mask
    /// blend), which only understand unsigned 8/16-bit channels. Returns
    /// [`DrawError::UnsupportedFloatRaster`] so the fallible wrappers surface a
    /// typed error instead of the panic [`channel_at`] would raise when it hit
    /// float bytes.
    fn reject_float_target(&self) -> Result<(), DrawError> {
        if self.format().is_float() {
            return Err(DrawError::UnsupportedFloatRaster {
                format: self.format(),
            });
        }
        Ok(())
    }

    /// Bounds check shared by the flood wrappers.
    fn flood_seed_in_bounds(&self, x: i32, y: i32) -> Result<(), DrawError> {
        if x < 0 || y < 0 || (x as i64) >= self.width() as i64 || (y as i64) >= self.height() as i64
        {
            return Err(DrawError::SeedOutOfBounds {
                x,
                y,
                image_w: self.width(),
                image_h: self.height(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelFormat;

    fn black(w: u32, h: u32) -> Raster {
        Raster::zeroed(w, h, PixelFormat::Gray8).unwrap()
    }

    fn at(im: &Raster, x: u32, y: u32) -> u8 {
        im.region(x, y, 1, 1).unwrap().pixel(0, 0).unwrap()[0]
    }

    /// put_pixel writes ink and clips off-canvas coordinates without panicking.
    #[test]
    fn put_pixel_writes_and_clips() {
        let mut im = black(4, 4);
        im.put_pixel(1, 2, &[200]);
        assert_eq!(at(&im, 1, 2), 200);
        // Off-canvas (negative and past-edge) are silent no-ops.
        im.put_pixel(-1, 0, &[9]);
        im.put_pixel(0, -1, &[9]);
        im.put_pixel(4, 0, &[9]);
        im.put_pixel(0, 4, &[9]);
        // Nothing else changed.
        assert_eq!(im.data().iter().filter(|&&b| b != 0).count(), 1);
    }

    /// put_pixel fills every channel of a multi-band pixel from the ink slice.
    #[test]
    fn put_pixel_multiband() {
        let mut im = Raster::zeroed(2, 1, PixelFormat::Rgb8).unwrap();
        im.put_pixel(0, 0, &[10, 20, 30]);
        assert_eq!(im.getpoint(0, 0), vec![10.0, 20.0, 30.0]);
    }

    /// Circle outline matches the libvips reference: the leftmost point is on
    /// the circle (ink), the pixel just inside it is untouched (0).
    #[test]
    fn draw_circle_outline_reference() {
        let mut im = black(100, 100);
        im.draw_circle(&[100], 50, 50, 25);
        assert_eq!(at(&im, 25, 50), 100, "pixel on circle");
        assert_eq!(
            at(&im, 26, 50),
            0,
            "pixel just inside outline is not filled"
        );
    }

    /// Filled circle: boundary and interior are ink, just-outside is untouched.
    #[test]
    fn draw_circle_filled_reference() {
        let mut im = black(100, 100);
        im.draw_circle_filled(&[100], 50, 50, 25);
        assert_eq!(at(&im, 25, 50), 100, "boundary");
        assert_eq!(at(&im, 26, 50), 100, "interior");
        assert_eq!(at(&im, 24, 50), 0, "just outside");
    }

    /// Filled coverage strictly exceeds outline coverage, and both agree on the
    /// four cardinal extreme points of the circle.
    #[test]
    fn fill_covers_more_than_outline() {
        let count = |im: &Raster| im.data().iter().filter(|&&b| b != 0).count();

        let mut outline = black(64, 64);
        outline.draw_circle(&[100], 32, 32, 20);
        let mut filled = black(64, 64);
        filled.draw_circle_filled(&[100], 32, 32, 20);

        assert!(
            count(&filled) > count(&outline),
            "fill ({}) should cover more pixels than outline ({})",
            count(&filled),
            count(&outline)
        );
        // The cardinal boundary points sit on both the outline and the fill.
        for (x, y) in [(12, 32), (52, 32), (32, 12), (32, 52)] {
            assert_eq!(at(&outline, x, y), 100, "outline cardinal ({x},{y})");
            assert_eq!(at(&filled, x, y), 100, "fill cardinal ({x},{y})");
        }
        // The centre is filled but not on the outline.
        assert_eq!(at(&filled, 32, 32), 100);
        assert_eq!(at(&outline, 32, 32), 0);
    }

    /// Rectangle outline paints only the border; the interior stays background.
    #[test]
    fn draw_rect_outline_border_only() {
        let mut im = black(20, 20);
        im.draw_rect(&[100], 5, 5, 10, 8);
        // Corners and edges are ink.
        assert_eq!(at(&im, 5, 5), 100);
        assert_eq!(at(&im, 14, 5), 100); // left+width-1
        assert_eq!(at(&im, 5, 12), 100); // top+height-1
        assert_eq!(at(&im, 14, 12), 100);
        // Interior is untouched.
        assert_eq!(at(&im, 9, 8), 0);
    }

    /// Filled rectangle paints the whole interior span, and clips at the edge.
    #[test]
    fn draw_rect_filled_covers_interior_and_clips() {
        let mut im = black(20, 20);
        im.draw_rect_filled(&[77], 5, 5, 10, 8);
        assert_eq!(at(&im, 5, 5), 77);
        assert_eq!(at(&im, 14, 12), 77);
        assert_eq!(at(&im, 9, 8), 77);
        assert_eq!(at(&im, 4, 5), 0, "left of rect");
        assert_eq!(at(&im, 15, 5), 0, "right of rect");

        // A rectangle straddling the top-left corner clips without panic.
        let mut im2 = black(10, 10);
        im2.draw_rect_filled(&[5], -3, -3, 6, 6);
        assert_eq!(at(&im2, 0, 0), 5);
        assert_eq!(at(&im2, 2, 2), 5);
        assert_eq!(at(&im2, 3, 3), 0);
    }

    /// A huge but in-range rectangle (fill and outline) terminates quickly by
    /// clipping its iteration to the canvas, instead of spinning through
    /// billions of off-canvas no-op writes (DoS regression). The painted
    /// pixels match libvips `vips_draw_rect`, which clips the rect to the
    /// image the same way.
    #[test]
    fn draw_rect_huge_extent_clips_and_terminates() {
        use std::time::Instant;

        // Fill: an extent 1e6 x 1e6 on a 10x10 canvas covers every pixel and
        // must finish essentially instantly. Unclipped this is ~1e12 writes.
        let mut fill = black(10, 10);
        let t0 = Instant::now();
        fill.draw_rect_filled(&[255], 0, 0, 1_000_000, 1_000_000);
        assert!(
            t0.elapsed().as_secs() < 5,
            "huge filled rect must clip, not iterate its full extent"
        );
        assert!(
            fill.data().iter().all(|&b| b == 255),
            "vips parity: a huge fill covers the whole canvas"
        );

        // Outline: a rect whose top-left corner is at (5,5) with a huge size
        // paints only the on-canvas top and left edges (the far edges are
        // off-canvas), exactly matching the oracle's CSV for the same call.
        let mut outline = black(10, 10);
        let t1 = Instant::now();
        outline.draw_rect(&[255], 5, 5, 1_000_000, 1_000_000);
        assert!(
            t1.elapsed().as_secs() < 5,
            "huge rect outline must clip, not iterate its full extent"
        );
        for x in 5..10 {
            assert_eq!(at(&outline, x, 5), 255, "top edge ({x},5)");
        }
        for y in 5..10 {
            assert_eq!(at(&outline, 5, y), 255, "left edge (5,{y})");
        }
        // Interior and everything left/above the corner stay background.
        assert_eq!(at(&outline, 6, 6), 0, "interior untouched");
        assert_eq!(at(&outline, 4, 5), 0, "left of the corner untouched");
        assert_eq!(at(&outline, 9, 9), 0, "far corner has no edge on-canvas");
    }

    /// A huge but in-range circle (filled and outline) terminates quickly by
    /// clipping each scanline to the canvas, instead of running an O(radius^2)
    /// span walk (filled) or spinning off-canvas (both). The painted pixels
    /// match libvips `vips_draw_circle`, which clips every scanline the same
    /// way. Regression for the `draw_circle --fill` DoS (#493).
    #[test]
    fn draw_circle_huge_radius_clips_and_terminates() {
        use std::time::Instant;

        // Filled: centre on a 10x10 canvas with a 1e6 radius covers every
        // pixel. Unclipped the inner span walk is ~radius per octant step, so
        // ~1e12 writes; clipped it is O(radius) octant steps of <=10px each.
        let mut fill = black(10, 10);
        let t0 = Instant::now();
        fill.draw_circle_filled(&[255], 5, 5, 1_000_000);
        assert!(
            t0.elapsed().as_secs() < 5,
            "huge filled circle must clip its scanlines, not iterate their full width"
        );
        assert!(
            fill.data().iter().all(|&b| b == 255),
            "vips parity: a huge filled disc covers the whole canvas"
        );

        // A circle whose bounding box is wholly off-canvas paints nothing and
        // returns immediately, never walking the O(radius) octant. Both fill
        // and outline short-circuit.
        let mut off_fill = black(10, 10);
        let t1 = Instant::now();
        off_fill.draw_circle_filled(&[255], -2_000_000_000, -2_000_000_000, 1_000_000_000);
        assert!(
            t1.elapsed().as_secs() < 5,
            "off-canvas filled circle short-circuits"
        );
        assert!(
            off_fill.data().iter().all(|&b| b == 0),
            "nothing painted off-canvas"
        );

        let mut off_outline = black(10, 10);
        let t2 = Instant::now();
        off_outline.draw_circle(&[255], 2_000_000_000, 2_000_000_000, 1_000_000_000);
        assert!(
            t2.elapsed().as_secs() < 5,
            "off-canvas circle outline short-circuits"
        );
        assert!(
            off_outline.data().iter().all(|&b| b == 0),
            "nothing painted off-canvas"
        );
    }

    /// A line whose bounding box is wholly off-canvas paints nothing and
    /// returns before the Bresenham walk, so a max-coordinate off-canvas line
    /// can't spin through ~1e9 no-op steps. On-canvas lines are unaffected
    /// (they still walk in full, matching vips's pointwise clipping).
    #[test]
    fn draw_line_offcanvas_bbox_short_circuits() {
        use std::time::Instant;

        let mut im = black(10, 10);
        let t0 = Instant::now();
        // Entirely to the right of the canvas: max-coordinate endpoints.
        im.draw_line(&[255], 1_000_000_000, 0, 1_000_000_000, 1_000_000_000);
        assert!(
            t0.elapsed().as_secs() < 5,
            "off-canvas line must short-circuit, not walk its full extent"
        );
        assert!(
            im.data().iter().all(|&b| b == 0),
            "nothing painted off-canvas"
        );
    }

    /// Degenerate inputs are safe no-ops: negative radius, zero-size rect.
    #[test]
    fn degenerate_ops_are_noops() {
        let mut im = black(8, 8);
        im.draw_circle(&[1], 4, 4, -1);
        im.draw_rect(&[1], 0, 0, 0, 5);
        im.draw_rect_filled(&[1], 0, 0, 5, 0);
        assert!(im.data().iter().all(|&b| b == 0));
    }

    /// The extensibility seam composes: a custom `DrawOp` from outside the
    /// module drives the same `Raster::draw` entry point as the built-ins, and
    /// a `&dyn DrawOp` erases the concrete type without changing behaviour.
    #[test]
    fn custom_draw_op_composes() {
        struct Diagonal<'a> {
            ink: &'a [u8],
        }
        impl DrawOp for Diagonal<'_> {
            fn apply(&self, raster: &mut Raster) {
                let n = raster.width().min(raster.height());
                for i in 0..n as i32 {
                    raster.put_pixel(i, i, self.ink);
                }
            }
        }

        let mut im = black(5, 5);
        let op = Diagonal { ink: &[255] };
        // Drive it both as a concrete op and behind a trait object.
        im.draw(&op);
        let dynamic: &dyn DrawOp = &op;
        im.draw(dynamic);
        for i in 0..5 {
            assert_eq!(at(&im, i, i), 255);
        }
        assert_eq!(at(&im, 0, 4), 0);
    }

    /// Count of non-background pixels in a Gray8 raster.
    fn inked(im: &Raster) -> usize {
        im.data().iter().filter(|&&b| b != 0).count()
    }

    // ---- line ----

    /// The ported reference: a horizontal line across row 0. The far endpoint
    /// (100, 0) is off-canvas and clips silently.
    #[test]
    fn draw_line_ported_reference() {
        let mut im = black(100, 100);
        im.draw_line(&[100], 0, 0, 100, 0);
        assert_eq!(at(&im, 0, 0), 100);
        assert_eq!(at(&im, 99, 0), 100);
        assert_eq!(at(&im, 0, 1), 0, "row 1 untouched");
        assert_eq!(inked(&im), 100, "exactly row 0 painted");
    }

    /// Both endpoints are painted, and a perfect diagonal visits exactly the
    /// lattice diagonal.
    #[test]
    fn draw_line_diagonal_inclusive_endpoints() {
        let mut im = black(5, 5);
        im.draw_line(&[9], 0, 0, 4, 4);
        for i in 0..5 {
            assert_eq!(at(&im, i, i), 9, "diagonal pixel ({i},{i})");
        }
        assert_eq!(inked(&im), 5);
    }

    /// A steep line paints exactly one pixel per row between the endpoint
    /// rows, with x monotonic between the endpoint columns.
    #[test]
    fn draw_line_steep_one_pixel_per_row() {
        let mut im = black(8, 8);
        im.draw_line(&[7], 1, 0, 3, 7);
        for y in 0..8u32 {
            let xs: Vec<u32> = (0..8u32).filter(|&x| at(&im, x, y) != 0).collect();
            assert_eq!(xs.len(), 1, "row {y} should hold exactly one pixel");
            assert!((1..=3).contains(&xs[0]), "row {y} x within endpoint span");
        }
        assert_eq!(at(&im, 1, 0), 7, "start endpoint");
        assert_eq!(at(&im, 3, 7), 7, "end endpoint");
    }

    /// Right-to-left horizontal, vertical, and degenerate one-point lines.
    #[test]
    fn draw_line_reverse_vertical_and_point() {
        let mut im = black(10, 10);
        im.draw_line(&[5], 9, 2, 0, 2);
        assert_eq!(inked(&im), 10, "reversed horizontal paints the whole row");

        let mut im = black(10, 10);
        im.draw_line(&[5], 4, 1, 4, 8);
        for y in 1..=8 {
            assert_eq!(at(&im, 4, y), 5);
        }
        assert_eq!(inked(&im), 8);

        let mut im = black(10, 10);
        im.draw_line(&[5], 3, 3, 3, 3);
        assert_eq!(at(&im, 3, 3), 5);
        assert_eq!(inked(&im), 1);
    }

    /// Lines with off-canvas endpoints clip per pixel and never panic.
    #[test]
    fn draw_line_clips_offcanvas_endpoints() {
        let mut im = black(8, 8);
        im.draw_line(&[255], -5, -5, 12, 12);
        for i in 0..8 {
            assert_eq!(at(&im, i, i), 255);
        }
        assert_eq!(inked(&im), 8);
    }

    // ---- flood ----

    /// The ported reference: flooding the interior of an outlined circle
    /// reproduces the filled circle exactly.
    #[test]
    fn draw_flood_outline_matches_filled_circle() {
        let mut im = black(100, 100);
        im.draw_circle(&[100], 50, 50, 25);
        im.draw_flood(&[100], 50, 50).unwrap();

        let mut filled = black(100, 100);
        filled.draw_circle_filled(&[100], 50, 50, 25);

        assert_eq!(im.data(), filled.data(), "flooded outline == filled disc");
    }

    /// The same equivalence holds across small radii, pinning the shared
    /// octant walk between outline and fill.
    #[test]
    fn draw_flood_outline_matches_filled_circle_small_radii() {
        for r in 1..=12 {
            let mut outlined = black(32, 32);
            outlined.draw_circle(&[100], 15, 15, r);
            outlined.draw_flood(&[100], 15, 15).unwrap();

            let mut filled = black(32, 32);
            filled.draw_circle_filled(&[100], 15, 15, r);

            assert_eq!(outlined.data(), filled.data(), "radius {r}");
        }
    }

    /// A bounded flood stops at the ink wall: only the seed's chamber fills.
    #[test]
    fn draw_flood_bounded_fills_only_seed_chamber() {
        let mut im = black(9, 5);
        // Vertical ink wall at x = 4.
        im.draw_line(&[100], 4, 0, 4, 4);
        im.draw_flood(&[100], 1, 2).unwrap();
        for y in 0..5 {
            for x in 0..9 {
                let expect = if x <= 4 { 100 } else { 0 };
                assert_eq!(at(&im, x, y), expect, "pixel ({x},{y})");
            }
        }
    }

    /// Flood is 4-connected: a diagonal-only gap does not leak, matching
    /// libvips connectivity.
    #[test]
    fn draw_flood_is_4_connected() {
        let mut im = black(2, 2);
        im.put_pixel(1, 0, &[100]);
        im.put_pixel(0, 1, &[100]);
        im.draw_flood(&[100], 0, 0).unwrap();
        assert_eq!(at(&im, 0, 0), 100, "seed painted");
        assert_eq!(at(&im, 1, 1), 0, "diagonal neighbour must not fill");
    }

    /// Seeding a bounded flood on an ink pixel is a no-op, and flooding an
    /// outlined rectangle reproduces the filled rectangle.
    #[test]
    fn draw_flood_rect_outline_and_ink_seed() {
        let mut im = black(20, 20);
        im.draw_rect(&[100], 5, 5, 10, 8);
        let boundary = im.clone();
        // Seed on the boundary: nothing changes.
        im.draw_flood(&[100], 5, 5).unwrap();
        assert_eq!(im.data(), boundary.data(), "ink seed is a no-op");
        // Seed inside: the outline pours full.
        im.draw_flood(&[100], 9, 8).unwrap();
        let mut filled = black(20, 20);
        filled.draw_rect_filled(&[100], 5, 5, 10, 8);
        assert_eq!(im.data(), filled.data());
    }

    /// All four ported out-of-bounds seeds error with the typed seed error.
    #[test]
    fn draw_flood_out_of_bounds_errors() {
        let mut im = black(100, 100);
        for (x, y) in [(200, 50), (50, 200), (-1, 50), (50, -1)] {
            let err = im.draw_flood(&[100], x, y).unwrap_err();
            assert!(
                matches!(
                    err,
                    DrawError::SeedOutOfBounds {
                        image_w: 100,
                        image_h: 100,
                        ..
                    }
                ),
                "seed ({x},{y})"
            );
            let msg = err.to_string();
            assert!(msg.contains("out of bounds"), "display: {msg}");
            let err = im.draw_flood_blob(&[100], x, y).unwrap_err();
            assert!(matches!(err, DrawError::SeedOutOfBounds { .. }));
        }
        assert!(im.data().iter().all(|&b| b == 0), "image untouched");
    }

    /// As a raw DrawOp (no Result surface), an off-canvas seed is a silent
    /// no-op, keeping the seam infallible.
    #[test]
    fn flood_op_offcanvas_seed_is_noop() {
        let mut im = black(4, 4);
        im.draw(&Flood::bounded(&[9], -1, 0));
        im.draw(&Flood::blob(&[9], 4, 4));
        assert!(im.data().iter().all(|&b| b == 0));
    }

    /// Blob flood recolours exactly the equal-valued 4-connected region.
    #[test]
    fn draw_flood_blob_recolours_equal_region() {
        let mut im = black(20, 20);
        im.draw_rect_filled(&[7], 2, 2, 4, 4);
        // Seed inside the 7-blob: only the blob changes.
        im.draw_flood_blob(&[200], 3, 3).unwrap();
        let mut expect = black(20, 20);
        expect.draw_rect_filled(&[200], 2, 2, 4, 4);
        assert_eq!(im.data(), expect.data());

        // Seed in the background: the connected zero region floods, flowing
        // around (but not into) the blob.
        let mut im = black(20, 20);
        im.draw_rect_filled(&[7], 2, 2, 4, 4);
        im.draw_flood_blob(&[9], 0, 0).unwrap();
        for y in 0..20 {
            for x in 0..20 {
                let inside = (2..6).contains(&x) && (2..6).contains(&y);
                assert_eq!(at(&im, x, y), if inside { 7 } else { 9 }, "({x},{y})");
            }
        }
    }

    /// Blob flood whose blob already wears the ink colour terminates as a
    /// no-op instead of looping.
    #[test]
    fn draw_flood_blob_seed_equal_ink_is_noop() {
        let mut im = black(8, 8);
        im.draw_rect_filled(&[7], 1, 1, 3, 3);
        let before = im.clone();
        im.draw_flood_blob(&[7], 2, 2).unwrap();
        assert_eq!(im.data(), before.data());
        im.draw_flood_blob(&[0], 7, 7).unwrap();
        assert_eq!(im.data(), before.data());
    }

    /// Multi-band floods compare whole pixels: a one-channel difference is a
    /// different colour.
    #[test]
    fn draw_flood_blob_multiband_exact_match() {
        let mut im = Raster::zeroed(4, 1, PixelFormat::Rgb8).unwrap();
        im.put_pixel(0, 0, &[10, 20, 30]);
        im.put_pixel(1, 0, &[10, 20, 30]);
        im.put_pixel(2, 0, &[10, 20, 31]);
        im.draw_flood_blob(&[1, 2, 3], 0, 0).unwrap();
        assert_eq!(im.getpoint(0, 0), vec![1.0, 2.0, 3.0]);
        assert_eq!(im.getpoint(1, 0), vec![1.0, 2.0, 3.0]);
        assert_eq!(
            im.getpoint(2, 0),
            vec![10.0, 20.0, 31.0],
            "near-match stays"
        );
        assert_eq!(im.getpoint(3, 0), vec![0.0, 0.0, 0.0], "background stays");
    }

    // ---- mask ----

    /// The ported reference: ink 200 through a 128-valued circle stencil onto
    /// black equals a directly drawn circle of 100 (200 * 128 / 255 = 100).
    #[test]
    fn draw_mask_ported_reference() {
        let mut mask = black(51, 51);
        mask.draw_circle_filled(&[128], 25, 25, 25);

        let mut im = black(100, 100);
        im.draw_mask(&[200], &mask, 25, 25);

        let mut expect = black(100, 100);
        expect.draw_circle_filled(&[100], 50, 50, 25);

        assert_eq!(im.data(), expect.data());
    }

    /// Mask extremes: 255 writes the ink verbatim, 0 leaves the pixel alone,
    /// and a partial mask blends against the existing value with truncating
    /// division.
    #[test]
    fn draw_mask_opacity_extremes_and_blend() {
        let mut mask = black(3, 1);
        mask.put_pixel(0, 0, &[255]);
        mask.put_pixel(1, 0, &[0]);
        mask.put_pixel(2, 0, &[128]);

        let mut im = black(3, 1);
        im.draw_rect_filled(&[100], 0, 0, 3, 1);
        im.draw_mask(&[200], &mask, 0, 0);

        assert_eq!(at(&im, 0, 0), 200, "opaque mask writes ink");
        assert_eq!(at(&im, 1, 0), 100, "transparent mask leaves pixel");
        // (200 * 128 + 100 * 127) / 255 = 150.19 -> 150.
        assert_eq!(at(&im, 2, 0), 150, "partial mask blends");
    }

    /// The mask clips where it overhangs the canvas.
    #[test]
    fn draw_mask_clips() {
        let mut mask = black(4, 4);
        mask.draw_rect_filled(&[255], 0, 0, 4, 4);
        let mut im = black(4, 4);
        im.draw_mask(&[9], &mask, -2, -2);
        for y in 0..4 {
            for x in 0..4 {
                let expect = if x < 2 && y < 2 { 9 } else { 0 };
                assert_eq!(at(&im, x, y), expect, "({x},{y})");
            }
        }
    }

    /// A mask that is not single-band 8-bit is a documented no-op.
    #[test]
    fn draw_mask_requires_single_band_8bit_mask() {
        let rgb_mask = Raster::zeroed(4, 4, PixelFormat::Rgb8).unwrap();
        let gray16_mask = Raster::zeroed(4, 4, PixelFormat::Gray16).unwrap();
        let mut im = black(4, 4);
        im.draw_mask(&[200], &rgb_mask, 0, 0);
        im.draw_mask(&[200], &gray16_mask, 0, 0);
        assert!(im.data().iter().all(|&b| b == 0));
    }

    /// Multi-band targets blend every channel against the same mask value.
    /// Mask 51 divides 255 exactly (ink * 51 / 255 = ink / 5).
    #[test]
    fn draw_mask_rgb_target_blends_per_channel() {
        let mut mask = black(1, 1);
        mask.put_pixel(0, 0, &[51]);
        let mut im = Raster::zeroed(2, 1, PixelFormat::Rgb8).unwrap();
        im.draw_mask(&[200, 100, 50], &mask, 0, 0);
        assert_eq!(im.getpoint(0, 0), vec![40.0, 20.0, 10.0]);
        assert_eq!(im.getpoint(1, 0), vec![0.0, 0.0, 0.0], "outside the mask");
    }

    /// 16-bit targets blend on decoded channel values: an opaque mask equals
    /// put_pixel, and mask 51 scales the decoded value by exactly 1/5.
    #[test]
    fn draw_mask_16bit_target() {
        let ink = 10_000u16.to_ne_bytes();

        let mut opaque = black(1, 1);
        opaque.put_pixel(0, 0, &[255]);
        let mut im = Raster::zeroed(1, 1, PixelFormat::Gray16).unwrap();
        im.draw_mask(&ink, &opaque, 0, 0);
        assert_eq!(im.getpoint(0, 0), vec![10_000.0]);

        let mut fifth = black(1, 1);
        fifth.put_pixel(0, 0, &[51]);
        let mut im = Raster::zeroed(1, 1, PixelFormat::Gray16).unwrap();
        im.draw_mask(&ink, &fifth, 0, 0);
        assert_eq!(im.getpoint(0, 0), vec![2_000.0]);
    }

    // ---- smudge ----

    /// The ported reference: smudging a region and pasting the original
    /// region back restores the original image, proving the smudge writes
    /// only inside its rect.
    #[test]
    fn draw_smudge_then_restore_is_identity() {
        let mut im = black(100, 100);
        im.draw_circle_filled(&[100], 50, 50, 25);

        let mut smudged = im.clone();
        smudged.draw_smudge(10, 10, 50, 50);
        assert_ne!(smudged.data(), im.data(), "smudge must change the region");

        let patch = im.extract(10, 10, 50, 50).unwrap();
        smudged.draw_image(&patch, 10, 10);
        assert_eq!(smudged.data(), im.data());
    }

    /// Pixels outside the smudge rect are byte-identical afterwards, and a
    /// uniform region is a fixed point of the box blur.
    #[test]
    fn draw_smudge_touches_only_rect_and_fixes_uniform() {
        let mut im = black(30, 30);
        im.draw_circle_filled(&[100], 15, 15, 10);
        let before = im.clone();
        im.draw_smudge(5, 5, 10, 10);
        for y in 0..30u32 {
            for x in 0..30u32 {
                if !(5..15).contains(&x) || !(5..15).contains(&y) {
                    assert_eq!(at(&im, x, y), at(&before, x, y), "outside ({x},{y})");
                }
            }
        }

        let mut uniform = black(10, 10);
        uniform.draw_rect_filled(&[100], 0, 0, 10, 10);
        let expect = uniform.clone();
        uniform.draw_smudge(2, 2, 6, 6);
        assert_eq!(uniform.data(), expect.data(), "uniform is unchanged");
    }

    /// Known 3x3 means: windows clip to the canvas (dividing by the sampled
    /// count) and every mean reads the pre-smudge snapshot, so a symmetric
    /// input keeps a symmetric result regardless of scan order.
    #[test]
    fn draw_smudge_box_average_snapshot_semantics() {
        let mut im = Raster::new(3, 1, PixelFormat::Gray8, vec![0, 90, 0]).unwrap();
        im.draw_smudge(0, 0, 3, 1);
        // (0+90)/2 = 45, (0+90+0)/3 = 30, (90+0)/2 = 45. An in-place scan
        // would instead see the already-smudged 45 when averaging x = 1.
        assert_eq!(im.data(), &[45, 30, 45]);

        // Rounding: means round to nearest, .5 away from zero.
        let mut im = Raster::new(2, 1, PixelFormat::Gray8, vec![0, 1]).unwrap();
        im.draw_smudge(0, 0, 2, 1);
        assert_eq!(im.data(), &[1, 1]);
    }

    /// Multi-band smudges average each channel independently.
    #[test]
    fn draw_smudge_rgb_channels_independent() {
        let data = vec![0, 0, 0, 90, 30, 60, 0, 0, 0];
        let mut im = Raster::new(3, 1, PixelFormat::Rgb8, data).unwrap();
        im.draw_smudge(1, 0, 1, 1);
        assert_eq!(im.getpoint(0, 0), vec![0.0, 0.0, 0.0], "outside rect");
        assert_eq!(im.getpoint(1, 0), vec![30.0, 10.0, 20.0]);
        assert_eq!(im.getpoint(2, 0), vec![0.0, 0.0, 0.0], "outside rect");
    }

    /// A smudge rect overhanging the canvas clips, and degenerate rects are
    /// no-ops.
    #[test]
    fn draw_smudge_clips_and_degenerate_noop() {
        let mut im = black(6, 6);
        im.draw_circle_filled(&[100], 3, 3, 2);
        let before = im.clone();
        im.draw_smudge(-3, -3, 6, 6);
        for y in 3..6u32 {
            for x in 0..6u32 {
                assert_eq!(at(&im, x, y), at(&before, x, y), "below rect ({x},{y})");
            }
        }

        let mut im = before.clone();
        im.draw_smudge(0, 0, 0, 6);
        im.draw_smudge(0, 0, 6, -1);
        im.draw_smudge(10, 10, 3, 3);
        assert_eq!(im.data(), before.data());
    }

    // ---- image (paste) ----

    /// The ported reference: pasting a filled-circle sub-image reproduces the
    /// directly drawn circle.
    #[test]
    fn draw_image_ported_reference() {
        let mut small = black(51, 51);
        small.draw_circle_filled(&[100], 25, 25, 25);

        let mut im = black(100, 100);
        im.draw_image(&small, 25, 25);

        let mut expect = black(100, 100);
        expect.draw_circle_filled(&[100], 50, 50, 25);

        assert_eq!(im.data(), expect.data());
    }

    /// Paste replaces pixels (libvips "set" mode); it does not blend.
    #[test]
    fn draw_image_replaces_pixels() {
        let mut im = black(6, 6);
        im.draw_rect_filled(&[50], 0, 0, 6, 6);
        let zeros = black(2, 2);
        im.draw_image(&zeros, 2, 2);
        for y in 0..6u32 {
            for x in 0..6u32 {
                let inside = (2..4).contains(&x) && (2..4).contains(&y);
                assert_eq!(at(&im, x, y), if inside { 0 } else { 50 }, "({x},{y})");
            }
        }
    }

    /// Overlays clip at every edge, including fully negative origins.
    #[test]
    fn draw_image_clips() {
        let mut nine = black(4, 4);
        nine.draw_rect_filled(&[9], 0, 0, 4, 4);

        let mut im = black(8, 8);
        im.draw_image(&nine, -2, -2);
        im.draw_image(&nine, 6, 6);
        let mut painted = 0;
        for y in 0..8u32 {
            for x in 0..8u32 {
                let expect = (x < 2 && y < 2) || (x >= 6 && y >= 6);
                assert_eq!(at(&im, x, y) == 9, expect, "({x},{y})");
                painted += usize::from(expect);
            }
        }
        assert_eq!(painted, 8);

        // Entirely off-canvas: a no-op.
        let mut im = black(8, 8);
        im.draw_image(&nine, 8, 0);
        assert!(im.data().iter().all(|&b| b == 0));
    }

    /// Multi-band pastes copy whole pixels at the right offsets.
    #[test]
    fn draw_image_rgb() {
        let mut sub = Raster::zeroed(1, 1, PixelFormat::Rgb8).unwrap();
        sub.put_pixel(0, 0, &[1, 2, 3]);
        let mut im = Raster::zeroed(3, 3, PixelFormat::Rgb8).unwrap();
        im.draw_image(&sub, 1, 1);
        assert_eq!(im.getpoint(1, 1), vec![1.0, 2.0, 3.0]);
        assert_eq!(im.getpoint(0, 1), vec![0.0, 0.0, 0.0]);
    }

    /// A format-mismatched overlay is a documented no-op.
    #[test]
    fn draw_image_format_mismatch_is_noop() {
        let rgb = Raster::zeroed(2, 2, PixelFormat::Rgb8).unwrap();
        let mut im = black(4, 4);
        im.draw_rect_filled(&[50], 0, 0, 4, 4);
        let before = im.clone();
        im.draw_image(&rgb, 0, 0);
        assert_eq!(im.data(), before.data());
    }

    // ---- seam ----

    /// The new ops are ordinary DrawOps: a boxed heterogeneous batch replayed
    /// through the generic entry point matches the inherent-method sequence.
    #[test]
    fn new_ops_compose_through_the_seam() {
        let small = {
            let mut s = black(3, 3);
            s.draw_rect_filled(&[40], 0, 0, 3, 3);
            s
        };
        let mask = {
            let mut m = black(2, 2);
            m.draw_rect_filled(&[255], 0, 0, 2, 2);
            m
        };

        let ops: Vec<Box<dyn DrawOp>> = vec![
            Box::new(Line::new(&[100], 0, 6, 15, 6)),
            Box::new(Rectangle::outline(&[100], 2, 8, 6, 5)),
            Box::new(Flood::bounded(&[100], 4, 10)),
            Box::new(Paste::new(&small, 10, 10)),
            Box::new(Mask::new(&[200], &mask, 12, 2)),
            Box::new(Smudge::new(10, 10, 3, 3)),
        ];
        let mut via_seam = black(16, 16);
        for op in &ops {
            via_seam.draw(op.as_ref());
        }

        let mut via_methods = black(16, 16);
        via_methods.draw_line(&[100], 0, 6, 15, 6);
        via_methods.draw_rect(&[100], 2, 8, 6, 5);
        via_methods.draw_flood(&[100], 4, 10).unwrap();
        via_methods.draw_image(&small, 10, 10);
        via_methods.draw_mask(&[200], &mask, 12, 2);
        via_methods.draw_smudge(10, 10, 3, 3);

        assert_eq!(via_seam.data(), via_methods.data());
    }

    // ---- ink channel validation (#294) ----

    /// A channel-count / depth mismatch is a typed error, not silent
    /// corruption: the `try_draw_*` forms reject ink that is not exactly one
    /// pixel wide and leave the raster untouched (no cycling, no truncation,
    /// no panic).
    #[test]
    fn try_draw_rejects_mismatched_ink_length() {
        // RGB ink (3 bytes) on an Rgba8 raster (4 bytes/pixel): previously
        // this painted alpha from the red byte.
        let mut rgba = Raster::zeroed(8, 8, PixelFormat::Rgba8).unwrap();
        let err = rgba
            .try_draw_rect_filled(&[10, 20, 30], 1, 1, 4, 4)
            .unwrap_err();
        assert!(
            matches!(
                err,
                DrawError::InkLengthMismatch {
                    ink_len: 3,
                    pixel_bytes: 4,
                    format: PixelFormat::Rgba8,
                }
            ),
            "got {err:?}"
        );
        assert!(err.to_string().contains("does not match"), "{err}");
        assert!(
            rgba.data().iter().all(|&b| b == 0),
            "rejected draw must not touch the raster"
        );

        // One-byte ink on Gray16 (2 bytes/pixel): previously wrote a repeated
        // low byte.
        let mut g16 = Raster::zeroed(4, 4, PixelFormat::Gray16).unwrap();
        let err = g16.try_draw_circle(&[200], 2, 2, 1).unwrap_err();
        assert!(
            matches!(
                err,
                DrawError::InkLengthMismatch {
                    ink_len: 1,
                    pixel_bytes: 2,
                    ..
                }
            ),
            "got {err:?}"
        );
        assert!(g16.data().iter().all(|&b| b == 0));

        // Too-long ink truncates under the old contract; now it errors too.
        let mut gray = black(4, 4);
        assert!(matches!(
            gray.try_draw_line(&[100, 200], 0, 0, 3, 3).unwrap_err(),
            DrawError::InkLengthMismatch {
                ink_len: 2,
                pixel_bytes: 1,
                ..
            }
        ));
        // Flood wrappers validate ink up front as well.
        assert!(matches!(
            gray.draw_flood(&[1, 2], 0, 0).unwrap_err(),
            DrawError::InkLengthMismatch { .. }
        ));
        assert!(gray.data().iter().all(|&b| b == 0), "nothing painted");
    }

    /// The panicking `draw_*` forms turn the same mismatch into an immediate
    /// panic tagged with the op name, mirroring the crate's `expect_*` pattern.
    #[test]
    #[should_panic(expected = "draw_rect: ink of 3 bytes does not match")]
    fn draw_rect_panics_on_mismatched_ink() {
        let mut rgba = Raster::zeroed(4, 4, PixelFormat::Rgba8).unwrap();
        rgba.draw_rect(&[10, 20, 30], 0, 0, 2, 2);
    }

    /// Correct-length ink still draws correctly across depths, and the fill is
    /// byte-identical to the raw op path (no behaviour change for valid ink).
    #[test]
    fn correct_length_ink_draws_and_matches_raw_path() {
        // Gray8 pixel round-trip through the fallible entry point.
        let mut gray = black(8, 8);
        gray.try_draw_rect_filled(&[123], 2, 2, 1, 1).unwrap();
        assert_eq!(gray.getpoint(2, 2), vec![123.0]);
        assert_eq!(gray.getpoint(0, 0), vec![0.0], "outside the paint");

        // Rgba8: all four channels come from their own byte; alpha is not
        // stolen from red, and it round-trips exactly.
        let mut rgba = Raster::zeroed(8, 8, PixelFormat::Rgba8).unwrap();
        rgba.try_draw_rect_filled(&[10, 20, 30, 40], 1, 1, 3, 3)
            .unwrap();
        assert_eq!(rgba.getpoint(2, 2), vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(rgba.getpoint(0, 0), vec![0.0, 0.0, 0.0, 0.0]);

        // Gray16: a 2-byte native-endian ink round-trips to the 16-bit value.
        let ink16 = 4_000u16.to_ne_bytes();
        let mut g16 = Raster::zeroed(6, 6, PixelFormat::Gray16).unwrap();
        g16.try_draw_circle_filled(&ink16, 3, 3, 2).unwrap();
        assert_eq!(g16.getpoint(3, 3), vec![4_000.0]);

        // Byte-for-byte parity: the validated entry point, the panicking
        // wrapper, and the raw op all produce the same bytes for valid ink.
        let mut via_try = Raster::zeroed(8, 8, PixelFormat::Rgba8).unwrap();
        via_try.try_draw_rect(&[9, 8, 7, 6], 1, 1, 5, 4).unwrap();
        let mut via_panic = Raster::zeroed(8, 8, PixelFormat::Rgba8).unwrap();
        via_panic.draw_rect(&[9, 8, 7, 6], 1, 1, 5, 4);
        let mut via_raw = Raster::zeroed(8, 8, PixelFormat::Rgba8).unwrap();
        via_raw.draw(&Rectangle::outline(&[9, 8, 7, 6], 1, 1, 5, 4));
        assert_eq!(via_try.data(), via_raw.data());
        assert_eq!(via_panic.data(), via_raw.data());
    }

    // ---- draw(&op) path ink validation (#294 follow-up, #346) ----

    /// The raw `draw(&op)` / direct-construction path is guarded against the
    /// #294 ink corruption exactly like the `draw_*` wrappers: a wrong-width
    /// ink through the `put_pixel` mechanism (here a 3-byte RGB ink on a 4-byte
    /// `Rgba8` pixel) panics instead of cycling and painting alpha from red.
    /// The `try_draw_*` wrappers reject it up front and never reach `apply`, so
    /// the guard fires only on this infallible entry point.
    #[test]
    #[should_panic(expected = "does not match")]
    fn draw_op_rect_path_panics_on_mismatched_ink() {
        let mut rgba = Raster::zeroed(4, 4, PixelFormat::Rgba8).unwrap();
        rgba.draw(&Rectangle::filled(&[10, 20, 30], 0, 0, 2, 2));
    }

    /// Same guard through `Circle` (also `put_pixel`): a 1-byte ink on a 2-byte
    /// `Gray16` pixel would repeat the low byte across both channels.
    #[test]
    #[should_panic(expected = "does not match")]
    fn draw_op_circle_path_panics_on_mismatched_ink() {
        let mut g16 = Raster::zeroed(4, 4, PixelFormat::Gray16).unwrap();
        g16.draw(&Circle::filled(&[200], 2, 2, 1));
    }

    /// Same guard through `Line` (also `put_pixel`).
    #[test]
    #[should_panic(expected = "does not match")]
    fn draw_op_line_path_panics_on_mismatched_ink() {
        let mut rgba = Raster::zeroed(4, 4, PixelFormat::Rgba8).unwrap();
        rgba.draw(&Line::new(&[10, 20, 30], 0, 0, 3, 3));
    }

    /// Same guard through the `ink_pixel` mechanism used by `Mask`.
    #[test]
    #[should_panic(expected = "does not match")]
    fn draw_op_mask_path_panics_on_mismatched_ink() {
        let mut mask = black(2, 2);
        mask.draw_rect_filled(&[255], 0, 0, 2, 2);
        let mut rgba = Raster::zeroed(4, 4, PixelFormat::Rgba8).unwrap();
        rgba.draw(&Mask::new(&[10, 20, 30], &mask, 0, 0));
    }

    /// Same guard through `Flood`, whose `ink_pixel` + whole-pixel copy is the
    /// third distinct ink mechanism.
    #[test]
    #[should_panic(expected = "does not match")]
    fn draw_op_flood_path_panics_on_mismatched_ink() {
        let mut rgba = Raster::zeroed(4, 4, PixelFormat::Rgba8).unwrap();
        rgba.draw(&Flood::bounded(&[10, 20, 30], 0, 0));
    }

    /// The op-application guard is invisible to correct-length ink: every
    /// ink-bearing op painted through the raw `draw(&op)` path stays
    /// byte-identical to its validated wrapper, so valid draws are unchanged.
    #[test]
    fn guarded_ops_match_wrappers_for_correct_ink() {
        let ink = [9u8, 8, 7, 6]; // one Rgba8 pixel
        let mut mask = black(4, 4);
        mask.draw_rect_filled(&[200], 0, 0, 4, 4);

        let mut raw = Raster::zeroed(8, 8, PixelFormat::Rgba8).unwrap();
        raw.draw(&Rectangle::filled(&ink, 1, 1, 3, 3));
        raw.draw(&Circle::outline(&ink, 5, 5, 2));
        raw.draw(&Line::new(&ink, 0, 0, 7, 7));
        raw.draw(&Mask::new(&ink, &mask, 2, 2));

        let mut wrapped = Raster::zeroed(8, 8, PixelFormat::Rgba8).unwrap();
        wrapped.try_draw_rect_filled(&ink, 1, 1, 3, 3).unwrap();
        wrapped.try_draw_circle(&ink, 5, 5, 2).unwrap();
        wrapped.try_draw_line(&ink, 0, 0, 7, 7).unwrap();
        wrapped.try_draw_mask(&ink, &mask, 2, 2).unwrap();

        assert_eq!(raw.data(), wrapped.data());

        // Flood on a fresh raster (order-independent).
        let mut raw_f = Raster::zeroed(4, 4, PixelFormat::Rgba8).unwrap();
        raw_f.draw(&Flood::bounded(&ink, 0, 0));
        let mut wrapped_f = Raster::zeroed(4, 4, PixelFormat::Rgba8).unwrap();
        wrapped_f.draw_flood(&ink, 0, 0).unwrap();
        assert_eq!(raw_f.data(), wrapped_f.data());
    }

    // ---- try_draw_mask on float rasters (#346) ----

    /// The fallible `try_draw_mask` returns a typed error on a 32-bit-float
    /// target instead of panicking inside the unsigned-sample decode. The ink
    /// is correct width (4 x f32 = 16 bytes), so the float format is the only
    /// reason to fail, and the raster is left untouched.
    #[test]
    fn try_draw_mask_float_target_is_typed_error() {
        let mut mask = black(2, 2);
        mask.draw_rect_filled(&[255], 0, 0, 2, 2);
        let ink: Vec<u8> = [0.0f32; 4].iter().flat_map(|f| f.to_ne_bytes()).collect();

        let mut im = Raster::zeroed(2, 2, PixelFormat::RgbaF32).unwrap();
        let err = im.try_draw_mask(&ink, &mask, 0, 0).unwrap_err();
        assert!(
            matches!(
                err,
                DrawError::UnsupportedFloatRaster {
                    format: PixelFormat::RgbaF32,
                }
            ),
            "got {err:?}"
        );
        assert!(err.to_string().contains("float"), "{err}");
        assert!(
            im.data().iter().all(|&b| b == 0),
            "a rejected float draw must not touch the raster"
        );

        // FloatF32(n) is rejected as well.
        let mut im = Raster::from_f32_samples(1, 1, PixelFormat::RgbaF32, &[0.0; 4]).unwrap();
        assert!(matches!(
            im.try_draw_mask(&ink, &mask, 0, 0).unwrap_err(),
            DrawError::UnsupportedFloatRaster { .. }
        ));
    }

    /// The panicking `draw_mask` surfaces the float rejection as a tagged panic,
    /// matching the `# Panics` contract.
    #[test]
    #[should_panic(expected = "draw_mask: draw operation does not support")]
    fn draw_mask_float_target_panics() {
        let mut mask = black(2, 2);
        mask.draw_rect_filled(&[255], 0, 0, 2, 2);
        let ink: Vec<u8> = [0.0f32; 4].iter().flat_map(|f| f.to_ne_bytes()).collect();
        let mut im = Raster::zeroed(2, 2, PixelFormat::RgbaF32).unwrap();
        im.draw_mask(&ink, &mask, 0, 0);
    }
}
