//! Extensible in-place raster drawing.
//!
//! This module is the drawing seam for [`Raster`]. Every shape or paint
//! operation is a value that implements the [`DrawOp`] trait, whose single
//! [`apply`](DrawOp::apply) method mutates a `Raster` in place. New ops (line,
//! polygon, ellipse, flood-fill, gradient, ...) plug in by adding one `impl
//! DrawOp` and, optionally, one convenience method on `Raster`; the core
//! [`Raster`] type and the existing ops never change. This keeps the drawing
//! surface open for extension and closed for modification.
//!
//! # Why a trait, not four hard-coded methods
//!
//! A `DrawOp` is a first-class, inspectable value. Callers can build a
//! `Vec<Box<dyn DrawOp>>` and replay it, wrap an op to log or clip it, or ship
//! a custom op from a downstream crate without a libviprs release. The built-in
//! shapes ([`Circle`], [`Rectangle`]) are ordinary implementors with no
//! privileged access, so a third-party op is exactly as capable as a built-in
//! one.
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
//! raster libraries. Drawing is therefore always infallible.
//!
//! # Ink
//!
//! `ink` is the raw pixel value to paint, as bytes. It is written verbatim to
//! each affected pixel, cycling if it is shorter than the pixel's byte width
//! (so `&[100]` fills a `Gray8` pixel, and `&[r, g, b]` fills an `Rgb8` one).
//! Ink longer than one pixel is truncated to the pixel width.
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

/// A circle, drawn as an outline or filled disc.
///
/// Built with [`Circle::outline`] / [`Circle::filled`], or applied through the
/// [`Raster::draw_circle`] / [`Raster::draw_circle_filled`] convenience
/// methods. The outline is a 1px midpoint circle; the fill is the solid disc of
/// the same radius, so an outline and a fill of identical parameters share the
/// same boundary pixels.
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
        if self.radius < 0 {
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
        if self.width <= 0 || self.height <= 0 {
            return;
        }
        // `right`/`bottom` are the inclusive far edges. Widen to i64 so a large
        // `left + width` cannot overflow i32.
        let right = (self.left as i64 + self.width as i64 - 1) as i32;
        let bottom = (self.top as i64 + self.height as i64 - 1) as i32;
        if self.fill {
            for y in self.top..=bottom {
                for x in self.left..=right {
                    raster.put_pixel(x, y, self.ink);
                }
            }
        } else {
            for x in self.left..=right {
                raster.put_pixel(x, self.top, self.ink);
                raster.put_pixel(x, bottom, self.ink);
            }
            for y in self.top..=bottom {
                raster.put_pixel(self.left, y, self.ink);
                raster.put_pixel(right, y, self.ink);
            }
        }
    }
}

/// Midpoint-circle outline: plot the eight symmetric octant points.
fn outline_circle(raster: &mut Raster, ink: &[u8], cx: i32, cy: i32, radius: i32) {
    if radius == 0 {
        raster.put_pixel(cx, cy, ink);
        return;
    }
    let mut x = radius;
    let mut y = 0;
    // Decision variable for the midpoint algorithm.
    let mut err = 1 - radius;
    while x >= y {
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
        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

/// Solid disc: paint the horizontal span for every scanline the circle covers.
fn fill_circle(raster: &mut Raster, ink: &[u8], cx: i32, cy: i32, radius: i32) {
    let r2 = radius as i64 * radius as i64;
    for dy in -radius..=radius {
        // Half-width of the span at this row: floor(sqrt(r^2 - dy^2)).
        let rem = r2 - dy as i64 * dy as i64;
        if rem < 0 {
            continue;
        }
        let dx = isqrt_i64(rem) as i32;
        let y = cy + dy;
        for x in (cx - dx)..=(cx + dx) {
            raster.put_pixel(x, y, ink);
        }
    }
}

/// Integer square root (floor) for non-negative `n`. Avoids float rounding at
/// the disc boundary so `fill_circle` and `outline_circle` agree there.
fn isqrt_i64(n: i64) -> i64 {
    if n < 2 {
        return n;
    }
    let mut x = (n as f64).sqrt() as i64;
    // Correct any float rounding in either direction.
    while x * x > n {
        x -= 1;
    }
    while (x + 1) * (x + 1) <= n {
        x += 1;
    }
    x
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
    pub fn draw_circle(&mut self, ink: &[u8], cx: i32, cy: i32, radius: i32) {
        self.draw(&Circle::outline(ink, cx, cy, radius));
    }

    /// Draw a filled disc (see [`Circle::filled`]).
    pub fn draw_circle_filled(&mut self, ink: &[u8], cx: i32, cy: i32, radius: i32) {
        self.draw(&Circle::filled(ink, cx, cy, radius));
    }

    /// Draw a rectangle outline (see [`Rectangle::outline`]).
    pub fn draw_rect(&mut self, ink: &[u8], left: i32, top: i32, width: i32, height: i32) {
        self.draw(&Rectangle::outline(ink, left, top, width, height));
    }

    /// Draw a filled rectangle (see [`Rectangle::filled`]).
    pub fn draw_rect_filled(&mut self, ink: &[u8], left: i32, top: i32, width: i32, height: i32) {
        self.draw(&Rectangle::filled(ink, left, top, width, height));
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
}
