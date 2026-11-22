//! A PNG of a published PVec artifact (issue #1629): strokes only, black on opaque white, at the
//! width the caller asks for.
//!
//! The picture is the artifact's declared bounds, so a paper layout (whose bounds are the paper)
//! fills the image edge to edge and a model-space artifact (whose bounds are the geometry's
//! extent) gets the margin the conversion asks for. The height follows the bounds' aspect, so a
//! long thin sheet gives a tall or a flat image. Y is up: the bottom of the bounds is the bottom
//! row. Text is not drawn, and every stroke is one pixel wide.

use std::path::Path;

use image::codecs::png::PngEncoder;
use image::{ExtendedColorType, ImageEncoder};
use pvec_facade::Curve;
use pvec_geometry::Curve2d;
use resvg::tiny_skia::{Color, Paint, PathBuilder, Pixmap, Stroke, Transform};
use thiserror::Error;

use super::publish::{OpenError, open_published};

/// Why a thumbnail could not be drawn.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ThumbnailError {
    /// The width was zero.
    #[error("a thumbnail needs a width of at least one pixel")]
    ZeroWidth,
    /// The artifact could not be read.
    #[error("the artifact could not be read: {0}")]
    Artifact(String),
    /// The picture could not be allocated or encoded.
    #[error("the picture could not be made: {0}")]
    Raster(String),
}

/// The artifact's curves as a PNG `width_px` wide, the height following the artifact's bounds.
/// The bounds fill the image with no margin.
///
/// # Errors
///
/// [`ThumbnailError`] for a zero width or an artifact that cannot be read.
pub fn thumbnail(artifact: &Path, width_px: u32) -> Result<Vec<u8>, ThumbnailError> {
    if width_px == 0 {
        return Err(ThumbnailError::ZeroWidth);
    }
    let reader =
        open_published(artifact).map_err(|e: OpenError| ThumbnailError::Artifact(e.to_string()))?;
    let bounds = reader.manifest().bounds;
    let curves = reader
        .entities()
        .map_err(|e| ThumbnailError::Artifact(e.to_string()))?;
    draw(
        bounds.min,
        bounds.max,
        curves.iter().map(|(_, c)| c),
        width_px,
        0.0,
    )
}

/// Draws `curves` inside the box `lo` to `hi` into a PNG. `margin` is the share of the longer
/// side of the box left empty on every side.
pub(crate) fn draw<'a>(
    lo: [f64; 2],
    hi: [f64; 2],
    curves: impl Iterator<Item = &'a Curve>,
    width_px: u32,
    margin: f64,
) -> Result<Vec<u8>, ThumbnailError> {
    let longer = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    let pad = if longer.is_finite() {
        longer * margin
    } else {
        0.0
    };
    let (x0, y0) = (lo[0] - pad, lo[1] - pad);
    let (w, h) = (hi[0] - lo[0] + 2.0 * pad, hi[1] - lo[1] + 2.0 * pad);
    // A bounds with no extent at all (a single point) has no aspect, so the picture is square.
    let (w, h) = if w > 0.0 && h > 0.0 {
        (w, h)
    } else {
        (1.0, 1.0)
    };
    let height_px = ((f64::from(width_px) * h / w).round() as u32).max(1);
    let (sx, sy) = (f64::from(width_px) / w, f64::from(height_px) / h);
    let to_px = |x: f64, y: f64| {
        (
            ((x - x0) * sx) as f32,
            (f64::from(height_px) - (y - y0) * sy) as f32,
        )
    };

    let mut path = PathBuilder::new();
    for curve in curves {
        stroke_curve(&mut path, curve, &to_px);
    }
    let mut pixmap = Pixmap::new(width_px, height_px)
        .ok_or_else(|| ThumbnailError::Raster(format!("{width_px} x {height_px} is not a size")))?;
    pixmap.fill(Color::WHITE);
    if let Some(path) = path.finish() {
        let mut paint = Paint::default();
        paint.set_color(Color::BLACK);
        paint.anti_alias = true;
        let stroke = Stroke {
            width: 1.0,
            ..Stroke::default()
        };
        pixmap.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    }
    // The background is opaque, so dropping the alpha loses nothing.
    let rgb: Vec<u8> = pixmap
        .data()
        .chunks_exact(4)
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&rgb, width_px, height_px, ExtendedColorType::Rgb8)
        .map_err(|e| ThumbnailError::Raster(e.to_string()))?;
    Ok(png)
}

/// How many segments a curved piece is drawn with.
const CURVED: usize = 96;

/// Adds the curve as one or more polylines. A bare line has no ends (a trimmed one does), so it
/// has nothing to draw.
fn stroke_curve(path: &mut PathBuilder, curve: &Curve, to_px: &dyn Fn(f64, f64) -> (f32, f32)) {
    let mut run = |points: &mut dyn Iterator<Item = (f64, f64)>| {
        let mut first = true;
        for (x, y) in points {
            let (px, py) = to_px(x, y);
            if !(px.is_finite() && py.is_finite()) {
                first = true;
            } else if first {
                path.move_to(px, py);
                first = false;
            } else {
                path.line_to(px, py);
            }
        }
    };
    let mut along = |c: &dyn Curve2d, n: usize| {
        let d = c.domain();
        let mut pts = (0..=n).filter_map(|i| {
            let t = d.lo() + d.width() * (i as f64) / (n as f64);
            c.eval(t.min(d.hi())).ok().map(|p| (p.x, p.y))
        });
        run(&mut pts);
    };
    match curve {
        Curve::Line(_) => {}
        Curve::Trimmed(t) if matches!(t.basis(), Curve::Line(_)) => along(t, 1),
        Curve::Polyline(p) => run(&mut p.vertices().iter().map(|v| (v.x, v.y))),
        Curve::Composite(c) => {
            for s in c.segments() {
                stroke_curve(path, s, to_px);
            }
        }
        other => along(other, CURVED),
    }
}
