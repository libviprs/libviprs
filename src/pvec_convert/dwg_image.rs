//! Placing a DWG `IMAGE` as a raster (issue #1667).
//!
//! A DWG IMAGE names a file and says where the picture sits; it carries no pixels. When the caller
//! says where the files are ([`PvecConversion::images`](super::PvecConversion::images)) the file is
//! read from that directory by its file name, decoded, and cut into the pyramid a PVec raster
//! stream stores, placed with the map the IMAGE's insertion point and pixel vectors give, composed
//! with the map of the leaf (a block, a viewport, a flipped sheet), so it lands in the frame of
//! the linework around it.
//!
//! * Pixel (px, py) is the picture's pixel, top-left origin, and sits at
//!   `insert + u * size.x * (px / w) + v * size.y * (1 - py / h)`. The file's own size (w, h) wins
//!   over the entity's `size` when they differ: the frame stays where the drawing puts it and the
//!   pixels fill it, as AutoCAD does.
//! * The clip is in the entity's pixel units (top-left origin, y down, an edge is a stored vertex
//!   plus one half). That origin is an open assumption: acadrust (pinned rev 5bf63c7) only says
//!   "image pixel space", the default boundary is symmetric so it does not tell, and I found no
//!   spec text on hand that settles it, so it wants checking against a drawing with an off-centre clip. A pixel is kept when its centre is inside the boundary (outside it for an
//!   inverted clip), inside the box the conversion cuts to and inside the viewport's polygon. A kept
//!   area that is a pixel rectangle only sets `visible_box`; anything else is baked into alpha.
//! * The file is looked up by the last component of the path the drawing carries, in the directory
//!   the caller gave and nowhere else, ASCII case ignored when the exact name is not there.
//! * What is not carried: brightness, contrast, fade and the bitonal transparency flag.

use std::fs;
use std::io::Cursor;
use std::path::Path;

use ::image::{DynamicImage, ImageDecoder, ImageReader, Limits};

use crate::pixel::PixelFormat;
use crate::planner::{Layout, PyramidPlanner};
use crate::pvec::{ImageFrame, Point3};
use crate::raster::Raster;
use crate::raster_pvec::pyramid::{
    RasterPlacement, RasterPyramid, RasterPyramidOptions, produce_raster_pyramid,
};

/// The most pixels one placed image may have, as for a PDF image.
pub(super) const MAX_PIXELS: u64 = 64 << 20;
/// The most images one conversion places.
pub(super) const MAX_IMAGES: usize = 256;
/// The most pixels all the images of one conversion may have together.
pub(super) const MAX_TOTAL_PIXELS: u64 = 256 << 20;
/// The most memory the decoder may allocate for one image, bytes.
const MAX_ALLOC: u64 = 1 << 30;
/// The largest file read.
const MAX_FILE_BYTES: u64 = 256 << 20;

/// The pyramid of `frame`'s picture read from `dir`, or why the image is not placed. `map` is the
/// leaf's map from block space into the artifact's frame; `clip_box` and `ring` are what the
/// conversion cuts to there (the sheet or a viewport's rectangle, and its polygon). `max_pixels` is
/// what this image may have: the per-image cap, or what is left of the conversion's total. The size
/// is read from the header and refused before a pixel is allocated.
pub(super) fn place(
    frame: &ImageFrame,
    map: impl Fn(Point3) -> [f64; 2],
    dir: &Path,
    clip_box: Option<[f64; 4]>,
    ring: Option<&[[f64; 2]]>,
    max_pixels: u64,
) -> Result<RasterPyramid, String> {
    if !frame.visible {
        return Err("its display is switched off".to_owned());
    }
    let picture = read(dir, &frame.file_name, max_pixels.min(MAX_PIXELS))?;
    let (w, h) = (picture.width(), picture.height());
    let (wf, hf) = (f64::from(w), f64::from(h));

    // Pixel (0, 0), and one pixel along each axis, through the leaf's map.
    let at = |px: f64, py: f64| -> [f64; 2] {
        let (s, t) = (px / wf, 1.0 - py / hf);
        map([
            frame.insert[0] + frame.u[0] * frame.size[0] * s + frame.v[0] * frame.size[1] * t,
            frame.insert[1] + frame.u[1] * frame.size[0] * s + frame.v[1] * frame.size[1] * t,
            frame.insert[2] + frame.u[2] * frame.size[0] * s + frame.v[2] * frame.size[1] * t,
        ])
    };
    let (o, x1, y1) = (at(0.0, 0.0), at(1.0, 0.0), at(0.0, 1.0));
    let m = [
        x1[0] - o[0],
        y1[0] - o[0],
        x1[1] - o[1],
        y1[1] - o[1],
        o[0],
        o[1],
    ];
    let det = m[0] * m[3] - m[1] * m[2];
    if !m.iter().all(|v| v.is_finite())
        || m[..4].iter().any(|v| v.abs() > f64::from(1u32 << 20))
        || det == 0.0
        || !det.is_finite()
    {
        return Err("its frame is not a placeable map".to_owned());
    }
    let m = m.map(|v| if v == 0.0 { 0.0 } else { v });

    // The clip in pixel edges of the file's grid.
    let (kx, ky) = (wf / frame.size[0], hf / frame.size[1]);
    let boundary: Vec<[f64; 2]> = match frame.clip.as_slice() {
        [] => Vec::new(),
        [a, b] => {
            let (x0, x1) = ((a[0].min(b[0]) + 0.5) * kx, (a[0].max(b[0]) + 0.5) * kx);
            let (y0, y1) = ((a[1].min(b[1]) + 0.5) * ky, (a[1].max(b[1]) + 0.5) * ky);
            vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        }
        many => many
            .iter()
            .map(|v| [(v[0] + 0.5) * kx, (v[1] + 0.5) * ky])
            .collect(),
    };
    if boundary.iter().flatten().any(|v| !v.is_finite()) {
        return Err("its clip boundary is not finite".to_owned());
    }
    let to_sheet = |cx: f64, cy: f64| [m[0] * cx + m[1] * cy + m[4], m[2] * cx + m[3] * cy + m[5]];
    let keep = |px: u32, py: u32| -> bool {
        let (cx, cy) = (f64::from(px) + 0.5, f64::from(py) + 0.5);
        if !boundary.is_empty() && inside(&boundary, [cx, cy]) == frame.clip_inverted {
            return false;
        }
        if clip_box.is_none() && ring.is_none() {
            return true;
        }
        let s = to_sheet(cx, cy);
        clip_box.is_none_or(|b| s[0] >= b[0] && s[0] <= b[2] && s[1] >= b[1] && s[1] <= b[3])
            && ring.is_none_or(|r| inside(r, s))
    };

    let cut = !boundary.is_empty() || clip_box.is_some() || ring.is_some();
    let (mut lo, mut hi, mut kept) = ([w, h], [0u32, 0u32], 0u64);
    let mut mask = Vec::new();
    if cut {
        mask.reserve(w as usize * h as usize);
        for py in 0..h {
            for px in 0..w {
                let k = keep(px, py);
                mask.push(k);
                if k {
                    lo = [lo[0].min(px), lo[1].min(py)];
                    hi = [hi[0].max(px + 1), hi[1].max(py + 1)];
                    kept += 1;
                }
            }
        }
        if kept == 0 {
            return Err("its clip leaves nothing".to_owned());
        }
    } else {
        (lo, hi) = ([0, 0], [w, h]);
    }
    let visible_box = [lo[0], lo[1], hi[0], hi[1]];
    // A kept area that is exactly its box needs no alpha.
    let boxed = u64::from(hi[0] - lo[0]) * u64::from(hi[1] - lo[1]) == kept;

    let alpha = picture.color().has_alpha() || (cut && !boxed);
    let (format, mut data) = if alpha {
        (PixelFormat::Rgba8, picture.into_rgba8().into_raw())
    } else {
        (PixelFormat::Rgb8, picture.into_rgb8().into_raw())
    };
    if cut && !boxed {
        for (k, px) in mask.iter().zip(data.chunks_exact_mut(4)) {
            if !k {
                px[3] = 0;
            }
        }
    }
    let raster = Raster::new(w, h, format, data).map_err(|e| e.to_string())?;
    let plan = PyramidPlanner::new(w, h, 256, 0, Layout::Google)
        .map_err(|e| e.to_string())?
        .plan();
    let mut pyramid = produce_raster_pyramid(
        &raster,
        &plan,
        &RasterPyramidOptions::new(RasterPlacement::Explicit(m)),
    )
    .map_err(|e| e.to_string())?;
    pyramid.visible_box = visible_box;
    Ok(pyramid)
}

/// The picture the drawing's path names, from `dir` by its last component. The file has to resolve,
/// symlinks followed, to a regular file inside `dir`, and its header size has to be within
/// `max_pixels` before anything is decoded.
fn read(dir: &Path, path: &str, max_pixels: u64) -> Result<DynamicImage, String> {
    let name = path.rsplit(['\\', '/']).next().unwrap_or_default();
    if name.is_empty() || name == "." || name == ".." || name.contains('\0') {
        return Err(format!("the path {path:?} names no file"));
    }
    let root = fs::canonicalize(dir).map_err(|e| format!("the image directory: {e}"))?;
    let exact = dir.join(name);
    let found = if fs::symlink_metadata(&exact).is_ok() {
        exact
    } else {
        fs::read_dir(dir)
            .map_err(|e| format!("{name} is not in the image directory ({e})"))?
            .filter_map(Result::ok)
            .find(|e| e.file_name().to_string_lossy().eq_ignore_ascii_case(name))
            .map(|e| e.path())
            .ok_or_else(|| format!("{name} is not in the image directory"))?
    };
    let file = fs::canonicalize(&found).map_err(|e| format!("{name}: {e}"))?;
    if !file.starts_with(&root) {
        return Err(format!("{name} resolves outside the image directory"));
    }
    let meta = fs::metadata(&file).map_err(|e| format!("{name}: {e}"))?;
    if !meta.is_file() {
        return Err(format!("{name} is not a regular file"));
    }
    if meta.len() > MAX_FILE_BYTES {
        return Err(format!(
            "{name} is {} bytes, over the {MAX_FILE_BYTES} limit",
            meta.len()
        ));
    }
    let bytes = fs::read(&file).map_err(|e| format!("{name}: {e}"))?;
    let mut decoder = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("{name}: {e}"))?
        .into_decoder()
        .map_err(|e| format!("{name} does not decode ({e})"))?;
    let (w, h) = decoder.dimensions();
    if w == 0 || h == 0 || u64::from(w) * u64::from(h) > max_pixels {
        return Err(format!(
            "{name} declares {w} x {h} pixels, over the {max_pixels} it may have"
        ));
    }
    let mut limits = Limits::default();
    limits.max_alloc = Some(MAX_ALLOC);
    decoder
        .set_limits(limits)
        .map_err(|e| format!("{name}: {e}"))?;
    DynamicImage::from_decoder(decoder).map_err(|e| format!("{name} does not decode ({e})"))
}

/// Even-odd crossing count of `p` against the closed polygon `poly`.
fn inside(poly: &[[f64; 2]], p: [f64; 2]) -> bool {
    let mut inside = false;
    for (i, a) in poly.iter().enumerate() {
        let b = &poly[(i + 1) % poly.len()];
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
    }
    inside
}
