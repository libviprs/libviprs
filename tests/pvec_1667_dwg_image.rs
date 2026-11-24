//! Issue #1667: a DWG IMAGE is placed as a raster. The frame was counted (#1334) and never drawn, so
//! a logo in a title block left a blank patch on the converted sheet.
//!
//! With `PvecConversion::images(dir)` the file the IMAGE's definition names is read from `dir` (by
//! its file name, never by the path the drawing carries) and written into the artifact's raster
//! stream at the IMAGE's insertion point, scale and rotation, cut to its clip, in the same frame as
//! the linework around it. Without the option nothing changes, so #1328 still holds.
//!
//! The drawings are written by acadrust's `DwgWriter` and converted through the real pipeline; each
//! test reads the placement and the tile back from the artifact and checks pixels in sheet space.
//! Pixel space of a clip boundary is taken as top-left origin, y down, pixel edges at integers
//! (the stored vertices are in pixel-centre units, so an edge is the vertex plus one half).
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::{Path, PathBuf};

use acadrust::entities::{ClipBoundary, Line, RasterImage};
use acadrust::objects::{ImageDefinition, ObjectType};
use acadrust::types::{Vector2, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::ViewProfile;
use libviprs::pvec::publish::open_published;
use libviprs::pvec_convert::{PvecConversion, PvecConversionOutput, PvecLayoutSet};
use pvec_facade::Curve;
use pvec_facade::unstable::RasterId;

const SHEET: &str = "Sheet A";

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("libviprs-1667-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// The colour of pixel (x, y) of the logo: every pixel is told apart by its red and green.
fn logo_pixel(x: u32, y: u32) -> [u8; 3] {
    [(60 * x + 30) as u8, (60 * y + 30) as u8, 128]
}

/// `logo.png`, 4 by 4 pixels, written into `dir`.
fn write_logo(dir: &Path) {
    let img = image::RgbaImage::from_fn(4, 4, |x, y| {
        let [r, g, b] = logo_pixel(x, y);
        image::Rgba([r, g, b, 255])
    });
    img.save(dir.join("logo.png")).unwrap();
}

/// A 4 by 4 pixel image of 10 units a pixel, its lower-left corner at (50, 40).
fn logo(def_path: &str) -> RasterImage {
    let mut img = RasterImage::new(def_path, Vector3::new(50.0, 40.0, 0.0), 4.0, 4.0);
    img.u_vector = Vector3::new(10.0, 0.0, 0.0);
    img.v_vector = Vector3::new(0.0, 10.0, 0.0);
    img
}

/// An A3 sheet, millimetres, 1 : 1, with a line along the bottom of the logo and the IMAGE.
fn sheet(def_path: &str, image: RasterImage) -> Vec<u8> {
    let mut doc = CadDocument::new();
    let layout = doc.add_layout(SHEET).expect("layout");
    let Some(ObjectType::Layout(l)) = doc.objects.get_mut(&layout) else {
        panic!("the layout object");
    };
    l.paper_size = "ISO_A3_(420.00_x_297.00_MM)".into();
    l.paper_width = 420.0;
    l.paper_height = 297.0;
    l.plot_paper_units = 1;
    l.plot_type = 5;
    l.plot_scale_numerator = 1.0;
    l.plot_scale_denominator = 1.0;
    l.plot_rotation = 0;
    l.plot_margin_left = 0.0;
    l.plot_margin_bottom = 0.0;
    l.plot_origin_x = 0.0;
    l.plot_origin_y = 0.0;
    l.plot_flags.print_lineweights = true;

    let mut def = ImageDefinition::new(def_path);
    def.handle = doc.allocate_handle();
    def.size_in_pixels = (4, 4);
    def.pixel_size = (1.0, 1.0);
    def.is_loaded = true;
    let def_handle = def.handle;
    doc.objects
        .insert(def_handle, ObjectType::ImageDefinition(def));

    let mut image = image;
    image.definition_handle = Some(def_handle);
    doc.add_entity_to_layout(
        EntityType::Line(Line::from_points(
            Vector3::new(50.0, 40.0, 0.0),
            Vector3::new(90.0, 40.0, 0.0),
        )),
        SHEET,
    )
    .expect("line");
    doc.add_entity_to_layout(EntityType::RasterImage(image), SHEET)
        .expect("image");
    let mut bytes = Vec::new();
    DwgWriter::write_to_writer(std::io::Cursor::new(&mut bytes), &doc).expect("write");
    bytes
}

fn convert(bytes: &[u8], dir: &Path, images: Option<&Path>) -> PvecConversionOutput {
    let dest = dir.join("sheet.pvec");
    let mut c = PvecConversion::new(CadSource::Bytes(bytes), &dest)
        .view_profile(ViewProfile::print_profile(SHEET));
    if let Some(images) = images {
        c = c.images(images);
    }
    c.convert().expect("the sheet converts")
}

/// One raster read back from an artifact: its placement and the full-resolution tile.
struct Placed {
    pixel_to_source: [f64; 6],
    visible_box: [u32; 4],
    size: [u32; 2],
    tile: image::RgbaImage,
}

impl Placed {
    /// The colour the sheet shows at (x, y) from this raster, or `None` where it draws nothing:
    /// outside the image, outside its visible box or on a transparent pixel.
    fn colour_at(&self, x: f64, y: f64) -> Option<[u8; 3]> {
        let [a, b, c, d, tx, ty] = self.pixel_to_source;
        let (dx, dy) = (x - tx, y - ty);
        let det = a * d - b * c;
        let px = (d * dx - b * dy) / det;
        let py = (-c * dx + a * dy) / det;
        if px < 0.0 || py < 0.0 || px >= f64::from(self.size[0]) || py >= f64::from(self.size[1]) {
            return None;
        }
        let (ix, iy) = (px.floor() as u32, py.floor() as u32);
        let [x0, y0, x1, y1] = self.visible_box;
        if ix < x0 || ix >= x1 || iy < y0 || iy >= y1 {
            return None;
        }
        let p = self.tile.get_pixel(ix, iy).0;
        (p[3] != 0).then_some([p[0], p[1], p[2]])
    }
}

fn rasters(out: &PvecConversionOutput) -> Vec<Placed> {
    let reader = open_published(&out.artifact).unwrap();
    reader
        .raster_placements()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(n, pl)| {
            let top = u32::from(pl.levels) - 1;
            let bytes = reader
                .raster_tile(RasterId::new(n as u32).unwrap(), top, 0, 0)
                .unwrap()
                .expect("the tile is stored");
            Placed {
                pixel_to_source: pl.pixel_to_source,
                visible_box: pl.visible_box,
                size: [pl.width, pl.height],
                tile: image::load_from_memory(&bytes).unwrap().to_rgba8(),
            }
        })
        .collect()
}

/// A trimmed line as (origin, unit direction, length).
fn segment(c: &Curve) -> ([f64; 2], [f64; 2], f64) {
    let Curve::Trimmed(t) = c else {
        panic!("a trimmed line, not {c:?}");
    };
    let Curve::Line(l) = t.basis() else {
        panic!("a trimmed line, not {c:?}");
    };
    let (o, d) = (l.origin(), l.direction());
    ([o.x, o.y], [d.x, d.y], t.trim().span)
}

fn near2(got: [f64; 2], want: [f64; 2]) {
    assert!(
        (got[0] - want[0]).abs() < 1e-6 && (got[1] - want[1]).abs() < 1e-6,
        "{got:?} is not {want:?}"
    );
}

fn near(got: [f64; 6], want: [f64; 6]) {
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "{got:?} is not {want:?}");
    }
}

/// The centre of pixel (x, y) of an unrotated logo in sheet space.
fn centre(x: u32, y: u32) -> (f64, f64) {
    (
        50.0 + 10.0 * (f64::from(x) + 0.5),
        80.0 - 10.0 * (f64::from(y) + 0.5),
    )
}

// ---- insertion and scale ----

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_an_image_is_placed_at_its_insertion_point_and_scale() {
    let dir = scratch("place");
    write_logo(&dir);
    let bytes = sheet("logo.png", logo("logo.png"));
    let out = convert(&bytes, &dir, Some(&dir));
    let placed = rasters(&out);
    assert_eq!(placed.len(), 1, "one IMAGE, one raster");
    // Pixel (0, 0) is the top-left corner, which is the insertion point plus four v vectors.
    near(
        placed[0].pixel_to_source,
        [10.0, 0.0, 0.0, -10.0, 50.0, 80.0],
    );
    assert_eq!(placed[0].size, [4, 4]);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_every_pixel_lands_where_the_drawing_puts_it() {
    let dir = scratch("pixels");
    write_logo(&dir);
    let bytes = sheet("logo.png", logo("logo.png"));
    let placed = rasters(&convert(&bytes, &dir, Some(&dir)));
    assert_eq!(placed.len(), 1, "the IMAGE is placed as one raster");
    for y in 0..4 {
        for x in 0..4 {
            let (sx, sy) = centre(x, y);
            assert_eq!(
                placed[0].colour_at(sx, sy),
                Some(logo_pixel(x, y)),
                "pixel ({x}, {y})"
            );
        }
    }
    // Just outside the frame on each side draws nothing.
    for (x, y) in [(49.0, 60.0), (91.0, 60.0), (70.0, 39.0), (70.0, 81.0)] {
        assert_eq!(placed[0].colour_at(x, y), None, "({x}, {y})");
    }
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_a_pixel_size_other_than_one_scales_the_raster() {
    let dir = scratch("scale");
    write_logo(&dir);
    let mut img = logo("logo.png");
    img.u_vector = Vector3::new(2.5, 0.0, 0.0);
    img.v_vector = Vector3::new(0.0, 5.0, 0.0);
    let bytes = sheet("logo.png", img);
    let placed = rasters(&convert(&bytes, &dir, Some(&dir)));
    assert_eq!(placed.len(), 1, "the IMAGE is placed as one raster");
    near(placed[0].pixel_to_source, [2.5, 0.0, 0.0, -5.0, 50.0, 60.0]);
    fs::remove_dir_all(&dir).unwrap();
}

// ---- rotation ----

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_a_rotated_image_is_placed_rotated() {
    let dir = scratch("rotate");
    write_logo(&dir);
    // Turned a quarter anticlockwise: u points up the sheet, v points left.
    let mut img = logo("logo.png");
    img.insertion_point = Vector3::new(100.0, 40.0, 0.0);
    img.u_vector = Vector3::new(0.0, 10.0, 0.0);
    img.v_vector = Vector3::new(-10.0, 0.0, 0.0);
    let bytes = sheet("logo.png", img);
    let placed = rasters(&convert(&bytes, &dir, Some(&dir)));
    assert_eq!(placed.len(), 1, "the IMAGE is placed as one raster");
    // Pixel (0, 0) sits at the insertion point plus four v vectors, (60, 40).
    near(
        placed[0].pixel_to_source,
        [0.0, 10.0, 10.0, 0.0, 60.0, 40.0],
    );
    // The top-left pixel is now the one nearest the sheet's bottom-left of the turned frame.
    assert_eq!(placed[0].colour_at(65.0, 45.0), Some(logo_pixel(0, 0)));
    // The bottom-right pixel is across and up from it.
    assert_eq!(placed[0].colour_at(95.0, 75.0), Some(logo_pixel(3, 3)));
    fs::remove_dir_all(&dir).unwrap();
}

// ---- clip ----

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_a_rectangular_clip_cuts_the_visible_box() {
    let dir = scratch("clip-rect");
    write_logo(&dir);
    let mut img = logo("logo.png");
    img.clipping_enabled = true;
    // Pixel columns 1 and 2, rows 0 and 1: edges x 1..3, y 0..2, so vertices half a pixel less.
    img.clip_boundary = ClipBoundary::rectangular(Vector2::new(0.5, -0.5), Vector2::new(2.5, 1.5));
    let bytes = sheet("logo.png", img);
    let placed = rasters(&convert(&bytes, &dir, Some(&dir)));
    assert_eq!(placed.len(), 1, "the IMAGE is placed as one raster");
    assert_eq!(placed[0].visible_box, [1, 0, 3, 2]);
    for (x, y, shown) in [
        (0, 0, false),
        (1, 0, true),
        (2, 1, true),
        (3, 1, false),
        (1, 2, false),
    ] {
        let (sx, sy) = centre(x, y);
        assert_eq!(
            placed[0].colour_at(sx, sy).is_some(),
            shown,
            "pixel ({x}, {y})"
        );
    }
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_a_polygonal_clip_makes_the_outside_transparent() {
    let dir = scratch("clip-poly");
    write_logo(&dir);
    let mut img = logo("logo.png");
    img.clipping_enabled = true;
    // The triangle (0, 0), (4, 0), (0, 4) in pixel edges: a pixel is in when x + y < 3.
    img.clip_boundary = ClipBoundary::polygonal(vec![
        Vector2::new(-0.5, -0.5),
        Vector2::new(3.5, -0.5),
        Vector2::new(-0.5, 3.5),
    ]);
    let bytes = sheet("logo.png", img);
    let placed = rasters(&convert(&bytes, &dir, Some(&dir)));
    assert_eq!(placed.len(), 1, "the IMAGE is placed as one raster");
    for y in 0..4u32 {
        for x in 0..4u32 {
            let (sx, sy) = centre(x, y);
            let want = (x + y < 3).then(|| logo_pixel(x, y));
            assert_eq!(placed[0].colour_at(sx, sy), want, "pixel ({x}, {y})");
        }
    }
    fs::remove_dir_all(&dir).unwrap();
}

// ---- the frame of the linework ----

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_the_raster_sits_in_the_frame_of_the_linework() {
    let dir = scratch("frame");
    write_logo(&dir);
    let bytes = sheet("logo.png", logo("logo.png"));
    let out = convert(&bytes, &dir, Some(&dir));
    let placed = rasters(&out);
    assert_eq!(placed.len(), 1, "the IMAGE is placed as one raster");
    // The line runs along the bottom of the IMAGE, (50, 40) to (90, 40).
    let curves = open_published(&out.artifact).unwrap().entities().unwrap();
    assert_eq!(curves.len(), 1);
    let (o, d, len) = segment(&curves[0].1);
    let start = o;
    let end = [o[0] + d[0] * len, o[1] + d[1] * len];
    // The raster's bottom-left corner (pixel x 0, y 4) is the line's start, its bottom-right the end.
    let [a, b, c, e, tx, ty] = placed[0].pixel_to_source;
    let at = |px: f64, py: f64| [a * px + b * py + tx, c * px + e * py + ty];
    near2(at(0.0, 4.0), start);
    near2(at(4.0, 4.0), end);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_a_layout_set_places_the_image_on_its_sheet() {
    let dir = scratch("set");
    write_logo(&dir);
    let out_dir = dir.join("out");
    fs::create_dir(&out_dir).unwrap();
    let bytes = sheet("logo.png", logo("logo.png"));
    let images = dir.clone();
    let set = PvecLayoutSet::new(CadSource::Bytes(&bytes), &out_dir)
        .stem("plan")
        .sheets(move |c| c.images(&images))
        .convert()
        .expect("the set converts");
    let sheet = set.written().next().expect("one sheet");
    let done = sheet.outcome.as_ref().unwrap();
    assert_eq!(rasters(done).len(), 1);
    fs::remove_dir_all(&dir).unwrap();
}

// ---- what is not placed, and says so ----

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_without_the_option_nothing_is_placed() {
    let dir = scratch("off");
    write_logo(&dir);
    let bytes = sheet("logo.png", logo("logo.png"));
    let out = convert(&bytes, &dir, None);
    assert!(rasters(&out).is_empty());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_the_drawings_own_path_is_never_followed() {
    // The definition names a path with directories, the Windows kind: only the file name is looked
    // up, in the directory the caller gave.
    let dir = scratch("path");
    write_logo(&dir);
    let bytes = sheet(
        "C:\\clients\\oct\\logos\\logo.png",
        logo("C:\\clients\\oct\\logos\\logo.png"),
    );
    assert_eq!(rasters(&convert(&bytes, &dir, Some(&dir))).len(), 1);
    // A path that climbs out of the directory finds nothing.
    let bytes = sheet("..\\..\\etc\\logo.png", logo("..\\..\\etc\\logo.png"));
    let sub = dir.join("only");
    fs::create_dir(&sub).unwrap();
    let out = convert(&bytes, &dir, Some(&sub));
    assert!(rasters(&out).is_empty());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_a_missing_file_is_reported_on_the_row() {
    let dir = scratch("missing");
    let bytes = sheet("logo.png", logo("logo.png"));
    let out = convert(&bytes, &dir, Some(&dir));
    assert!(rasters(&out).is_empty());
    let row = out
        .report
        .entities
        .iter()
        .find(|r| r.source_class == "IMAGE")
        .expect("a row for the IMAGE");
    assert!(
        row.warnings
            .iter()
            .any(|w| w.contains("IMAGE") && w.contains("logo.png")),
        "{row:#?}"
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_a_hidden_image_is_not_drawn() {
    let dir = scratch("hidden");
    write_logo(&dir);
    let mut img = logo("logo.png");
    img.set_visible(false);
    let bytes = sheet("logo.png", img);
    assert!(rasters(&convert(&bytes, &dir, Some(&dir))).is_empty());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1667_a_placed_image_says_so_on_its_row() {
    let dir = scratch("row");
    write_logo(&dir);
    let bytes = sheet("logo.png", logo("logo.png"));
    let out = convert(&bytes, &dir, Some(&dir));
    let row = out
        .report
        .entities
        .iter()
        .find(|r| r.source_class == "IMAGE")
        .expect("a row for the IMAGE");
    assert!(
        row.warnings
            .iter()
            .any(|w| w.contains("placed as a raster")),
        "{row:#?}"
    );
    fs::remove_dir_all(&dir).unwrap();
}
