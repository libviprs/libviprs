//! Issue #1330: a 16-bit source is converted to 8 bits before tiling and the report says so,
//! under the name `converted-bit-depth`. An 8-bit source gets no such line.

use libviprs::raster_pvec::pyramid::{
    RasterPlacement, RasterPyramidOptions, produce_raster_pyramid,
};
use libviprs::{Layout, PixelFormat, PyramidPlanner, Raster};

#[path = "common/tile_headers.rs"]
mod tile_headers;

fn run(format: PixelFormat) -> libviprs::raster_pvec::pyramid::RasterPyramid {
    let bpp = format.bytes_per_pixel();
    let data: Vec<u8> = (0..300 * 200 * bpp).map(|i| (i % 253) as u8).collect();
    let r = Raster::new(300, 200, format, data).unwrap();
    let plan = PyramidPlanner::new(300, 200, 256, 0, Layout::Google)
        .unwrap()
        .plan();
    let o = RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, 0.0, 0.0, -1.0, 0.0, 200.0]));
    produce_raster_pyramid(&r, &plan, &o).unwrap()
}

#[test]
fn a_16_bit_source_yields_8_bit_tiles_and_a_named_report_line() {
    for (format, name) in [
        (PixelFormat::Gray16, "Gray16 -> Gray8"),
        (PixelFormat::Rgb16, "Rgb16 -> Rgb8"),
        (PixelFormat::Rgba16, "Rgba16 -> Rgba8"),
    ] {
        let p = run(format);
        assert_eq!(p.bit_depth, 8);
        assert_eq!(
            p.report.conversions,
            vec![format!("converted-bit-depth: {name}")],
            "{format:?}"
        );
        for (_, _, _, bytes) in p.tiles() {
            assert_eq!(tile_headers::png(bytes).bit_depth, 8);
        }
    }
}

#[test]
fn an_8_bit_source_gets_no_line() {
    for format in [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8] {
        assert!(run(format).report.conversions.is_empty(), "{format:?}");
    }
}
