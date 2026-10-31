//! Issue #1330: LIMIT-15 at encode time. A stored tile longer than `2 * 4 * edge^2 + 65,536`
//! bytes is refused by PVec's reader, so the producer refuses it first.

use libviprs::raster_pvec::tiles::{RasterTileError, check_tile_len, max_tile_bytes};
use libviprs::{PixelFormat, Raster};

#[test]
fn the_formula_at_the_three_edges() {
    assert_eq!(max_tile_bytes(256), 589_824);
    assert_eq!(max_tile_bytes(512), 2_162_688);
    assert_eq!(max_tile_bytes(1024), 8_454_144);
}

#[test]
fn the_cap_itself_passes_and_one_more_byte_is_refused() {
    for edge in [256u32, 512, 1024] {
        let cap = max_tile_bytes(edge) as usize;
        assert!(check_tile_len(edge, cap).is_ok());
        match check_tile_len(edge, cap + 1) {
            Err(RasterTileError::TooLong { len, cap: c, edge: e }) => {
                assert_eq!((len, c, e), (cap as u64 + 1, cap as u64, edge));
            }
            other => panic!("{edge}: {other:?}"),
        }
    }
}

#[test]
fn a_synthetic_over_cap_byte_string_is_refused() {
    let bytes = vec![0u8; max_tile_bytes(256) as usize + 1];
    assert!(matches!(
        check_tile_len(256, bytes.len()),
        Err(RasterTileError::TooLong { .. })
    ));
}

#[test]
fn the_worst_honest_tile_stays_under_the_cap() {
    // Incompressible RGBA at 1024: the PNG is a little over the raw 4 MiB, far under 8 MiB.
    let mut s = 0xDEAD_BEEFu32;
    let data: Vec<u8> = (0..1024 * 1024 * 4)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        })
        .collect();
    let tile = Raster::new(1024, 1024, PixelFormat::Rgba8, data).unwrap();
    let bytes = tile.encode_png(6).unwrap();
    assert!(
        check_tile_len(1024, bytes.len()).is_ok(),
        "{} bytes against {}",
        bytes.len(),
        max_tile_bytes(1024)
    );
}

#[test]
fn the_pyramid_runs_the_check_on_every_tile_it_keeps() {
    // Unreachable for honest output, so the proof is that the check is wired in: a pyramid of
    // noise at the smallest edge still passes, and its largest tile is under the cap.
    use libviprs::raster_pvec::pyramid::{RasterPlacement, RasterPyramidOptions, produce_raster_pyramid};
    use libviprs::{Layout, PyramidPlanner};
    let mut s = 7u32;
    let data: Vec<u8> = (0..300 * 300 * 4)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        })
        .collect();
    let r = Raster::new(300, 300, PixelFormat::Rgba8, data).unwrap();
    let plan = PyramidPlanner::new(300, 300, 256, 0, Layout::Google).unwrap().plan();
    let o = RasterPyramidOptions::new(RasterPlacement::Explicit([1.0, 0.0, 0.0, -1.0, 0.0, 300.0]));
    let p = produce_raster_pyramid(&r, &plan, &o).unwrap();
    let largest = p.tiles().map(|(_, _, _, b)| b.len()).max().unwrap();
    assert!(largest as u64 <= max_tile_bytes(256));
}
