//! Issue #1330: every seed libviprs writes passes PVec's three tile sniffers, and the sniffers
//! really are checking depth and channels (a seed under the wrong placement is refused). This is
//! the interop check of the acceptance criteria; the automated version for whole pyramids is
//! libviprs/libviprs-tests#305. Needs `--features pvec-raster`.

#![cfg(feature = "pvec-raster")]

use pvec_facade::unstable::{
    Band, PixelFormat as Channels, Placement, RasterId, TileCodec, TileIn,
};
use pvec_facade::{Bounds2, Curve, EntityId, Manifest, PvecReader, PvecWriter};
use pvec_geometry::{Point2, Polyline};

const DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/pvec-raster-seeds"
);

fn placement(codec: TileCodec, channels: Channels, bit_depth: u8) -> Placement {
    Placement {
        pixel_to_source: [1.0, 0.0, 0.0, -1.0, 0.0, 256.0],
        width: 256,
        height: 256,
        visible_box: [0, 0, 256, 256],
        tile_edge: 256,
        levels: 1,
        tile_codec: codec,
        pixel_format: channels,
        bit_depth,
        opacity: 255,
        band: Band::Underlay,
        layer_index: None,
        source_digest: None,
        attribution: String::new(),
    }
}

/// Write one tile under `p`; the artifact comes back only if the writer took it.
fn through_writer(p: &Placement, bytes: Vec<u8>) -> Option<PvecReader> {
    let mut w = PvecWriter::new(Manifest {
        metres_per_unit: 1.0,
        bounds: Bounds2 {
            min: [0.0, 0.0],
            max: [256.0, 256.0],
        },
        capabilities: Vec::new(),
        producer_id: "libviprs-1330".into(),
        producer_version: "1".into(),
    })
    .unwrap();
    let ring = Polyline::new(vec![
        Point2 { x: 0.0, y: 0.0 },
        Point2 { x: 256.0, y: 0.0 },
        Point2 { x: 256.0, y: 256.0 },
        Point2 { x: 0.0, y: 0.0 },
    ])
    .unwrap();
    w.push(EntityId::new(0).unwrap(), &Curve::Polyline(ring))
        .unwrap();
    w.rasters(std::slice::from_ref(p)).ok()?;
    w.raster_tiles(
        RasterId::new(0).unwrap(),
        vec![TileIn {
            level: 0,
            col: 0,
            row: 0,
            bytes,
        }],
    )
    .ok()?;
    PvecReader::open(w.finish().ok()?).ok()
}

fn seed(name: &str) -> Vec<u8> {
    std::fs::read(format!("{DIR}/{name}")).unwrap()
}

#[test]
fn every_seed_passes_the_sniffer_for_its_codec_and_comes_back_byte_equal() {
    let cases = [
        ("png-gray1.png", TileCodec::Png, Channels::Gray, 1),
        ("png-gray8.png", TileCodec::Png, Channels::Gray, 8),
        ("png-rgb8.png", TileCodec::Png, Channels::Rgb, 8),
        ("png-rgba8.png", TileCodec::Png, Channels::Rgba, 8),
        (
            "jpeg-gray8-baseline.jpg",
            TileCodec::Jpeg,
            Channels::Gray,
            8,
        ),
        ("jpeg-rgb8-baseline.jpg", TileCodec::Jpeg, Channels::Rgb, 8),
        ("webp-vp8l-rgb8.webp", TileCodec::Webp, Channels::Rgb, 8),
        ("webp-vp8l-rgba8.webp", TileCodec::Webp, Channels::Rgba, 8),
    ];
    for (name, codec, channels, depth) in cases {
        let bytes = seed(name);
        let reader = through_writer(&placement(codec, channels, depth), bytes.clone())
            .unwrap_or_else(|| panic!("{name} was refused"));
        let got = reader
            .raster_tile(RasterId::new(0).unwrap(), 0, 0, 0)
            .unwrap()
            .expect("stored");
        assert_eq!(got, bytes, "{name}");
    }
}

#[test]
fn the_sniffers_check_depth_and_channels_so_the_pass_above_means_something() {
    // A 1-bit tile under an 8-bit placement, an RGB tile under a gray one, a gray JPEG under RGB.
    assert!(
        through_writer(
            &placement(TileCodec::Png, Channels::Gray, 8),
            seed("png-gray1.png")
        )
        .is_none()
    );
    assert!(
        through_writer(
            &placement(TileCodec::Png, Channels::Gray, 8),
            seed("png-rgb8.png")
        )
        .is_none()
    );
    assert!(
        through_writer(
            &placement(TileCodec::Jpeg, Channels::Rgb, 8),
            seed("jpeg-gray8-baseline.jpg")
        )
        .is_none()
    );
    // Trailing bytes after the stream end are refused (the polyglot rule).
    let mut long = seed("png-gray8.png");
    long.extend_from_slice(b"trailing");
    assert!(through_writer(&placement(TileCodec::Png, Channels::Gray, 8), long).is_none());
}
