//! The archive the web viewer's specs load (issue #1033).
//!
//! The viewer's visual regression is only meaningful if the archive under it
//! does not drift, so this test rebuilds the archive and fails if what it
//! holds differs from the committed one. Run with
//! `LIBVIPRS_UPDATE_VIEWER_FIXTURE=1` to rewrite it after an intended change,
//! and regenerate the viewer's snapshots in the same commit.
//!
//! What is compared is the archive's *content*: the layer schema, the header
//! facts a viewer reads, the metadata, and every tile, inflated back to raw
//! MVT. It is not compared as file bytes, because the tiles are gzipped through
//! `flate2` and the compressed bytes belong to whichever backend the build
//! picked: `--features packfile` brings in `zip`, whose default features turn
//! `flate2` over to zlib-rs, and the same tiles then deflate to a different
//! (valid) stream, 79939 bytes against 77814 here (issue #1259). A byte-exact
//! check made this pass or fail by feature set, and said nothing about the
//! tiler.

use std::collections::BTreeMap;
use std::path::Path;

use libviprs::cad::{Arc, Circle, Ellipse, Line, Polyline, Primitive, Spline, Text};
use libviprs::pmtiles::Metadata;
use libviprs::vector::VectorArchive;
use libviprs::vector::archive::LayerSchema;
use libviprs::vector::tiler::{TilerConfig, build_archive};

const Z: [f64; 3] = [0.0, 0.0, 1.0];
const FIXTURE: &str = "web/vector-viewer/e2e/fixtures/golden.pmtiles";

fn line(a: [f64; 2], b: [f64; 2]) -> Primitive {
    Line::new([a[0], a[1], 0.0], [b[0], b[1], 0.0])
        .unwrap()
        .into()
}

/// A 1000 x 600 sheet: a frame, a circle, an arc, an ellipse, a spline, a
/// polyline with a bulge, a hatch, two diagonals and three labels.
fn drawing() -> Vec<Primitive> {
    let mut p: Vec<Primitive> = vec![
        Polyline::new(
            &[
                [0.0, 0.0, 0.0],
                [1000.0, 0.0, 0.0],
                [1000.0, 600.0, 0.0],
                [0.0, 600.0, 0.0],
            ],
            true,
            Z,
            &[],
        )
        .unwrap()
        .into(),
        Circle::new([300.0, 300.0, 0.0], 150.0, Z).unwrap().into(),
        Arc::new([700.0, 300.0, 0.0], 200.0, 0.4, 4.0, Z)
            .unwrap()
            .into(),
        Ellipse::new([500.0, 300.0, 0.0], [120.0, 40.0, 0.0], 0.5, 0.0, 6.0, Z)
            .unwrap()
            .into(),
        Spline::new(
            3,
            0,
            &[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            &[
                [50.0, 50.0, 0.0],
                [200.0, 500.0, 0.0],
                [400.0, 100.0, 0.0],
                [600.0, 550.0, 0.0],
                [900.0, 80.0, 0.0],
            ],
            &[],
        )
        .unwrap()
        .into(),
        Polyline::new(
            &[
                [100.0, 100.0, 0.0],
                [300.0, 120.0, 0.0],
                [350.0, 250.0, 0.0],
            ],
            false,
            Z,
            &[0.0, 0.7, 0.0],
        )
        .unwrap()
        .into(),
        line([0.0, 0.0], [1000.0, 600.0]),
        line([0.0, 600.0], [1000.0, 0.0]),
        Text::new([60.0, 540.0, 0.0], 30.0, 0.0, "GROUND FLOOR")
            .unwrap()
            .into(),
        Text::new([600.0, 20.0, 0.0], 12.0, std::f64::consts::FRAC_PI_2, "N")
            .unwrap()
            .into(),
        Text::new([700.0, 540.0, 0.0], 18.0, 0.0, "\u{2205}45 at 90\u{b0}")
            .unwrap()
            .into(),
    ];
    for i in 0..40 {
        let x = 600.0 + f64::from(i) * 5.0;
        p.push(line([x, 400.0], [x + 30.0, 470.0]));
    }
    p
}

/// Everything about an archive that does not depend on how its tiles were
/// compressed.
#[derive(Debug, PartialEq)]
struct Content {
    layers: Vec<LayerSchema>,
    zooms: (u8, u8),
    bounds: (f64, f64, f64, f64),
    addressed_tiles: u64,
    metadata: Metadata,
    /// Every tile that exists, as raw MVT.
    tiles: BTreeMap<(u8, u32, u32), Vec<u8>>,
}

fn content(path: &Path) -> Content {
    let archive = VectorArchive::try_open(path).expect("the archive opens");
    let reader = archive.pmtiles();
    let (min, max) = (reader.min_zoom(), reader.max_zoom());
    let mut tiles = BTreeMap::new();
    for z in min..=max {
        for x in 0..1u32 << z {
            for y in 0..1u32 << z {
                if let Some(bytes) = archive.tile_bytes(z, x, y).expect("a tile reads") {
                    tiles.insert((z, x, y), bytes);
                }
            }
        }
    }
    Content {
        layers: archive.layers().to_vec(),
        zooms: (min, max),
        bounds: reader.bounds(),
        addressed_tiles: reader.header().addressed_tiles_count,
        metadata: reader.metadata().expect("the metadata reads").clone(),
        tiles,
    }
}

#[test]
#[cfg_attr(miri, ignore)] // touches the filesystem, which Miri isolation blocks
fn the_committed_viewer_fixture_is_what_the_tiler_makes() {
    let dir = std::env::temp_dir().join(format!("fixture1033-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let built = dir.join("golden.pmtiles");
    build_archive(
        &drawing(),
        &TilerConfig::default().with_max_zoom(6),
        &built,
        "golden",
    )
    .unwrap();
    let bytes = std::fs::read(&built).unwrap();

    let committed = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    if std::env::var_os("LIBVIPRS_UPDATE_VIEWER_FIXTURE").is_some() {
        std::fs::write(&committed, &bytes).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    assert!(
        committed.exists(),
        "the viewer fixture is committed at {FIXTURE}"
    );
    let have = content(&committed);
    let rebuilt = content(&built);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !have.tiles.is_empty(),
        "positive control: the committed fixture holds no tiles, so the comparison below \
         would pass on two empty archives"
    );
    assert!(
        have == rebuilt,
        "{FIXTURE} is stale (it holds {} tiles, the tiler makes {}); rerun with \
         LIBVIPRS_UPDATE_VIEWER_FIXTURE=1 and regenerate the viewer snapshots",
        have.tiles.len(),
        rebuilt.tiles.len()
    );
    assert!(
        bytes.len() < 200_000,
        "the fixture is {} bytes",
        bytes.len()
    );
}
