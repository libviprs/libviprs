//! The archive the web viewer's specs load (issue #1033).
//!
//! The viewer's visual regression is only meaningful if the archive under it
//! does not drift, so the committed bytes are pinned here: this test rebuilds
//! the archive and fails if it differs. Run with
//! `LIBVIPRS_UPDATE_VIEWER_FIXTURE=1` to rewrite it after an intended change,
//! and regenerate the viewer's snapshots in the same commit.

use libviprs::cad::{Arc, Circle, Ellipse, Line, Polyline, Primitive, Spline, Text};
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
    let _ = std::fs::remove_dir_all(&dir);

    let committed = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    if std::env::var_os("LIBVIPRS_UPDATE_VIEWER_FIXTURE").is_some() {
        std::fs::write(&committed, &bytes).unwrap();
        return;
    }
    let have = std::fs::read(&committed).expect("the viewer fixture is committed");
    assert!(
        have == bytes,
        "{FIXTURE} is stale ({} bytes vs {} rebuilt); rerun with LIBVIPRS_UPDATE_VIEWER_FIXTURE=1 and regenerate the viewer snapshots",
        have.len(),
        bytes.len()
    );
    assert!(
        bytes.len() < 200_000,
        "the fixture is {} bytes",
        bytes.len()
    );
}
