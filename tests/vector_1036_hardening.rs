//! Hardening the vector tiler against hostile and pathological input
//! (issue #1036): cancellation, a wall-clock deadline, a bound on what one text
//! feature carries, and extreme values that must never panic.

use std::time::{Duration, Instant};

use libviprs::CancelToken;
use libviprs::cad::{
    Arc, Circle, DiagnosticCode, Ellipse, Line, Polyline, Primitive, Spline, Text,
};
use libviprs::vector::mvt::{Geometry, Value};
use libviprs::vector::tiler::{
    CollectTiles, TilerConfig, TilerError, build_archive, tile_primitives,
};
use proptest::prelude::*;

const Z: [f64; 3] = [0.0, 0.0, 1.0];

fn line(a: [f64; 2], b: [f64; 2]) -> Primitive {
    Line::new([a[0], a[1], 0.0], [b[0], b[1], 0.0])
        .unwrap()
        .into()
}

/// Two corner-to-corner lines. At a deep zoom each one crosses millions of
/// tiles, which is the tile explosion a hostile drawing is after.
fn explosive() -> Vec<Primitive> {
    vec![
        line([0.0, 0.0], [1.0e5, 1.0e5]),
        line([0.0, 1.0e5], [1.0e5, 0.0]),
    ]
}

// ---------------------------------------------------------------------------
// Cancellation and the deadline
// ---------------------------------------------------------------------------

#[test]
fn a_cancelled_token_stops_the_run_before_any_tile() {
    let token = CancelToken::new();
    token.cancel();
    let cfg = TilerConfig::default()
        .with_max_zoom(6)
        .with_cancel(token.clone());
    let mut out = CollectTiles::default();
    let err = tile_primitives(&explosive(), &cfg, &mut out).unwrap_err();
    assert!(matches!(err, TilerError::Cancelled), "got {err:?}");
    assert!(out.tiles.is_empty(), "nothing is written after a cancel");
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_cancelled_build_leaves_nothing_behind() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.pmtiles");
    let token = CancelToken::new();
    token.cancel();
    let cfg = TilerConfig::default()
        .with_max_zoom(6)
        .with_cancel(token.clone());
    let err = build_archive(&explosive(), &cfg, &path, "x").unwrap_err();
    assert!(matches!(err, TilerError::Cancelled), "got {err:?}");
    let left: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert!(left.is_empty(), "a cancelled run left {left:?} behind");
}

#[test]
fn a_deadline_stops_a_tile_explosion_inside_one_primitive() {
    // Two primitives only, so the stop has to come from inside the splitting
    // of one line over its tiles and not from a check between primitives.
    let limit = Duration::from_millis(300);
    let cfg = TilerConfig::default()
        .with_max_zoom(18)
        .with_deadline(limit);
    let mut out = CollectTiles::default();
    let t = Instant::now();
    let err = tile_primitives(&explosive(), &cfg, &mut out).unwrap_err();
    let took = t.elapsed();
    assert!(
        matches!(err, TilerError::DeadlineExceeded { limit: l } if l == limit),
        "got {err:?}"
    );
    assert!(
        took < Duration::from_secs(5),
        "a {limit:?} deadline took {took:?} to take effect"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_deadline_stops_a_failed_build_without_a_partial_archive() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.pmtiles");
    let cfg = TilerConfig::default()
        .with_max_zoom(18)
        .with_deadline(Duration::from_millis(200));
    let err = build_archive(&explosive(), &cfg, &path, "x").unwrap_err();
    assert!(
        matches!(err, TilerError::DeadlineExceeded { .. }),
        "{err:?}"
    );
    assert!(!path.exists());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[test]
fn cancelling_from_another_thread_stops_a_running_tiler() {
    let token = CancelToken::new();
    let cfg = TilerConfig::default()
        .with_max_zoom(18)
        .with_cancel(token.clone());
    let canceller = std::thread::spawn({
        let token = token.clone();
        move || {
            std::thread::sleep(Duration::from_millis(150));
            token.cancel();
        }
    });
    let mut out = CollectTiles::default();
    let t = Instant::now();
    let err = tile_primitives(&explosive(), &cfg, &mut out).unwrap_err();
    canceller.join().unwrap();
    assert!(matches!(err, TilerError::Cancelled), "got {err:?}");
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
}

#[test]
fn a_run_that_finishes_inside_its_deadline_is_unchanged() {
    let prims = vec![line([0.0, 0.0], [100.0, 50.0])];
    let plain = TilerConfig::default().with_max_zoom(4);
    let timed = plain
        .clone()
        .with_deadline(Duration::from_secs(60))
        .with_cancel(CancelToken::new());
    let (mut a, mut b) = (CollectTiles::default(), CollectTiles::default());
    let (ra, _) = tile_primitives(&prims, &plain, &mut a).unwrap();
    let (rb, _) = tile_primitives(&prims, &timed, &mut b).unwrap();
    assert_eq!(ra.tiles, rb.tiles);
    assert_eq!(a.tiles.len(), b.tiles.len());
}

// ---------------------------------------------------------------------------
// Huge strings
// ---------------------------------------------------------------------------

fn text_features(out: &CollectTiles) -> Vec<String> {
    let mut found = Vec::new();
    for tile in out.tiles.values() {
        for layer in tile.layers.iter().filter(|l| l.name == "text") {
            for f in &layer.features {
                assert!(matches!(f.geometry, Geometry::Points(_)));
                for (k, v) in &f.properties {
                    if k == "text"
                        && let Value::String(s) = v
                    {
                        found.push(s.clone());
                    }
                }
            }
        }
    }
    found
}

#[test]
fn one_text_feature_never_carries_more_than_the_configured_bytes() {
    // 2000 two-byte characters, a cap of 255 bytes: an odd cap, so a cut that
    // ignored character boundaries would split an "e acute" in half.
    let long = "\u{e9}".repeat(2000);
    let prims = vec![
        line([0.0, 0.0], [12_000.0, 100.0]),
        Text::new([10.0, 10.0, 0.0], 10.0, 0.0, &long)
            .unwrap()
            .into(),
        Text::new([10.0, 50.0, 0.0], 10.0, 0.0, "short")
            .unwrap()
            .into(),
    ];
    let cfg = TilerConfig::default()
        .with_max_zoom(5)
        .with_max_text_bytes(255);
    let mut out = CollectTiles::default();
    let (report, _) = tile_primitives(&prims, &cfg, &mut out).unwrap();
    let texts = text_features(&out);
    assert!(!texts.is_empty(), "the labels were drawn at some zoom");
    for t in &texts {
        assert!(t.len() <= 255, "a label carried {} bytes", t.len());
    }
    assert!(
        texts.iter().any(|t| t == "short"),
        "a short label is intact"
    );
    assert!(report.text_truncated >= 1, "the cut is counted");
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code() == DiagnosticCode::TEXT_TRUNCATED),
        "the cut is a diagnostic, not a silence"
    );
}

#[test]
fn the_default_cap_leaves_a_long_general_note_whole() {
    // The longest label in the fixtures is 8,192 characters.
    let note = "N".repeat(8192);
    let prims = vec![
        line([0.0, 0.0], [20_000.0, 100.0]),
        Text::new([10.0, 10.0, 0.0], 20.0, 0.0, &note)
            .unwrap()
            .into(),
    ];
    let mut out = CollectTiles::default();
    let (report, _) =
        tile_primitives(&prims, &TilerConfig::default().with_max_zoom(8), &mut out).unwrap();
    assert_eq!(report.text_truncated, 0);
    assert!(text_features(&out).iter().all(|t| t.len() == 8192));
}

#[test]
fn a_label_inside_the_cap_is_not_reported() {
    let prims = vec![
        line([0.0, 0.0], [1000.0, 100.0]),
        Text::new([10.0, 10.0, 0.0], 10.0, 0.0, "GROUND FLOOR")
            .unwrap()
            .into(),
    ];
    let mut out = CollectTiles::default();
    let (report, _) =
        tile_primitives(&prims, &TilerConfig::default().with_max_zoom(5), &mut out).unwrap();
    assert_eq!(report.text_truncated, 0);
}

// ---------------------------------------------------------------------------
// Extreme but constructible values: never a panic, never a hang
// ---------------------------------------------------------------------------

fn run_guarded(name: &str, prims: &[Primitive], cfg: &TilerConfig) {
    let t = Instant::now();
    let cfg = cfg.clone().with_deadline(Duration::from_secs(20));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut out = CollectTiles::default();
        let _ = tile_primitives(prims, &cfg, &mut out);
    }));
    assert!(caught.is_ok(), "{name} panicked");
    assert!(
        t.elapsed() < Duration::from_secs(40),
        "{name} took {:?}",
        t.elapsed()
    );
}

fn extreme_batches() -> Vec<(&'static str, Vec<Primitive>)> {
    let big = 1.0e300;
    let tiny = 1.0e-300;
    let mut v: Vec<(&'static str, Vec<Primitive>)> = vec![
        (
            "huge coordinates",
            vec![line([-big, -big], [big, big]), line([0.0, 0.0], [1.0, 1.0])],
        ),
        (
            "subnormal extent",
            vec![
                line([0.0, 0.0], [tiny, tiny]),
                line([0.0, 0.0], [0.0, tiny]),
            ],
        ),
        (
            "mixed scales",
            vec![
                line([0.0, 0.0], [1.0e12, 1.0]),
                line([0.0, 0.0], [1.0e-9, 1.0e-9]),
            ],
        ),
        (
            "coincident points",
            vec![line([5.0, 5.0], [5.0, 5.0]), line([5.0, 5.0], [5.0, 5.0])],
        ),
        (
            "colossal circle",
            vec![Circle::new([0.0, 0.0, 0.0], big, Z).unwrap().into()],
        ),
        (
            "microscopic circle beside a large one",
            vec![
                Circle::new([0.0, 0.0, 0.0], tiny, Z).unwrap().into(),
                Circle::new([1.0e6, 0.0, 0.0], 1.0e6, Z).unwrap().into(),
            ],
        ),
        (
            "arc over a full turn and back",
            vec![
                Arc::new([0.0, 0.0, 0.0], 10.0, 0.0, 100.0, Z)
                    .unwrap()
                    .into(),
                Arc::new([0.0, 0.0, 0.0], 10.0, -50.0, 1.0e6, Z)
                    .unwrap()
                    .into(),
            ],
        ),
        (
            "flat ellipse",
            vec![
                Ellipse::new([0.0, 0.0, 0.0], [1.0e6, 0.0, 0.0], 1.0e-12, 0.0, 6.0, Z)
                    .unwrap()
                    .into(),
            ],
        ),
        (
            "bulge storm",
            vec![
                Polyline::new(
                    &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 5.0, 0.0]],
                    true,
                    Z,
                    &[1.0e9, -1.0e9, 1.0e-12],
                )
                .unwrap()
                .into(),
            ],
        ),
        (
            "text at the edge of the range",
            vec![
                Text::new([big, big, 0.0], 1.0e150, 0.0, "X")
                    .unwrap()
                    .into(),
                Text::new([0.0, 0.0, 0.0], tiny, 1.0e9, "Y").unwrap().into(),
            ],
        ),
        (
            "control characters and escapes",
            vec![
                line([0.0, 0.0], [100.0, 100.0]),
                Text::new(
                    [1.0, 1.0, 0.0],
                    5.0,
                    0.0,
                    "\0\u{1}\u{202e}\u{feff}A\u{1f600}",
                )
                .unwrap()
                .into(),
                Text::new([1.0, 20.0, 0.0], 5.0, 0.0, "{{{{{\\\\\\\\\\H1;\\S1^;{\\f;")
                    .unwrap()
                    .into(),
                Text::new([1.0, 40.0, 0.0], 5.0, 0.0, "\\").unwrap().into(),
            ],
        ),
    ];
    // A closed polygon of a hundred thousand identical points and one of a
    // hundred thousand points on a line: both degenerate after quantisation.
    let same: Vec<[f64; 3]> = (0..100_000).map(|_| [3.0, 3.0, 0.0]).collect();
    let collinear: Vec<[f64; 3]> = (0..100_000).map(|i| [f64::from(i), 0.0, 0.0]).collect();
    v.push((
        "identical vertices",
        vec![
            Polyline::new(&same, true, Z, &[]).unwrap().into(),
            line([0.0, 0.0], [10.0, 10.0]),
        ],
    ));
    v.push((
        "collinear vertices",
        vec![Polyline::new(&collinear, true, Z, &[]).unwrap().into()],
    ));
    // A spline whose controls span nothing and one whose span is enormous.
    if let Ok(s) = Spline::new(
        3,
        0,
        &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        &[
            [0.0, 0.0, 0.0],
            [big, big, 0.0],
            [-big, big, 0.0],
            [0.0, 0.0, 0.0],
        ],
        &[],
    ) {
        v.push(("huge spline", vec![s.into(), line([0.0, 0.0], [1.0, 1.0])]));
    }
    v
}

#[test]
fn extreme_values_neither_panic_nor_hang() {
    for cfg in [
        TilerConfig::default().with_max_zoom(6),
        TilerConfig::default().with_max_zoom(24).with_extent(16),
        TilerConfig::default().with_max_zoom(0),
    ] {
        for (name, prims) in extreme_batches() {
            run_guarded(name, &prims, &cfg);
        }
    }
}

fn finite() -> impl Strategy<Value = f64> {
    prop_oneof![
        -1.0e6..1.0e6f64,
        Just(0.0),
        Just(1.0e300),
        Just(-1.0e300),
        Just(1.0e-300),
        any::<f64>().prop_filter("finite", |v| v.is_finite()),
    ]
}

fn primitive() -> impl Strategy<Value = Primitive> {
    prop_oneof![
        (finite(), finite(), finite(), finite()).prop_filter_map("line", |(a, b, c, d)| {
            Line::new([a, b, 0.0], [c, d, 0.0])
                .ok()
                .map(Primitive::from)
        }),
        (finite(), finite(), finite()).prop_filter_map("circle", |(x, y, r)| {
            Circle::new([x, y, 0.0], r, Z).ok().map(Primitive::from)
        }),
        (finite(), finite(), finite(), finite(), finite())
            .prop_filter_map("arc", |(x, y, r, a, b)| Arc::new([x, y, 0.0], r, a, b, Z)
                .ok()
                .map(Primitive::from)),
        (finite(), finite(), finite(), ".{0,64}").prop_filter_map("text", |(x, y, h, s)| {
            Text::new([x, y, 0.0], h, 0.5, &s).ok().map(Primitive::from)
        }),
        (
            prop::collection::vec((finite(), finite(), finite()), 0..24),
            any::<bool>()
        )
            .prop_filter_map("polyline", |(pts, closed)| {
                let v: Vec<[f64; 3]> = pts.iter().map(|p| [p.0, p.1, 0.0]).collect();
                let b: Vec<f64> = pts.iter().map(|p| p.2).collect();
                Polyline::new(&v, closed, Z, &b).ok().map(Primitive::from)
            }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 96, failure_persistence: None, .. ProptestConfig::default() })]

    #[test]
    fn arbitrary_batches_never_panic(
        prims in prop::collection::vec(primitive(), 0..24),
        zoom in 0u8..=8,
        extent in prop_oneof![Just(16u32), Just(256), Just(4096)],
    ) {
        let cfg = TilerConfig {
            buffer: 8,
            ..TilerConfig::default().with_max_zoom(zoom).with_extent(extent)
        }
        .with_deadline(Duration::from_secs(10));
        let mut out = CollectTiles::default();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = tile_primitives(&prims, &cfg, &mut out);
        }));
        prop_assert!(r.is_ok(), "the tiler panicked");
    }
}
