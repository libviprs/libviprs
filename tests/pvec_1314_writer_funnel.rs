//! Issue #1314 (N6.1b): the one funnel that writes normalized geometry into a PVec artifact
//! through the facade, and reads back what PVec's reader says. Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use libviprs::CancelToken;
use libviprs::cad::ItemHandle;
use libviprs::pvec::writer::{LeafConfig, WriteConfig, WriteError, write_geometry};
use libviprs::pvec::{
    EntityId, EntityStyle, FidelityClass, InstancePath, Normalized2DGeometry, OrderingPolicy,
    ProvenanceStub, RawEntity, RawGeometry, SourceUnit, order_entities,
};
use pvec_facade::{Curve, PvecReader};

const EXACT: FidelityClass = FidelityClass::SupportedExact;

fn ids(n: usize) -> Vec<EntityId> {
    let raw = (0..n)
        .map(|i| RawEntity {
            style: EntityStyle::absent(),
            geometry: RawGeometry::Line {
                start: [i as f64, 0.0, 0.0],
                end: [i as f64 + 1.0, 0.0, 0.0],
            },
            provenance: ProvenanceStub {
                handle: Some(ItemHandle::new(i as u64 + 1)),
                class_name: "LINE".into(),
                block_path: InstancePath::root(),
                layer: "0".into(),
            },
        })
        .collect();
    order_entities("t", "1", &OrderingPolicy::new(), raw)
        .entities
        .into_iter()
        .map(|o| o.id)
        .collect()
}

/// A circle, an arc, an ellipse, a polyline and a spline, in that order.
fn mixed() -> Vec<Normalized2DGeometry> {
    vec![
        Normalized2DGeometry::circle([10.0, 10.0], [1.0, 0.0], 5.0, EXACT).unwrap(),
        Normalized2DGeometry::circular_arc(
            [30.0, 10.0],
            [1.0, 0.0],
            5.0,
            0.0,
            std::f64::consts::FRAC_PI_2,
            true,
            EXACT,
        )
        .unwrap(),
        Normalized2DGeometry::ellipse([50.0, 10.0], [1.0, 0.0], 8.0, 3.0, EXACT).unwrap(),
        Normalized2DGeometry::polyline(&[[0.0, 0.0], [4.0, 3.0], [8.0, 0.0]], EXACT).unwrap(),
        Normalized2DGeometry::b_spline(
            2,
            &[[10.0, 40.0], [20.0, 60.0], [30.0, 40.0]],
            &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            EXACT,
        )
        .unwrap(),
    ]
}

fn config() -> WriteConfig {
    WriteConfig {
        unit: SourceUnit::Millimeters,
        leaves: None,
    }
}

fn write(
    config: &WriteConfig,
    geoms: &[Normalized2DGeometry],
) -> Result<libviprs::pvec::writer::GeometryArtifact, WriteError> {
    let ids = ids(geoms.len());
    write_geometry(
        config,
        ids.into_iter().zip(geoms),
        Vec::new(),
        &CancelToken::new(),
    )
}

fn kind(c: &Curve) -> &'static str {
    match c {
        Curve::Circle(_) => "circle",
        Curve::Ellipse(_) => "ellipse",
        Curve::Polyline(_) => "polyline",
        Curve::Trimmed(_) => "trimmed",
        Curve::BSpline(_) => "bspline",
        _ => "other",
    }
}

#[test]
fn issue_1314_every_geometry_kind_reads_back_as_the_kind_that_went_in() {
    let art = write(&config(), &mixed()).unwrap();
    assert_eq!((art.written, art.skipped), (5, 0));
    let r = PvecReader::open(art.bytes).unwrap();
    assert_eq!(r.entity_count().unwrap(), 5);
    let kinds: Vec<_> = r.entities().unwrap().iter().map(|(_, c)| kind(c)).collect();
    assert_eq!(
        kinds,
        ["circle", "trimmed", "ellipse", "polyline", "bspline"]
    );
}

#[test]
fn issue_1314_the_pvec_ids_are_dense_and_source_ids_map_them_back() {
    let art = write(&config(), &mixed()).unwrap();
    let r = PvecReader::open(art.bytes).unwrap();
    let got: Vec<u32> = r
        .entities()
        .unwrap()
        .iter()
        .map(|(id, _)| id.index())
        .collect();
    assert_eq!(got, [0, 1, 2, 3, 4]);
}

#[test]
fn issue_1314_an_entity_without_geometry_is_counted_and_leaves_no_gap_in_the_ids() {
    let mut geoms = mixed();
    geoms.insert(
        2,
        Normalized2DGeometry::without_geometry(FidelityClass::Unsupported).unwrap(),
    );
    let all = ids(geoms.len());
    let art = write(&config(), &geoms).unwrap();
    assert_eq!((art.written, art.skipped), (5, 1));
    // The libviprs ids of the five written entities, skipping index 2.
    let want: Vec<_> = [0, 1, 3, 4, 5].iter().map(|i| all[*i]).collect();
    assert_eq!(art.source_ids, want);
    let r = PvecReader::open(art.bytes).unwrap();
    let got: Vec<u32> = r
        .entities()
        .unwrap()
        .iter()
        .map(|(id, _)| id.index())
        .collect();
    assert_eq!(
        got,
        [0, 1, 2, 3, 4],
        "PVec's ids are dense, as Tier 0 needs"
    );
}

#[test]
fn issue_1314_an_indexed_write_with_a_skipped_entity_still_gets_its_tier0_blocks() {
    let mut geoms: Vec<_> = (0..20)
        .map(|i| {
            let x = f64::from(i) * 10.0;
            Normalized2DGeometry::polyline(&[[x, 0.0], [x + 4.0, 3.0]], EXACT).unwrap()
        })
        .collect();
    geoms.insert(
        7,
        Normalized2DGeometry::without_geometry(FidelityClass::IgnoredByPolicy).unwrap(),
    );
    let c = WriteConfig {
        leaves: Some(LeafConfig {
            cell_capacity: 8,
            tier0_max_error: Some(0.05),
        }),
        ..config()
    };
    let art = write(&c, &geoms).unwrap();
    let r = PvecReader::open(art.bytes).unwrap();
    let dir = r.spatial_directory().unwrap().expect("a directory");
    assert!(!r.tier0_leaves(&dir).unwrap().is_empty());
}

#[test]
fn issue_1314_the_manifest_carries_the_unit_the_drawing_bounds_and_the_producer() {
    let art = write(&config(), &mixed()).unwrap();
    let drawing = *art.bounds.drawing_bounds().expect("a drawing box");
    let r = PvecReader::open(art.bytes).unwrap();
    let m = r.manifest();
    assert_eq!(m.metres_per_unit, 0.001);
    assert_eq!(m.producer_id, "libviprs");
    assert_eq!(m.producer_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(m.bounds.min, [drawing.min().x, drawing.min().y]);
    assert_eq!(m.bounds.max, [drawing.max().x, drawing.max().y]);
}

#[test]
fn issue_1314_a_unit_pvec_cannot_declare_is_refused_not_rescaled() {
    for unit in [SourceUnit::Yards, SourceUnit::Unitless, SourceUnit::Unknown] {
        let c = WriteConfig { unit, ..config() };
        assert!(
            matches!(write(&c, &mixed()), Err(WriteError::UnsupportedUnit(u)) if u == unit),
            "{unit:?}"
        );
    }
}

#[test]
fn issue_1314_nothing_to_write_is_an_error_not_an_empty_artifact() {
    assert!(matches!(write(&config(), &[]), Err(WriteError::NoGeometry)));
    let only_dropped =
        vec![Normalized2DGeometry::without_geometry(FidelityClass::IgnoredByPolicy).unwrap()];
    assert!(matches!(
        write(&config(), &only_dropped),
        Err(WriteError::NoGeometry)
    ));
}

#[test]
fn issue_1314_two_runs_write_the_same_bytes() {
    let a = write(&config(), &mixed()).unwrap().bytes;
    let b = write(&config(), &mixed()).unwrap().bytes;
    assert_eq!(a, b);
}

#[test]
fn issue_1314_a_cancelled_write_returns_cancelled_and_no_bytes() {
    let geoms = mixed();
    let token = CancelToken::new();
    token.cancel();
    let r = write_geometry(
        &config(),
        ids(5).into_iter().zip(&geoms),
        Vec::new(),
        &token,
    );
    assert!(matches!(r, Err(WriteError::Cancelled)));
}

#[test]
fn issue_1314_an_indexed_write_lays_a_directory_and_tier0_blocks() {
    let geoms: Vec<_> = (0..40)
        .map(|i| {
            let x = f64::from(i % 8) * 10.0;
            let y = f64::from(i / 8) * 10.0;
            Normalized2DGeometry::polyline(&[[x, y], [x + 4.0, y + 3.0]], EXACT).unwrap()
        })
        .collect();
    let c = WriteConfig {
        leaves: Some(LeafConfig {
            cell_capacity: 8,
            tier0_max_error: Some(0.05),
        }),
        ..config()
    };
    let indexed = write(&c, &geoms).unwrap();
    let plain = write(&config(), &geoms).unwrap();
    assert_ne!(indexed.bytes, plain.bytes);
    let r = PvecReader::open(indexed.bytes).unwrap();
    let dir = r.spatial_directory().unwrap().expect("a directory");
    assert_eq!(dir.entities(), 40);
    let leaves = r.tier0_leaves(&dir).unwrap();
    assert!(!leaves.is_empty(), "Tier 0 blocks were asked for");
    assert!(
        PvecReader::open(plain.bytes)
            .unwrap()
            .spatial_directory()
            .unwrap()
            .is_none()
    );
}

#[test]
fn issue_1314_the_indexed_artifact_keeps_the_analytic_curves() {
    // The spline is advanced, so it stays in no leaf, and what the reader gives back for it is the
    // curve that went in, not a preview.
    let c = WriteConfig {
        leaves: Some(LeafConfig {
            cell_capacity: 8,
            tier0_max_error: Some(0.05),
        }),
        ..config()
    };
    let art = write(&c, &mixed()).unwrap();
    let kinds: Vec<_> = PvecReader::open(art.bytes)
        .unwrap()
        .entities()
        .unwrap()
        .iter()
        .map(|(_, c)| kind(c))
        .collect();
    assert_eq!(
        kinds,
        ["circle", "trimmed", "ellipse", "polyline", "bspline"]
    );
}
