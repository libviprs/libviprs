//! Issue #1093 (N6.7): the conversion report. Every entity is in it exactly once, the identity
//! between the rows and the entities encountered is enforced, the bytes are deterministic, and the
//! schema cannot change without the version changing.

use std::sync::Mutex;

use libviprs::cad::ItemHandle;
use libviprs::pvec::fidelity::Classification;
use libviprs::pvec::report::{
    BoundsSplit, ConversionReport, EntityRow, ReportBuilder, ReportError, SCHEMA_VERSION,
    UnresolvedXrefRow, ViewSummary, XrefSummary,
};
use libviprs::pvec::{
    Basis, EntityId, EntityStyle, FidelityClass, InstancePath, OrderingPolicy, ProvenanceStub,
    RawEntity, RawGeometry, ReasonCode, order_entities,
};

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

/// One entity in each of the six classes, then more of the common ones.
fn rows(n: usize) -> Vec<EntityRow> {
    ids(n)
        .into_iter()
        .enumerate()
        .map(|(i, id)| {
            let (c, semantic) = match i % 6 {
                0 => (
                    Classification::supported(Basis::Structural, "LINE"),
                    Some("polyline"),
                ),
                1 => (
                    Classification::supported(Basis::Equivalent, "ARC"),
                    Some("trimmed_curve"),
                ),
                2 => (
                    Classification::refused(ReasonCode::Spline3d, "SPLINE"),
                    None,
                ),
                3 => (
                    Classification::refused(ReasonCode::UnmappedSourceClass, "HATCH"),
                    None,
                ),
                4 => (
                    Classification::ignored_by_policy("layers.off", "LINE"),
                    None,
                ),
                _ => (
                    Classification::refused(ReasonCode::SplineKnotsNotReported, "SPLINE"),
                    None,
                ),
            };
            EntityRow::new(id, &c, Some(i as u64 + 1), semantic, vec![])
        })
        .collect()
}

fn build(rows: Vec<EntityRow>, encountered: u64) -> Result<ConversionReport, ReportError> {
    let mut b = ReportBuilder::new();
    for r in rows {
        b.push(r);
    }
    b.finish(
        encountered,
        ViewSummary::default(),
        XrefSummary::default(),
        BoundsSplit::default(),
    )
}

#[test]
fn issue_1093_the_class_counts_add_up_to_the_entities_encountered() {
    for n in [1usize, 6, 7, 60, 601] {
        let r = build(rows(n), n as u64).unwrap();
        let sum: u64 = r.totals.by_fidelity.values().sum();
        assert_eq!(
            (sum, r.totals.encountered, r.entities.len() as u64),
            (n as u64, n as u64, n as u64)
        );
        assert_eq!(
            r.totals.by_fidelity.len(),
            FidelityClass::ALL.len(),
            "all six classes appear, zeros too"
        );
    }
}

#[test]
fn issue_1093_a_dropped_entity_turns_the_build_red() {
    // The mutation check: the conversion met 12 entities and 11 rows reach the builder.
    let mut r = rows(12);
    r.remove(7);
    assert!(matches!(
        build(r, 12),
        Err(ReportError::CountMismatch {
            rows: 11,
            encountered: 12
        })
    ));
    // And one counted twice.
    let mut r = rows(12);
    r.push(r[3].clone());
    assert!(matches!(
        build(r, 12),
        Err(ReportError::CountMismatch { .. })
    ));
}

#[test]
fn issue_1093_two_rows_for_one_entity_are_refused() {
    let mut r = rows(6);
    r[5].id = r[4].id;
    assert!(matches!(build(r, 6), Err(ReportError::DuplicateId(_))));
}

#[test]
fn issue_1093_each_refused_or_ignored_entity_names_exactly_one_reason() {
    let r = build(rows(60), 60).unwrap();
    for row in &r.entities {
        match row.fidelity.as_str() {
            "IgnoredByPolicy" => {
                assert!(
                    row.policy_field.is_some() && row.reason.is_none(),
                    "{row:?}"
                );
            }
            "Unsupported" | "RequiresExtension" | "RequiresPVecCapability" => {
                assert!(
                    row.reason.is_some() && row.policy_field.is_none(),
                    "{row:?}"
                );
            }
            _ => assert!(
                row.reason.is_none() && row.policy_field.is_none(),
                "{row:?}"
            ),
        }
    }
    // A row that names none, or two, cannot be built.
    let mut bad = rows(6);
    bad[2].reason = None;
    assert!(matches!(build(bad, 6), Err(ReportError::Inconsistent(_))));
    let mut bad = rows(6);
    bad[4].reason = Some("spline-3d".into());
    bad[4].policy_field = None;
    assert!(matches!(build(bad, 6), Err(ReportError::Inconsistent(_))));
}

#[test]
fn issue_1093_the_aggregates_match_the_rows() {
    let r = build(rows(60), 60).unwrap();
    assert_eq!(r.totals.by_fidelity["SupportedExact"], 10);
    assert_eq!(r.totals.by_fidelity["IgnoredByPolicy"], 10);
    assert_eq!(r.totals.by_source_class["SPLINE"], 20);
    assert_eq!(r.totals.by_source_class["HATCH"], 10);
    assert_eq!(r.totals.by_semantic_type["polyline"], 10);
    assert_eq!(r.totals.by_semantic_type.values().sum::<u64>(), 20);
}

#[test]
fn issue_1093_the_bytes_do_not_depend_on_the_order_rows_arrive_in() {
    let forward = build(rows(60), 60).unwrap().to_json_string().unwrap();
    let mut rev = rows(60);
    rev.reverse();
    assert_eq!(forward, build(rev, 60).unwrap().to_json_string().unwrap());
    let again = build(rows(60), 60).unwrap().to_json_string().unwrap();
    assert_eq!(forward, again);
}

#[test]
fn issue_1093_one_worker_and_eight_write_the_same_bytes() {
    let all = rows(800);
    let one = build(all.clone(), 800).unwrap().to_json_string().unwrap();

    let builder = Mutex::new(ReportBuilder::new());
    std::thread::scope(|s| {
        for chunk in all.chunks(100) {
            let builder = &builder;
            s.spawn(move || {
                for row in chunk {
                    builder.lock().unwrap().push(row.clone());
                }
            });
        }
    });
    let eight = builder
        .into_inner()
        .unwrap()
        .finish(
            800,
            ViewSummary::default(),
            XrefSummary::default(),
            BoundsSplit::default(),
        )
        .unwrap()
        .to_json_string()
        .unwrap();
    assert_eq!(one, eight);
}

#[test]
fn issue_1093_an_unresolved_xref_appears_with_its_name_and_path() {
    let xrefs = XrefSummary {
        unresolved: vec![UnresolvedXrefRow {
            name: "site".into(),
            path: "..\\site.dwg".into(),
            resolved_path: None,
            path_refusal: Some("EscapesRoot".into()),
        }],
        resolved: vec!["floor1".into()],
    };
    let mut b = ReportBuilder::new();
    for r in rows(6) {
        b.push(r);
    }
    let json = b
        .finish(
            6,
            ViewSummary::default(),
            xrefs,
            BoundsSplit {
                exact: 4,
                hull_upper_bound: 1,
            },
        )
        .unwrap()
        .to_json_string()
        .unwrap();
    let back = ConversionReport::from_json_slice(json.as_bytes()).unwrap();
    assert_eq!(back.xrefs.unresolved[0].name, "site");
    assert_eq!(back.xrefs.unresolved[0].path, "..\\site.dwg");
    assert_eq!(back.xrefs.resolved, ["floor1"]);
    assert_eq!((back.bounds.exact, back.bounds.hull_upper_bound), (4, 1));
}

#[test]
fn issue_1093_a_report_reads_back_equal() {
    let r = build(rows(60), 60).unwrap();
    let back = ConversionReport::from_json_slice(r.to_json_string().unwrap().as_bytes()).unwrap();
    assert_eq!(back, r);
}

#[test]
fn issue_1093_a_report_of_another_version_is_refused() {
    let json = build(rows(6), 6).unwrap().to_json_string().unwrap();
    let other = json.replace(
        &format!("\"schema_version\": {SCHEMA_VERSION}"),
        &format!("\"schema_version\": {}", SCHEMA_VERSION + 1),
    );
    assert_ne!(json, other);
    assert!(matches!(
        ConversionReport::from_json_slice(other.as_bytes()),
        Err(ReportError::Schema(_))
    ));
}

/// Every JSON key path of a report, `a.b` for nested keys and `a[]` for array elements. Adding,
/// renaming or removing a field changes this list, and this list is pinned to the version.
fn key_paths(v: &serde_json::Value, prefix: &str, out: &mut Vec<String>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, v) in m {
                // Maps keyed by data (class names, layers) are values, not schema.
                if matches!(
                    prefix,
                    "totals.by_fidelity"
                        | "totals.by_source_class"
                        | "totals.by_semantic_type"
                        | "view.excluded_by_field"
                        | "view.excluded_by_layer"
                ) {
                    continue;
                }
                let p = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                out.push(p.clone());
                key_paths(v, &p, out);
            }
        }
        serde_json::Value::Array(a) => {
            if let Some(first) = a.first() {
                key_paths(first, &format!("{prefix}[]"), out);
            }
        }
        _ => {}
    }
}

#[test]
fn issue_1093_the_schema_cannot_change_without_the_version_changing() {
    let mut b = ReportBuilder::new();
    for r in rows(6) {
        b.push(r);
    }
    let xrefs = XrefSummary {
        unresolved: vec![UnresolvedXrefRow {
            name: "n".into(),
            path: "p".into(),
            resolved_path: None,
            path_refusal: None,
        }],
        resolved: vec!["r".into()],
    };
    let r = b
        .finish(6, ViewSummary::default(), xrefs, BoundsSplit::default())
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&r.to_json_string().unwrap()).unwrap();
    let mut keys = Vec::new();
    key_paths(&v, "", &mut keys);
    keys.sort();
    // If this list changes, bump SCHEMA_VERSION (or make the field ignorable and say so in
    // docs/pvec-conversion-report.md), and update the pinned pair below.
    let pinned: (u32, &[&str]) = (
        1,
        &[
            "bounds",
            "bounds.exact",
            "bounds.hull_upper_bound",
            "entities",
            "entities[].fidelity",
            "entities[].id",
            "entities[].policy_field",
            "entities[].reason",
            "entities[].semantic_type",
            "entities[].source_class",
            "entities[].source_handle",
            "entities[].warnings",
            "schema_version",
            "totals",
            "totals.by_fidelity",
            "totals.by_semantic_type",
            "totals.by_source_class",
            "totals.encountered",
            "view",
            "view.excluded_by_field",
            "view.excluded_by_layer",
            "view.included",
            "view.space",
            "xrefs",
            "xrefs.resolved",
            "xrefs.unresolved",
            "xrefs.unresolved[].name",
            "xrefs.unresolved[].path",
            "xrefs.unresolved[].path_refusal",
            "xrefs.unresolved[].resolved_path",
        ],
    );
    assert_eq!(
        SCHEMA_VERSION, pinned.0,
        "the schema changed: the version and this list move together"
    );
    assert_eq!(keys, pinned.1);
}

#[test]
fn issue_1093_the_doc_names_every_field() {
    let doc = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/docs/pvec-conversion-report.md"
    ))
    .expect("docs/pvec-conversion-report.md");
    let r = build(rows(6), 6).unwrap();
    let v: serde_json::Value = serde_json::from_str(&r.to_json_string().unwrap()).unwrap();
    let mut keys = Vec::new();
    key_paths(&v, "", &mut keys);
    for k in keys {
        let leaf = k.rsplit('.').next().unwrap().trim_end_matches("[]");
        assert!(
            doc.contains(&format!("`{leaf}`")),
            "the doc does not describe `{leaf}`"
        );
    }
    assert!(doc.contains("schema_version"));
}
