//! Issue #1089 (N6.2): the one table that maps normalized geometry to PVec capabilities.
//!
//! One group per acceptance criterion. The table is data, so the tests read it as data.
#![cfg(feature = "pvec-normalized")]

use libviprs::pvec::mapping::{
    AbsentCapability, ConformanceClass, NORMALIZED_TO_PVEC, NormalizedKind, kind_of,
    render_markdown, row_for,
};
use libviprs::pvec::{FidelityClass, Normalized2DGeometry};
use pvec_geometry::{Bezier, Curve, Line, Point2, Trimmed};

const DOC: &str = include_str!("../docs/pvec-mapping.md");
const SRC: &str = include_str!("../src/pvec/mapping.rs");

fn p(x: f64, y: f64) -> Point2 {
    Point2::checked(x, y).unwrap()
}

/// One normalized record per kind, built the way the extractors build them.
fn fixture(kind: NormalizedKind) -> Normalized2DGeometry {
    let exact = FidelityClass::SupportedExact;
    let equiv = FidelityClass::SupportedEquivalent;
    match kind {
        NormalizedKind::Line => {
            let line = Line::new(p(0.0, 0.0), p(0.0, 0.0).vector_to(p(3.0, 4.0))).unwrap();
            let t = Trimmed::new(Curve::Line(line), 0.0, 5.0, true).unwrap();
            Normalized2DGeometry::from_curve(Curve::Trimmed(t), exact).unwrap()
        }
        NormalizedKind::Circle => {
            Normalized2DGeometry::circle([1.0, 2.0], [1.0, 0.0], 3.0, exact).unwrap()
        }
        NormalizedKind::CircularArc => {
            Normalized2DGeometry::circular_arc([1.0, 2.0], [1.0, 0.0], 3.0, 0.25, 1.5, true, exact)
                .unwrap()
        }
        NormalizedKind::Ellipse => {
            Normalized2DGeometry::ellipse([0.0, 0.0], [1.0, 0.0], 4.0, 2.0, exact).unwrap()
        }
        NormalizedKind::EllipticalArc => {
            let e = Normalized2DGeometry::ellipse([0.0, 0.0], [1.0, 0.0], 4.0, 2.0, exact).unwrap();
            let basis = e.geometry().unwrap().clone();
            let c = Curve::trimmed_conic(basis, 0.1, 1.2, true).unwrap();
            Normalized2DGeometry::from_curve(c, exact).unwrap()
        }
        NormalizedKind::Polyline => {
            Normalized2DGeometry::polyline(&[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], exact).unwrap()
        }
        NormalizedKind::Composite => {
            // a bulged vertex pair joined to a straight run, as a bulge polyline normalizes
            let arc = Trimmed::from_bulge(p(0.0, 0.0), p(2.0, 0.0), 0.5).unwrap();
            let run = Curve::Polyline(
                pvec_geometry::Polyline::new(vec![p(2.0, 0.0), p(2.0, 3.0)]).unwrap(),
            );
            let c = pvec_geometry::Composite::new(vec![Curve::Trimmed(arc), run]).unwrap();
            Normalized2DGeometry::from_curve(Curve::Composite(c), equiv).unwrap()
        }
        NormalizedKind::Bezier => {
            let b = Bezier::new(vec![p(0.0, 0.0), p(1.0, 2.0), p(2.0, 2.0), p(3.0, 0.0)]).unwrap();
            Normalized2DGeometry::from_curve(Curve::Bezier(b), exact).unwrap()
        }
        NormalizedKind::BSpline => Normalized2DGeometry::b_spline(
            2,
            &[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]],
            &[0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            exact,
        )
        .unwrap(),
        NormalizedKind::RationalBSpline => Normalized2DGeometry::rational_b_spline(
            2,
            &[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]],
            &[0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            &[1.0, 2.0, 1.0, 1.0],
            exact,
        )
        .unwrap(),
    }
}

// -- one row per variant, in a fixed order ------------------------------------------------------

#[test]
fn the_table_has_exactly_one_row_per_kind_in_declared_order() {
    assert_eq!(NORMALIZED_TO_PVEC.len(), NormalizedKind::ALL.len());
    for (i, kind) in NormalizedKind::ALL.iter().enumerate() {
        assert_eq!(NORMALIZED_TO_PVEC[i].kind, *kind, "row {i}");
        assert_eq!(row_for(*kind).kind, *kind);
    }
}

#[test]
fn the_row_lookup_is_an_exhaustive_match_so_a_new_variant_cannot_lack_a_row() {
    // The compile-time half cannot be a runtime assertion, so this pins the shape: the lookup
    // is a `match` over `NormalizedKind` with no wildcard arm.
    let body = SRC
        .split("pub const fn row_index")
        .nth(1)
        .expect("row_index exists");
    let body = body.split("\n}\n").next().unwrap();
    assert!(body.contains("match kind"), "{body}");
    assert!(
        !body.contains("_ =>"),
        "a wildcard arm defeats the check: {body}"
    );
}

// -- the doc is rendered from the table ---------------------------------------------------------

#[test]
fn the_doc_is_the_rendered_table_byte_for_byte() {
    if std::env::var_os("PVEC_MAPPING_BLESS").is_some() {
        std::fs::write(
            concat!(env!("CARGO_MANIFEST_DIR"), "/docs/pvec-mapping.md"),
            render_markdown(),
        )
        .unwrap();
        return;
    }
    assert_eq!(
        DOC,
        render_markdown(),
        "docs/pvec-mapping.md is generated: regenerate it with \
         `PVEC_MAPPING_BLESS=1 cargo test --features pvec-normalized --test pvec_1089_mapping`"
    );
}

#[test]
fn a_hand_edit_of_the_doc_is_caught() {
    let edited = DOC.replacen("circle", "cirkle", 1);
    assert_ne!(edited, render_markdown());
}

#[test]
fn the_rendering_is_deterministic() {
    assert_eq!(render_markdown(), render_markdown());
}

// -- tessellation is never an answer ------------------------------------------------------------

#[test]
fn no_curve_row_maps_to_a_polyline_or_a_point_set() {
    for row in NORMALIZED_TO_PVEC.iter() {
        if row.kind == NormalizedKind::Polyline {
            continue;
        }
        let t = row.pvec_type.to_ascii_lowercase();
        assert!(
            !t.contains("polyline") && !t.contains("point"),
            "{:?} maps to {}",
            row.kind,
            row.pvec_type
        );
    }
}

#[test]
fn only_the_polyline_row_names_the_polyline_capability() {
    for row in NORMALIZED_TO_PVEC.iter() {
        let is_polyline_cap = row.capability.name == "polyline";
        assert_eq!(
            is_polyline_cap,
            row.kind == NormalizedKind::Polyline,
            "{:?}",
            row.kind
        );
    }
}

#[test]
fn the_absent_cell_is_one_of_three_and_none_of_them_is_tessellation() {
    for row in NORMALIZED_TO_PVEC.iter() {
        let f = row.absent.fidelity();
        assert!(
            matches!(
                f,
                FidelityClass::RequiresPVecCapability
                    | FidelityClass::RequiresExtension
                    | FidelityClass::Unsupported
            ),
            "{:?}: {f:?}",
            row.kind
        );
        assert!(
            matches!(
                row.absent,
                AbsentCapability::RequiresPVecCapability
                    | AbsentCapability::RequiresExtension
                    | AbsentCapability::Unsupported
            ),
            "{:?}",
            row.kind
        );
    }
    // A fourth value is unrepresentable: the type has exactly these three.
    assert_eq!(AbsentCapability::ALL.len(), 3);
}

#[test]
fn the_source_never_computes_a_curve() {
    for word in [
        "tessellat",
        "flatten(",
        "evaluate(",
        "insert_knot",
        "subdivide",
    ] {
        let code: String = SRC
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!code.contains(word), "`{word}` in mapping.rs");
    }
}

// -- a fixture per row --------------------------------------------------------------------------

#[test]
fn every_row_has_a_fixture_that_lands_on_its_kind() {
    for kind in NormalizedKind::ALL {
        let g = fixture(kind);
        assert_eq!(kind_of(&g), Some(kind), "{kind:?}");
    }
}

#[test]
fn every_fixtures_fidelity_is_the_one_its_conformance_class_promises() {
    for kind in NormalizedKind::ALL {
        let row = row_for(kind);
        let g = fixture(kind);
        assert_eq!(g.fidelity(), row.fidelity_when_present(), "{kind:?}");
    }
}

#[test]
fn the_pvec_type_names_the_curve_the_fixture_actually_holds() {
    for kind in NormalizedKind::ALL {
        let row = row_for(kind);
        let curve = fixture(kind).geometry().unwrap().clone();
        let held = match &curve {
            Curve::Circle(_) => "circle".to_string(),
            Curve::Ellipse(_) => "ellipse".to_string(),
            Curve::Polyline(_) => "polyline".to_string(),
            Curve::Composite(_) => "composite_curve".to_string(),
            Curve::Bezier(_) => "bezier_curve".to_string(),
            Curve::BSpline(_) => "b_spline_curve".to_string(),
            Curve::RationalBSpline(_) => "rational_b_spline_curve".to_string(),
            Curve::Trimmed(t) => match t.basis() {
                Curve::Line(_) => "trimmed_curve(line)".to_string(),
                Curve::Circle(_) => "trimmed_curve(circle)".to_string(),
                Curve::Ellipse(_) => "trimmed_curve(ellipse)".to_string(),
                other => panic!("unexpected basis {other:?}"),
            },
            other => panic!("unexpected curve {other:?}"),
        };
        assert_eq!(row.pvec_type, held, "{kind:?}");
    }
}

#[test]
fn a_record_with_no_geometry_has_no_kind() {
    let g = Normalized2DGeometry::without_geometry(FidelityClass::Unsupported).unwrap();
    assert_eq!(kind_of(&g), None);
}

#[test]
fn a_curve_with_no_row_is_reported_not_guessed() {
    let t = Trimmed::new(
        Curve::Bezier(Bezier::new(vec![p(0.0, 0.0), p(1.0, 1.0), p(2.0, 0.0)]).unwrap()),
        0.0,
        0.5,
        true,
    );
    // PVec only trims lines, circles and ellipses here, so a Bezier basis may be refused. Either
    // way no row is invented for it.
    if let Ok(t) = t {
        let g = Normalized2DGeometry::from_curve(Curve::Trimmed(t), FidelityClass::SupportedExact)
            .unwrap();
        assert_eq!(kind_of(&g), None);
    }
}

#[test]
fn the_structural_rows_claim_structural_exactness_and_the_bulge_composite_does_not() {
    for row in NORMALIZED_TO_PVEC.iter() {
        let expect = if row.kind == NormalizedKind::Composite {
            ConformanceClass::GeometricExactness
        } else {
            ConformanceClass::StructuralExactness
        };
        assert_eq!(row.conformance, expect, "{:?}", row.kind);
    }
    // Display fidelity means a densified output. No normalized row is one.
    assert!(
        NORMALIZED_TO_PVEC
            .iter()
            .all(|r| r.conformance != ConformanceClass::DisplayFidelity)
    );
}

#[test]
fn capability_ids_are_unique_per_name_and_non_zero() {
    for a in NORMALIZED_TO_PVEC.iter() {
        assert!(a.capability.id != 0);
        for b in NORMALIZED_TO_PVEC.iter() {
            if a.capability.id == b.capability.id {
                assert_eq!(a.capability.name, b.capability.name);
            }
        }
    }
}
