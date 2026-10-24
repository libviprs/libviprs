//! Issue #1095 (N3.1): the `Normalized2DGeometry` adapter.
//!
//! One test per acceptance criterion, plus the behaviour the ADR (`NG-02` to `NG-08`) fixes.
//! The record holds PVec values and does no mathematics of its own, so most of what is checked
//! here is that PVec's answer comes back untouched.
#![cfg(feature = "pvec-normalized")]

use libviprs::pvec::{FidelityClass, NormalizeError, Normalized2DGeometry, ProvenanceHandle};
use pvec_geometry::{Curve, Curve2d};

const SRC: &str = include_str!("../src/pvec/normalized.rs");

fn code_only(src: &str) -> String {
    // Doc and line comments say what is banned, so the scans below read code, not prose.
    src.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

const GEO: [FidelityClass; 3] = [
    FidelityClass::SupportedExact,
    FidelityClass::SupportedEquivalent,
    FidelityClass::RequiresExtension,
];
const NO_GEO: [FidelityClass; 3] = [
    FidelityClass::RequiresPVecCapability,
    FidelityClass::Unsupported,
    FidelityClass::IgnoredByPolicy,
];

#[test]
fn non_finite_coordinate_is_rejected_at_construction_with_a_typed_error() {
    let f = FidelityClass::SupportedExact;
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let errs: Vec<NormalizeError> = vec![
            Normalized2DGeometry::circle([bad, 0.0], [1.0, 0.0], 1.0, f).unwrap_err(),
            Normalized2DGeometry::circle([0.0, bad], [1.0, 0.0], 1.0, f).unwrap_err(),
            Normalized2DGeometry::circle([0.0, 0.0], [1.0, 0.0], bad, f).unwrap_err(),
            Normalized2DGeometry::circle([0.0, 0.0], [bad, 0.0], 1.0, f).unwrap_err(),
            Normalized2DGeometry::ellipse([0.0, 0.0], [1.0, 0.0], bad, 1.0, f).unwrap_err(),
            Normalized2DGeometry::circular_arc([0.0, 0.0], [1.0, 0.0], 1.0, bad, 1.0, true, f)
                .unwrap_err(),
            Normalized2DGeometry::polyline(&[[0.0, 0.0], [bad, 1.0]], f).unwrap_err(),
            Normalized2DGeometry::b_spline(1, &[[0.0, 0.0], [1.0, 1.0]], &[0.0, 0.0, bad, 1.0], f)
                .unwrap_err(),
            Normalized2DGeometry::rational_b_spline(
                1,
                &[[0.0, 0.0], [1.0, 1.0]],
                &[0.0, 0.0, 1.0, 1.0],
                &[1.0, bad],
                f,
            )
            .unwrap_err(),
        ];
        for e in errs {
            assert!(
                e.is_non_finite(),
                "{bad}: expected a non-finite refusal, got {e:?}"
            );
        }
    }
}

#[test]
fn the_refused_field_is_named() {
    let e = Normalized2DGeometry::polyline(
        &[[0.0, 0.0], [f64::NAN, 1.0]],
        FidelityClass::SupportedExact,
    )
    .unwrap_err();
    assert!(e.is_non_finite());
    assert_eq!(e.rule(), Some("NP-01"));
}

#[test]
fn pvec_refusals_pass_through_with_their_rule() {
    // A zero radius is PVec's NP-07, not a check this crate makes.
    let e =
        Normalized2DGeometry::circle([0.0, 0.0], [1.0, 0.0], 0.0, FidelityClass::SupportedExact)
            .unwrap_err();
    assert!(!e.is_non_finite());
    assert_eq!(e.rule(), Some("NP-07"));
    // A zero reference direction is NP-10.
    let e =
        Normalized2DGeometry::circle([0.0, 0.0], [0.0, 0.0], 1.0, FidelityClass::SupportedExact)
            .unwrap_err();
    assert_eq!(e.rule(), Some("NP-10"));
}

#[test]
fn a_circle_is_a_pvec_circle_with_pvecs_own_bounds() {
    let g =
        Normalized2DGeometry::circle([2.0, 3.0], [1.0, 0.0], 5.0, FidelityClass::SupportedExact)
            .unwrap();
    let curve = g.geometry().expect("a circle carries geometry");
    assert!(matches!(curve, Curve::Circle(_)));
    // The box is whatever PVec says for that curve, carried and not recomputed.
    assert_eq!(g.source_bounds().copied(), Some(curve.bounds().unwrap()));
    assert_eq!(g.fidelity(), FidelityClass::SupportedExact);
    assert!(g.provenance().is_none());
}

#[test]
fn an_arc_is_a_trimmed_circle() {
    let g = Normalized2DGeometry::circular_arc(
        [0.0, 0.0],
        [1.0, 0.0],
        2.0,
        0.0,
        1.0,
        true,
        FidelityClass::SupportedExact,
    )
    .unwrap();
    match g.geometry().unwrap() {
        Curve::Trimmed(t) => assert!(matches!(t.basis(), Curve::Circle(_))),
        other => panic!("expected a trimmed circle, got {other:?}"),
    }
}

#[test]
fn an_ellipse_is_a_pvec_ellipse() {
    let g = Normalized2DGeometry::ellipse(
        [0.0, 0.0],
        [1.0, 0.0],
        4.0,
        2.0,
        FidelityClass::SupportedExact,
    )
    .unwrap();
    assert!(matches!(g.geometry().unwrap(), Curve::Ellipse(_)));
}

#[test]
fn a_polyline_keeps_its_vertices_and_a_closed_one_keeps_the_repeated_vertex() {
    let pts = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 0.0]];
    let g = Normalized2DGeometry::polyline(&pts, FidelityClass::SupportedExact).unwrap();
    match g.geometry().unwrap() {
        Curve::Polyline(p) => {
            let v: Vec<[f64; 2]> = p.vertices().iter().map(|q| [q.x, q.y]).collect();
            assert_eq!(v, pts.to_vec());
        }
        other => panic!("expected a polyline, got {other:?}"),
    }
}

#[test]
fn a_spline_passes_its_knots_and_control_points_through_whole() {
    let pts = [[0.0, 0.0], [1.0, 2.0], [3.0, 2.0], [4.0, 0.0]];
    let knots = [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
    // Not asserted as a literal: a knot vector this short is only legal when PVec says so.
    let g = Normalized2DGeometry::b_spline(2, &pts, &knots, FidelityClass::SupportedExact).unwrap();
    match g.geometry().unwrap() {
        Curve::BSpline(s) => {
            assert_eq!(s.degree(), 2);
            assert_eq!(s.knots(), &knots);
            assert_eq!(s.control_points().len(), pts.len());
        }
        other => panic!("expected a b-spline, got {other:?}"),
    }
}

#[test]
fn weights_present_means_rational_and_is_never_demoted() {
    let pts = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]];
    let knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let w = [1.0, 0.5, 1.0];
    let g =
        Normalized2DGeometry::rational_b_spline(2, &pts, &knots, &w, FidelityClass::SupportedExact)
            .unwrap();
    match g.geometry().unwrap() {
        Curve::RationalBSpline(s) => assert_eq!(s.weights(), &w),
        other => panic!("expected a rational b-spline, got {other:?}"),
    }
}

#[test]
fn geometry_presence_follows_the_fidelity_class() {
    let curve_of = || {
        Normalized2DGeometry::circle([0.0, 0.0], [1.0, 0.0], 1.0, FidelityClass::SupportedExact)
            .unwrap()
            .geometry()
            .unwrap()
            .clone()
    };
    for f in GEO {
        assert!(f.produces_geometry());
        let g = Normalized2DGeometry::from_curve(curve_of(), f).unwrap();
        assert!(g.geometry().is_some() && g.source_bounds().is_some());
        assert!(matches!(
            Normalized2DGeometry::without_geometry(f),
            Err(NormalizeError::FidelityGeometryMismatch { .. })
        ));
    }
    for f in NO_GEO {
        assert!(!f.produces_geometry());
        let g = Normalized2DGeometry::without_geometry(f).unwrap();
        assert!(g.geometry().is_none() && g.source_bounds().is_none());
        assert_eq!(g.fidelity(), f);
        assert!(matches!(
            Normalized2DGeometry::from_curve(curve_of(), f),
            Err(NormalizeError::FidelityGeometryMismatch { .. })
        ));
        // The constructors that build a curve refuse a class that produces none.
        assert!(matches!(
            Normalized2DGeometry::circle([0.0, 0.0], [1.0, 0.0], 1.0, f),
            Err(NormalizeError::FidelityGeometryMismatch { .. })
        ));
    }
}

#[test]
fn provenance_is_optional_and_does_not_touch_the_geometry() {
    let bare =
        Normalized2DGeometry::circle([0.0, 0.0], [1.0, 0.0], 1.0, FidelityClass::SupportedExact)
            .unwrap();
    let tagged = bare.clone().with_provenance(ProvenanceHandle::new(42));
    assert_eq!(tagged.provenance().map(ProvenanceHandle::get), Some(42));
    assert!(bare.provenance().is_none());
    assert_eq!(bare.geometry(), tagged.geometry());
    assert_eq!(bare.source_bounds(), tagged.source_bounds());
    assert_eq!(bare.fidelity(), tagged.fidelity());
}

#[test]
fn the_record_has_no_equality_of_its_own() {
    // NG-B6: a derived PartialEq over f64 fields would be bit comparison dressed as geometry.
    let code = code_only(SRC);
    for needle in [
        "PartialEq for Normalized2DGeometry",
        "Hash for Normalized2DGeometry",
    ] {
        assert!(!code.contains(needle), "{needle}");
    }
    let before = code
        .split("pub struct Normalized2DGeometry")
        .next()
        .unwrap();
    let derive = before
        .lines()
        .rev()
        .find(|l| l.contains("derive("))
        .unwrap_or("");
    assert!(
        !derive.contains("PartialEq") && !derive.contains("Hash"),
        "{derive}"
    );
}

#[test]
fn the_module_declares_no_tolerance_and_no_comparison_threshold() {
    let code = code_only(SRC).to_lowercase();
    for needle in [
        "epsilon",
        "tolerance",
        "tol_",
        "_tol",
        "abs_diff",
        "approx",
        "1e-",
        "1.0e-",
        "ulp",
    ] {
        assert!(
            !code.contains(needle),
            "normalized.rs must not contain `{needle}`"
        );
    }
    // No named numeric constant of any kind (`pub const fn` is a function, not a constant).
    for line in code.lines().map(str::trim_start) {
        let line = line.strip_prefix("pub ").unwrap_or(line);
        assert!(
            !(line.starts_with("const ") && !line.starts_with("const fn"))
                && !line.starts_with("static "),
            "normalized.rs must not declare a constant: {line}"
        );
    }
}

#[test]
fn the_module_contains_no_evaluator_tessellator_or_knot_manipulation() {
    let code = code_only(SRC);
    for needle in [
        ".sin(",
        ".cos(",
        ".tan(",
        ".atan",
        ".sqrt(",
        ".powi(",
        ".powf(",
        ".hypot(",
        ".eval(",
        ".derivative(",
        "flatten",
        "tessellat",
        "insert_knot",
        "remove_knot",
        "elevate",
        "to_rational_b_spline",
        "TAU",
        "PI",
    ] {
        assert!(
            !code.contains(needle),
            "normalized.rs must not contain `{needle}` (NG-B1 to NG-B4)"
        );
    }
}

#[test]
fn no_provider_or_physical_format_type_appears_in_the_module() {
    for needle in [
        "cad::",
        "crate::cad",
        "Primitive",
        "ProviderError",
        "DwgProvider",
        "RawEntity",
        "RawGeometry",
        "acadrust",
        "acadsharp",
        "lopdf",
        "pvec_codec",
        "pvec_container",
        "pvec_components",
        "pvec_entity",
    ] {
        assert!(
            !code_only(SRC).contains(needle),
            "normalized.rs must not name `{needle}` (NG-08)"
        );
    }
}

#[test]
fn the_module_documentation_states_the_delegation_rule() {
    let docs: String = SRC
        .lines()
        .take_while(|l| l.starts_with("//!") || l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    for needle in [
        "NG-01",
        "delegat",
        "pvec_geometry",
        "reviewable violation",
        "NG-B1",
    ] {
        assert!(docs.contains(needle), "module docs must mention `{needle}`");
    }
}

#[test]
fn the_crate_root_re_exports_the_public_types_only() {
    let _: Option<libviprs::Normalized2DGeometry> = None;
    let _: Option<libviprs::NormalizeError> = None;
    let _: Option<libviprs::FidelityClass> = None;
    let _: Option<libviprs::ProvenanceHandle> = None;
}
