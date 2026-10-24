//! Issue #1081 (N5.2): native cubic Bezier mathematics through normalization.
//!
//! The fixtures are content streams built in memory so the operators under test are visible
//! here. Every assertion about control points compares raw bit patterns, because a mapping that
//! rounds, snaps or re-derives a control point would still pass an `==` with a tolerance.
#![cfg(feature = "pvec-normalized")]

use libviprs::pvec::pdf_bezier::{PdfNormalizedPath, normalize_page, normalize_path};
use libviprs::pvec::pdf_vector::{
    PdfPathSegment as Seg, PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes,
};
use libviprs::pvec::{FidelityClass, ReasonCode};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};
use pvec_geometry::Curve;

fn one_page(body: &str) -> Vec<u8> {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let content = doc.add_object(Stream::new(Dictionary::new(), body.as_bytes().to_vec()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 200.into(), 100.into()],
        "Contents" => content,
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![Object::from(page)],
            "Count" => 1,
        }),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog);
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

fn page(body: &str) -> PdfVectorPage {
    extract_page_paths_from_bytes(&one_page(body), 1, &PdfVectorLimits::default()).unwrap()
}

fn normalized(body: &str) -> Vec<PdfNormalizedPath> {
    normalize_page(&page(body))
}

fn bits(p: [f64; 2]) -> [u64; 2] {
    [p[0].to_bits(), p[1].to_bits()]
}

fn bezier_bits(c: &Curve) -> Vec<[u64; 2]> {
    match c {
        Curve::Bezier(b) => b
            .control_points()
            .iter()
            .map(|p| [p.x.to_bits(), p.y.to_bits()])
            .collect(),
        other => panic!("expected a Bezier, got {other:?}"),
    }
}

fn polyline_ends(c: &Curve) -> Vec<[u64; 2]> {
    match c {
        Curve::Polyline(p) => p
            .vertices()
            .iter()
            .map(|p| [p.x.to_bits(), p.y.to_bits()])
            .collect(),
        other => panic!("expected a polyline, got {other:?}"),
    }
}

const S_CURVE: &str = "10.25 20.5 m 30.125 80.75 60.5 80.25 90.5 50.5 c \
                       120.75 20.25 150.5 20.125 180.25 50.75 c S";

#[test]
fn n5_2_s_curve_of_two_c_is_exactly_two_curve_entities() {
    let paths = normalized(S_CURVE);
    assert_eq!(paths.len(), 1);
    assert_eq!(paths[0].entities.len(), 2);
    for e in &paths[0].entities {
        assert!(matches!(e.geometry.geometry(), Some(Curve::Bezier(_))));
        assert_eq!(e.geometry.fidelity(), FidelityClass::SupportedExact);
    }
}

#[test]
fn n5_2_control_points_are_bit_identical_to_the_stream() {
    let p = page(S_CURVE);
    let paths = normalize_path(&p.entities[0]);
    let expect = [
        [[10.25, 20.5], [30.125, 80.75], [60.5, 80.25], [90.5, 50.5]],
        [
            [90.5, 50.5],
            [120.75, 20.25],
            [150.5, 20.125],
            [180.25, 50.75],
        ],
    ];
    assert_eq!(paths.entities.len(), 2);
    for (e, want) in paths.entities.iter().zip(expect) {
        let got = bezier_bits(e.geometry.geometry().unwrap());
        let want: Vec<_> = want.iter().map(|p| bits(*p)).collect();
        assert_eq!(got, want);
    }
}

#[test]
fn n5_2_curve_is_cubic_with_four_control_points_never_an_approximation() {
    // An N-segment approximation (any N, including 4096) is a polyline or a higher-degree curve,
    // never a degree-3 Bezier with exactly four control points.
    let paths = normalized(S_CURVE);
    assert_eq!(paths[0].entities.len(), 2, "no extra entities from sampling");
    for e in &paths[0].entities {
        match e.geometry.geometry().unwrap() {
            Curve::Bezier(b) => {
                assert_eq!(b.degree(), 3);
                assert_eq!(b.control_points().len(), 4);
            }
            other => panic!("a curve operator produced {other:?}"),
        }
    }
}

#[test]
fn n5_2_v_and_y_carry_the_expanded_points_verbatim() {
    let p = page("10 10 m 20.5 40.25 60 10 v 70 70 80.5 90.25 y S");
    let n = normalize_path(&p.entities[0]);
    assert_eq!(n.entities.len(), 2);
    let Seg::CurveTo { c1, c2, end } = p.entities[0].segments[1] else {
        panic!("v did not expand to CurveTo");
    };
    assert_eq!(
        bezier_bits(n.entities[0].geometry.geometry().unwrap()),
        vec![bits([10.0, 10.0]), bits(c1), bits(c2), bits(end)]
    );
    let Seg::CurveTo { c1, c2, end } = p.entities[0].segments[2] else {
        panic!("y did not expand to CurveTo");
    };
    assert_eq!(
        bezier_bits(n.entities[1].geometry.geometry().unwrap()),
        vec![bits([60.0, 10.0]), bits(c1), bits(c2), bits(end)]
    );
}

#[test]
fn n5_2_l_is_one_line_segment() {
    let n = &normalized("10 10 m 50.5 40.25 l S")[0];
    assert_eq!(n.entities.len(), 1);
    assert_eq!(
        polyline_ends(n.entities[0].geometry.geometry().unwrap()),
        vec![bits([10.0, 10.0]), bits([50.5, 40.25])]
    );
    assert_eq!(n.subpaths.len(), 1);
    assert!(!n.subpaths[0].closed);
}

#[test]
fn n5_2_h_closes_with_a_segment_and_marks_the_subpath_closed() {
    let n = &normalized("10 10 m 50 10 l 50 40 l h S")[0];
    assert_eq!(n.entities.len(), 3, "two l segments plus the closing one");
    assert!(n.subpaths[0].closed);
    assert_eq!(
        polyline_ends(n.entities[2].geometry.geometry().unwrap()),
        vec![bits([50.0, 40.0]), bits([10.0, 10.0])]
    );
    for e in &n.entities {
        assert!(matches!(e.geometry.geometry(), Some(Curve::Polyline(_))));
    }
}

#[test]
fn n5_2_h_after_a_curve_ending_at_the_start_adds_no_degenerate_segment() {
    let n = &normalized("10 10 m 20 50 60 50 10 10 c h f")[0];
    assert_eq!(n.entities.len(), 1);
    assert!(matches!(
        n.entities[0].geometry.geometry(),
        Some(Curve::Bezier(_))
    ));
    assert!(n.subpaths[0].closed);
}

#[test]
fn n5_2_re_is_four_segments_and_a_closed_subpath() {
    let n = &normalized("10 20 30 40 re f")[0];
    assert_eq!(n.entities.len(), 4);
    assert_eq!(n.subpaths.len(), 1);
    assert!(n.subpaths[0].closed);
    assert_eq!(n.subpaths[0].entity_count, 4);
    let corners = [[10.0, 20.0], [40.0, 20.0], [40.0, 60.0], [10.0, 60.0]];
    for (i, e) in n.entities.iter().enumerate() {
        assert_eq!(
            polyline_ends(e.geometry.geometry().unwrap()),
            vec![bits(corners[i]), bits(corners[(i + 1) % 4])],
            "side {i}"
        );
        assert!(!matches!(e.geometry.geometry(), Some(Curve::Bezier(_))));
    }
}

#[test]
fn n5_2_each_subpath_keeps_its_own_closure_and_shares_one_group() {
    let n = &normalized("0 0 m 10 0 l 10 10 l h 50 50 m 60 50 l 60 60 l S")[0];
    assert_eq!(n.subpaths.len(), 2);
    assert!(n.subpaths[0].closed);
    assert!(!n.subpaths[1].closed);
    assert_eq!(n.entities.len(), 3 + 2);
    assert!(
        n.entities
            .iter()
            .all(|e| e.compound_group == n.compound_group)
    );
    assert_eq!(n.entities[0].subpath, 0);
    assert_eq!(n.entities[4].subpath, 1);
    assert_eq!(n.subpaths[0].first_entity, 0);
    assert_eq!(n.subpaths[1].first_entity, 3);
}

#[test]
fn n5_2_separate_paths_get_separate_groups() {
    let ps = normalized("0 0 m 10 10 l S 20 20 m 30 30 l S");
    assert_eq!(ps.len(), 2);
    assert_ne!(ps[0].compound_group, ps[1].compound_group);
}

#[test]
fn n5_2_lines_are_equivalent_curves_are_exact() {
    let n = &normalized("0 0 m 10 0 l 20 10 30 10 40 0 c S")[0];
    assert_eq!(
        n.entities[0].geometry.fidelity(),
        FidelityClass::SupportedEquivalent
    );
    assert_eq!(
        n.entities[1].geometry.fidelity(),
        FidelityClass::SupportedExact
    );
}

#[test]
fn n5_2_a_refused_segment_is_counted_and_classified_not_dropped() {
    // A zero-length line is degenerate to PVec. It stays an entity, with a class and a reason,
    // and nothing replaces it.
    let n = &normalized("10 10 m 10 10 l 50 50 l S")[0];
    assert_eq!(n.entities.len(), 2);
    let e = &n.entities[0];
    assert!(e.geometry.geometry().is_none());
    assert_eq!(e.geometry.fidelity(), FidelityClass::Unsupported);
    assert_eq!(e.classification.class(), FidelityClass::Unsupported);
    assert_eq!(
        e.classification.reason(),
        Some(ReasonCode::SplinePolicyRefused)
    );
    assert!(e.refusal.is_some());
    assert!(n.entities[1].geometry.geometry().is_some());
}

#[test]
fn n5_2_entities_follow_content_stream_order() {
    let n = &normalized("0 0 m 10 0 l 20 5 30 5 40 0 c 40 20 l S")[0];
    let kinds: Vec<_> = n
        .entities
        .iter()
        .map(|e| matches!(e.geometry.geometry(), Some(Curve::Bezier(_))))
        .collect();
    assert_eq!(kinds, vec![false, true, false]);
}

/// `(function name, parameter name)` for every `fn` in a source file, from the signature text.
fn fn_params(src: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let code: String = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let mut rest = code.as_str();
    while let Some(i) = rest.find("fn ") {
        let after = &rest[i + 3..];
        let name: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let Some(open) = after.find('(') else { break };
        let mut depth = 0usize;
        let mut end = open;
        for (j, c) in after[open..].char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + j;
                        break;
                    }
                }
                _ => {}
            }
        }
        for param in after[open + 1..end].split(',') {
            if let Some((pname, _)) = param.split_once(':') {
                out.push((name.clone(), pname.trim().to_ascii_lowercase()));
            }
        }
        rest = &after[end..];
    }
    out
}

#[test]
fn n5_2_no_function_takes_a_tolerance_step_count_or_segment_budget() {
    const BANNED: [&str; 9] = [
        "tol",
        "epsilon",
        "step",
        "budget",
        "sample",
        "flatness",
        "subdiv",
        "resolution",
        "eps",
    ];
    let sources = [
        ("pdf_vector.rs", include_str!("../src/pvec/pdf_vector.rs")),
        ("pdf_bezier.rs", include_str!("../src/pvec/pdf_bezier.rs")),
    ];
    for (file, src) in sources {
        let params = fn_params(src);
        assert!(!params.is_empty(), "{file}: the scanner found no signatures");
        for (f, p) in params {
            assert!(
                !BANNED.iter().any(|b| p.contains(b)) && p != "n" && p != "segments",
                "{file}: fn {f} takes `{p}`"
            );
        }
    }
    // The scanner must see a banned name when one is there.
    let planted = fn_params("fn flatten(path: &P, tolerance: f64) {}");
    assert!(planted.iter().any(|(_, p)| p.contains("tol")));
}

#[test]
fn n5_2_the_mapping_never_tessellates() {
    let src = include_str!("../src/pvec/pdf_bezier.rs");
    let code: String = src
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    for banned in ["flatten", "tessellat", "subdivide", "to_polyline", "eval("] {
        assert!(!code.contains(banned), "pdf_bezier.rs contains `{banned}`");
    }
    assert!(
        code.contains("Bezier::new"),
        "the mapping must go through PVec's Bezier"
    );
}
