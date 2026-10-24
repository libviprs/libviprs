//! Issue #1085 (N5.6): the hatch and fill policy.
//!
//! The PDF fixtures are content streams built in memory so the operators under test are visible
//! here. The DWG counts come from the committed `g13_hatch` and `g13_solid` drawings, which are
//! the in-tree stand-ins until the O2.1/O2.2 corpus is integrated (see the policy doc).

use libviprs::pvec::fill_policy::{
    FILL_POLICY, FillConstruct, FillDecision, FillReport, policy_for,
};
use libviprs::pvec::pdf_vector::{PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes};
use libviprs::pvec::FidelityClass;
use lopdf::{Dictionary, Document, Object, Stream, dictionary};

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

const RECT: &str = "10 10 40 20 re";

#[test]
fn n5_6_every_construct_has_exactly_one_row_with_a_reason() {
    for c in FillConstruct::ALL {
        let rows: Vec<_> = FILL_POLICY.iter().filter(|r| r.construct == c).collect();
        assert_eq!(rows.len(), 1, "{c:?}");
        assert!(!rows[0].reason.is_empty());
    }
    let mut reasons: Vec<_> = FILL_POLICY.iter().map(|r| r.reason).collect();
    reasons.sort_unstable();
    reasons.dedup();
    assert_eq!(reasons.len(), FILL_POLICY.len(), "reasons are unique");
}

#[test]
fn n5_6_no_decision_carries_area_as_triangles() {
    // The only decisions on offer keep existing geometry or count and drop. A tessellating
    // answer is not representable, which is panel finding 6 as a type.
    for r in FILL_POLICY {
        assert!(matches!(
            r.decision,
            FillDecision::GeometryOnly | FillDecision::IgnoredCounted
        ));
    }
    assert_eq!(
        policy_for(FillConstruct::PdfShading).bucket,
        FidelityClass::RequiresExtension
    );
    assert_eq!(
        policy_for(FillConstruct::PdfPattern).bucket,
        FidelityClass::RequiresExtension
    );
}

#[test]
fn n5_6_the_doc_has_a_row_for_every_construct_and_reason() {
    let doc = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/pvec-hatch-policy.md"),
    )
    .expect("docs/pvec-hatch-policy.md");
    for r in FILL_POLICY {
        assert!(
            doc.contains(&format!("`{}`", r.reason)),
            "the doc never names reason {}",
            r.reason
        );
    }
    for needle in ["HATCH", "SOLID", "TRACE", "f*", "sh", "/Pattern", "g13_hatch", "g13_solid"] {
        assert!(doc.contains(needle), "the doc never mentions {needle}");
    }
}

#[test]
fn n5_6_pdf_fill_operators_are_counted_by_winding_rule() {
    let body = format!(
        "{RECT} f {RECT} F {RECT} f* {RECT} B {RECT} B* {RECT} b {RECT} b* {RECT} S {RECT} s {RECT} n"
    );
    let mut report = FillReport::new();
    report.count_pdf_page(&page(&body));
    assert_eq!(report.count(FillConstruct::PdfFillNonZero), 4); // f F B b
    assert_eq!(report.count(FillConstruct::PdfFillEvenOdd), 3); // f* B* b*
    assert_eq!(report.count(FillConstruct::PdfShading), 0);
    assert_eq!(report.count(FillConstruct::PdfPattern), 0);
}

#[test]
fn n5_6_pdf_shadings_and_patterns_are_counted_not_dropped_silently() {
    let body = format!("/Sh1 sh /Pattern cs /P1 scn {RECT} f /P2 SCN 0.5 g /Sh2 sh");
    let p = page(&body);
    assert_eq!(p.shading_paints, 2);
    assert_eq!(p.pattern_selections, 2);
    let mut report = FillReport::new();
    report.count_pdf_page(&p);
    let rows: Vec<_> = report.rows().collect();
    let shading = rows
        .iter()
        .find(|r| r.construct == FillConstruct::PdfShading)
        .unwrap();
    assert_eq!(shading.count, 2);
    assert_eq!(shading.policy.reason, "fill-shading-requires-extension");
    // A plain colour operand is not a pattern.
    let plain = page("1 0 0 scn 0 1 0 SCN 10 10 5 5 re f");
    assert_eq!(plain.pattern_selections, 0);
}

#[cfg(feature = "pvec-normalized")]
mod normalized {
    use super::*;
    use libviprs::pvec::dump::dump_to_string;
    use libviprs::pvec::pdf_bezier::normalize_page;
    use pvec_geometry::Curve;

    const CIRCLE: &str = "100 50 m 100 66.5 86.5 80 70 80 c 53.5 80 40 66.5 40 50 c \
                          40 33.5 53.5 20 70 20 c 86.5 20 100 33.5 100 50 c h";

    #[test]
    fn n5_6_a_filled_curve_stays_a_bezier_never_a_triangle_or_point_set() {
        let paths = normalize_page(&page(&format!("{CIRCLE} f*")));
        let mut curves = 0;
        for e in paths.iter().flat_map(|p| &p.entities) {
            match e.geometry.geometry() {
                Some(Curve::Bezier(_)) => curves += 1,
                other => panic!("a filled cubic became {other:?}"),
            }
        }
        assert_eq!(curves, 4);
    }

    #[test]
    fn n5_6_dropping_the_fill_does_not_change_geometry_bytes() {
        let dump = |body: &str| {
            let paths = normalize_page(&page(body));
            dump_to_string(
                paths
                    .iter()
                    .flat_map(|p| &p.entities)
                    .map(|e| (&e.geometry, None)),
            )
        };
        let stroked = dump(&format!("{CIRCLE} S"));
        assert!(!stroked.is_empty());
        assert_eq!(stroked, dump(&format!("{CIRCLE} f")));
        assert_eq!(stroked, dump(&format!("{CIRCLE} f*")));
        assert_eq!(stroked, dump(&format!("{CIRCLE} B*")));
    }
}

#[cfg(feature = "dwg-acadrust")]
mod dwg {
    use super::*;
    use libviprs::cad::CadSource;
    use libviprs::pvec::dwg_acadrust::AcadrustProvider;
    use libviprs::pvec::{DwgProvider, EntityScope, RawEntity, RawGeometry};

    /// Every entity of every scope in the drawing, blocks included.
    fn all(name: &str) -> Vec<RawEntity> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/cad/g13")
            .join(format!("{name}.dwg"));
        let p = AcadrustProvider::open(CadSource::Path(&path)).unwrap();
        let mut out = Vec::new();
        for l in p.layouts().unwrap() {
            out.extend(
                p.entities(&EntityScope::Layout(l.id))
                    .unwrap()
                    .collect_all()
                    .unwrap(),
            );
        }
        out
    }

    #[test]
    fn n5_6_g13_hatch_reports_its_hatches() {
        let es = all("g13_hatch");
        let mut r = FillReport::new();
        r.count_dwg_entities(&es);
        assert_eq!(r.count(FillConstruct::DwgHatch), 1);
        assert_eq!(r.count(FillConstruct::DwgSolid), 0);
        // Nothing a hatch brings is a mapped triangle or point set.
        assert!(es
            .iter()
            .filter(|e| e.provenance.class_name.eq_ignore_ascii_case("HATCH"))
            .all(|e| matches!(e.geometry, RawGeometry::Unmapped { .. })));
    }

    #[test]
    fn n5_6_g13_solid_reports_its_solids() {
        let mut r = FillReport::new();
        r.count_dwg_entities(&all("g13_solid"));
        assert_eq!(r.count(FillConstruct::DwgSolid), 1);
        assert_eq!(r.count(FillConstruct::DwgHatch), 0);
    }
}
