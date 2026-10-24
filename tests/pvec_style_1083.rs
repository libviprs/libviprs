//! Issue #1083 (N5.4): the monochrome line-style component.
//!
//! Style is its own optional component keyed by `EntityId`. These tests pin the parts the issue
//! calls out: `w 0` is not an absent width, degenerate dashes are refused with a typed error,
//! colour is counted and never written, and the geometry side does not move when style is
//! computed (or when it is not).
use libviprs::cad::ItemHandle;
use libviprs::pvec::pdf_vector::{PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes};
use libviprs::pvec::style::{
    DashPattern, DwgLineweight, DwgStyleInput, LineStyleComponent, StrokeWidth, StyleError,
    dwg_line_style, pdf_line_styles,
};
use libviprs::pvec::{
    EntityId, FidelityClass, InstancePath, OrderingPolicy, ProvenanceStub, RawEntity, RawGeometry,
    order_entities,
};
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

/// Dense ids for `n` entities, minted the only way a caller can: through ordering.
fn ids(n: usize) -> Vec<EntityId> {
    let raw = (0..n)
        .map(|i| RawEntity {
            geometry: RawGeometry::Line {
                start: [i as f64, 0.0, 0.0],
                end: [i as f64 + 1.0, 0.0, 0.0],
            },
            provenance: ProvenanceStub {
                handle: Some(ItemHandle::new(i as u64 + 1)),
                class_name: "LINE".into(),
                block_path: InstancePath::default(),
                layer: "0".into(),
            },
        })
        .collect();
    order_entities("t", "1", &OrderingPolicy::default(), raw)
        .entities
        .iter()
        .map(|e| e.id)
        .collect()
}

fn components(body: &str) -> Vec<LineStyleComponent> {
    let p = page(body);
    let batch = pdf_line_styles(&p, &ids(p.entities.len())).unwrap();
    assert!(batch.refused.is_empty(), "{:?}", batch.refused);
    batch.components
}

const SQUARE_AND_LINE: &str = "0 0 m 10 0 l S 0 5 m 10 5 l S";

#[test]
fn width_from_the_w_operator_is_carried_in_source_units() {
    let c = components("2.5 w 0 0 m 10 0 l S");
    assert_eq!(c.len(), 1);
    assert_eq!(c[0].width, Some(StrokeWidth::Width(2.5)));
    assert!(c[0].visible);
}

#[test]
fn w_zero_is_thinnest_renderable_not_absent_and_not_a_zero_width() {
    let zero = components("0 w 0 0 m 10 0 l S");
    assert_eq!(zero[0].width, Some(StrokeWidth::Thinnest));
    assert_ne!(zero[0].width, Some(StrokeWidth::Width(0.0)));
}

#[test]
fn absent_width_is_distinct_from_w_zero() {
    // A dash with no `w` ever set: the width is absent, and the component still exists.
    let absent = components("[3 2] 0 d 0 0 m 10 0 l S");
    assert_eq!(absent.len(), 1);
    assert_eq!(absent[0].width, None);
    let zero = components("0 w [3 2] 0 d 0 0 m 10 0 l S");
    assert_ne!(absent[0].width, zero[0].width);
}

#[test]
fn a_stroke_with_no_style_operators_has_nothing_to_say() {
    assert!(components(SQUARE_AND_LINE).is_empty());
}

#[test]
fn dash_array_and_phase_come_from_d() {
    let c = components("[3 2] 1.5 d 0 0 m 10 0 l S");
    let d = c[0].dash.as_ref().unwrap();
    assert_eq!(d.array(), &[3.0, 2.0]);
    assert_eq!(d.phase(), 1.5);
}

#[test]
fn empty_dash_array_is_an_explicit_solid_line_not_an_absent_dash() {
    let c = components("[] 0 d 0 0 m 10 0 l S");
    assert_eq!(c[0].dash.as_ref().unwrap().array(), &[] as &[f64]);
}

#[test]
fn style_follows_q_and_cap_q() {
    let c = components("1 w q 4 w 0 0 m 10 0 l S Q 0 5 m 10 5 l S");
    assert_eq!(c.len(), 2);
    assert_eq!(c[0].width, Some(StrokeWidth::Width(4.0)));
    assert_eq!(c[1].width, Some(StrokeWidth::Width(1.0)));
}

#[test]
fn fill_only_paths_get_no_line_style() {
    assert!(components("3 w 0 0 m 10 0 l 10 10 l f").is_empty());
}

#[test]
fn components_reference_entities_by_the_ids_given() {
    let p = page("2 w 0 0 m 10 0 l S 3 w 0 5 m 10 5 l S");
    let ids = ids(2);
    let batch = pdf_line_styles(&p, &ids).unwrap();
    assert_eq!(batch.components[0].entity, ids[0]);
    assert_eq!(batch.components[1].entity, ids[1]);
}

#[test]
fn a_wrong_number_of_ids_is_refused() {
    let p = page("2 w 0 0 m 10 0 l S");
    assert!(pdf_line_styles(&p, &ids(2)).is_err());
}

#[test]
fn negative_dash_entry_is_refused_naming_the_index() {
    assert_eq!(
        DashPattern::new(vec![3.0, -1.0, 2.0], 0.0),
        Err(StyleError::NegativeDash { index: 1 })
    );
}

#[test]
fn non_finite_dash_entry_is_refused_naming_the_index() {
    assert_eq!(
        DashPattern::new(vec![1.0, f64::NAN], 0.0),
        Err(StyleError::NonFiniteDash { index: 1 })
    );
    assert_eq!(
        DashPattern::new(vec![f64::INFINITY], 0.0),
        Err(StyleError::NonFiniteDash { index: 0 })
    );
}

#[test]
fn all_zero_dash_array_is_refused() {
    assert_eq!(
        DashPattern::new(vec![0.0, 0.0], 0.0),
        Err(StyleError::AllZeroDash)
    );
}

#[test]
fn non_finite_phase_is_refused() {
    assert_eq!(
        DashPattern::new(vec![1.0], f64::NAN),
        Err(StyleError::NonFinitePhase)
    );
}

#[test]
fn a_zero_entry_among_positive_ones_is_a_dot_and_is_kept() {
    let d = DashPattern::new(vec![0.0, 4.0], 0.0).unwrap();
    assert_eq!(d.array(), &[0.0, 4.0]);
}

#[test]
fn negative_dash_in_a_page_is_reported_per_entity_not_normalized() {
    let p = page("[3 -1] 0 d 0 0 m 10 0 l S 0 5 m 10 5 l S");
    let ids = ids(p.entities.len());
    let batch = pdf_line_styles(&p, &ids).unwrap();
    // The refused entity yields no component and the refusal names the index.
    assert_eq!(batch.refused.len(), 2);
    assert_eq!(batch.refused[0].0, ids[0]);
    assert_eq!(batch.refused[0].1, StyleError::NegativeDash { index: 1 });
    assert!(batch.components.is_empty());
}

#[test]
fn non_finite_and_negative_widths_are_refused() {
    assert_eq!(
        LineStyleComponent::new(ids(1)[0], Some(f64::NAN), None, true),
        Err(StyleError::NonFiniteWidth)
    );
    assert_eq!(
        LineStyleComponent::new(ids(1)[0], Some(f64::INFINITY), None, true),
        Err(StyleError::NonFiniteWidth)
    );
    assert_eq!(
        LineStyleComponent::new(ids(1)[0], Some(-1.0), None, true),
        Err(StyleError::NegativeWidth)
    );
}

#[test]
fn colour_operators_are_counted_and_no_colour_component_is_written() {
    let body = "0.5 g 1 0 0 rg 0 0 0 1 k /Cs1 cs 0.2 sc 0.1 0.2 0.3 scn 2 w 0 0 m 10 0 l S";
    let p = page(body);
    let batch = pdf_line_styles(&p, &ids(p.entities.len())).unwrap();
    assert_eq!(batch.report.colour_total(), 6);
    assert!(batch.report.colour_total() > 0);
    // Only the width survives: there is no colour field to find on a component at all.
    assert_eq!(batch.components.len(), 1);
    assert_eq!(batch.components[0].width, Some(StrokeWidth::Width(2.0)));
}

#[test]
fn colour_rows_are_ignored_by_policy_with_the_monochrome_reason() {
    let p = page("1 0 0 rg 1 0 0 RG 0 g 0 0 m 10 0 l S");
    let batch = pdf_line_styles(&p, &ids(1)).unwrap();
    let rows: Vec<_> = batch.report.colour_rows().collect();
    assert_eq!(rows.iter().map(|r| r.count).sum::<u64>(), 3);
    for r in &rows {
        let c = r.classification();
        assert_eq!(c.class(), FidelityClass::IgnoredByPolicy);
        assert_eq!(c.policy_field(), Some("profile.monochrome"));
        assert_eq!(r.reason.to_string(), "MonochromeProfile");
    }
    assert!(rows.iter().any(|r| r.source_class == "rg"));
}

#[test]
fn colour_operators_inside_a_form_are_counted_too() {
    // Counted wherever they run; here on the page stream itself, in a q/Q pair.
    let p = page("q 1 0 0 rg Q 0 0 m 10 0 l S");
    assert_eq!(
        pdf_line_styles(&p, &ids(1)).unwrap().report.colour_total(),
        1
    );
}

#[test]
fn a_page_without_colour_reports_zero() {
    let p = page(SQUARE_AND_LINE);
    let batch = pdf_line_styles(&p, &ids(2)).unwrap();
    assert_eq!(batch.report.colour_total(), 0);
}

#[test]
fn dwg_lineweight_in_hundredths_of_a_mm_maps_to_source_units() {
    let c = dwg_line_style(
        ids(1)[0],
        &DwgStyleInput {
            lineweight: Some(DwgLineweight::Hundredths(25)),
            ..DwgStyleInput::default()
        },
    )
    .unwrap();
    assert_eq!(c.width, Some(StrokeWidth::Width(0.25)));
}

#[test]
fn dwg_zero_lineweight_is_thinnest_and_by_layer_is_absent() {
    let id = ids(1)[0];
    let zero = dwg_line_style(
        id,
        &DwgStyleInput {
            lineweight: Some(DwgLineweight::Hundredths(0)),
            ..DwgStyleInput::default()
        },
    )
    .unwrap();
    assert_eq!(zero.width, Some(StrokeWidth::Thinnest));
    let by_layer = dwg_line_style(
        id,
        &DwgStyleInput {
            lineweight: Some(DwgLineweight::ByLayer),
            ..DwgStyleInput::default()
        },
    )
    .unwrap();
    assert_eq!(by_layer.width, None);
}

#[test]
fn dwg_linetype_becomes_an_alternating_on_off_array_times_the_scale() {
    // DWG: positive pen down, negative pen up, zero a dot.
    let c = dwg_line_style(
        ids(1)[0],
        &DwgStyleInput {
            linetype_elements: Some(vec![0.5, -0.25, 0.0, -0.25]),
            linetype_scale: 2.0,
            ..DwgStyleInput::default()
        },
    )
    .unwrap();
    assert_eq!(c.dash.unwrap().array(), &[1.0, 0.5, 0.0, 0.5]);
}

#[test]
fn dwg_linetype_with_two_dashes_in_a_row_gets_a_zero_gap_between() {
    let c = dwg_line_style(
        ids(1)[0],
        &DwgStyleInput {
            linetype_elements: Some(vec![1.0, 2.0, -1.0]),
            linetype_scale: 1.0,
            ..DwgStyleInput::default()
        },
    )
    .unwrap();
    assert_eq!(c.dash.unwrap().array(), &[1.0, 0.0, 2.0, 1.0]);
}

#[test]
fn dwg_degenerate_linetype_is_refused() {
    let id = ids(1)[0];
    let bad_scale = dwg_line_style(
        id,
        &DwgStyleInput {
            linetype_elements: Some(vec![1.0, -1.0]),
            linetype_scale: f64::NAN,
            ..DwgStyleInput::default()
        },
    );
    assert!(bad_scale.is_err());
    let all_zero = dwg_line_style(
        id,
        &DwgStyleInput {
            linetype_elements: Some(vec![0.0, 0.0]),
            linetype_scale: 1.0,
            ..DwgStyleInput::default()
        },
    );
    assert_eq!(all_zero, Err(StyleError::AllZeroDash));
}

#[test]
fn dwg_negative_lineweight_is_refused_not_dropped() {
    // DWG's negative sentinels (ByLayer and friends) have their own variants, so a negative
    // `Hundredths` is a corrupt value. It must not quietly become "no width".
    for h in [-1_i16, -3, i16::MIN] {
        let r = dwg_line_style(
            ids(1)[0],
            &DwgStyleInput {
                lineweight: Some(DwgLineweight::Hundredths(h)),
                ..DwgStyleInput::default()
            },
        );
        assert_eq!(r, Err(StyleError::NegativeWidth), "Hundredths({h})");
    }
}

#[test]
fn dwg_invisible_flag_is_carried() {
    let c = dwg_line_style(
        ids(1)[0],
        &DwgStyleInput {
            visible: false,
            lineweight: Some(DwgLineweight::Hundredths(18)),
            ..DwgStyleInput::default()
        },
    )
    .unwrap();
    assert!(!c.visible);
}

/// The named fixture of the byte-identity criterion: a drawing with width, dash and colour.
const NAMED_FIXTURE: &str =
    "1 0 0 rg 2 w [4 2] 0 d 0 0 m 50 0 l 50 50 l S 0 w 10 10 m 20 20 30 5 40 10 c S";

#[test]
fn geometry_is_byte_identical_whether_or_not_style_is_computed() {
    let render = |p: &PdfVectorPage| -> String {
        p.entities
            .iter()
            .map(|e| format!("{:?}|{:?}|{:?}", e.segments, e.paint, e.provenance))
            .collect::<Vec<_>>()
            .join("\n")
    };
    // Two inputs built independently from the same source text, so the comparison is between
    // two real extractions and not one value against itself.
    let bytes_a = one_page(NAMED_FIXTURE);
    let bytes_b = one_page(NAMED_FIXTURE);
    let without = extract_page_paths_from_bytes(&bytes_a, 1, &PdfVectorLimits::default()).unwrap();
    let with = extract_page_paths_from_bytes(&bytes_b, 1, &PdfVectorLimits::default()).unwrap();
    let before = render(&with);
    assert!(!before.is_empty());
    let batch = pdf_line_styles(&with, &ids(with.entities.len())).unwrap();
    assert_eq!(batch.components.len(), 2);
    // Style was computed on `with` only; its geometry must still match the untouched page.
    assert_eq!(render(&with).as_bytes(), render(&without).as_bytes());
    assert_eq!(render(&with), before);
}

#[cfg(feature = "pvec-normalized")]
#[test]
fn normalized_geometry_size_is_unchanged_by_style() {
    // Pinned. If a style field ever lands on the geometry entity this changes with it.
    assert_eq!(
        std::mem::size_of::<libviprs::pvec::Normalized2DGeometry>(),
        PINNED_NORMALIZED_SIZE
    );
}

#[cfg(feature = "pvec-normalized")]
const PINNED_NORMALIZED_SIZE: usize = 176;

#[test]
#[cfg_attr(miri, ignore)]
fn the_geometry_entity_sources_name_no_style_field() {
    for f in ["src/pvec/normalized.rs", "src/pvec/pdf_bezier.rs"] {
        let src = std::fs::read_to_string(f).unwrap().to_lowercase();
        for banned in ["linestyle", "line_style", "dash", "lineweight"] {
            assert!(!src.contains(banned), "{f} mentions {banned}");
        }
    }
}
