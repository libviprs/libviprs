//! Issue #1578, L2 and the alpha part of L4: `pdf_page_to_pvec` writes a shading pattern fill and
//! an `sh` as a gradient in stream 25 (stops sampled from the PDF function, the transform to
//! source space), the middle colour in the hatch record as the fallback, and the fill alpha (`ca`)
//! in the hatch colour. The page's own extraction is the oracle. Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use libviprs::pvec::pdf_page_to_pvec;
use libviprs::pvec::pdf_vector::{PdfVectorLimits, extract_page_paths_from_bytes};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};
use pvec_facade::PvecReader;
use pvec_facade::components::{
    FillPaintStream, FillSource, GradientCoords, HatchStream, ItemKind, PaintOrderStream,
};

/// A one page PDF (100 by 100) whose resources are `resources`, converted and decoded.
fn convert(
    resources: Dictionary,
    content: &str,
) -> (
    Option<FillPaintStream>,
    HatchStream,
    Option<PaintOrderStream>,
    Vec<u8>,
) {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let contents = doc.add_object(Stream::new(Dictionary::new(), content.as_bytes().to_vec()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
        "Contents" => contents, "Resources" => resources,
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages", "Kids" => vec![Object::from(page)], "Count" => 1,
        }),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog);
    let mut pdf = Vec::new();
    doc.save_to(&mut pdf).unwrap();
    let page = extract_page_paths_from_bytes(&pdf, 1, &PdfVectorLimits::default()).unwrap();
    let bytes = pdf_page_to_pvec(&page, "test", "0").unwrap();
    let (streams, _) = PvecReader::open(bytes.clone()).unwrap().streams().unwrap();
    let find = |cap| streams.iter().find(|s| s.capability == cap);
    (
        find(FillPaintStream::CAPABILITY).map(|s| FillPaintStream::decode(&s.payload).unwrap()),
        HatchStream::decode(
            &find(HatchStream::CAPABILITY)
                .expect("a hatch stream")
                .payload,
        )
        .unwrap(),
        find(PaintOrderStream::CAPABILITY).map(|s| PaintOrderStream::decode(&s.payload).unwrap()),
        bytes,
    )
}

fn nums(v: &[f64]) -> Object {
    Object::Array(v.iter().map(|x| Object::Real(*x as f32)).collect())
}

fn red_to_blue() -> Object {
    Object::Dictionary(dictionary! {
        "FunctionType" => 2, "Domain" => nums(&[0.0, 1.0]),
        "C0" => nums(&[1.0, 0.0, 0.0]), "C1" => nums(&[0.0, 0.0, 1.0]), "N" => 1,
    })
}

fn axial() -> Object {
    Object::Dictionary(dictionary! {
        "ShadingType" => 2, "ColorSpace" => "DeviceRGB",
        "Coords" => nums(&[0.0, 0.0, 100.0, 0.0]),
        "Extend" => vec![Object::Boolean(true), Object::Boolean(false)],
        "Function" => red_to_blue(),
    })
}

fn pattern_resources(matrix: Option<&[f64]>, shading: Object) -> Dictionary {
    let mut pattern = dictionary! { "PatternType" => 2, "Shading" => shading };
    if let Some(m) = matrix {
        pattern.set("Matrix", nums(m));
    }
    dictionary! { "Pattern" => dictionary! { "P1" => pattern } }
}

fn sh_resources(shading: Object) -> Dictionary {
    dictionary! { "Shading" => dictionary! { "Sh1" => shading } }
}

#[test]
fn an_axial_pattern_fill_is_a_gradient_in_stream_25() {
    let (fills, hatch, _, _) = convert(
        pattern_resources(None, axial()),
        "/Pattern cs /P1 scn 10 10 80 80 re f",
    );
    let fills = fills.expect("stream 25");
    assert_eq!(fills.gradients.len(), 1);
    assert_eq!(fills.assignments.len(), 1);
    assert_eq!(fills.assignments[0].paint, FillSource::Gradient(0));
    let g = &fills.gradients[0];
    assert_eq!(
        g.coords,
        GradientCoords::Axial {
            x0: 0.0,
            y0: 0.0,
            x1: 100.0,
            y1: 0.0
        }
    );
    assert!(g.extend_before && !g.extend_after);
    assert_eq!(g.stops.len(), 2);
    assert_eq!((g.stops[0].offset, g.stops[0].rgba), (0.0, 0xFF00_00FF));
    assert_eq!((g.stops[1].offset, g.stops[1].rgba), (1.0, 0x0000_FFFF));
    // PDF points to inches, y flipped: the page is 100 points high.
    let s = 1.0 / 72.0;
    assert_eq!(g.transform, [s, 0.0, 0.0, -s, 0.0, 100.0 * s]);
    // A reader that does not know 25 draws the colour in the middle of the ramp.
    match &hatch.records[0].fill {
        pvec_facade::components::Fill::Solid { rgba } => assert_eq!(*rgba, 0x8000_80FF),
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_pattern_matrix_moves_the_gradient_not_the_ctm() {
    let (fills, _, _, _) = convert(
        pattern_resources(Some(&[2.0, 0.0, 0.0, 2.0, 10.0, 20.0]), axial()),
        "3 0 0 3 5 5 cm /Pattern cs /P1 scn 0 0 10 10 re f",
    );
    let s = 1.0 / 72.0;
    assert_eq!(
        fills.unwrap().gradients[0].transform,
        [2.0 * s, 0.0, 0.0, -2.0 * s, 10.0 * s, 80.0 * s]
    );
}

#[test]
fn a_radial_pattern_keeps_both_circles() {
    let radial = Object::Dictionary(dictionary! {
        "ShadingType" => 3, "ColorSpace" => "DeviceRGB",
        "Coords" => nums(&[50.0, 50.0, 0.0, 50.0, 50.0, 40.0]),
        "Function" => red_to_blue(),
    });
    let (fills, _, _, _) = convert(
        pattern_resources(None, radial),
        "/Pattern cs /P1 scn 0 0 100 100 re f",
    );
    let g = &fills.unwrap().gradients[0];
    assert_eq!(
        g.coords,
        GradientCoords::Radial {
            x0: 50.0,
            y0: 50.0,
            r0: 0.0,
            x1: 50.0,
            y1: 50.0,
            r1: 40.0
        }
    );
    assert!(!g.extend_before && !g.extend_after);
}

#[test]
fn two_fills_of_one_pattern_share_one_gradient() {
    let (fills, hatch, _, _) = convert(
        pattern_resources(None, axial()),
        "/Pattern cs /P1 scn 0 0 10 10 re f 20 0 10 10 re f",
    );
    let fills = fills.unwrap();
    assert_eq!(hatch.records.len(), 2);
    assert_eq!(fills.gradients.len(), 1);
    assert!(
        fills
            .assignments
            .iter()
            .all(|a| a.paint == FillSource::Gradient(0))
    );
}

#[test]
fn a_plain_fill_after_a_pattern_fill_stays_flat() {
    let (fills, _, _, _) = convert(
        pattern_resources(None, axial()),
        "/Pattern cs /P1 scn 0 0 10 10 re f 1 0 0 rg 20 0 10 10 re f",
    );
    let sources: Vec<_> = fills.unwrap().assignments.iter().map(|a| a.paint).collect();
    assert_eq!(sources, [FillSource::Gradient(0), FillSource::Own]);
}

#[test]
fn sh_under_a_rectangular_clip_is_a_hatch_record_over_the_clip() {
    let (fills, hatch, order, _) = convert(sh_resources(axial()), "10 10 50 50 re W n /Sh1 sh");
    assert_eq!(hatch.records.len(), 1);
    assert_eq!(hatch.records[0].boundary.len(), 1);
    let fills = fills.expect("stream 25");
    assert_eq!(fills.assignments[0].paint, FillSource::Gradient(0));
    // The rule is nonzero: the boundary is one rectangle.
    let order = order.expect("paint order");
    assert_eq!(
        order.items().map(|(k, _)| k).collect::<Vec<_>>(),
        [ItemKind::Fill]
    );
}

#[test]
fn sh_without_a_clip_covers_the_page() {
    let (_, hatch, _, _) = convert(sh_resources(axial()), "/Sh1 sh");
    assert_eq!(hatch.records.len(), 1);
}

#[test]
fn sh_follows_the_paths_painted_before_it_in_the_paint_order() {
    let (fills, _, order, _) = convert(
        sh_resources(axial()),
        "0 0 10 10 re f q 20 20 30 30 re W n /Sh1 sh Q 0 0 5 5 re f",
    );
    let kinds: Vec<_> = order.unwrap().items().map(|(k, _)| k).collect();
    assert_eq!(kinds, [ItemKind::Fill, ItemKind::Fill, ItemKind::Fill]);
    // The second hatch record is the shading, so it was written between the two paths.
    let sources: Vec<_> = fills.unwrap().assignments.iter().map(|a| a.paint).collect();
    assert_eq!(
        sources,
        [FillSource::Own, FillSource::Gradient(0), FillSource::Own]
    );
}

#[test]
fn fill_alpha_is_the_alpha_byte_of_the_hatch_colour() {
    let resources = dictionary! { "ExtGState" => dictionary! {
    "G1" => dictionary! { "ca" => Object::Real(0.5), "CA" => Object::Real(0.25) } } };
    let (_, hatch, _, bytes) = convert(resources, "/G1 gs 1 0 0 rg 0 0 50 50 re f");
    let alpha = match &hatch.records[0].fill {
        pvec_facade::components::Fill::Solid { rgba } => (*rgba & 0xFF) as u8,
        other => panic!("{other:?}"),
    };
    assert!((127..=128).contains(&alpha), "{alpha}");
    drop(bytes);
}

#[test]
fn fill_alpha_reaches_the_stops_of_a_gradient() {
    let resources = {
        let mut r = pattern_resources(None, axial());
        r.set(
            "ExtGState",
            dictionary! { "G1" => dictionary! { "ca" => Object::Real(0.5) } },
        );
        r
    };
    let (fills, hatch, _, _) = convert(resources, "/G1 gs /Pattern cs /P1 scn 0 0 100 100 re f");
    let g = &fills.unwrap().gradients[0];
    assert!(
        g.stops
            .iter()
            .all(|s| (127..=128).contains(&(s.rgba & 0xFF)))
    );
    match &hatch.records[0].fill {
        pvec_facade::components::Fill::Solid { rgba } => {
            assert!((127..=128).contains(&(rgba & 0xFF)))
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_opaque_fill_is_still_opaque() {
    let (_, hatch, _, _) = convert(Dictionary::new(), "1 0 0 rg 0 0 50 50 re f");
    match &hatch.records[0].fill {
        pvec_facade::components::Fill::Solid { rgba } => assert_eq!(*rgba, 0xFF00_00FF),
        other => panic!("{other:?}"),
    }
}
