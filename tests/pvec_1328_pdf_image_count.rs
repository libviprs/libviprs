//! Issue #1328 (R2.4): PDF images are counted, with their CTMs, beside the fill constructs.
//!
//! No placement is written for a PDF image in this task (the bake rules are the follow-up), so
//! the report says "counted, not placed". Fixtures are built in memory with `lopdf`.

use libviprs::pvec::fill_policy::{FillConstruct, FillDecision, FillReport, policy_for};
use libviprs::pvec::pdf_vector::{PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};

fn image_xobject(doc: &mut Document) -> lopdf::ObjectId {
    doc.add_object(Stream::new(
        dictionary! {
            "Type" => "XObject", "Subtype" => "Image",
            "Width" => 1, "Height" => 1, "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8,
        },
        vec![0x80],
    ))
}

/// A one page PDF. `form_body` is the content of a form `/Fm` that the page also invokes.
fn pdf(body: &str, images: usize, form_body: Option<&str>) -> Vec<u8> {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let mut xobjects = Dictionary::new();
    for i in 0..images {
        let id = image_xobject(&mut doc);
        xobjects.set(format!("Im{}", i + 1), id);
    }
    if let Some(form_body) = form_body {
        // The form shares the page's image names, as a form without its own resources does.
        let form = doc.add_object(Stream::new(
            dictionary! {
                "Type" => "XObject", "Subtype" => "Form",
                "BBox" => vec![0.into(), 0.into(), 1000.into(), 1000.into()],
                "Matrix" => vec![2.into(), 0.into(), 0.into(), 2.into(), 0.into(), 0.into()],
            },
            form_body.as_bytes().to_vec(),
        ));
        xobjects.set("Fm", form);
    }
    let content = doc.add_object(Stream::new(Dictionary::new(), body.as_bytes().to_vec()));
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        "Contents" => content,
        "Resources" => dictionary! { "XObject" => xobjects },
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages", "Kids" => vec![Object::from(page)], "Count" => 1,
        }),
    );
    let catalog = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog);
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

fn page(bytes: &[u8]) -> PdfVectorPage {
    extract_page_paths_from_bytes(bytes, 1, &PdfVectorLimits::default()).unwrap()
}

const INLINE: &str = "BI /W 2 /H 2 /CS /Gray /BPC 8 ID \u{1}\u{2}\u{3}\u{4} EI";

#[test]
fn n_image_xobjects_and_m_inline_images_are_counted_with_their_ctms() {
    // Two image XObject paints (the second under a nested cm), two inline images (one under cm).
    let body = format!(
        "q 100 0 0 50 10 20 cm /Im1 Do Q \
         q 0.5 0 0 0.5 0 0 cm 200 0 0 100 300 400 cm /Im2 Do Q \
         {INLINE} \
         q 30 0 0 30 5 6 cm {INLINE} Q"
    );
    let p = page(&pdf(&body, 2, None));

    let xobjects: Vec<_> = p.images.iter().filter(|i| i.name.is_some()).collect();
    let inline: Vec<_> = p.images.iter().filter(|i| i.name.is_none()).collect();
    assert_eq!((xobjects.len(), inline.len()), (2, 2));

    assert_eq!(xobjects[0].name.as_deref(), Some("Im1"));
    assert_eq!(xobjects[0].ctm, [100.0, 0.0, 0.0, 50.0, 10.0, 20.0]);
    assert_eq!(xobjects[1].name.as_deref(), Some("Im2"));
    assert_eq!(xobjects[1].ctm, [100.0, 0.0, 0.0, 50.0, 150.0, 200.0]);
    assert_eq!(inline[0].ctm, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    assert_eq!(inline[1].ctm, [30.0, 0.0, 0.0, 30.0, 5.0, 6.0]);
}

#[test]
fn an_image_inside_a_form_carries_the_form_path_and_the_form_matrix() {
    let p = page(&pdf(
        "q 1 0 0 1 10 10 cm /Fm Do Q",
        1,
        Some("q 10 0 0 10 0 0 cm /Im1 Do Q"),
    ));
    assert_eq!(p.images.len(), 1);
    assert_eq!(p.images[0].form_path, vec!["Fm".to_string()]);
    // Form /Matrix (scale 2) then the page's translate, then the cm inside the form.
    assert_eq!(p.images[0].ctm, [20.0, 0.0, 0.0, 20.0, 10.0, 10.0]);
}

#[test]
fn a_form_xobject_is_not_an_image_and_a_pdf_without_images_reports_zero() {
    let p = page(&pdf("10 10 40 20 re f", 0, None));
    assert!(p.images.is_empty());
    let mut report = FillReport::new();
    report.count_pdf_page(&p);
    assert_eq!(report.count(FillConstruct::PdfImageXObject), 0);
    assert_eq!(report.count(FillConstruct::PdfInlineImage), 0);
    assert!(report.rows().all(|r| !matches!(
        r.construct,
        FillConstruct::PdfImageXObject | FillConstruct::PdfInlineImage
    )));
}

#[test]
fn the_report_counts_images_and_says_no_placement_is_written() {
    let body = format!("q 10 0 0 10 0 0 cm /Im1 Do Q /Im1 Do {INLINE}");
    let p = page(&pdf(&body, 1, None));
    let mut report = FillReport::new();
    report.count_pdf_page(&p);
    assert_eq!(report.count(FillConstruct::PdfImageXObject), 2);
    assert_eq!(report.count(FillConstruct::PdfInlineImage), 1);
    for c in [
        FillConstruct::PdfImageXObject,
        FillConstruct::PdfInlineImage,
    ] {
        let row = policy_for(c);
        assert_eq!(row.decision, FillDecision::IgnoredCounted, "{c:?}");
        assert!(row.reason.contains("not-placed"), "{}", row.reason);
    }
}

#[test]
fn counting_is_deterministic_and_a_malformed_inline_image_is_still_an_error() {
    let body = format!("{INLINE} q 2 0 0 2 0 0 cm {INLINE} Q");
    let bytes = pdf(&body, 0, None);
    assert_eq!(page(&bytes).images, page(&bytes).images);
    let broken = pdf("BI /W 2 /H 2 ID 1234", 0, None);
    assert!(extract_page_paths_from_bytes(&broken, 1, &PdfVectorLimits::default()).is_err());
}
