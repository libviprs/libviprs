//! A white fill with an alpha under 255 still reaches the PVec (planviewer needs it to blend a
//! white wash over a raster); a fully opaque white fill is still dropped, the viewer's page is
//! white anyway. Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use libviprs::pvec::pdf_page_to_pvec;
use libviprs::pvec::pdf_vector::{PdfVectorLimits, extract_page_paths_from_bytes};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};
use pvec_facade::PvecReader;
use pvec_facade::components::{Fill, HatchStream};

/// The fill colours (`0xRRGGBBAA`) of the hatch records a 100 by 100 page with `content` converts to.
fn hatch_colours(content: &str, alpha: Option<f32>) -> Vec<u32> {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let contents = doc.add_object(Stream::new(Dictionary::new(), content.as_bytes().to_vec()));
    let resources = match alpha {
        Some(a) => dictionary! { "ExtGState" => dictionary! {
        "G1" => dictionary! { "ca" => Object::Real(a) } } },
        None => Dictionary::new(),
    };
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
    let (streams, _) = PvecReader::open(bytes).unwrap().streams().unwrap();
    let Some(hatch) = streams
        .iter()
        .find(|s| s.capability == HatchStream::CAPABILITY)
    else {
        return Vec::new();
    };
    HatchStream::decode(&hatch.payload)
        .unwrap()
        .records
        .iter()
        .map(|r| match &r.fill {
            Fill::Solid { rgba } => *rgba,
            other => panic!("{other:?}"),
        })
        .collect()
}

#[test]
fn a_half_transparent_white_fill_keeps_its_record_with_the_alpha() {
    let got = hatch_colours("/G1 gs 1 1 1 rg 0 0 50 50 re f", Some(0.5));
    assert_eq!(got.len(), 1, "{got:x?}");
    assert_eq!(got[0] >> 8, 0xFF_FFFF);
    assert!((127..=128).contains(&(got[0] & 0xFF)), "{:x}", got[0]);
}

#[test]
fn an_opaque_white_fill_is_still_dropped() {
    assert!(hatch_colours("1 1 1 rg 0 0 50 50 re f", None).is_empty());
    assert!(hatch_colours("/G1 gs 1 1 1 rg 0 0 50 50 re f", Some(1.0)).is_empty());
}

#[test]
fn a_half_transparent_grey_fill_is_unchanged() {
    let got = hatch_colours("/G1 gs 0.5 g 0 0 50 50 re f", Some(0.5));
    assert_eq!(got.len(), 1);
}
