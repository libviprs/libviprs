//! N5.5 (#1084): the v1 text capability policy. Text is omitted and counted.
//!
//! The PDFs are built in memory with `lopdf`, so the content streams and the font
//! dictionaries under test are visible here. The decision itself, and the measurements
//! behind it, are in `docs/pvec-text-policy.md`.

use libviprs::cad::ItemHandle;
use libviprs::pvec::pdf_vector::{PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes};
use libviprs::pvec::text_policy::{TextFontOutcome as F, TextReport, shx_outcome};
use libviprs::pvec::{InstancePath, ProvenanceStub, RawEntity, RawGeometry};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};

struct Fixture {
    doc: Document,
    pages_id: lopdf::ObjectId,
    kids: Vec<Object>,
}

impl Fixture {
    fn new() -> Self {
        let mut doc = Document::with_version("1.7");
        let pages_id = doc.new_object_id();
        Self {
            doc,
            pages_id,
            kids: Vec::new(),
        }
    }

    fn stream(&mut self, body: &str) -> lopdf::ObjectId {
        self.doc
            .add_object(Stream::new(Dictionary::new(), body.as_bytes().to_vec()))
    }

    fn form(&mut self, body: &str, extra: Dictionary) -> lopdf::ObjectId {
        let mut d = dictionary! {
            "Type" => "XObject",
            "Subtype" => "Form",
            "BBox" => vec![0.into(), 0.into(), 1000.into(), 1000.into()],
        };
        for (k, v) in extra.iter() {
            d.set(k.clone(), v.clone());
        }
        self.doc
            .add_object(Stream::new(d, body.as_bytes().to_vec()))
    }

    fn page_with(
        &mut self,
        media: [i64; 4],
        contents: Object,
        resources: Option<Dictionary>,
        extra: Dictionary,
    ) {
        let mut page = dictionary! {
            "Type" => "Page",
            "Parent" => self.pages_id,
            "MediaBox" => media.iter().map(|v| Object::Integer(*v)).collect::<Vec<_>>(),
            "Contents" => contents,
        };
        if let Some(r) = resources {
            page.set("Resources", r);
        }
        for (k, v) in extra.iter() {
            page.set(k.clone(), v.clone());
        }
        let id = self.doc.add_object(page);
        self.kids.push(id.into());
    }

    fn bytes(mut self) -> Vec<u8> {
        let n = self.kids.len() as i64;
        self.doc.objects.insert(
            self.pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => self.kids.clone(),
                "Count" => n,
            }),
        );
        let catalog = self.doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => self.pages_id,
        });
        self.doc.trailer.set("Root", catalog);
        let mut out = Vec::new();
        self.doc.save_to(&mut out).unwrap();
        out
    }
}

fn run(bytes: &[u8]) -> PdfVectorPage {
    extract_page_paths_from_bytes(bytes, 1, &PdfVectorLimits::default()).unwrap()
}

/// A one-page PDF with `fonts` as the page's `/Font` dictionary.
fn with_fonts(body: &str, fonts: Dictionary, extra: Vec<(&str, Object)>) -> Vec<u8> {
    let mut f = Fixture::new();
    let s = f.stream(body);
    let mut res = dictionary! { "Font" => fonts };
    for (k, v) in extra {
        res.set(k, v);
    }
    f.page_with([0, 0, 200, 100], s.into(), Some(res), Dictionary::new());
    f.bytes()
}

fn font_file(doc_fixture: &mut Fixture) -> lopdf::ObjectId {
    doc_fixture.stream("not a real font program")
}

fn descriptor(f: &mut Fixture, name: &str, embedded: bool) -> lopdf::ObjectId {
    let mut d = dictionary! { "Type" => "FontDescriptor", "FontName" => Object::Name(name.as_bytes().to_vec()) };
    if embedded {
        let ff = font_file(f);
        d.set("FontFile2", ff);
    }
    f.doc.add_object(d)
}

/// One page showing `/F1`, whose dictionary `make` builds.
fn one_font(make: impl FnOnce(&mut Fixture) -> Dictionary) -> PdfVectorPage {
    let mut f = Fixture::new();
    let font = make(&mut f);
    let font_id = f.doc.add_object(font);
    let s = f.stream("BT /F1 12 Tf (Hello) Tj ET");
    let res = dictionary! { "Font" => dictionary! { "F1" => font_id } };
    f.page_with([0, 0, 200, 100], s.into(), Some(res), Dictionary::new());
    run(&f.bytes())
}

fn only(page: &PdfVectorPage, outcome: F) {
    assert_eq!(page.text.pdf.total(), 1);
    assert_eq!(
        page.text
            .by_outcome
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect::<Vec<_>>(),
        vec![(outcome, 1)],
        "wrong bucket"
    );
}

#[test]
fn n5_5_every_text_showing_operator_is_counted() {
    let page = run(&with_fonts(
        "BT /F1 12 Tf (a) Tj (b) Tj (c) Tj [(d) -20 (e)] TJ [(f)] TJ (g) ' 1 2 (h) \" ET 0 0 m 5 5 l S",
        dictionary! {},
        vec![],
    ));
    assert_eq!(page.text.pdf.tj, 3);
    assert_eq!(page.text.pdf.tj_array, 2);
    assert_eq!(page.text.pdf.quote, 1);
    assert_eq!(page.text.pdf.double_quote, 1);
    assert_eq!(page.text.omitted(), 7);
}

#[test]
fn n5_5_text_inside_a_form_is_counted_once_per_invocation() {
    let mut f = Fixture::new();
    let form = f.form("BT /F1 9 Tf (in form) Tj [(x)] TJ ET", Dictionary::new());
    let s = f.stream("BT (page) Tj ET /Fm Do /Fm Do");
    let res = dictionary! { "XObject" => dictionary! { "Fm" => form } };
    f.page_with([0, 0, 200, 100], s.into(), Some(res), Dictionary::new());
    let page = run(&f.bytes());
    assert_eq!(page.text.pdf.tj, 1 + 2);
    assert_eq!(page.text.pdf.tj_array, 2);
}

#[test]
fn n5_5_a_page_without_text_reports_zero() {
    let page = run(&with_fonts("0 0 m 9 9 l S", dictionary! {}, vec![]));
    assert_eq!(page.text, TextReport::default());
    assert_eq!(page.text.omitted(), 0);
}

#[test]
fn n5_5_text_never_becomes_geometry() {
    let with = run(&with_fonts(
        "10 10 m 50 10 l S BT /F1 12 Tf 20 20 Td (label) Tj ET 10 20 m 50 20 l S",
        dictionary! {},
        vec![],
    ));
    let without = run(&with_fonts(
        "10 10 m 50 10 l S 10 20 m 50 20 l S",
        dictionary! {},
        vec![],
    ));
    assert_eq!(with.text.omitted(), 1);
    // Same entities, same order, apart from the sequence-free fields being equal.
    assert_eq!(with.entities, without.entities);
}

#[test]
fn n5_5_non_embedded_font() {
    let page = one_font(|f| {
        let d = descriptor(f, "Arial", false);
        dictionary! { "Type" => "Font", "Subtype" => "TrueType", "BaseFont" => "Arial", "FontDescriptor" => d }
    });
    only(&page, F::NotEmbedded);
}

#[test]
fn n5_5_standard_font_without_a_descriptor_is_not_embedded() {
    let page = one_font(
        |_| dictionary! { "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica" },
    );
    only(&page, F::NotEmbedded);
}

#[test]
fn n5_5_subset_embedded_font() {
    let page = one_font(|f| {
        let d = descriptor(f, "ABCDEF+Arial", true);
        dictionary! { "Type" => "Font", "Subtype" => "TrueType", "BaseFont" => "ABCDEF+Arial", "FontDescriptor" => d }
    });
    only(&page, F::SubsetEmbedded);
}

#[test]
fn n5_5_fully_embedded_font_is_not_a_subset() {
    let page = one_font(|f| {
        let d = descriptor(f, "Arial", true);
        dictionary! { "Type" => "Font", "Subtype" => "TrueType", "BaseFont" => "Arial", "FontDescriptor" => d }
    });
    only(&page, F::Embedded);
}

fn type0(f: &mut Fixture, to_unicode: bool) -> Dictionary {
    let d = descriptor(f, "ABCDEF+Cid", true);
    let desc = f.doc.add_object(dictionary! {
        "Type" => "Font", "Subtype" => "CIDFontType2", "BaseFont" => "ABCDEF+Cid",
        "FontDescriptor" => d,
    });
    let mut font = dictionary! {
        "Type" => "Font", "Subtype" => "Type0", "BaseFont" => "ABCDEF+Cid",
        "Encoding" => "Identity-H", "DescendantFonts" => vec![desc.into()],
    };
    if to_unicode {
        let tu = f.stream("/CIDInit /ProcSet findresource begin end");
        font.set("ToUnicode", tu);
    }
    font
}

#[test]
fn n5_5_cid_font_without_tounicode() {
    only(&one_font(|f| type0(f, false)), F::CidNoToUnicode);
}

#[test]
fn n5_5_cid_font_with_tounicode_follows_its_embedding() {
    only(&one_font(|f| type0(f, true)), F::SubsetEmbedded);
}

#[test]
fn n5_5_a_font_that_does_not_resolve() {
    let page = run(&with_fonts(
        "BT /Nope 12 Tf (x) Tj ET BT (y) Tj ET",
        dictionary! {},
        vec![],
    ));
    assert_eq!(page.text.by_outcome.get(&F::Unresolved), Some(&2));
}

#[test]
fn n5_5_the_font_is_graphics_state_and_q_restores_it() {
    let mut f = Fixture::new();
    let arial = f.doc.add_object(
        dictionary! { "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica" },
    );
    let s = f.stream("BT /F1 12 Tf (a) Tj ET q /F2 9 Tf (b) Tj Q (c) Tj");
    let res = dictionary! { "Font" => dictionary! { "F1" => arial } };
    f.page_with([0, 0, 200, 100], s.into(), Some(res), Dictionary::new());
    let page = run(&f.bytes());
    // a and c use F1 (not embedded), b names a font that is missing.
    assert_eq!(page.text.by_outcome.get(&F::NotEmbedded), Some(&2));
    assert_eq!(page.text.by_outcome.get(&F::Unresolved), Some(&1));
}

fn entity(class: &str) -> RawEntity {
    RawEntity {
        style: libviprs::pvec::EntityStyle::absent(),
        geometry: RawGeometry::Unmapped {
            reason: format!("no mapping for {class}"),
        },
        provenance: ProvenanceStub {
            handle: Some(ItemHandle::new(1)),
            class_name: class.into(),
            block_path: InstancePath::default(),
            layer: "0".into(),
        },
    }
}

#[test]
fn n5_5_dwg_text_and_mtext_are_counted_and_other_classes_are_not() {
    let mut r = TextReport::default();
    r.count_dwg(&[
        entity("TEXT"),
        entity("TEXT"),
        entity("MTEXT"),
        entity("LINE"),
        entity("ATTRIB"),
    ]);
    assert_eq!((r.dwg.text, r.dwg.mtext, r.omitted()), (2, 1, 3));
}

#[test]
fn n5_5_a_missing_shx_font_has_its_own_bucket() {
    assert_eq!(shx_outcome("romans.shx", &["txt.shx"]), F::ShxMissing);
    assert_eq!(shx_outcome("ROMANS", &["romans.SHX"]), F::Embedded);
    let mut r = TextReport::default();
    r.count_dwg(&[entity("TEXT"), entity("MTEXT")]);
    r.bucket_dwg("gothice.shx", &[], 2);
    assert_eq!(r.by_outcome.get(&F::ShxMissing), Some(&2));
}

#[test]
#[cfg_attr(miri, ignore)]
fn n5_5_outcome_codes_are_stable_and_documented() {
    let doc = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/docs/pvec-text-policy.md"
    ))
    .expect("docs/pvec-text-policy.md");
    for o in F::ALL {
        assert!(
            doc.contains(o.code()),
            "{} is not in the policy doc",
            o.code()
        );
    }
    assert!(doc.contains("Decision: omit text and report the counts"));
    // Exactly one option is chosen.
    assert_eq!(doc.matches("Decision:").count(), 1);
}

/// Real DWG files, read through the acadrust provider. `g13_text.dwg` holds one `TEXT` and one
/// `MTEXT`; `sample_AC1018.dwg` holds 29 and 5 (counted by `examples/text_policy_measure.rs`).
#[cfg(feature = "dwg-acadrust")]
mod dwg_files {
    use super::*;
    use libviprs::cad::CadSource;
    use libviprs::pvec::dwg_acadrust::AcadrustProvider;
    use libviprs::pvec::{DwgProvider, EntityScope};

    fn count(rel: &str) -> TextReport {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let p = AcadrustProvider::open(CadSource::Path(&path)).unwrap();
        let mut all = Vec::new();
        for l in p.layouts().unwrap() {
            all.extend(
                p.entities(&EntityScope::Layout(l.id))
                    .unwrap()
                    .collect_all()
                    .unwrap(),
            );
        }
        let mut r = TextReport::default();
        r.count_dwg(&all);
        r
    }

    #[test]
    fn n5_5_g13_text_has_one_text_and_one_mtext() {
        let r = count("tests/fixtures/cad/g13/g13_text.dwg");
        assert_eq!((r.dwg.text, r.dwg.mtext), (1, 1));
    }

    #[test]
    fn n5_5_a_dwg_without_text_counts_zero() {
        assert_eq!(count("tests/fixtures/cad/g13/g13_line.dwg").omitted(), 0);
    }

    #[test]
    fn n5_5_sample_ac1018_counts() {
        let r = count("tests/fixtures/cad/sample_AC1018.dwg");
        assert_eq!((r.dwg.text, r.dwg.mtext), (29, 5));
    }
}
