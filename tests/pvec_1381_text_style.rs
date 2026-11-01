//! Issue #1381 (N5.5b, slice of #1086 and #1313): the text style crosses the provider boundary,
//! so a DWG whose style names an SHX font that is not installed reaches `TEXT_SHX_MISSING`
//! through a real drawing, not through a hand-built entity.
//!
//! The drawings are written by acadrust's `DwgWriter` and read back through the provider. Needs
//! `--features "dwg-acadrust pvec-normalized"`.
#![cfg(all(feature = "dwg-acadrust", feature = "pvec-normalized"))]

use acadrust::entities::{MText, Text};
use acadrust::tables::TextStyle;
use acadrust::types::Vector3;
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::text_policy::{TextFontOutcome, TextReport};
use libviprs::pvec::{
    Availability, DwgProvider, EntityScope, Facet, LayoutKind, ProviderError, RawEntity,
};

fn style(name: &str, font: &str) -> TextStyle {
    let mut s = TextStyle::new(name);
    s.font_file = font.to_string();
    s
}

/// A TEXT on style GONE (font nope.shx), an MTEXT on style OK (font txt.shx), a TEXT on a style
/// the table does not have, and a LINE-free model space otherwise.
fn drawing() -> Vec<u8> {
    let mut doc = CadDocument::new();
    doc.text_styles.add(style("GONE", "nope.shx")).unwrap();
    doc.text_styles.add(style("OK", "txt.shx")).unwrap();
    let mut a = Text::new();
    a.value = "a".into();
    a.insertion_point = Vector3::new(1.0, 1.0, 0.0);
    a.height = 2.0;
    a.style = "GONE".into();
    doc.add_entity(EntityType::Text(a)).unwrap();
    let mut b = MText::new();
    b.value = "b".into();
    b.style = "OK".into();
    doc.add_entity(EntityType::MText(b)).unwrap();
    let mut c = Text::new();
    c.value = "c".into();
    c.height = 2.0;
    c.style = "NOT_IN_THE_TABLE".into();
    doc.add_entity(EntityType::Text(c)).unwrap();
    DwgWriter::write_to_vec(&doc).expect("write")
}

fn read(bytes: &[u8]) -> (AcadrustProvider, EntityScope, Vec<RawEntity>) {
    let p = AcadrustProvider::open(CadSource::Bytes(bytes)).expect("open");
    let m = p
        .layouts()
        .unwrap()
        .into_iter()
        .find(|l| l.kind == LayoutKind::Model)
        .unwrap();
    let scope = EntityScope::Layout(m.id);
    let es = p.entities(&scope).unwrap().collect_all().unwrap();
    (p, scope, es)
}

#[test]
fn the_provider_claims_the_text_style_facet() {
    assert!(Facet::ALL.contains(&Facet::TextStyle));
    let (p, _, _) = read(&drawing());
    assert_eq!(
        p.caps().availability(Facet::TextStyle),
        Availability::Reported
    );
}

#[test]
fn text_and_mtext_carry_their_style_name_and_font_files() {
    let (p, scope, es) = read(&drawing());
    let styles = p.text_styles(&scope).unwrap();
    let text: Vec<_> = es
        .iter()
        .filter(|e| matches!(e.provenance.class_name.as_str(), "TEXT" | "MTEXT"))
        .collect();
    assert_eq!(
        text.len(),
        3,
        "{:?}",
        es.iter()
            .map(|e| &e.provenance.class_name)
            .collect::<Vec<_>>()
    );
    let of = |e: &RawEntity| styles.get(&e.provenance.handle.unwrap()).cloned();
    let gone = of(text[0]).expect("TEXT on GONE");
    assert_eq!(gone.style_name, "GONE");
    assert_eq!(gone.font_file, "nope.shx");
    let ok = of(text[1]).expect("MTEXT on OK");
    assert_eq!(ok.style_name, "OK");
    assert_eq!(ok.font_file, "txt.shx");
    // A style the table does not have still reports the name the entity used.
    let missing = of(text[2]).expect("TEXT on a style that is not in the table");
    assert_eq!(missing.style_name, "NOT_IN_THE_TABLE");
    assert!(missing.font_file.is_empty());
}

#[test]
fn a_scope_the_drawing_does_not_have_is_a_scope_error() {
    let (p, _, _) = read(&drawing());
    let e = p
        .text_styles(&EntityScope::BlockDefinition {
            name: "NOPE".into(),
            reached_via: libviprs::pvec::InstancePath::root(),
        })
        .unwrap_err();
    assert!(matches!(e, ProviderError::NoSuchScope(_)), "{e:?}");
}

#[test]
fn a_missing_shx_font_reaches_text_shx_missing_through_a_real_drawing() {
    let (p, scope, es) = read(&drawing());
    let styles = p.text_styles(&scope).unwrap();
    let mut r = TextReport::default();
    r.count_dwg(&es);
    r.bucket_dwg_entities(&es, &styles, &["txt.shx"]);
    assert_eq!((r.dwg.text, r.dwg.mtext), (2, 1));
    let n = |o: TextFontOutcome| r.by_outcome.get(&o).copied().unwrap_or(0);
    assert_eq!(n(TextFontOutcome::ShxMissing), 1, "{:?}", r.by_outcome);
    assert_eq!(
        n(TextFontOutcome::Embedded),
        1,
        "txt.shx is installed: {:?}",
        r.by_outcome
    );
    assert_eq!(
        n(TextFontOutcome::Unresolved),
        1,
        "a style the table lacks is unresolved, not missing: {:?}",
        r.by_outcome
    );
    assert_eq!(r.by_outcome.values().sum::<u64>(), r.dwg.total());
    assert_eq!(TextFontOutcome::ShxMissing.code(), "TEXT_SHX_MISSING");
}

#[test]
fn nothing_is_installed_means_every_resolved_style_is_missing() {
    let (p, scope, es) = read(&drawing());
    let styles = p.text_styles(&scope).unwrap();
    let mut r = TextReport::default();
    r.count_dwg(&es);
    r.bucket_dwg_entities(&es, &styles, &[]);
    assert_eq!(r.by_outcome.get(&TextFontOutcome::ShxMissing), Some(&2));
}
