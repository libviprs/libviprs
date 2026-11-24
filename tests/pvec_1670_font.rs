//! Issue #1670: a plot sets its text in an embedded face whose advance widths are Arial's, so a
//! label takes the room it takes in the plotted PDF. The face is a Latin subset of Liberation Sans
//! (SIL Open Font License 1.1, renamed as the licence asks) in `fonts/`, packaged with its licence
//! text. Every text record is set in it, because a text record does not say which face the drawing
//! named; `embedded_font(false)` goes back to the standard Helvetica. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;
use std::path::PathBuf;

use acadrust::entities::Text;
use acadrust::objects::ObjectType;
use acadrust::types::Vector3;
use acadrust::{CadDocument, DwgWriter, EntityType};
use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::PdfPlot;
use lopdf::{Dictionary, Document, Object};

fn plot(tag: &str, embedded: bool) -> Document {
    let dir = std::env::temp_dir().join(format!("libviprs-1670f-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mut doc = CadDocument::new();
    doc.add_layout("A3").unwrap();
    let layout = doc
        .objects
        .values_mut()
        .find_map(|o| match o {
            ObjectType::Layout(l) if l.name == "A3" => Some(l),
            _ => None,
        })
        .unwrap();
    layout.paper_width = 420.0;
    layout.paper_height = 297.0;
    layout.plot_paper_units = 1;
    layout.plot_type = 5;
    layout.plot_scale_numerator = 1.0;
    layout.plot_scale_denominator = 1.0;
    layout.paper_size = "ISO_A3_(420.00_x_297.00_MM)".into();
    layout.plot_flags.print_lineweights = true;
    let text = Text::with_value("Plan", Vector3::new(20.0, 20.0, 0.0)).with_height(5.0);
    doc.add_entity_to_layout(EntityType::Text(text), "A3")
        .unwrap();
    let bytes = DwgWriter::write_to_vec(&doc).unwrap();
    let dest = dir.join("out.pdf");
    PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("A3"))
        .layouts(["A3"])
        .texts(true)
        .embedded_font(embedded)
        .convert_with::<AcadrustProvider>()
        .unwrap();
    let pdf = Document::load(&dest).unwrap();
    let _ = fs::remove_dir_all(dir);
    pdf
}

fn font(pdf: &Document) -> Dictionary {
    let page = pdf.get_pages()[&1];
    let fonts = pdf.get_page_fonts(page).unwrap();
    assert_eq!(fonts.len(), 1, "one font on the page");
    (*fonts.values().next().unwrap()).clone()
}

fn name(o: &Object) -> String {
    String::from_utf8_lossy(o.as_name().unwrap()).into_owned()
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_text_is_set_in_an_embedded_true_type_face() {
    let pdf = plot("embedded", true);
    let f = font(&pdf);
    assert_eq!(name(f.get(b"Subtype").unwrap()), "TrueType");
    assert!(name(f.get(b"BaseFont").unwrap()).ends_with("LibviprsLatin-Regular"));
    let descriptor = pdf
        .get_dictionary(f.get(b"FontDescriptor").unwrap().as_reference().unwrap())
        .unwrap();
    let file = pdf
        .get_object(
            descriptor
                .get(b"FontFile2")
                .unwrap()
                .as_reference()
                .unwrap(),
        )
        .unwrap()
        .as_stream()
        .unwrap();
    let data = file
        .decompressed_content()
        .unwrap_or_else(|_| file.content.clone());
    assert_eq!(&data[..4], &[0, 1, 0, 0], "a TrueType font program");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_advance_widths_are_arials() {
    // Arial's widths in 1/1000 em: P 667, l 222, a 556, n 556.
    let pdf = plot("widths", true);
    let f = font(&pdf);
    let first = f.get(b"FirstChar").unwrap().as_i64().unwrap();
    let widths = f.get(b"Widths").unwrap().as_array().unwrap();
    let width = |c: u8| {
        let w = &widths[usize::from(c) - first as usize];
        w.as_float()
            .map(f64::from)
            .unwrap_or_else(|_| w.as_i64().unwrap() as f64)
    };
    assert_eq!(
        [b'P', b'l', b'a', b'n'].map(width),
        [667.0, 222.0, 556.0, 556.0]
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_text_is_selectable() {
    let pdf = plot("selectable", true);
    let text = pdf.extract_text(&[1]).unwrap();
    assert!(text.contains("Plan"), "{text:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_without_the_embedded_face_the_text_is_standard_helvetica() {
    let pdf = plot("helvetica", false);
    let f = font(&pdf);
    assert_eq!(name(f.get(b"BaseFont").unwrap()), "Helvetica");
    assert!(f.get(b"FontDescriptor").is_err());
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_licence_text_is_in_the_repository_and_packaged() {
    let licence = fs::read_to_string(root().join("fonts/LIBERATION-OFL.txt")).unwrap();
    assert!(licence.contains("SIL OPEN FONT LICENSE Version 1.1"));
    assert!(licence.contains("Red Hat"));
    assert!(root().join("fonts/LibviprsLatin-Regular.ttf").exists());
    // No include or exclude list in the manifest takes the directory out of the package.
    let manifest = fs::read_to_string(root().join("Cargo.toml")).unwrap();
    for line in manifest.lines().filter(|l| {
        let l = l.trim_start();
        l.starts_with("exclude") || l.starts_with("include")
    }) {
        assert!(!line.contains("fonts"), "{line}");
    }
    // The crate declares the face's licence too.
    assert!(
        manifest.contains("OFL-1.1"),
        "license in Cargo.toml does not name the font's licence"
    );
    // The renamed face does not carry the reserved name as its family.
    let ttf = fs::read(root().join("fonts/LibviprsLatin-Regular.ttf")).unwrap();
    let as_utf16: Vec<u8> = "Liberation Sans"
        .encode_utf16()
        .flat_map(u16::to_be_bytes)
        .collect();
    assert!(
        !ttf.windows(as_utf16.len())
            .any(|w| w == as_utf16.as_slice()),
        "the reserved name is still a name in the font"
    );
}

/// The name records of a TrueType font as (name id, text), read from its `name` table.
fn name_records(ttf: &[u8]) -> Vec<(u16, String)> {
    let be16 = |at: usize| u16::from_be_bytes([ttf[at], ttf[at + 1]]);
    let be32 = |at: usize| u32::from_be_bytes([ttf[at], ttf[at + 1], ttf[at + 2], ttf[at + 3]]);
    let tables = usize::from(be16(4));
    let table = (0..tables)
        .map(|i| 12 + 16 * i)
        .find(|&at| &ttf[at..at + 4] == b"name")
        .expect("a name table");
    let base = be32(table + 8) as usize;
    let count = usize::from(be16(base + 2));
    let strings = base + usize::from(be16(base + 4));
    (0..count)
        .map(|i| {
            let r = base + 6 + 12 * i;
            let (platform, id, len, off) = (
                be16(r),
                be16(r + 6),
                usize::from(be16(r + 8)),
                usize::from(be16(r + 10)),
            );
            let raw = &ttf[strings + off..strings + off + len];
            let text = if platform == 3 || platform == 0 {
                String::from_utf16_lossy(
                    &raw.chunks(2)
                        .map(|c| u16::from_be_bytes([c[0], c[1]]))
                        .collect::<Vec<_>>(),
                )
            } else {
                String::from_utf8_lossy(raw).into_owned()
            };
            (id, text)
        })
        .collect()
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1670_the_faces_names_carry_no_reserved_name() {
    // Name ids 1 (family), 3 (unique id), 4 (full name), 6 (PostScript name), 10 (description) and 16
    // (typographic family) name the font, and a modified font may not use a reserved name there: the
    // licence reserves Liberation, Arimo, Tinos and Cousine. The copyright (0) and trademark (7) notices
    // stay as the licence asks.
    let ttf = fs::read(root().join("fonts/LibviprsLatin-Regular.ttf")).unwrap();
    let records = name_records(&ttf);
    for (id, text) in &records {
        if [1u16, 3, 4, 6, 10, 16].contains(id) {
            for reserved in ["Liberation", "Arimo", "Tinos", "Cousine"] {
                assert!(!text.contains(reserved), "name {id} reads {text:?}");
            }
        }
    }
    let family = records.iter().find(|(id, _)| *id == 1).unwrap();
    assert_eq!(family.1, "Libviprs Latin");
}
