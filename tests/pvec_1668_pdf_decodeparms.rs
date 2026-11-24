//! Issue #1668: a content stream that carries `/DecodeParms` is read, not refused.
//!
//! The owner's plan (MASTER_FLOOR_PLAN_Rev.6.0) stops on "stream filter: /DecodeParms is not
//! supported" on every page. Its content streams are `/Filter /FlateDecode` with
//! `/DecodeParms <</Predictor 1>>`, which means "no prediction". The same stage can also carry
//! the PNG and TIFF predictors (what other writers emit), a `[null]` array that follows the filter
//! chain by position, or a predictor on a Form XObject. Every fixture is built in memory.
//!
//! The last test reads the real plan when `LIBVIPRS_1668_PDF` points at it. The file is the
//! owner's, it is never committed, and the test is ignored by default.

use std::io::Write;

use libviprs::pvec::pdf_vector::{
    PdfVectorError, PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes,
};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};

/// Three stroked segments, long enough to span several predictor rows.
const CONTENT: &str = "0 0 m 10 10 l S\n20 20 m 30 30 l S\n40 40 m 50 50 l S\n";

fn flate(raw: &[u8]) -> Vec<u8> {
    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    e.write_all(raw).unwrap();
    e.finish().unwrap()
}

fn name(n: &str) -> Object {
    Object::Name(n.as_bytes().to_vec())
}

/// A one-page PDF whose content stream has `dict` and `body`; `form` also paints a Form XObject.
fn pdf(dict: Dictionary, body: Vec<u8>, form: Option<(Dictionary, Vec<u8>)>) -> Vec<u8> {
    let mut doc = Document::with_version("1.7");
    let pages_id = doc.new_object_id();
    let mut resources = Dictionary::new();
    let mut content = Stream::new(dict, body);
    if let Some((mut fd, fbody)) = form {
        fd.set("Type", "XObject");
        fd.set("Subtype", "Form");
        fd.set("BBox", vec![0.into(), 0.into(), 100.into(), 100.into()]);
        let fid = doc.add_object(Stream::new(fd, fbody));
        resources.set("XObject", dictionary! { "Fm1" => fid });
        content = Stream::new(Dictionary::new(), b"/Fm1 Do".to_vec());
    }
    let cid = doc.add_object(content);
    let page = doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages_id,
        "MediaBox" => vec![0.into(), 0.into(), 200.into(), 100.into()],
        "Contents" => cid, "Resources" => resources,
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages", "Kids" => vec![page.into()], "Count" => 1,
        }),
    );
    let root = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", root);
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

fn run(bytes: &[u8]) -> Result<PdfVectorPage, PdfVectorError> {
    extract_page_paths_from_bytes(bytes, 1, &PdfVectorLimits::default())
}

fn read_ok(bytes: &[u8]) {
    let page = run(bytes).unwrap_or_else(|e| panic!("refused: {e}"));
    assert_eq!(page.entities.len(), 3, "all three segments are read");
}

/// Forward PNG predictor, 1 byte per pixel: row `i` uses filter `tags[i % tags.len()]`.
fn png(raw: &[u8], cols: usize, tags: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut prev = vec![0u8; cols];
    for (i, row) in raw.chunks(cols).enumerate() {
        let mut row = row.to_vec();
        row.resize(cols, b' ');
        let tag = tags[i % tags.len()];
        out.push(tag);
        for x in 0..cols {
            let a = if x > 0 { row[x - 1] } else { 0 };
            let b = prev[x];
            let c = if x > 0 { prev[x - 1] } else { 0 };
            let p = |a: u8, b: u8, c: u8| {
                let (ia, ib, ic) = (i16::from(a), i16::from(b), i16::from(c));
                let est = ia + ib - ic;
                let (pa, pb, pc) = ((est - ia).abs(), (est - ib).abs(), (est - ic).abs());
                if pa <= pb && pa <= pc {
                    a
                } else if pb <= pc {
                    b
                } else {
                    c
                }
            };
            let sub = match tag {
                0 => 0,
                1 => a,
                2 => b,
                3 => ((u16::from(a) + u16::from(b)) / 2) as u8,
                _ => p(a, b, c),
            };
            out.push(row[x].wrapping_sub(sub));
        }
        prev = row;
    }
    out
}

/// Forward TIFF predictor 2, 1 byte per pixel.
fn tiff(raw: &[u8], cols: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for row in raw.chunks(cols) {
        let mut line = row.to_vec();
        for x in (1..line.len()).rev() {
            line[x] = line[x].wrapping_sub(line[x - 1]);
        }
        out.extend(line);
    }
    out
}

fn parms(predictor: i64, columns: i64) -> Object {
    Object::Dictionary(dictionary! {
        "Predictor" => predictor, "Colors" => 1, "BitsPerComponent" => 8, "Columns" => columns,
    })
}

#[test]
fn flate_with_predictor_1_is_read() {
    // The owner's plan: /Filter /FlateDecode /DecodeParms <</Predictor 1>>.
    let dict = dictionary! {
        "Filter" => "FlateDecode",
        "DecodeParms" => dictionary! { "Predictor" => 1 },
    };
    read_ok(&pdf(dict, flate(CONTENT.as_bytes()), None));
}

#[test]
fn flate_with_a_null_decode_parms_array_is_read() {
    let dict = dictionary! {
        "Filter" => vec![name("FlateDecode")],
        "DecodeParms" => vec![Object::Null],
    };
    read_ok(&pdf(dict, flate(CONTENT.as_bytes()), None));
}

#[test]
fn ascii_hex_then_flate_with_two_null_parms_is_read() {
    let mut hex: String = flate(CONTENT.as_bytes())
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect();
    hex.push('>');
    let dict = dictionary! {
        "Filter" => vec![name("ASCIIHexDecode"), name("FlateDecode")],
        "DecodeParms" => vec![Object::Null, Object::Null],
    };
    read_ok(&pdf(dict, hex.into_bytes(), None));
}

#[test]
fn flate_with_a_png_up_predictor_is_read() {
    let dict = dictionary! { "Filter" => "FlateDecode", "DecodeParms" => parms(12, 16) };
    read_ok(&pdf(dict, flate(&png(CONTENT.as_bytes(), 16, &[2])), None));
}

#[test]
fn flate_with_an_optimum_png_predictor_and_mixed_row_filters_is_read() {
    let dict = dictionary! { "Filter" => "FlateDecode", "DecodeParms" => parms(15, 7) };
    read_ok(&pdf(
        dict,
        flate(&png(CONTENT.as_bytes(), 7, &[0, 1, 2, 3, 4])),
        None,
    ));
}

#[test]
fn flate_with_a_tiff_predictor_is_read() {
    let dict = dictionary! { "Filter" => "FlateDecode", "DecodeParms" => parms(2, 9) };
    read_ok(&pdf(dict, flate(&tiff(CONTENT.as_bytes(), 9)), None));
}

#[test]
fn a_form_xobject_with_a_predictor_is_read() {
    let fd = dictionary! { "Filter" => "FlateDecode", "DecodeParms" => parms(12, 16) };
    let body = flate(&png(CONTENT.as_bytes(), 16, &[2]));
    read_ok(&pdf(Dictionary::new(), Vec::new(), Some((fd, body))));
}

#[test]
fn an_unknown_predictor_is_a_typed_refusal_that_names_the_predictor() {
    let dict = dictionary! { "Filter" => "FlateDecode", "DecodeParms" => parms(99, 16) };
    let err = run(&pdf(dict, flate(CONTENT.as_bytes()), None)).expect_err("refused");
    let text = err.to_string();
    assert!(text.contains("/Predictor 99"), "{text}");
}

/// The owner's plan: pages 1, 2, 11, 32 and 64 of 64 all stopped on /DecodeParms.
#[test]
#[ignore = "needs LIBVIPRS_1668_PDF, the owner's plan, which is never committed"]
fn the_owners_plan_converts_on_the_pages_that_stopped() {
    let path = std::env::var("LIBVIPRS_1668_PDF").expect("LIBVIPRS_1668_PDF");
    let bytes = std::fs::read(path).unwrap();
    for page in [1, 2, 11, 32, 64] {
        let p = extract_page_paths_from_bytes(&bytes, page, &PdfVectorLimits::default())
            .unwrap_or_else(|e| panic!("page {page}: {e}"));
        assert!(!p.entities.is_empty(), "page {page} has paths");
    }
}
