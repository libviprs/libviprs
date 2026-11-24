//! Issue #1687: text in an embedded CID TrueType font (Type0, `/Identity-H` or `/Identity-V`) is
//! read, not dropped.
//!
//! A plotted drawing's labels are often Type0 fonts whose `/ToUnicode` CMap is stored as it is
//! (no `/Filter`), and the text reader only took a CMap it could inflate, so such a font looked
//! like it had no map and every label on the page was left out. The fonts here carry no program
//! (the text path never reads one); what they carry is what the reader does read: the encoding,
//! the `/ToUnicode` CMap, the `/W` widths and the `/CIDToGIDMap`. Every PDF is built in memory.
//!
//! The last test reads a real plan when `LIBVIPRS_1687_PDF` points at one. That file is the
//! owner's, it is never committed, and the test is ignored by default.

use std::io::Write;

use libviprs::pvec::pdf_vector::{
    PdfVectorError, PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes,
};
use libviprs::pvec::text_policy::TextFontOutcome;
use lopdf::{Dictionary, Document, Object, Stream, dictionary};

/// A CMap with three codes: 1 is A, 2 is B, 3 is C.
const CMAP: &str = "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
    /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
    1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
    3 beginbfchar\n<0001> <0041>\n<0002> <0042>\n<0003> <0043>\nendbfchar\n\
    endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n";

fn flate(raw: &[u8]) -> Vec<u8> {
    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    e.write_all(raw).unwrap();
    e.finish().unwrap()
}

/// How a font's `/ToUnicode` is stored.
enum Map {
    None,
    Plain(Vec<u8>),
    Flate(Vec<u8>),
}

/// A 200 by 100 page that shows `content` in the Type0 font `/F1`.
struct Fixture {
    encoding: &'static str,
    map: Map,
    cid_to_gid: Option<Object>,
    /// `/W`, when the font has one. Without it every glyph is the default 1000 units wide.
    widths: Option<Vec<Object>>,
    rotate: Option<i64>,
    content: &'static str,
}

impl Fixture {
    fn new(map: Map) -> Self {
        Self {
            encoding: "Identity-H",
            map,
            cid_to_gid: None,
            widths: None,
            rotate: None,
            content: "BT /F1 10 Tf 20 30 Td <000100020003> Tj ET",
        }
    }

    fn build(self) -> Vec<u8> {
        let mut doc = Document::with_version("1.7");
        let pages_id = doc.new_object_id();
        let content = doc.add_object(Stream::new(
            Dictionary::new(),
            self.content.as_bytes().to_vec(),
        ));
        let descriptor = doc.add_object(dictionary! {
            "Type" => "FontDescriptor", "FontName" => "SampleSans", "Flags" => 4,
            "CapHeight" => 700,
        });
        let mut cid = dictionary! {
            "Type" => "Font", "Subtype" => "CIDFontType2", "BaseFont" => "SampleSans",
            "FontDescriptor" => descriptor,
            "CIDSystemInfo" => dictionary! {
                "Registry" => Object::string_literal("Adobe"),
                "Ordering" => Object::string_literal("Identity"), "Supplement" => 0,
            },
        };
        if let Some(map) = self.cid_to_gid {
            cid.set("CIDToGIDMap", map);
        }
        if let Some(w) = self.widths {
            cid.set("W", w);
        }
        let cid = doc.add_object(cid);
        let mut font = dictionary! {
            "Type" => "Font", "Subtype" => "Type0", "BaseFont" => "SampleSans",
            "Encoding" => self.encoding, "DescendantFonts" => vec![cid.into()],
        };
        match self.map {
            Map::None => {}
            Map::Plain(body) => {
                let id = doc.add_object(Stream::new(Dictionary::new(), body));
                font.set("ToUnicode", id);
            }
            Map::Flate(body) => {
                let id = doc.add_object(Stream::new(
                    dictionary! { "Filter" => "FlateDecode" },
                    flate(&body),
                ));
                font.set("ToUnicode", id);
            }
        }
        let font = doc.add_object(font);
        let mut page = dictionary! {
            "Type" => "Page", "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), 200.into(), 100.into()],
            "Contents" => content,
            "Resources" => dictionary! { "Font" => dictionary! { "F1" => font } },
        };
        if let Some(r) = self.rotate {
            page.set("Rotate", r);
        }
        let page = doc.add_object(page);
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

    fn page(self) -> PdfVectorPage {
        extract_page_paths_from_bytes(&self.build(), 1, &PdfVectorLimits::default()).unwrap()
    }

    fn refusal(self, limits: &PdfVectorLimits) -> PdfVectorError {
        extract_page_paths_from_bytes(&self.build(), 1, limits).unwrap_err()
    }
}

fn plain() -> Map {
    Map::Plain(CMAP.as_bytes().to_vec())
}

/// The stored CMap, the case that reads 0 glyphs before the fix.
#[test]
fn an_identity_h_font_with_a_stored_tounicode_map_gives_its_text() {
    let page = Fixture::new(plain()).page();
    assert_eq!(page.texts.len(), 1, "one run for the one Tj");
    let run = &page.texts[0];
    assert_eq!(run.text, "ABC");
    assert_eq!(run.start, [20.0, 30.0]);
    // 3 glyphs at the default 1000 units, size 10: 10 points each.
    assert_eq!(run.end, [50.0, 30.0]);
    // `/CapHeight` 700 at size 10.
    assert!((run.cap_height - 7.0).abs() < 1e-9, "{}", run.cap_height);
}

#[test]
fn an_inflated_tounicode_map_still_gives_its_text() {
    let page = Fixture::new(Map::Flate(CMAP.as_bytes().to_vec())).page();
    assert_eq!(page.texts.len(), 1);
    assert_eq!(page.texts[0].text, "ABC");
}

#[test]
fn a_cid_to_gid_map_stream_does_not_change_the_text() {
    // Codes are CIDs and the map is by CID, so the CID to glyph map only picks glyphs: an
    // identity map written as a stream of 3 entries (6 bytes) reads the same.
    let bytes = with_cid_to_gid(Fixture::new(plain()), vec![0, 0, 0, 1, 0, 2, 0, 3]);
    let page = extract_page_paths_from_bytes(&bytes, 1, &PdfVectorLimits::default()).unwrap();
    assert_eq!(page.texts.len(), 1);
    assert_eq!(page.texts[0].text, "ABC");
}

#[test]
fn a_cid_to_gid_map_named_identity_does_not_change_the_text() {
    let mut f = Fixture::new(plain());
    f.cid_to_gid = Some(Object::Name(b"Identity".to_vec()));
    let page = f.page();
    assert_eq!(page.texts[0].text, "ABC");
}

/// Replaces the descendant font's `/CIDToGIDMap` with a stream holding `body`.
fn with_cid_to_gid(f: Fixture, body: Vec<u8>) -> Vec<u8> {
    let mut doc = Document::load_mem(&f.build()).unwrap();
    let stream = doc.add_object(Stream::new(Dictionary::new(), body));
    let ids: Vec<_> = doc
        .objects
        .iter()
        .filter(|(_, o)| {
            o.as_dict().is_ok_and(|d| {
                d.get(b"Subtype")
                    .is_ok_and(|s| s.as_name().is_ok_and(|n| n == b"CIDFontType2"))
            })
        })
        .map(|(id, _)| *id)
        .collect();
    for id in ids {
        if let Ok(Object::Dictionary(d)) = doc.get_object_mut(id) {
            d.set("CIDToGIDMap", stream);
        }
    }
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    out
}

#[test]
fn bfrange_entries_and_widths_are_read() {
    let cmap = CMAP.replace(
        "3 beginbfchar\n<0001> <0041>\n<0002> <0042>\n<0003> <0043>\nendbfchar",
        "1 beginbfrange\n<0001> <0003> <0041>\nendbfrange",
    );
    let mut f = Fixture::new(Map::Plain(cmap.into_bytes()));
    // CID 1 to 3 are 500 units wide.
    f.widths = Some(vec![1.into(), 3.into(), 500.into()]);
    let page = f.page();
    assert_eq!(page.texts[0].text, "ABC");
    assert_eq!(page.texts[0].end, [35.0, 30.0], "3 glyphs of 5 points");
}

#[test]
fn an_identity_v_font_reads_its_text_and_advances_down_the_page() {
    let mut f = Fixture::new(Map::Flate(CMAP.as_bytes().to_vec()));
    f.encoding = "Identity-V";
    let page = f.page();
    assert_eq!(page.texts.len(), 1, "Identity-V is read like Identity-H");
    let run = &page.texts[0];
    assert_eq!(run.text, "ABC");
    assert_eq!(run.start[0], run.end[0], "a vertical run keeps its x");
    // Default vertical advance is one em (1000 units) a glyph: 3 glyphs of 10 points, downward.
    assert_eq!(run.end[1], run.start[1] - 30.0);
}

#[test]
fn the_page_rotation_turns_the_run() {
    let mut f = Fixture::new(plain());
    f.rotate = Some(90);
    let page = f.page();
    let run = &page.texts[0];
    assert_eq!(run.text, "ABC");
    // On a page turned a quarter, a baseline that ran along x runs along y.
    assert!((run.end[0] - run.start[0]).abs() < 1e-9, "{run:?}");
    assert!(
        ((run.end[1] - run.start[1]).abs() - 30.0).abs() < 1e-9,
        "{run:?}"
    );
}

#[test]
fn a_font_without_tounicode_is_flagged_and_the_page_still_converts() {
    let page = Fixture::new(Map::None).page();
    assert!(page.texts.is_empty(), "there is nothing to decode it with");
    assert_eq!(page.text.pdf.total(), 1, "the operator is still counted");
    assert_eq!(
        page.text.by_outcome.get(&TextFontOutcome::CidNoToUnicode),
        Some(&1),
        "and the omission is flagged, not silent: {:?}",
        page.text.by_outcome
    );
}

#[test]
fn a_tounicode_map_over_the_font_byte_bound_is_a_typed_refusal() {
    // 1 MiB of comment text inflates from a few hundred bytes, against a 4 KiB bound.
    let mut body = CMAP.as_bytes().to_vec();
    body.extend(std::iter::repeat_n(b'%', 1 << 20));
    let limits = PdfVectorLimits {
        max_font_bytes: 4096,
        ..PdfVectorLimits::default()
    };
    match Fixture::new(Map::Flate(body)).refusal(&limits) {
        PdfVectorError::FontLimitExceeded {
            what,
            limit,
            observed,
        } => {
            assert_eq!(what, "font bytes");
            assert_eq!(limit, 4096);
            assert!(observed > 4096, "{observed}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_cmap_with_too_many_entries_is_a_typed_refusal() {
    // Three full ranges are 196 608 codes, past the 131 072 a font may map.
    let cmap = CMAP.replace(
        "3 beginbfchar\n<0001> <0041>\n<0002> <0042>\n<0003> <0043>\nendbfchar",
        "3 beginbfrange\n<0000> <FFFF> <0041>\n<0000> <FFFF> <0041>\n<0000> <FFFF> <0041>\nendbfrange",
    );
    match Fixture::new(Map::Flate(cmap.into_bytes())).refusal(&PdfVectorLimits::default()) {
        PdfVectorError::FontLimitExceeded { what, limit, .. } => {
            assert_eq!(what, "cmap entries");
            assert_eq!(limit, 131_072);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_cid_to_gid_map_with_more_than_65536_glyphs_is_a_typed_refusal() {
    let bytes = with_cid_to_gid(Fixture::new(plain()), vec![0u8; 2 * 65_536 + 2]);
    match extract_page_paths_from_bytes(&bytes, 1, &PdfVectorLimits::default()).unwrap_err() {
        PdfVectorError::FontLimitExceeded {
            what,
            limit,
            observed,
        } => {
            assert_eq!(what, "glyphs");
            assert_eq!(limit, 65_536);
            assert_eq!(observed, 65_537);
        }
        other => panic!("{other:?}"),
    }
}

/// One range over every code with a destination of `digits` hex digits.
fn range_cmap(digits: usize) -> String {
    CMAP.replace(
        "3 beginbfchar\n<0001> <0041>\n<0002> <0042>\n<0003> <0043>\nendbfchar",
        &format!(
            "1 beginbfrange\n<0000> <FFFF> <{}>\nendbfrange",
            "0041".repeat(digits / 4)
        ),
    )
}

#[test]
fn a_bfrange_destination_over_64_units_is_a_typed_refusal_before_anything_is_built() {
    // 65 536 codes times 5 000 units is 650 MB if each code keeps its own copy.
    let started = std::time::Instant::now();
    match Fixture::new(Map::Flate(range_cmap(20_000).into_bytes()))
        .refusal(&PdfVectorLimits::default())
    {
        PdfVectorError::FontLimitExceeded {
            what,
            limit,
            observed,
        } => {
            assert_eq!(what, "cmap destination");
            assert_eq!(limit, 64);
            assert_eq!(observed, 5_000);
        }
        other => panic!("{other:?}"),
    }
    assert!(started.elapsed().as_secs() < 2, "{:?}", started.elapsed());
}

#[test]
fn built_cmap_strings_count_against_the_font_byte_bound() {
    // 64 units a code is allowed, but 65 536 codes of 64 bytes are 4 MiB against a 1 MiB bound.
    let limits = PdfVectorLimits {
        max_font_bytes: 1 << 20,
        ..PdfVectorLimits::default()
    };
    match Fixture::new(Map::Flate(range_cmap(256).into_bytes())).refusal(&limits) {
        PdfVectorError::FontLimitExceeded { what, limit, .. } => {
            assert_eq!(what, "font bytes");
            assert_eq!(limit, 1 << 20);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_short_bfrange_destination_still_reads() {
    let cmap = CMAP.replace(
        "3 beginbfchar\n<0001> <0041>\n<0002> <0042>\n<0003> <0043>\nendbfchar",
        "1 beginbfrange\n<0001> <0003> <00410042>\nendbfrange",
    );
    // Two units a code, the last one counting up: AB, AC, AD.
    let page = Fixture::new(Map::Flate(cmap.into_bytes())).page();
    assert_eq!(page.texts[0].text, "ABACAD");
}

#[test]
fn a_w_array_with_too_many_widths_is_a_typed_refusal() {
    // `0 65535 500` three times: 196 608 widths.
    let mut w = Vec::new();
    for _ in 0..3 {
        w.extend([Object::from(0), Object::from(65_535), Object::from(500)]);
    }
    let mut f = Fixture::new(plain());
    f.widths = Some(w);
    match f.refusal(&PdfVectorLimits::default()) {
        PdfVectorError::FontLimitExceeded { what, limit, .. } => {
            assert_eq!(what, "cid widths");
            assert_eq!(limit, 131_072);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_default_font_bound_is_16_mib() {
    assert_eq!(PdfVectorLimits::default().max_font_bytes, 16 << 20);
}

/// A page's text reaches the PVec text stream as one record.
#[cfg(feature = "pvec-write")]
#[test]
fn the_text_of_a_stored_map_font_is_a_pvec_text_record() {
    use libviprs::pvec::pdf_page_to_pvec_with;
    use pvec_facade::PvecReader;
    use pvec_facade::components::TextStream;
    let mut f = Fixture::new(plain());
    // A page with no path at all has nothing to convert, so it gets a rule.
    f.content = "0 0 m 100 0 l S BT /F1 10 Tf 20 30 Td <000100020003> Tj ET";
    let page = f.page();
    let bytes = pdf_page_to_pvec_with(&page, "test", "0", false).unwrap();
    let streams = PvecReader::open(bytes).unwrap().streams().unwrap().0;
    let texts = TextStream::decode(
        &streams
            .iter()
            .find(|s| s.capability == TextStream::CAPABILITY)
            .expect("a text stream")
            .payload,
    )
    .unwrap();
    assert_eq!(texts.records.len(), 1);
}

/// A real plan: every page named in `LIBVIPRS_1687_PAGES` (default 1, 2, 3 and 11) gives text.
#[test]
#[cfg_attr(miri, ignore)]
#[ignore = "needs LIBVIPRS_1687_PDF, a real plan, which is never committed"]
fn a_real_plan_with_cid_truetype_labels_gives_text() {
    let path = std::env::var("LIBVIPRS_1687_PDF").expect("LIBVIPRS_1687_PDF");
    let pages = std::env::var("LIBVIPRS_1687_PAGES").unwrap_or_else(|_| "1,2,3,11".into());
    let bytes = std::fs::read(path).unwrap();
    for page in pages.split(',').map(|p| p.trim().parse::<usize>().unwrap()) {
        let p = extract_page_paths_from_bytes(&bytes, page, &PdfVectorLimits::default())
            .unwrap_or_else(|e| panic!("page {page}: {e}"));
        eprintln!(
            "page {page}: {} runs of {} operators",
            p.texts.len(),
            p.text.pdf.total()
        );
        assert!(!p.texts.is_empty(), "page {page} has text runs");
    }
}
