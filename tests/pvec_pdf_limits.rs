//! N5.8 (#1087): malformed-PDF and resource-limit tests for the vector path.
//!
//! Every fixture is built in memory inside the test, so there is nothing to tear down
//! and nothing external to depend on. Each limit has a fixture that trips it and an
//! error that names the limit, the configured value and the observed value, in the
//! shape `tests/decode_alloc_refusal_shape.rs` uses for the raster side (a typed
//! variant a caller matches on, never a string).
//!
//! Peak allocation is measured with a counting allocator that tracks the calling
//! thread only, so the other tests in this binary do not move the numbers.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::Write;

use libviprs::pvec::pdf_vector::{
    PdfVectorError, PdfVectorLimits, PdfVectorPage, extract_page_paths_from_bytes,
};
use lopdf::{Dictionary, Document, Object, Stream, dictionary};

struct Counting;

thread_local! {
    static LIVE: Cell<usize> = const { Cell::new(0) };
    static PEAK: Cell<usize> = const { Cell::new(0) };
}

fn bump(n: usize) {
    let _ = LIVE.try_with(|l| {
        l.set(l.get() + n);
        let _ = PEAK.try_with(|p| p.set(p.get().max(l.get())));
    });
}

fn drop_live(n: usize) {
    let _ = LIVE.try_with(|l| l.set(l.get().saturating_sub(n)));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        drop_live(layout.size());
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        drop_live(layout.size());
        bump(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Bytes above the starting level that `f` held live at its high-water mark.
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let start = LIVE.with(Cell::get);
    PEAK.with(|p| p.set(start));
    let out = f();
    (out, PEAK.with(Cell::get).saturating_sub(start))
}

const MIB: usize = 1 << 20;

struct Pdf {
    doc: Document,
    pages_id: lopdf::ObjectId,
}

impl Pdf {
    fn new() -> Self {
        let mut doc = Document::with_version("1.7");
        let pages_id = doc.new_object_id();
        Self { doc, pages_id }
    }

    fn add_stream(&mut self, dict: Dictionary, body: Vec<u8>) -> lopdf::ObjectId {
        self.doc.add_object(Stream::new(dict, body))
    }

    fn form(&mut self, body: &str, resources: Option<Dictionary>) -> lopdf::ObjectId {
        let mut d = dictionary! {
            "Type" => "XObject",
            "Subtype" => "Form",
            "BBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
        };
        if let Some(r) = resources {
            d.set("Resources", r);
        }
        self.add_stream(d, body.as_bytes().to_vec())
    }

    /// Finish with one page and return the file bytes.
    fn finish(mut self, media: Object, contents: Object, resources: Option<Dictionary>) -> Vec<u8> {
        let mut page = dictionary! {
            "Type" => "Page",
            "Parent" => self.pages_id,
            "MediaBox" => media,
            "Contents" => contents,
        };
        if let Some(r) = resources {
            page.set("Resources", r);
        }
        let page_id = self.doc.add_object(page);
        self.doc.objects.insert(
            self.pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => vec![page_id.into()],
                "Count" => 1,
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

fn media() -> Object {
    Object::Array(vec![0.into(), 0.into(), 200.into(), 100.into()])
}

fn page_bytes(body: &str) -> Vec<u8> {
    let mut p = Pdf::new();
    let s = p.add_stream(Dictionary::new(), body.as_bytes().to_vec());
    p.finish(media(), s.into(), None)
}

fn flate(raw: &[u8]) -> Vec<u8> {
    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    e.write_all(raw).unwrap();
    e.finish().unwrap()
}

fn run(bytes: &[u8], limits: &PdfVectorLimits) -> Result<PdfVectorPage, PdfVectorError> {
    extract_page_paths_from_bytes(bytes, 1, limits)
}

/// Run and require an error, and that nothing panicked on the way.
fn refused(bytes: &[u8], limits: &PdfVectorLimits) -> PdfVectorError {
    let (bytes, limits) = (bytes.to_vec(), *limits);
    std::panic::catch_unwind(move || run(&bytes, &limits))
        .expect("the vector path panicked")
        .expect_err("expected a typed refusal")
}

// ---- the limits ----------------------------------------------------------------

#[test]
fn n5_8_1087_content_bytes_plain_stream() {
    let bytes = page_bytes(&"0 0 m 1 1 l S\n".repeat(100));
    let limits = PdfVectorLimits {
        max_content_bytes: 100,
        ..PdfVectorLimits::default()
    };
    match refused(&bytes, &limits) {
        PdfVectorError::ContentBytesExceeded { limit, observed } => {
            assert_eq!(limit, 100);
            assert!(observed > 100, "observed {observed}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn n5_8_1087_content_bytes_flate_bomb_is_cut_off_while_decoding() {
    // 64 MiB of whitespace deflates to a few tens of KiB.
    let mut p = Pdf::new();
    let packed = flate(&vec![b' '; 64 * MIB]);
    assert!(packed.len() < MIB);
    let s = p.add_stream(dictionary! { "Filter" => "FlateDecode" }, packed);
    let bytes = p.finish(media(), s.into(), None);
    let limits = PdfVectorLimits {
        max_content_bytes: MIB as u64,
        ..PdfVectorLimits::default()
    };
    let (err, peak) = peak_during(|| refused(&bytes, &limits));
    assert!(
        matches!(
            err,
            PdfVectorError::ContentBytesExceeded { limit, observed }
                if limit == MIB as u64 && observed > limit
        ),
        "{err:?}"
    );
    assert!(
        peak < 8 * MIB,
        "peak {peak} grew with the decompressed size"
    );
}

#[test]
fn n5_8_1087_content_bytes_count_across_form_expansion() {
    let mut p = Pdf::new();
    let body = format!("{}0 0 m 1 1 l S\n", " ".repeat(600));
    let f = p.form(&body, None);
    let s = p.add_stream(Dictionary::new(), b"/F Do /F Do /F Do".to_vec());
    let res = dictionary! { "XObject" => dictionary! { "F" => f } };
    let bytes = p.finish(media(), s.into(), Some(res));
    let limits = PdfVectorLimits {
        max_content_bytes: 1000,
        ..PdfVectorLimits::default()
    };
    assert!(matches!(
        refused(&bytes, &limits),
        PdfVectorError::ContentBytesExceeded { limit: 1000, .. }
    ));
}

#[test]
fn n5_8_1087_operator_count() {
    let bytes = page_bytes(&"0 0 m\n".repeat(100));
    let limits = PdfVectorLimits {
        max_operations: 10,
        ..PdfVectorLimits::default()
    };
    assert_eq!(
        refused(&bytes, &limits),
        PdfVectorError::OperationLimitExceeded {
            limit: 10,
            observed: 11
        }
    );
}

#[test]
fn n5_8_1087_path_segments_per_path() {
    let bytes = page_bytes(&format!("0 0 m {}S", "1 1 l ".repeat(500)));
    let limits = PdfVectorLimits {
        max_segments_per_path: 100,
        ..PdfVectorLimits::default()
    };
    match refused(&bytes, &limits) {
        PdfVectorError::PathSegmentsExceeded { limit, observed } => {
            assert_eq!(limit, 100);
            assert!(observed > 100 && observed < 110, "observed {observed}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn n5_8_1087_path_segments_reset_between_paths() {
    // 3 paths of 60 segments each are fine under a limit of 100.
    let one = format!("0 0 m {}S\n", "1 1 l ".repeat(59));
    let bytes = page_bytes(&one.repeat(3));
    let limits = PdfVectorLimits {
        max_segments_per_path: 100,
        ..PdfVectorLimits::default()
    };
    assert_eq!(run(&bytes, &limits).unwrap().entities.len(), 3);
}

#[test]
fn n5_8_1087_graphics_stack_depth() {
    let bytes = page_bytes(&format!("{}0 0 m 1 1 l S", "q ".repeat(300)));
    assert_eq!(
        refused(&bytes, &PdfVectorLimits::default()),
        PdfVectorError::GraphicsStackExceeded {
            limit: 256,
            observed: 257
        }
    );
}

#[test]
fn n5_8_1087_ten_mib_of_q_refuses_at_the_nesting_bound_with_flat_allocation() {
    let body = "q\n".repeat(5 * MIB);
    assert_eq!(body.len(), 10 * MIB);
    let bytes = page_bytes(&body);
    drop(body);
    let limits = PdfVectorLimits {
        max_content_bytes: 16 * MIB as u64,
        max_operations: u64::MAX / 2,
        ..PdfVectorLimits::default()
    };
    let (err, peak) = peak_during(|| refused(&bytes, &limits));
    assert_eq!(
        err,
        PdfVectorError::GraphicsStackExceeded {
            limit: 256,
            observed: 257
        }
    );
    // Parsing the file holds the stream once and reading it holds it again; what must
    // not happen is an operation list that grows with the stream (5M operations is
    // hundreds of MiB).
    assert!(
        peak < 6 * 10 * MIB,
        "peak {peak} bytes for a {} byte stream",
        bytes.len()
    );
}

#[test]
fn n5_8_1087_q_nesting_across_forms_counts_the_whole_expansion() {
    let mut p = Pdf::new();
    let f = p.form(&"q ".repeat(40), None);
    let s = p.add_stream(Dictionary::new(), b"/F Do /F Do".to_vec());
    let res = dictionary! { "XObject" => dictionary! { "F" => f } };
    let _ = (&f, &res);
    // Each form closes its own q's on exit, so two forms of 40 stay at 40.
    let bytes = p.finish(media(), s.into(), Some(res));
    let limits = PdfVectorLimits {
        max_graphics_stack: 50,
        ..PdfVectorLimits::default()
    };
    assert!(run(&bytes, &limits).is_ok());
    let tight = PdfVectorLimits {
        max_graphics_stack: 30,
        ..PdfVectorLimits::default()
    };
    assert_eq!(
        refused(&bytes, &tight),
        PdfVectorError::GraphicsStackExceeded {
            limit: 30,
            observed: 31
        }
    );
}

#[test]
fn n5_8_1087_form_that_draws_itself() {
    let mut p = Pdf::new();
    let f = p.form("/F Do", None);
    // Point the form's own resources back at itself.
    if let Some(Object::Stream(s)) = p.doc.objects.get_mut(&f) {
        s.dict.set(
            "Resources",
            dictionary! { "XObject" => dictionary! { "F" => f } },
        );
    }
    let s = p.add_stream(Dictionary::new(), b"/F Do".to_vec());
    let res = dictionary! { "XObject" => dictionary! { "F" => f } };
    let bytes = p.finish(media(), s.into(), Some(res));
    assert_eq!(
        refused(&bytes, &PdfVectorLimits::default()),
        PdfVectorError::FormDepthExceeded {
            limit: 16,
            observed: 17
        }
    );
}

#[test]
fn n5_8_1087_form_fan_out_is_bounded_by_total_expansion() {
    // Ten levels, each invoking the next ten times: 10^10 operations if it were let run.
    let mut p = Pdf::new();
    let mut inner = p.form("0 0 m 1 1 l S", None);
    for _ in 0..9 {
        let body = "/N Do ".repeat(10);
        let res = dictionary! { "XObject" => dictionary! { "N" => inner } };
        inner = p.form(&body, Some(res));
    }
    let s = p.add_stream(Dictionary::new(), b"/N Do".to_vec());
    let res = dictionary! { "XObject" => dictionary! { "N" => inner } };
    let bytes = p.finish(media(), s.into(), Some(res));
    let limits = PdfVectorLimits {
        max_operations: 10_000,
        ..PdfVectorLimits::default()
    };
    let (err, peak) = peak_during(|| refused(&bytes, &limits));
    assert!(
        matches!(
            err,
            PdfVectorError::OperationLimitExceeded {
                limit: 10_000,
                observed: 10_001
            }
        ),
        "{err:?}"
    );
    assert!(peak < 32 * MIB, "peak {peak}");
}

#[test]
fn n5_8_1087_resources_parent_cycle() {
    let mut p = Pdf::new();
    let s = p.add_stream(Dictionary::new(), b"0 0 m 1 1 l S".to_vec());
    let a = p.doc.new_object_id();
    let b = p.doc.new_object_id();
    // A page with no /Resources whose /Parent chain loops without ever reaching one.
    p.doc.objects.insert(
        a,
        Object::Dictionary(dictionary! { "Type" => "Pages", "Parent" => b }),
    );
    p.doc.objects.insert(
        b,
        Object::Dictionary(dictionary! { "Type" => "Pages", "Parent" => a }),
    );
    let page = p.doc.add_object(dictionary! {
        "Type" => "Page", "Parent" => a, "MediaBox" => media(), "Contents" => s,
    });
    let pages = p.pages_id;
    p.doc.objects.insert(
        pages,
        Object::Dictionary(dictionary! {
            "Type" => "Pages", "Kids" => vec![page.into()], "Count" => 1,
        }),
    );
    let cat = p
        .doc
        .add_object(dictionary! { "Type" => "Catalog", "Pages" => pages });
    p.doc.trailer.set("Root", cat);
    let mut bytes = Vec::new();
    p.doc.save_to(&mut bytes).unwrap();
    assert!(matches!(
        refused(&bytes, &PdfVectorLimits::default()),
        PdfVectorError::ResourceCycle { .. }
    ));
}

#[test]
fn n5_8_1087_resources_reference_chain_cycle() {
    let mut p = Pdf::new();
    let a = p.doc.new_object_id();
    let b = p.doc.new_object_id();
    p.doc.objects.insert(a, Object::Reference(b));
    p.doc.objects.insert(b, Object::Reference(a));
    let s = p.add_stream(Dictionary::new(), b"0 0 m 1 1 l S".to_vec());
    let mut page = Pdf::new();
    page.doc = p.doc;
    page.pages_id = p.pages_id;
    let bytes = page.finish(media(), s.into(), None);
    // Re-open to attach the looping /Resources to the page.
    let mut doc = Document::load_mem(&bytes).unwrap();
    let pid = *doc.get_pages().get(&1).unwrap();
    if let Some(Object::Dictionary(d)) = doc.objects.get_mut(&pid) {
        d.set("Resources", Object::Reference(a));
    }
    let mut out = Vec::new();
    doc.save_to(&mut out).unwrap();
    assert!(matches!(
        refused(&out, &PdfVectorLimits::default()),
        PdfVectorError::ResourceCycle { .. }
    ));
}

/// The vector path reads no fonts yet (text is N5.5's), so there is no font or CMap
/// limit to trip. This pins that: a 24 MiB embedded font is never touched, which is
/// why the issue's last table row is carried by a follow-up and not faked here.
#[test]
fn n5_8_1087_embedded_font_bytes_are_never_read() {
    let mut p = Pdf::new();
    let font_file = p.add_stream(Dictionary::new(), vec![0u8; 24 * MIB]);
    let desc = p.doc.add_object(dictionary! {
        "Type" => "FontDescriptor", "FontName" => "Big", "FontFile2" => font_file,
    });
    let font = p.doc.add_object(dictionary! {
        "Type" => "Font", "Subtype" => "TrueType", "BaseFont" => "Big",
        "FontDescriptor" => desc,
    });
    let s = p.add_stream(
        Dictionary::new(),
        b"BT /F1 12 Tf (hi) Tj ET 0 0 m 5 5 l S".to_vec(),
    );
    let res = dictionary! { "Font" => dictionary! { "F1" => font } };
    let bytes = p.finish(media(), s.into(), Some(res));
    let (page, peak) = peak_during(|| run(&bytes, &PdfVectorLimits::default()).unwrap());
    assert_eq!(page.entities.len(), 1);
    // The file itself is loaded once; nothing decodes the font on top of that.
    assert!(peak < 3 * bytes.len(), "peak {peak}, file {}", bytes.len());
}

// ---- malformed input: a typed error, never a panic -----------------------------

fn content_error(bytes: &[u8]) {
    let e = refused(bytes, &PdfVectorLimits::default());
    assert!(
        matches!(e, PdfVectorError::ContentStream { page: 1, .. }),
        "{e:?}"
    );
}

#[test]
fn n5_8_1087_truncated_flate_stream() {
    let packed = flate(b"0 0 m 100 100 l 200 0 l S ".repeat(200).as_slice());
    let cut = packed[..packed.len() / 2].to_vec();
    let mut p = Pdf::new();
    let s = p.add_stream(dictionary! { "Filter" => "FlateDecode" }, cut);
    content_error(&p.finish(media(), s.into(), None));
}

#[test]
fn n5_8_1087_corrupt_flate_stream() {
    let mut p = Pdf::new();
    let s = p.add_stream(dictionary! { "Filter" => "FlateDecode" }, vec![0xde; 64]);
    content_error(&p.finish(media(), s.into(), None));
}

#[test]
fn n5_8_1087_unknown_filter_in_the_chain() {
    let mut p = Pdf::new();
    let s = p.add_stream(
        dictionary! { "Filter" => vec![Object::Name(b"Bogus".to_vec()), Object::Name(b"FlateDecode".to_vec())] },
        flate(b"0 0 m 1 1 l S"),
    );
    content_error(&p.finish(media(), s.into(), None));
}

#[test]
fn n5_8_1087_contents_points_at_a_non_stream() {
    let mut p = Pdf::new();
    let d = p.doc.add_object(dictionary! { "Not" => "a stream" });
    content_error(&p.finish(media(), d.into(), None));
}

#[test]
fn n5_8_1087_unbalanced_q_and_restore_do_not_panic() {
    // Extra Q is counted (documented N5.1 behaviour), open q is closed at the end.
    let page = run(
        &page_bytes("Q Q q q 0 0 m 1 1 l S"),
        &PdfVectorLimits::default(),
    )
    .unwrap();
    assert_eq!(page.unbalanced_restores, 2);
    assert_eq!(page.entities.len(), 1);
}

#[test]
fn n5_8_1087_operand_arity() {
    for body in [
        "1 m",
        "1 2 3 m",
        "0 0 m 1 l",
        "0 0 m 1 2 3 4 5 c",
        "0 0 m 1 2 3 v",
        "1 2 3 re",
        "1 2 3 4 5 6 7 cm",
        "1 h",
        "0 0 m 1 1 l 5 S",
        "/Name Do Do",
    ] {
        content_error(&page_bytes(body));
    }
}

#[test]
fn n5_8_1087_non_numeric_operand() {
    content_error(&page_bytes("/a /b m"));
    content_error(&page_bytes("(x) 0 m"));
}

#[test]
fn n5_8_1087_out_of_range_numbers() {
    let huge = "9".repeat(60);
    // An integer past i64 is refused outright, so it cannot read as a shorter page.
    content_error(&page_bytes(&format!("{huge} 0 m 1 1 l S")));
    let f32_max = "340282350000000000000000000000000000000.0";
    // Neither the literal nor what a transform does to it may reach the output as a
    // non-finite coordinate, and neither may panic.
    for body in [
        format!("{huge} 0 m 1 1 l S"),
        format!("{f32_max} 0 0 {f32_max} 0 0 cm {f32_max} 0 m 1 1 l S"),
        format!("{f32_max} 0 0 {f32_max} 0 0 cm {f32_max} 0 0 {f32_max} 0 0 cm 5 5 m 6 6 l S"),
    ] {
        let bytes = page_bytes(&body);
        let outcome = std::panic::catch_unwind(|| run(&bytes, &PdfVectorLimits::default()))
            .expect("panicked");
        match outcome {
            Err(_) => {}
            Ok(page) => {
                for e in &page.entities {
                    for s in &e.segments {
                        format!("{s:?}")
                            .split(|c: char| !(c.is_alphanumeric() || c == '.' || c == '-'))
                            .for_each(|t| assert!(!t.contains("inf") && !t.contains("NaN"), "{t}"));
                    }
                }
            }
        }
    }
    // Nine f32::MAX scales compose past f64::MAX: that has to be an error, not a
    // quiet infinity in the output.
    let scale = format!("{f32_max} 0 0 {f32_max} 0 0 cm ");
    content_error(&page_bytes(&format!("{}5 5 m 6 6 l S", scale.repeat(9))));
}

fn page_with_media(m: Object) -> Vec<u8> {
    let mut p = Pdf::new();
    let s = p.add_stream(Dictionary::new(), b"0 0 m 1 1 l S".to_vec());
    p.finish(m, s.into(), None)
}

#[test]
fn n5_8_1087_media_box_with_no_extent() {
    for m in [
        vec![0, 0, 0, 0],
        vec![0, 0, 0, 100],
        vec![0, 0, 100, 0],
        vec![50, 50, 50, 50],
    ] {
        let bytes = page_with_media(Object::Array(m.into_iter().map(Object::Integer).collect()));
        let e = refused(&bytes, &PdfVectorLimits::default());
        assert!(matches!(e, PdfVectorError::InvalidPageBox { .. }), "{e:?}");
    }
}

#[test]
fn n5_8_1087_media_box_with_non_finite_or_junk_entries() {
    let junk = Object::Array(vec![0.into(), 0.into(), Object::Real(f32::NAN), 100.into()]);
    let inf = Object::Array(vec![
        0.into(),
        0.into(),
        Object::Real(f32::INFINITY),
        100.into(),
    ]);
    let names = Object::Array(vec![
        Object::Name(b"a".to_vec()),
        0.into(),
        1.into(),
        2.into(),
    ]);
    for m in [junk, inf, names] {
        let bytes = page_with_media(m);
        let outcome = std::panic::catch_unwind(|| run(&bytes, &PdfVectorLimits::default()))
            .expect("panicked");
        if let Ok(page) = outcome {
            // Falling back to a default page is allowed, a non-finite size is not.
            assert!(page.width.is_finite() && page.width > 0.0);
            assert!(page.height.is_finite() && page.height > 0.0);
        }
    }
}
