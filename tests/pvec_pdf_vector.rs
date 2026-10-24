//! The PDF vector-path adapter (N5.1, #1080), over fixtures built in memory with
//! `lopdf` so the content streams under test are visible in the test itself.
//!
//! Covers: path construction (`m l c v y h re`), `v`/`y` expansion compared bit for
//! bit against explicit `c`, graphics state, painting and clipping intent, Form
//! XObjects (inherited resources, `/Matrix`, depth and expansion limits), page
//! geometry (box origin, `/Rotate`), tokenizer failures never read as an empty page,
//! determinism, the no-`pdfium` configuration and the no-`lopdf`-in-signatures rule.

use libviprs::pvec::pdf_vector::{
    PdfFillRule, PdfPathSegment as Seg, PdfVectorError, PdfVectorLimits, PdfVectorPage,
    extract_page_paths_from_bytes,
};
use lopdf::content::{Content, Operation};
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

    fn page(&mut self, body: &str) {
        let s = self.stream(body);
        self.page_with([0, 0, 200, 100], s.into(), None, Dictionary::new());
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

fn one_page(body: &str) -> Vec<u8> {
    let mut f = Fixture::new();
    f.page(body);
    f.bytes()
}

fn run(bytes: &[u8]) -> Result<PdfVectorPage, PdfVectorError> {
    extract_page_paths_from_bytes(bytes, 1, &PdfVectorLimits::default())
}

fn ok(body: &str) -> PdfVectorPage {
    run(&one_page(body)).unwrap()
}

fn bits(p: [f64; 2]) -> [u64; 2] {
    [p[0].to_bits(), p[1].to_bits()]
}

#[test]
fn n5_1_vector_pdf_yields_paths() {
    let page = ok("10 10 m 50 10 l 50 40 l h S");
    assert_eq!(page.entities.len(), 1);
    assert_eq!(
        page.entities[0].segments,
        vec![
            Seg::MoveTo([10.0, 10.0]),
            Seg::LineTo([50.0, 10.0]),
            Seg::LineTo([50.0, 40.0]),
            Seg::Close
        ]
    );
    assert!(page.entities[0].paint.stroke);
}

/// The vector path adds no dependency and needs no pdfium: this build has the
/// feature off and the same call answers.
#[cfg(not(feature = "pdfium"))]
#[test]
fn n5_1_paths_without_the_pdfium_feature() {
    let page = ok("0 0 m 10 10 l S");
    assert_eq!(page.entities.len(), 1);
}

#[test]
fn n5_1_rect_expands_to_four_corners_and_close() {
    let page = ok("10 20 30 40 re f");
    assert_eq!(
        page.entities[0].segments,
        vec![
            Seg::MoveTo([10.0, 20.0]),
            Seg::LineTo([40.0, 20.0]),
            Seg::LineTo([40.0, 60.0]),
            Seg::LineTo([10.0, 60.0]),
            Seg::Close
        ]
    );
}

#[test]
fn n5_1_v_and_y_match_explicit_c_bit_for_bit() {
    // The same two curves in both spellings: `v` repeats the current point as the
    // first control point, `y` repeats the end point as the second.
    let spelled = ok("1.25 2.5 m 7.5 8.25 20.125 4.75 v 20.125 4.75 m 3.5 9.75 30.5 11.25 y S");
    let explicit = ok(
        "1.25 2.5 m 1.25 2.5 7.5 8.25 20.125 4.75 c 20.125 4.75 m 3.5 9.75 30.5 11.25 30.5 11.25 c S",
    );
    let a = &spelled.entities[0].segments;
    let b = &explicit.entities[0].segments;
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        match (x, y) {
            (
                Seg::CurveTo { c1, c2, end },
                Seg::CurveTo {
                    c1: d1,
                    c2: d2,
                    end: de,
                },
            ) => {
                assert_eq!(bits(*c1), bits(*d1));
                assert_eq!(bits(*c2), bits(*d2));
                assert_eq!(bits(*end), bits(*de));
            }
            (p, q) => assert_eq!(p, q),
        }
    }
    assert!(matches!(a[1], Seg::CurveTo { .. }));
}

#[test]
fn n5_1_v_and_y_use_the_spec_control_points() {
    let page = ok("0 0 m 1 2 3 4 v 10 10 m 5 6 7 8 y S");
    let s = &page.entities[0].segments;
    assert_eq!(
        s[1],
        Seg::CurveTo {
            c1: [0.0, 0.0],
            c2: [1.0, 2.0],
            end: [3.0, 4.0]
        }
    );
    assert_eq!(
        s[3],
        Seg::CurveTo {
            c1: [5.0, 6.0],
            c2: [7.0, 8.0],
            end: [7.0, 8.0]
        }
    );
}

#[test]
fn n5_1_cm_q_and_restore() {
    let page = ok("q 2 0 0 2 5 5 cm 1 1 m 2 2 l S Q 1 1 m 2 2 l S");
    assert_eq!(
        page.entities[0].segments,
        vec![Seg::MoveTo([7.0, 7.0]), Seg::LineTo([9.0, 9.0])]
    );
    assert_eq!(
        page.entities[1].segments,
        vec![Seg::MoveTo([1.0, 1.0]), Seg::LineTo([2.0, 2.0])]
    );
}

#[test]
fn n5_1_cm_concatenates_in_pdf_order() {
    // The later cm is applied first to a point: translate then scale.
    let page = ok("2 0 0 2 0 0 cm 1 0 0 1 10 0 cm 0 0 m 1 0 l S");
    assert_eq!(
        page.entities[0].segments,
        vec![Seg::MoveTo([20.0, 0.0]), Seg::LineTo([22.0, 0.0])]
    );
}

#[test]
fn n5_1_unbalanced_restore_is_counted_not_fatal() {
    let page = ok("Q 0 0 m 1 1 l S");
    assert_eq!(page.unbalanced_restores, 1);
    assert_eq!(page.entities.len(), 1);
}

#[test]
fn n5_1_graphics_stack_is_bounded() {
    let body = "q ".repeat(10);
    let bytes = one_page(&format!("{body} 0 0 m 1 1 l S"));
    let limits = PdfVectorLimits {
        max_graphics_stack: 4,
        ..PdfVectorLimits::default()
    };
    let err = extract_page_paths_from_bytes(&bytes, 1, &limits).unwrap_err();
    assert!(
        matches!(
            err,
            PdfVectorError::GraphicsStackExceeded {
                limit: 4,
                observed: 5
            }
        ),
        "{err:?}"
    );
}

#[test]
fn n5_1_paint_operators_are_recorded_as_intent() {
    let cases: [(&str, Option<PdfFillRule>, bool, bool); 9] = [
        ("S", None, true, false),
        ("s", None, true, true),
        ("f", Some(PdfFillRule::NonZero), false, false),
        ("F", Some(PdfFillRule::NonZero), false, false),
        ("f*", Some(PdfFillRule::EvenOdd), false, false),
        ("B", Some(PdfFillRule::NonZero), true, false),
        ("B*", Some(PdfFillRule::EvenOdd), true, false),
        ("b", Some(PdfFillRule::NonZero), true, true),
        ("b*", Some(PdfFillRule::EvenOdd), true, true),
    ];
    for (op, fill, stroke, close) in cases {
        let page = ok(&format!("0 0 m 5 0 l 5 5 l {op}"));
        let e = &page.entities[0];
        assert_eq!(e.paint.fill, fill, "{op}");
        assert_eq!(e.paint.stroke, stroke, "{op}");
        assert_eq!(e.paint.close, close, "{op}");
        assert_eq!(e.provenance.operator, op);
        assert_eq!(e.clip, None);
    }
}

#[test]
fn n5_1_clip_is_recorded_and_n_paints_nothing() {
    let page = ok("0 0 10 10 re W n 0 0 10 10 re W* n 1 1 m 2 2 l S");
    assert_eq!(page.entities.len(), 3);
    assert_eq!(page.entities[0].clip, Some(PdfFillRule::NonZero));
    assert_eq!(page.entities[1].clip, Some(PdfFillRule::EvenOdd));
    for e in &page.entities[..2] {
        assert_eq!(e.paint.fill, None);
        assert!(!e.paint.stroke);
    }
    assert_eq!(page.entities[2].clip, None);
}

#[test]
fn n5_1_contents_array_is_one_logical_stream() {
    let mut f = Fixture::new();
    let a = f.stream("0 0 m 10 0 l");
    let b = f.stream("10 10 l S");
    f.page_with(
        [0, 0, 200, 100],
        vec![Object::from(a), Object::from(b)].into(),
        None,
        Dictionary::new(),
    );
    let page = run(&f.bytes()).unwrap();
    assert_eq!(page.entities.len(), 1);
    assert_eq!(page.entities[0].segments.len(), 3);
}

#[test]
fn n5_1_page_origin_is_removed() {
    let mut f = Fixture::new();
    let s = f.stream("100 50 m 110 60 l S");
    f.page_with([100, 50, 300, 250], s.into(), None, Dictionary::new());
    let page = run(&f.bytes()).unwrap();
    assert_eq!(
        page.entities[0].segments,
        vec![Seg::MoveTo([0.0, 0.0]), Seg::LineTo([10.0, 10.0])]
    );
    assert_eq!((page.width, page.height), (200.0, 200.0));
}

#[test]
fn n5_1_cropbox_origin_wins_over_mediabox() {
    let mut f = Fixture::new();
    let s = f.stream("30 30 m 40 40 l S");
    f.page_with(
        [0, 0, 200, 100],
        s.into(),
        None,
        dictionary! { "CropBox" => vec![20.into(), 10.into(), 120.into(), 90.into()] },
    );
    let page = run(&f.bytes()).unwrap();
    assert_eq!(
        page.entities[0].segments,
        vec![Seg::MoveTo([10.0, 20.0]), Seg::LineTo([20.0, 30.0])]
    );
    assert_eq!((page.width, page.height), (100.0, 80.0));
}

#[test]
fn n5_1_rotate_is_composed_explicitly() {
    // A 200x100 page; the point (10, 20) is 10 from the left and 20 from the bottom.
    let want = [
        (0, [10.0, 20.0], (200.0, 100.0)),
        (90, [20.0, 190.0], (100.0, 200.0)),
        (180, [190.0, 80.0], (200.0, 100.0)),
        (270, [80.0, 10.0], (100.0, 200.0)),
        (-90, [80.0, 10.0], (100.0, 200.0)),
    ];
    for (rot, pt, dims) in want {
        let mut f = Fixture::new();
        let s = f.stream("10 20 m 11 20 l S");
        f.page_with(
            [0, 0, 200, 100],
            s.into(),
            None,
            dictionary! { "Rotate" => rot },
        );
        let page = run(&f.bytes()).unwrap();
        let Seg::MoveTo(p) = page.entities[0].segments[0] else {
            panic!()
        };
        assert_eq!(p, pt, "rotate {rot}");
        assert_eq!((page.width, page.height), dims, "rotate {rot}");
        assert_eq!(page.rotation_degrees, i64::from(rot).rem_euclid(360));
    }
}

#[test]
fn n5_1_rotate_not_a_multiple_of_90_is_typed() {
    let mut f = Fixture::new();
    let s = f.stream("0 0 m 1 1 l S");
    f.page_with(
        [0, 0, 200, 100],
        s.into(),
        None,
        dictionary! { "Rotate" => 45 },
    );
    assert_eq!(
        run(&f.bytes()).unwrap_err(),
        PdfVectorError::UnsupportedRotation(45)
    );
}

fn form_page(page_body: &str, form_body: &str, form_extra: Dictionary) -> Vec<u8> {
    let mut f = Fixture::new();
    let form = f.form(form_body, form_extra);
    let s = f.stream(page_body);
    f.page_with(
        [0, 0, 200, 100],
        s.into(),
        Some(dictionary! { "XObject" => dictionary! { "Fm1" => form } }),
        Dictionary::new(),
    );
    f.bytes()
}

#[test]
fn n5_1_form_matrix_composes_into_the_ctm() {
    let bytes = form_page(
        "q 2 0 0 2 0 0 cm /Fm1 Do Q",
        "1 1 m 2 1 l S",
        dictionary! { "Matrix" => vec![1.into(), 0.into(), 0.into(), 1.into(), 10.into(), 0.into()] },
    );
    let page = run(&bytes).unwrap();
    // form matrix first (translate 10), then the invoking scale of 2.
    assert_eq!(
        page.entities[0].segments,
        vec![Seg::MoveTo([22.0, 2.0]), Seg::LineTo([24.0, 2.0])]
    );
    assert_eq!(page.entities[0].provenance.form_path, vec!["Fm1"]);
}

#[test]
fn n5_1_form_state_does_not_leak_out() {
    let bytes = form_page(
        "/Fm1 Do 0 0 m 1 1 l S",
        "5 0 0 5 0 0 cm q q",
        Dictionary::new(),
    );
    let page = run(&bytes).unwrap();
    assert_eq!(
        page.entities[0].segments,
        vec![Seg::MoveTo([0.0, 0.0]), Seg::LineTo([1.0, 1.0])]
    );
}

#[test]
fn n5_1_form_inherits_the_invokers_resources() {
    // The form has no /Resources of its own and invokes Fm2, which only the page
    // names, so the lookup has to fall back to the invoking resources.
    let mut f = Fixture::new();
    let inner = f.form("0 0 m 3 3 l S", Dictionary::new());
    let outer = f.form("/Fm2 Do", Dictionary::new());
    let s = f.stream("/Fm1 Do");
    f.page_with(
        [0, 0, 200, 100],
        s.into(),
        Some(dictionary! { "XObject" => dictionary! { "Fm1" => outer, "Fm2" => inner } }),
        Dictionary::new(),
    );
    let page = run(&f.bytes()).unwrap();
    assert_eq!(page.entities.len(), 1);
    assert_eq!(page.entities[0].provenance.form_path, vec!["Fm1", "Fm2"]);
}

#[test]
fn n5_1_form_own_resources_win() {
    let mut f = Fixture::new();
    let page_fm = f.form("0 0 m 1 1 l S", Dictionary::new());
    let own = f.form("0 0 m 9 9 l S", Dictionary::new());
    let outer = f.form(
        "/Fm2 Do",
        dictionary! { "Resources" => dictionary! { "XObject" => dictionary! { "Fm2" => own } } },
    );
    let s = f.stream("/Fm1 Do");
    f.page_with(
        [0, 0, 200, 100],
        s.into(),
        Some(dictionary! { "XObject" => dictionary! { "Fm1" => outer, "Fm2" => page_fm } }),
        Dictionary::new(),
    );
    let page = run(&f.bytes()).unwrap();
    assert_eq!(page.entities[0].segments[1], Seg::LineTo([9.0, 9.0]));
}

#[test]
fn n5_1_self_referencing_form_hits_the_depth_limit() {
    let mut f = Fixture::new();
    let id = f.doc.new_object_id();
    let dict = dictionary! {
        "Type" => "XObject", "Subtype" => "Form",
        "BBox" => vec![0.into(), 0.into(), 10.into(), 10.into()],
        "Resources" => dictionary! { "XObject" => dictionary! { "Fm1" => id } },
    };
    f.doc
        .objects
        .insert(id, Stream::new(dict, b"/Fm1 Do".to_vec()).into());
    let s = f.stream("/Fm1 Do");
    f.page_with(
        [0, 0, 200, 100],
        s.into(),
        Some(dictionary! { "XObject" => dictionary! { "Fm1" => id } }),
        Dictionary::new(),
    );
    let limits = PdfVectorLimits {
        max_form_depth: 3,
        ..PdfVectorLimits::default()
    };
    let err = extract_page_paths_from_bytes(&f.bytes(), 1, &limits).unwrap_err();
    assert_eq!(
        err,
        PdfVectorError::FormDepthExceeded {
            limit: 3,
            observed: 4
        }
    );
    assert!(err.to_string().contains("max_form_depth=3"));
}

#[test]
fn n5_1_total_expansion_is_bounded() {
    // One form invoked many times: depth stays 1, the operation count does not.
    let bytes = form_page(&"/Fm1 Do ".repeat(200), "0 0 m 1 1 l S", Dictionary::new());
    let limits = PdfVectorLimits {
        max_operations: 100,
        ..PdfVectorLimits::default()
    };
    let err = extract_page_paths_from_bytes(&bytes, 1, &limits).unwrap_err();
    assert!(
        matches!(
            err,
            PdfVectorError::OperationLimitExceeded { limit: 100, .. }
        ),
        "{err:?}"
    );
}

#[test]
fn n5_1_entity_limit_is_typed() {
    let bytes = one_page(&"0 0 m 1 1 l S ".repeat(10));
    let limits = PdfVectorLimits {
        max_entities: 5,
        ..PdfVectorLimits::default()
    };
    assert_eq!(
        extract_page_paths_from_bytes(&bytes, 1, &limits).unwrap_err(),
        PdfVectorError::EntityLimitExceeded { limit: 5 }
    );
}

#[test]
fn n5_1_untokenizable_stream_is_an_error_never_an_empty_page() {
    for body in [
        "0 0 m 1 1 l S (unterminated",
        "0 0 m 1 1 l S [1 2",
        "0 0 m 1 1 l S <<",
        "0 0 m 1 1 l S <ab",
        "0 0 m 1 1 l ] S",
    ] {
        let r = run(&one_page(body));
        assert!(
            matches!(r, Err(PdfVectorError::ContentStream { page: 1, .. })),
            "{body}: {r:?}"
        );
    }
}

#[test]
fn n5_1_balanced_strings_and_comments_still_tokenize() {
    let page = ok("% a ( comment\n/Span <</MCID 0>> BDC (a \\) (nested) b) Tj EMC 0 0 m 1 1 l S");
    assert_eq!(page.entities.len(), 1);
}

#[test]
fn n5_1_malformed_operators_are_errors_too() {
    for body in [
        "0 0 m 1 l S",             // l with one operand
        "1 1 l S",                 // l with no current point
        "1 2 3 4 v S",             // v with no current point
        "0 0 m a b l S",           // non-numeric operand
        "0 0 m 1 1 1 1 1 1 1 c S", // too many operands
    ] {
        let r = run(&one_page(body));
        assert!(
            matches!(r, Err(PdfVectorError::ContentStream { .. })),
            "{body}: {r:?}"
        );
    }
}

#[test]
fn n5_1_a_page_with_no_paths_is_an_empty_ok() {
    let page = ok("BT ET");
    assert!(page.entities.is_empty());
}

#[test]
fn n5_1_page_out_of_range_and_garbage_are_typed() {
    let bytes = one_page("0 0 m 1 1 l S");
    assert!(matches!(
        extract_page_paths_from_bytes(&bytes, 2, &PdfVectorLimits::default()),
        Err(PdfVectorError::PageOutOfRange { page: 2, total: 1 })
    ));
    assert!(matches!(
        extract_page_paths_from_bytes(b"not a pdf", 1, &PdfVectorLimits::default()),
        Err(PdfVectorError::Read(_))
    ));
}

#[test]
fn n5_1_two_runs_give_identical_sequences() {
    let bytes = form_page(
        "0 0 m 5 5 l S q 3 0 0 3 1 1 cm /Fm1 Do Q 9 9 m 1 2 3 4 5 6 c f*",
        "0 0 5 5 re B 1 1 m 2 2 3 3 v S",
        Dictionary::new(),
    );
    let a = run(&bytes).unwrap();
    let b = run(&bytes).unwrap();
    assert_eq!(a, b);
    let seq: Vec<u64> = a.entities.iter().map(|e| e.provenance.sequence).collect();
    assert_eq!(seq, (0..a.entities.len() as u64).collect::<Vec<_>>());
}

#[test]
fn n5_1_content_encoding_round_trips_through_lopdf_content() {
    // Guard on the fixture builder itself: the Content encoder and our raw text
    // agree on what `v` means, so a fixture written either way reads the same.
    let c = Content {
        operations: vec![
            Operation::new("m", vec![0.into(), 0.into()]),
            Operation::new("v", vec![1.into(), 2.into(), 3.into(), 4.into()]),
            Operation::new("S", vec![]),
        ],
    };
    let mut f = Fixture::new();
    let s = f.stream(&String::from_utf8(c.encode().unwrap()).unwrap());
    f.page_with([0, 0, 200, 100], s.into(), None, Dictionary::new());
    let page = run(&f.bytes()).unwrap();
    assert_eq!(page.entities[0].segments.len(), 2);
}

// ---- the boundary: no lopdf type in any public signature -------------------------

fn strip_comments(src: &str) -> String {
    src.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every `pub` item header (not `pub(crate)`), to the first `{`, `;` or top-level `,`.
fn public_items(src: &str) -> Vec<String> {
    let text = strip_comments(src)
        .replace("->", " to ")
        .replace("=>", " to ");
    let chars: Vec<char> = text.chars().collect();
    let starts = |i: usize, kw: &str| {
        let k: Vec<char> = kw.chars().collect();
        i + k.len() < chars.len()
            && chars[i..i + k.len()] == k[..]
            && (i == 0 || !(chars[i - 1].is_alphanumeric() || chars[i - 1] == '_'))
            && (chars[i + k.len()] == ' ' || chars[i + k.len()] == '(')
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        // Public items only: `pub(crate)` and private helpers may name `lopdf`.
        if !starts(i, "pub") || chars[i + 3..].starts_with(&['(']) {
            i += 1;
            continue;
        }
        let mut depth = 0i32;
        let mut j = i;
        while j < chars.len() {
            match chars[j] {
                '<' | '(' | '[' => depth += 1,
                '>' | ')' | ']' => depth -= 1,
                '{' | ';' => break,
                ',' if depth <= 0 => break,
                _ => {}
            }
            j += 1;
        }
        let item: String = chars[i..j.min(chars.len())].iter().collect();
        out.push(item.split_whitespace().collect::<Vec<_>>().join(" "));
        i = j.max(i + 1);
    }
    out
}

const FOREIGN: [&str; 8] = [
    "lopdf",
    "Object",
    "ObjectId",
    "Dictionary",
    "Stream",
    "Document",
    "Operation",
    "Content",
];

fn foreign_words(item: &str) -> Vec<&'static str> {
    let words: Vec<&str> = item
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .collect();
    FOREIGN
        .iter()
        .copied()
        .filter(|f| words.contains(f))
        .collect()
}

#[test]
fn n5_1_scanner_can_fail() {
    let items = public_items("pub fn leak(d: &lopdf::Dictionary) -> lopdf::Object { x }");
    assert!(
        items.iter().any(|i| !foreign_words(i).is_empty()),
        "{items:?}"
    );
}

#[test]
fn n5_1_no_lopdf_type_in_any_public_signature() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(root.join("src/pvec/pdf_vector.rs")).unwrap();
    let items = public_items(&src);
    assert!(items.len() > 20, "scanner found {} items", items.len());
    for item in &items {
        assert!(foreign_words(item).is_empty(), "foreign type in `{item}`");
    }
    // The re-exports reach normalization through `pvec`, and say only our names.
    let m = std::fs::read_to_string(root.join("src/pvec/mod.rs")).unwrap();
    for item in public_items(&m) {
        assert!(foreign_words(&item).is_empty(), "foreign type in `{item}`");
    }
}
