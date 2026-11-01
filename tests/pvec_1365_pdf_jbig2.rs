//! Issue #1365 (RL2.1 follow-up 4/6): `JBIG2Decode` through `hayro-jbig2`, behind the non-default
//! `jbig2` feature.
//!
//! The streams are in `tests/fixtures/jbig2/` (provenance and licence in `PROVENANCE.md`). Each is
//! a whole JBIG2 file; `embedded` turns it into what a PDF stores (Annex D.3: no file header,
//! the page 0 segments apart as `/JBIG2Globals`). Every case is baked and read back through
//! PVec's reader, and the pixels must equal the reference picture: tolerance 0, black is 0.
//! Needs `--features pvec-raster`; the decoding cases also need `jbig2`.

#![cfg(feature = "pvec-raster")]
#![allow(dead_code)]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::pdf_image::{PdfImageBake, PdfImageNote, PdfImageOptions, bake_page_images};
use libviprs::raster_pvec::container::write_rasters;
use libviprs::raster_pvec::pyramid::RasterPyramid;
use lopdf::{Dictionary, Object, Stream, dictionary};
use pvec_facade::unstable::RasterId;
use pvec_facade::{Bounds2, Curve, EntityId, Manifest, PvecReader, PvecWriter};
use pvec_geometry::{Point2, Polyline};
use std::path::PathBuf;

fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/jbig2")
        .join(name)
}

/// The stream's segments as `(page association, bytes of header and data)`, file header dropped.
fn segments(file: &[u8]) -> Vec<(u32, Vec<u8>)> {
    assert_eq!(&file[..8], b"\x97JB2\r\n\x1a\n", "a JBIG2 file header");
    let flags = file[8];
    assert_eq!(flags & 1, 1, "sequential organisation");
    let mut at = if flags & 2 == 0 { 13 } else { 9 };
    let mut out = Vec::new();
    while at < file.len() {
        let start = at;
        let number = u32::from_be_bytes(file[at..at + 4].try_into().unwrap());
        let seg_flags = file[at + 4];
        at += 5;
        let count = if file[at] >> 5 == 7 {
            let n = u32::from_be_bytes(file[at..at + 4].try_into().unwrap()) & 0x1FFF_FFFF;
            at += 4 + (n as usize + 1).div_ceil(8);
            n as usize
        } else {
            let n = (file[at] >> 5) as usize;
            at += 1;
            n
        };
        at += count
            * if number <= 256 {
                1
            } else if number <= 65536 {
                2
            } else {
                4
            };
        let page = if seg_flags & 0x40 != 0 {
            at += 4;
            u32::from_be_bytes(file[at - 4..at].try_into().unwrap())
        } else {
            at += 1;
            u32::from(file[at - 1])
        };
        let len = u32::from_be_bytes(file[at..at + 4].try_into().unwrap()) as usize;
        assert_ne!(
            len, 0xFFFF_FFFF,
            "unknown length is not used by these streams"
        );
        at += 4 + len;
        // End of file (51) has no place in an embedded stream.
        if seg_flags & 0x3F != 51 {
            out.push((page, file[start..at].to_vec()));
        }
    }
    out
}

/// `(page stream, globals)` as a PDF holds them: page 0 segments are the globals.
fn embedded(name: &str) -> (Vec<u8>, Option<Vec<u8>>) {
    let (mut page, mut globals) = (Vec::new(), Vec::new());
    for (assoc, bytes) in segments(&std::fs::read(path(name)).unwrap()) {
        if assoc == 0 {
            globals.extend(bytes);
        } else {
            page.extend(bytes);
        }
    }
    (page, (!globals.is_empty()).then_some(globals))
}

/// All segments in the page stream, no globals object, even those that were page 0.
fn embedded_inline(name: &str) -> Vec<u8> {
    let (mut page, globals) = embedded(name);
    if let Some(mut g) = globals {
        g.append(&mut page);
        page = g;
    }
    page
}

/// Rows of the reference picture as 8-bit gray, black 0, and its size.
fn reference(name: &str) -> (u32, u32, Vec<u8>) {
    let img = image::open(path(name)).unwrap().to_luma8();
    let px = img
        .pixels()
        .map(|p| if p.0[0] < 128 { 0 } else { 255 })
        .collect();
    (img.width(), img.height(), px)
}

fn jbig2(w: i64, h: i64, data: Vec<u8>, extra: Dictionary) -> Stream {
    let mut d = dictionary! {
        "ColorSpace" => "DeviceGray", "BitsPerComponent" => 1, "Filter" => "JBIG2Decode",
    };
    for (k, v) in extra.iter() {
        d.set(k.clone(), v.clone());
    }
    image(w, h, d, data)
}

/// A PDF with one JBIG2 image painted at one pixel per point; `globals` become `/JBIG2Globals`.
fn pdf_with(w: i64, h: i64, page: Vec<u8>, globals: Option<Vec<u8>>, extra: Dictionary) -> Vec<u8> {
    let mut d = doc();
    let mut extra = extra;
    if let Some(g) = globals {
        let id = d.add_object(Stream::new(Dictionary::new(), g));
        extra.set(
            "DecodeParms",
            dictionary! { "JBIG2Globals" => Object::Reference(id) },
        );
    }
    pdf(
        d,
        Resources::images(vec![("Im1", Object::Stream(jbig2(w, h, page, extra)))]),
        format!("q {w} 0 0 {h} 0 0 cm /Im1 Do Q").as_bytes(),
    )
}

fn bake(bytes: &[u8]) -> PdfImageBake {
    bake_page_images(&page(bytes), &PdfImageOptions::default())
}

fn through_pvec(b: &PdfImageBake) -> PvecReader {
    let mut w = PvecWriter::new(Manifest {
        metres_per_unit: 1.0,
        bounds: Bounds2 {
            min: [0.0, 0.0],
            max: [612.0, 792.0],
        },
        capabilities: Vec::new(),
        producer_id: "libviprs-1365".into(),
        producer_version: "1".into(),
    })
    .unwrap();
    // An artifact needs one entity (WRT-01): the page outline.
    let ring = Polyline::new(vec![
        Point2 { x: 0.0, y: 0.0 },
        Point2 { x: 612.0, y: 0.0 },
        Point2 { x: 612.0, y: 792.0 },
        Point2 { x: 0.0, y: 792.0 },
        Point2 { x: 0.0, y: 0.0 },
    ])
    .unwrap();
    w.push(EntityId::new(0).unwrap(), &Curve::Polyline(ring))
        .unwrap();
    write_rasters(&mut w, b.written.iter().map(|w| &w.pyramid)).unwrap();
    PvecReader::open(w.finish().unwrap()).unwrap()
}

/// The gray pixels of raster 0, cropped from the padded tile.
fn luma(reader: &PvecReader, p: &RasterPyramid) -> Vec<u8> {
    let bytes = reader
        .raster_tile(RasterId::new(0).unwrap(), u32::from(p.levels) - 1, 0, 0)
        .unwrap()
        .expect("a stored tile");
    let t = image::load_from_memory(&bytes).unwrap().to_luma8();
    let mut out = Vec::new();
    for y in 0..p.height as usize {
        let start = y * t.width() as usize;
        out.extend_from_slice(&t.as_raw()[start..start + p.width as usize]);
    }
    out
}

fn only(b: &PdfImageBake) -> &RasterPyramid {
    assert!(b.refused.is_empty(), "{:?}", b.refused);
    assert_eq!(b.written.len(), 1);
    &b.written[0].pyramid
}

/// Bake `stream` (with its globals split off or not) and compare with `reference`.
fn check(stream: &str, picture: &str, with_globals: bool) {
    let (w, h, want) = reference(picture);
    let bytes = if with_globals {
        let (page, globals) = embedded(stream);
        pdf_with(w.into(), h.into(), page, globals, Dictionary::new())
    } else {
        pdf_with(
            w.into(),
            h.into(),
            embedded_inline(stream),
            None,
            Dictionary::new(),
        )
    };
    let b = bake(&bytes);
    let p = only(&b);
    assert_eq!(p.bit_depth, 1, "{stream}: 1-bit tiles");
    assert!(b.written[0].notes.contains(&PdfImageNote::Bilevel));
    assert_eq!(luma(&through_pvec(&b), p), want, "{stream}");
}

#[cfg(feature = "jbig2")]
mod enabled {
    use super::*;

    #[test]
    fn a_generic_region_with_arithmetic_coding_decodes_to_the_reference() {
        check("generic_template0.jbig2", "truchet.png", true);
    }

    #[test]
    fn a_generic_region_with_mmr_decodes_to_the_reference() {
        check("generic_mmr.jbig2", "truchet.png", true);
    }

    #[test]
    fn a_symbol_dictionary_and_text_region_with_arithmetic_coding_decode() {
        check("symbol.jbig2", "honeycomb.png", true);
    }

    #[test]
    fn a_symbol_dictionary_and_text_region_with_huffman_tables_decode() {
        check("huffman_standard_tables.jbig2", "honeycomb.png", true);
    }

    #[test]
    fn refinement_in_symbols_and_text_decodes() {
        check("symbol_refine_text_refine.jbig2", "honeycomb.png", true);
    }

    #[test]
    fn a_generic_refinement_region_decodes() {
        check("refine_template0.jbig2", "truchet.png", true);
    }

    #[test]
    fn a_halftone_region_decodes() {
        check("halftone.jbig2", "maze.png", true);
    }

    #[test]
    fn a_symbol_dictionary_in_the_globals_stream_decodes() {
        let (_, globals) = embedded("symbol_global.jbig2");
        assert!(globals.is_some(), "the fixture has page 0 segments");
        check("symbol_global.jbig2", "honeycomb.png", true);
    }

    #[test]
    fn a_pattern_dictionary_in_the_globals_stream_decodes() {
        let (_, globals) = embedded("halftone_global.jbig2");
        assert!(globals.is_some());
        check("halftone_global.jbig2", "maze.png", true);
    }

    #[test]
    fn the_same_stream_without_a_globals_object_decodes_alike() {
        check("symbol_global.jbig2", "honeycomb.png", false);
        check("halftone_global.jbig2", "maze.png", false);
    }

    #[test]
    fn black_pixels_are_zero_and_decode_one_zero_flips_them() {
        let (w, h, want) = reference("truchet.png");
        let (page, globals) = embedded("generic_template0.jbig2");
        let decode = dictionary! { "Decode" => vec![1.into(), 0.into()] };
        let b = bake(&pdf_with(w.into(), h.into(), page, globals, decode));
        let flipped: Vec<u8> = want.iter().map(|v| 255 - v).collect();
        assert_eq!(luma(&through_pvec(&b), only(&b)), flipped);
        assert!(want.contains(&0) && want.contains(&255));
    }

    #[test]
    fn a_page_size_that_disagrees_with_the_dictionary_is_refused() {
        let (w, h, _) = reference("truchet.png");
        let (page, globals) = embedded("generic_template0.jbig2");
        let b = bake(&pdf_with(
            i64::from(w) + 1,
            h.into(),
            page,
            globals,
            Dictionary::new(),
        ));
        assert!(b.written.is_empty());
        assert_eq!(b.refused[0].error.reason(), "decode-failed");
    }

    #[test]
    fn a_truncated_stream_is_a_refusal_not_a_partial_picture() {
        let (w, h, _) = reference("truchet.png");
        let (mut page, globals) = embedded("generic_template0.jbig2");
        page.truncate(page.len() - 120);
        let b = bake(&pdf_with(
            w.into(),
            h.into(),
            page,
            globals,
            Dictionary::new(),
        ));
        assert!(b.written.is_empty());
        assert_eq!(b.refused[0].error.reason(), "decode-failed");
    }

    #[test]
    fn garbage_is_a_refusal() {
        let b = bake(&pdf_with(8, 8, vec![0xFF; 64], None, Dictionary::new()));
        assert!(b.written.is_empty());
        assert_eq!(b.refused[0].error.reason(), "decode-failed");
    }
}

#[cfg(not(feature = "jbig2"))]
mod disabled {
    use super::*;
    use std::error::Error;

    #[test]
    fn without_the_feature_the_reason_is_filter_not_enabled() {
        let (w, h, _) = reference("truchet.png");
        let (page, globals) = embedded("generic_template0.jbig2");
        let b = bake(&pdf_with(
            w.into(),
            h.into(),
            page,
            globals,
            Dictionary::new(),
        ));
        assert!(b.written.is_empty());
        assert_eq!(
            b.refused[0].error.reason(),
            "filter-not-enabled:JBIG2Decode"
        );
        assert_eq!(
            b.report.refused.get("filter-not-enabled:JBIG2Decode"),
            Some(&1)
        );
    }

    #[test]
    fn the_error_names_the_feature() {
        let b = bake(&pdf_with(8, 8, vec![0; 8], None, Dictionary::new()));
        let e = &b.refused[0].error;
        let text = format!("{e} {:?}", e.source());
        assert!(text.contains("jbig2"), "{text}");
    }
}
