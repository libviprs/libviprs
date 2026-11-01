//! Issue #1363: `CCITTFaxDecode` images of a PDF page become PVec rasters.
//!
//! Every stream is encoded outside the crate under test. The bytes in `common/ccitt_fixtures.rs`
//! come from libtiff (`tiffcp -c g4`, `g3:1d`, `g3:2d`, Debian bookworm
//! `libtiff-tools`, the generator is `tests/common/ccitt_gen.py`), so a decoder bug cannot hide
//! behind an encoder with the same bug. The `EncodedByteAlign` stream is the `g3:1d` strip with
//! its EOLs removed and each row padded to a byte (`byte_aligned` in the generator). The picture is `ink_small` / `ink_wide` below; the
//! expected gray samples are worked out by hand from the PDF rules:
//!
//! * `/BlackIs1 false` (the default): a black pixel is a 0 bit, so ink is gray 0.
//! * `/BlackIs1 true`: a black pixel is a 1 bit, so DeviceGray reads ink as white.
//! * `/Decode [1 0]` on top of that flips it back.
//!
//! Needs `--features pvec-raster`. The corpus test needs `LIBVIPRS_PDF_CANDIDATES` (the
//! `pdf-candidates` checkout) and says so when it is not set.

#![cfg(feature = "pvec-raster")]

#[path = "common/ccitt_fixtures.rs"]
mod ccitt_fixtures;
#[path = "common/pdf_images.rs"]
mod fixture;

use ccitt_fixtures::*;
use fixture::*;
use libviprs::pvec::pdf_image::{
    FilterError, PdfImageBake, PdfImageError, PdfImageNote, PdfImageOptions, bake_page_images,
};
use libviprs::raster_pvec::container::write_rasters;
use libviprs::raster_pvec::pyramid::RasterPyramid;
use lopdf::{Dictionary, Object, dictionary};
use pvec_facade::unstable::RasterId;
use pvec_facade::{Bounds2, Curve, EntityId, Manifest, PvecReader, PvecWriter};
use pvec_geometry::{Point2, Polyline};
use std::error::Error;

/// Must match `ink_small` in `tests/common/ccitt_gen.py`.
fn ink_small(x: i64, y: i64) -> bool {
    (x + 2 * y) % 11 < 4 || (x >= 30 && y % 2 == 0)
}

/// Must match `ink_wide` in `tests/common/ccitt_gen.py`.
fn ink_wide(x: i64, y: i64) -> bool {
    (x / 3 + y) % 5 == 0 || x > 1700
}

fn picture(w: i64, h: i64, ink: fn(i64, i64) -> bool, ink_gray: u8) -> Vec<u8> {
    let paper = 255 - ink_gray;
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| if ink(x, y) { ink_gray } else { paper })
        .collect()
}

/// An image XObject around `data`, with `/DecodeParms` `parms` and extra dictionary entries.
fn ccitt(w: i64, h: i64, parms: Dictionary, extra: Dictionary, data: &[u8]) -> Vec<u8> {
    let mut dict = dictionary! {
        "ColorSpace" => "DeviceGray",
        "BitsPerComponent" => 1,
        "Filter" => "CCITTFaxDecode",
        "DecodeParms" => parms,
    };
    for (k, v) in extra.iter() {
        dict.set(k.clone(), v.clone());
    }
    one(
        image(w, h, dict, data.to_vec()),
        "q 20 0 0 20 100 200 cm /Im1 Do Q",
    )
}

fn bake(bytes: &[u8]) -> PdfImageBake {
    // One tile per image, so a tile read back is the whole picture.
    let mut options = PdfImageOptions::default();
    options.tile_edge = 2048;
    bake_page_images(&page(bytes), &options)
}

fn through_pvec(b: &PdfImageBake) -> PvecReader {
    let mut w = PvecWriter::new(Manifest {
        metres_per_unit: 1.0,
        bounds: Bounds2 {
            min: [0.0, 0.0],
            max: [612.0, 792.0],
        },
        capabilities: Vec::new(),
        producer_id: "libviprs-1363".into(),
        producer_version: "1".into(),
    })
    .unwrap();
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

fn tile_bytes(reader: &PvecReader, p: &RasterPyramid) -> Vec<u8> {
    reader
        .raster_tile(RasterId::new(0).unwrap(), u32::from(p.levels) - 1, 0, 0)
        .unwrap()
        .expect("a stored tile")
}

/// The gray samples of the one raster `b` wrote, read back through PVec.
fn luma(b: &PdfImageBake) -> Vec<u8> {
    assert!(b.refused.is_empty(), "{:?}", b.refused);
    assert_eq!(b.written.len(), 1);
    let p = &b.written[0].pyramid;
    let t = image::load_from_memory(&tile_bytes(&through_pvec(b), p))
        .unwrap()
        .to_luma8();
    let stride = t.width() as usize;
    let mut out = Vec::new();
    for y in 0..p.height as usize {
        out.extend_from_slice(&t.as_raw()[y * stride..y * stride + p.width as usize]);
    }
    out
}

fn decoded(bytes: &[u8]) -> Vec<u8> {
    luma(&bake(bytes))
}

/// `got` equals `want`, and if not, say where the first pixel differs.
fn same_picture(w: usize, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "sample count");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "first difference at x {} y {}: got {} want {}",
            i % w,
            i / w,
            got[i],
            want[i]
        );
    }
}

fn filter_error(e: &PdfImageError) -> &FilterError {
    e.source()
        .and_then(|s| s.downcast_ref::<FilterError>())
        .unwrap_or_else(|| panic!("not a filter error: {e:?}"))
}

fn refusal(bytes: &[u8]) -> PdfImageError {
    let b = bake(bytes);
    assert!(b.written.is_empty(), "wrote {} images", b.written.len());
    b.refused.into_iter().next().expect("a refusal").error
}

const K4: i64 = 4;

fn small_parms(k: i64) -> Dictionary {
    dictionary! { "K" => k, "Columns" => 40, "Rows" => 8 }
}

// ---- the three coding schemes ----

#[test]
fn a_group_4_stream_decodes_to_the_picture() {
    let pdf = ccitt(40, 8, small_parms(-1), dictionary! {}, G4_SMALL);
    same_picture(40, &decoded(&pdf), &picture(40, 8, ink_small, 0));
}

#[test]
fn a_group_3_one_dimensional_stream_decodes_to_the_picture() {
    let pdf = ccitt(40, 8, small_parms(0), dictionary! {}, G3_1D_SMALL);
    same_picture(40, &decoded(&pdf), &picture(40, 8, ink_small, 0));
}

#[test]
fn a_group_3_mixed_stream_decodes_to_the_picture() {
    let pdf = ccitt(40, 8, small_parms(K4), dictionary! {}, G3_2D_SMALL);
    same_picture(40, &decoded(&pdf), &picture(40, 8, ink_small, 0));
}

// ---- polarity ----

#[test]
fn black_is_1_true_reads_the_ink_as_white_and_decode_1_0_puts_it_back() {
    let mut p = small_parms(-1);
    p.set("BlackIs1", true);
    let plain = ccitt(40, 8, p.clone(), dictionary! {}, G4_SMALL);
    // The same black and white geometry, the opposite gray: 1 bits are white in DeviceGray.
    same_picture(40, &decoded(&plain), &picture(40, 8, ink_small, 255));
    let flipped = ccitt(
        40,
        8,
        p,
        dictionary! { "Decode" => vec![Object::Integer(1), Object::Integer(0)] },
        G4_SMALL,
    );
    // The shape of the corpus file: BlackIs1 true with /Decode [1 0] is black ink again.
    same_picture(40, &decoded(&flipped), &picture(40, 8, ink_small, 0));
}

#[test]
fn black_is_1_false_and_true_with_decode_give_the_same_picture() {
    let mut off = small_parms(0);
    off.set("BlackIs1", false);
    let mut on = small_parms(0);
    on.set("BlackIs1", true);
    let a = ccitt(40, 8, off, dictionary! {}, G3_1D_SMALL);
    let b = ccitt(
        40,
        8,
        on,
        dictionary! { "Decode" => vec![Object::Integer(1), Object::Integer(0)] },
        G3_1D_SMALL,
    );
    assert_eq!(decoded(&a), decoded(&b));
}

// ---- the dictionary defaults ----

#[test]
fn columns_absent_means_1728() {
    let pdf = ccitt(
        1728,
        4,
        dictionary! { "K" => -1, "Rows" => 4 },
        dictionary! {},
        G4_WIDE,
    );
    same_picture(1728, &decoded(&pdf), &picture(1728, 4, ink_wide, 0));
}

#[test]
fn rows_absent_with_end_of_block_stops_at_the_marker() {
    let pdf = ccitt(
        40,
        8,
        dictionary! { "K" => -1, "Columns" => 40, "EndOfBlock" => true },
        dictionary! {},
        G4_SMALL,
    );
    same_picture(40, &decoded(&pdf), &picture(40, 8, ink_small, 0));
}

#[test]
fn encoded_byte_align_true_reads_rows_that_start_on_a_byte() {
    let pdf = ccitt(
        40,
        8,
        dictionary! {
            "K" => 0, "Columns" => 40, "Rows" => 8,
            "EncodedByteAlign" => true,
        },
        dictionary! {},
        G3_1D_ALIGNED_SMALL,
    );
    same_picture(40, &decoded(&pdf), &picture(40, 8, ink_small, 0));
}

// ---- refusals that are typed, never a silent resize ----

#[test]
fn a_stream_short_by_one_row_is_short_output() {
    // The stream holds 8 rows; the dictionary says 9.
    let mut p = small_parms(-1);
    p.set("Rows", 9);
    let e = refusal(&ccitt(40, 9, p, dictionary! {}, G4_SMALL));
    assert_eq!(e.reason(), "decode-failed");
    assert!(
        matches!(filter_error(&e), FilterError::ShortOutput { .. }),
        "{e:?}"
    );
}

#[test]
fn columns_that_disagree_with_width_are_parms() {
    let mut p = small_parms(-1);
    p.set("Columns", 48);
    let e = refusal(&ccitt(40, 8, p, dictionary! {}, G4_SMALL));
    assert_eq!(e.reason(), "decode-failed");
    assert!(matches!(filter_error(&e), FilterError::Parms(_)), "{e:?}");
}

#[test]
fn rows_that_disagree_with_height_are_parms() {
    let mut p = small_parms(-1);
    p.set("Rows", 7);
    let e = refusal(&ccitt(40, 8, p, dictionary! {}, G4_SMALL));
    assert!(matches!(filter_error(&e), FilterError::Parms(_)), "{e:?}");
}

#[test]
fn garbage_is_a_refusal_not_a_panic() {
    // Zero bits are no G4 code, and an empty stream has no rows: both are refusals.
    for junk in [&[0u8; 7][..], &[][..]] {
        let e = refusal(&ccitt(40, 8, small_parms(-1), dictionary! {}, junk));
        assert_eq!(e.reason(), "decode-failed");
    }
}

// ---- what the report says ----

#[test]
fn the_written_image_is_bilevel_with_one_bit_tiles_and_counted_by_filter() {
    let pdf = ccitt(40, 8, small_parms(-1), dictionary! {}, G4_SMALL);
    let b = bake(&pdf);
    assert!(b.refused.is_empty(), "{:?}", b.refused);
    assert_eq!(b.report.bilevel, 1);
    assert!(b.written[0].notes.contains(&PdfImageNote::Bilevel));
    assert_eq!(b.report.written_by_filter.get("CCITTFaxDecode"), Some(&1));
    assert!(b.report.refused.is_empty());
    // The tile is a PNG of one bit per sample: IHDR bit depth is byte 24 of the file.
    let bytes = tile_bytes(&through_pvec(&b), &b.written[0].pyramid);
    assert_eq!(&bytes[1..4], b"PNG");
    assert_eq!(bytes[24], 1, "bit depth");
}

#[test]
fn a_flate_stage_in_front_of_the_fax_stage_takes_the_second_decode_parms() {
    use flate2::{Compression, write::ZlibEncoder};
    use std::io::Write;
    let mut z = ZlibEncoder::new(Vec::new(), Compression::default());
    z.write_all(G4_SMALL).unwrap();
    let flated = z.finish().unwrap();
    let pdf = ccitt_chain(&flated);
    same_picture(40, &decoded(&pdf), &picture(40, 8, ink_small, 0));
}

fn ccitt_chain(data: &[u8]) -> Vec<u8> {
    let dict = dictionary! {
        "ColorSpace" => "DeviceGray",
        "BitsPerComponent" => 1,
        "Filter" => vec![name("FlateDecode"), name("CCITTFaxDecode")],
        "DecodeParms" => vec![Object::Null, Object::Dictionary(small_parms(-1))],
    };
    one(
        image(40, 8, dict, data.to_vec()),
        "q 20 0 0 20 100 200 cm /Im1 Do Q",
    )
}

// ---- the corpus ----

/// `usace_swf_gpw_daca67-06-d-0029.pdf`: seven pages, each one page-sized 400 dpi G4 scan with
/// `/BlackIs1 true` and `/Decode [1 0]`, no path operators. Before this issue it converted to
/// nothing. The content stream of page N is `q 1584 0 0 H 0 0 cm /Im0 Do Q`; the image is 8800
/// by ROWS, so the placement is `[a/w, -c/h, b/w, -d/h, c+e, d+f]` with c = e = 0.
#[test]
fn the_scanned_sheet_converts_to_seven_underlays() {
    let Some(dir) = std::env::var_os("LIBVIPRS_PDF_CANDIDATES") else {
        eprintln!("LIBVIPRS_PDF_CANDIDATES is not set: the 5 MB corpus file is not in the repo");
        return;
    };
    let path = std::path::Path::new(&dir).join("arch/usace_swf_gpw_daca67-06-d-0029.pdf");
    let bytes = std::fs::read(&path).expect("the corpus file");
    assert_eq!(bytes.len(), 5_355_305, "not the file CANDIDATES.md pins");
    // (page height in points as the content stream writes it, image rows)
    let pages: [(&str, u32); 7] = [
        ("1094.9400024", 6083),
        ("1096.7400055", 6093),
        ("1098.7200012", 6104),
        ("1088.8200073", 6049),
        ("1087.0200043", 6039),
        ("1091.6999969", 6065),
        ("1100.3399963", 6113),
    ];
    let mut written = 0;
    for (n, (d_text, rows)) in pages.iter().enumerate() {
        let page = libviprs::pvec::pdf_vector::extract_page_paths_from_bytes(
            &bytes,
            n + 1,
            &libviprs::pvec::pdf_vector::PdfVectorLimits::default(),
        )
        .unwrap();
        let b = bake_page_images(&page, &PdfImageOptions::default());
        assert!(b.refused.is_empty(), "page {}: {:?}", n + 1, b.refused);
        assert_eq!(b.written.len(), 1, "page {}", n + 1);
        assert_eq!(b.report.written_by_filter.get("CCITTFaxDecode"), Some(&1));
        assert_eq!(b.report.bilevel, 1);
        let p = &b.written[0].pyramid;
        assert_eq!((p.width, p.height), (8800, *rows), "page {}", n + 1);
        let d = f64::from(d_text.parse::<f32>().unwrap());
        let expect = [1584.0 / 8800.0, 0.0, 0.0, -d / f64::from(*rows), 0.0, d];
        assert_eq!(
            p.pixel_to_source.map(f64::to_bits),
            expect.map(f64::to_bits),
            "page {}",
            n + 1
        );
        written += 1;
    }
    assert_eq!(written, 7);
}
