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
    assert_eq!(b.report.written_by_filter.get("JBIG2Decode"), Some(&1));
    assert_eq!(luma(&through_pvec(&b), p), want, "{stream}");
}

#[cfg(feature = "jbig2")]
mod enabled {
    use super::*;
    use libviprs::pvec::pdf_image::{FilterError, PdfImageError};
    use libviprs::pvec::pdf_vector::{PdfVectorLimits, extract_page_paths_from_bytes};

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_generic_region_with_arithmetic_coding_decodes_to_the_reference() {
        check("generic_template0.jbig2", "truchet.png", true);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_generic_region_with_mmr_decodes_to_the_reference() {
        check("generic_mmr.jbig2", "truchet.png", true);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_symbol_dictionary_and_text_region_with_arithmetic_coding_decode() {
        check("symbol.jbig2", "honeycomb.png", true);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_symbol_dictionary_and_text_region_with_huffman_tables_decode() {
        check("huffman_standard_tables.jbig2", "honeycomb.png", true);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn refinement_in_symbols_and_text_decodes() {
        check("symbol_refine_text_refine.jbig2", "honeycomb.png", true);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_generic_refinement_region_decodes() {
        check("refine_template0.jbig2", "truchet.png", true);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_halftone_region_decodes() {
        check("halftone.jbig2", "maze.png", true);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_symbol_dictionary_in_the_globals_stream_decodes() {
        let (_, globals) = embedded("symbol_global.jbig2");
        assert!(globals.is_some(), "the fixture has page 0 segments");
        check("symbol_global.jbig2", "honeycomb.png", true);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_pattern_dictionary_in_the_globals_stream_decodes() {
        let (_, globals) = embedded("halftone_global.jbig2");
        assert!(globals.is_some());
        check("halftone_global.jbig2", "maze.png", true);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn the_same_stream_without_a_globals_object_decodes_alike() {
        check("symbol_global.jbig2", "honeycomb.png", false);
        check("halftone_global.jbig2", "maze.png", false);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
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
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_flate_compressed_globals_stream_is_inflated_first() {
        use std::io::Write;
        let (w, h, want) = reference("honeycomb.png");
        let (page, globals) = embedded("symbol_global.jbig2");
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        z.write_all(&globals.unwrap()).unwrap();
        let mut d = doc();
        let g = d.add_object(Stream::new(
            dictionary! { "Filter" => "FlateDecode" },
            z.finish().unwrap(),
        ));
        let parms = dictionary! { "JBIG2Globals" => Object::Reference(g) };
        let s = jbig2(
            w.into(),
            h.into(),
            page,
            dictionary! { "DecodeParms" => parms },
        );
        let bytes = pdf(
            d,
            Resources::images(vec![("Im1", Object::Stream(s))]),
            format!("q {w} 0 0 {h} 0 0 cm /Im1 Do Q").as_bytes(),
        );
        let b = bake(&bytes);
        assert_eq!(luma(&through_pvec(&b), only(&b)), want);
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_globals_stream_counts_against_the_per_stream_cap() {
        let (w, h, _) = reference("honeycomb.png");
        let (page, globals) = embedded("symbol_global.jbig2");
        let total = page.len() + globals.as_ref().unwrap().len();
        let bytes = pdf_with(w.into(), h.into(), page, globals, Dictionary::new());
        let limits = |cap: usize| PdfVectorLimits {
            max_image_stream_bytes: cap as u64,
            ..PdfVectorLimits::default()
        };
        let over = extract_page_paths_from_bytes(&bytes, 1, &limits(total - 1)).unwrap();
        let b = bake_page_images(&over, &PdfImageOptions::default());
        assert!(matches!(
            b.refused[0].error,
            PdfImageError::StreamTooLarge { .. }
        ));
        let fits = extract_page_paths_from_bytes(&bytes, 1, &limits(total)).unwrap();
        let b = bake_page_images(&fits, &PdfImageOptions::default());
        only(&b);
    }

    /// #1492: the Flate stage on `/JBIG2Globals` gets the room an image stream of this size gets,
    /// and never more than the default per-stream cap (64 MiB), so a deflate bomb in the globals
    /// is refused while inflating. The first case is a small image, where the room is the bound;
    /// the second is a large one, where the room is over 64 MiB and the cap is the bound.
    #[test]
    fn a_flate_bomb_in_the_globals_is_capped_1492() {
        use flate2::{Compression, write::ZlibEncoder};
        use libviprs::pdf::PdfError;
        use std::io::Write;
        for (w, h, inflated) in [
            (163i64, 109i64, 8usize << 20),
            (4096, 4096, (64 << 20) + (512 << 10)),
        ] {
            let mut z = ZlibEncoder::new(Vec::new(), Compression::best());
            z.write_all(&vec![0u8; inflated]).unwrap();
            let bomb = z.finish().unwrap();
            assert!(
                bomb.len() < 128 << 10,
                "the bomb has to be small on the wire"
            );
            let mut d = doc();
            let id = d.add_object(Stream::new(dictionary! { "Filter" => "FlateDecode" }, bomb));
            let bytes = pdf(
                d,
                Resources::images(vec![(
                    "Im1",
                    Object::Stream(jbig2(
                        w,
                        h,
                        vec![0; 16],
                        dictionary! { "DecodeParms" => dictionary! { "JBIG2Globals" => Object::Reference(id) } },
                    )),
                )]),
                format!("q {w} 0 0 {h} 0 0 cm /Im1 Do Q").as_bytes(),
            );
            let page =
                extract_page_paths_from_bytes(&bytes, 1, &PdfVectorLimits::default()).unwrap();
            let b = bake_page_images(&page, &PdfImageOptions::default());
            assert!(
                matches!(
                    b.refused[0].error,
                    PdfImageError::Pdf(PdfError::DecompressionLimitExceeded { .. })
                ),
                "{w}x{h}: the inflate should have hit its cap, got {:?}",
                b.refused[0].error
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn the_refusals_are_typed() {
        let (w, h, _) = reference("truchet.png");
        let (mut page, globals) = embedded("generic_template0.jbig2");
        let wrong = bake(&pdf_with(
            i64::from(w) + 1,
            h.into(),
            page.clone(),
            globals.clone(),
            Dictionary::new(),
        ));
        assert!(matches!(
            &wrong.refused[0].error,
            PdfImageError::Filter {
                source: FilterError::Parms(_),
                ..
            }
        ));
        page.truncate(page.len() - 120);
        let cut = bake(&pdf_with(
            w.into(),
            h.into(),
            page,
            globals,
            Dictionary::new(),
        ));
        assert!(matches!(
            &cut.refused[0].error,
            PdfImageError::Filter {
                source: FilterError::Jbig2(_),
                ..
            }
        ));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
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
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
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

    /// #1495: strict decoding refuses streams that lenient decoding repairs. Each of these is a
    /// valid fixture with one byte changed, and each still gives a picture in lenient mode: a
    /// reserved segment type, a segment that refers to one that is not there, a wrong flag.
    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_stream_that_lenient_decoding_repairs_is_refused_1495() {
        let (w, h, _) = reference("truchet.png");
        for (name, at, xor) in [
            ("generic_template0.jbig2", 34, 0xFF),
            ("generic_template0.jbig2", 37, 0x01),
            ("generic_template0.jbig2", 527, 0x01),
            ("generic_mmr.jbig2", 34, 0xFF),
            ("symbol.jbig2", 30, 0x01),
        ] {
            let (mut page, globals) = embedded(name);
            page[at] ^= xor;
            let b = bake(&pdf_with(
                w.into(),
                h.into(),
                page,
                globals,
                Dictionary::new(),
            ));
            assert!(b.written.is_empty(), "{name} byte {at}: written");
            assert_eq!(
                b.refused[0].error.reason(),
                "decode-failed",
                "{name} byte {at}"
            );
        }
    }

    /// One sequential segment: number, type, page 1, no referred-to segments.
    fn segment(number: u32, kind: u8, data: &[u8]) -> Vec<u8> {
        let mut out = number.to_be_bytes().to_vec();
        out.extend([kind, 0, 1]);
        out.extend((data.len() as u32).to_be_bytes());
        out.extend(data);
        out
    }

    /// An 8 by 8 page information segment.
    fn page_info() -> Vec<u8> {
        let mut d = 8u32.to_be_bytes().to_vec();
        d.extend(8u32.to_be_bytes());
        d.extend([0; 8]);
        d.extend([0, 0, 0]);
        segment(0, 48, &d)
    }

    fn refused_reason(stream: Vec<u8>) -> String {
        let b = bake(&pdf_with(8, 8, stream, None, Dictionary::new()));
        assert!(b.written.is_empty());
        let PdfImageError::Filter {
            source: FilterError::Jbig2(m),
            ..
        } = &b.refused[0].error
        else {
            panic!("{:?}", b.refused[0].error)
        };
        m.clone()
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_symbol_dictionary_declaring_more_symbols_than_it_has_bits_is_refused_unread() {
        // Arithmetic coding, template 0: flags, eight AT bytes, SDNUMEXSYMS, SDNUMNEWSYMS.
        let mut d = vec![0, 0, 3, 0xFF, 0xFD, 0xFF, 2, 0xFE, 0xFE, 0xFE];
        d.extend(65_535u32.to_be_bytes());
        d.extend(65_535u32.to_be_bytes());
        d.extend([0xAA; 4]);
        let mut stream = page_info();
        stream.extend(segment(1, 0, &d));
        assert!(refused_reason(stream).contains("symbol dictionary"));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_region_larger_than_the_page_is_refused_before_it_is_allocated() {
        let mut d = 4000u32.to_be_bytes().to_vec();
        d.extend(4000u32.to_be_bytes());
        d.extend([0; 9]);
        let mut stream = page_info();
        stream.extend(segment(1, 38, &d));
        assert!(refused_reason(stream).contains("4000x4000"));
    }

    /// #1495: the decoder is `hayro-jbig2` 0.4 (with `hayro-ccitt` 0.5, so one copy of that crate)
    /// and the call site asks for strict decoding, which 0.3 has no setting for.
    #[test]
    #[cfg_attr(miri, ignore)] // reads the manifest and the module from disk
    fn the_decoder_is_hayro_jbig2_0_4_and_decodes_strictly_1495() {
        let root = env!("CARGO_MANIFEST_DIR");
        let manifest = std::fs::read_to_string(format!("{root}/Cargo.toml")).unwrap();
        assert!(
            manifest.contains("hayro-jbig2 = { version = \"0.4"),
            "hayro-jbig2 is not 0.4"
        );
        assert!(
            manifest.contains("hayro-ccitt = \"0.5\""),
            "hayro-ccitt is not 0.5"
        );
        let module =
            std::fs::read_to_string(format!("{root}/src/pvec/pdf_filters/jbig2.rs")).unwrap();
        assert!(
            module.contains("strict: true"),
            "the decode settings are not strict"
        );
    }

    /// #1493: an intermediate region (types 4, 20, 36, 40) larger than the page is refused by the
    /// prescan like the immediate and lossless forms are, whatever else the segment holds.
    #[test]
    fn an_oversized_intermediate_region_is_refused_before_it_is_allocated_1493() {
        for kind in [4u8, 20, 36, 40] {
            let mut d = 1_000_000u32.to_be_bytes().to_vec();
            d.extend(1_000_000u32.to_be_bytes());
            d.extend([0; 9]);
            let mut stream = page_info();
            stream.extend(segment(1, kind, &d));
            let reason = refused_reason(stream);
            assert!(
                reason.contains("1000000x1000000 region on a 8x8 page"),
                "type {kind}: {reason}"
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn garbage_is_a_refusal() {
        let b = bake(&pdf_with(8, 8, vec![0xFF; 64], None, Dictionary::new()));
        assert!(b.written.is_empty());
        assert_eq!(b.refused[0].error.reason(), "decode-failed");
    }
}

/// Streams the fuzz target found that `hayro-jbig2` cannot decode in bounded time or memory
/// (measured on 0.3.1, and 0.4.0 fixes none), reported upstream. Each one lives in arithmetic
/// coded data (the symbol sizes, the export runs, the halftone grid), so the segment pre-scan
/// cannot see it and nothing here can refuse it before the decoder runs: that is what `jbig2`
/// being off by default is for. They are `#[ignore]`d because running one is the finding (an
/// endless loop or gigabytes), and they stay here so the day an upstream fix lands,
/// `cargo test --features jbig2 -- --ignored` shows it.
#[cfg(feature = "jbig2")]
mod known_upstream {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn bakes_in_bounds(stream: &str, w: i64, h: i64) {
        let b = bake(&pdf_with(w, h, hex(stream), None, Dictionary::new()));
        assert_eq!(b.report.images, 1);
    }

    #[test]
    #[ignore = "LaurenzV/hayro#1700: export_symbols never ends (8.6 billion iterations in 60 s)"]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn an_export_run_of_zero_never_ends_hayro_1700() {
        bakes_in_bounds(
            concat!(
                "0000000030000100000013000000a30000006d00001ec200001ec2010000000000010001010000005d000003fffdff02",
                "fefefe00000004000000042a8ade999613e94932eee83551c80ae15399fd6506245d849c71418296d3754909f9d98b18",
                "667bbfcf8f79d6494c4811cea9ff520ef87046190be51a1ae8bf15dfa2f4c6c025cf5035ffac00000002072001010000",
                "004f000000a30000006d0000000000000000000c180000005a98f4702d3fedd5814de8a0a8d33ea0bb7c21e2a3553e56",
                "d595c58c794c355eb6b138beb8a8210221fe35ad84411d042bc434c9132927ffac0000000331000100000000",
            ),
            163,
            109,
        );
    }

    #[test]
    #[ignore = "LaurenzV/hayro#1701: 1.4 GB and 20 s in symbol bitmaps"]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn symbols_of_24000_pixels_across_cost_1_4_gb_hayro_1701() {
        bakes_in_bounds(
            concat!(
                "0000000030000100000013000000a30000006d00001ec200001ec2010000000000010001010000005d000003fffdff02",
                "fefefe00000004000000fe2a8ade999613e94903eee8357fc80ae15399fd6506245d849c71418296d375fc09f9d98b18",
                "667bbfa18f79d6494c4855cea9ff520ef87046190be51a1ae8bf15dfa2f4c6c025f85035ff5500000002072001010000",
                "004f000000a30000006d0000000000000000001b180000005a98f4702d3fedd5814de8a0a8d33ea0bb7c21e2a3553e56",
                "d595c58c79fd355eb6b138beb8a8210221fe35ad84411d042bc434c9132927ffac0000fc0331000100000000",
            ),
            163,
            109,
        );
    }

    #[test]
    #[ignore = "LaurenzV/hayro#1598: 2^31 empty height classes, 39 s"]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn empty_height_classes_run_to_overflow_hayro_1598() {
        bakes_in_bounds(
            concat!(
                "0000000030000100000013000000a30000006d00001ec200001ec2010000000000010001010000005d000003fffdff02",
                "fefefe00000004000000fe2ad8de99ef13e94932eee8357fc80ae15399fd6506245d84fc71418296d3754909f9d98b18",
                "667bbfa1ef79d6494c4811cea9ff520ef87046190be5fe1ae8bf15dfa2f4c6c025f85035ff5500000002072001010000",
                "004f000000a30000006d00000000000000fd001b180000005a98f4702d3fedd5814de8a0a8d33ea0bb7c21e2a3553e56",
                "d595c58c794c355eb6b138beb8a8210221fe35ad84411d042bc434c9132927ffac0000ef0331000100000000",
            ),
            163,
            109,
        );
    }

    #[test]
    #[ignore = "LaurenzV/hayro#1550: a 64786x24332 grid, 6.4 GB and 28 s"]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn a_halftone_grid_of_1_6_billion_cells_hayro_1550() {
        bakes_in_bounds(
            concat!(
                "00000000300001000000130000009b0000006500001ec200001ec2010000000000011001010000001c00090900000003",
                "e9fe4aea9b3fcbf636e6d528b7efc58f9641bfffac00000002172001010000005f0000009b0000006500000000000000",
                "0000000000fd1200005f0c00003f0000000000090000001e44d543ec5900b651bf1d5af117e9cd03ca8621010a5feb0b",
                "93bdf2457f5a40ef594395eefb557f0d84764b15892809a54eccf9e86851ffac0000000331000100000000",
            ),
            155,
            101,
        );
    }
}

#[cfg(not(feature = "jbig2"))]
mod disabled {
    use super::*;
    use std::error::Error;

    #[test]
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
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
    #[cfg_attr(miri, ignore)] // reads the fixtures from disk
    fn the_error_names_the_feature() {
        let b = bake(&pdf_with(8, 8, vec![0; 8], None, Dictionary::new()));
        let e = &b.refused[0].error;
        let text = format!("{e} {:?}", e.source());
        assert!(text.contains("jbig2"), "{text}");
    }
}
