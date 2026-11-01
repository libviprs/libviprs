//! Issue #1332 (RL2.1): the images of a PDF page become PVec rasters.
//!
//! One construct per PDF, built in memory. Every written case is read back **through PVec's
//! reader**: the decoded tile equals the pixels worked out by hand here (tolerance 0, every case
//! is lossless but the `DCTDecode` one), and the placement equals the hand-computed
//! coefficients bit for bit. Needs `--features pvec-raster`.

#![cfg(feature = "pvec-raster")]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::pdf_image::bake_page_images;
use libviprs::pvec::pdf_image::{PdfImageBake, PdfImageError, PdfImageNote, PdfImageOptions};
use libviprs::pvec::pdf_vector::{PdfVectorError, PdfVectorLimits, extract_page_paths_from_bytes};
use libviprs::raster_pvec::container::write_rasters;
use libviprs::raster_pvec::pyramid::RasterPyramid;
use lopdf::{Object, Stream, dictionary};
use pvec_facade::unstable::{Band, PixelFormat as Channels, RasterId};
use pvec_facade::{Bounds2, Curve, EntityId, Manifest, PvecReader, PvecWriter};
use pvec_geometry::{Point2, Polyline};

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
        producer_id: "libviprs-1332".into(),
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

fn bits(m: [f64; 6]) -> [u64; 6] {
    m.map(f64::to_bits)
}

/// The full resolution tile (0, 0) of raster `id`, read through PVec and decoded.
fn tile(reader: &PvecReader, id: u32, p: &RasterPyramid) -> image::DynamicImage {
    let bytes = reader
        .raster_tile(RasterId::new(id).unwrap(), u32::from(p.levels) - 1, 0, 0)
        .unwrap()
        .expect("a stored tile");
    image::load_from_memory(&bytes).unwrap()
}

fn luma(reader: &PvecReader, id: u32, p: &RasterPyramid) -> Vec<u8> {
    let t = tile(reader, id, p).to_luma8();
    crop(t.width(), p.width, p.height, 1, t.as_raw())
}

fn rgb(reader: &PvecReader, id: u32, p: &RasterPyramid) -> Vec<u8> {
    let t = tile(reader, id, p).to_rgb8();
    crop(t.width(), p.width, p.height, 3, t.as_raw())
}

fn rgba(reader: &PvecReader, id: u32, p: &RasterPyramid) -> Vec<u8> {
    let t = tile(reader, id, p).to_rgba8();
    crop(t.width(), p.width, p.height, 4, t.as_raw())
}

/// The top-left `w` by `h` of a padded tile.
fn crop(stride_px: u32, w: u32, h: u32, bpp: usize, raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for y in 0..h as usize {
        let start = y * stride_px as usize * bpp;
        out.extend_from_slice(&raw[start..start + w as usize * bpp]);
    }
    out
}

fn only(b: &PdfImageBake) -> &RasterPyramid {
    assert!(b.refused.is_empty(), "{:?}", b.refused);
    assert_eq!(b.written.len(), 1);
    &b.written[0].pyramid
}

#[test]
fn a_gray_8_bit_image_is_written_as_gray_with_the_hand_computed_placement() {
    let b = bake(&one(
        gray8(2, 2, &[0x00, 0x40, 0x80, 0xFF]),
        "q 20 0 0 20 100 200 cm /Im1 Do Q",
    ));
    let p = only(&b);
    assert_eq!(
        bits(p.pixel_to_source),
        bits([10.0, 0.0, 0.0, -10.0, 100.0, 220.0])
    );
    let reader = through_pvec(&b);
    let pl = &reader.raster_placements().unwrap()[0];
    assert_eq!(bits(pl.pixel_to_source), bits(p.pixel_to_source));
    assert_eq!(pl.pixel_format, Channels::Gray);
    assert_eq!((pl.width, pl.height, pl.visible_box), (2, 2, [0, 0, 2, 2]));
    assert_eq!(luma(&reader, 0, p), vec![0x00, 0x40, 0x80, 0xFF]);
}

#[test]
fn a_rotated_sheared_ctm_gives_the_coefficients_of_the_rasters_document() {
    // CTM [8 4 -2 6 10 20] over 4 by 2 pixels: [A/w -C/h B/w -D/h C+E D+F].
    let data: Vec<u8> = (0..24).map(|i| i * 9).collect();
    let b = bake(&one(rgb8(4, 2, &data), "q 8 4 -2 6 10 20 cm /Im1 Do Q"));
    let p = only(&b);
    assert_eq!(
        bits(p.pixel_to_source),
        bits([2.0, 1.0, 1.0, -3.0, 8.0, 26.0])
    );
    let reader = through_pvec(&b);
    assert_eq!(rgb(&reader, 0, p), data);
}

#[test]
fn a_16_bit_rgb_image_is_reduced_to_8_bits_and_says_so() {
    // Two pixels of u16 big-endian samples: (0xFFFF 0x0000 0x8000) and (0x1234 0xFFFF 0x0001).
    let data = [
        0xFF, 0xFF, 0x00, 0x00, 0x80, 0x00, 0x12, 0x34, 0xFF, 0xFF, 0x00, 0x01,
    ];
    let s = image(
        2,
        1,
        dictionary! { "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 16 },
        data.to_vec(),
    );
    let b = bake(&one(s, "q 2 0 0 1 0 0 cm /Im1 Do Q"));
    let p = only(&b);
    let reader = through_pvec(&b);
    // v * 255 / 65535, rounded.
    assert_eq!(rgb(&reader, 0, p), vec![255, 0, 128, 18, 255, 0]);
    assert!(b.written[0].notes.contains(&PdfImageNote::BitDepthReduced));
    assert_eq!(b.report.bit_depth_reduced, 1);
}

#[test]
fn an_indexed_image_is_expanded_through_its_lookup() {
    let cs = Object::Array(vec![
        name("Indexed"),
        name("DeviceRGB"),
        2.into(),
        literal(&[255, 0, 0, 0, 255, 0, 0, 0, 255]),
    ]);
    let s = image(
        2,
        2,
        dictionary! { "ColorSpace" => cs, "BitsPerComponent" => 8 },
        vec![0, 1, 2, 1],
    );
    let b = bake(&one(s, "q 2 0 0 2 0 0 cm /Im1 Do Q"));
    let p = only(&b);
    let reader = through_pvec(&b);
    assert_eq!(
        rgb(&reader, 0, p),
        vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 0]
    );
}

#[test]
fn an_icc_based_image_is_taken_as_its_device_space_and_reported_approximated() {
    let mut d = doc();
    let icc = d.add_object(Stream::new(dictionary! { "N" => 3 }, vec![0u8; 16]));
    let cs = Object::Array(vec![name("ICCBased"), Object::Reference(icc)]);
    let s = image(
        2,
        1,
        dictionary! { "ColorSpace" => cs, "BitsPerComponent" => 8 },
        vec![1, 2, 3, 250, 251, 252],
    );
    let bytes = pdf(
        d,
        Resources::images(vec![("Im1", Object::Stream(s))]),
        b"q 2 0 0 1 0 0 cm /Im1 Do Q",
    );
    let b = bake(&bytes);
    let p = only(&b);
    let reader = through_pvec(&b);
    assert_eq!(rgb(&reader, 0, p), vec![1, 2, 3, 250, 251, 252]);
    assert!(
        b.written[0]
            .notes
            .contains(&PdfImageNote::ColourSpaceApproximated)
    );
    assert_eq!(b.report.colour_approximated, 1);
}

#[test]
fn an_image_mask_is_painted_in_the_fill_colour_with_decode_honoured() {
    // 8 by 2, one bit each: 0xAA = 10101010, 0xF0 = 11110000.
    let mask = |decode: Option<Vec<Object>>| {
        let mut extra = dictionary! { "ImageMask" => true, "BitsPerComponent" => 1 };
        if let Some(d) = decode {
            extra.set("Decode", d);
        }
        image(8, 2, extra, vec![0xAA, 0xF0])
    };
    let content = "1 0 0 rg q 8 0 0 2 0 0 cm /Im1 Do Q";

    // Decode [1 0]: a 1 paints.
    let b = bake(&one(mask(Some(vec![1.into(), 0.into()])), content));
    let p = only(&b);
    let reader = through_pvec(&b);
    let alpha = |painted: bool| if painted { 255 } else { 0 };
    let mut want = Vec::new();
    for byte in [0xAAu8, 0xF0] {
        for bit in (0..8).rev() {
            want.extend_from_slice(&[255, 0, 0, alpha(byte >> bit & 1 == 1)]);
        }
    }
    assert_eq!(rgba(&reader, 0, p), want);
    assert_eq!(
        reader.raster_placements().unwrap()[0].pixel_format,
        Channels::Rgba
    );

    // The default Decode [0 1]: a 0 paints.
    let b = bake(&one(mask(None), content));
    let p = only(&b);
    let reader = through_pvec(&b);
    let mut want = Vec::new();
    for byte in [0xAAu8, 0xF0] {
        for bit in (0..8).rev() {
            want.extend_from_slice(&[255, 0, 0, alpha(byte >> bit & 1 == 0)]);
        }
    }
    assert_eq!(rgba(&reader, 0, p), want);
}

#[test]
fn an_smask_is_resampled_to_the_image_size_and_written_into_alpha() {
    let mut d = doc();
    let sm = d.add_object(gray8(2, 2, &[0, 85, 170, 255]));
    let mut s = rgb8(2, 2, &[255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30]);
    s.dict.set("SMask", Object::Reference(sm));
    let bytes = pdf(
        d,
        Resources::images(vec![("Im1", Object::Stream(s))]),
        b"q 2 0 0 2 0 0 cm /Im1 Do Q",
    );
    let b = bake(&bytes);
    let p = only(&b);
    let reader = through_pvec(&b);
    assert_eq!(
        rgba(&reader, 0, p),
        vec![255, 0, 0, 0, 0, 255, 0, 85, 0, 0, 255, 170, 10, 20, 30, 255]
    );

    // A smaller mask is stretched over the image: one gray pixel covers all four.
    let mut d = doc();
    let sm = d.add_object(gray8(1, 1, &[200]));
    let mut s = gray8(2, 2, &[1, 2, 3, 4]);
    s.dict.set("SMask", Object::Reference(sm));
    let bytes = pdf(
        d,
        Resources::images(vec![("Im1", Object::Stream(s))]),
        b"q 2 0 0 2 0 0 cm /Im1 Do Q",
    );
    let b = bake(&bytes);
    let p = only(&b);
    let reader = through_pvec(&b);
    assert_eq!(
        rgba(&reader, 0, p),
        vec![1, 1, 1, 200, 2, 2, 2, 200, 3, 3, 3, 200, 4, 4, 4, 200]
    );
}

#[test]
fn an_inline_image_is_baked_like_an_xobject() {
    let mut content = b"q 4 0 0 2 10 10 cm BI /W 2 /H 2 /CS /DeviceGray /BPC 8 ID ".to_vec();
    content.extend_from_slice(&[0x10, 0x20, 0x30, 0x40]);
    content.extend_from_slice(b" EI Q");
    let bytes = pdf(doc(), Resources::images(Vec::new()), &content);
    let b = bake(&bytes);
    let p = only(&b);
    assert_eq!(
        bits(p.pixel_to_source),
        bits([2.0, 0.0, 0.0, -1.0, 10.0, 12.0])
    );
    let reader = through_pvec(&b);
    assert_eq!(luma(&reader, 0, p), vec![0x10, 0x20, 0x30, 0x40]);
}

#[test]
fn a_flate_image_with_a_png_predictor_is_unpredicted() {
    use std::io::Write;
    // Row 0: Sub, 10 then +5. Row 1: Up, +1 then +2 over row 0.
    let raw = [1u8, 10, 5, 2, 1, 2];
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    z.write_all(&raw).unwrap();
    let s = image(
        2,
        2,
        dictionary! {
            "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8, "Filter" => "FlateDecode",
            "DecodeParms" => dictionary! { "Predictor" => 15, "Colors" => 1, "BitsPerComponent" => 8, "Columns" => 2 },
        },
        z.finish().unwrap(),
    );
    let b = bake(&one(s, "q 2 0 0 2 0 0 cm /Im1 Do Q"));
    let p = only(&b);
    let reader = through_pvec(&b);
    assert_eq!(luma(&reader, 0, p), vec![10, 15, 11, 17]);
}

#[test]
fn a_dct_image_is_decoded_and_retiled_and_the_report_says_the_tiles_are_a_re_encode() {
    let mut src = Vec::new();
    for y in 0..16u8 {
        for x in 0..16u8 {
            src.extend_from_slice(&[x * 10, y * 10, 128]);
        }
    }
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 95)
        .encode(&src, 16, 16, image::ExtendedColorType::Rgb8)
        .unwrap();
    let want = image::load_from_memory(&jpeg).unwrap().to_rgb8();
    let s = image(
        16,
        16,
        dictionary! { "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8, "Filter" => "DCTDecode" },
        jpeg,
    );
    let b = bake(&one(s, "q 16 0 0 16 0 0 cm /Im1 Do Q"));
    let p = only(&b);
    let reader = through_pvec(&b);
    assert_eq!(rgb(&reader, 0, p), want.as_raw().clone());
    assert!(b.written[0].notes.contains(&PdfImageNote::ReEncodedLossy));
    assert_eq!(b.report.re_encoded_lossy, 1);
}

fn refused_reason(bytes: &[u8]) -> (PdfImageError, String) {
    let b = bake(bytes);
    assert!(b.written.is_empty(), "{:?}", b.written);
    assert_eq!(b.refused.len(), 1);
    assert_eq!(b.report.refused_total(), 1);
    let r = b.refused.into_iter().next().unwrap();
    let reason = r.error.reason();
    assert_eq!(b.report.refused.get(&reason), Some(&1));
    (r.error, reason)
}

#[test]
fn a_jpx_image_is_not_enabled_by_name() {
    let s = image(
        2,
        2,
        dictionary! { "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8, "Filter" => "JPXDecode" },
        vec![0u8; 32],
    );
    let (e, reason) = refused_reason(&one(s, "q 2 0 0 2 0 0 cm /Im1 Do Q"));
    assert!(
        matches!(&e, PdfImageError::FilterNotEnabled { filter, .. } if filter == "JPXDecode"),
        "{e:?}"
    );
    assert_eq!(reason, "filter-not-enabled:JPXDecode");
}

#[test]
fn a_separation_image_is_refused_by_name() {
    let cs = Object::Array(vec![
        name("Separation"),
        name("Spot"),
        name("DeviceGray"),
        Object::Dictionary(dictionary! { "FunctionType" => 2, "N" => 1 }),
    ]);
    let s = image(
        1,
        1,
        dictionary! { "ColorSpace" => cs, "BitsPerComponent" => 8 },
        vec![7],
    );
    let (e, _) = refused_reason(&one(s, "q 1 0 0 1 0 0 cm /Im1 Do Q"));
    assert!(
        matches!(&e, PdfImageError::UnsupportedColourSpace(n) if n == "Separation"),
        "{e:?}"
    );
}

#[test]
fn an_image_under_a_graphics_state_soft_mask_is_refused() {
    let gs = Object::Dictionary(dictionary! {
        "Type" => "ExtGState",
        "SMask" => dictionary! { "Type" => "Mask", "S" => "Alpha" },
    });
    let bytes = pdf(
        doc(),
        Resources {
            xobjects: vec![("Im1", Object::Stream(gray8(1, 1, &[7])))],
            gstates: vec![("GS1", gs)],
        },
        b"/GS1 gs q 1 0 0 1 0 0 cm /Im1 Do Q",
    );
    let (e, _) = refused_reason(&bytes);
    assert!(matches!(e, PdfImageError::GraphicsStateSoftMask), "{e:?}");
}

#[test]
fn a_singular_ctm_is_refused() {
    let (e, _) = refused_reason(&one(gray8(1, 1, &[7]), "q 0 0 0 0 5 5 cm /Im1 Do Q"));
    assert!(matches!(e, PdfImageError::SingularCtm), "{e:?}");
}

#[test]
fn the_raster_id_follows_content_order_and_the_band_follows_the_paths() {
    // Im1 before every path (underlay), Im2 after every path (overlay).
    let bytes = pdf(
        doc(),
        Resources::images(vec![
            ("Im1", Object::Stream(gray8(1, 1, &[10]))),
            ("Im2", Object::Stream(gray8(1, 1, &[20]))),
            ("Im3", Object::Stream(gray8(1, 1, &[30]))),
        ]),
        b"q 5 0 0 5 0 0 cm /Im2 Do Q 0 0 10 10 re f q 5 0 0 5 20 0 cm /Im1 Do Q q 5 0 0 5 40 0 cm /Im3 Do Q",
    );
    let b = bake(&bytes);
    assert_eq!(b.written.len(), 3);
    // Content order is Im2, Im1, Im3: the ids are those positions.
    let reader = through_pvec(&b);
    let pls = reader.raster_placements().unwrap();
    assert_eq!(pls.len(), 3);
    assert_eq!(pls[0].band, Band::Underlay, "Im2 came before the only path");
    assert_eq!(pls[1].band, Band::Overlay);
    assert_eq!(pls[2].band, Band::Overlay);
    assert_eq!(luma(&reader, 0, &b.written[0].pyramid), vec![20]);
    assert_eq!(luma(&reader, 1, &b.written[1].pyramid), vec![10]);
    assert_eq!(luma(&reader, 2, &b.written[2].pyramid), vec![30]);
    assert_eq!(
        b.written.iter().map(|w| w.index).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}

#[test]
fn an_image_painted_between_paths_is_band_0_with_an_interleaved_line() {
    let bytes = pdf(
        doc(),
        Resources::images(vec![("Im1", Object::Stream(gray8(1, 1, &[10])))]),
        b"0 0 10 10 re f q 5 0 0 5 20 0 cm /Im1 Do Q 30 0 10 10 re f",
    );
    let b = bake(&bytes);
    assert!(b.written[0].notes.contains(&PdfImageNote::Interleaved));
    assert_eq!(b.report.interleaved, 1);
    let reader = through_pvec(&b);
    assert_eq!(reader.raster_placements().unwrap()[0].band, Band::Underlay);
}

#[test]
fn the_image_count_is_limited_and_the_refusal_is_typed() {
    let bytes = pdf(
        doc(),
        Resources::images(vec![("Im1", Object::Stream(gray8(1, 1, &[7])))]),
        b"/Im1 Do /Im1 Do /Im1 Do",
    );
    let limits = PdfVectorLimits {
        max_images: 2,
        ..PdfVectorLimits::default()
    };
    let e = extract_page_paths_from_bytes(&bytes, 1, &limits).unwrap_err();
    assert_eq!(e, PdfVectorError::ImageLimitExceeded { limit: 2 });

    let limits = PdfVectorLimits {
        max_image_bytes: 1,
        ..PdfVectorLimits::default()
    };
    let bytes = pdf(
        doc(),
        Resources::images(vec![("Im1", Object::Stream(gray8(2, 2, &[1, 2, 3, 4])))]),
        b"/Im1 Do",
    );
    let e = extract_page_paths_from_bytes(&bytes, 1, &limits).unwrap_err();
    assert!(
        matches!(e, PdfVectorError::ImageBytesExceeded { limit: 1, .. }),
        "{e:?}"
    );
}

#[test]
fn an_inline_image_mask_still_fails_the_page_because_lopdf_wants_a_colour_space() {
    // The limit is the parser's, recorded in docs/pvec-raster-producer.md: an inline image with
    // no /CS (an inline image mask) does not tokenize, so the page is an error, not a silent skip.
    let mut content = b"BI /W 8 /H 1 /IM true /BPC 1 ID ".to_vec();
    content.push(0xAA);
    content.extend_from_slice(b" EI");
    let bytes = pdf(doc(), Resources::images(Vec::new()), &content);
    assert!(extract_page_paths_from_bytes(&bytes, 1, &PdfVectorLimits::default()).is_err());
}

// ---- close-out of #1332: colour spaces by name, typed refusals, 1-bit tiles ----

fn within(got: &[u8], want: &[u8], tol: i16) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!(
            (i16::from(*g) - i16::from(*w)).abs() <= tol,
            "{got:?} vs {want:?}"
        );
    }
}

#[test]
fn a_cal_rgb_image_goes_through_its_matrix_into_srgb_and_is_reported() {
    // Gamma 1 and the sRGB primaries with a D65 white: (1,1,1) is white, (0,0,0) black and
    // (1,0,0) is sRGB red, worked out by hand from the matrix rows.
    let cs = Object::Array(vec![
        name("CalRGB"),
        Object::Dictionary(dictionary! {
            "WhitePoint" => vec![Object::Real(0.9505), Object::Real(1.0), Object::Real(1.089)],
            "Gamma" => vec![Object::Real(1.0), Object::Real(1.0), Object::Real(1.0)],
            "Matrix" => vec![
                Object::Real(0.4124), Object::Real(0.2126), Object::Real(0.0193),
                Object::Real(0.3576), Object::Real(0.7152), Object::Real(0.1192),
                Object::Real(0.1805), Object::Real(0.0722), Object::Real(0.9505),
            ],
        }),
    ]);
    let s = image(
        3,
        1,
        dictionary! { "ColorSpace" => cs, "BitsPerComponent" => 8 },
        vec![255, 255, 255, 0, 0, 0, 255, 0, 0],
    );
    let b = bake(&one(s, "q 3 0 0 1 0 0 cm /Im1 Do Q"));
    let p = only(&b);
    let reader = through_pvec(&b);
    within(&rgb(&reader, 0, p), &[255, 255, 255, 0, 0, 0, 255, 0, 0], 2);
    assert!(
        b.written[0]
            .notes
            .contains(&PdfImageNote::ColourSpaceApproximated)
    );
    assert_eq!(b.report.colour_approximated, 1);
}

#[test]
fn a_lab_image_goes_through_the_cie_formulas_into_srgb_and_is_reported() {
    // L 100 and L 0 with a and b at the middle of the range: white and black. L byte 128 is
    // L* 50.2, Y = ((50.2 + 16) / 116)^3 = 0.1859, sRGB 0.4685 = 119.5.
    let cs = Object::Array(vec![
        name("Lab"),
        Object::Dictionary(dictionary! {
            "WhitePoint" => vec![Object::Real(0.9505), Object::Real(1.0), Object::Real(1.089)],
            "Range" => vec![(-128).into(), 127.into(), (-128).into(), 127.into()],
        }),
    ]);
    let s = image(
        3,
        1,
        dictionary! { "ColorSpace" => cs, "BitsPerComponent" => 8 },
        vec![255, 128, 128, 0, 128, 128, 128, 128, 128],
    );
    let b = bake(&one(s, "q 3 0 0 1 0 0 cm /Im1 Do Q"));
    let p = only(&b);
    let reader = through_pvec(&b);
    within(
        &rgb(&reader, 0, p),
        &[255, 255, 255, 0, 0, 0, 120, 120, 120],
        3,
    );
    assert!(
        b.written[0]
            .notes
            .contains(&PdfImageNote::ColourSpaceApproximated)
    );
}

#[test]
fn a_cal_gray_image_is_reported_approximated() {
    let cs = Object::Array(vec![
        name("CalGray"),
        Object::Dictionary(dictionary! {
            "WhitePoint" => vec![Object::Real(0.9505), Object::Real(1.0), Object::Real(1.089)],
            "Gamma" => Object::Real(1.0),
        }),
    ]);
    let s = image(
        2,
        1,
        dictionary! { "ColorSpace" => cs, "BitsPerComponent" => 8 },
        vec![0, 255],
    );
    let b = bake(&one(s, "q 2 0 0 1 0 0 cm /Im1 Do Q"));
    let p = only(&b);
    assert_eq!(luma(&through_pvec(&b), 0, p), vec![0, 255]);
    assert_eq!(b.report.colour_approximated, 1);
}

#[test]
fn a_cmyk_image_is_converted_and_reported_approximated() {
    let s = image(
        2,
        1,
        dictionary! { "ColorSpace" => "DeviceCMYK", "BitsPerComponent" => 8 },
        vec![0, 0, 0, 0, 0, 0, 0, 255],
    );
    let b = bake(&one(s, "q 2 0 0 1 0 0 cm /Im1 Do Q"));
    let p = only(&b);
    assert_eq!(rgb(&through_pvec(&b), 0, p), vec![255, 255, 255, 0, 0, 0]);
    assert_eq!(b.report.colour_approximated, 1);
}

#[test]
fn a_decode_array_inverts_a_gray_image() {
    let s = image(
        2,
        1,
        dictionary! {
            "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8,
            "Decode" => vec![1.into(), 0.into()],
        },
        vec![0, 255],
    );
    let b = bake(&one(s, "q 2 0 0 1 0 0 cm /Im1 Do Q"));
    let p = only(&b);
    assert_eq!(luma(&through_pvec(&b), 0, p), vec![255, 0]);
}

#[test]
fn ccitt_jbig2_devicen_and_pattern_are_each_refused_by_name_and_counted() {
    let cases: [(&str, Object, &str, &str); 4] = [
        (
            "CCITTFaxDecode",
            name("DeviceGray"),
            "filter-not-enabled:CCITTFaxDecode",
            "f",
        ),
        (
            "JBIG2Decode",
            name("DeviceGray"),
            "filter-not-enabled:JBIG2Decode",
            "f",
        ),
        (
            "",
            Object::Array(vec![
                name("DeviceN"),
                Object::Array(vec![name("A"), name("B")]),
                name("DeviceRGB"),
                Object::Dictionary(dictionary! { "FunctionType" => 2, "N" => 1 }),
            ]),
            "unsupported-colour-space:DeviceN",
            "c",
        ),
        ("", name("Pattern"), "unsupported-colour-space:Pattern", "c"),
    ];
    for (filter, cs, reason, kind) in cases {
        let mut extra = dictionary! { "ColorSpace" => cs, "BitsPerComponent" => 1 };
        if !filter.is_empty() {
            extra.set("Filter", name(filter));
        }
        let b = bake(&one(
            image(8, 1, extra, vec![0xAA]),
            "q 8 0 0 1 0 0 cm /Im1 Do Q",
        ));
        assert!(b.written.is_empty());
        assert_eq!(b.report.images, 1);
        assert_eq!(b.report.refused.get(reason), Some(&1), "{reason}");
        assert_eq!(b.refused[0].error.reason(), reason);
        match (&b.refused[0].error, kind) {
            (PdfImageError::FilterNotEnabled { filter: n, .. }, "f") => assert_eq!(n, filter),
            (PdfImageError::UnsupportedColourSpace(_), "c") => {}
            (e, _) => panic!("{e:?}"),
        }
    }
}

/// The bit depth byte of a PNG's `IHDR`.
fn png_depth(bytes: &[u8]) -> u8 {
    assert_eq!(&bytes[1..4], b"PNG");
    bytes[24]
}

fn full_tile(reader: &PvecReader, p: &RasterPyramid) -> Vec<u8> {
    reader
        .raster_tile(RasterId::new(0).unwrap(), u32::from(p.levels) - 1, 0, 0)
        .unwrap()
        .expect("a stored tile")
}

/// 10 by 2 at one bit: row 0 is 1011 0010 01, row 1 is 0100 1101 10.
const BILEVEL_10X2: [u8; 4] = [0b1011_0010, 0b0100_0000, 0b0100_1101, 0b1000_0000];
const BILEVEL_ROW0: [u8; 10] = [255, 0, 255, 255, 0, 0, 255, 0, 0, 255];
const BILEVEL_ROW1: [u8; 10] = [0, 255, 0, 0, 255, 255, 0, 255, 255, 0];

fn bilevel(extra: Dictionary) -> Stream {
    let mut d = dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 1 };
    for (k, v) in extra.iter() {
        d.set(k.clone(), v.clone());
    }
    image(10, 2, d, BILEVEL_10X2.to_vec())
}

use lopdf::Dictionary;

#[test]
fn a_one_bit_gray_image_is_written_as_one_bit_tiles_and_the_report_counts_it() {
    let b = bake(&one(
        bilevel(Dictionary::new()),
        "q 10 0 0 2 0 0 cm /Im1 Do Q",
    ));
    let p = only(&b);
    assert_eq!(p.bit_depth, 1);
    assert!(b.written[0].notes.contains(&PdfImageNote::Bilevel));
    assert_eq!(b.report.bilevel, 1);
    let reader = through_pvec(&b);
    let pl = &reader.raster_placements().unwrap()[0];
    assert_eq!((pl.bit_depth, pl.pixel_format), (1, Channels::Gray));
    assert_eq!(png_depth(&full_tile(&reader, p)), 1);
    let mut want = BILEVEL_ROW0.to_vec();
    want.extend_from_slice(&BILEVEL_ROW1);
    assert_eq!(luma(&reader, 0, p), want);
}

#[test]
fn a_one_bit_image_with_decode_one_zero_is_inverted_in_its_one_bit_tiles() {
    let b = bake(&one(
        bilevel(dictionary! { "Decode" => vec![1.into(), 0.into()] }),
        "q 10 0 0 2 0 0 cm /Im1 Do Q",
    ));
    let p = only(&b);
    assert_eq!(p.bit_depth, 1);
    let want: Vec<u8> = BILEVEL_ROW0
        .iter()
        .chain(&BILEVEL_ROW1)
        .map(|v| 255 - v)
        .collect();
    assert_eq!(luma(&through_pvec(&b), 0, p), want);
}

#[test]
fn a_one_bit_image_with_a_soft_mask_has_alpha_and_so_stays_eight_bit_rgba() {
    let mut d = doc();
    let sm = d.add_object(gray8(10, 2, &[255; 20]));
    let s = bilevel(dictionary! { "SMask" => Object::Reference(sm) });
    let bytes = pdf(
        d,
        Resources::images(vec![("Im1", Object::Stream(s))]),
        b"q 10 0 0 2 0 0 cm /Im1 Do Q",
    );
    let b = bake(&bytes);
    let p = only(&b);
    assert_eq!(p.bit_depth, 8);
    assert_eq!(b.report.bilevel, 0);
}

#[test]
fn a_clipped_one_bit_image_has_baked_alpha_and_stays_eight_bit() {
    let b = bake(&one(
        bilevel(Dictionary::new()),
        "q 0 0 m 5 0 l 0 2 l h W n 10 0 0 2 0 0 cm /Im1 Do Q",
    ));
    let p = only(&b);
    assert_eq!(p.bit_depth, 8);
    assert_eq!(b.report.bilevel, 0);
}

#[test]
fn an_eight_bit_gray_image_of_only_black_and_white_is_not_bilevel() {
    let b = bake(&one(gray8(2, 1, &[0, 255]), "q 2 0 0 1 0 0 cm /Im1 Do Q"));
    assert_eq!(only(&b).bit_depth, 8);
    assert_eq!(b.report.bilevel, 0);
}

#[test]
fn a_one_bit_image_under_jpeg_tiles_stays_eight_bit_gray() {
    let mut o = PdfImageOptions::default();
    o.tile_format = libviprs::TileFormat::Jpeg { quality: 90 };
    let page = page(&one(
        bilevel(Dictionary::new()),
        "q 10 0 0 2 0 0 cm /Im1 Do Q",
    ));
    let b = bake_page_images(&page, &o);
    assert_eq!(only(&b).bit_depth, 8);
    assert_eq!(b.report.bilevel, 0);
}
