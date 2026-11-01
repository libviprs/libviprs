//! Issue #1364: `JPXDecode` images of a PDF page become PVec rasters, through the `jp2k` feature.
//!
//! The codestreams are written by `Raster::encode_jp2k` in each test (lossless, so the decoded
//! samples equal the ones written), and the expected pixels are worked out by hand. Every written
//! case is read back through PVec's reader. Needs `--features pvec-raster`; the decoding cases
//! also need `jp2k`, and the one case for the build without it needs the opposite. The corpus
//! cases need `LIBVIPRS_PDF_CANDIDATES` (the `pdf-candidates` checkout) and say so when it is not
//! set.

#![cfg(feature = "pvec-raster")]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::pdf_image::{PdfImageBake, PdfImageError, PdfImageOptions, bake_page_images};
use lopdf::{Dictionary, dictionary};

fn bake(bytes: &[u8]) -> PdfImageBake {
    bake_page_images(&page(bytes), &PdfImageOptions::default())
}

/// A one-image page whose image is a `JPXDecode` stream of `data`, `w` by `h` in the dictionary.
fn jpx(w: i64, h: i64, extra: Dictionary, data: Vec<u8>) -> Vec<u8> {
    let mut dict = dictionary! { "Filter" => "JPXDecode" };
    for (k, v) in extra.iter() {
        dict.set(k.clone(), v.clone());
    }
    one(image(w, h, dict, data), "q 20 0 0 20 100 200 cm /Im1 Do Q")
}

fn refusal(bytes: &[u8]) -> PdfImageError {
    let b = bake(bytes);
    assert!(b.written.is_empty(), "wrote {} images", b.written.len());
    b.refused.into_iter().next().expect("a refusal").error
}

#[cfg(feature = "jp2k")]
mod with_jp2k {
    use super::*;
    use libviprs::pixel::PixelFormat;
    use libviprs::pvec::pdf_image::{FilterError, PdfImageNote};
    use libviprs::raster::Raster;
    use libviprs::raster_pvec::container::write_rasters;
    use libviprs::raster_pvec::pyramid::RasterPyramid;
    use pvec_facade::unstable::RasterId;
    use pvec_facade::{Bounds2, Curve, EntityId, Manifest, PvecReader, PvecWriter};
    use pvec_geometry::{Point2, Polyline};
    use std::error::Error;
    use std::num::NonZeroU16;

    fn through_pvec(b: &PdfImageBake) -> PvecReader {
        let mut w = PvecWriter::new(Manifest {
            metres_per_unit: 1.0,
            bounds: Bounds2 {
                min: [0.0, 0.0],
                max: [612.0, 792.0],
            },
            capabilities: Vec::new(),
            producer_id: "libviprs-1364".into(),
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

    /// The samples of the one raster `b` wrote, `bpp` bytes a pixel, tiles stitched.
    fn samples(b: &PdfImageBake, bpp: usize) -> Vec<u8> {
        assert!(b.refused.is_empty(), "{:?}", b.refused);
        assert_eq!(b.written.len(), 1);
        let p: &RasterPyramid = &b.written[0].pyramid;
        let reader = through_pvec(b);
        let (w, h, edge) = (p.width as usize, p.height as usize, p.tile_edge as usize);
        let mut out = vec![0u8; w * h * bpp];
        for row in 0..h.div_ceil(edge) {
            for col in 0..w.div_ceil(edge) {
                let bytes = reader
                    .raster_tile(
                        RasterId::new(0).unwrap(),
                        u32::from(p.levels) - 1,
                        col as u32,
                        row as u32,
                    )
                    .unwrap()
                    .expect("a stored tile");
                let img = image::load_from_memory(&bytes).unwrap();
                let (raw, stride) = match bpp {
                    1 => (img.to_luma8().into_raw(), img.width() as usize),
                    _ => (img.to_rgb8().into_raw(), img.width() as usize),
                };
                for y in 0..edge.min(h - row * edge) {
                    let n = edge.min(w - col * edge);
                    let at = ((row * edge + y) * w + col * edge) * bpp;
                    let from = y * stride * bpp;
                    out[at..at + n * bpp].copy_from_slice(&raw[from..from + n * bpp]);
                }
            }
        }
        out
    }

    fn codestream(w: u32, h: u32, format: PixelFormat, data: Vec<u8>) -> Vec<u8> {
        Raster::new(w, h, format, data)
            .unwrap()
            .encode_jp2k(Default::default())
            .expect("a JPEG 2000 encoder")
    }

    fn multi8(n: u16) -> PixelFormat {
        PixelFormat::Multi8(NonZeroU16::new(n).unwrap())
    }

    #[test]
    fn an_8_bit_gray_codestream_is_written_as_gray() {
        let px = vec![0x00, 0x40, 0x80, 0xFF, 0x10, 0x20, 0x30, 0x40];
        let cs = codestream(4, 2, PixelFormat::Gray8, px.clone());
        let b = bake(&jpx(
            4,
            2,
            dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8 },
            cs,
        ));
        assert_eq!(samples(&b, 1), px);
        assert!(b.written[0].notes.contains(&PdfImageNote::ReEncodedLossy));
        assert_eq!(b.report.written_by_filter.get("JPXDecode"), Some(&1));
    }

    #[test]
    fn an_8_bit_rgb_codestream_is_written_as_rgb() {
        let px: Vec<u8> = (0..12).map(|i| i * 20).collect();
        let cs = codestream(2, 2, PixelFormat::Rgb8, px.clone());
        let b = bake(&jpx(
            2,
            2,
            dictionary! { "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8 },
            cs,
        ));
        assert_eq!(samples(&b, 3), px);
    }

    #[test]
    fn a_cmyk_codestream_goes_through_the_bakes_formula_and_is_approximated() {
        // White, then black by the K ink alone: the formula of the unfiltered CMYK case.
        let cs = codestream(2, 1, multi8(4), vec![0, 0, 0, 0, 0, 0, 0, 255]);
        let b = bake(&jpx(
            2,
            1,
            dictionary! { "ColorSpace" => "DeviceCMYK", "BitsPerComponent" => 8 },
            cs,
        ));
        assert_eq!(samples(&b, 3), vec![255, 255, 255, 0, 0, 0]);
        assert_eq!(b.report.colour_approximated, 1);
        assert!(
            b.written[0]
                .notes
                .contains(&PdfImageNote::ColourSpaceApproximated)
        );
    }

    #[test]
    fn no_color_space_in_the_dictionary_lets_the_codestream_decide() {
        let px: Vec<u8> = (0..12).map(|i| 255 - i * 10).collect();
        let cs = codestream(2, 2, PixelFormat::Rgb8, px.clone());
        let b = bake(&jpx(2, 2, dictionary! {}, cs));
        assert_eq!(samples(&b, 3), px);
        let g = codestream(2, 2, PixelFormat::Gray8, vec![1, 2, 3, 4]);
        let b = bake(&jpx(2, 2, dictionary! {}, g));
        assert_eq!(samples(&b, 1), vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_dictionary_space_that_disagrees_in_component_count_loses() {
        let px: Vec<u8> = (0..12).map(|i| i * 7).collect();
        let cs = codestream(2, 2, PixelFormat::Rgb8, px.clone());
        let b = bake(&jpx(
            2,
            2,
            dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8 },
            cs,
        ));
        assert_eq!(samples(&b, 3), px);
    }

    #[test]
    fn decode_is_ignored_for_a_non_indexed_space() {
        let px = vec![0, 10, 200, 255];
        let cs = codestream(2, 2, PixelFormat::Gray8, px.clone());
        let b = bake(&jpx(
            2,
            2,
            dictionary! {
                "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8,
                "Decode" => vec![1.into(), 0.into()],
            },
            cs,
        ));
        assert_eq!(samples(&b, 1), px);
    }

    #[test]
    fn sixteen_bit_components_are_reduced_and_say_so() {
        // Two RGB pixels, high bytes 0xFF 0x00 0x80 and 0x12 0xFF 0x00.
        let words: [u16; 6] = [0xFFFF, 0x0000, 0x8000, 0x1234, 0xFFFF, 0x0001];
        let data: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        let cs = codestream(2, 1, PixelFormat::Rgb16, data);
        let b = bake(&jpx(
            2,
            1,
            dictionary! { "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 16 },
            cs,
        ));
        assert_eq!(samples(&b, 3), vec![0xFF, 0x00, 0x80, 0x12, 0xFF, 0x00]);
        assert!(b.written[0].notes.contains(&PdfImageNote::BitDepthReduced));
        assert_eq!(b.report.bit_depth_reduced, 1);
    }

    #[test]
    fn a_codestream_whose_size_disagrees_with_width_and_height_is_refused() {
        let cs = codestream(4, 2, PixelFormat::Gray8, vec![0; 8]);
        let e = refusal(&jpx(
            3,
            2,
            dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8 },
            cs,
        ));
        assert_eq!(e.reason(), "decode-failed");
    }

    #[test]
    fn smask_in_data_is_refused_by_name() {
        let cs = codestream(2, 2, PixelFormat::Gray8, vec![0; 4]);
        let e = refusal(&jpx(
            2,
            2,
            dictionary! {
                "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8, "SMaskInData" => 1,
            },
            cs,
        ));
        assert_eq!(e.reason(), "decode-failed");
        assert!(
            matches!(
                e.source().and_then(|s| s.downcast_ref::<FilterError>()),
                Some(FilterError::SMaskInData(1))
            ),
            "{e:?}"
        );
    }

    #[test]
    fn smask_in_data_zero_is_the_default_and_decodes() {
        let cs = codestream(2, 2, PixelFormat::Gray8, vec![9, 8, 7, 6]);
        let b = bake(&jpx(
            2,
            2,
            dictionary! {
                "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8, "SMaskInData" => 0,
            },
            cs,
        ));
        assert_eq!(samples(&b, 1), vec![9, 8, 7, 6]);
    }

    #[test]
    fn a_codestream_with_alpha_loses_it_when_smask_in_data_is_zero() {
        let rgba: Vec<u8> = vec![10, 20, 30, 255, 40, 50, 60, 0];
        let cs = codestream(2, 1, PixelFormat::Rgba8, rgba);
        let b = bake(&jpx(
            2,
            1,
            dictionary! { "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8 },
            cs,
        ));
        assert_eq!(samples(&b, 3), vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn garbage_is_a_refusal_not_a_panic() {
        let e = refusal(&jpx(
            2,
            2,
            dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8 },
            vec![1, 2, 3, 4, 5, 6, 7, 8],
        ));
        assert_eq!(e.reason(), "decode-failed");
    }

    /// Hoskins Creek: one page, 23 JPX images, none with a `/ColorSpace` of its own that the
    /// codestream does not already say.
    #[test]
    #[cfg_attr(miri, ignore)] // reads a file from outside the repo
    fn the_hoskins_creek_sheet_writes_23_images() {
        let Some(bytes) = corpus("arch/usace_norfolk_hoskins_creek_dredging_plans.pdf") else {
            return;
        };
        let page = libviprs::pvec::pdf_vector::extract_page_paths_from_bytes(
            &bytes,
            1,
            &libviprs::pvec::pdf_vector::PdfVectorLimits::default(),
        )
        .unwrap();
        let b = bake_page_images(&page, &PdfImageOptions::default());
        assert_eq!(b.report.written_by_filter.get("JPXDecode"), Some(&23));
        assert!(
            !b.report.refused.keys().any(|k| k.contains("JPXDecode")),
            "{:?}",
            b.report.refused
        );
    }

    /// tm11b3: 20 pages, four JPX images in all, CMYK among them.
    #[test]
    #[cfg_attr(miri, ignore)] // reads a file from outside the repo
    fn the_tm11b3_report_writes_4_images_with_cmyk_approximated() {
        let Some(bytes) = corpus("us/tm11b3.pdf") else {
            return;
        };
        let (mut written, mut approximated) = (0, 0);
        for n in 1..=20 {
            let page = libviprs::pvec::pdf_vector::extract_page_paths_from_bytes(
                &bytes,
                n,
                &libviprs::pvec::pdf_vector::PdfVectorLimits::default(),
            )
            .unwrap();
            let b = bake_page_images(&page, &PdfImageOptions::default());
            assert!(
                !b.report.refused.keys().any(|k| k.contains("JPXDecode")),
                "page {n}: {:?}",
                b.report.refused
            );
            written += b.report.written_by_filter.get("JPXDecode").unwrap_or(&0);
            approximated += b.report.colour_approximated;
        }
        assert_eq!(written, 4);
        assert!(approximated >= 1, "no CMYK image was approximated");
    }

    fn corpus(rel: &str) -> Option<Vec<u8>> {
        let Some(dir) = std::env::var_os("LIBVIPRS_PDF_CANDIDATES") else {
            eprintln!("LIBVIPRS_PDF_CANDIDATES is not set: the corpus files are not in the repo");
            return None;
        };
        Some(std::fs::read(std::path::Path::new(&dir).join(rel)).expect("the corpus file"))
    }
}

#[cfg(not(feature = "jp2k"))]
#[test]
fn without_the_feature_the_reason_is_filter_not_enabled_and_names_the_feature() {
    let e = refusal(&jpx(
        2,
        2,
        dictionary! { "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8 },
        vec![0; 16],
    ));
    assert_eq!(e.reason(), "filter-not-enabled:JPXDecode");
    assert!(
        matches!(
            &e,
            PdfImageError::FilterNotEnabled {
                feature: "jp2k",
                ..
            }
        ),
        "{e:?}"
    );
}
