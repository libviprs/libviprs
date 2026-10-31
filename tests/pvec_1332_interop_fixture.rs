//! Issue #1332: the `.pvec` files the interop check against maplibre-pvec's raster harness reads.
//!
//! Two PDFs, each a 64 by 48 gray "scan" (a horizontal ramp, `40 + 3 * x`) painted 100 by 75
//! points at the origin, and one stroked diagonal across it. In `scan_under_linework` the image
//! is painted first (band 0, an underlay) and in `scan_over_linework` after the stroke (band 1).
//! Page space goes into source space by `+ (25, 10)`, which centres the sheet under the harness
//! camera. The test always checks the bake; with `PVEC_INTEROP_OUT=<dir>` it also writes
//! `<name>.pvec`, which is how `tests/fixtures/pvec-raster/libviprs/` in libviprs/maplibre-pvec
//! was made. With `PVEC_INTEROP_THREE_IMAGES=<path to three_images.pdf>` (the committed fixture of
//! libviprs-tests) it writes `three_images.pvec` too. Needs `--features pvec-raster`.

#![cfg(feature = "pvec-raster")]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::pdf_image::{PdfImageOptions, bake_page_images};
use libviprs::pvec::pdf_vector::{PdfPathSegment, PdfVectorLimits, extract_page_paths_from_bytes};
use libviprs::raster_pvec::container::write_rasters;
use lopdf::Object;
use pvec_facade::unstable::Band;
use pvec_facade::{Bounds2, Curve, EntityId, Manifest, PvecReader, PvecWriter};
use pvec_geometry::{Point2, Polyline};

const SHIFT: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 25.0, 10.0];

fn scan() -> lopdf::Stream {
    let mut px = Vec::new();
    for _ in 0..48 {
        for x in 0..64u32 {
            px.push((40 + 3 * x) as u8);
        }
    }
    gray8(64, 48, &px)
}

fn sheet(image_first: bool) -> Vec<u8> {
    let image = "q 100 0 0 75 0 0 cm /Scan Do Q ";
    let line = "1 0 0 RG 2 w 0 0 m 100 75 l S ";
    let content = if image_first {
        format!("{image}{line}")
    } else {
        format!("{line}{image}")
    };
    pdf(
        doc(),
        Resources::images(vec![("Scan", Object::Stream(scan()))]),
        content.as_bytes(),
    )
}

fn writer() -> PvecWriter {
    PvecWriter::new(Manifest {
        metres_per_unit: 1.0,
        bounds: Bounds2 {
            min: [0.0, 0.0],
            max: [700.0, 900.0],
        },
        capabilities: Vec::new(),
        producer_id: "libviprs-1332".into(),
        producer_version: "1".into(),
    })
    .unwrap()
}

/// The page's paths as polylines in source space, then the baked rasters, as one `.pvec`.
fn convert(bytes: &[u8]) -> (Vec<u8>, Vec<Band>) {
    let page = extract_page_paths_from_bytes(bytes, 1, &PdfVectorLimits::default()).unwrap();
    let mut options = PdfImageOptions::default();
    options.page_to_source = SHIFT;
    let bake = bake_page_images(&page, &options);
    assert!(bake.refused.is_empty(), "{:?}", bake.refused);
    let mut w = writer();
    let mut next = 0;
    for e in &page.entities {
        let pts: Vec<Point2> = e
            .segments
            .iter()
            .filter_map(|s| match s {
                PdfPathSegment::MoveTo(p) | PdfPathSegment::LineTo(p) => Some(Point2 {
                    x: p[0] + SHIFT[4],
                    y: p[1] + SHIFT[5],
                }),
                _ => None,
            })
            .collect();
        if pts.len() >= 2 {
            w.push(
                EntityId::new(next).unwrap(),
                &Curve::Polyline(Polyline::new(pts).unwrap()),
            )
            .unwrap();
            next += 1;
        }
    }
    if next == 0 {
        // A file needs one entity (WRT-01): a page with only images gets its outline.
        let ring = [
            (0.0, 0.0),
            (page.width, 0.0),
            (page.width, page.height),
            (0.0, 0.0),
        ]
        .map(|(x, y)| Point2 {
            x: x + SHIFT[4],
            y: y + SHIFT[5],
        });
        w.push(
            EntityId::new(0).unwrap(),
            &Curve::Polyline(Polyline::new(ring.to_vec()).unwrap()),
        )
        .unwrap();
    }
    write_rasters(&mut w, bake.written.iter().map(|b| &b.pyramid)).unwrap();
    let bytes = w.finish().expect("finish");
    let bands = PvecReader::open(bytes.clone())
        .unwrap()
        .raster_placements()
        .unwrap()
        .iter()
        .map(|p| p.band)
        .collect();
    (bytes, bands)
}

#[test]
fn a_scan_under_linework_and_one_over_it_convert_with_their_bands() {
    let (under, b0) = convert(&sheet(true));
    let (over, b1) = convert(&sheet(false));
    assert_eq!(b0, vec![Band::Underlay]);
    assert_eq!(b1, vec![Band::Overlay]);
    if let Ok(dir) = std::env::var("PVEC_INTEROP_OUT") {
        std::fs::write(format!("{dir}/scan_under_linework.pvec"), &under).unwrap();
        std::fs::write(format!("{dir}/scan_over_linework.pvec"), &over).unwrap();
        if let Ok(three) = std::env::var("PVEC_INTEROP_THREE_IMAGES") {
            let (bytes, bands) = convert(&std::fs::read(three).unwrap());
            assert_eq!(bands.len(), 3);
            std::fs::write(format!("{dir}/three_images.pvec"), bytes).unwrap();
        }
    }
}
