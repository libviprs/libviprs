//! Issue #1732: a PDF image whose stream or soft mask goes through `ASCII85Decode` (or
//! `ASCIIHexDecode`) is refused, so the whole picture is missing from the PVec.
//!
//! AutoCAD's plot driver writes the coloured regions of a site plan as `/DeviceRGB` images with an
//! `/SMask` whose `/Filter` is `[/FlateDecode /ASCII85Decode]`. At main every one of those images
//! is refused as `unsupported-filter:ASCII85Decode`, and the page converts with black outlines only.
//! Each test builds a tiny PDF in memory with one construct, so the pixels are known by hand. These
//! tests are red until the ASCII stages are read in the image filter chain.

#![cfg(feature = "pvec-write")]

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::pdf_image::{PdfImageBake, PdfImageOptions, bake_page_images};
use libviprs::pvec::pdf_page_to_pvec_report;
use lopdf::{Object, dictionary};
use std::io::Write;

/// ASCII85 with the `~>` terminator, no `z` shortcut.
fn ascii85(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for chunk in data.chunks(4) {
        let mut word = [0u8; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        let mut value = u32::from_be_bytes(word);
        let mut digits = [0u8; 5];
        for digit in digits.iter_mut().rev() {
            *digit = (value % 85) as u8 + 33;
            value /= 85;
        }
        out.extend_from_slice(&digits[..chunk.len() + 1]);
    }
    out.extend_from_slice(b"~>");
    out
}

fn ascii_hex(data: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = data
        .iter()
        .flat_map(|b| format!("{b:02X}").into_bytes())
        .collect();
    out.push(b'>');
    out
}

fn zlib(data: &[u8]) -> Vec<u8> {
    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    e.write_all(data).unwrap();
    e.finish().unwrap()
}

fn filters(names: &[&str]) -> Object {
    Object::Array(names.iter().map(|n| name(n)).collect())
}

/// A `w` by `h` image of `colorspace` whose stored bytes are `stored`, read back through `chain`.
fn encoded(w: i64, h: i64, colorspace: &str, chain: &[&str], stored: Vec<u8>) -> lopdf::Stream {
    image(
        w,
        h,
        dictionary! {
            "ColorSpace" => colorspace, "BitsPerComponent" => 8, "Filter" => filters(chain),
        },
        stored,
    )
}

fn page_with(image: lopdf::Stream) -> Vec<u8> {
    one(image, "q 2 0 0 2 0 0 cm /Im1 Do Q")
}

/// The page also strokes one rule, so it has a path and the conversion does not stop at `NoPaths`.
fn with_mask(mask: lopdf::Stream, rgb: lopdf::Stream) -> Vec<u8> {
    let mut d = doc();
    let sm = d.add_object(mask);
    let mut s = rgb;
    s.dict.set("SMask", Object::Reference(sm));
    pdf(
        d,
        Resources::images(vec![("Im1", Object::Stream(s))]),
        b"10 10 m 100 10 l S q 2 0 0 2 0 0 cm /Im1 Do Q",
    )
}

const PINK: [u8; 12] = [250, 200, 220, 250, 200, 220, 250, 200, 220, 250, 200, 220];
const ALPHA: [u8; 4] = [0, 85, 170, 255];

fn bake(bytes: &[u8]) -> PdfImageBake {
    bake_page_images(&page(bytes), &PdfImageOptions::default())
}

fn assert_written(b: &PdfImageBake) {
    assert!(
        b.refused.is_empty() && b.report.refused.is_empty(),
        "the image was refused: {:?}",
        b.report.refused
    );
    assert_eq!(b.written.len(), 1, "report: {:?}", b.report);
    assert_eq!(b.report.written, 1);
}

#[test]
fn issue_1732_a_soft_mask_through_flate_then_ascii85_is_read() {
    // The chain of the AutoCAD plot: the stored bytes are Flate, and Flate gives ASCII85 text.
    let mask = encoded(
        2,
        2,
        "DeviceGray",
        &["FlateDecode", "ASCII85Decode"],
        zlib(&ascii85(&ALPHA)),
    );
    let rgb = encoded(2, 2, "DeviceRGB", &["FlateDecode"], zlib(&PINK));
    assert_written(&bake(&with_mask(mask, rgb)));
}

#[test]
fn issue_1732_a_soft_mask_through_ascii85_alone_is_read() {
    let mask = encoded(2, 2, "DeviceGray", &["ASCII85Decode"], ascii85(&ALPHA));
    let rgb = encoded(2, 2, "DeviceRGB", &["FlateDecode"], zlib(&PINK));
    assert_written(&bake(&with_mask(mask, rgb)));
}

#[test]
fn issue_1732_a_soft_mask_through_asciihex_is_read() {
    let mask = encoded(2, 2, "DeviceGray", &["ASCIIHexDecode"], ascii_hex(&ALPHA));
    let rgb = encoded(2, 2, "DeviceRGB", &["FlateDecode"], zlib(&PINK));
    assert_written(&bake(&with_mask(mask, rgb)));
}

#[test]
fn issue_1732_an_image_through_flate_then_ascii85_is_read() {
    let rgb = encoded(
        2,
        2,
        "DeviceRGB",
        &["FlateDecode", "ASCII85Decode"],
        zlib(&ascii85(&PINK)),
    );
    assert_written(&bake(&page_with(rgb)));
}

#[test]
fn issue_1732_an_image_through_ascii85_alone_is_read() {
    let rgb = encoded(2, 2, "DeviceRGB", &["ASCII85Decode"], ascii85(&PINK));
    assert_written(&bake(&page_with(rgb)));
}

#[test]
fn issue_1732_an_image_through_asciihex_alone_is_read() {
    let rgb = encoded(2, 2, "DeviceRGB", &["ASCIIHexDecode"], ascii_hex(&PINK));
    assert_written(&bake(&page_with(rgb)));
}

#[test]
fn issue_1732_the_converted_page_carries_the_coloured_image_and_says_it_is_baked() {
    // The whole path the server takes: path extraction, then pdf_page_to_pvec_report with the bake.
    let mask = encoded(
        2,
        2,
        "DeviceGray",
        &["FlateDecode", "ASCII85Decode"],
        zlib(&ascii85(&ALPHA)),
    );
    let rgb = encoded(2, 2, "DeviceRGB", &["FlateDecode"], zlib(&PINK));
    let bytes = with_mask(mask, rgb);
    let (_, report) = pdf_page_to_pvec_report(&page(&bytes), "issue-1732", "0", true).unwrap();
    assert!(report.refused.is_empty(), "refused: {:?}", report.refused);
    assert_eq!(report.written, 1, "report: {report:?}");
    assert!(report.baked(), "report: {report:?}");
}
