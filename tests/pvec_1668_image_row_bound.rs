//! Issue #1668 (review): the image decoder bounds a predictor row by the inflated data, so a
//! hostile `/Columns`, `/Colors` or `/BitsPerComponent` is a typed refusal and never an allocation
//! that aborts. Needs `--features pvec-raster`.

#![cfg(feature = "pvec-raster")]

#[path = "common/pdf_images.rs"]
mod fixture;

use std::io::Write;

use fixture::*;
use libviprs::pvec::pdf_image::{PdfImageError, PdfImageOptions, bake_page_images};
use lopdf::{Object, dictionary};

fn flate(raw: &[u8]) -> Vec<u8> {
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    z.write_all(raw).unwrap();
    z.finish().unwrap()
}

fn hostile(columns: i64, colors: i64, bpc: i64) -> PdfImageError {
    let s = image(
        2,
        2,
        dictionary! {
            "ColorSpace" => "DeviceGray", "BitsPerComponent" => 8,
            "Filter" => vec![name("FlateDecode")],
            "DecodeParms" => vec![Object::Dictionary(dictionary! {
                "Predictor" => 15, "Colors" => colors, "BitsPerComponent" => bpc,
                "Columns" => columns,
            })],
        },
        flate(&[1u8, 10, 5, 2, 1, 2]),
    );
    let bytes = one(s, "q 2 0 0 2 0 0 cm /Im1 Do Q");
    let b = bake_page_images(&page(&bytes), &PdfImageOptions::default());
    assert!(b.written.is_empty(), "{:?}", b.written);
    assert_eq!(b.refused.len(), 1);
    b.refused.into_iter().next().unwrap().error
}

#[test]
fn a_columns_value_near_two_to_the_sixty_second_is_a_typed_refusal() {
    let e = hostile(1 << 62, 1, 8);
    assert!(matches!(e, PdfImageError::Decode(_)), "{e:?}");
}

#[test]
fn a_huge_columns_colors_and_bits_per_component_product_is_a_typed_refusal() {
    let e = hostile(1 << 31, 4, 16);
    assert!(matches!(e, PdfImageError::Decode(_)), "{e:?}");
}
