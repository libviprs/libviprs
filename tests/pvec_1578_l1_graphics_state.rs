//! Issue #1578, L1: a painted PDF path reports its stroke colour, its fill and stroke alpha and
//! the scale of the CTM it was painted under, so a converter can write a stroke colour, an alpha
//! byte and a width in source units.

#[path = "common/pdf_images.rs"]
mod fixture;

use fixture::*;
use libviprs::pvec::pdf_vector::PdfPathEntity;
use lopdf::{Object, dictionary};

fn entities(content: &str) -> Vec<PdfPathEntity> {
    let bytes = pdf(doc(), Resources::images(Vec::new()), content.as_bytes());
    page(&bytes).entities
}

fn with_alpha(content: &str, ca: f32, stroke_ca: f32) -> Vec<PdfPathEntity> {
    let gs = Object::Dictionary(dictionary! {"ca" => ca, "CA" => stroke_ca});
    let resources = Resources {
        xobjects: Vec::new(),
        gstates: vec![("GS1", gs)],
    };
    page(&pdf(doc(), resources, content.as_bytes())).entities
}

#[test]
fn a_stroke_reports_the_stroke_colour_not_the_fill() {
    let e = entities("1 0 0 RG 0 0 1 rg 0 0 10 10 re B");
    assert_eq!(e[0].stroke_colour, [255, 0, 0]);
    assert_eq!(e[0].fill_colour, [0, 0, 255]);
}

#[test]
fn a_stroke_with_no_colour_set_is_black_and_a_gray_stroke_is_gray() {
    assert_eq!(entities("0 0 10 10 re S")[0].stroke_colour, [0, 0, 0]);
    assert_eq!(entities("0.5 G 0 0 10 10 re S")[0].stroke_colour, [128; 3]);
}

#[test]
fn a_cmyk_stroke_is_converted_and_a_fill_operator_leaves_it_alone() {
    let e = entities("0 0 0 0 K 1 0 0 rg 0 0 10 10 re S");
    assert_eq!(e[0].stroke_colour, [255, 255, 255]);
    let e = entities("0 1 0 RG 0 0 0 1 k 0 0 10 10 re S");
    assert_eq!(e[0].stroke_colour, [0, 255, 0]);
}

#[test]
fn big_q_restores_the_stroke_colour() {
    let e = entities("q 1 0 0 RG 0 0 10 10 re S Q 20 0 10 10 re S");
    assert_eq!(e[0].stroke_colour, [255, 0, 0]);
    assert_eq!(e[1].stroke_colour, [0, 0, 0]);
}

#[test]
fn sc_and_scn_set_the_stroke_in_their_upper_case_forms() {
    let e = entities("/DeviceRGB CS 0 0 1 SC 0 0 10 10 re S");
    assert_eq!(e[0].stroke_colour, [0, 0, 255]);
    let e = entities("0 1 1 SCN 0 0 10 10 re S");
    assert_eq!(e[0].stroke_colour, [0, 255, 255]);
}

#[test]
fn alpha_defaults_to_opaque() {
    let e = entities("0 0 10 10 re B");
    assert_eq!((e[0].fill_alpha, e[0].stroke_alpha), (255, 255));
}

#[test]
fn ca_and_big_ca_set_the_fill_and_stroke_alpha_separately() {
    let e = with_alpha("/GS1 gs 0 0 10 10 re B", 0.5, 0.25);
    assert_eq!((e[0].fill_alpha, e[0].stroke_alpha), (128, 64));
}

#[test]
fn alpha_is_scoped_by_q_and_clamped() {
    let e = with_alpha("q /GS1 gs 0 0 10 10 re f Q 20 0 10 10 re f", 0.5, 2.0);
    assert_eq!(e[0].fill_alpha, 128);
    assert_eq!(e[0].stroke_alpha, 255);
    assert_eq!(e[1].fill_alpha, 255);
}

#[test]
fn the_ctm_scale_is_the_square_root_of_the_determinant() {
    // The page base is a plain point scale, so `cm` alone moves it.
    let e = entities("0 0 10 10 re S 2 0 0 2 0 0 cm 0 0 10 10 re S");
    let base = e[0].ctm_scale;
    assert!((base - 1.0).abs() < 1e-9, "{base}");
    assert!((e[1].ctm_scale - 2.0).abs() < 1e-9);
    assert!((e[1].ctm_anisotropy - 1.0).abs() < 1e-9);
}

#[test]
fn a_rotation_keeps_the_scale_and_a_stretch_reports_its_anisotropy() {
    let e = entities("0 3 -3 0 0 0 cm 0 0 10 10 re S");
    assert!((e[0].ctm_scale - 3.0).abs() < 1e-9);
    assert!((e[0].ctm_anisotropy - 1.0).abs() < 1e-9);
    let e = entities("4 0 0 1 0 0 cm 0 0 10 10 re S");
    assert!((e[0].ctm_scale - 2.0).abs() < 1e-9);
    assert!((e[0].ctm_anisotropy - 4.0).abs() < 1e-9);
}
