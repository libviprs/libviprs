//! Pins the compositing call surface required by the libviprs-tests
//! ported suite (libviprs-tests issue #55, the `test_composite` and
//! `test_composite_non_separable` call sites in
//! `tests/ported_conversion.rs`).
//!
//! Integration tests compile as an external crate, exactly the position
//! the ported tests are in, so this file proves the surface they call
//! compiles and behaves: `base.composite(&overlay, CompositeMode::...)`
//! with the mode names the ported suite constructs, composed with the
//! already-landed `add_const` / `bandjoin_const` / `getpoint` surface.
//! Behaviour depth is covered by the unit tests in `src/composite.rs`;
//! this file is the API contract.
//!
//! The ported `test_composite` fixture is `mask_ideal(100, 100, 0.5,
//! reject, optical)`, whose (0, 0) pixel is 0, making `colour`'s (0, 0)
//! pixel `[2, 3, 4]`. The fixture is reproduced with those exact corner
//! values so the ported expected values (51.8 / 52.8 / 53.8) stay
//! literal. The ported non-separable test additionally casts its inputs
//! to a float pixel format before compositing; that cast target arrives
//! with the float-format batch, so the non-separable pin runs on the
//! integer inputs the operation accepts today and keeps the ported
//! geometry assertions.

use libviprs::{CompositeMode, PixelFormat, Raster};

/// Stand-in for the ported `colour` fixture: a 100x100 Rgb8 image whose
/// (0, 0) pixel is `[2, 3, 4]` (the libvips mask_ideal optical-reject
/// origin value) with a radial texture elsewhere.
fn make_test_colour() -> Raster {
    let w = 100u32;
    let h = 100u32;
    let mut data = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 / w as f64;
            let dy = y as f64 / h as f64;
            let r = (dx * dx + dy * dy).sqrt();
            let v = if r > 0.5 { (r * 200.0).min(255.0) } else { 0.0 };
            let off = ((y * w + x) * 3) as usize;
            data[off] = ((v + 2.0) as u16).min(255) as u8;
            data[off + 1] = ((v * 2.0 + 3.0) as u16).min(255) as u8;
            data[off + 2] = ((v * 3.0 + 4.0) as u16).min(255) as u8;
        }
    }
    Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
}

/// The ported `test_composite` body: overlay `colour` with alpha 128
/// Over `colour + 100`, expecting the libvips values at (0, 0).
#[test]
fn ported_composite_call_site() {
    let colour = make_test_colour();
    let overlay = colour.bandjoin_const(128.0);
    let base = colour.add_const(100.0);

    let comp = base.composite(&overlay, CompositeMode::Over);
    let px = comp.getpoint(0, 0);
    assert!((px[0] - 51.8).abs() < 1.0, "{px:?}");
    assert!((px[1] - 52.8).abs() < 1.0, "{px:?}");
    assert!((px[2] - 53.8).abs() < 1.0, "{px:?}");
}

/// The ported `test_composite_non_separable` body (minus the
/// float-format cast; see the module docs): every non-separable mode
/// preserves geometry and produces a 4-channel result.
#[test]
fn ported_composite_non_separable_call_site() {
    let colour = make_test_colour();
    let base = colour.add_const(100.0).bandjoin_const(255.0);
    let overlay = colour.bandjoin_const(128.0);

    for mode in &[
        CompositeMode::Hue,
        CompositeMode::Saturation,
        CompositeMode::Colour,
        CompositeMode::Luminosity,
    ] {
        let comp = base.composite(&overlay, *mode);
        assert_eq!(comp.width(), base.width());
        assert_eq!(comp.height(), base.height());
        assert_eq!(comp.format().channels(), 4);
    }
}

/// `composite2` is the binary libvips name for the same call shape.
#[test]
fn composite2_call_site() {
    let colour = make_test_colour();
    let overlay = colour.bandjoin_const(128.0);
    let a = colour.composite(&overlay, CompositeMode::Over);
    let b = colour.composite2(&overlay, CompositeMode::Over);
    assert_eq!(a.data(), b.data());
}

/// Compile-position pin for every blend mode name the ported suites can
/// construct: the full Porter-Duff set, the PDF separable set, and the
/// PDF non-separable set.
#[test]
fn all_mode_names_construct() {
    let modes = [
        CompositeMode::Clear,
        CompositeMode::Source,
        CompositeMode::Over,
        CompositeMode::In,
        CompositeMode::Out,
        CompositeMode::Atop,
        CompositeMode::Dest,
        CompositeMode::DestOver,
        CompositeMode::DestIn,
        CompositeMode::DestOut,
        CompositeMode::DestAtop,
        CompositeMode::Xor,
        CompositeMode::Add,
        CompositeMode::Saturate,
        CompositeMode::Multiply,
        CompositeMode::Screen,
        CompositeMode::Overlay,
        CompositeMode::Darken,
        CompositeMode::Lighten,
        CompositeMode::ColourDodge,
        CompositeMode::ColourBurn,
        CompositeMode::HardLight,
        CompositeMode::SoftLight,
        CompositeMode::Difference,
        CompositeMode::Exclusion,
        CompositeMode::Hue,
        CompositeMode::Saturation,
        CompositeMode::Colour,
        CompositeMode::Luminosity,
    ];
    let base = Raster::zeroed(2, 2, PixelFormat::Rgba8).unwrap();
    let overlay = Raster::zeroed(2, 2, PixelFormat::Rgba8).unwrap();
    for mode in modes {
        let out = base.composite(&overlay, mode);
        assert_eq!(out.format().channels(), 4, "{mode:?}");
    }
}
