//! Pins the extract / crop / geometry-placement call surface required by
//! the libviprs-tests ported suite (libviprs-tests issue #55,
//! `tests/ported_conversion.rs`, plus the `insert` call sites in
//! `ported_arithmetic.rs`).
//!
//! Integration tests compile as an external crate, exactly the position the
//! ported tests are in, so this file proves the surface they call compiles
//! and behaves: method names, argument types (including `&str` compass
//! directions), enum variants, and tuple return types. Behavior depth is
//! covered by the unit tests in `src/extract.rs`; this file is the API
//! contract.
//!
//! Where a ported test's setup uses a fixture decode (`sample.jpg`,
//! `rgba.png`) or an operation from a later batch, the setup is reproduced
//! with direct `Raster` construction and the extract expressions are kept
//! literal. The fixture-derived attention coordinates the ported suite
//! asserts (199/234 and 20/124) are covered by the module's deferral note,
//! not here.

use libviprs::{Extend, PixelFormat, Raster, SmartcropInteresting};

/// The ported `make_test_mono`: a 100x100 Gray8 band-reject ring image.
fn make_test_mono() -> Raster {
    let w = 100u32;
    let h = 100u32;
    let mut data = vec![0u8; (w * h) as usize];
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;
    for y in 0..h {
        for x in 0..w {
            let dx = (x as f64 - cx) / cx;
            let dy = (y as f64 - cy) / cy;
            let r = (dx * dx + dy * dy).sqrt();
            let v = if r > 0.5 {
                (r * 200.0).min(255.0) as u8
            } else {
                0
            };
            data[(y * w + x) as usize] = v;
        }
    }
    Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
}

/// The ported `make_test_colour`: `mono * [1, 2, 3] + [2, 3, 4]` as Rgb8.
fn make_test_colour() -> Raster {
    let mono = make_test_mono();
    let w = mono.width();
    let h = mono.height();
    let md = mono.data();
    let mut data = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
        let v = md[i] as u16;
        data[i * 3] = (v + 2).min(255) as u8;
        data[i * 3 + 1] = (v * 2 + 3).min(255) as u8;
        data[i * 3 + 2] = (v * 3 + 4).min(255) as u8;
    }
    Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
}

/// A 100x150 Rgba8 image with an opaque textured band, for the smartcrop
/// RGBA call sites (the ported tests decode `rgba.png`).
fn make_test_rgba() -> Raster {
    let w = 100u32;
    let h = 150u32;
    let mut data = vec![0u8; (w * h * 4) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = (y * w as usize + x) * 4;
            data[i] = ((x * 5) % 256) as u8;
            data[i + 1] = ((y * 3) % 256) as u8;
            data[i + 2] = ((x + y) % 256) as u8;
            data[i + 3] = 255;
        }
    }
    Raster::new(w, h, PixelFormat::Rgba8, data).unwrap()
}

/// The ported `test_embed` call sites (`ported_conversion.rs`).
#[test]
fn ported_surface_embed() {
    let colour = make_test_colour();
    let w = colour.width();
    let h = colour.height();

    // Black extend
    let im = colour.embed(20, 20, w + 40, h + 40, Extend::Black, None);
    assert_eq!(im.width(), w + 40);
    assert_eq!(im.height(), h + 40);
    let px = im.getpoint(10, 10);
    assert!(px[0].abs() < 1.0 && px[1].abs() < 1.0 && px[2].abs() < 1.0);
    let px = im.getpoint(30, 30);
    let orig = colour.getpoint(10, 10);
    for (a, b) in px.iter().zip(orig.iter()) {
        assert!((a - b).abs() < 1.0);
    }

    // White extend
    let im = colour.embed(20, 20, w + 40, h + 40, Extend::White, None);
    let px = im.getpoint(10, 10);
    assert!((px[0] - 255.0).abs() < 1.0);

    // Background extend
    let im = colour.embed(
        20,
        20,
        w + 40,
        h + 40,
        Extend::Background,
        Some(&[7.0, 8.0, 9.0]),
    );
    let px = im.getpoint(10, 10);
    assert!((px[0] - 7.0).abs() < 1.0);
    assert!((px[1] - 8.0).abs() < 1.0);
    assert!((px[2] - 9.0).abs() < 1.0);

    // Copy extend
    let im = colour.embed(20, 20, w + 40, h + 40, Extend::Copy, None);
    let px = im.getpoint(10, 10);
    let corner = colour.getpoint(0, 0);
    assert!((px[0] - corner[0]).abs() < 1.0);
}

/// The ported `test_gravity` call sites: string directions on a 1x1 pixel.
#[test]
fn ported_surface_gravity() {
    let im = Raster::new(1, 1, PixelFormat::Gray8, vec![255u8]).unwrap();

    let positions: &[(&str, u32, u32)] = &[
        ("centre", 1, 1),
        ("north", 1, 0),
        ("south", 1, 2),
        ("east", 2, 1),
        ("west", 0, 1),
        ("north-east", 2, 0),
        ("south-east", 2, 2),
        ("south-west", 0, 2),
        ("north-west", 0, 0),
    ];

    for &(direction, x, y) in positions {
        let im2 = im.gravity(direction, 3, 3);
        let px = im2.getpoint(x, y);
        assert!(
            (px[0] - 255.0).abs() < 1.0,
            "gravity({direction}): pixel at ({x},{y}) should be 255"
        );
        assert!(
            (im2.avg() - 255.0 / 9.0).abs() < 1.0,
            "gravity({direction}): avg should be ~28.3"
        );
    }
}

/// The ported `test_extract` / `test_crop` call sites.
#[test]
fn ported_surface_extract_area_and_crop() {
    let colour = make_test_colour();

    let sub = colour.extract_area(25, 25, 10, 10);
    assert_eq!(sub.width(), 10);
    assert_eq!(sub.height(), 10);
    let px = sub.getpoint(5, 5);
    let orig = colour.getpoint(30, 30);
    assert!((px[0] - orig[0]).abs() < 1.0);
    assert!((px[1] - orig[1]).abs() < 1.0);

    let sub = colour.crop(25, 25, 10, 10);
    let px = sub.getpoint(5, 5);
    assert!((px[0] - orig[0]).abs() < 1.0);

    // The rot tests' setup crop (rot itself is a later batch).
    let test = colour.crop(0, 0, 51, 51);
    assert_eq!(test.width(), 51);
    assert_eq!(test.height(), 51);
}

/// The ported `test_smartcrop` call site (fixture decode reproduced with a
/// synthetic raster).
#[test]
fn ported_surface_smartcrop() {
    let im = make_test_colour().replicate(3, 3);
    let result = im.smartcrop(100, 100, SmartcropInteresting::Entropy);
    assert_eq!(result.width(), 100);
    assert_eq!(result.height(), 100);
}

/// The ported `test_smartcrop` attention call site with coordinates.
#[test]
fn ported_surface_smartcrop_attention() {
    let im = make_test_colour().replicate(3, 3);
    let (result, attention_x, attention_y) =
        im.smartcrop_with_coords(100, 100, SmartcropInteresting::Attention);
    assert_eq!(result.width(), 100);
    assert_eq!(result.height(), 100);
    assert!(attention_x >= 0 && attention_x < im.width() as i32);
    assert!(attention_y >= 0 && attention_y < im.height() as i32);
}

/// The ported RGBA smartcrop call sites, including the premultiplied form.
#[test]
fn ported_surface_smartcrop_rgba_premultiplied() {
    let im = make_test_rgba();
    let (result, attention_x, attention_y) =
        im.smartcrop_with_coords(80, 60, SmartcropInteresting::Attention);
    assert_eq!(result.width(), 80);
    assert_eq!(result.height(), 60);
    assert!(attention_x >= 0 && attention_y >= 0);

    let im = im.premultiply();
    let (result, attention_x, attention_y) =
        im.smartcrop_with_coords_premultiplied(80, 60, SmartcropInteresting::Attention, true);
    assert_eq!(result.width(), 80);
    assert_eq!(result.height(), 60);
    assert!(attention_x >= 0 && attention_y >= 0);
}

/// The ported `test_insert` call sites (`ported_conversion.rs`).
#[test]
fn ported_surface_insert() {
    let mono = make_test_mono();
    let colour = make_test_colour();

    let result = mono.insert(&colour, 10, 10, false);
    assert_eq!(result.width(), mono.width());
    assert_eq!(result.height(), mono.height());
    let px = result.getpoint(10, 10);
    let orig = colour.getpoint(0, 0);
    for (a, b) in px.iter().zip(orig.iter()) {
        assert!((a - b).abs() < 1.0);
    }

    let result = mono.insert(&colour, 10, 10, true);
    assert_eq!(result.width(), mono.width() + 10);
    assert_eq!(result.height(), mono.height() + 10);
}

/// The `insert` shape the ported arithmetic suite builds its half/half
/// images with (`left.insert(&right, 50, 0, true)`).
#[test]
fn ported_surface_insert_half_half() {
    let left = Raster::new(50, 100, PixelFormat::Gray8, vec![0u8; 5000]).unwrap();
    let right = Raster::new(50, 100, PixelFormat::Gray8, vec![10u8; 5000]).unwrap();
    let combined = left.insert(&right, 50, 0, true);
    assert_eq!(combined.width(), 100);
    assert_eq!(combined.height(), 100);
    assert!((combined.avg() - 5.0).abs() < 1e-9);
}

/// The ported `test_replicate` and `test_grid` setup call sites.
#[test]
fn ported_surface_replicate() {
    let colour = make_test_colour();
    let result = colour.replicate(10, 10);
    assert_eq!(result.width(), colour.width() * 10);
    assert_eq!(result.height(), colour.height() * 10);

    let before = colour.getpoint(10, 10);
    let after = result.getpoint(10 + colour.width() * 2, 10 + colour.height() * 2);
    for (b, a) in before.iter().zip(after.iter()) {
        assert!((a - b).abs() < 1.0);
    }

    // The grid test's setup (grid itself is a later batch).
    let tall = colour.replicate(1, 12);
    assert_eq!(tall.height(), colour.height() * 12);
}

/// The ported `test_subsample` call site.
#[test]
fn ported_surface_subsample() {
    let colour = make_test_colour();
    let result = colour.subsample(3, 3);
    assert_eq!(result.width(), colour.width() / 3);
    assert_eq!(result.height(), colour.height() / 3);

    let before = colour.getpoint(60, 60);
    let after = result.getpoint(20, 20);
    for (b, a) in before.iter().zip(after.iter()) {
        assert!((a - b).abs() < 1.0);
    }
}

/// The ported `test_zoom` call site.
#[test]
fn ported_surface_zoom() {
    let colour = make_test_colour();
    let result = colour.zoom(3, 3);
    assert_eq!(result.width(), colour.width() * 3);
    assert_eq!(result.height(), colour.height() * 3);

    let before = colour.getpoint(50, 50);
    let after = result.getpoint(150, 150);
    for (b, a) in before.iter().zip(after.iter()) {
        assert!((a - b).abs() < 1.0);
    }
}
