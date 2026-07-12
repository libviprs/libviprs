//! Pins the resample call surface required by the libviprs-tests ported
//! suite (libviprs-tests issue #55, `tests/ported_resample.rs`).
//!
//! Integration tests compile as an external crate, exactly the position
//! the ported tests are in, so this file proves the surface they call
//! compiles and behaves: `shrink` / `reduce` / `resize` / `affine` /
//! `similarity` / `rotate` / `mapim` on [`Raster`], the libvips nickname
//! strings for interpolators and kernels, and the `constant_u8` helper,
//! with each ported test body reproduced literally. Behaviour depth is
//! covered by the unit tests in `src/resample.rs`.
//!
//! The ported cell decodes `sample.jpg` from the fetched reference suite;
//! this file substitutes a deterministic synthetic stand-in with the same
//! 290x442 portrait dimensions, so every size assertion exercises the
//! same rounding the ported cell pins. The stand-in is a smooth gradient,
//! which the average-preservation thresholds (`< 1`, `< 2`) require.
//!
//! Two documented adaptations, neither an assertion change:
//!
//! * The ported `test_affine` iterates the interpolators
//!   `["nearest", "bicubic", "bilinear", "nohalo", "lbb"]`. The `nohalo`
//!   and `lbb` resamplers (libvips `nohalo.cpp` / `lbb.cpp`, roughly 2400
//!   lines of minmod subdivision arithmetic) are a dedicated later batch;
//!   `Interpolator::from_name` recognises the names and fails loudly with
//!   `ResampleError::InterpolatorNotImplemented` instead of silently
//!   aliasing them. The loop here runs the three implemented
//!   interpolators with the asserted round-trip identity untouched, and a
//!   companion test pins the loud typed failure for the deferred two.
//! * The ported `test_thumbnail`, `test_thumbnail_uhdr_linear`, and
//!   `test_thumbnail_icc` bodies are not reproduced: the thumbnail family
//!   is not part of this batch. `ported_foreign.rs` declares a
//!   conflicting `thumbnail` signature (`fn thumbnail(path, width) ->
//!   Result<Raster, DecodeError>` versus this cell's four-argument
//!   `Raster::thumbnail`), `ported_infrastructure.rs` a third
//!   (`im.thumbnail(200)`), and the ICC body needs `Raster::max_value`,
//!   which belongs to the colour / create batches; reconciling the three
//!   call surfaces needs its own coordinated batch.

use libviprs::{PixelFormat, Raster};

/// The synthetic stand-in for `sample.jpg`: a smooth Rgb8 gradient with
/// the reference image's 290x442 portrait dimensions.
fn sample() -> Raster {
    libviprs::source::generate_test_raster(290, 442).unwrap()
}

// ---------------------------------------------------------------------------
// 5.1 Resize
// ---------------------------------------------------------------------------
mod resize {
    use super::*;

    /// Subset of libvips test_resample.py::test_resize (the ported
    /// test_resize_quarter body, verbatim).
    #[test]
    fn test_resize_quarter() {
        let src = libviprs::source::generate_test_raster(256, 256).unwrap();
        let half = libviprs::resize::downscale_half(&src).unwrap();
        assert_eq!(half.width(), 128);
        assert_eq!(half.height(), 128);
        let quarter = libviprs::resize::downscale_half(&half).unwrap();
        assert_eq!(quarter.width(), 64);
        assert_eq!(quarter.height(), 64);
    }

    /// The ported test_resize_rounding body: resize sizing rounds to
    /// nearest, including the thin-image and double-precision cases.
    #[test]
    fn test_resize_rounding() {
        let im = sample();
        let im2 = im.resize(0.25);
        assert_eq!(im2.width(), ((im.width() as f64 / 4.0) + 0.5) as u32);
        assert_eq!(im2.height(), ((im.height() as f64 / 4.0) + 0.5) as u32);

        // Edge case: thin image
        let im = Raster::black(100, 1);
        let x = im.resize(0.5);
        assert_eq!(x.width(), 50);
        assert_eq!(x.height(), 1);

        // Double-precision calculation
        let im = Raster::black(1600, 1000);
        let x = im.resize(10.0 / 1600.0);
        assert_eq!(x.width(), 10);
        assert_eq!(x.height(), 6);
    }

    /// The ported test_shrink body: box shrink sizing and average
    /// preservation for integer and fractional factors.
    #[test]
    fn test_shrink() {
        let im = sample();

        let im2 = im.shrink(4.0, 4.0);
        assert_eq!(im2.width(), ((im.width() as f64 / 4.0) + 0.5) as u32);
        assert_eq!(im2.height(), ((im.height() as f64 / 4.0) + 0.5) as u32);
        assert!((im.avg() - im2.avg()).abs() < 1.0);

        let im2 = im.shrink(2.5, 2.5);
        assert_eq!(im2.width(), ((im.width() as f64 / 2.5) + 0.5) as u32);
        assert_eq!(im2.height(), ((im.height() as f64 / 2.5) + 0.5) as u32);
        assert!((im.avg() - im2.avg()).abs() < 1.0);
    }

    /// The ported test_shrink_average body: box-average shrink preserves
    /// the mean across scale factors.
    #[test]
    fn test_shrink_average() {
        let im = sample();

        for &factor in &[2.0, 3.0, 4.0, 5.0, 8.0] {
            let shrunk = im.shrink(factor, factor);
            assert!(
                (im.avg() - shrunk.avg()).abs() < 1.0,
                "Average should be preserved when shrinking by {factor}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5.2 Affine Transform
// ---------------------------------------------------------------------------
mod affine {
    use super::*;

    /// The ported test_affine body over the implemented interpolators:
    /// four 90-degree rotations with the [0, 1, 1, 0] matrix are the
    /// identity, byte for byte. The `nohalo` / `lbb` entries of the
    /// ported list are deferred (see the file header); their loud typed
    /// failure is pinned below.
    #[test]
    fn test_affine() {
        let im = sample();

        for interp in &["nearest", "bicubic", "bilinear"] {
            let mut x = im.clone();
            for _ in 0..4 {
                x = x.affine([0.0, 1.0, 1.0, 0.0], interp);
            }

            let max_diff = im
                .data()
                .iter()
                .zip(x.data().iter())
                .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
                .max()
                .unwrap_or(0);
            assert_eq!(
                max_diff, 0,
                "4x rotation round-trip should be identity for {interp}"
            );
        }
    }

    /// The deferred interpolators fail loudly with a typed error instead
    /// of silently aliasing to bicubic.
    #[test]
    fn test_affine_deferred_interpolators_fail_loudly() {
        for name in ["nohalo", "lbb"] {
            let err = libviprs::Interpolator::from_name(name).unwrap_err();
            assert!(
                matches!(
                    err,
                    libviprs::ResampleError::InterpolatorNotImplemented { .. }
                ),
                "{name} should be recognised but unimplemented, got {err}"
            );
        }
    }

    /// The ported test_similarity body: similarity(90) approximately
    /// matches the equivalent affine rotation.
    #[test]
    fn test_similarity() {
        let im = sample();

        let im2 = im.similarity(90.0, 1.0);
        let im3 = im.affine([0.0, -1.0, 1.0, 0.0], "bilinear");

        let max_diff = im2
            .data()
            .iter()
            .zip(im3.data().iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs())
            .max()
            .unwrap_or(0);
        assert!(
            max_diff < 50,
            "similarity(90) vs affine rotation: max_diff={max_diff}"
        );
    }

    /// The ported test_similarity_scale body: similarity(0, 2) matches
    /// affine([2, 0, 0, 2]) exactly.
    #[test]
    fn test_similarity_scale() {
        let im = sample();

        let im2 = im.similarity(0.0, 2.0);
        let im3 = im.affine([2.0, 0.0, 0.0, 2.0], "bilinear");

        let max_diff = im2
            .data()
            .iter()
            .zip(im3.data().iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs())
            .max()
            .unwrap_or(0);
        assert_eq!(max_diff, 0, "similarity(scale=2) should match affine(2x)");
    }

    /// The ported test_rotate body: rotate(90) approximately matches the
    /// equivalent affine rotation.
    #[test]
    fn test_rotate() {
        let im = sample();

        let im2 = im.rotate(90.0);
        let im3 = im.affine([0.0, -1.0, 1.0, 0.0], "bilinear");

        let max_diff = im2
            .data()
            .iter()
            .zip(im3.data().iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs())
            .max()
            .unwrap_or(0);
        assert!(max_diff < 50, "rotate(90) vs affine: max_diff={max_diff}");
    }
}

// ---------------------------------------------------------------------------
// 5.3 Advanced Resampling
// ---------------------------------------------------------------------------
mod advanced_resampling {
    use super::*;

    /// The ported test_reduce body: average preservation for every
    /// kernel at fractional factors, and exact constant preservation.
    #[test]
    fn test_reduce() {
        let im = sample();

        let kernels = ["nearest", "linear", "cubic", "lanczos2", "lanczos3"];

        for &fac in &[1.0, 1.1, 1.5, 1.999] {
            for kernel in &kernels {
                let r = im.reduce(fac, fac, kernel);
                let d = (r.avg() - im.avg()).abs();
                assert!(d < 2.0, "reduce(fac={fac}, kernel={kernel}) avg diff={d}");
            }
        }

        // Constant image preservation
        for &val in &[0u8, 1, 2, 254, 255] {
            let constant = Raster::constant_u8(10, 10, val);
            for kernel in &kernels {
                let r = constant.reduce(2.0, 2.0, kernel);
                assert!(
                    (r.avg() - val as f64).abs() < 0.001,
                    "Constant {val} should be preserved by reduce with {kernel}"
                );
            }
        }
    }

    /// The ported test_mapim body: an identity coordinate image
    /// preserves the average exactly.
    #[test]
    fn test_mapim() {
        let im = sample();

        // Identity remap: coordinate image
        let mp = Raster::xyz(im.width(), im.height());
        let remapped = im.mapim(&mp, "bicubic");
        assert!(
            (im.avg() - remapped.avg()).abs() < 0.001,
            "Identity mapim should preserve average exactly"
        );
    }

    /// The ported cell's kernel strings and the typed enum agree; the
    /// nickname set matches libvips `VipsKernel` (plus `mitchell`, which
    /// the cell does not exercise).
    #[test]
    fn test_kernel_nicknames() {
        for name in [
            "nearest", "linear", "cubic", "mitchell", "lanczos2", "lanczos3",
        ] {
            libviprs::ReduceKernel::from_name(name).unwrap();
        }
    }

    /// The ported surface types re-export from the crate root.
    #[test]
    fn test_root_reexports() {
        let _: libviprs::Interpolator = libviprs::Interpolator::Bilinear;
        let _: libviprs::ReduceKernel = libviprs::ReduceKernel::Lanczos3;
        let _ = libviprs::AffineOptions::default();
        let _ = libviprs::ResizeOptions::default();
        let im = Raster::constant_u8(4, 4, 9);
        assert_eq!(im.format(), PixelFormat::Gray8);
    }
}
