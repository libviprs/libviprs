//! Issue #1328 (R2.4): the composed raster placement.
//!
//! `pixel -> source = (1 / metres_per_unit) * inverse(metres -> native) o (pixel -> native)`,
//! written in PVec's `Transform2` order `[a b c d tx ty]` (rows `[a b tx]` and `[c d ty]`),
//! with libviprs's `[a b c d e f]` permuted to `[a b d e c f]`.
//!
//! The vectors below are dyadic on purpose, so every product and sum is exact and a bit-for-bit
//! comparison means something. They are worked by hand in the comments. They are the producer's
//! side of the vector; `docs/RASTER.md` in libviprs/pvec carries the shared one.

use std::error::Error;

use libviprs::pvec::raster_placement::{PlacementError, SourceCalibration, pixel_to_source};
use libviprs::{GeoTransform, PixelOrigin};

/// P (pixel -> native), rotated and sheared:
/// `x = 0.5 px + 0.25 py + 1000`, `y = 0.125 px - 0.5 py + 2000`.
fn pixel_to_native() -> GeoTransform {
    GeoTransform::new(0.5, 0.25, 1000.0, 0.125, -0.5, 2000.0)
}

/// B (metres -> native): `x' = 2 x + y + 900`, `y' = y + 1800`. det = 2, inverse is exact.
fn calibration(metres_per_unit: f64) -> SourceCalibration {
    SourceCalibration::Affine {
        native_crs: "EPSG:2056".to_string(),
        metres_to_native: GeoTransform::new(2.0, 1.0, 900.0, 0.0, 1.0, 1800.0),
        metres_per_unit,
    }
}

fn bits(m: [f64; 6]) -> [u64; 6] {
    m.map(f64::to_bits)
}

#[test]
fn the_vector_composes_to_six_coefficients_bit_for_bit() {
    // B^-1 = [[0.5, -0.5], [0, 1]]; A = [[0.5, 0.25], [0.125, -0.5]]
    // B^-1 A = [[0.1875, 0.375], [0.125, -0.5]], times 1 / 0.5 = 2: [[0.375, 0.75], [0.25, -1]]
    // t - s = (100, 200); B^-1 (100, 200) = (-50, 200); times 2 = (-100, 400)
    // libviprs order [a b c d e f] = [0.375 0.75 -100 0.25 -1 400]; PVec order [a b c d tx ty]:
    let got = pixel_to_source(
        &pixel_to_native(),
        PixelOrigin::Corner,
        "EPSG:2056",
        &calibration(0.5),
    )
    .expect("the vector composes");
    assert_eq!(bits(got), bits([0.375, 0.75, 0.25, -1.0, -100.0, 400.0]));
}

#[test]
fn the_permutation_is_a_b_d_e_c_f_and_not_the_identity() {
    // A pure translation in pixel space with distinct linear entries, identity calibration:
    // the result must put libviprs's (c, f) in the last two slots and (d, e) in slots 2 and 3.
    let p = GeoTransform::new(1.0, 2.0, 30.0, 3.0, 8.0, 60.0);
    let identity = SourceCalibration::Affine {
        native_crs: "X".to_string(),
        metres_to_native: GeoTransform::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        metres_per_unit: 1.0,
    };
    let got = pixel_to_source(&p, PixelOrigin::Corner, "X", &identity).unwrap();
    assert_eq!(bits(got), bits([1.0, 2.0, 3.0, 8.0, 30.0, 60.0]));
}

#[test]
fn a_centre_origin_is_shifted_by_half_a_pixel_to_the_corner_convention() {
    // P_corner(q) = P_centre(q - 0.5), so t' = t - A (0.5, 0.5):
    // (1000 - 0.25 - 0.125, 2000 - 0.0625 + 0.25) = (999.625, 2000.1875), which composes to
    // tx = -100.5625 and ty = 400.375 with the same linear part as the corner vector.
    let got = pixel_to_source(
        &pixel_to_native(),
        PixelOrigin::Centre,
        "EPSG:2056",
        &calibration(0.5),
    )
    .expect("composes");
    assert_eq!(
        bits(got),
        bits([0.375, 0.75, 0.25, -1.0, -100.5625, 400.375])
    );
}

#[test]
fn the_metres_per_unit_factor_is_folded_in() {
    // Millimetres: 0.001 metres a unit. The linear part scales by 1000 against the metre case.
    let m = pixel_to_source(
        &pixel_to_native(),
        PixelOrigin::Corner,
        "EPSG:2056",
        &calibration(1.0),
    )
    .unwrap();
    let mm = pixel_to_source(
        &pixel_to_native(),
        PixelOrigin::Corner,
        "EPSG:2056",
        &calibration(0.001),
    )
    .unwrap();
    for (i, (a, b)) in m.iter().zip(mm).enumerate() {
        assert!(
            (a * 1000.0 - b).abs() <= 1e-9 * b.abs().max(1.0),
            "slot {i}: {a} x 1000 vs {b}"
        );
    }
}

#[test]
fn negative_zero_is_normalised_to_positive_zero() {
    let p = GeoTransform::new(1.0, -0.0, 5.0, -0.0, -1.0, 7.0);
    let identity = SourceCalibration::Affine {
        native_crs: "X".to_string(),
        metres_to_native: GeoTransform::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        metres_per_unit: 1.0,
    };
    let got = pixel_to_source(&p, PixelOrigin::Corner, "X", &identity).unwrap();
    assert_eq!(bits(got), bits([1.0, 0.0, 0.0, -1.0, 5.0, 7.0]));
    assert!(
        got.iter().all(|c| *c != 0.0 || c.is_sign_positive()),
        "no -0.0 may leave: {got:?}"
    );
}

#[test]
fn a_raster_in_another_crs_is_refused_by_name_and_never_resampled() {
    let err = pixel_to_source(
        &pixel_to_native(),
        PixelOrigin::Corner,
        "EPSG:4326",
        &calibration(1.0),
    )
    .expect_err("the raster's CRS is not the calibration's native CRS");
    match &err {
        PlacementError::CrsMismatch {
            raster,
            calibration,
        } => {
            assert_eq!(
                (raster.as_str(), calibration.as_str()),
                ("EPSG:4326", "EPSG:2056")
            );
        }
        other => panic!("wrong refusal: {other:?}"),
    }
    assert!(err.to_string().contains("EPSG:4326"), "{err}");
}

#[test]
fn a_projective_calibration_is_refused_by_name() {
    let cal = SourceCalibration::Projective {
        native_crs: "EPSG:2056".to_string(),
        homography: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.001, 0.0, 1.0],
        metres_per_unit: 1.0,
    };
    let err =
        pixel_to_source(&pixel_to_native(), PixelOrigin::Corner, "EPSG:2056", &cal).unwrap_err();
    assert!(
        matches!(err, PlacementError::ProjectiveCalibration),
        "{err:?}"
    );
}

#[test]
fn degenerate_inputs_are_typed_refusals() {
    let singular = SourceCalibration::Affine {
        native_crs: "EPSG:2056".to_string(),
        metres_to_native: GeoTransform::new(1.0, 2.0, 0.0, 2.0, 4.0, 0.0),
        metres_per_unit: 1.0,
    };
    assert!(matches!(
        pixel_to_source(
            &pixel_to_native(),
            PixelOrigin::Corner,
            "EPSG:2056",
            &singular
        ),
        Err(PlacementError::SingularCalibration)
    ));
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                pixel_to_source(
                    &pixel_to_native(),
                    PixelOrigin::Corner,
                    "EPSG:2056",
                    &calibration(bad)
                ),
                Err(PlacementError::InvalidUnitScale(_))
            ),
            "metres per unit {bad}"
        );
    }
    let nan = GeoTransform::new(f64::NAN, 0.0, 0.0, 0.0, 1.0, 0.0);
    assert!(matches!(
        pixel_to_source(&nan, PixelOrigin::Corner, "EPSG:2056", &calibration(1.0)),
        Err(PlacementError::NonFinite)
    ));
    let flat = GeoTransform::new(1.0, 2.0, 0.0, 2.0, 4.0, 0.0);
    assert!(matches!(
        pixel_to_source(&flat, PixelOrigin::Corner, "EPSG:2056", &calibration(1.0)),
        Err(PlacementError::SingularRaster)
    ));
}

#[test]
fn the_refusal_survives_a_source_chain() {
    // The producer hands this error up through `EngineError::Source`, as the strip-source
    // failures do (tests/error_source_typing.rs): the typed value must downcast back.
    let err = pixel_to_source(
        &pixel_to_native(),
        PixelOrigin::Corner,
        "EPSG:4326",
        &calibration(1.0),
    )
    .unwrap_err();
    let wrapped = libviprs::EngineError::Source(Box::new(err));
    let source = Error::source(&wrapped).expect("EngineError::Source exposes its source()");
    assert!(matches!(
        source.downcast_ref::<PlacementError>(),
        Some(PlacementError::CrsMismatch { .. })
    ));
}
