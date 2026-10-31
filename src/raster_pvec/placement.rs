//! The georeference composition for a raster in a PVec (issue #1328, R2.4).
//!
//! ADR 0018 (libviprs/pvec) keeps one georeference per drawing and puts the raster in the
//! drawing's **source space**, so a raster that starts from a geo-referenced scan needs the map
//! from its pixels into source space:
//!
//! ```text
//! pixel -> source = inverse(source -> native) o (pixel -> native)
//!                 = (1 / metres_per_unit) * inverse(metres -> native) o (pixel -> native)
//! ```
//!
//! `source -> native` is the drawing's calibration (source units to metres, then the affine
//! calibration into the CRS's native units, which is the order maplibre-pvec composes in). The
//! result is written in PVec's `Transform2` order `[a b c d tx ty]`, rows `[a b tx]` and
//! `[c d ty]`, so libviprs's `[a b c d e f]` is permuted to `[a b d e c f]`. The pixel origin is
//! the top-left **corner** of pixel (0, 0); a transform stored with [`PixelOrigin::Centre`] is
//! shifted by half a pixel first. Source space is y-up and a raster's rows run down, so an
//! upright image has a negative determinant here: that is the normal case.
//!
//! The composition is affine only when the raster's CRS is the calibration's native CRS and the
//! calibration is itself affine. Anything else is a typed [`PlacementError`], never a silent
//! resample: reprojecting pixels is lossy, and it is the producer's business before this call.
//!
//! The calibration is passed in explicitly. The georeference handoff of issue #1048 (N7.3) will
//! produce one; until it lands the caller supplies it.

use thiserror::Error;

use crate::geo::{GeoTransform, PixelOrigin};

/// The drawing's calibration, as far as a raster placement needs it.
///
/// Input to the calibration is **metres in the drawing frame**; output is native CRS units.
/// `metres_per_unit` converts the drawing's source unit (`0.001` for millimetres).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SourceCalibration {
    /// An affine calibration.
    Affine {
        /// The CRS the calibration's output is in, as an identifier such as `EPSG:2056`.
        native_crs: String,
        /// Metres in the drawing frame to native CRS units, in [`GeoTransform`] order.
        metres_to_native: GeoTransform,
        /// Metres per source unit, finite and above zero.
        metres_per_unit: f64,
    },
    /// A row-major 3x3 homography (a rubber-sheeted or scanned plan). A raster cannot ride it as
    /// an affine placement, so composing against it is refused.
    Projective {
        /// The CRS the calibration's output is in.
        native_crs: String,
        /// The homography, row-major.
        homography: [f64; 9],
        /// Metres per source unit.
        metres_per_unit: f64,
    },
}

/// Why a placement could not be composed.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum PlacementError {
    /// The raster is in a different CRS from the calibration's native one. Composing would need
    /// a reprojection, which resamples pixels, so it is not done here.
    #[error(
        "the raster is in {raster} but the calibration's native CRS is {calibration}; \
         reproject the raster first, a placement never resamples"
    )]
    CrsMismatch {
        /// The raster's CRS.
        raster: String,
        /// The calibration's native CRS.
        calibration: String,
    },
    /// The calibration is projective, so `source -> native` is not affine.
    #[error("the calibration is projective, so a raster cannot be placed with an affine map")]
    ProjectiveCalibration,
    /// The calibration's linear part has no inverse.
    #[error("the calibration is singular, so pixel space cannot be mapped back into source space")]
    SingularCalibration,
    /// The raster's own transform has no inverse (a flat or degenerate pixel size).
    #[error("the raster's pixel to geo transform is singular")]
    SingularRaster,
    /// `metres_per_unit` was not finite and above zero.
    #[error("metres per unit must be finite and above zero, got {0}")]
    InvalidUnitScale(f64),
    /// A coefficient of an input, or of the result, was not finite.
    #[error("a coefficient was not finite")]
    NonFinite,
}

/// Compose `pixel -> source` for a raster.
///
/// `pixel_to_native` is the raster's [`GeoTransform`] under `pixel_origin`, and `raster_crs` the
/// CRS its coordinates are in. The result is `[a b c d tx ty]` in PVec's `Transform2` order with
/// `-0.0` normalised to `+0.0`, ready for a placement.
///
/// # Errors
///
/// A [`PlacementError`] names the precondition that failed: a non-finite input, a bad unit
/// scale, a CRS mismatch, a projective or singular calibration, or a singular raster transform.
///
/// # Examples
///
/// ```
/// use libviprs::raster_pvec::placement::{SourceCalibration, pixel_to_source};
/// use libviprs::{GeoTransform, PixelOrigin};
///
/// // A 0.5 m pixel scan whose rows run down, in a drawing calibrated one to one.
/// let pixel_to_native = GeoTransform::new(0.5, 0.0, 100.0, 0.0, -0.5, 200.0);
/// let calibration = SourceCalibration::Affine {
///     native_crs: "EPSG:2056".to_string(),
///     metres_to_native: GeoTransform::new(1.0, 0.0, 100.0, 0.0, 1.0, 100.0),
///     metres_per_unit: 1.0,
/// };
/// let m = pixel_to_source(&pixel_to_native, PixelOrigin::Corner, "EPSG:2056", &calibration)?;
/// assert_eq!(m, [0.5, 0.0, 0.0, -0.5, 0.0, 100.0]);
/// # Ok::<(), libviprs::raster_pvec::placement::PlacementError>(())
/// ```
pub fn pixel_to_source(
    pixel_to_native: &GeoTransform,
    pixel_origin: PixelOrigin,
    raster_crs: &str,
    calibration: &SourceCalibration,
) -> Result<[f64; 6], PlacementError> {
    let (native_crs, metres_per_unit) = match calibration {
        SourceCalibration::Affine {
            native_crs,
            metres_per_unit,
            ..
        }
        | SourceCalibration::Projective {
            native_crs,
            metres_per_unit,
            ..
        } => (native_crs, *metres_per_unit),
    };
    if !(metres_per_unit.is_finite() && metres_per_unit > 0.0) {
        return Err(PlacementError::InvalidUnitScale(metres_per_unit));
    }
    if !raster_crs.trim().eq_ignore_ascii_case(native_crs.trim()) {
        return Err(PlacementError::CrsMismatch {
            raster: raster_crs.to_string(),
            calibration: native_crs.clone(),
        });
    }
    let SourceCalibration::Affine {
        metres_to_native, ..
    } = calibration
    else {
        return Err(PlacementError::ProjectiveCalibration);
    };

    let p = pixel_to_native;
    let all_finite = [p.a, p.b, p.c, p.d, p.e, p.f]
        .iter()
        .chain([
            &metres_to_native.a,
            &metres_to_native.b,
            &metres_to_native.c,
            &metres_to_native.d,
            &metres_to_native.e,
            &metres_to_native.f,
        ])
        .all(|v| v.is_finite());
    if !all_finite {
        return Err(PlacementError::NonFinite);
    }
    if p.inverse().is_none() {
        return Err(PlacementError::SingularRaster);
    }
    let s = metres_to_native
        .inverse()
        .ok_or(PlacementError::SingularCalibration)?;

    // Half a pixel in from the corner: P_corner(q) = P_centre(q - 0.5), so only the offset moves.
    let (c, f) = match pixel_origin {
        PixelOrigin::Centre => (p.c - 0.5 * (p.a + p.b), p.f - 0.5 * (p.d + p.e)),
        _ => (p.c, p.f),
    };

    // S o P with S = native -> metres, then metres -> source units.
    let a = (s.a * p.a + s.b * p.d) / metres_per_unit;
    let b = (s.a * p.b + s.b * p.e) / metres_per_unit;
    let tx = (s.a * c + s.b * f + s.c) / metres_per_unit;
    let c2 = (s.d * p.a + s.e * p.d) / metres_per_unit;
    let d = (s.d * p.b + s.e * p.e) / metres_per_unit;
    let ty = (s.d * c + s.e * f + s.f) / metres_per_unit;

    let out = [a, b, c2, d, tx, ty];
    if !out.iter().all(|v| v.is_finite()) {
        return Err(PlacementError::NonFinite);
    }
    // NP-03: -0.0 would make two equal placements differ in their stored bits.
    Ok(out.map(|v| if v == 0.0 { 0.0 } else { v }))
}
