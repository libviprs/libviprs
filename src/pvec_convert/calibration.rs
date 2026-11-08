//! A drawing's calibration (issue #1546, slice 1 of #1543): what one source unit stands for in the
//! world, and where the drawing sits on the earth.
//!
//! The design is the panel decision on libviprs/pvec#263. PVec carries a calibration in one optional
//! block (capability 22, block kind 11) with two optional records, a scale and a georeference, and
//! the types here are libviprs's side of that block. The geometry stays in source space either way:
//! nothing here is applied to a coordinate.
//!
//! [`Calibration::new`] checks every rule the block will hold the records to, so a calibration that
//! gets this far is one PVec's writer takes. The rules, with the block's `CAL-NN` names:
//!
//! | rule | refused as |
//! |---|---|
//! | CAL-03 more than 1,024 control points, a text over its cap (64 byte CRS id, 16 KiB definition, 255 byte note and external id) | [`CalibrationError::TooMany`], [`CalibrationError::Text`] |
//! | CAL-04 a number that is not finite and normal (NP-01), a coordinate over 2^40 (NP-02), a scale outside `[2^-40, 2^40]`, an uncertainty outside `[0, 1)` | [`CalibrationError::OutOfRange`] |
//! | CAL-05 an affine that fails NP-19 (a linear entry over 2^20, or singular) | [`CalibrationError::Affine`] |
//! | CAL-06 one source point with two world points, a measured pair closer than NP-06's degenerate length | [`CalibrationError::Contradictory`], [`CalibrationError::Degenerate`] |
//! | CAL-07 a control character (a definition may hold LF and TAB), U+2028 or U+2029, a bidi override, or a CRS id outside `[A-Za-z0-9:_.+-]` | [`CalibrationError::Text`] |
//! | CAL-09 neither a scale nor a georeference | [`CalibrationError::Empty`] |
//! | CAL-10 evidence a provenance does not use or lacks; a `Georeference` scale with no georeference; a fit from fewer than two points; a residual on anything but a fit of three points or more | [`CalibrationError::Evidence`] |
//! | a georeference with neither a CRS id nor a definition | [`CalibrationError::NoCrs`] |
//!
//! A projective transform (a 3x3 with a perspective row) has no representation in the block, so
//! [`Georeference::affine_from_matrix`] refuses it with [`CalibrationError::Projective`] rather than
//! dropping the perspective and moving points.

use thiserror::Error;

/// NP-02's `MAX_ABS_COORD`, and the scale factor's bounds are this and its inverse.
const MAX_ABS: f64 = 1_099_511_627_776.0;

/// A scale, a georeference, or both, for one drawing.
#[derive(Debug, Clone, PartialEq)]
pub struct Calibration {
    scale: Option<ScaleClaim>,
    georeference: Option<Georeference>,
    external_id: Option<String>,
}

/// What one source unit stands for in the world, and the evidence for it.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleClaim {
    /// Metres in the world per source unit. Not the manifest's unit: on a 1:100 sheet drawn in
    /// millimetres the manifest says a unit is 0.001 m and this says it stands for 0.1 m.
    pub metres_per_source_unit: f64,
    /// Where the number came from.
    pub provenance: ScaleProvenance,
    /// A relative uncertainty in `[0, 1)`, when one is known.
    pub uncertainty: Option<f64>,
    /// Plain text, at most 255 bytes: the ratio as the sheet prints it, say.
    pub note: String,
}

/// Where a scale came from, with the evidence that provenance needs.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ScaleProvenance {
    /// The source declares a unit the manifest's table cannot name, such as a PDF's points.
    DeclaredUnit,
    /// A ratio stated on the sheet: `numerator` paper units to `denominator` world units, with
    /// one source unit being `paper_metres_per_unit` metres of paper.
    PlanScale {
        /// The paper side of the ratio, at least 1.
        numerator: u32,
        /// The world side of the ratio, at least 1.
        denominator: u32,
        /// Metres of paper per source unit.
        paper_metres_per_unit: f64,
    },
    /// Two source points and the real distance between them, in metres.
    Measured {
        /// The first point, in source units.
        p0: [f64; 2],
        /// The second point, in source units.
        p1: [f64; 2],
        /// The distance between them in the world, in metres.
        metres: f64,
    },
    /// Derived from the georeference.
    Georeference,
    /// Entered by hand.
    Manual,
}

/// Where the drawing sits: an affine from source space to a CRS, and the evidence for it.
#[derive(Debug, Clone, PartialEq)]
pub struct Georeference {
    /// `x' = a x + b y + tx`, `y' = c x + d y + ty`, in PVec's `[a b c d tx ty]` order. The output is
    /// (easting, northing) for a projected or engineering CRS and (longitude, latitude) for a
    /// geographic one.
    pub affine: [f64; 6],
    /// What kind of CRS the affine lands in.
    pub crs_kind: CrsKind,
    /// How the affine was arrived at.
    pub method: GeoreferenceMethod,
    /// An authority code such as `EPSG:27700`, at most 64 bytes of `[A-Za-z0-9:_.+-]`.
    pub crs_id: Option<String>,
    /// The CRS's definition, opaque to libviprs and PVec, handed to the host as it is.
    pub definition: Option<CrsDefinition>,
    /// The evidence, at most 1,024 pairs.
    pub control_points: Vec<ControlPoint>,
    /// The largest distance, in CRS units, between a control point's world position and where the
    /// affine puts its source position. Zero unless the affine was fitted to three points or more.
    pub residual_max: f64,
}

/// What kind of CRS a georeference lands in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CrsKind {
    /// A projected CRS: eastings and northings.
    Projected,
    /// A geographic CRS: longitude and latitude, in that order.
    Geographic,
    /// An engineering or local CRS, such as a site grid.
    Engineering,
}

/// How a georeference's affine was arrived at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeoreferenceMethod {
    /// The source states the affine.
    Declared,
    /// Fitted by libviprs to the control points.
    Fitted,
    /// Entered by hand.
    Manual,
}

/// A CRS definition as text, in the format it was given in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrsDefinition {
    /// The text's format.
    pub format: DefinitionFormat,
    /// At most 16 KiB of plain text; LF and TAB are allowed.
    pub text: String,
}

/// The format of a [`CrsDefinition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DefinitionFormat {
    /// Well-known text, as a GeoPDF's `/GCS /WKT` carries it.
    Wkt,
    /// PROJJSON.
    Projjson,
}

/// One control point: a source position and the world position it stands for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlPoint {
    /// In source units.
    pub source: [f64; 2],
    /// In the CRS's units, in the affine's output order.
    pub world: [f64; 2],
}

/// Why a calibration was refused. Nothing is written for a refused calibration.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum CalibrationError {
    /// Neither a scale nor a georeference (CAL-09).
    #[error("a calibration needs a scale, a georeference or both")]
    Empty,
    /// A number that is not finite and normal, or outside its range (CAL-04).
    #[error("{field} is not a finite number in its range")]
    OutOfRange {
        /// The field.
        field: &'static str,
    },
    /// Evidence that does not fit the provenance or the method (CAL-10).
    #[error("the evidence does not match the scale's provenance or the georeference's method")]
    Evidence,
    /// A measured pair too close to measure (CAL-06, NP-06).
    #[error("the two measured points are too close to give a length")]
    Degenerate,
    /// The affine fails NP-19 (CAL-05).
    #[error("the affine is singular or has an entry out of range")]
    Affine,
    /// A text over its cap or with a character it may not hold (CAL-03, CAL-07).
    #[error("{field} is too long or holds a character it may not")]
    Text {
        /// The field.
        field: &'static str,
    },
    /// A georeference with neither a CRS id nor a definition.
    #[error("a georeference needs a CRS id or a CRS definition")]
    NoCrs,
    /// More control points than the block holds (CAL-03).
    #[error("{count} control points, and the most a calibration holds is 1024")]
    TooMany {
        /// How many were given.
        count: usize,
    },
    /// Two control points with one source position and different world positions (CAL-06).
    #[error("control points {first} and {second} put one source point in two places")]
    Contradictory {
        /// The first point's index.
        first: usize,
        /// The second point's index.
        second: usize,
    },
    /// A projective transform, which no PVec calibration can hold.
    #[error("a projective transform cannot be stored; only an affine can")]
    Projective,
}

impl Calibration {
    /// A calibration from a scale, a georeference or both, and the application's own identifier
    /// for it (at most 255 bytes).
    ///
    /// # Errors
    ///
    /// [`CalibrationError`] for anything PVec's calibration block would refuse.
    pub fn new(
        scale: Option<ScaleClaim>,
        georeference: Option<Georeference>,
        external_id: Option<String>,
    ) -> Result<Self, CalibrationError> {
        fn check_scale(s: &ScaleClaim) -> Result<(), CalibrationError> {
            let factor = |v: f64| normal(v) && (1.0 / MAX_ABS..=MAX_ABS).contains(&v);
            if !factor(s.metres_per_source_unit) {
                return Err(CalibrationError::OutOfRange {
                    field: "metres_per_source_unit",
                });
            }
            if let Some(u) = s.uncertainty
                && !(normal(u) && (0.0..1.0).contains(&u))
            {
                return Err(CalibrationError::OutOfRange {
                    field: "uncertainty",
                });
            }
            check_text(&s.note, 255, false, "note")?;
            match s.provenance {
                ScaleProvenance::PlanScale {
                    numerator,
                    denominator,
                    paper_metres_per_unit,
                } if numerator == 0 || denominator == 0 || !factor(paper_metres_per_unit) => {
                    Err(CalibrationError::Evidence)
                }
                ScaleProvenance::Measured { p0, p1, metres } => {
                    measured_length(p0, p1)?;
                    if factor(metres) {
                        Ok(())
                    } else {
                        Err(CalibrationError::Evidence)
                    }
                }
                _ => Ok(()),
            }
        }

        fn check_georeference(g: &Georeference) -> Result<(), CalibrationError> {
            let [a, b, c, d, tx, ty] = g.affine;
            let linear = [a, b, c, d];
            let big = linear.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if !g.affine.iter().all(|&v| normal(v))
                || tx.abs() > MAX_ABS
                || ty.abs() > MAX_ABS
                || big > 1_048_576.0
                || (a * d - b * c).abs() <= 1e-10 * big * big
            {
                return Err(CalibrationError::Affine);
            }
            if let Some(id) = &g.crs_id {
                let class = |c: char| c.is_ascii_alphanumeric() || ":_.+-".contains(c);
                if id.is_empty() || id.len() > 64 || !id.chars().all(class) {
                    return Err(CalibrationError::Text { field: "crs_id" });
                }
            }
            if let Some(def) = &g.definition {
                if def.text.is_empty() {
                    return Err(CalibrationError::Text {
                        field: "definition",
                    });
                }
                check_text(&def.text, 16 * 1024, true, "definition")?;
            }
            if g.crs_id.is_none() && g.definition.is_none() {
                return Err(CalibrationError::NoCrs);
            }
            let points = &g.control_points;
            if points.len() > 1024 {
                return Err(CalibrationError::TooMany {
                    count: points.len(),
                });
            }
            if points
                .iter()
                .any(|p| !coordinate(p.source) || !coordinate(p.world))
            {
                return Err(CalibrationError::OutOfRange {
                    field: "control point",
                });
            }
            for (first, p) in points.iter().enumerate() {
                if let Some(n) = points[first + 1..]
                    .iter()
                    .position(|q| q.source == p.source && q.world != p.world)
                {
                    return Err(CalibrationError::Contradictory {
                        first,
                        second: first + 1 + n,
                    });
                }
            }
            if !(normal(g.residual_max) && g.residual_max >= 0.0) {
                return Err(CalibrationError::OutOfRange {
                    field: "residual_max",
                });
            }
            if g.residual_max != 0.0 && (g.method != GeoreferenceMethod::Fitted || points.len() < 3)
            {
                return Err(CalibrationError::Evidence);
            }
            if g.method == GeoreferenceMethod::Fitted && points.len() < 2 {
                return Err(CalibrationError::Evidence);
            }
            Ok(())
        }

        if scale.is_none() && georeference.is_none() {
            return Err(CalibrationError::Empty);
        }
        if let Some(s) = &scale {
            check_scale(s)?;
        }
        if let Some(g) = &georeference {
            check_georeference(g)?;
        }
        if let Some(id) = &external_id {
            check_text(id, 255, false, "external_id")?;
        }
        if scale
            .as_ref()
            .is_some_and(|s| s.provenance == ScaleProvenance::Georeference)
            && georeference.is_none()
        {
            return Err(CalibrationError::Evidence);
        }
        // NP-03: the block stores -0.0 as +0.0, and adding +0.0 is what turns one into the other.
        let mut scale = scale;
        let mut georeference = georeference;
        if let Some(s) = &mut scale {
            s.metres_per_source_unit += 0.0;
            if let Some(u) = &mut s.uncertainty {
                *u += 0.0;
            }
            match &mut s.provenance {
                ScaleProvenance::PlanScale {
                    paper_metres_per_unit,
                    ..
                } => *paper_metres_per_unit += 0.0,
                ScaleProvenance::Measured { p0, p1, metres } => {
                    for v in p0.iter_mut().chain(p1.iter_mut()) {
                        *v += 0.0;
                    }
                    *metres += 0.0;
                }
                _ => {}
            }
        }
        if let Some(g) = &mut georeference {
            for v in g.affine.iter_mut().chain(
                g.control_points
                    .iter_mut()
                    .flat_map(|p| p.source.iter_mut().chain(p.world.iter_mut())),
            ) {
                *v += 0.0;
            }
            g.residual_max += 0.0;
        }
        Ok(Self {
            scale,
            georeference,
            external_id,
        })
    }

    /// The scale record.
    #[must_use]
    pub fn scale(&self) -> Option<&ScaleClaim> {
        self.scale.as_ref()
    }

    /// The georeference record.
    #[must_use]
    pub fn georeference(&self) -> Option<&Georeference> {
        self.georeference.as_ref()
    }

    /// The application's identifier for this calibration.
    #[must_use]
    pub fn external_id(&self) -> Option<&str> {
        self.external_id.as_deref()
    }
}

impl ScaleClaim {
    /// The scale two source points and the real distance between them give: `metres` over their
    /// length. No uncertainty and an empty note.
    ///
    /// # Errors
    ///
    /// [`CalibrationError::Degenerate`] when the points are too close to give a length,
    /// [`CalibrationError::OutOfRange`] for a coordinate that is not a finite number.
    pub fn measured(p0: [f64; 2], p1: [f64; 2], metres: f64) -> Result<Self, CalibrationError> {
        let length = measured_length(p0, p1)?;
        Ok(Self {
            metres_per_source_unit: metres / length,
            provenance: ScaleProvenance::Measured { p0, p1, metres },
            uncertainty: None,
            note: String::new(),
        })
    }
}

impl Georeference {
    /// The six affine numbers, in PVec's `[a b c d tx ty]` order, of a 3x3 matrix in row order
    /// (`[[a, b, tx], [c, d, ty], [0, 0, w]]`), divided through by `w`.
    ///
    /// # Errors
    ///
    /// [`CalibrationError::Projective`] when the last row has a perspective term,
    /// [`CalibrationError::Affine`] when `w` is zero or a number is not finite.
    pub fn affine_from_matrix(m: [[f64; 3]; 3]) -> Result<[f64; 6], CalibrationError> {
        if m.iter().flatten().any(|v| !v.is_finite()) {
            return Err(CalibrationError::Affine);
        }
        if m[2][0] != 0.0 || m[2][1] != 0.0 {
            return Err(CalibrationError::Projective);
        }
        let w = m[2][2];
        if w == 0.0 {
            return Err(CalibrationError::Affine);
        }
        Ok([
            m[0][0] / w,
            m[0][1] / w,
            m[1][0] / w,
            m[1][1] / w,
            m[0][2] / w,
            m[1][2] / w,
        ])
    }
}

/// NP-01: finite, and zero or normal.
fn normal(v: f64) -> bool {
    v.is_finite() && (v == 0.0 || v.abs() >= f64::MIN_POSITIVE)
}

/// NP-01 and NP-02 on a point.
fn coordinate(p: [f64; 2]) -> bool {
    p.iter().all(|&v| normal(v) && v.abs() <= MAX_ABS)
}

/// The length between two measured points, refused when NP-06 calls it degenerate. The square
/// root is the only function used, and it is correctly rounded, so the length is the same bits
/// everywhere.
fn measured_length(p0: [f64; 2], p1: [f64; 2]) -> Result<f64, CalibrationError> {
    if !coordinate(p0) || !coordinate(p1) {
        return Err(CalibrationError::OutOfRange {
            field: "measured point",
        });
    }
    let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
    let length = (dx * dx + dy * dy).sqrt();
    let mag = p0.iter().chain(&p1).fold(0.0f64, |m, v| m.max(v.abs()));
    // ulp(mag): the gap to the next larger f64.
    let ulp = f64::from_bits(mag.to_bits() + 1) - mag;
    if !length.is_finite() || length <= 1e-12f64.max(8.0 * ulp) {
        return Err(CalibrationError::Degenerate);
    }
    Ok(length)
}

/// CAL-03 and CAL-07: at most `max` bytes, no control character (LF and TAB only when `lines`),
/// no line or paragraph separator, no bidi override.
fn check_text(
    s: &str,
    max: usize,
    lines: bool,
    field: &'static str,
) -> Result<(), CalibrationError> {
    let bad = |c: char| {
        (c.is_control() && !(lines && (c == '\n' || c == '\t')))
            || matches!(c, '\u{2028}' | '\u{2029}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
    };
    if s.len() > max || s.chars().any(bad) {
        return Err(CalibrationError::Text { field });
    }
    Ok(())
}
