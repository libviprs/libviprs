//! The georeference handoff (issue #1415, N7.3): what libviprs gives an application so a drawing
//! can be placed on a map without a GIS transform ever entering PVec coordinates.
//!
//! `src/geo.rs` is raster georeferencing and belongs to the pyramid path. A drawing has no pixel
//! grid, its geometry is in source CAD units, and the epic keeps georeferencing out of source
//! geometry. So a conversion writes one small sidecar beside the artifact, `<dest>.georef.json`,
//! and the report names it.
//!
//! # The rule
//!
//! PVec coordinates are always source space. libviprs never writes a `GeoTransform`, a CRS string
//! or a projected coordinate into the artifact, and the sidecar carries none either: its control
//! points are drawing coordinates with no world coordinate attached, and the calibration is an
//! opaque identifier naming a record the application holds. Re-calibrating a converted drawing
//! rewrites the sidecar ([`GeoreferenceHandoff::recalibrate`]) and touches nothing else, so the
//! artifact's bytes, and its checksum, do not move.
//!
//! # The sidecar
//!
//! Versioned the way `src/manifest.rs` is: `schema_version` is the serde tag, a file with another
//! version is refused with [`GeoreferenceError::UnknownVersion`] and never parsed leniently.
//!
//! | field | meaning |
//! |---|---|
//! | `bounds` | The exact drawing box, `f64`, the one the artifact's manifest declares (N6.4). Absent for a drawing with no geometry. |
//! | `unit.source` | The unit the drawing's header declares, `Unknown` when the provider reports none. Never defaulted. |
//! | `unit.insunits` | The header's `$INSUNITS` code, absent when there is none. |
//! | `unit.physical` | Whether the source unit is a real length unit. |
//! | `unit.evidence` | What the source unit was read from, in words. |
//! | `unit.applied` | The unit the artifact declares. It differs from `source` only when the caller overrode it. |
//! | `calibration_id` | The caller's opaque calibration identifier, absent when none was given. |
//! | `control_points` | Source-space control points, drawing coordinates only. |

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::pvec::SourceUnit;

/// Where the sidecar for the artifact at `dest` is published: `dest` with `.georef.json` appended.
#[must_use]
pub fn georef_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".georef.json");
    PathBuf::from(s)
}

/// An axis-aligned box in source space.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SourceBox {
    /// The smallest corner, `[x, y]`.
    pub min: [f64; 2],
    /// The largest corner, `[x, y]`.
    pub max: [f64; 2],
}

/// The unit of the drawing and the unit the artifact declares.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitRecord {
    /// The header's unit, or `Unknown`.
    pub source: String,
    /// The header's `$INSUNITS` code, when there is one.
    pub insunits: Option<i16>,
    /// Whether `source` is a real length unit.
    pub physical: bool,
    /// What `source` was read from.
    pub evidence: String,
    /// The unit the artifact declares.
    pub applied: String,
}

impl UnitRecord {
    /// The record for a drawing whose header says `source`, written as `applied`.
    #[must_use]
    pub fn new(source: SourceUnit, applied: SourceUnit) -> Self {
        let insunits = source.insunits_code();
        UnitRecord {
            source: source.name().to_owned(),
            insunits,
            physical: source.is_physical(),
            evidence: match insunits {
                Some(code) => format!("the drawing header declares $INSUNITS = {code}"),
                None => "the provider reports no unit for this drawing".to_owned(),
            },
            applied: applied.name().to_owned(),
        }
    }
}

/// Schema 1 of the sidecar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HandoffV1 {
    /// The exact drawing box.
    pub bounds: Option<SourceBox>,
    /// The unit and its evidence.
    pub unit: UnitRecord,
    /// The caller's opaque calibration identifier.
    pub calibration_id: Option<String>,
    /// Source-space control points.
    pub control_points: Vec<[f64; 2]>,
}

/// The versioned sidecar. `schema_version` is the serde tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "schema_version")]
pub enum GeoreferenceHandoff {
    /// Schema 1.
    #[serde(rename = "1")]
    V1(HandoffV1),
}

/// Why a sidecar was not written, read or built.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GeoreferenceError {
    /// The file could not be read or written.
    #[error("the georeference sidecar could not be read or written")]
    Io(#[source] io::Error),
    /// The JSON is malformed.
    #[error("the georeference sidecar is not valid JSON of its schema")]
    Json(#[source] serde_json::Error),
    /// The file names a schema version this reader does not know.
    #[error("the georeference sidecar has the unknown schema_version {0:?}")]
    UnknownVersion(String),
    /// A control point has a coordinate that is not finite.
    #[error("control point {index} has a coordinate that is not finite")]
    NonFinite {
        /// The point's index.
        index: usize,
    },
}

impl GeoreferenceHandoff {
    /// The schema 1 payload.
    #[must_use]
    pub fn as_v1(&self) -> &HandoffV1 {
        match self {
            GeoreferenceHandoff::V1(v) => v,
        }
    }

    /// The sidecar as pretty JSON. The same sidecar gives the same bytes.
    ///
    /// # Errors
    ///
    /// [`GeoreferenceError::Json`] if serialising fails.
    pub fn to_json_string(&self) -> Result<String, GeoreferenceError> {
        serde_json::to_string_pretty(self).map_err(GeoreferenceError::Json)
    }

    /// Reads a sidecar, refusing a `schema_version` other than `"1"`.
    ///
    /// # Errors
    ///
    /// [`GeoreferenceError::UnknownVersion`] for another version, [`GeoreferenceError::Json`] for
    /// malformed JSON.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, GeoreferenceError> {
        let value: serde_json::Value =
            serde_json::from_slice(bytes).map_err(GeoreferenceError::Json)?;
        match value.get("schema_version").and_then(|v| v.as_str()) {
            Some("1") => serde_json::from_value(value).map_err(GeoreferenceError::Json),
            other => Err(GeoreferenceError::UnknownVersion(
                other.unwrap_or("").to_owned(),
            )),
        }
    }

    /// Writes the sidecar to `path`, replacing it atomically.
    ///
    /// # Errors
    ///
    /// [`GeoreferenceError`].
    pub fn write_to(&self, path: &Path) -> Result<(), GeoreferenceError> {
        let json = self.to_json_string()?;
        crate::resume::atomic_write(path, json.as_bytes()).map_err(GeoreferenceError::Io)
    }

    /// Reads the sidecar at `path`.
    ///
    /// # Errors
    ///
    /// [`GeoreferenceError`].
    pub fn read_from(path: &Path) -> Result<Self, GeoreferenceError> {
        Self::from_json_slice(&fs::read(path).map_err(GeoreferenceError::Io)?)
    }

    /// Re-calibrates an already converted drawing: rewrites the sidecar at `path` with a new
    /// calibration and control points, and nothing else.
    ///
    /// # Errors
    ///
    /// [`GeoreferenceError`], including [`GeoreferenceError::NonFinite`] for a control point.
    pub fn recalibrate(
        path: &Path,
        calibration_id: Option<String>,
        control_points: Vec<[f64; 2]>,
    ) -> Result<(), GeoreferenceError> {
        check_finite(&control_points)?;
        let GeoreferenceHandoff::V1(mut v1) = Self::read_from(path)?;
        v1.calibration_id = calibration_id;
        v1.control_points = control_points;
        GeoreferenceHandoff::V1(v1).write_to(path)
    }
}

/// The control points a handoff may carry: finite, as P1.2 requires of every source coordinate.
pub(super) fn check_finite(points: &[[f64; 2]]) -> Result<(), GeoreferenceError> {
    match points
        .iter()
        .position(|p| !p[0].is_finite() || !p[1].is_finite())
    {
        Some(index) => Err(GeoreferenceError::NonFinite { index }),
        None => Ok(()),
    }
}
