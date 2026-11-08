//! The retired georeference sidecar (issues #1415 and #1554): a reader for one minor release, so a
//! drawing converted before the calibration moved into the artifact can still be placed.
//!
//! A conversion used to write `<dest>.georef.json` beside the artifact: the drawing box, the unit
//! and its evidence, an opaque `calibration_id` naming a record the application holds, and
//! source-space control points with no world side. Since #1552 the calibration is written into the
//! artifact's calibration block, and since #1554 no sidecar is written at all.
//!
//! [`import_sidecar`] brings an old one in: it reads the sidecar, asks the application's resolver
//! for the calibration its `calibration_id` names (the sidecar holds no world coordinate, so it
//! cannot become a georeference by itself), and writes that calibration into the artifact with the
//! id as its `external_id`. When there is no id, or nothing resolves, nothing is written.
//! [`GeoreferenceHandoff`] reads the file and is deprecated with it; both go in the next minor
//! release.
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

use super::calibration::{Calibration, CalibrationError};
use crate::pvec::writer::WriteError;

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

/// Schema 1 of the sidecar.
#[deprecated(note = "the georeference sidecar is retired; use `import_sidecar` to bring one in")]
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
#[deprecated(note = "the georeference sidecar is retired; use `import_sidecar` to bring one in")]
#[allow(deprecated)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "schema_version")]
pub enum GeoreferenceHandoff {
    /// Schema 1.
    #[serde(rename = "1")]
    V1(HandoffV1),
}

/// Why a sidecar could not be read or imported.
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
    /// The resolved calibration, with the sidecar's id as its `external_id`, was refused.
    #[error("the resolved calibration was refused")]
    Calibration(#[source] CalibrationError),
    /// The artifact could not be rewritten with the calibration.
    #[error("the artifact could not be rewritten with the calibration")]
    Rewrite(#[source] WriteError),
}

/// What [`import_sidecar`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SidecarImport {
    /// The resolved calibration is in the artifact, with the sidecar's id as its `external_id`.
    Imported,
    /// The sidecar names no calibration, so there was nothing to import. Nothing was written.
    NoCalibrationId,
    /// The resolver knew no calibration by this id. Nothing was written.
    Unresolved(String),
}

/// Brings an old `.georef.json` sidecar into the artifact at `artifact` (issue #1554): reads it,
/// hands its `calibration_id` to `resolve`, and writes what comes back into the artifact's
/// calibration block, keeping the id as its `external_id`, through [`recalibrate`](super::recalibrate).
/// The sidecar's control points have no world side and are not carried over; the resolved
/// calibration's are. The sidecar itself is left where it is.
///
/// # Errors
///
/// [`GeoreferenceError`] for a sidecar that cannot be read or has another schema version, a resolved
/// calibration refused with the id attached, or an artifact that cannot be rewritten. Nothing is
/// written then.
#[allow(deprecated)]
pub fn import_sidecar(
    artifact: &Path,
    sidecar: &Path,
    resolve: impl FnOnce(&str) -> Option<Calibration>,
) -> Result<SidecarImport, GeoreferenceError> {
    let GeoreferenceHandoff::V1(v1) = GeoreferenceHandoff::read_from(sidecar)?;
    let Some(id) = v1.calibration_id else {
        return Ok(SidecarImport::NoCalibrationId);
    };
    let Some(found) = resolve(&id) else {
        return Ok(SidecarImport::Unresolved(id));
    };
    let calibration = Calibration::new(
        found.scale().cloned(),
        found.georeference().cloned(),
        Some(id),
    )
    .map_err(GeoreferenceError::Calibration)?;
    super::recalibrate(artifact, Some(&calibration)).map_err(GeoreferenceError::Rewrite)?;
    Ok(SidecarImport::Imported)
}

#[allow(deprecated)]
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

    /// Reads the sidecar at `path`.
    ///
    /// # Errors
    ///
    /// [`GeoreferenceError`].
    pub fn read_from(path: &Path) -> Result<Self, GeoreferenceError> {
        Self::from_json_slice(&fs::read(path).map_err(GeoreferenceError::Io)?)
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
