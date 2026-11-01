//! Atomic publication of a PVec artifact and its report (issue #1314, N6.8).
//!
//! STUB: red tests first.

use std::path::{Path, PathBuf};

use crate::cancel::CancelToken;

/// A point in the publish, reported before the step runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Before the artifact's `.pvec.partial` exists.
    ArtifactWriting,
    /// Artifact written, not yet synced.
    ArtifactSyncing,
    /// Before the report's partial exists.
    ReportWriting,
    /// Report written, not yet synced.
    ReportSyncing,
    /// Both synced, before the renames.
    Committing,
    /// Both renamed.
    Committed,
}

/// Why a publish failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum PublishError {
    /// Not written yet.
    Unimplemented,
    /// Cancelled.
    Cancelled,
    /// The bytes are not a complete artifact.
    NotAnArtifact(pvec_facade::PvecError),
    /// IO.
    Io(std::io::Error),
}
impl std::fmt::Display for PublishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "publish")
    }
}
impl std::error::Error for PublishError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PublishError::Io(e) => Some(e),
            PublishError::NotAnArtifact(e) => Some(e),
            _ => None,
        }
    }
}

/// Why a published artifact could not be opened.
#[derive(Debug)]
#[non_exhaustive]
pub enum OpenError {
    /// Not written yet.
    Unimplemented,
    /// A partial.
    Partial(PathBuf),
    /// PVec refused.
    Pvec(pvec_facade::PvecError),
}

/// The report's path for an artifact at `dest`.
pub fn report_path(dest: &Path) -> PathBuf {
    dest.to_path_buf()
}

/// Publish.
///
/// # Errors
///
/// [`PublishError`].
pub fn publish(
    _dest: &Path,
    _artifact: &[u8],
    _report: &[u8],
    _cancel: &CancelToken,
    _at: &mut dyn FnMut(Stage),
) -> Result<(), PublishError> {
    Err(PublishError::Unimplemented)
}

/// Open.
///
/// # Errors
///
/// [`OpenError`].
pub fn open_published(_path: &Path) -> Result<pvec_facade::PvecReader, OpenError> {
    Err(OpenError::Unimplemented)
}
