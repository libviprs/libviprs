//! A PNG of a published PVec artifact (issue #1629): strokes only, black on white, at the width
//! the caller asks for.

use std::path::Path;

use thiserror::Error;

/// Why a thumbnail could not be drawn.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ThumbnailError {
    /// The width was zero.
    #[error("a thumbnail needs a width of at least one pixel")]
    ZeroWidth,
    /// The artifact could not be read.
    #[error("the artifact could not be read: {0}")]
    Artifact(String),
}

/// The artifact's hot streams as a PNG `width_px` wide, the height following the artifact's bounds.
///
/// # Errors
///
/// [`ThumbnailError`] for a zero width or an artifact that cannot be read.
pub fn thumbnail(artifact: &Path, width_px: u32) -> Result<Vec<u8>, ThumbnailError> {
    let _ = (artifact, width_px);
    todo!("#1629: the rasteriser")
}
