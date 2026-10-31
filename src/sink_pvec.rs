//! Red stub (issue #1331): the PVec sink is not written yet.

use std::path::Path;

use pvec_facade::PvecWriter;
use thiserror::Error;

use crate::raster_pvec::pyramid::RasterPyramid;
use crate::sink::{SinkError, Tile, TileFormat, TileSink};

/// Stub.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PvecSinkError {
    /// Stub.
    #[error("stub")]
    Stub,
}

/// Stub.
pub struct PvecSink;

impl PvecSink {
    /// Stub.
    ///
    /// # Errors
    ///
    /// Always.
    pub fn new(
        _writer: PvecWriter,
        _pyramid: &RasterPyramid,
        _format: TileFormat,
    ) -> Result<Self, PvecSinkError> {
        Err(PvecSinkError::Stub)
    }

    /// Stub.
    #[must_use]
    pub fn into_writer(self) -> PvecWriter {
        unimplemented!()
    }

    /// Stub.
    ///
    /// # Errors
    ///
    /// Always.
    pub fn finish_to(self, _dest: &Path, _staging: &Path) -> Result<(), PvecSinkError> {
        Err(PvecSinkError::Stub)
    }
}

impl TileSink for PvecSink {
    fn write_tile(&self, _tile: &Tile) -> Result<(), SinkError> {
        Err(SinkError::Other("stub".into()))
    }
}
