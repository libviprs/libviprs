//! Stub, replaced by the implementation in the next commit (N5.2, #1081).

use super::fidelity::Classification;
use super::normalized::{NormalizeError, Normalized2DGeometry};
use super::pdf_vector::{PdfPathEntity, PdfProvenance, PdfVectorPage};

/// One normalized entity.
#[derive(Debug, Clone)]
pub struct PdfNormalizedEntity {
    /// The PVec record.
    pub geometry: Normalized2DGeometry,
    /// Its classification.
    pub classification: Classification,
    /// Subpath index.
    pub subpath: usize,
    /// Compound-path group id.
    pub compound_group: u64,
    /// PVec's refusal, if any.
    pub refusal: Option<NormalizeError>,
}

/// One subpath.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdfSubpath {
    /// Closed or not.
    pub closed: bool,
    /// First entity index.
    pub first_entity: usize,
    /// Entity count.
    pub entity_count: usize,
}

/// One painted path, normalized.
#[derive(Debug, Clone)]
pub struct PdfNormalizedPath {
    /// Group id.
    pub compound_group: u64,
    /// Entities.
    pub entities: Vec<PdfNormalizedEntity>,
    /// Subpaths.
    pub subpaths: Vec<PdfSubpath>,
    /// Source provenance.
    pub provenance: PdfProvenance,
}

/// Normalize one path.
pub fn normalize_path(path: &PdfPathEntity) -> PdfNormalizedPath {
    PdfNormalizedPath {
        compound_group: path.provenance.sequence,
        entities: Vec::new(),
        subpaths: Vec::new(),
        provenance: path.provenance.clone(),
    }
}

/// Normalize a page.
pub fn normalize_page(page: &PdfVectorPage) -> Vec<PdfNormalizedPath> {
    page.entities.iter().map(normalize_path).collect()
}
