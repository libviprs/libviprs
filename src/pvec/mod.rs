//! The analytic path's side of the DWG boundary (Epic N).
//!
//! `cad` is the tessellating path: a [`CadDecoder`](crate::cad::CadDecoder) pushes
//! primitives at a tiler, and the tiler flattens curves against a zoom-dependent
//! budget. This module is the other consumer. Normalization wants entities as the
//! drawing stores them (a circle is a centre and a radius, a spline keeps its
//! knots), plus the facts a rule engine needs (layer state, block names, which
//! INSERT chain a leaf was reached through), and it wants to ask for them through
//! a seam a different reader can sit behind.
//!
//! What is here is that seam and nothing that computes with it:
//!
//! ```text
//! DwgProvider::open / probe          one drawing, header sniffed or fully read
//!   layouts()  layers()              LayoutState, LayerState
//!   entities(EntityScope)  ->  EntityCursor  ->  RawEntity, one deterministic pass
//!   caps()                           ProviderCaps: what this provider can and cannot report
//! ```
//!
//! [`provider`] holds the traits and the boundary types. No ACadSharp, .NET, FFI
//! or `lopdf` type appears in any signature there, and `tests/pvec_provider_boundary.rs`
//! enumerates the public items and fails on a foreign path. The first provider,
//! `AcadrustProvider` in `dwg_acadrust`, is behind the `dwg-acadrust`
//! feature and the module compiles without it.
//!
//! `cad::CadDecoder` and `cad::Origin` are untouched: the tile path keeps the
//! decoder it has, and this module only reuses two of its plain value types
//! ([`CadSource`](crate::cad::CadSource) and [`ItemHandle`](crate::cad::ItemHandle)).

pub mod classify;
pub mod fidelity;
pub mod limits;
#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod normalized;
pub mod pdf_vector;
#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod provenance;
pub mod provider;
#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod transform;
pub mod units;
pub mod view;
pub mod view_profile;

#[cfg(feature = "dwg-acadrust")]
#[cfg_attr(docsrs, doc(cfg(feature = "dwg-acadrust")))]
pub mod dwg_acadrust;

#[cfg(feature = "pvec-spline")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-spline")))]
pub mod dwg_spline;

pub use classify::{
    ClassSource, Classification, ClassificationReport, ElementClass, ElementRule, ElementRules,
    RuleMatch,
};
pub use fidelity::{
    Basis, FidelityClass, FidelityReport, ReasonCode, Tally, TallyError, classify_unmapped,
};
pub use limits::{Limit, LimitError, SourceGuard, SourceLimits};
pub use pdf_vector::{
    PdfFillRule, PdfPaintIntent, PdfPathEntity, PdfPathSegment, PdfPoint, PdfProvenance,
    PdfVectorError, PdfVectorLimits, PdfVectorPage, extract_page_paths,
    extract_page_paths_from_bytes,
};
pub use provider::{
    ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityScope, Facet, FacetCap,
    FacetSource, InstancePath, InstanceStep, LayerState, LayoutId, LayoutKind, LayoutState, Point3,
    ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry, RawVertex, SourceFact,
    SourceTransform,
};

#[cfg(feature = "pvec-normalized")]
pub use normalized::{NormalizeError, Normalized2DGeometry, ProvenanceHandle, SourceBounds};
pub use units::{DrawingExtents, INSUNITS_TABLE, SourceSpace, SourceUnit};
pub use view_profile::{
    ExcludedEntity, Exclusion, InsertLayerRule, Knob, LayerStateRule, LayerVisibility,
    NestedInsertPolicy, SpaceSelection, ViewOutcome, ViewProfile, ViewProfileError, ViewReport,
    ViewedEntity, ViewportClipping, ViewportFreezeRule, XrefPolicy,
};
pub use view::{
    Decision, EntityState, EntityStates, IgnoredByPolicy, PolicyReason, ViewportFreezeMode,
    ViewportFreezeNote, ViewportInput, VisibilityDisagreement, VisibilityFilter,
    VisibilityOutcome, select_layout,
};
