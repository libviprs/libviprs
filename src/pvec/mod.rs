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

#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod bounds;
pub mod classify;
#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod differential;
#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod dump;
#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod dwg_basic;
pub mod fidelity;
pub mod fill_policy;
#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod flatten;
pub mod limits;
#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod mapping;
#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod normalized;
pub mod ordering;
#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod pdf_bezier;
pub mod pdf_clip;
mod pdf_filters;
pub mod pdf_image;
mod pdf_image_decode;
pub mod pdf_vector;
#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod provenance;
pub mod provider;
#[cfg(feature = "pvec-write")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-write")))]
pub mod publish;
pub mod style;
pub mod text_policy;
#[cfg(feature = "pvec-normalized")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-normalized")))]
pub mod transform;
pub mod units;
pub mod view;
pub mod view_profile;
#[cfg(feature = "pvec-write")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-write")))]
pub mod writer;
pub mod xref;

#[cfg(feature = "dwg-acadrust")]
#[cfg_attr(docsrs, doc(cfg(feature = "dwg-acadrust")))]
pub mod dwg_acadrust;

#[cfg(feature = "pvec-spline")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-spline")))]
pub mod dwg_spline;

#[cfg(feature = "pvec-normalized")]
pub use bounds::{
    BoundsError, BoundsMethod, BoundsReport, EntityBounds, bounds_inputs, curve_bounds,
    pvec_entity_id,
};
pub use classify::{
    ClassSource, Classification, ClassificationReport, ElementClass, ElementRule, ElementRules,
    RuleMatch,
};
pub use fidelity::{
    Basis, FidelityClass, FidelityReport, ReasonCode, Tally, TallyError, classify_unmapped,
};
pub use limits::{Limit, LimitError, SourceGuard, SourceLimits};
pub use ordering::{
    EntityId, OrderedEntity, OrderedSource, OrderingBasis, OrderingPolicy, OrderingReport,
    canonical_cmp, order_entities,
};
pub use pdf_vector::{
    PdfDash, PdfFillRule, PdfImagePlacement, PdfPaintIntent, PdfPathEntity, PdfPathSegment,
    PdfPoint, PdfProvenance, PdfStrokeParams, PdfVectorError, PdfVectorLimits, PdfVectorPage,
    extract_page_paths, extract_page_paths_from_bytes,
};
pub use provider::{
    ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityFacts, EntityScope,
    EntityStyle, Facet, FacetCap, FacetSource, InstancePath, InstanceStep, LayerState, LayoutId,
    LayoutKind, LayoutState, Point3, ProvenanceStub, ProviderCaps, ProviderError, RawEntity,
    RawGeometry, RawVertex, SourceColour, SourceFact, SourceLineweight, SourceTransform,
    TextStyleRef, ViewportEntity, XrefAttachment,
};
pub use text_policy::{
    DwgTextCounts, PdfTextOperatorCounts, TextFontOutcome, TextReport, shx_outcome,
};

#[cfg(feature = "pvec-normalized")]
pub use normalized::{NormalizeError, Normalized2DGeometry, ProvenanceHandle, SourceBounds};
pub use units::{DrawingExtents, INSUNITS_TABLE, SourceSpace, SourceUnit};
pub use view::{
    Decision, EntityState, EntityStates, IgnoredByPolicy, PolicyReason, ViewportFreezeMode,
    ViewportFreezeNote, ViewportInput, VisibilityDisagreement, VisibilityFilter, VisibilityOutcome,
    select_layout,
};
pub use view_profile::{
    ExcludedEntity, Exclusion, InsertLayerRule, Knob, LayerStateRule, LayerVisibility,
    NestedInsertPolicy, SpaceSelection, ViewOutcome, ViewProfile, ViewProfileError, ViewReport,
    ViewedEntity, ViewportClipping, ViewportFreezeRule, XrefPolicy,
};
