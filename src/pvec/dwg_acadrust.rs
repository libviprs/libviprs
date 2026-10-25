//! The first DWG provider: acadrust, behind the `dwg-acadrust` feature.
//!
//! This is the one place in libviprs that names acadrust. Everything that leaves it
//! is a [`DwgProvider`], an [`EntityCursor`] or one of the boundary types of
//! [`provider`](crate::pvec::provider): no `acadrust` type appears in a public
//! signature, and `tests/pvec_provider_boundary.rs` checks that.
//!
//! ```text
//! acadrust::CadDocument
//!   layers        ->  LayerState      (off, frozen, locked, as the table stores them)
//!   Layout objects ->  LayoutState     (model first, then paper layouts in tab order)
//!   block record's entities -> EntityCursor -> RawEntity   (storage order, not expanded)
//!   entity invisible flag  ->  EntityFacts      (own flag, and the provider's own answer)
//!   paper-space viewports  ->  ViewportEntity   (visibility, clip boundary, frozen layer names)
//! ```
//!
//! # Where the facts come from
//!
//! Every facet is read from acadrust's object model directly, which is the
//! [`FacetSource::DecoderDirect`] row of [`ProviderCaps`]. The existing
//! [`CadDecoder`](crate::cad::CadDecoder) adapter carries none of them (no layer
//! name, no block name, no insert chain), and it is untouched.
//!
//! # What this provider does not do
//!
//! It does not expand INSERTs, apply layer visibility, or flatten a curve. A spline
//! with no control points (a fit-point spline) is [`RawGeometry::Unmapped`], and so
//! is every class with no mapping yet: counted, with the class name, never dropped.
//! A reader panic on a hostile file is caught and reported as
//! [`ProviderError::Open`], because the open failure is the thing Gate 0 counts.
//!
//! # Supported DWG versions
//!
//! [`probe`](DwgProvider::probe) accepts the six byte stamps the existing decoder
//! accepts, [`SUPPORTED_DWG_VERSIONS`] (AC1014 through AC1032), because those are the
//! versions the corpus of `docs/spikes/acadrust-gate0.md` was run over.

use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::panic::{AssertUnwindSafe, catch_unwind};

use acadrust as acr;
use acr::objects::ObjectType;
use acr::{DwgReader, EntityType};

use crate::cad::{CadSource, ItemHandle};
use crate::pvec::provider::{
    ArrayGrid, Availability, BlockRef, DwgProvider, EntityCursor, EntityFacts, EntityScope,
    EntityStyle, Facet, FacetSource, InstancePath, LayerState, LayoutId, LayoutKind, LayoutState,
    Point3, ProvenanceStub, ProviderCaps, ProviderError, RawEntity, RawGeometry, RawVertex,
    SourceColour, SourceFact, SourceLineweight, SourceTransform, ViewportEntity,
};
use crate::pvec::units::SourceUnit;

/// What this provider calls itself in [`ProviderError`].
const NAME: &str = "acadrust";

/// The DWG version stamps accepted: R14, 2000, 2004, 2007, 2010, 2013 and 2018.
pub const SUPPORTED_DWG_VERSIONS: [&[u8; 6]; 7] = [
    b"AC1014", b"AC1015", b"AC1018", b"AC1021", b"AC1024", b"AC1027", b"AC1032",
];

fn stamp_supported(head: &[u8]) -> bool {
    head.len() >= 6 && SUPPORTED_DWG_VERSIONS.iter().any(|v| head[..6] == v[..])
}

fn read_head(source: CadSource<'_>) -> Result<Vec<u8>, ProviderError> {
    match source {
        CadSource::Bytes(b) => Ok(b[..b.len().min(6)].to_vec()),
        CadSource::Path(p) => {
            let mut head = Vec::with_capacity(6);
            let file = std::fs::File::open(p).map_err(|e| ProviderError::Open {
                provider: NAME,
                detail: format!("{}: {e}", p.display()),
            })?;
            file.take(6)
                .read_to_end(&mut head)
                .map_err(|e| ProviderError::Open {
                    provider: NAME,
                    detail: e.to_string(),
                })?;
            Ok(head)
        }
    }
}

/// The first provider.
pub struct AcadrustProvider {
    doc: acr::CadDocument,
    caps: ProviderCaps,
    layouts: Vec<LayoutState>,
}

impl AcadrustProvider {
    /// Every block definition's name, in the block table's order.
    ///
    /// Not part of [`DwgProvider`]: a consumer follows INSERTs by name and has no
    /// need of the list. It is here for a report that wants to say how many blocks
    /// a drawing exposes.
    #[must_use]
    pub fn block_names(&self) -> Vec<String> {
        self.doc
            .block_records
            .iter()
            .map(|b| b.name.clone())
            .collect()
    }

    /// The unit of the drawing's coordinates, read from the header's `$INSUNITS`.
    ///
    /// Header only: no inference from the coordinates and no default. The value is
    /// handed on as the drawing stores it, and the coordinates the cursor yields
    /// are not scaled by it.
    #[must_use]
    pub fn source_unit(&self) -> SourceUnit {
        SourceUnit::from_insunits(Some(self.doc.header.insertion_units))
    }

    fn capabilities() -> ProviderCaps {
        Facet::ALL.iter().fold(ProviderCaps::new(), |caps, f| {
            caps.with(*f, Availability::Reported, FacetSource::DecoderDirect)
        })
    }

    fn collect_layouts(doc: &acr::CadDocument) -> Vec<LayoutState> {
        let block_of = |handle: acr::Handle| {
            doc.block_records
                .iter()
                .find(|b| b.handle == handle)
                .map(|b| b.name.clone())
        };
        let mut found: Vec<(i16, u64, String, String)> = Vec::new();
        for o in doc.objects.values() {
            if let ObjectType::Layout(l) = o
                && let Some(block) = block_of(l.block_record)
            {
                found.push((l.tab_order, l.handle.value(), l.name.clone(), block));
            }
        }
        if found.is_empty() {
            // Drawings before R2000 have no Layout objects: the block table is all
            // there is, so name the layouts after their block records.
            for b in doc.block_records.iter() {
                let lower = b.name.to_ascii_lowercase();
                if lower.starts_with("*model_space") || lower.starts_with("*paper_space") {
                    let name = if lower.starts_with("*model_space") {
                        "Model".to_string()
                    } else {
                        b.name.clone()
                    };
                    found.push((0, b.handle.value(), name, b.name.clone()));
                }
            }
        }
        // The objects live in a hash map, so the order is made here, not inherited.
        let is_model = |block: &str| block.eq_ignore_ascii_case("*Model_Space");
        found.sort_by_key(|a| (!is_model(&a.3), a.0, a.1));
        found
            .into_iter()
            .enumerate()
            .map(|(i, (_, _, name, block))| LayoutState {
                id: LayoutId(u32::try_from(i).unwrap_or(u32::MAX)),
                name,
                kind: if is_model(&block) {
                    LayoutKind::Model
                } else {
                    LayoutKind::Paper
                },
                block_name: block,
            })
            .collect()
    }
}

impl DwgProvider for AcadrustProvider {
    fn probe(source: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(stamp_supported(&read_head(source)?))
    }

    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        let head = read_head(source)?;
        if !stamp_supported(&head) {
            return Err(ProviderError::Unrecognised {
                provider: NAME,
                detail: format!("version stamp {:?}", String::from_utf8_lossy(&head)),
            });
        }
        let result = catch_unwind(AssertUnwindSafe(|| match source {
            CadSource::Path(p) => DwgReader::from_file(p).and_then(|mut r| r.read()),
            CadSource::Bytes(b) => DwgReader::from_stream(Cursor::new(b)).read(),
        }));
        let doc = match result {
            Ok(Ok(doc)) => doc,
            Ok(Err(e)) => {
                return Err(ProviderError::Open {
                    provider: NAME,
                    detail: e.to_string(),
                });
            }
            Err(_) => {
                return Err(ProviderError::Open {
                    provider: NAME,
                    detail: "the reader panicked".into(),
                });
            }
        };
        let layouts = Self::collect_layouts(&doc);
        Ok(Self {
            doc,
            caps: Self::capabilities(),
            layouts,
        })
    }

    fn name(&self) -> &'static str {
        NAME
    }

    fn caps(&self) -> &ProviderCaps {
        &self.caps
    }

    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        Ok(self.layouts.clone())
    }

    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        Ok(self
            .doc
            .layers
            .iter()
            .map(|l| LayerState {
                name: l.name.clone(),
                handle: item_handle(l.handle),
                off: l.is_off(),
                frozen: l.is_frozen(),
                locked: l.is_locked(),
            })
            .collect())
    }

    fn entity_facts(
        &self,
        scope: &EntityScope,
    ) -> Result<BTreeMap<ItemHandle, EntityFacts>, ProviderError> {
        let (record, _) = self.resolve(scope)?;
        let mut out = BTreeMap::new();
        for handle in &record.entity_handles {
            let Some(e) = self.doc.get_entity(*handle) else {
                continue;
            };
            if matches!(e, EntityType::Block(_) | EntityType::BlockEnd(_)) {
                continue;
            }
            let Some(key) = item_handle(*handle) else {
                continue;
            };
            let common = e.common();
            // A drawing stores one layer per entity, so there is no second one to report.
            // That is a fact about the drawing, not a gap in the reader.
            let layer_hides = self
                .doc
                .layers
                .get(&common.layer)
                .is_some_and(|l| l.is_off() || l.is_frozen());
            out.insert(
                key,
                EntityFacts {
                    visible: SourceFact::Present(!common.invisible),
                    layer_override: SourceFact::Absent,
                    provider_visible: SourceFact::Present(!common.invisible && !layer_hides),
                },
            );
        }
        Ok(out)
    }

    fn viewports(&self) -> Result<Vec<ViewportEntity>, ProviderError> {
        let mut out = Vec::new();
        for layout in self.layouts.iter().filter(|l| l.kind == LayoutKind::Paper) {
            let Some(record) = self.doc.block_records.get(&layout.block_name) else {
                continue;
            };
            for handle in &record.entity_handles {
                let (Some(EntityType::Viewport(v)), Some(key)) =
                    (self.doc.get_entity(*handle), item_handle(*handle))
                else {
                    continue;
                };
                // Viewport id 1 is the paper-space view itself, not a window onto the model.
                if v.id == 1 {
                    continue;
                }
                out.push(ViewportEntity {
                    handle: key,
                    layout: layout.id,
                    visible: v.status.is_on && !v.common.invisible,
                    clip: self.viewport_clip(v),
                    frozen_layers: v
                        .frozen_layers
                        .iter()
                        .filter_map(|h| {
                            self.doc
                                .layers
                                .iter()
                                .find(|l| l.handle == *h)
                                .map(|l| l.name.clone())
                        })
                        .collect(),
                });
            }
        }
        Ok(out)
    }

    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let (record, path) = self.resolve(scope)?;
        Ok(Box::new(AcadrustCursor {
            doc: &self.doc,
            handles: record.entity_handles.iter(),
            path,
            base_points: BTreeMap::new(),
        }))
    }
}

impl AcadrustProvider {
    fn resolve(
        &self,
        scope: &EntityScope,
    ) -> Result<(&acr::tables::BlockRecord, InstancePath), ProviderError> {
        let (block, path) = match scope {
            EntityScope::Layout(id) => {
                let l = self
                    .layouts
                    .get(id.0 as usize)
                    .ok_or_else(|| ProviderError::NoSuchScope(format!("layout {}", id.0)))?;
                (l.block_name.clone(), InstancePath::root())
            }
            EntityScope::BlockDefinition { name, reached_via } => {
                (name.clone(), reached_via.clone())
            }
        };
        let record = self
            .doc
            .block_records
            .get(&block)
            .ok_or_else(|| ProviderError::NoSuchScope(format!("block {block}")))?;
        Ok((record, path))
    }

    /// The viewport's clip boundary: the polyline it is clipped to when it has one, else
    /// the rectangle its centre, width and height describe.
    fn viewport_clip(&self, v: &acr::entities::Viewport) -> Vec<[f64; 2]> {
        if !v.clip_boundary_handle.is_null() {
            match self.doc.get_entity(v.clip_boundary_handle) {
                Some(EntityType::LwPolyline(p)) => {
                    return p
                        .vertices
                        .iter()
                        .map(|q| [q.location.x, q.location.y])
                        .collect();
                }
                Some(EntityType::Polyline2D(p)) => {
                    return p
                        .vertices
                        .iter()
                        .map(|q| [q.location.x, q.location.y])
                        .collect();
                }
                _ => {}
            }
        }
        let (cx, cy) = (v.center.x, v.center.y);
        let (hw, hh) = (v.width / 2.0, v.height / 2.0);
        vec![
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
        ]
    }
}

struct AcadrustCursor<'a> {
    doc: &'a acr::CadDocument,
    handles: std::slice::Iter<'a, acr::Handle>,
    path: InstancePath,
    /// Block base points looked up so far, by block name.
    base_points: BTreeMap<String, [f64; 2]>,
}

impl EntityCursor for AcadrustCursor<'_> {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        for handle in self.handles.by_ref() {
            let Some(e) = self.doc.get_entity(*handle) else {
                return Ok(Some(RawEntity {
                    geometry: RawGeometry::Unmapped {
                        reason: "the block lists a handle the document has no entity for".into(),
                    },
                    provenance: ProvenanceStub {
                        handle: item_handle(*handle),
                        class_name: "UNKNOWN".into(),
                        block_path: self.path.clone(),
                        layer: String::new(),
                    },
                    style: EntityStyle::absent(),
                }));
            };
            if matches!(e, EntityType::Block(_) | EntityType::BlockEnd(_)) {
                continue;
            }
            return Ok(Some(self.convert(e)));
        }
        Ok(None)
    }
}

fn item_handle(h: acr::Handle) -> Option<ItemHandle> {
    if h.is_null() {
        None
    } else {
        Some(ItemHandle::new(h.value()))
    }
}

/// A stored colour as a fact. `ByLayer` and `None` carry no colour of their own.
fn fact_colour(c: acr::Color) -> SourceFact<SourceColour> {
    match c {
        acr::Color::Index(i) => SourceFact::Present(SourceColour::Aci(i)),
        acr::Color::Rgb { r, g, b } => SourceFact::Present(SourceColour::True { r, g, b }),
        acr::Color::ByBlock => SourceFact::Present(SourceColour::ByBlock),
        acr::Color::ByLayer | acr::Color::None => SourceFact::Absent,
    }
}

/// A stored lineweight as a fact. `ByLayer` carries no weight of its own, and a
/// negative value that is not one of the three markers is not a weight.
fn fact_lineweight(w: acr::LineWeight) -> SourceFact<SourceLineweight> {
    match w {
        acr::LineWeight::Value(v) => u16::try_from(v).map_or(SourceFact::Absent, |h| {
            SourceFact::Present(SourceLineweight::Hundredths(h))
        }),
        acr::LineWeight::Default => SourceFact::Present(SourceLineweight::Default),
        acr::LineWeight::ByBlock => SourceFact::Present(SourceLineweight::ByBlock),
        acr::LineWeight::ByLayer => SourceFact::Absent,
    }
}

fn p3(v: acr::Vector3) -> Point3 {
    [v.x, v.y, v.z]
}

impl AcadrustCursor<'_> {
    fn base_point(&mut self, block: &str) -> [f64; 2] {
        if let Some(b) = self.base_points.get(block) {
            return *b;
        }
        let b = self
            .doc
            .block_records
            .get(block)
            .map_or([0.0, 0.0], |r| [r.base_point.x, r.base_point.y]);
        self.base_points.insert(block.to_string(), b);
        b
    }

    /// The entity's colour and lineweight, BYLAYER resolved through the layer table.
    ///
    /// BYBLOCK is handed on as it is: this cursor sees one block definition and does
    /// not know the INSERT that is drawing it, so a layer whose own value is BYBLOCK
    /// gives `Present(ByBlock)` too. A BYLAYER entity on a layer the table does not
    /// have, or on one whose own value is BYLAYER or no colour, has nothing to
    /// resolve to and reports [`SourceFact::Absent`].
    fn style_of(&self, common: &acr::entities::EntityCommon) -> EntityStyle {
        let layer = self.doc.layers.get(&common.layer).or_else(|| {
            self.doc
                .layers
                .iter()
                .find(|l| l.name.eq_ignore_ascii_case(&common.layer))
        });
        let colour = match common.color {
            acr::Color::ByLayer => layer.map_or(SourceFact::Absent, |l| fact_colour(l.color)),
            c => fact_colour(c),
        };
        let lineweight = match common.line_weight {
            acr::LineWeight::ByLayer => {
                layer.map_or(SourceFact::Absent, |l| fact_lineweight(l.line_weight))
            }
            w => fact_lineweight(w),
        };
        EntityStyle { colour, lineweight }
    }

    fn convert(&mut self, e: &EntityType) -> RawEntity {
        let common = e.common();
        let class_name = e.as_entity().entity_type().to_string();
        let provenance = ProvenanceStub {
            handle: item_handle(common.handle),
            class_name: class_name.clone(),
            block_path: self.path.clone(),
            layer: common.layer.clone(),
        };
        let unmapped = |reason: String| RawGeometry::Unmapped { reason };
        let geometry = match e {
            EntityType::Line(l) => RawGeometry::Line {
                start: p3(l.start),
                end: p3(l.end),
            },
            EntityType::Circle(c) => RawGeometry::Circle {
                center: p3(c.center),
                radius: c.radius,
                normal: p3(c.normal),
            },
            EntityType::Arc(a) => RawGeometry::Arc {
                center: p3(a.center),
                radius: a.radius,
                start_angle: a.start_angle,
                end_angle: a.end_angle,
                normal: p3(a.normal),
            },
            EntityType::Ellipse(el) => RawGeometry::Ellipse {
                center: p3(el.center),
                major_axis: p3(el.major_axis),
                minor_ratio: el.minor_axis_ratio,
                start_parameter: el.start_parameter,
                end_parameter: el.end_parameter,
                normal: p3(el.normal),
            },
            EntityType::LwPolyline(p) => RawGeometry::Polyline {
                vertices: p
                    .vertices
                    .iter()
                    .map(|v| RawVertex {
                        x: v.location.x,
                        y: v.location.y,
                        bulge: v.bulge,
                    })
                    .collect(),
                closed: p.is_closed,
                elevation: p.elevation,
                normal: p3(p.normal),
            },
            EntityType::Polyline2D(p) => RawGeometry::Polyline {
                vertices: p
                    .vertices
                    .iter()
                    .map(|v| RawVertex {
                        x: v.location.x,
                        y: v.location.y,
                        bulge: v.bulge,
                    })
                    .collect(),
                closed: p.flags.is_closed(),
                elevation: p.elevation,
                normal: p3(p.normal),
            },
            EntityType::Spline(s) => match u32::try_from(s.degree) {
                _ if s.control_points.is_empty() => unmapped(
                    "a spline with no control points (fit points only) has no control polygon to report"
                        .into(),
                ),
                Err(_) => unmapped(format!("a spline of degree {}", s.degree)),
                Ok(degree) => RawGeometry::Spline {
                    degree,
                    control_points: s.control_points.iter().copied().map(p3).collect(),
                    knots: if s.knots.is_empty() {
                        SourceFact::Absent
                    } else {
                        SourceFact::Present(s.knots.clone())
                    },
                    weights: if s.weights.is_empty() {
                        SourceFact::Absent
                    } else {
                        SourceFact::Present(s.weights.clone())
                    },
                    periodic: SourceFact::Present(s.flags.periodic),
                    closed: s.flags.closed,
                },
            },
            EntityType::Insert(i) => {
                let base = self.base_point(&i.block_name);
                let (sin, cos) = i.rotation.sin_cos();
                let (sx, sy) = (i.x_scale(), i.y_scale());
                let (m00, m01, m10, m11) = (cos * sx, -sin * sy, sin * sx, cos * sy);
                let tx = i.insert_point.x - (m00 * base[0] + m01 * base[1]);
                let ty = i.insert_point.y - (m10 * base[0] + m11 * base[1]);
                match SourceTransform::new([m00, m01, m10, m11, tx, ty]) {
                    Ok(transform) => RawGeometry::Insert(BlockRef {
                        block_name: i.block_name.clone(),
                        transform,
                        normal: p3(i.normal),
                        grid: ArrayGrid {
                            rows: u32::from(i.row_count).max(1),
                            columns: u32::from(i.column_count).max(1),
                            row_spacing: i.row_spacing,
                            column_spacing: i.column_spacing,
                        },
                    }),
                    Err(err) => unmapped(format!("an INSERT of {}: {err}", i.block_name)),
                }
            }
            _ => unmapped(format!("no mapping for {class_name}")),
        };
        let style = self.style_of(common);
        RawEntity {
            geometry,
            provenance,
            style,
        }
    }
}
