//! The acadrust DWG provider behind [`CadDecoder`], behind the `dwg-acadrust` feature (issue #1485).
//!
//! acadrust is the default DWG provider, and it is pure Rust: no native library, no loader, no
//! handshake. This module does not read acadrust a second way. It sits on
//! [`AcadrustProvider`], the `pvec::DwgProvider` the PVec path already uses, and turns what that
//! provider reports into [`Primitive`]s:
//!
//! ```text
//! AcadrustProvider  ->  BlockResolver (INSERTs expanded, bounded)  ->  FlatEntity + matrix
//!                                                                  ->  Primitive  ->  sink
//! ```
//!
//! # What crosses, and what is said about the rest
//!
//! Lines, polylines (bulges verbatim), arcs, circles, ellipses, splines and text cross. So do the
//! loops of a HATCH (as polygons, when their edges are straight, circular or a polyline), the
//! lines of a LEADER or MULTILEADER, the text of a MULTILEADER or TOLERANCE, and the visible
//! attributes of an INSERT. The
//! block transform of an INSERT is applied to the geometry, so a primitive from an expanded
//! INSERT already sits in the view's coordinates ([`Origin::from_expanded_insert`] says so).
//! A transform that keeps shapes (a rotation, a uniform scale, a mirror) is exact for every
//! kind. A transform that does not (an INSERT with unequal X and Y scale) is exact for lines,
//! open polylines without bulges and splines, and refused for the curves, with
//! [`DiagnosticCode::NON_UNIFORM_BLOCK_SCALE`]: an arc that has become an ellipse is not an arc
//! any more, and a made-up approximation would be a silent error.
//!
//! Anything this reader does not flatten (solids, points, mlines, images...) is
//! reported with [`DiagnosticCode::UNSUPPORTED_ENTITY`] and its class name, never dropped
//! quietly. An entity with an extrusion normal other than +Z inside an INSERT is reported the
//! same way.
//!
//! # Layers
//!
//! An entity on an off or frozen layer is not drawn, nor is an entity inside an INSERT that sits
//! on one. An entity on layer `0` inside a block takes the visibility of the INSERT, which is how
//! the drawing's own application treats it.
//!
//! # Limits
//!
//! The INSERT walk always runs under the default [`FlattenBounds`] (depth, instance count, grid
//! size), so a hostile block tree is refused with a typed error. [`AcadrustDecoder::hardened`]
//! adds the 256 MiB input cap the ACadSharp provider's hardened profile has, checked before the
//! file is read. acadrust has no streaming reader, so the input cap is the one limit that has to
//! be applied up front.

use std::collections::HashMap;

use super::text::{decode_escapes, display_text};
use super::{
    Arc, CadDecoder, CadDrawing, CadError, CadSource, CadView, Circle, DecodeReport, Diagnostic,
    DiagnosticCode, Ellipse, ItemHandle, Line, Origin, Polygon, Polyline, Primitive, PrimitiveSink,
    Spline, Text,
};
use crate::pvec::dwg_acadrust::{AcadrustProvider, TextFact};
use crate::pvec::flatten::{BlockResolver, FlatEntity, FlattenBounds, LineageStore, SkipReason};
use crate::pvec::provenance::NoProvenance;
use crate::pvec::{DwgProvider, EntityScope, InstancePath, LayoutKind, ProviderError, RawGeometry};

const NAME: &str = "acadrust";

/// The input size the hardened profile refuses, which is the ACadSharp profile's cap.
const HARDENED_MAX_INPUT_BYTES: u64 = 256 << 20;

/// A DWG reader that needs nothing installed.
#[derive(Debug, Clone, Copy, Default)]
pub struct AcadrustDecoder {
    max_input_bytes: Option<u64>,
}

impl AcadrustDecoder {
    /// A decoder with no input cap, right for a drawing you made yourself.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_input_bytes: None,
        }
    }

    /// A decoder that refuses an input of more than 256 MiB before reading it, right for a
    /// drawing a stranger uploaded.
    #[must_use]
    pub const fn hardened() -> Self {
        Self {
            max_input_bytes: Some(HARDENED_MAX_INPUT_BYTES),
        }
    }
}

fn translate(error: ProviderError) -> CadError {
    match error {
        ProviderError::Unrecognised { .. } => CadError::UnrecognisedSource { decoder: NAME },
        other => CadError::provider(NAME, other),
    }
}

impl CadDecoder for AcadrustDecoder {
    fn probe(&self, source: CadSource<'_>) -> Result<bool, CadError> {
        match AcadrustProvider::probe(source) {
            Ok(found) => Ok(found),
            // The provider reports a file it cannot open as an `Open` error, which the contract
            // says is `CadError::Io` for a probe.
            Err(ProviderError::Open { detail, .. }) => Err(std::io::Error::other(detail).into()),
            Err(other) => Err(translate(other)),
        }
    }

    fn open<'a>(&'a self, source: CadSource<'a>) -> Result<Box<dyn CadDrawing + 'a>, CadError> {
        if let Some(cap) = self.max_input_bytes {
            let size = match source {
                CadSource::Bytes(b) => b.len() as u64,
                CadSource::Path(p) => std::fs::metadata(p)?.len(),
            };
            if size > cap {
                return Err(CadError::provider(
                    NAME,
                    format!("LimitExceeded: the input is {size} bytes and the cap is {cap}"),
                ));
            }
        }
        let provider = AcadrustProvider::open(source).map_err(translate)?;
        Ok(Box::new(AcadrustDrawing { provider }))
    }
}

struct AcadrustDrawing {
    provider: AcadrustProvider,
}

/// Why the drawing would hide an entity.
#[derive(Debug, Clone, Copy, Default)]
struct Hidden {
    /// The entity's own invisible flag.
    flag: bool,
    /// Its layer is off or frozen.
    layer: bool,
}

impl AcadrustDrawing {
    fn scopes(&self) -> Result<Vec<EntityScope>, CadError> {
        let mut scopes: Vec<EntityScope> = self
            .provider
            .layouts()
            .map_err(translate)?
            .iter()
            .map(|l| EntityScope::Layout(l.id))
            .collect();
        scopes.extend(self.provider.block_names().into_iter().map(|name| {
            EntityScope::BlockDefinition {
                name,
                reached_via: InstancePath::root(),
            }
        }));
        Ok(scopes)
    }

    fn entity_count(&self, scope: &EntityScope) -> u64 {
        let Ok(mut cursor) = self.provider.entities(scope) else {
            return 0;
        };
        let mut n = 0;
        while matches!(cursor.next_entity(), Ok(Some(_))) {
            n += 1;
        }
        n
    }

    fn hidden(&self) -> Result<HashMap<ItemHandle, Hidden>, CadError> {
        let mut out = HashMap::new();
        for scope in self.scopes()? {
            let Ok(facts) = self.provider.entity_facts(&scope) else {
                continue;
            };
            for (handle, f) in facts {
                let flag = f.visible.present() == Some(&false);
                let layer = !flag && f.provider_visible.present() == Some(&false);
                out.insert(handle, Hidden { flag, layer });
            }
        }
        Ok(out)
    }
}

impl CadDrawing for AcadrustDrawing {
    fn views(&self) -> Result<Vec<CadView>, CadError> {
        let layouts = self.provider.layouts().map_err(translate)?;
        Ok(layouts
            .iter()
            .map(|l| {
                let model = l.kind == LayoutKind::Model;
                CadView::new(
                    l.id.0,
                    u32::from(!model),
                    if model {
                        self.provider.model_space_extents()
                    } else {
                        [1e20, 1e20, -1e20, -1e20]
                    },
                    self.entity_count(&EntityScope::Layout(l.id)),
                    l.name.clone(),
                )
            })
            .collect())
    }

    fn decode(
        &self,
        view: u32,
        sink: &mut dyn PrimitiveSink,
        report: &mut DecodeReport,
    ) -> Result<(), CadError> {
        let layouts = self.provider.layouts().map_err(translate)?;
        let Some(layout) = layouts.iter().find(|l| l.id.0 == view) else {
            return Err(CadError::NoSuchView {
                index: view,
                views: u32::try_from(layouts.len()).unwrap_or(u32::MAX),
            });
        };
        let flat = BlockResolver::new(&self.provider, FlattenBounds::default())
            .flatten(
                &EntityScope::Layout(layout.id),
                NAME,
                &layout.name,
                &mut NoProvenance,
                &mut LineageStore::new(),
            )
            .map_err(|e| {
                report.push(
                    Diagnostic::from_code(DiagnosticCode::DECODE_FAILED, e.to_string())
                        .with_view(view),
                );
                CadError::provider(NAME, e.to_string())
            })?;

        for skipped in &flat.skipped {
            let code = match skipped.reason {
                SkipReason::NormalNotZ => DiagnosticCode::UNSUPPORTED_ENTITY,
                _ => DiagnosticCode::UNRESOLVED_BLOCK,
            };
            let mut d = Diagnostic::from_code(
                code,
                format!(
                    "the INSERT of `{}` was not expanded ({})",
                    skipped.block_name,
                    skipped.reason.code()
                ),
            )
            .with_view(view);
            if let Some(h) = skipped.insert_handle {
                d = d.with_entity(h);
            }
            report.push(d);
        }

        let hidden = self.hidden()?;
        let mut run = Run {
            view,
            hidden: &hidden,
            sink,
            report,
        };
        self.emit(&flat.entities, IDENTITY, true, &mut run)?;
        self.emit_attributes(&EntityScope::Layout(layout.id), &mut run)?;
        report.mark_complete();
        sink.finish()
    }
}

const IDENTITY: [f64; 6] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// What one decode hands each leaf: the view, which entities the drawing hides, and where the
/// result goes.
struct Run<'a> {
    view: u32,
    hidden: &'a HashMap<ItemHandle, Hidden>,
    sink: &'a mut dyn PrimitiveSink,
    report: &'a mut DecodeReport,
}

/// `outer` after `inner`: the map that does `inner` first.
fn compose(outer: [f64; 6], inner: [f64; 6]) -> [f64; 6] {
    let [oa, ob, oc, od, otx, oty] = outer;
    let [ia, ib, ic, id, itx, ity] = inner;
    [
        oa * ia + ob * ic,
        oa * ib + ob * id,
        oc * ia + od * ic,
        oc * ib + od * id,
        oa * itx + ob * ity + otx,
        oc * itx + od * ity + oty,
    ]
}

impl AcadrustDrawing {
    /// Sends every visible leaf to the sink, `outer` applied after the leaf's own map. A
    /// DIMENSION is a leaf with no geometry of its own: its lines and text live in the anonymous
    /// block it names, which is walked once more (`nest` is false inside it, so a block that
    /// names itself cannot loop).
    fn emit(
        &self,
        leaves: &[FlatEntity],
        outer: [f64; 6],
        nest: bool,
        run: &mut Run<'_>,
    ) -> Result<(), CadError> {
        let state = |h: &ItemHandle| run.hidden.get(h).copied().unwrap_or_default();
        for leaf in leaves {
            let prov = &leaf.entity.provenance;
            let ancestors = prov
                .block_path
                .steps()
                .iter()
                .map(|s| state(&s.insert_handle));
            let own = prov.handle.as_ref().map(state).unwrap_or_default();
            // An entity on layer 0 inside a block takes its INSERT's layer, so its own layer's
            // state does not hide it. Its invisible flag still does.
            let takes_insert_layer = !prov.block_path.is_root() && prov.layer == "0";
            let hides = own.flag
                || (own.layer && !takes_insert_layer)
                || ancestors.into_iter().any(|a| a.flag || a.layer);
            if hides {
                continue;
            }
            let matrix = compose(outer, leaf.matrix());
            if nest
                && let Some(block) = self.dimension_block(leaf)
                && let Some(handle) = prov.handle
            {
                self.emit_dimension_block(&block, handle, leaf, matrix, run)?;
                continue;
            }
            if let Some((drawn, notes)) = convert_drawn(&self.provider, leaf, matrix) {
                for p in drawn {
                    let kind = p.kind();
                    run.sink.primitive(p)?;
                    run.report.record(kind);
                }
                for d in notes {
                    let mut d = d.with_view(run.view);
                    if let Some(h) = prov.handle {
                        d = d.with_entity(h);
                    }
                    run.report.push(d);
                }
                continue;
            }
            match convert(&self.provider, leaf, matrix) {
                Ok(Some(p)) => {
                    let kind = p.kind();
                    run.sink.primitive(p)?;
                    run.report.record(kind);
                }
                Ok(None) => {}
                Err(d) => {
                    let mut d = d.with_view(run.view);
                    if let Some(h) = prov.handle {
                        d = d.with_entity(h);
                    }
                    run.report.push(d);
                }
            }
        }
        Ok(())
    }

    /// The visible attributes of each INSERT that sits directly in the view. The block walk
    /// expands an INSERT into its definition's entities, which do not include the attribute
    /// values the INSERT carries, so they are read from the INSERT itself. They are stored in the
    /// view's own coordinates.
    fn emit_attributes(&self, scope: &EntityScope, run: &mut Run<'_>) -> Result<(), CadError> {
        let Ok(mut cursor) = self.provider.entities(scope) else {
            return Ok(());
        };
        while let Ok(Some(e)) = cursor.next_entity() {
            let (RawGeometry::Insert(_), Some(handle)) = (&e.geometry, e.provenance.handle) else {
                continue;
            };
            let state = run.hidden.get(&handle).copied().unwrap_or_default();
            if state.flag || state.layer {
                continue;
            }
            let map = Map::new(IDENTITY);
            let origin = Origin::from_handle(handle);
            for fact in self.provider.attribute_texts(handle) {
                match text_primitive(&fact, &map, origin, "ATTRIB") {
                    Ok(Some(p)) => {
                        let kind = p.kind();
                        run.sink.primitive(p)?;
                        run.report.record(kind);
                    }
                    Ok(None) => {}
                    Err(d) => run.report.push(d.with_view(run.view).with_entity(handle)),
                }
            }
        }
        Ok(())
    }

    fn dimension_block(&self, leaf: &FlatEntity) -> Option<String> {
        if !leaf.entity.provenance.class_name.starts_with("DIMENSION") {
            return None;
        }
        self.provider
            .dimension_block(leaf.entity.provenance.handle?)
    }

    fn emit_dimension_block(
        &self,
        block: &str,
        handle: ItemHandle,
        leaf: &FlatEntity,
        matrix: [f64; 6],
        run: &mut Run<'_>,
    ) -> Result<(), CadError> {
        let scope = EntityScope::BlockDefinition {
            name: block.to_owned(),
            reached_via: leaf.entity.provenance.block_path.child(block, handle),
        };
        match BlockResolver::new(&self.provider, FlattenBounds::default()).flatten(
            &scope,
            NAME,
            block,
            &mut NoProvenance,
            &mut LineageStore::new(),
        ) {
            Ok(inner) => self.emit(&inner.entities, matrix, false, run),
            Err(e) => {
                run.report.push(
                    Diagnostic::from_code(DiagnosticCode::UNRESOLVED_BLOCK, e.to_string())
                        .with_view(run.view)
                        .with_entity(handle),
                );
                Ok(())
            }
        }
    }
}

/// A block-to-view map read as what it does to shapes.
#[derive(Debug, Clone, Copy)]
struct Map {
    m: [f64; 6],
    /// Rotation, uniform scale and mirror only: circles stay circles.
    similarity: bool,
    scale: f64,
    mirror: bool,
    identity: bool,
}

impl Map {
    fn new(m: [f64; 6]) -> Self {
        let [a, b, c, d, _, _] = m;
        let det = a * d - b * c;
        let (col0, col1) = (a.hypot(c), b.hypot(d));
        let tolerance = 1e-9 * col0.max(col1).max(1.0);
        Self {
            m,
            similarity: (col0 - col1).abs() <= tolerance
                && (a * b + c * d).abs() <= tolerance * tolerance.max(col0),
            scale: det.abs().sqrt(),
            mirror: det < 0.0,
            identity: m == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }
    }

    fn point(&self, p: [f64; 3]) -> [f64; 3] {
        let [a, b, c, d, tx, ty] = self.m;
        [a * p[0] + b * p[1] + tx, c * p[0] + d * p[1] + ty, p[2]]
    }

    fn vector(&self, v: [f64; 3]) -> [f64; 3] {
        let [a, b, c, d, _, _] = self.m;
        [a * v[0] + b * v[1], c * v[0] + d * v[1], v[2]]
    }

    /// The direction angle `theta` has after the map.
    fn angle(&self, theta: f64) -> f64 {
        let v = self.vector([theta.cos(), theta.sin(), 0.0]);
        v[1].atan2(v[0])
    }
}

fn is_plus_z(n: [f64; 3]) -> bool {
    n[0].abs() < 1e-12 && n[1].abs() < 1e-12 && n[2] > 0.0
}

fn refused(code: DiagnosticCode, class: &str, why: &str) -> Diagnostic {
    Diagnostic::from_code(code, format!("{class}: {why}"))
}

/// Off the +Z plane, inside a block map, is a 3D placement this 2D map cannot carry.
fn flat_normal(map: &Map, class: &str, n: [f64; 3]) -> Result<(), Diagnostic> {
    if map.identity || is_plus_z(n) {
        Ok(())
    } else {
        Err(refused(
            DiagnosticCode::UNSUPPORTED_ENTITY,
            class,
            "an extrusion normal off +Z inside an INSERT",
        ))
    }
}

fn origin_of(leaf: &FlatEntity) -> Origin {
    leaf.entity
        .provenance
        .handle
        .map_or(Origin::UNKNOWN, Origin::from_handle)
        .with_expanded_insert(!leaf.entity.provenance.block_path.is_root())
}

/// A TEXT, MTEXT, MULTILEADER, TOLERANCE or ATTRIB string as a [`Text`], or nothing when it
/// shows no characters.
fn text_primitive(
    fact: &TextFact,
    map: &Map,
    origin: Origin,
    class: &str,
) -> Result<Option<Primitive>, Diagnostic> {
    flat_normal(map, class, fact.normal)?;
    let shown = if fact.multiline {
        display_text(&fact.content).text
    } else {
        fact.content.clone()
    };
    let content = decode_escapes(&shown);
    if content.trim().is_empty() {
        return Ok(None);
    }
    let text = Text::new(
        map.point(fact.position),
        fact.height * map.scale,
        map.angle(fact.rotation),
        content,
    )
    .map_err(|e| e.to_diagnostic())?;
    Ok(Some(text.with_origin(origin).into()))
}

/// What a HATCH, LEADER, MULTILEADER or TOLERANCE leaf draws, and what it says it left out, or
/// [`None`] when the leaf is some other entity.
fn convert_drawn(
    provider: &AcadrustProvider,
    leaf: &FlatEntity,
    matrix: [f64; 6],
) -> Option<(Vec<Primitive>, Vec<Diagnostic>)> {
    if !matches!(leaf.entity.geometry, RawGeometry::Unmapped { .. }) {
        return None;
    }
    let class = leaf.entity.provenance.class_name.as_str();
    let fact = provider.drawn_fact(leaf.entity.provenance.handle?)?;
    let (map, origin) = (Map::new(matrix), origin_of(leaf));
    let (mut drawn, mut notes) = (Vec::new(), Vec::new());
    for why in &fact.skipped {
        notes.push(refused(DiagnosticCode::UNSUPPORTED_ENTITY, class, why));
    }
    if !fact.rings.is_empty() {
        let bulged = fact.rings.iter().flatten().any(|v| v.bulge != 0.0);
        if let Err(d) = flat_normal(&map, class, fact.normal) {
            notes.push(d);
        } else if !map.similarity && bulged {
            notes.push(refused(
                DiagnosticCode::NON_UNIFORM_BLOCK_SCALE,
                class,
                "an INSERT that scales X and Y unequally cannot carry a bulged hatch loop",
            ));
        } else {
            let sign = if map.mirror { -1.0 } else { 1.0 };
            let normal = if map.identity {
                fact.normal
            } else {
                [0.0, 0.0, 1.0]
            };
            for ring in &fact.rings {
                let points: Vec<[f64; 3]> = ring
                    .iter()
                    .map(|v| map.point([v.x, v.y, fact.elevation]))
                    .collect();
                let bulges: Vec<f64> = if bulged {
                    ring.iter().map(|v| v.bulge * sign).collect()
                } else {
                    Vec::new()
                };
                match Polygon::new(&points, normal, &bulges) {
                    Ok(p) => drawn.push(p.with_origin(origin).into()),
                    Err(e) => notes.push(e.to_diagnostic()),
                }
            }
        }
    }
    for stroke in &fact.strokes {
        for pair in stroke.windows(2) {
            match Line::new(map.point(pair[0]), map.point(pair[1])) {
                Ok(l) => drawn.push(l.with_origin(origin).into()),
                Err(e) => notes.push(e.to_diagnostic()),
            }
        }
    }
    for text in &fact.texts {
        match text_primitive(text, &map, origin, class) {
            Ok(Some(p)) => drawn.push(p),
            Ok(None) => {}
            Err(d) => notes.push(d),
        }
    }
    Some((drawn, notes))
}

fn convert(
    provider: &AcadrustProvider,
    leaf: &FlatEntity,
    matrix: [f64; 6],
) -> Result<Option<Primitive>, Diagnostic> {
    let entity = &leaf.entity;
    let class = entity.provenance.class_name.as_str();
    let map = Map::new(matrix);
    let origin = origin_of(leaf);
    let bad = |e: CadError| e.to_diagnostic();
    let unmoved = |code, why: &str| refused(code, class, why);
    let curve = |what: &str| {
        unmoved(
            DiagnosticCode::NON_UNIFORM_BLOCK_SCALE,
            &format!("an INSERT that scales X and Y unequally cannot carry {what}"),
        )
    };
    let flat_normal = |n: [f64; 3]| flat_normal(&map, class, n);
    let primitive: Primitive = match &entity.geometry {
        RawGeometry::Line { start, end } => Line::new(map.point(*start), map.point(*end))
            .map_err(bad)?
            .with_origin(origin)
            .into(),
        RawGeometry::Polyline {
            vertices,
            closed,
            elevation,
            normal,
        } => {
            flat_normal(*normal)?;
            let has_bulge = vertices.iter().any(|v| v.bulge != 0.0);
            if !map.similarity && has_bulge {
                return Err(curve("a bulged polyline"));
            }
            let points: Vec<[f64; 3]> = vertices
                .iter()
                .map(|v| map.point([v.x, v.y, *elevation]))
                .collect();
            let bulges: Vec<f64> = if has_bulge {
                let sign = if map.mirror { -1.0 } else { 1.0 };
                vertices.iter().map(|v| v.bulge * sign).collect()
            } else {
                Vec::new()
            };
            let normal = if map.identity {
                *normal
            } else {
                [0.0, 0.0, 1.0]
            };
            Polyline::new(&points, *closed, normal, &bulges)
                .map_err(bad)?
                .with_origin(origin)
                .into()
        }
        RawGeometry::Circle {
            center,
            radius,
            normal,
        } => {
            flat_normal(*normal)?;
            if !map.similarity {
                return Err(curve("a circle"));
            }
            let normal = if map.identity {
                *normal
            } else {
                [0.0, 0.0, 1.0]
            };
            Circle::new(map.point(*center), radius * map.scale, normal)
                .map_err(bad)?
                .with_origin(origin)
                .into()
        }
        RawGeometry::Arc {
            center,
            radius,
            start_angle,
            end_angle,
            normal,
        } => {
            flat_normal(*normal)?;
            if !map.similarity {
                return Err(curve("an arc"));
            }
            let (s, e) = (map.angle(*start_angle), map.angle(*end_angle));
            // A mirror reverses the direction of travel, so the ends swap.
            let (s, e) = if map.mirror { (e, s) } else { (s, e) };
            let normal = if map.identity {
                *normal
            } else {
                [0.0, 0.0, 1.0]
            };
            Arc::new(map.point(*center), radius * map.scale, s, e, normal)
                .map_err(bad)?
                .with_origin(origin)
                .into()
        }
        RawGeometry::Ellipse {
            center,
            major_axis,
            minor_ratio,
            start_parameter,
            end_parameter,
            normal,
        } => {
            flat_normal(*normal)?;
            if !map.similarity {
                return Err(curve("an ellipse"));
            }
            // A mirror reverses the parameter's direction of travel.
            let (s, e) = if map.mirror {
                (-end_parameter, -start_parameter)
            } else {
                (*start_parameter, *end_parameter)
            };
            let normal = if map.identity {
                *normal
            } else {
                [0.0, 0.0, 1.0]
            };
            Ellipse::new(
                map.point(*center),
                map.vector(*major_axis),
                *minor_ratio,
                s,
                e,
                normal,
            )
            .map_err(bad)?
            .with_origin(origin)
            .into()
        }
        RawGeometry::Spline {
            degree,
            control_points,
            knots,
            weights,
            periodic,
            closed,
        } => {
            let Some(knots) = knots.present() else {
                return Err(unmoved(
                    DiagnosticCode::UNSUPPORTED_ENTITY,
                    "a spline with no stored knot vector",
                ));
            };
            let weights = weights.present().cloned().unwrap_or_default();
            let mut flags = 0;
            if *closed {
                flags |= Spline::CLOSED;
            }
            if !weights.is_empty() {
                flags |= Spline::RATIONAL;
            }
            if periodic.present().copied().unwrap_or(false) {
                flags |= Spline::PERIODIC;
            }
            let controls: Vec<[f64; 3]> = control_points.iter().map(|p| map.point(*p)).collect();
            Spline::new(*degree, flags, knots, &controls, &weights)
                .map_err(bad)?
                .with_origin(origin)
                .into()
        }
        RawGeometry::Unmapped { reason } => {
            let Some(fact) = entity.provenance.handle.and_then(|h| provider.text_fact(h)) else {
                return Err(unmoved(DiagnosticCode::UNSUPPORTED_ENTITY, reason));
            };
            return text_primitive(&fact, &map, origin, class);
        }
        // `Insert` never reaches a leaf: the walk expanded it or listed it as skipped.
        _ => {
            return Err(unmoved(
                DiagnosticCode::UNSUPPORTED_ENTITY,
                "a geometry kind this build has no primitive for",
            ));
        }
    };
    Ok(Some(primitive))
}

#[cfg(test)]
mod tests {
    use super::{Map, is_plus_z};
    use std::f64::consts::{FRAC_PI_2, PI};

    #[test]
    fn a_rotation_keeps_shapes_and_turns_directions() {
        let m = Map::new([0.0, -1.0, 1.0, 0.0, 5.0, 6.0]);
        assert!(m.similarity && !m.mirror && !m.identity);
        assert!((m.scale - 1.0).abs() < 1e-12);
        assert!((m.angle(0.0) - FRAC_PI_2).abs() < 1e-12);
        assert_eq!(m.point([1.0, 0.0, 7.0]), [5.0, 7.0, 7.0]);
    }

    #[test]
    fn a_mirror_is_a_similarity_that_reverses_travel() {
        let m = Map::new([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert!(m.similarity && m.mirror);
        assert!((m.angle(0.0).abs() - PI).abs() < 1e-12);
        assert!((m.angle(FRAC_PI_2) - FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn a_uniform_scale_is_a_similarity_and_an_unequal_one_is_not() {
        let four = Map::new([4.0, 0.0, 0.0, 4.0, 0.0, 0.0]);
        assert!(four.similarity && (four.scale - 4.0).abs() < 1e-12);
        let skew = Map::new([2.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert!(!skew.similarity);
        assert!(Map::new([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).identity);
    }

    #[test]
    fn only_plus_z_is_the_plane_a_block_map_can_carry() {
        assert!(is_plus_z([0.0, 0.0, 1.0]));
        assert!(!is_plus_z([0.0, 0.0, -1.0]));
        assert!(!is_plus_z([1.0, 0.0, 0.0]));
    }
}
