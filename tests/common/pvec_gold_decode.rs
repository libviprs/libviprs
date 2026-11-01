//! Opening a corpus drawing and turning its model space into world-space curves, the way the
//! conversion does (acadrust provider, block flatten, basic and spline extraction, promotion
//! through the composed instance transform). Shared by the conformance suites that compare libviprs'
//! output with something from the pinned corpus: `pvec_dual_oracle` (N2.2) and the later ones.
//!
//! The suites `#[path]`-include this file after `pvec_gold.rs`. Every comparison helper here takes
//! its tolerance from `pvec_numeric::Policy::V1`; there is no literal threshold in this file.

#![allow(dead_code)] // each suite uses its own part

use libviprs::cad::{CadSource, ItemHandle};
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::dwg_basic::{self, BasicKind};
use libviprs::pvec::dwg_spline::{self, SplineForm};
use libviprs::pvec::flatten::{
    BlockResolver, FlatEntity, FlattenBounds, FlattenError, Flattened, LineageStore,
};
use libviprs::pvec::provenance::NoProvenance;
use libviprs::pvec::transform::{ComposedTransform, promote};
use libviprs::pvec::{
    DwgProvider, EntityScope, FidelityClass, InstanceStep, LayoutKind, Normalized2DGeometry,
    RawGeometry, ViewProfile,
};
use pvec_geometry::{Curve, Curve2d};
use pvec_numeric::Policy;

pub const POLICY: Policy = Policy::V1;

/// Opens a drawing from memory.
pub fn open(bytes: &[u8]) -> Result<AcadrustProvider, String> {
    AcadrustProvider::open(CadSource::Bytes(bytes)).map_err(|e| format!("does not open: {e}"))
}

/// Flattens model space under the default view profile's bounds.
pub fn flatten(provider: &AcadrustProvider, source: &str) -> Result<Flattened, FlattenError> {
    let layouts = provider.layouts().expect("layouts");
    let model = layouts
        .iter()
        .find(|l| l.kind == LayoutKind::Model)
        .expect("model space");
    BlockResolver::new(
        provider,
        FlattenBounds::from_profile(&ViewProfile::default_profile()),
    )
    .flatten(
        &EntityScope::Layout(model.id),
        source,
        &model.name,
        &mut NoProvenance,
        &mut LineageStore::new(),
    )
}

/// A leaf as a world-space curve with the semantic class libviprs gave it, or the reason it has
/// none (a refusal or a class this slice does not map).
pub fn world(f: &FlatEntity) -> Result<(&'static str, Curve), String> {
    let (class, curve) = match &f.entity.geometry {
        RawGeometry::Spline { .. } => {
            let s = dwg_spline::extract(&f.entity).map_err(|e| format!("refused: {e}"))?;
            let pts: Vec<[f64; 2]> = s.control_points.iter().map(|p| [p[0], p[1]]).collect();
            let degree = s.degree as usize;
            let (class, rec) = match &s.form {
                SplineForm::NonRational => (
                    "bspline",
                    Normalized2DGeometry::b_spline(
                        degree,
                        &pts,
                        &s.knots,
                        FidelityClass::SupportedExact,
                    ),
                ),
                SplineForm::Rational(w) => (
                    "nurbs",
                    Normalized2DGeometry::rational_b_spline(
                        degree,
                        &pts,
                        &s.knots,
                        w,
                        FidelityClass::SupportedExact,
                    ),
                ),
                _ => return Err("refused: unknown spline form".into()),
            };
            let rec = rec.map_err(|e| format!("refused: {e}"))?;
            (class, rec.geometry().expect("geometry").clone())
        }
        RawGeometry::Insert(_) => return Err("an insert that was not expanded".into()),
        _ => {
            let b = dwg_basic::extract(&f.entity).map_err(|e| format!("refused: {e}"))?;
            let class = match b.kind {
                BasicKind::Line => "line",
                BasicKind::Polyline => "polyline",
                BasicKind::Circle => "circle",
                BasicKind::Arc => "circular_arc",
                BasicKind::Ellipse => "ellipse",
                BasicKind::EllipticalArc => "elliptical_arc",
                _ => return Err("a basic kind this slice does not map".into()),
            };
            (class, b.curve)
        }
    };
    let record = Normalized2DGeometry::from_curve(curve, FidelityClass::SupportedEquivalent)
        .map_err(|e| format!("refused: {e}"))?;
    let step = InstanceStep {
        block_name: "composed".to_string(),
        insert_handle: ItemHandle::new(0),
    };
    let composed = ComposedTransform::identity()
        .push_inner_coefficients(step, f.matrix())
        .map_err(|e| format!("refused: {e}"))?;
    let promoted = promote(&record, &composed).map_err(|e| format!("refused: {e:?}"))?;
    let curve = promoted
        .record()
        .geometry()
        .ok_or_else(|| "refused: the promoted record carries no geometry".to_string())?
        .clone();
    Ok((class, curve))
}

/// `a` and `b` are the same value at feature size `scale` (NP-05).
pub fn near(a: f64, b: f64, scale: f64) -> bool {
    POLICY.near(a, b, scale)
}

/// Both coordinates are `near`.
pub fn near_pt(a: [f64; 2], b: [f64; 2], scale: f64) -> bool {
    POLICY.near_point(a, b, scale)
}

/// The size of the features a record talks about, for `Policy::near`.
pub fn scale_of(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0)
}

/// Whether `p` lies on the ellipse with centre `c`, unit major direction `u` and semi-axes `a`, `b`
/// (the implicit equation of a corpus record, not a computation of libviprs').
pub fn on_ellipse(p: [f64; 2], c: [f64; 2], u: [f64; 2], a: f64, b: f64, scale: f64) -> bool {
    let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
    let (s, t) = (dx * u[0] + dy * u[1], -dx * u[1] + dy * u[0]);
    near((s / a).hypot(t / b), 1.0, scale.max(1.0))
}

/// Evaluates a curve at `n + 1` evenly spaced parameters across its domain.
pub fn sample(curve: &Curve, n: u32) -> Vec<[f64; 2]> {
    let d = curve.domain();
    (0..=n)
        .map(|k| {
            let p = curve
                .eval(d.lo() + d.width() * f64::from(k) / f64::from(n))
                .expect("a curve evaluates inside its domain");
            [p.x, p.y]
        })
        .collect()
}
