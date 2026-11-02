//! Drift check: a `.pvec` against the fixture it was generated from (issue #1471 follow-up, EPIC N).
//!
//! The artifact is read back through PVec's public reader and its geometry is compared, per entity
//! id, with what the fixture knows about itself:
//!
//! - a `KNOWN_TRUTH_EXACT` fixture is compared with its `expected.math.json`, the independently
//!   authored truth, so a pass here says the artifact is the drawing and not merely the same as last
//!   time;
//! - every other evidence class (`DUAL_ORACLE_EQUIVALENT`, `REAL_WORLD_DIFFERENTIAL`) has no truth,
//!   so the artifact is compared with the previously accepted output, a canonical dump kept in the
//!   repository;
//! - a `QUARANTINED` fixture is compared the same way, and the report says it does not gate.
//!
//! A known-truth fixture is also compared with its accepted dump when one is passed in, so the report
//! can say both "this is not the drawing" and "this is not what we accepted last time".
//!
//! Every comparison is within PVec's shared numeric policy ([`Policy::V1`]); there is no epsilon of
//! libviprs's own. The curve mathematics (evaluation, bounds, trims) is PVec's, reached through its
//! public API. What this module adds is the bookkeeping: which entity is which, and what kind of
//! drift a difference is.
//!
//! The canonical dump ([`canonical_dump`]) prints every number as its IEEE bits next to the shortest
//! decimal, so its sha256 ([`digest`]) changes when a single bit of the stored geometry does. That
//! digest is what `tests/fixtures/pvec-drift/` commits, one per fixture.
//!
//! See `docs/pvec-drift-check.md`.

use std::fmt::{self, Write as _};
use std::path::Path;

use pvec_facade::{Curve, PvecReader};
use pvec_geometry::{Curve2d, CurveKind};
use pvec_numeric::Policy;
use serde_json::Value;
use sha2::{Digest, Sha256};

const POLICY: Policy = Policy::V1;

/// What kind of drift a difference is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DriftKind {
    /// A vertex, endpoint, centre, control point, knot or weight is not where it should be.
    MovedVertex,
    /// A radius or semi-axis changed.
    ChangedRadius,
    /// An entity the fixture has is not in the artifact.
    DroppedEntity,
    /// The artifact has an entity the fixture does not.
    ExtraEntity,
    /// The right geometry is under another entity id.
    ReorderedIds,
    /// The manifest bounds are not the bounds of the geometry.
    ChangedBounds,
    /// The manifest unit is not the fixture's.
    ChangedUnit,
    /// The entity is another kind of curve than the fixture says.
    SwappedClass,
}

impl DriftKind {
    /// The name used in the machine-readable report.
    pub fn name(self) -> &'static str {
        match self {
            Self::MovedVertex => "moved-vertex",
            Self::ChangedRadius => "changed-radius",
            Self::DroppedEntity => "dropped-entity",
            Self::ExtraEntity => "extra-entity",
            Self::ReorderedIds => "reordered-ids",
            Self::ChangedBounds => "changed-bounds",
            Self::ChangedUnit => "changed-unit",
            Self::SwappedClass => "swapped-class",
        }
    }
}

/// What a drift was found against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Against {
    /// The fixture's `expected.math.json`.
    Truth,
    /// The previously accepted output.
    Accepted,
}

impl Against {
    fn name(self) -> &'static str {
        match self {
            Self::Truth => "known-truth",
            Self::Accepted => "accepted-output",
        }
    }
}

/// One difference between the artifact and what it was checked against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Drift {
    /// The PVec entity id, when the difference belongs to one.
    pub entity: Option<u32>,
    /// What kind of difference.
    pub kind: DriftKind,
    /// What it was compared with.
    pub against: Against,
    /// The difference, in words.
    pub detail: String,
}

/// The result of checking one artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriftReport {
    /// The fixture id.
    pub fixture: String,
    /// The fixture's evidence class, as the corpus names it.
    pub evidence_class: String,
    /// Whether a drift fails the check: false for a quarantined fixture.
    pub gating: bool,
    /// Every difference found, sorted by entity id and then kind.
    pub drifts: Vec<Drift>,
    /// sha256 of the canonical dump of the artifact.
    pub digest: String,
}

impl DriftReport {
    /// No drift was found.
    pub fn is_clean(&self) -> bool {
        self.drifts.is_empty()
    }

    /// The report fails the check: it has a drift and the fixture gates.
    pub fn fails(&self) -> bool {
        self.gating && !self.drifts.is_empty()
    }

    /// The machine-readable report: one JSON object, keys in a fixed order.
    pub fn to_json(&self) -> String {
        let drifts: Vec<Value> = self
            .drifts
            .iter()
            .map(|d| {
                serde_json::json!({
                    "entity": d.entity,
                    "kind": d.kind.name(),
                    "against": d.against.name(),
                    "detail": d.detail,
                })
            })
            .collect();
        let mut out = serde_json::to_string_pretty(&serde_json::json!({
            "fixture": self.fixture,
            "evidence_class": self.evidence_class,
            "gating": self.gating,
            "clean": self.is_clean(),
            "digest": self.digest,
            "drifts": drifts,
        }))
        .expect("a json value serialises");
        out.push('\n');
        out
    }

    /// The report for a person.
    pub fn to_text(&self) -> String {
        let mut out = format!(
            "{} ({}): {}\n",
            self.fixture,
            self.evidence_class,
            if self.is_clean() {
                "no drift".to_owned()
            } else {
                format!(
                    "{} drift(s){}",
                    self.drifts.len(),
                    if self.gating { "" } else { ", not gating" }
                )
            }
        );
        for d in &self.drifts {
            let who = d
                .entity
                .map_or_else(|| "artifact".to_owned(), |e| format!("entity {e}"));
            let _ = writeln!(
                out,
                "  {who}: {} against {}: {}",
                d.kind.name(),
                d.against.name(),
                d.detail
            );
        }
        let _ = writeln!(out, "  digest {}", self.digest);
        out
    }
}

/// Why a check could not be made (as opposed to a drift being found).
#[derive(Debug)]
pub enum DriftError {
    /// A file could not be read.
    Io(String),
    /// The fixture's files are not what the corpus promises.
    Fixture(String),
    /// PVec's reader refused the artifact.
    Read(String),
}

impl fmt::Display for DriftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(s) => write!(f, "io: {s}"),
            Self::Fixture(s) => write!(f, "fixture: {s}"),
            Self::Read(s) => write!(f, "the artifact does not read: {s}"),
        }
    }
}

impl std::error::Error for DriftError {}

// ------------------------------------------------------------------------------- entry points

/// Checks the `.pvec` at `pvec` against the fixture directory `fixture_dir` (one directory of the
/// pinned corpus: `verdict.json`, and `expected.math.json` for a known-truth fixture). `accepted` is
/// the fixture's accepted dump, when there is one ([`accepted_section`]).
pub fn check_fixture(
    fixture_dir: &Path,
    pvec: &Path,
    accepted: Option<&str>,
) -> Result<DriftReport, DriftError> {
    let read = |name: &str| -> Result<Option<Value>, DriftError> {
        let p = fixture_dir.join(name);
        if !p.exists() {
            return Ok(None);
        }
        let bytes =
            std::fs::read(&p).map_err(|e| DriftError::Io(format!("{}: {e}", p.display())))?;
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| DriftError::Fixture(format!("{}: {e}", p.display())))
    };
    let verdict = read("verdict.json")?.ok_or_else(|| {
        DriftError::Fixture(format!("{} has no verdict.json", fixture_dir.display()))
    })?;
    let id = verdict["fixture_id"].as_str().unwrap_or_default();
    let class = verdict["evidence_class"].as_str().unwrap_or_default();
    let truth = read("expected.math.json")?;
    let bytes =
        std::fs::read(pvec).map_err(|e| DriftError::Io(format!("{}: {e}", pvec.display())))?;
    check(id, class, truth.as_ref(), accepted, &bytes)
}

/// Checks artifact bytes. `truth` is the fixture's `expected.math.json` (used when the evidence class
/// is `KNOWN_TRUTH_EXACT` and it is present), `accepted` the accepted dump of this fixture.
pub fn check(
    fixture: &str,
    evidence_class: &str,
    truth: Option<&Value>,
    accepted: Option<&str>,
    pvec: &[u8],
) -> Result<DriftReport, DriftError> {
    let art = Artifact::read(pvec)?;
    let mut drifts = Vec::new();
    if evidence_class == "KNOWN_TRUTH_EXACT" {
        let t = truth.ok_or_else(|| {
            DriftError::Fixture(format!(
                "{fixture} is known truth and has no expected.math.json"
            ))
        })?;
        against_truth(&art, t, &mut drifts);
    } else if accepted.is_none() {
        return Err(DriftError::Fixture(format!(
            "{fixture} is {evidence_class}: it has no truth, so it needs an accepted dump to be checked against"
        )));
    }
    if let Some(a) = accepted {
        against_accepted(&art, a, &mut drifts);
    }
    drifts.sort_by(|a, b| {
        (a.entity, a.kind, a.against as u8).cmp(&(b.entity, b.kind, b.against as u8))
    });
    Ok(DriftReport {
        fixture: fixture.to_owned(),
        evidence_class: evidence_class.to_owned(),
        gating: evidence_class != "QUARANTINED",
        drifts,
        digest: digest(&art.dump(fixture)),
    })
}

/// The canonical dump of an artifact: the manifest and every entity, numbers as bits.
pub fn canonical_dump(fixture: &str, pvec: &[u8]) -> Result<String, DriftError> {
    Ok(Artifact::read(pvec)?.dump(fixture))
}

/// The sha256 of a canonical dump, in hex.
pub fn digest(dump: &str) -> String {
    Sha256::digest(dump.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The dump of `fixture` inside an accepted-output file, which holds one `== <fixture>` section per
/// fixture. `None` when the file has no section for it.
pub fn accepted_section<'a>(file: &'a str, fixture: &str) -> Option<&'a str> {
    let head = format!("== {fixture}\n");
    let start = file
        .match_indices(&head)
        .find(|(i, _)| *i == 0 || file[..*i].ends_with('\n'))?
        .0
        + head.len();
    let rest = &file[start..];
    let end = rest.find("\n== ").map_or(rest.len(), |i| i + 1);
    Some(&rest[..end])
}

// ----------------------------------------------------------------------------- the artifact

/// What the reader hands back, plus the numbers the dump and the comparisons use.
struct Artifact {
    metres_per_unit: f64,
    bounds: [[f64; 2]; 2],
    curves: Vec<(u32, Curve)>,
}

impl Artifact {
    fn read(bytes: &[u8]) -> Result<Artifact, DriftError> {
        let fail = |e: pvec_facade::PvecError| DriftError::Read(e.to_string());
        let reader = PvecReader::open(bytes.to_vec()).map_err(fail)?;
        let m = reader.manifest();
        let curves = reader
            .entities()
            .map_err(fail)?
            .into_iter()
            .map(|(id, c)| (id.index(), c))
            .collect();
        Ok(Artifact {
            metres_per_unit: m.metres_per_unit,
            bounds: [m.bounds.min, m.bounds.max],
            curves,
        })
    }

    fn dump(&self, fixture: &str) -> String {
        let mut out = format!("fixture {fixture}\n");
        let _ = writeln!(out, "unit {}", show(self.metres_per_unit));
        let _ = writeln!(
            out,
            "bounds {} {} {} {}",
            show(self.bounds[0][0]),
            show(self.bounds[0][1]),
            show(self.bounds[1][0]),
            show(self.bounds[1][1])
        );
        let _ = writeln!(out, "entities {}", self.curves.len());
        for (id, c) in &self.curves {
            let (class, params) = describe(c);
            let _ = write!(out, "e {id} {class}");
            for (name, v) in params {
                let _ = write!(out, " {name}={}", show(v));
            }
            out.push('\n');
        }
        out
    }
}

/// A number as the shortest decimal that reparses to it, then its bits.
fn show(v: f64) -> String {
    format!("{v}[{:016x}]", v.to_bits())
}

/// The bits out of a [`show`]n number.
fn unshow(s: &str) -> Option<f64> {
    let hex = s.split_once('[')?.1.strip_suffix(']')?;
    u64::from_str_radix(hex, 16).ok().map(f64::from_bits)
}

/// The class name and the named numbers that define a curve, read through the public getters.
fn describe(c: &Curve) -> (String, Vec<(String, f64)>) {
    let mut p = Vec::new();
    let class = describe_into(c, "", &mut p);
    (class, p)
}

fn describe_into(c: &Curve, prefix: &str, p: &mut Vec<(String, f64)>) -> String {
    let mut put = |name: &str, v: f64| p.push((format!("{prefix}{name}"), v));
    match c {
        Curve::Line(l) => {
            put("ox", l.origin().x);
            put("oy", l.origin().y);
            put("dx", l.direction().x);
            put("dy", l.direction().y);
            "line".into()
        }
        Curve::Circle(k) => {
            let (o, a) = (k.placement().origin(), k.placement().x_axis());
            put("cx", o.x);
            put("cy", o.y);
            put("ax", a.x);
            put("ay", a.y);
            put("radius", k.radius());
            "circle".into()
        }
        Curve::Ellipse(k) => {
            let (o, a) = (k.placement().origin(), k.placement().x_axis());
            put("cx", o.x);
            put("cy", o.y);
            put("ax", a.x);
            put("ay", a.y);
            put("rx", k.semi_axis_1());
            put("ry", k.semi_axis_2());
            "ellipse".into()
        }
        Curve::Polyline(k) => {
            for (i, v) in k.vertices().iter().enumerate() {
                put(&format!("v{i}x"), v.x);
                put(&format!("v{i}y"), v.y);
            }
            "polyline".into()
        }
        Curve::Trimmed(t) => {
            let tr = t.trim();
            put("trim.start", tr.start);
            put("trim.span", tr.span);
            put("trim.reversed", f64::from(u8::from(tr.reversed)));
            let inner = describe_into(t.basis(), &format!("{prefix}b."), p);
            format!("trimmed-{inner}")
        }
        Curve::Composite(k) => {
            for (i, s) in k.segments().iter().enumerate() {
                let inner = describe_into(s, &format!("{prefix}s{i}."), p);
                p.push((format!("{prefix}s{i}.class#{inner}"), 0.0));
            }
            "composite".into()
        }
        Curve::Bezier(k) => {
            put("degree", k.degree() as f64);
            for (i, v) in k.control_points().iter().enumerate() {
                put(&format!("p{i}x"), v.x);
                put(&format!("p{i}y"), v.y);
            }
            "bezier".into()
        }
        Curve::BSpline(k) => {
            put("degree", k.degree() as f64);
            for (i, v) in k.control_points().iter().enumerate() {
                put(&format!("p{i}x"), v.x);
                put(&format!("p{i}y"), v.y);
            }
            for (i, v) in k.knots().iter().enumerate() {
                put(&format!("k{i}"), *v);
            }
            "bspline".into()
        }
        Curve::RationalBSpline(k) => {
            put("degree", k.degree() as f64);
            for (i, v) in k.control_points().iter().enumerate() {
                put(&format!("p{i}x"), v.x);
                put(&format!("p{i}y"), v.y);
            }
            for (i, v) in k.knots().iter().enumerate() {
                put(&format!("k{i}"), *v);
            }
            for (i, v) in k.weights().iter().enumerate() {
                put(&format!("w{i}"), *v);
            }
            "rational-bspline".into()
        }
        other => format!("{:?}", other.kind()).to_lowercase(),
    }
}

/// The kind of a parameter name: a radius or semi-axis, or something that moved.
fn kind_of_param(name: &str) -> DriftKind {
    let last = name.rsplit('.').next().unwrap_or(name);
    if matches!(last, "radius" | "rx" | "ry") {
        DriftKind::ChangedRadius
    } else {
        DriftKind::MovedVertex
    }
}

// ------------------------------------------------------------------------ against the truth

type Problems = Vec<String>;

fn near(a: f64, b: f64, scale: f64) -> bool {
    POLICY.near(a, b, scale)
}

fn near_pt(a: [f64; 2], b: [f64; 2], scale: f64) -> bool {
    POLICY.near_point(a, b, scale)
}

fn scale_of(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0)
}

/// A truth number, `{"decimal": .., "f64_hex": ..}`: the hex is the value. A malformed one is NaN,
/// which no comparison accepts, so it shows up as a drift and never as a pass.
pub fn num(v: &Value) -> f64 {
    v["f64_hex"]
        .as_str()
        .and_then(|h| u64::from_str_radix(h.trim_start_matches("0x"), 16).ok())
        .map_or(f64::NAN, f64::from_bits)
}

/// A list of truth numbers.
pub fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .map_or_else(Vec::new, |l| l.iter().map(num).collect())
}

fn points(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .map_or_else(Vec::new, |l| l.iter().map(nums).collect())
}

fn end_points(c: &Curve) -> [[f64; 2]; 2] {
    let d = c.domain();
    let at = |u: f64| c.eval(u).map_or([f64::NAN; 2], |p| [p.x, p.y]);
    [at(d.lo()), at(d.hi())]
}

/// The radii and the first axis of a conic's truth record.
fn conic_frame(class: &str, g: &Value) -> (f64, f64, [f64; 2]) {
    let (rx, ry, axis) = match class {
        "circle" | "circular_arc" => (num(&g["radius"]), num(&g["radius"]), nums(&g["x_axis"])),
        _ => (
            num(&g["semi_major"]),
            num(&g["semi_minor"]),
            nums(&g["major_axis"]),
        ),
    };
    let axis = if axis.len() >= 2 {
        axis
    } else {
        vec![f64::NAN; 2]
    };
    let n = axis[0].hypot(axis[1]);
    (rx, ry, [axis[0] / n, axis[1] / n])
}

/// The curve classes a truth class may come back as. A line is stored as a trimmed unbounded line,
/// an arc as a trimmed circle or ellipse, and a Bezier as a B-spline (PVec's reader documents that).
fn classes_of(truth_class: &str) -> &'static [CurveKind] {
    match truth_class {
        "line" => &[CurveKind::TrimmedCurve],
        "circle" => &[CurveKind::Circle],
        "ellipse" => &[CurveKind::Ellipse],
        "circular_arc" | "elliptical_arc" => &[
            CurveKind::TrimmedCurve,
            CurveKind::Circle,
            CurveKind::Ellipse,
        ],
        "polyline" => &[CurveKind::Polyline],
        "bspline" | "nurbs" => &[CurveKind::BSpline, CurveKind::RationalBSpline],
        "bezier" => &[CurveKind::Bezier, CurveKind::BSpline],
        _ => &[],
    }
}

/// The reasons `curve` is not what the truth record `class`/`g` describes (`t` is the whole truth
/// entity, for closure). Empty when it is. Also used by the end-to-end suite (`tests/pvec_end_to_end.rs`).
pub fn differs(class: &str, g: &Value, t: &Value, curve: &Curve) -> Problems {
    let mut out = Problems::new();
    match class {
        "line" => {
            let (s, e) = (nums(&g["start"]), nums(&g["end"]));
            if s.len() < 2 || e.len() < 2 {
                return vec!["line: the truth record has no endpoints".into()];
            }
            let got = end_points(curve);
            let sc = scale_of(&[s[0], s[1], e[0], e[1]]);
            if !near_pt(got[0], [s[0], s[1]], sc) || !near_pt(got[1], [e[0], e[1]], sc) {
                out.push(format!(
                    "line: runs {:?} to {:?}, truth {s:?} to {e:?}",
                    got[0], got[1]
                ));
            }
        }
        "circle" | "ellipse" => {
            let c = nums(&g["center"]);
            if c.len() < 2 {
                return vec![format!("{class}: the truth record has no centre")];
            }
            let (rx, ry, u) = conic_frame(class, g);
            let sc = scale_of(&[c[0], c[1], rx, ry]);
            let d = curve.domain();
            for k in 0..8 {
                let Ok(p) = curve.eval(d.lo() + d.width() * f64::from(k) / 8.0) else {
                    out.push(format!("{class}: the curve does not evaluate"));
                    break;
                };
                let (dx, dy) = (p.x - c[0], p.y - c[1]);
                let (s, q) = (dx * u[0] + dy * u[1], -dx * u[1] + dy * u[0]);
                if !near((s / rx).hypot(q / ry), 1.0, sc.max(1.0)) {
                    out.push(format!(
                        "{class}: the point ({}, {}) is off the truth",
                        p.x, p.y
                    ));
                    break;
                }
            }
        }
        "circular_arc" | "elliptical_arc" => {
            let c = nums(&g["center"]);
            if c.len() < 2 {
                return vec![format!("{class}: the truth record has no centre")];
            }
            let (rx, ry, ax) = conic_frame(class, g);
            // The y axis follows the normal, so a reflected frame is honoured.
            let nz = g
                .get("normal")
                .map_or(1.0, |n| nums(n).get(2).copied().unwrap_or(1.0));
            let ay = [-nz * ax[1], nz * ax[0]];
            let (a, sweep) = (num(&g["start_angle"]), num(&g["sweep_angle"]));
            let at = |th: f64| {
                [
                    c[0] + rx * th.cos() * ax[0] + ry * th.sin() * ay[0],
                    c[1] + rx * th.cos() * ax[1] + ry * th.sin() * ay[1],
                ]
            };
            let got = end_points(curve);
            let sc = scale_of(&[rx, ry, c[0], c[1]]);
            if !near_pt(got[0], at(a), sc) || !near_pt(got[1], at(a + sweep), sc) {
                out.push(format!(
                    "{class}: runs {:?} to {:?}, truth {:?} to {:?}",
                    got[0],
                    got[1],
                    at(a),
                    at(a + sweep)
                ));
            }
        }
        "polyline" => polyline(&mut out, g, t, curve),
        "bspline" | "nurbs" | "bezier" => spline(&mut out, class, g, t, curve),
        other => out.push(format!("no comparison for the truth class {other}")),
    }
    out
}

fn polyline(out: &mut Problems, g: &Value, t: &Value, curve: &Curve) {
    let v = points(&g["vertices"]);
    if v.is_empty() || v.iter().any(|p| p.len() < 2) {
        out.push("polyline: the truth record has no vertices".into());
        return;
    }
    let closed = t["closure"]["closed_by_endpoint"]
        .as_bool()
        .unwrap_or(false);
    let got = end_points(curve);
    let (first, last) = (&v[0], if closed { &v[0] } else { &v[v.len() - 1] });
    let sc = scale_of(&v.iter().flatten().copied().collect::<Vec<_>>());
    if !near_pt(got[0], [first[0], first[1]], sc) || !near_pt(got[1], [last[0], last[1]], sc) {
        out.push(format!(
            "polyline: runs {:?} to {:?}, truth {first:?} to {last:?}",
            got[0], got[1]
        ));
    }
    if let Curve::Polyline(p) = curve {
        let want = v.len() + usize::from(closed);
        if p.vertices().len() != want {
            out.push(format!(
                "polyline: {} vertices, truth {want}",
                p.vertices().len()
            ));
        }
        for (i, (a, w)) in p.vertices().iter().zip(&v).enumerate() {
            if !near_pt([a.x, a.y], [w[0], w[1]], sc) {
                out.push(format!(
                    "polyline: vertex {i} is ({}, {}), truth {w:?}",
                    a.x, a.y
                ));
            }
        }
    }
}

fn spline(out: &mut Problems, class: &str, g: &Value, t: &Value, curve: &Curve) {
    let degree = g["degree"].as_u64().unwrap_or(0) as usize;
    let want_cps = points(&g["control_points"]);
    let (got_degree, cps, knots, weights) = match curve {
        Curve::BSpline(s) => (
            s.degree(),
            s.control_points()
                .iter()
                .map(|p| [p.x, p.y])
                .collect::<Vec<[f64; 2]>>(),
            Some(s.knots().to_vec()),
            None::<Vec<f64>>,
        ),
        Curve::RationalBSpline(s) => (
            s.degree(),
            s.control_points().iter().map(|p| [p.x, p.y]).collect(),
            Some(s.knots().to_vec()),
            Some(s.weights().to_vec()),
        ),
        Curve::Bezier(b) => (
            b.degree(),
            b.control_points().iter().map(|p| [p.x, p.y]).collect(),
            None,
            None,
        ),
        other => {
            out.push(format!("{class}: came back as {:?}", other.kind()));
            return;
        }
    };
    if got_degree != degree {
        out.push(format!("degree: re-read {got_degree}, truth {degree}"));
    }
    if t["closure"]["periodic"].as_bool().unwrap_or(false) {
        // Not comparable structurally: PVec stores a clamped spline, so a periodic source comes back
        // with its control points and knots rewritten. What survives the rewrite is the degree
        // (above) and that the curve still closes on itself.
        let [a, b] = end_points(curve);
        let sc = scale_of(&[a[0], a[1], b[0], b[1]]);
        if !near_pt(a, b, sc) {
            out.push(format!(
                "closed state: a periodic source's curve runs {a:?} to {b:?}"
            ));
        }
        return;
    }
    if cps.len() != want_cps.len() {
        out.push(format!(
            "control points: re-read {}, truth {}",
            cps.len(),
            want_cps.len()
        ));
    } else {
        for (i, (a, w)) in cps.iter().zip(&want_cps).enumerate() {
            if w.len() < 2 || a[0] != w[0] || a[1] != w[1] {
                out.push(format!(
                    "control point {i}: re-read ({}, {}), truth {w:?}",
                    a[0], a[1]
                ));
            }
        }
    }
    if let (Some(got), Some(list)) = (&knots, g.get("knots").and_then(Value::as_array)) {
        let want: Vec<f64> = list
            .iter()
            .flat_map(|k| {
                let m = k["multiplicity"].as_u64().unwrap_or(1) as usize;
                std::iter::repeat_n(num(&k["value"]), m)
            })
            .collect();
        if *got != want {
            out.push(format!(
                "knots: re-read {got:?}, truth (with multiplicities) {want:?}"
            ));
        }
    }
    // A non-rational spline has no weight array at all, not a row of ones.
    match (&weights, g.get("weights")) {
        (None, None) => {}
        (Some(w), Some(want)) if *w == nums(want) => {}
        (Some(w), Some(want)) => out.push(format!("weights: re-read {w:?}, truth {want}")),
        (None, Some(_)) => out.push("weights: the truth has weights and the artifact none".into()),
        (Some(_), None) => out.push("weights: a weight array on a non-rational spline".into()),
    }
}

/// The geometry a truth entity says the artifact holds: the world result when the entity is
/// instanced under a transform, the curve itself otherwise.
pub fn target(t: &Value) -> (&str, &Value) {
    match t.get("world") {
        Some(w) => (w["class"].as_str().unwrap_or("?"), &w["geometry"]),
        None => (t["class"].as_str().unwrap_or("?"), &t["geometry"]),
    }
}

/// The truth entities the artifact is expected to hold, in truth order: not an equivalent form of
/// another, and not on a layer that is off or frozen (the view profile excludes those).
pub fn live(truth: &Value) -> Vec<&Value> {
    truth["entities"]
        .as_array()
        .map_or_else(Vec::new, |l| l.iter().collect::<Vec<_>>())
        .into_iter()
        .filter(|e| !e["id"].as_str().is_some_and(|i| i.contains("-equivalent-")))
        .filter(|e| {
            let l = &e["layer"];
            l["on"].as_bool().unwrap_or(true) && !l["frozen"].as_bool().unwrap_or(false)
        })
        .collect()
}

/// Whether entity `t` of the truth is the curve `c`: the right class and the right geometry.
fn is(t: &Value, c: &Curve) -> bool {
    let (class, g) = target(t);
    classes_of(class).contains(&kind_of(c)) && differs(class, g, t, c).is_empty()
}

/// The class a curve is stored as, looking through a trim at what is trimmed.
fn kind_of(c: &Curve) -> CurveKind {
    match c {
        Curve::Trimmed(t) if matches!(t.basis(), Curve::Circle(_) | Curve::Ellipse(_)) => {
            // an arc is a trimmed conic, and the conic kind says which
            t.basis().kind()
        }
        other => other.kind(),
    }
}

fn push(
    out: &mut Vec<Drift>,
    entity: Option<u32>,
    kind: DriftKind,
    against: Against,
    detail: String,
) {
    out.push(Drift {
        entity,
        kind,
        against,
        detail,
    });
}

/// Entity `i` of the live truth belongs under id `i`: the converter assigns ids in drawing order, so
/// an entity found under another id is a reorder, and a truth entity found nowhere is dropped.
fn against_truth(art: &Artifact, truth: &Value, out: &mut Vec<Drift>) {
    use Against::Truth;
    let want = live(truth);
    let n = art
        .curves
        .iter()
        .map(|(id, _)| *id as usize + 1)
        .max()
        .unwrap_or(0)
        .max(want.len());
    // curves by id; ids are dense from 0 in a well-formed artifact, but a gap is reported as such
    let curve_at = |i: usize| {
        art.curves
            .iter()
            .find(|(id, _)| *id as usize == i)
            .map(|(_, c)| c)
    };
    let mut used = vec![false; n];
    let mut settled = vec![false; want.len()];
    for (i, t) in want.iter().enumerate() {
        if let Some(c) = curve_at(i).filter(|c| is(t, c)) {
            let _ = c;
            used[i] = true;
            settled[i] = true;
        }
    }
    // the right geometry under the wrong id
    for (i, t) in want.iter().enumerate() {
        if settled[i] {
            continue;
        }
        if let Some(j) = (0..n).find(|j| !used[*j] && curve_at(*j).is_some_and(|c| is(t, c))) {
            used[j] = true;
            settled[i] = true;
            push(
                out,
                Some(j as u32),
                DriftKind::ReorderedIds,
                Truth,
                format!(
                    "{} is under id {j}, the drawing order puts it at id {i}",
                    t["id"].as_str().unwrap_or("?")
                ),
            );
        }
    }
    // what is left is changed in place, dropped, or extra
    for (i, t) in want.iter().enumerate() {
        if settled[i] {
            continue;
        }
        let (class, g) = target(t);
        let name = t["id"].as_str().unwrap_or("?");
        match curve_at(i).filter(|_| !used[i]) {
            Some(c) => {
                used[i] = true;
                if !classes_of(class).contains(&kind_of(c)) {
                    push(
                        out,
                        Some(i as u32),
                        DriftKind::SwappedClass,
                        Truth,
                        format!(
                            "{name} is a {class} in the truth and a {:?} in the artifact",
                            kind_of(c)
                        ),
                    );
                } else {
                    let why = differs(class, g, t, c);
                    let radius = why
                        .iter()
                        .any(|w| w.contains("is off the truth") || w.contains("semi"))
                        && matches!(class, "circle" | "ellipse");
                    let kind = if radius {
                        DriftKind::ChangedRadius
                    } else {
                        DriftKind::MovedVertex
                    };
                    push(
                        out,
                        Some(i as u32),
                        kind,
                        Truth,
                        format!("{name}: {}", why.join("; ")),
                    );
                }
            }
            None => push(
                out,
                Some(i as u32),
                DriftKind::DroppedEntity,
                Truth,
                format!("{name} ({class}) is not in the artifact"),
            ),
        }
    }
    for (id, c) in &art.curves {
        if !used[*id as usize] {
            push(
                out,
                Some(*id),
                DriftKind::ExtraEntity,
                Truth,
                format!("a {:?} the fixture does not have", kind_of(c)),
            );
        }
    }
    unit_and_bounds(art, truth_unit(truth), out, Truth);
}

fn truth_unit(truth: &Value) -> Option<f64> {
    let v = &truth["units"]["scale_to_canonical"];
    v.get("f64_hex").is_some().then(|| num(v))
}

/// The unit and bounds checks both comparisons share. The bounds must be those of the geometry
/// itself, measured by PVec: equal for curves PVec bounds in closed form, and between the
/// endpoints' box and the control hull for a spline (which PVec bounds by its hull).
fn unit_and_bounds(art: &Artifact, unit: Option<f64>, out: &mut Vec<Drift>, against: Against) {
    if let Some(u) = unit
        && art.metres_per_unit.to_bits() != u.to_bits()
    {
        push(
            out,
            None,
            DriftKind::ChangedUnit,
            against,
            format!(
                "the manifest says {} metres per unit, the fixture {u}",
                art.metres_per_unit
            ),
        );
    }
    let mut hull: Option<[[f64; 2]; 2]> = None;
    let mut ends: Option<[[f64; 2]; 2]> = None;
    let mut exact = true;
    let grow = |b: &mut Option<[[f64; 2]; 2]>, lo: [f64; 2], hi: [f64; 2]| {
        *b = Some(match *b {
            None => [lo, hi],
            Some(o) => [
                [o[0][0].min(lo[0]), o[0][1].min(lo[1])],
                [o[1][0].max(hi[0]), o[1][1].max(hi[1])],
            ],
        });
    };
    for (_, c) in &art.curves {
        if let Ok(b) = c.bounds() {
            grow(&mut hull, [b.min().x, b.min().y], [b.max().x, b.max().y]);
        }
        let [a, z] = end_points(c);
        grow(
            &mut ends,
            [a[0].min(z[0]), a[1].min(z[1])],
            [a[0].max(z[0]), a[1].max(z[1])],
        );
        exact &= !matches!(
            c,
            Curve::BSpline(_) | Curve::RationalBSpline(_) | Curve::Bezier(_) | Curve::Composite(_)
        );
    }
    let (Some(hull), Some(ends)) = (hull, ends) else {
        return;
    };
    let got = art.bounds;
    let sc = scale_of(&[hull[0][0], hull[0][1], hull[1][0], hull[1][1]]);
    let same = |a: [f64; 2], b: [f64; 2]| near_pt(a, b, sc);
    let ok = if exact {
        same(got[0], hull[0]) && same(got[1], hull[1])
    } else {
        // inside the hull, around the endpoints
        let le = |a: f64, b: f64| a <= b || near(a, b, sc);
        le(hull[0][0], got[0][0])
            && le(hull[0][1], got[0][1])
            && le(got[1][0], hull[1][0])
            && le(got[1][1], hull[1][1])
            && le(got[0][0], ends[0][0])
            && le(got[0][1], ends[0][1])
            && le(ends[1][0], got[1][0])
            && le(ends[1][1], got[1][1])
    };
    if !ok {
        push(
            out,
            None,
            DriftKind::ChangedBounds,
            against,
            format!("the manifest bounds are {got:?}, the geometry's are {hull:?}"),
        );
    }
}

// ---------------------------------------------------------------------- against the accepted

/// An entity as the artifact has it: id, class, named numbers.
type Described = (u32, String, Vec<(String, f64)>);

/// One entity line of a dump.
struct Line<'a> {
    id: u32,
    class: &'a str,
    params: Vec<(&'a str, f64)>,
}

fn parse_entity(line: &str) -> Option<Line<'_>> {
    let mut it = line.strip_prefix("e ")?.split(' ');
    let id = it.next()?.parse().ok()?;
    let class = it.next()?;
    let params = it
        .map(|kv| {
            let (k, v) = kv.split_once('=')?;
            Some((k, unshow(v)?))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Line { id, class, params })
}

fn against_accepted(art: &Artifact, accepted: &str, out: &mut Vec<Drift>) {
    use Against::Accepted;
    let field = |key: &str| {
        accepted
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .map(|r| r.split(' ').filter_map(unshow).collect::<Vec<f64>>())
    };
    let want: Vec<Line<'_>> = accepted.lines().filter_map(parse_entity).collect();
    let got: Vec<Described> = art
        .curves
        .iter()
        .map(|(id, c)| {
            let (class, p) = describe(c);
            (*id, class, p)
        })
        .collect();
    let same = |w: &Line<'_>, g: &Described| {
        w.class == g.1
            && w.params.len() == g.2.len()
            && w.params.iter().zip(&g.2).all(|(a, b)| {
                a.0 == b.0 && {
                    let sc = scale_of(&[a.1, b.1]);
                    near(a.1, b.1, sc)
                }
            })
    };
    let mut used = vec![false; got.len()];
    let mut settled = vec![false; want.len()];
    for (i, w) in want.iter().enumerate() {
        if let Some(j) = got.iter().position(|g| g.0 == w.id)
            && same(w, &got[j])
        {
            used[j] = true;
            settled[i] = true;
        }
    }
    for (i, w) in want.iter().enumerate() {
        if settled[i] {
            continue;
        }
        if let Some(j) = (0..got.len()).find(|j| !used[*j] && same(w, &got[*j])) {
            used[j] = true;
            settled[i] = true;
            push(
                out,
                Some(got[j].0),
                DriftKind::ReorderedIds,
                Accepted,
                format!(
                    "the {} accepted at id {} is now at id {}",
                    w.class, w.id, got[j].0
                ),
            );
        }
    }
    for (i, w) in want.iter().enumerate() {
        if settled[i] {
            continue;
        }
        match got.iter().position(|g| g.0 == w.id).filter(|j| !used[*j]) {
            Some(j) => {
                used[j] = true;
                let g = &got[j];
                if g.1 != w.class {
                    push(
                        out,
                        Some(w.id),
                        DriftKind::SwappedClass,
                        Accepted,
                        format!("accepted a {}, now a {}", w.class, g.1),
                    );
                } else if w.params.len() != g.2.len() {
                    push(
                        out,
                        Some(w.id),
                        DriftKind::MovedVertex,
                        Accepted,
                        format!("accepted {} numbers, now {}", w.params.len(), g.2.len()),
                    );
                } else {
                    for (a, b) in w.params.iter().zip(&g.2) {
                        let sc = scale_of(&[a.1, b.1]);
                        if a.0 != b.0 || !near(a.1, b.1, sc) {
                            push(
                                out,
                                Some(w.id),
                                kind_of_param(a.0),
                                Accepted,
                                format!("{}: accepted {}, now {}", a.0, a.1, b.1),
                            );
                        }
                    }
                }
            }
            None => push(
                out,
                Some(w.id),
                DriftKind::DroppedEntity,
                Accepted,
                format!("the accepted {} is gone", w.class),
            ),
        }
    }
    for (j, g) in got.iter().enumerate() {
        if !used[j] {
            push(
                out,
                Some(g.0),
                DriftKind::ExtraEntity,
                Accepted,
                format!("a {} that was not accepted", g.1),
            );
        }
    }
    let unit = field("unit ").and_then(|v| v.first().copied());
    unit_and_bounds(art, unit, out, Accepted);
    if let Some(b) = field("bounds ").filter(|b| b.len() == 4) {
        let sc = scale_of(&b);
        let moved =
            !near_pt(art.bounds[0], [b[0], b[1]], sc) || !near_pt(art.bounds[1], [b[2], b[3]], sc);
        let already = out
            .iter()
            .any(|d| d.kind == DriftKind::ChangedBounds && d.against == Accepted);
        if moved && !already {
            push(
                out,
                None,
                DriftKind::ChangedBounds,
                Accepted,
                format!("accepted bounds {b:?}, now {:?}", art.bounds),
            );
        }
    }
}
