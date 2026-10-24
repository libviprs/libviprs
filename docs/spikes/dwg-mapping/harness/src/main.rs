//! N1.3 measurement harness: the decoded records of the r0-open corpus, mapped into PVec's public
//! semantic API, compared with each fixture's `expected.math.json`. Prints a TSV on stdout.
//!
//! What this is not: an evaluator of record, a tessellator or a tolerance. Every curve operation in
//! a mapping is a call into `pvec-geometry`, and every comparison goes through
//! `pvec_numeric::Policy::V1` (NP-04 for the tolerance, NP-05 for equality). The few places where
//! this file does arithmetic of its own are labelled `truth side` (it turns a truth record's
//! parameters into points so the two can be compared) or `candidate` (it computes what a missing
//! PVec constructor would have to return, so that the missing capability is measured and not just
//! asserted). Neither belongs in libviprs: NG-B1 and NG-B5 of the N1.2 ADR forbid both there.
//!
//! The input is the ACadSharp read record of each probe (`read.acadsharp.json`). Its fields are the
//! fields of the decoder IR (`cad::Primitive`, #1029) one for one: centre, radius, angles, major axis
//! and ratio, vertices and bulges, degree, knots, controls, weights, closed and periodic flags. The
//! other reader's record (`read.acadrust.json`) must be identical, and this harness checks that.

use pvec_geometry::{
    BSpline, Bezier, Circle, Composite, Curve, Curve2d, Domain, Ellipse, GeometryError, Line,
    Placement2, Point2, Polyline, RationalBSpline, Side, Transform2, Trimmed, Vector2,
};
use pvec_numeric::Policy;
use serde_json::Value;
use std::f64::consts::TAU;
use std::fs;
use std::path::{Path, PathBuf};

const P: Policy = Policy::V1;
type Res<T> = Result<T, GeometryError>;

// ------------------------------------------------------------------------------------ json

fn hx(v: &Value) -> f64 {
    let s = v
        .as_str()
        .unwrap_or_else(|| panic!("not a hex string: {v}"));
    f64::from_bits(u64::from_str_radix(s.trim_start_matches("0x"), 16).expect("hex f64"))
}

/// A truth number: `{"decimal": .., "f64_hex": ..}`; the hex is the value.
fn th(v: &Value) -> f64 {
    hx(&v["f64_hex"])
}

fn hv(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(hx).collect()
}

fn tv(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(th).collect()
}

fn xy(v: &[f64]) -> Point2 {
    Point2::new(v[0], v[1])
}

struct Fx {
    id: String,
    sharp: Value,
    truth: Value,
    readers_identical: bool,
}

fn load(dir: &Path, id: &str) -> Fx {
    let rd = |f: &str| -> Value {
        let p = dir.join("fixtures").join(id).join(f);
        serde_json::from_str(
            &fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())),
        )
        .expect("json")
    };
    let (sharp, rust, truth) = (
        rd("read.acadsharp.json"),
        rd("read.acadrust.json"),
        rd("expected.math.json"),
    );
    let strip = |v: &Value| {
        let mut v = v.clone();
        if let Some(o) = v.as_object_mut() {
            o.remove("reader");
            o.remove("dwg_sha256");
        }
        v
    };
    let readers_identical = strip(&sharp) == strip(&rust);
    Fx {
        id: id.to_string(),
        sharp,
        truth,
        readers_identical,
    }
}

impl Fx {
    fn block(&self, name: &str) -> &Vec<Value> {
        self.sharp["blocks"]
            .as_array()
            .and_then(|bs| bs.iter().find(|b| b["name"] == name))
            .and_then(|b| b["entities"].as_array())
            .unwrap_or_else(|| panic!("{}: no block {name}", self.id))
    }
    fn model(&self) -> &Vec<Value> {
        self.block("*Model_Space")
    }
    fn truth_ent(&self, i: usize) -> &Value {
        &self.truth["entities"][i]
    }
}

// ------------------------------------------------------------------------------------ results

fn fmt_e(x: f64) -> String {
    format!("{x:.3e}")
}

fn fmt_p(x: f64) -> String {
    format!("{:.9}", x + 0.0)
}

fn fmt_pt(p: Point2) -> String {
    format!("({},{})", fmt_p(p.x), fmt_p(p.y))
}

#[derive(Default)]
struct Acc {
    n: usize,
    worst: f64,
    ratio: f64,
    fails: Vec<String>,
    notes: Vec<(String, String)>,
}

impl Acc {
    fn note(&mut self, k: &str, v: impl ToString) {
        let v = v.to_string().replace([';', '\t', '\n'], " ");
        self.notes.push((k.to_string(), v));
    }

    /// A field the DWG stores verbatim: the PVec value must hold the same f64.
    fn bits(&mut self, name: &str, want: f64, got: f64) {
        self.n += 1;
        if want != got {
            let d = (want - got).abs();
            self.worst = self.worst.max(d);
            self.ratio = f64::INFINITY;
            self.fails
                .push(format!("{name}: want {want:e} got {got:e}"));
        }
    }

    /// A field the DWG stores as a derived value, or that PVec recomputes: NP-05 at feature scale.
    fn near(&mut self, name: &str, want: f64, got: f64, scale: f64) {
        self.n += 1;
        let d = (want - got).abs();
        let tol = P.tol(scale, want.abs().max(got.abs()));
        self.worst = self.worst.max(d);
        self.ratio = self.ratio.max(d / tol);
        if !P.near(want, got, scale) {
            self.fails
                .push(format!("{name}: want {want:e} got {got:e} tol {tol:e}"));
        }
    }

    fn near_pt(&mut self, name: &str, want: Point2, got: Point2, scale: f64) {
        self.near(&format!("{name}.x"), want.x, got.x, scale);
        self.near(&format!("{name}.y"), want.y, got.y, scale);
    }

    fn truth(&mut self, name: &str, ok: bool) {
        self.n += 1;
        if !ok {
            self.fails.push(name.to_string());
        }
    }
}

struct Rec {
    id: String,
    kind: &'static str,
    fixture: String,
    dwg_class: String,
    target: String,
    result: String,
    reason: String,
    acc: Acc,
}

impl Rec {
    fn line(&self) -> String {
        let a = &self.acc;
        let notes = a
            .notes
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(";");
        let ratio = if a.ratio.is_infinite() {
            "inf".to_string()
        } else {
            fmt_e(a.ratio)
        };
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.id,
            self.kind,
            self.fixture,
            self.dwg_class,
            self.target,
            self.result,
            if self.reason.is_empty() {
                "-"
            } else {
                &self.reason
            },
            a.n,
            fmt_e(a.worst),
            ratio,
            notes
        )
    }
}

fn kind_name(c: &Curve) -> String {
    match c {
        Curve::Line(_) => "line".into(),
        Curve::Circle(_) => "circle".into(),
        Curve::Ellipse(_) => "ellipse".into(),
        Curve::Polyline(_) => "polyline".into(),
        Curve::Trimmed(t) => format!("trimmed_curve over {}", kind_name(t.basis())),
        Curve::Composite(_) => "composite_curve".into(),
        Curve::Bezier(_) => "bezier_curve".into(),
        Curve::BSpline(_) => "b_spline_curve_with_knots".into(),
        Curve::RationalBSpline(_) => "rational_b_spline_curve".into(),
        _ => "unknown".into(),
    }
}

fn refusal(e: &GeometryError) -> String {
    format!("{e}").replace([';', '\t', '='], " ")
}

fn sense_word(c: &Curve) -> &'static str {
    match c {
        Curve::Trimmed(t) if t.trim().reversed => "cw",
        Curve::Trimmed(_) => "ccw",
        _ => "n/a",
    }
}

fn bounds_note(acc: &mut Acc, c: &Curve) {
    match c.bounds() {
        Ok(b) => acc.note("bounds", format!("{}-{}", fmt_pt(b.min()), fmt_pt(b.max()))),
        Err(e) => acc.note("bounds", refusal(&e)),
    }
}

/// Parameter domain of the PVec value against the truth record's `parameter_domain`. Returns true
/// when they are the same interval (NP-05 at the interval's own scale).
fn domain_cmp(acc: &mut Acc, truth_dom: &Value, d: Domain) -> bool {
    let (tlo, thi) = (th(&truth_dom["start"]), th(&truth_dom["end"]));
    let scale = (thi - tlo).abs().max(d.width());
    let same = P.near(tlo, d.lo(), scale) && P.near(thi, d.hi(), scale);
    acc.note("domain_truth", format!("[{},{}]", fmt_p(tlo), fmt_p(thi)));
    acc.note(
        "domain_pvec",
        format!("[{},{}]", fmt_p(d.lo()), fmt_p(d.hi())),
    );
    acc.note("domain_preserved", same);
    same
}

fn finish(
    fx: &Fx,
    kind: &'static str,
    dwg_class: &str,
    target: String,
    mut acc: Acc,
    domain_same: bool,
    capability: Option<&str>,
) -> Rec {
    acc.note("readers_identical", fx.readers_identical);
    let (result, reason) = match capability {
        _ if !acc.fails.is_empty() => {
            acc.note("fails", acc.fails.join(" | "));
            ("FAILED".to_string(), String::new())
        }
        Some(c) => ("RequiresPVecCapability".to_string(), c.to_string()),
        None if domain_same => ("exact".to_string(), String::new()),
        None => ("equivalent".to_string(), String::new()),
    };
    Rec {
        id: fx.id.clone(),
        kind,
        fixture: fx.id.clone(),
        dwg_class: dwg_class.to_string(),
        target,
        result,
        reason,
        acc,
    }
}

fn refused(
    fx: &Fx,
    dwg_class: &str,
    target: &str,
    e: &GeometryError,
    capability: &str,
    mut acc: Acc,
) -> Rec {
    acc.note("pvec_refusal", refusal(e));
    acc.note("pvec_refusal_rule", e.rule());
    finish(
        fx,
        "row",
        dwg_class,
        format!("{target} (RequiresPVecCapability: {capability})"),
        acc,
        true,
        Some(capability),
    )
}

// ------------------------------------------------------------------------------------ truth side

/// Truth side: a point of the conic `c + a cos t X + b sin t Y`, from the truth record's own numbers.
fn truth_conic_point(c: [f64; 2], x: [f64; 2], y: [f64; 2], a: f64, b: f64, t: f64) -> Point2 {
    let (s, co) = t.sin_cos();
    Point2::new(
        c[0] + a * co * x[0] + b * s * y[0],
        c[1] + a * co * x[1] + b * s * y[1],
    )
}

/// Truth side: the frame of a truth conic record. The second axis is `normal x x_axis`.
fn truth_frame(g: &Value) -> ([f64; 2], [f64; 2], [f64; 2]) {
    let c = tv(&g["center"]);
    let x = tv(&g.get("x_axis").unwrap_or(&g["major_axis"]));
    let n = tv(&g["normal"]);
    ([c[0], c[1]], [x[0], x[1]], [-n[2] * x[1], n[2] * x[0]])
}

// ------------------------------------------------------------------------------------ mappings

fn is_plus_z(n: &[f64]) -> bool {
    n[0] == 0.0 && n[1] == 0.0 && n[2] == 1.0
}

/// DWG LINE -> trimmed_curve over line. Every operation is PVec's: the direction is normalised by
/// `Line::new`, the length is `Point2::distance`.
fn map_line(e: &Value) -> Res<Curve> {
    let (a, b) = (xy(&hv(&e["start"])), xy(&hv(&e["end"])));
    let line = Line::new(a, a.vector_to(b))?;
    Ok(Curve::Trimmed(Trimmed::new(
        Curve::Line(line),
        0.0,
        a.distance(b),
        true,
    )?))
}

/// DWG CIRCLE -> circle. The frame's reference direction is the x axis, a convention of the adapter
/// because a DWG circle has none.
fn map_circle(e: &Value) -> Res<Curve> {
    let c = xy(&hv(&e["center"]));
    let frame = Placement2::new(c, Vector2::new(1.0, 0.0))?;
    Ok(Curve::Circle(Circle::new(frame, hx(&e["radius"]))?))
}

/// DWG ARC -> trimmed_curve over circle, counter-clockwise from the start angle to the end angle.
fn map_arc(e: &Value) -> Res<Curve> {
    let circle = map_circle(e)?;
    Ok(Curve::Trimmed(Trimmed::new(
        circle,
        hx(&e["start_angle"]),
        hx(&e["end_angle"]),
        true,
    )?))
}

fn line_truth_checks(acc: &mut Acc, c: &Curve, start: Point2, end: Point2) {
    let len = start.distance(end);
    let d = c.domain();
    if let (Ok(s), Ok(t)) = (c.eval(d.lo()), c.eval(d.hi())) {
        acc.near_pt("start", start, s, len);
        acc.near_pt("end", end, t, len);
    }
}

fn row_line(fx: &Fx) -> Rec {
    let e = &fx.model()[0];
    let mut acc = Acc::default();
    let t = fx.truth_ent(0);
    let (start, end) = (tv(&t["geometry"]["start"]), tv(&t["geometry"]["end"]));
    let c = match map_line(e) {
        Ok(c) => c,
        Err(er) => {
            return refused(
                fx,
                "LINE",
                "trimmed_curve over line",
                &er,
                "a segment from two points",
                acc,
            )
        }
    };
    if let Curve::Trimmed(tr) = &c {
        if let Curve::Line(l) = tr.basis() {
            acc.bits("origin.x", start[0], l.origin().x);
            acc.bits("origin.y", start[1], l.origin().y);
        }
    }
    line_truth_checks(&mut acc, &c, xy(&start), xy(&end));
    let same = domain_cmp(&mut acc, &t["parameter_domain"], c.domain());
    acc.note("end_stored_verbatim", false);
    bounds_note(&mut acc, &c);
    finish(fx, "row", "LINE", kind_name(&c), acc, same, None)
}

/// The vertices of a DWG LWPOLYLINE or POLYLINE record: `[x, y, bulge]`.
fn vertices(e: &Value) -> (Vec<Point2>, Vec<f64>) {
    let vs = e["vertices"].as_array().expect("vertices");
    let pts = vs.iter().map(|v| xy(&hv(v))).collect();
    let bulges = vs.iter().map(|v| hv(v)[2]).collect();
    (pts, bulges)
}

fn row_polyline(fx: &Fx) -> Rec {
    let e = &fx.model()[0];
    let closed = e["closed"].as_bool().unwrap_or(false);
    let class = match (e["type"].as_str().unwrap(), closed) {
        ("LWPOLYLINE", false) => "LWPOLYLINE (open)".to_string(),
        ("LWPOLYLINE", true) => "LWPOLYLINE (closed)".to_string(),
        (o, false) => format!("{o} (2D, open)"),
        (o, true) => format!("{o} (2D, closed)"),
    };
    let mut acc = Acc::default();
    let t = fx.truth_ent(0);
    let (mut pts, bulges) = vertices(e);
    acc.truth("no bulge on this fixture", bulges.iter().all(|b| *b == 0.0));
    // A closed polyline has no flag in PVec: the first vertex is repeated (profile section 5).
    let n_src = pts.len();
    if closed {
        pts.push(pts[0]);
    }
    let pl = match Polyline::new(pts) {
        Ok(p) => p,
        Err(er) => return refused(fx, &class, "polyline", &er, "polyline", acc),
    };
    let tv_ = t["geometry"]["vertices"].as_array().unwrap();
    acc.truth("vertex count", tv_.len() == n_src);
    for (i, v) in tv_.iter().enumerate() {
        let w = tv(v);
        acc.bits(&format!("v{i}.x"), w[0], pl.vertices()[i].x);
        acc.bits(&format!("v{i}.y"), w[1], pl.vertices()[i].y);
    }
    if closed {
        let (f, l) = (pl.vertices()[0], pl.vertices()[n_src]);
        acc.bits("closing.x", f.x, l.x);
        acc.bits("closing.y", f.y, l.y);
        acc.truth(
            "truth says closed by end point",
            t["closure"]["closed_by_endpoint"] == true,
        );
    }
    acc.note("source_closed_flag", closed);
    acc.note("source_vertices", n_src);
    acc.note("pvec_vertices", pl.vertices().len());
    let c = Curve::Polyline(pl);
    let same = domain_cmp(&mut acc, &t["parameter_domain"], c.domain());
    if let Ok(b) = c.bounds() {
        // Truth side: the extent of the vertices, to say whether PVec's polyline bounds are tight.
        let (mut lo, mut hi) = (b.min(), b.max());
        for v in tv_ {
            let w = tv(v);
            lo = Point2::new(lo.x.min(w[0]), lo.y.min(w[1]));
            hi = Point2::new(hi.x.max(w[0]), hi.y.max(w[1]));
        }
        acc.bits("bounds.min.x", lo.x, b.min().x);
        acc.bits("bounds.max.y", hi.y, b.max().y);
    }
    bounds_note(&mut acc, &c);
    finish(fx, "row", &class, kind_name(&c), acc, same, None)
}

/// CANDIDATE, not a mapping. What a PVec bulge-to-arc constructor would have to return: the centre,
/// radius, canonical trim start, sweep and sense of the arc a chord and a bulge describe. It exists
/// so the missing capability is measured against the fixture's own expected arc. The chord and the
/// bulge are the only inputs; the standard relations are theta = 4 atan b, r = L (1 + b^2) / (4 |b|)
/// and the centre at (L / 2) (1 - b^2) / (2 b) along the left normal of the chord from its midpoint.
struct CandidateArc {
    centre: Point2,
    radius: f64,
    start: f64,
    sweep: f64,
    ccw: bool,
}

fn candidate_bulge_arc(p0: Point2, p1: Point2, b: f64) -> CandidateArc {
    let c = p0.vector_to(p1);
    let l = c.length();
    let left = Vector2::new(-c.y / l, c.x / l);
    let mid = Point2::new((p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0);
    let off = (l / 2.0) * (1.0 - b * b) / (2.0 * b);
    let centre = Point2::new(mid.x + off * left.x, mid.y + off * left.y);
    let radius = l * (1.0 + b * b) / (4.0 * b.abs());
    let a0 = (p0.y - centre.y).atan2(p0.x - centre.x);
    let a1 = (p1.y - centre.y).atan2(p1.x - centre.x);
    let ccw = b > 0.0;
    CandidateArc {
        centre,
        radius,
        start: (if ccw { a0 } else { a1 }).rem_euclid(TAU),
        sweep: 4.0 * b.abs().atan(),
        ccw,
    }
}

fn row_bulge(fx: &Fx) -> Rec {
    let e = &fx.model()[0];
    let mut acc = Acc::default();
    let t = fx.truth_ent(0);
    let arc_t = fx.truth_ent(1);
    let class = format!(
        "{} (bulge {})",
        e["type"].as_str().unwrap(),
        if vertices(e).1[0] > 0.0 {
            "positive"
        } else {
            "negative"
        }
    );
    assert_eq!(
        arc_t["class"], "circular_arc",
        "{}: second truth entity",
        fx.id
    );
    let (pts, bulges) = vertices(e);
    let (p0, p1, b) = (pts[0], pts[1], bulges[0]);
    acc.note("compared_to", "expected.math.json");
    acc.note("bulge", format!("{b}"));
    // Vertices and bulge are stored verbatim.
    let tvx = t["geometry"]["vertices"].as_array().unwrap();
    for (i, v) in tvx.iter().enumerate() {
        let w = tv(v);
        acc.bits(&format!("v{i}.x"), w[0], pts[i].x);
        acc.bits(&format!("v{i}.y"), w[1], pts[i].y);
    }
    acc.bits("bulge", th(&t["geometry"]["bulges"][0]), b);

    // The expected arc, from the fixture's own record.
    let g = &arc_t["geometry"];
    let (ec, er, es, ew) = (
        tv(&g["center"]),
        th(&g["radius"]),
        th(&g["start_angle"]),
        th(&g["sweep_angle"]),
    );
    // The record states the sense only when it is reversed against the source; for a positive bulge it
    // is silent, and I read that as counter-clockwise. The arc end points below would not meet the
    // polyline vertices if that reading were wrong.
    let esense = arc_t["orientation"]["sense_about_normal"]
        .as_str()
        .unwrap_or("CCW")
        .to_string();
    acc.note(
        "expected_sense_stated",
        arc_t["orientation"]["sense_about_normal"].is_string(),
    );
    acc.note("expected_centre", fmt_pt(xy(&ec)));
    acc.note("expected_radius", fmt_p(er));
    acc.note("expected_start_angle", fmt_p(es));
    acc.note("expected_sweep", fmt_p(ew));
    acc.note("expected_sense", &esense);

    // Candidate against expected.
    let cand = candidate_bulge_arc(p0, p1, b);
    let chord = p0.distance(p1);
    acc.near_pt("candidate centre", xy(&ec), cand.centre, er);
    acc.near("candidate radius", er, cand.radius, er);
    acc.near("candidate start angle", es, cand.start, TAU);
    acc.near("candidate sweep", ew, cand.sweep, TAU);
    acc.note("candidate_centre", fmt_pt(cand.centre));
    acc.note("candidate_radius", fmt_p(cand.radius));
    acc.note("candidate_start_angle", fmt_p(cand.start));
    acc.note("candidate_sweep", fmt_p(cand.sweep));
    let csense = if cand.ccw { "CCW" } else { "CW" };
    acc.note("candidate_sense", csense);
    acc.truth(
        "sweep sign (candidate sense equals expected sense)",
        csense == esense,
    );

    // PVec holds the expected arc itself: a trimmed circle built from the truth numbers, whose end
    // points must be the polyline's vertices. The source runs from p0 to p1.
    let ccw = esense == "CCW";
    let build = |centre: Point2, radius: f64, start: f64, sweep: f64, ccw: bool| -> Res<Curve> {
        let circle = Curve::Circle(Circle::new(
            Placement2::new(centre, Vector2::new(1.0, 0.0))?,
            radius,
        )?);
        let (t1, t2) = if ccw {
            (start, start + sweep)
        } else {
            (start + sweep, start)
        };
        Ok(Curve::Trimmed(Trimmed::new(circle, t1, t2, ccw)?))
    };
    match build(xy(&ec), er, es, ew, ccw) {
        Ok(arc) => {
            acc.note("pvec_arc_from_expected", "ok");
            acc.note("pvec_sense", sense_word(&arc));
            if let Curve::Trimmed(tr) = &arc {
                let tt = tr.trim();
                acc.near("pvec trim start", es, tt.start, TAU);
                acc.near("pvec trim span", ew, tt.span, TAU);
                acc.truth("pvec reversed flag follows the sign", tt.reversed == !ccw);
            }
            let d = arc.domain();
            if let (Ok(s), Ok(en)) = (arc.eval(d.lo()), arc.eval(d.hi())) {
                acc.near_pt("pvec arc start vs vertex 0", p0, s, chord);
                acc.near_pt("pvec arc end vs vertex 1", p1, en, chord);
            }
            // The composite of that one segment is the polyline's PVec form.
            match Composite::new(vec![arc.clone()]) {
                Ok(comp) => {
                    let cc = Curve::Composite(comp);
                    acc.note("pvec_composite", kind_name(&cc));
                    let _ = domain_cmp(&mut acc, &t["parameter_domain"], cc.domain());
                    bounds_note(&mut acc, &cc);
                }
                Err(er2) => acc.note("pvec_composite", refusal(&er2)),
            }
        }
        Err(er2) => acc.note("pvec_arc_from_expected", refusal(&er2)),
    }
    // And from the candidate, to see that its numbers are good enough for PVec to accept and close.
    if let Ok(arc) = build(cand.centre, cand.radius, cand.start, cand.sweep, cand.ccw) {
        let d = arc.domain();
        if let (Ok(s), Ok(en)) = (arc.eval(d.lo()), arc.eval(d.hi())) {
            acc.near_pt("candidate arc start vs vertex 0", p0, s, chord);
            acc.near_pt("candidate arc end vs vertex 1", p1, en, chord);
        }
    }
    let cap = "bulge-to-arc constructor (chord and bulge to centre, radius and trim)";
    finish(
        fx,
        "row",
        &class,
        format!("composite_curve of trimmed_curve over circle (RequiresPVecCapability: {cap})"),
        acc,
        true,
        Some(cap),
    )
}

fn row_circle(fx: &Fx) -> Rec {
    let e = &fx.model()[0];
    let mut acc = Acc::default();
    let t = fx.truth_ent(0);
    let g = &t["geometry"];
    acc.truth("normal is +z", is_plus_z(&hv(&e["normal"])));
    let c = match map_circle(e) {
        Ok(c) => c,
        Err(er) => return refused(fx, "CIRCLE", "circle", &er, "circle", acc),
    };
    if let Curve::Circle(ci) = &c {
        let tc = tv(&g["center"]);
        acc.bits("centre.x", tc[0], ci.placement().origin().x);
        acc.bits("centre.y", tc[1], ci.placement().origin().y);
        acc.bits("radius", th(&g["radius"]), ci.radius());
        acc.bits("x_axis.x", tv(&g["x_axis"])[0], ci.placement().x_axis().x);
    }
    let same = domain_cmp(&mut acc, &t["parameter_domain"], c.domain());
    if let Ok(b) = c.bounds() {
        // Truth side: a circle's extent is its centre plus and minus the radius.
        let (tc, r) = (tv(&g["center"]), th(&g["radius"]));
        acc.near("bounds.min.x", tc[0] - r, b.min().x, r);
        acc.near("bounds.max.y", tc[1] + r, b.max().y, r);
    }
    bounds_note(&mut acc, &c);
    finish(fx, "row", "CIRCLE", kind_name(&c), acc, same, None)
}

/// Truth side: samples of a trimmed conic against the truth record's own parameters. PVec's tau runs
/// from the source start, so fraction f of PVec's domain is truth parameter `start + f * sweep`.
fn locus(acc: &mut Acc, c: &Curve, g: &Value, a: f64, b: f64, start: f64, sweep: f64, scale: f64) {
    let (ctr, x, y) = truth_frame(g);
    let d = c.domain();
    for k in 0..=8 {
        let f = k as f64 / 8.0;
        if let Ok(p) = c.eval(d.lo() + f * d.width()) {
            let q = truth_conic_point(ctr, x, y, a, b, start + f * sweep);
            acc.near_pt(&format!("locus {k}/8"), Point2::new(q.x, q.y), p, scale);
        }
    }
    acc.note("locus_samples", 9);
}

fn row_arc(fx: &Fx) -> Rec {
    let e = &fx.model()[0];
    let mut acc = Acc::default();
    let t = fx.truth_ent(0);
    let g = &t["geometry"];
    acc.truth("normal is +z", is_plus_z(&hv(&e["normal"])));
    let c = match map_arc(e) {
        Ok(c) => c,
        Err(er) => {
            return refused(
                fx,
                "ARC",
                "trimmed_curve over circle",
                &er,
                "circular arc",
                acc,
            )
        }
    };
    let (r, s, w) = (
        th(&g["radius"]),
        th(&g["start_angle"]),
        th(&g["sweep_angle"]),
    );
    if let Curve::Trimmed(tr) = &c {
        if let Curve::Circle(ci) = tr.basis() {
            let tc = tv(&g["center"]);
            acc.bits("centre.x", tc[0], ci.placement().origin().x);
            acc.bits("centre.y", tc[1], ci.placement().origin().y);
            acc.bits("radius", r, ci.radius());
        }
        // The DWG stores the end angle; the sweep is derived by PVec's trim normalisation (NP-11).
        acc.near("trim start", s, tr.trim().start, TAU);
        acc.near("trim span", w, tr.trim().span, TAU);
        acc.note("source_end_angle", fmt_p(hx(&e["end_angle"])));
        acc.note("seam_crossed", hx(&e["end_angle"]) < hx(&e["start_angle"]));
    }
    locus(&mut acc, &c, g, r, r, s, w, r);
    let same = domain_cmp(&mut acc, &t["parameter_domain"], c.domain());
    bounds_note(&mut acc, &c);
    finish(fx, "row", "ARC", kind_name(&c), acc, same, None)
}

/// CANDIDATE, not a mapping: the semi-axes of a DWG ELLIPSE are the length of its major-axis vector
/// and that length times the ratio. The length is PVec's (`Vector2::length`); the product is the
/// one operation libviprs would have to do itself, which is why the row is a gap.
fn ellipse_axes(e: &Value) -> (Vector2, f64, f64) {
    let m = hv(&e["major_axis"]);
    let major = Vector2::new(m[0], m[1]);
    let a = major.length();
    (major, a, hx(&e["ratio"]) * a)
}

fn row_ellipse(fx: &Fx) -> Rec {
    let e = &fx.model()[0];
    let class = "ELLIPSE";
    let mut acc = Acc::default();
    let t = fx.truth_ent(0);
    let g = &t["geometry"];
    acc.truth("normal is +z", is_plus_z(&hv(&e["normal"])));
    let (major, a, b) = ellipse_axes(e);
    let (s_param, e_param) = (hx(&e["start_param"]), hx(&e["end_param"]));
    let frame = match Placement2::new(xy(&hv(&e["center"])), major) {
        Ok(f) => f,
        Err(er) => return refused(fx, class, "ellipse", &er, "ellipse", acc),
    };
    let ell = match Ellipse::new(frame, a, b) {
        Ok(x) => x,
        Err(er) => return refused(fx, class, "ellipse", &er, "ellipse", acc),
    };
    let tc = tv(&g["center"]);
    let tm = tv(&g["major_axis"]);
    let (sm, sn) = (th(&g["semi_major"]), th(&g["semi_minor"]));
    acc.bits("centre.x", tc[0], ell.placement().origin().x);
    acc.bits("centre.y", tc[1], ell.placement().origin().y);
    // Derived by the DWG: a unit axis, and the two semi-axes. Policy comparison, as in the corpus.
    acc.near("major axis x", tm[0], ell.placement().x_axis().x, 1.0);
    acc.near("major axis y", tm[1], ell.placement().x_axis().y, 1.0);
    acc.near("semi_axis_1", sm, ell.semi_axis_1(), sm);
    acc.near("semi_axis_2", sn, ell.semi_axis_2(), sm);
    acc.note("source_ratio", fmt_p(hx(&e["ratio"])));
    acc.note(
        "libviprs_side_arithmetic",
        "one product: ratio times the major length",
    );

    let whole = Curve::Ellipse(ell);
    let c = match Trimmed::new(whole.clone(), s_param, e_param, true) {
        Ok(tr) => Curve::Trimmed(tr),
        Err(er) => {
            return refused(
                fx,
                class,
                "trimmed_curve over ellipse",
                &er,
                "ellipse trim",
                acc,
            )
        }
    };
    let is_arc = t["class"] == "elliptical_arc";
    let (ts, tw) = if is_arc {
        (th(&g["start_angle"]), th(&g["sweep_angle"]))
    } else {
        (0.0, TAU)
    };
    if let Curve::Trimmed(tr) = &c {
        acc.near("trim start", ts, tr.trim().start, TAU);
        acc.near("trim span", tw, tr.trim().span, TAU);
        acc.note("pvec_trim_span_is_full_period", tr.trim().span == TAU);
    }
    locus(&mut acc, &c, g, sm, sn, ts, tw, sm);
    let same = domain_cmp(&mut acc, &t["parameter_domain"], c.domain());
    bounds_note(&mut acc, &c);
    let cap = "ellipse from a major-axis vector and a ratio";
    let target = format!("{} (RequiresPVecCapability: {cap})", kind_name(&c));
    let mut rec = finish(fx, "row", class, target, acc, same, Some(cap));
    rec.dwg_class = if is_arc {
        "ELLIPSE (arc)".into()
    } else {
        "ELLIPSE".into()
    };
    rec
}

/// Probe: the untrimmed `ellipse` built from the truth's own semi-axes, so PVec's own value is
/// measured with no libviprs arithmetic at all.
fn probe_ellipse_from_truth(fx: &Fx, id: &str) -> Rec {
    let e = &fx.model()[0];
    let t = fx.truth_ent(0);
    let g = &t["geometry"];
    let mut acc = Acc::default();
    let (sm, sn) = (th(&g["semi_major"]), th(&g["semi_minor"]));
    let tm = tv(&g["major_axis"]);
    let frame = Placement2::new(xy(&hv(&e["center"])), Vector2::new(tm[0], tm[1])).expect("frame");
    let ell = Ellipse::new(frame, sm, sn).expect("ellipse from truth semi-axes");
    acc.bits("semi_axis_1", sm, ell.semi_axis_1());
    acc.bits("semi_axis_2", sn, ell.semi_axis_2());
    acc.bits("centre.x", tv(&g["center"])[0], ell.placement().origin().x);
    let c = if t["class"] == "elliptical_arc" {
        let (ts, tw) = (th(&g["start_angle"]), th(&g["sweep_angle"]));
        let tr = Trimmed::new(Curve::Ellipse(ell), ts, ts + tw, true).expect("trim");
        acc.near("trim start", ts, tr.trim().start, TAU);
        acc.near("trim span", tw, tr.trim().span, TAU);
        Curve::Trimmed(tr)
    } else {
        Curve::Ellipse(ell)
    };
    let same = domain_cmp(&mut acc, &t["parameter_domain"], c.domain());
    acc.note(
        "purpose",
        "PVec holds the truth ellipse with no libviprs arithmetic",
    );
    let mut r = finish(fx, "probe", "ELLIPSE", kind_name(&c), acc, same, None);
    r.id = id.to_string();
    r
}

// ------------------------------------------------------------------------------------ splines

fn expand_knots(k: &Value) -> Vec<f64> {
    let mut out = Vec::new();
    for e in k.as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        for _ in 0..e["multiplicity"].as_u64().unwrap() {
            out.push(th(&e["value"]));
        }
    }
    out
}

fn spline_curve(e: &Value) -> Res<Curve> {
    let degree = e["degree"].as_u64().unwrap() as usize;
    let cps: Vec<Point2> = e["control_points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| xy(&hv(v)))
        .collect();
    let knots = hv(&e["knots"]);
    let weights = hv(&e["weights"]);
    // Weights present means rational, never demoted (NG-03).
    if e["rational"] == true || !weights.is_empty() {
        let mut s = RationalBSpline::new(degree, cps, knots, weights)?;
        if e["closed"] == true {
            s = s.declare_closed()?;
        }
        Ok(Curve::RationalBSpline(s))
    } else {
        let mut s = BSpline::new(degree, cps, knots)?;
        if e["closed"] == true {
            s = s.declare_closed()?;
        }
        Ok(Curve::BSpline(s))
    }
}

fn row_spline(fx: &Fx) -> Rec {
    let e = &fx.model()[0];
    let mut acc = Acc::default();
    let t = fx.truth_ent(0);
    let g = &t["geometry"];
    let class = "SPLINE";
    let tk = expand_knots(&g["knots"]);
    let tcps: Vec<Vec<f64>> = g["control_points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| tv(v))
        .collect();
    acc.note("degree", g["degree"].as_u64().unwrap());
    acc.note("control_points", tcps.len());
    acc.note(
        "knots",
        if tk.is_empty() {
            hv(&e["knots"]).len()
        } else {
            tk.len()
        },
    );
    acc.note("source_periodic", e["periodic"] == true);
    acc.note("source_closed_flag", e["closed"] == true);
    acc.note("source_rational", e["rational"] == true);
    let c = match spline_curve(e) {
        Ok(c) => c,
        Err(er) => {
            let n = tcps.len();
            let deg = g["degree"].as_u64().unwrap() as usize;
            acc.note("pvec_needs_knots", n + deg + 1);
            let cap = "periodic (wrapped) B-spline, or an exact unwrap to the clamped form";
            return refused(fx, class, "b_spline_curve_with_knots", &er, cap, acc);
        }
    };
    let (deg, cps, knots, weights): (usize, Vec<Point2>, Vec<f64>, Vec<f64>) = match &c {
        Curve::BSpline(s) => (
            s.degree(),
            s.control_points().to_vec(),
            s.knots().to_vec(),
            vec![],
        ),
        Curve::RationalBSpline(s) => (
            s.degree(),
            s.control_points().to_vec(),
            s.knots().to_vec(),
            s.weights().to_vec(),
        ),
        _ => unreachable!(),
    };
    acc.truth("degree", deg as u64 == g["degree"].as_u64().unwrap());
    acc.truth("control point count", cps.len() == tcps.len());
    for (i, p) in tcps.iter().enumerate() {
        acc.bits(&format!("cp{i}.x"), p[0], cps[i].x);
        acc.bits(&format!("cp{i}.y"), p[1], cps[i].y);
    }
    if tk.is_empty() {
        // A Bezier record has no knot array; the single-span clamped vector is implied.
        acc.note("truth_knots", "absent (single span implied)");
    } else {
        acc.truth("knot count", knots.len() == tk.len());
        for (i, k) in tk.iter().enumerate() {
            acc.bits(&format!("knot{i}"), *k, knots[i]);
        }
        acc.note("knots_identical", knots == tk);
    }
    if let Some(tw) = g.get("weights") {
        let tw = tv(tw);
        acc.truth("weight count", weights.len() == tw.len());
        for (i, w) in tw.iter().enumerate() {
            acc.bits(&format!("w{i}"), *w, weights[i]);
        }
        acc.note("weights_identical", weights == tw);
        let (mx, mn) = (
            tw.iter().cloned().fold(0.0, f64::max),
            tw.iter().cloned().fold(f64::INFINITY, f64::min),
        );
        acc.note("weight_max_over_min", fmt_p(mx / mn));
    }
    // Interior knots of full or repeated multiplicity, from the truth record's own multiplicities.
    for kn in g["knots"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
        let m = kn["multiplicity"].as_u64().unwrap();
        let v = th(&kn["value"]);
        let d = c.domain();
        if m > 1 && v > d.lo() && v < d.hi() {
            acc.note("interior_knot", format!("{v}"));
            acc.note("interior_multiplicity", m);
            let two = c.derivative(v, 1, Side::Both);
            acc.note(
                "two_sided_d1_at_knot",
                if two.is_err() { "refused" } else { "answered" },
            );
            if let Err(er) = &two {
                acc.note("two_sided_refusal_rule", er.rule());
            }
            let (l, r) = (
                c.derivative(v, 1, Side::Left),
                c.derivative(v, 1, Side::Right),
            );
            if let (Ok(l), Ok(r)) = (l, r) {
                acc.note(
                    "one_sided_d1_left",
                    format!("({},{})", fmt_p(l.x), fmt_p(l.y)),
                );
                acc.note(
                    "one_sided_d1_right",
                    format!("({},{})", fmt_p(r.x), fmt_p(r.y)),
                );
            }
            if let Ok(p) = c.eval(v) {
                acc.note("point_at_knot", fmt_pt(p));
            }
        }
    }
    let same = domain_cmp(&mut acc, &t["parameter_domain"], c.domain());
    // Closure: the source flag is what is declared (NP-20); the geometric fact is not inferred.
    let truth_closed =
        t["closure"]["closed_by_endpoint"] == true || t["closure"]["closed_by_knots"] == true;
    let declared = match &c {
        Curve::BSpline(s) => s.is_declared_closed(),
        Curve::RationalBSpline(s) => s.is_declared_closed(),
        _ => false,
    };
    acc.note("truth_closed", truth_closed);
    acc.note("pvec_declared_closed", declared);
    if let Ok(b) = c.bounds() {
        let (mut lo, mut hi) = (cps[0], cps[0]);
        for p in &cps {
            lo = Point2::new(lo.x.min(p.x), lo.y.min(p.y));
            hi = Point2::new(hi.x.max(p.x), hi.y.max(p.y));
        }
        acc.note("bounds_is_control_hull", b.min() == lo && b.max() == hi);
    }
    bounds_note(&mut acc, &c);
    let class_truth = t["class"].as_str().unwrap();
    let dwg = match class_truth {
        "bezier" => "SPLINE (single span)",
        "nurbs" => "SPLINE (rational)",
        _ => "SPLINE",
    };
    finish(
        fx,
        "row",
        dwg,
        kind_name(&c),
        acc,
        same && class_truth != "bezier",
        None,
    )
}

/// Probe: closed-by-end-point. The source says not closed, the truth says closed by the end point;
/// PVec never infers the flag (NP-20), and accepts the claim when it is made.
fn probe_declare_closed(fx: &Fx, id: &str) -> Rec {
    let e = &fx.model()[0];
    let mut acc = Acc::default();
    let degree = e["degree"].as_u64().unwrap() as usize;
    let cps: Vec<Point2> = e["control_points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| xy(&hv(v)))
        .collect();
    let s = BSpline::new(degree, cps, hv(&e["knots"])).expect("spline");
    acc.note("undeclared_is_declared_closed", s.is_declared_closed());
    let declared = s.declare_closed();
    acc.note(
        "declare_closed",
        if declared.is_ok() { "ok" } else { "refused" },
    );
    let mut r = finish(
        fx,
        "probe",
        "SPLINE",
        "b_spline_curve_with_knots".into(),
        acc,
        true,
        None,
    );
    r.id = id.to_string();
    r
}

/// Probe: the single-span cubic as PVec's `bezier_curve`, sampled against the same data as a
/// b-spline. Self-consistency of PVec's evaluators; the truth gives control points only.
fn probe_bezier(fx: &Fx, id: &str) -> Rec {
    let e = &fx.model()[0];
    let t = fx.truth_ent(0);
    let mut acc = Acc::default();
    let cps: Vec<Point2> = e["control_points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| xy(&hv(v)))
        .collect();
    let bz = Bezier::new(cps.clone()).expect("bezier");
    let bs = BSpline::new(e["degree"].as_u64().unwrap() as usize, cps, hv(&e["knots"]))
        .expect("bspline");
    let (cb, cs) = (Curve::Bezier(bz), Curve::BSpline(bs));
    for k in 0..=16 {
        let u = k as f64 / 16.0;
        let (a, b) = (cb.eval(u).unwrap(), cs.eval(u).unwrap());
        acc.near_pt(&format!("sample {k}/16"), a, b, 1.0);
    }
    let same = domain_cmp(&mut acc, &t["parameter_domain"], cb.domain());
    acc.note(
        "purpose",
        "bezier_curve against the b-spline of the same data, PVec evaluator against itself",
    );
    let mut r = finish(
        fx,
        "probe",
        "SPLINE (single span)",
        kind_name(&cb),
        acc,
        same,
        None,
    );
    r.id = id.to_string();
    r
}

/// Probe: a two-vertex polyline as the carrier of a LINE. It keeps both end points and the [0, 1]
/// domain, and is a polyline, not a line.
fn probe_line_as_polyline(fx: &Fx, id: &str) -> Rec {
    let e = &fx.model()[0];
    let t = fx.truth_ent(0);
    let mut acc = Acc::default();
    let (a, b) = (xy(&hv(&e["start"])), xy(&hv(&e["end"])));
    let pl = Polyline::new(vec![a, b]).expect("polyline");
    let (s, en) = (tv(&t["geometry"]["start"]), tv(&t["geometry"]["end"]));
    acc.bits("start.x", s[0], pl.vertices()[0].x);
    acc.bits("end.y", en[1], pl.vertices()[1].y);
    let c = Curve::Polyline(pl);
    let same = domain_cmp(&mut acc, &t["parameter_domain"], c.domain());
    acc.note(
        "purpose",
        "end points stored verbatim, domain preserved, class is polyline",
    );
    let mut r = finish(fx, "probe", "LINE", kind_name(&c), acc, same, None);
    r.id = id.to_string();
    r
}

// ------------------------------------------------------------------------------------ inserts

struct Insert {
    block: String,
    at: Vec<f64>,
    rotation: f64,
    scale: Vec<f64>,
}

/// The chain of INSERTs from model space down to the entity, outermost first, and the leaf entity.
fn chain(fx: &Fx) -> (Vec<Insert>, Value) {
    let mut out = Vec::new();
    let mut ents = fx.model().clone();
    loop {
        let e = ents[0].clone();
        if e["type"] != "INSERT" {
            return (out, e);
        }
        let blk = e["block"].as_str().unwrap().to_string();
        out.push(Insert {
            block: blk.clone(),
            at: hv(&e["insert_point"]),
            rotation: hx(&e["rotation"]),
            scale: hv(&e["scale"]),
        });
        ents = fx.block(&blk).clone();
    }
}

/// An INSERT as a PVec map: scale, then rotate, then translate. Only PVec constructors and `then`.
fn insert_map(i: &Insert) -> Res<Transform2> {
    Transform2::scale(i.scale[0], i.scale[1])?
        .then(&Transform2::rotate(i.rotation)?)?
        .then(&Transform2::translate(i.at[0], i.at[1])?)
}

fn row_inserts(fx: &Fx) -> Rec {
    let mut acc = Acc::default();
    let t = fx.truth_ent(0);
    let (ins, leaf) = chain(fx);
    let leaf_type = leaf["type"].as_str().unwrap().to_string();
    let dwg = format!("INSERT x{} over {leaf_type}", ins.len());
    acc.note("insert_levels", ins.len());
    acc.note(
        "blocks",
        ins.iter()
            .map(|i| i.block.clone())
            .collect::<Vec<_>>()
            .join(">"),
    );
    let leaf_curve = match leaf_type.as_str() {
        "LINE" => map_line(&leaf),
        "CIRCLE" => map_circle(&leaf),
        "ARC" => map_arc(&leaf),
        o => panic!("{}: leaf {o}", fx.id),
    };
    let leaf_curve = match leaf_curve {
        Ok(c) => c,
        Err(er) => return refused(fx, &dwg, "leaf", &er, "leaf entity", acc),
    };
    let dropped_z: f64 = ins.iter().map(|i| i.at[2].abs()).sum();
    if dropped_z != 0.0 {
        acc.note("dropped_insert_z", fmt_p(dropped_z));
    }
    // Compose innermost first: the child is applied before its parent.
    let mut total = Ok(Transform2::IDENTITY);
    for i in ins.iter().rev() {
        total = total.and_then(|acc_m| insert_map(i).and_then(|m| acc_m.then(&m)));
    }
    let total = match total {
        Ok(m) => m,
        Err(er) => {
            // The corpus expects a rejection: near-singular transform, quarantine, no curve.
            let expects = t.get("rejection").is_some();
            acc.truth("a rejection is what the truth expects", expects);
            acc.note(
                "expected_rejection",
                t["rejection"]["classification"].as_str().unwrap_or("none"),
            );
            acc.note("pvec_refusal", refusal(&er));
            acc.note("pvec_refusal_rule", er.rule());
            let det: f64 = ins.iter().map(|i| i.scale[0] * i.scale[1]).product();
            acc.note("source_determinant", format!("{det:e}"));
            return finish(
                fx,
                "row",
                &dwg,
                "refused with no curve emitted".into(),
                acc,
                true,
                None,
            )
            .into_refused(expects);
        }
    };
    // The composed matrix against the truth record's own matrix (2x2 and translation of the 4x4).
    let tm: Vec<f64> = match t.get("transform_chain") {
        Some(c) => tv(&c["composed"]["matrix"]),
        None => tv(&t["transform"]["matrix"]),
    };
    let m = total.matrix();
    let pick = [0usize, 1, 4, 5, 3, 7];
    let names = ["a", "b", "c", "d", "tx", "ty"];
    for k in 0..6 {
        acc.near(&format!("matrix {}", names[k]), tm[pick[k]], m[k], 1.0);
    }
    acc.note("determinant", format!("{:e}", total.determinant()));
    let (world, pm) = match leaf_curve.transform(&total) {
        Ok(w) => w,
        Err(er) => return refused(fx, &dwg, "transform", &er, "transform of the entity", acc),
    };
    acc.note(
        "param_map",
        format!("offset {} scale {}", fmt_p(pm.offset), fmt_p(pm.scale)),
    );
    let w = &t["world"];
    let wg = &w["geometry"];
    let wclass = w["class"].as_str().unwrap();
    let domain_same;
    match wclass {
        "line" => {
            let (s, e) = (tv(&wg["start"]), tv(&wg["end"]));
            line_truth_checks(&mut acc, &world, xy(&s), xy(&e));
            if s[2] != 0.0 {
                acc.note("dropped_world_z", fmt_p(s[2]));
            }
            domain_same = domain_cmp(&mut acc, &w["parameter_domain"], world.domain());
        }
        "circle" => {
            if let Curve::Circle(ci) = &world {
                let c = tv(&wg["center"]);
                let r = th(&wg["radius"]);
                acc.near_pt("centre", xy(&c), ci.placement().origin(), r);
                acc.near("radius", r, ci.radius(), r);
            } else {
                acc.truth("world is a circle", false);
            }
            domain_same = domain_cmp(&mut acc, &w["parameter_domain"], world.domain());
        }
        "ellipse" => {
            if let Curve::Ellipse(el) = &world {
                let c = tv(&wg["center"]);
                let (sm, sn) = (th(&wg["semi_major"]), th(&wg["semi_minor"]));
                let tm = tv(&wg["major_axis"]);
                let (a, b) = (el.semi_axis_1(), el.semi_axis_2());
                let (maj, mn, dir) = if a >= b {
                    (a, b, el.placement().x_axis())
                } else {
                    (b, a, el.placement().y_axis())
                };
                acc.near_pt("centre", xy(&c), el.placement().origin(), sm);
                acc.near("semi major", sm, maj, sm);
                acc.near("semi minor", sn, mn, sm);
                // Truth side: the major direction up to sign.
                let sgn = if dir.x * tm[0] + dir.y * tm[1] >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                acc.near("major dir x", tm[0], sgn * dir.x, 1.0);
                acc.near("major dir y", tm[1], sgn * dir.y, 1.0);
                acc.note("pvec_semi_axes", format!("({},{})", fmt_p(a), fmt_p(b)));
                acc.note("circle_became_ellipse", true);
            } else {
                acc.truth("world is an ellipse", false);
            }
            domain_same = domain_cmp(&mut acc, &w["parameter_domain"], world.domain());
        }
        "circular_arc" => {
            let r = th(&wg["radius"]);
            let (s, sw) = (th(&wg["start_angle"]), th(&wg["sweep_angle"]));
            locus(&mut acc, &world, wg, r, r, s, sw, r);
            acc.note(
                "expected_sense",
                t["orientation"]["sense_about_normal"]
                    .as_str()
                    .unwrap_or("-"),
            );
            acc.note("pvec_sense_before", sense_word(&leaf_curve));
            acc.note("pvec_sense_after", sense_word(&world));
            acc.note(
                "det_sign",
                if total.determinant() < 0.0 {
                    "negative"
                } else {
                    "positive"
                },
            );
            domain_same = domain_cmp(&mut acc, &w["parameter_domain"], world.domain());
        }
        o => panic!("{}: world class {o}", fx.id),
    }
    acc.note("world_kind", kind_name(&world));
    bounds_note(&mut acc, &world);
    finish(fx, "row", &dwg, kind_name(&world), acc, domain_same, None)
}

impl Rec {
    /// The truth for this fixture is a rejection, so a refusal is the expected outcome, not a failure.
    fn into_refused(mut self, expected: bool) -> Rec {
        if expected && self.acc.fails.is_empty() {
            self.result = "refused".into();
            self.reason = "TRANSFORM_SINGULAR".into();
        }
        self
    }
}

// ------------------------------------------------------------------------------------ layers

fn row_layers(fx: &Fx) -> Rec {
    let mut acc = Acc::default();
    let layers = fx.sharp["layers"].as_array().unwrap();
    let mut all_same = true;
    let mut mapped = 0;
    for (i, e) in fx.model().iter().enumerate() {
        let t = fx.truth_ent(i);
        let tl = &t["layer"];
        let name = e["layer"].as_str().unwrap();
        let st = layers.iter().find(|l| l["name"] == name);
        acc.truth(
            &format!("entity {i} layer name"),
            name == tl["name"].as_str().unwrap(),
        );
        acc.truth(
            &format!("entity {i} layer state"),
            st.map(|l| l["on"] == tl["on"] && l["frozen"] == tl["frozen"])
                .unwrap_or(false),
        );
        let (start, end) = (tv(&t["geometry"]["start"]), tv(&t["geometry"]["end"]));
        match map_line(e) {
            Ok(c) => {
                mapped += 1;
                line_truth_checks(&mut acc, &c, xy(&start), xy(&end));
                all_same &= domain_cmp(&mut Acc::default(), &t["parameter_domain"], c.domain());
            }
            Err(er) => acc.note("refused", refusal(&er)),
        }
        acc.note(
            &format!("entity{i}"),
            format!("{} on={} frozen={}", name, tl["on"], tl["frozen"]),
        );
    }
    acc.note("entities_mapped", mapped);
    acc.note("entities_encountered", fx.model().len());
    acc.note(
        "pvec_layer_stream_carries",
        "names only, no on or frozen flag",
    );
    acc.note("decoder_layer_table_carries", "name, on, frozen");
    finish(
        fx,
        "row",
        "LINE x3 on three layers",
        "trimmed_curve over line".into(),
        acc,
        all_same,
        None,
    )
}

/// Probe, synthetic: there is no truth record for it. A closed polyline whose last vertex already
/// repeats the first (common in drawings) becomes a ring with a zero-length last segment once the
/// closing vertex is appended. It is here to see what PVec says, not to be compared with anything.
fn probe_closed_with_repeated_vertex(fx: &Fx, id: &str) -> Rec {
    let e = &fx.model()[0];
    let mut acc = Acc::default();
    let (mut pts, _) = vertices(e);
    let first = pts[0];
    pts.push(first);
    pts.push(first);
    let (result, reason) = match Polyline::new(pts) {
        Ok(_) => ("accepted".to_string(), String::new()),
        Err(er) => {
            acc.note("pvec_refusal", refusal(&er));
            acc.note("pvec_refusal_rule", er.rule());
            ("refused".to_string(), "DEGENERATE_SEGMENT".to_string())
        }
    };
    acc.note("purpose", "synthetic input, no expected.math.json: first vertex repeated by the source and again by the closing rule");
    Rec {
        id: id.into(),
        kind: "probe",
        fixture: fx.id.clone(),
        dwg_class: "LWPOLYLINE (closed, repeated vertex)".into(),
        target: "polyline".into(),
        result,
        reason,
        acc,
    }
}

// ------------------------------------------------------------------------------------ driver

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .expect("usage: harness <corpus r0-open dir>")
        .into();
    println!(
        "id\tkind\tfixture\tdwg_class\ttarget\tresult\treason\tchecks\tworst_dev\tratio\tnotes"
    );
    let mut recs: Vec<Rec> = Vec::new();
    let mut ids: Vec<String> = fs::read_dir(dir.join("fixtures"))
        .expect("fixtures dir")
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    ids.sort();
    for id in &ids {
        let fx = load(&dir, id);
        let class = fx.truth_ent(0)["class"].as_str().unwrap().to_string();
        let structure = fx.truth_ent(0)["structure"].as_str().unwrap().to_string();
        let rec = if id == "layer-state-001" {
            row_layers(&fx)
        } else if structure == "INSTANCE_PLUS_TRANSFORM" {
            row_inserts(&fx)
        } else {
            match class.as_str() {
                "line" => row_line(&fx),
                "polyline" => {
                    let (_, b) = vertices(&fx.model()[0]);
                    if b.iter().any(|x| *x != 0.0) {
                        row_bulge(&fx)
                    } else {
                        row_polyline(&fx)
                    }
                }
                "circle" => row_circle(&fx),
                "circular_arc" => row_arc(&fx),
                "ellipse" | "elliptical_arc" => row_ellipse(&fx),
                "bspline" | "nurbs" | "bezier" => row_spline(&fx),
                o => panic!("{id}: unhandled class {o}"),
            }
        };
        recs.push(rec);
    }
    // Alternatives, measured so the choice between them is not argued.
    let p = |id: &str| load(&dir, id);
    recs.push(probe_line_as_polyline(&p("line-001"), "P1"));
    recs.push(probe_ellipse_from_truth(&p("ellipse-001"), "P2"));
    recs.push(probe_ellipse_from_truth(&p("elliptical-arc-001"), "P3"));
    recs.push(probe_bezier(&p("bezier-d3-001"), "P4"));
    recs.push(probe_declare_closed(&p("closed-by-endpoint-001"), "P5"));
    recs.push(probe_closed_with_repeated_vertex(
        &p("lwpolyline-closed-001"),
        "P6",
    ));
    for r in &recs {
        println!("{}", r.line());
    }
}
