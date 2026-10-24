//! N1.6 measurement harness: real PDFs through a content-stream path mapper, then through PVec.
//!
//! What this is not: an evaluator, a flattener or a tolerance. `lopdf` is the object and stream
//! reader, the mapper below only turns path operators into segments (a line keeps its two end points,
//! a cubic keeps its four control points), and every acceptance or comparison goes through
//! `pvec_geometry` constructors or `pvec_numeric::Policy::V1`. Prints a long-format TSV on stdout:
//! `fixture <TAB> key <TAB> value`.
//!
//! Usage: `harness <dir-with-pdfs>`; synthetic content-stream checks run first and are reported
//! under the fixture name `synthetic`.

use lopdf::content::Content;
use lopdf::{Dictionary, Document, Object};
use pvec_geometry::{Bezier, Point2, Polyline, Transform2};
use pvec_numeric::Policy;
use std::collections::BTreeMap;

type P = [f64; 2];

#[derive(Clone, Debug)]
enum Seg {
    Line(P),
    Cubic(P, P, P),
}

#[derive(Clone, Debug)]
struct Sub {
    start: P,
    pstart: P,
    segs: Vec<Seg>,
    closed: bool,
    rect: bool,
}

#[derive(Default)]
struct Stats(BTreeMap<String, u64>);

impl Stats {
    fn inc(&mut self, k: &str) {
        *self.0.entry(k.to_string()).or_insert(0) += 1;
    }
    fn add(&mut self, k: &str, n: u64) {
        *self.0.entry(k.to_string()).or_insert(0) += n;
    }
    fn max(&mut self, k: &str, n: u64) {
        let e = self.0.entry(k.to_string()).or_insert(0);
        *e = (*e).max(n);
    }
    fn get(&self, k: &str) -> u64 {
        self.0.get(k).copied().unwrap_or(0)
    }
}

const CM_CLASSES: &[&str] = &[
    "identity",
    "translation",
    "rotation",
    "uniform scale",
    "non-uniform scale",
    "reflection",
    "shear / general affine",
    "singular",
];

const OPS: &[&str] = &[
    "m", "l", "c", "v", "y", "re", "h", "cm", "q", "Q", "W", "W*", "S", "s", "f", "F", "f*", "B", "B*", "b", "b*",
    "n", "Do", "BDC", "BMC", "EMC",
];

/// PDF `[a b c d e f]` is `x' = a x + c y + e`, `y' = b x + d y + f`; PVec's `Transform2` takes
/// `[a b tx; c d ty]`, so the PDF order maps to `[a, c, b, d, e, f]`.
fn to_pvec(m: [f64; 6]) -> [f64; 6] {
    [m[0], m[2], m[1], m[3], m[4], m[5]]
}

/// Counting triage for `cm` into the seven N1.5 classes. The decision stays PVec's (N3.4 calls the
/// transform and checks the kind it returns); this only tallies, and every comparison is `Policy::V1`.
fn classify(m: [f64; 6]) -> &'static str {
    let t = match Transform2::new(to_pvec(m)) {
        Ok(t) => t,
        Err(_) => return "singular",
    };
    let p = &Policy::V1;
    let (a, b, c, d) = (m[0], m[1], m[2], m[3]);
    let lin_id = p.near(a, 1.0, 1.0) && p.near(d, 1.0, 1.0) && p.near(b, 0.0, 1.0) && p.near(c, 0.0, 1.0);
    if lin_id {
        return if p.near(m[4], 0.0, 1.0) && p.near(m[5], 0.0, 1.0) { "identity" } else { "translation" };
    }
    if t.determinant() < 0.0 {
        return "reflection";
    }
    if p.near(a, d, 1.0) && p.near(b, -c, 1.0) {
        return if p.near(a * a + b * b, 1.0, 1.0) { "rotation" } else { "uniform scale" };
    }
    if p.near(b, 0.0, 1.0) && p.near(c, 0.0, 1.0) {
        return "non-uniform scale";
    }
    "shear / general affine"
}

fn axis_aligned(t: &Transform2) -> bool {
    let m = t.matrix(); // PVec order: [a b tx; c d ty]
    Policy::V1.near(m[1], 0.0, 1.0) && Policy::V1.near(m[2], 0.0, 1.0)
}

struct Interp<'a> {
    doc: Option<&'a Document>,
    st: Stats,
    ctm: Option<Transform2>,
    stack: Vec<Option<Transform2>>,
    subs: Vec<Sub>,
    cur: P,
    pending_clip: Option<bool>,
    log: Option<Vec<Vec<Sub>>>,
    seen_forms: Vec<u32>,
    unique_forms: Vec<u32>,
    depth: usize,
}

fn num(o: &Object) -> Option<f64> {
    match o {
        Object::Integer(i) => Some(*i as f64),
        Object::Real(r) => Some(*r as f64),
        _ => None,
    }
}

impl<'a> Interp<'a> {
    fn new(doc: Option<&'a Document>, keep_log: bool) -> Self {
        Interp {
            doc,
            st: Stats::default(),
            ctm: Some(Transform2::IDENTITY),
            stack: Vec::new(),
            subs: Vec::new(),
            cur: [0.0, 0.0],
            pending_clip: None,
            log: if keep_log { Some(Vec::new()) } else { None },
            seen_forms: Vec::new(),
            unique_forms: Vec::new(),
            depth: 0,
        }
    }

    fn page(&self, p: P) -> P {
        match &self.ctm {
            Some(t) => {
                let q = t.apply_point(Point2::new(p[0], p[1]));
                [q.x, q.y]
            }
            None => p,
        }
    }

    /// One segment through PVec, at the CTM in force when it was built.
    fn check_line(&mut self, a: P, b: P) {
        let (a, b) = (self.page(a), self.page(b));
        match Polyline::new(vec![Point2::new(a[0], a[1]), Point2::new(b[0], b[1])]) {
            Ok(_) => self.st.inc("pvec.ok"),
            Err(e) => {
                self.st.inc("pvec.refused");
                self.st.inc(&format!("pvec.refused.line.{}", e.rule()));
            }
        }
    }

    fn check_cubic(&mut self, p0: P, c1: P, c2: P, p3: P) {
        let pts: Vec<Point2> = [p0, c1, c2, p3]
            .iter()
            .map(|p| {
                let q = self.page(*p);
                Point2::new(q[0], q[1])
            })
            .collect();
        match Bezier::new(pts) {
            Ok(_) => self.st.inc("pvec.ok"),
            Err(e) => {
                self.st.inc("pvec.refused");
                self.st.inc(&format!("pvec.refused.cubic.{}", e.rule()));
            }
        }
    }

    fn line_to(&mut self, p: P) {
        let from = self.cur;
        self.st.inc("segs.line");
        self.check_line(from, p);
        if let Some(s) = self.subs.last_mut() {
            s.segs.push(Seg::Line(p));
        }
        self.cur = p;
    }

    fn cubic_to(&mut self, c1: P, c2: P, p: P) {
        let from = self.cur;
        self.st.inc("segs.cubic");
        self.check_cubic(from, c1, c2, p);
        if let Some(s) = self.subs.last_mut() {
            s.segs.push(Seg::Cubic(c1, c2, p));
        }
        self.cur = p;
    }

    /// `h` is a flag on the subpath. The straight edge back to the start is real geometry, so it is
    /// counted on its own (`segs.close`) and goes through PVec like any other line.
    fn close(&mut self) {
        let Some(s) = self.subs.last() else { return };
        if s.closed {
            return;
        }
        let (start, cur) = (s.start, self.cur);
        if !near_p(start, cur) {
            self.st.inc("segs.close");
            self.check_line(cur, start);
        }
        if let Some(s) = self.subs.last_mut() {
            s.closed = true;
        }
        self.cur = start;
    }

    fn signed_area(&self, s: &Sub) -> f64 {
        let mut pts = vec![self.page(s.start)];
        for g in &s.segs {
            match g {
                Seg::Line(p) | Seg::Cubic(_, _, p) => pts.push(self.page(*p)),
            }
        }
        let mut a = 0.0;
        for i in 0..pts.len() {
            let (p, q) = (pts[i], pts[(i + 1) % pts.len()]);
            a += p[0] * q[1] - q[0] * p[1];
        }
        a / 2.0
    }

    fn paint(&mut self, op: &str) {
        if matches!(op, "s" | "b" | "b*") {
            self.close();
        }
        let subs = std::mem::take(&mut self.subs);
        if !subs.is_empty() {
            self.st.inc("paths");
            self.st.add("subpaths", subs.len() as u64);
            self.st.add("subpaths.closed", subs.iter().filter(|s| s.closed).count() as u64);
            if subs.len() > 1 {
                self.st.inc("paths.compound");
                let signs: Vec<bool> = subs.iter().map(|s| self.signed_area(s) > 0.0).collect();
                if signs.iter().any(|s| *s != signs[0]) {
                    self.st.inc("compound.mixed_winding");
                }
            }
        }
        match op {
            "f" | "F" | "B" | "b" => self.st.inc("fill.nonzero"),
            "f*" | "B*" | "b*" => self.st.inc("fill.evenodd"),
            _ => {}
        }
        if matches!(op, "S" | "s" | "B" | "B*" | "b" | "b*") {
            self.st.inc("stroke");
        }
        if let Some(_evenodd) = self.pending_clip.take() {
            // Exact: a single axis-aligned rectangle (a `re`, closed, four edges) under an axis-aligned
            // CTM is a rectangle in page space with no approximation. Anything else is bounded-only.
            let exact = subs.len() == 1
                && subs[0].rect
                && self.ctm.as_ref().map(axis_aligned).unwrap_or(false);
            self.st.inc(if exact { "clip.exact" } else { "clip.bounded" });
        }
        if let Some(log) = &mut self.log {
            log.push(subs);
        }
    }

    fn run(&mut self, content: &Content, res: Option<&Dictionary>) {
        for op in &content.operations {
            let o = op.operator.as_str();
            if OPS.contains(&o) {
                self.st.inc(&format!("ops.{o}"));
            }
            let a: Vec<f64> = op.operands.iter().filter_map(num).collect();
            match o {
                "m" if a.len() == 2 => {
                    let ps = self.page([a[0], a[1]]);
                    self.subs.push(Sub { start: [a[0], a[1]], pstart: ps, segs: vec![], closed: false, rect: false });
                    self.cur = [a[0], a[1]];
                }
                "l" if a.len() == 2 => self.line_to([a[0], a[1]]),
                "c" if a.len() == 6 => self.cubic_to([a[0], a[1]], [a[2], a[3]], [a[4], a[5]]),
                // `v`: first control point is the current point. `y`: second control point is the end point.
                "v" if a.len() == 4 => {
                    let c = self.cur;
                    self.cubic_to(c, [a[0], a[1]], [a[2], a[3]])
                }
                "y" if a.len() == 4 => self.cubic_to([a[0], a[1]], [a[2], a[3]], [a[2], a[3]]),
                "re" if a.len() == 4 => {
                    let (x, y, w, h) = (a[0], a[1], a[2], a[3]);
                    self.st.inc("segs.rect");
                    let ps = self.page([x, y]);
                    self.subs.push(Sub { start: [x, y], pstart: ps, segs: vec![], closed: false, rect: true });
                    self.cur = [x, y];
                    self.line_to([x + w, y]);
                    self.line_to([x + w, y + h]);
                    self.line_to([x, y + h]);
                    self.close();
                }
                "h" => self.close(),
                "W" => self.pending_clip = Some(false),
                "W*" => self.pending_clip = Some(true),
                "S" | "s" | "f" | "F" | "f*" | "B" | "B*" | "b" | "b*" | "n" => self.paint(o),
                "q" => {
                    self.stack.push(self.ctm);
                    self.st.max("q.max_depth", self.stack.len() as u64);
                }
                "Q" => match self.stack.pop() {
                    Some(c) => self.ctm = c,
                    None => self.st.inc("q.unbalanced"),
                },
                "cm" if a.len() == 6 => {
                    self.st.inc(&format!("cm.{}", classify([a[0], a[1], a[2], a[3], a[4], a[5]])));
                    self.fold(&a);
                }
                "Do" => self.do_xobject(op.operands.first(), res),
                "BDC" => {
                    if let (Some(Object::Name(tag)), Some(p)) = (op.operands.first(), op.operands.get(1)) {
                        if tag == b"OC" {
                            self.st.inc("bdc.oc");
                        }
                        let _ = p;
                    }
                }
                _ => {}
            }
        }
        if !self.stack.is_empty() {
            self.st.inc("q.unclosed_at_end");
        }
    }

    /// Fold a `cm` into the CTM through PVec's `then`. A refusal (singular, out of range) leaves the
    /// CTM unusable until the matching `Q`, and is counted.
    fn fold(&mut self, a: &[f64]) {
        let m = Transform2::new(to_pvec([a[0], a[1], a[2], a[3], a[4], a[5]]));
        self.ctm = match (m, &self.ctm) {
            (Ok(m), Some(c)) => match m.then(c) {
                Ok(t) => Some(t),
                Err(_) => {
                    self.st.inc("ctm.refused");
                    None
                }
            },
            _ => {
                self.st.inc("ctm.refused");
                None
            }
        };
    }

    fn do_xobject(&mut self, name: Option<&Object>, res: Option<&Dictionary>) {
        let (Some(doc), Some(Object::Name(n)), Some(res)) = (self.doc, name, res) else { return };
        let Some(xo) = res.get(b"XObject").ok().and_then(|o| doc.dereference(o).ok()).and_then(|(_, o)| o.as_dict().ok()) else {
            return;
        };
        let Some(obj) = xo.get(n).ok() else { return };
        let id = match obj {
            Object::Reference(id) => Some(id.0),
            _ => None,
        };
        let Some(stream) = doc.dereference(obj).ok().and_then(|(_, o)| o.as_stream().ok()) else { return };
        let sub = stream.dict.get(b"Subtype").ok().and_then(|o| o.as_name().ok()).unwrap_or(b"");
        if sub != b"Form" {
            self.st.inc("xobject.image_do");
            return;
        }
        self.st.inc("forms.do");
        if let Some(i) = id {
            if !self.unique_forms.contains(&i) {
                self.unique_forms.push(i);
            }
            if self.seen_forms.contains(&i) || self.depth >= 8 {
                self.st.inc("forms.skipped_cycle_or_depth");
                return;
            }
            self.seen_forms.push(i);
        }
        let bytes = stream.decompressed_content().unwrap_or_else(|_| stream.content.clone());
        let Ok(content) = Content::decode(&bytes) else { return };
        // The form's own resources win; otherwise it inherits the caller's.
        let own = stream.dict.get(b"Resources").ok().and_then(|o| doc.dereference(o).ok()).and_then(|(_, o)| o.as_dict().ok());
        self.stack.push(self.ctm);
        if let Ok(arr) = stream.dict.get(b"Matrix").and_then(|o| o.as_array()) {
            let v: Vec<f64> = arr.iter().filter_map(num).collect();
            if v.len() == 6 {
                self.fold(&v);
            }
        }
        let saved_subs = std::mem::take(&mut self.subs);
        self.depth += 1;
        self.run(&content, own.or(Some(res)));
        self.depth -= 1;
        self.subs = saved_subs;
        self.ctm = self.stack.pop().unwrap_or(None);
        if let Some(i) = id {
            self.seen_forms.retain(|x| *x != i);
        }
    }
}

// ---------------------------------------------------------------------------------- synthetic

fn run_bytes(src: &str) -> Interp<'static> {
    let content = Content::decode(src.as_bytes()).expect("synthetic stream parses");
    let mut it = Interp::new(None, true);
    it.run(&content, None);
    it
}

fn near2(a: P, b: P) -> bool {
    Policy::V1.near_point(a, b, 1.0)
}

fn near_p(a: P, b: P) -> bool {
    Policy::V1.near_point(a, b, 1.0)
}

fn synthetic(out: &mut Vec<(String, String, String)>) {
    let mut put = |id: &str, ok: bool, detail: String| {
        out.push(("synthetic".into(), id.into(), format!("{}: {detail}", if ok { "pass" } else { "FAIL" })));
    };

    // S1 m, l
    let it = run_bytes("0 0 m 10 0 l 10 10 l S");
    let s = &it.log.as_ref().unwrap()[0][0];
    put("S1", s.segs.len() == 2 && !s.closed && matches!(s.segs[0], Seg::Line(_)), "m l l is one open subpath of two line segments".into());

    // S2 c: the four control points survive untouched, one segment, never subdivided.
    let it = run_bytes("0 0 m 1 2 3 4 5 0 c S");
    let s = &it.log.as_ref().unwrap()[0][0];
    let ok = s.segs.len() == 1
        && matches!(&s.segs[0], Seg::Cubic(c1, c2, p) if near2(*c1, [1.0, 2.0]) && near2(*c2, [3.0, 4.0]) && near2(*p, [5.0, 0.0]))
        && it.st.get("segs.cubic") == 1;
    put("S2", ok, "c keeps its two control points and end point as one cubic segment".into());

    // S3 v
    let it = run_bytes("0 0 m 3 4 5 0 v S");
    let s = &it.log.as_ref().unwrap()[0][0];
    let ok = matches!(&s.segs[0], Seg::Cubic(c1, c2, p) if near2(*c1, [0.0, 0.0]) && near2(*c2, [3.0, 4.0]) && near2(*p, [5.0, 0.0]));
    put("S3", ok, "v makes the first control point the current point, still one exact cubic".into());

    // S4 y
    let it = run_bytes("0 0 m 1 2 5 0 y S");
    let s = &it.log.as_ref().unwrap()[0][0];
    let ok = matches!(&s.segs[0], Seg::Cubic(c1, c2, p) if near2(*c1, [1.0, 2.0]) && near2(*c2, [5.0, 0.0]) && near2(*p, [5.0, 0.0]));
    put("S4", ok, "y makes the second control point the end point, still one exact cubic".into());

    // S5 re
    let it = run_bytes("1 2 10 20 re f");
    let s = &it.log.as_ref().unwrap()[0][0];
    let pts: Vec<P> = s.segs.iter().map(|g| match g { Seg::Line(p) | Seg::Cubic(_, _, p) => *p }).collect();
    let ok = s.closed && s.rect && pts.len() == 3 && near2(pts[0], [11.0, 2.0]) && near2(pts[1], [11.0, 22.0]) && near2(pts[2], [1.0, 22.0]) && it.st.get("segs.close") == 1;
    put("S5", ok, "re is one closed subpath: three line segments plus the closing edge, flagged as a rectangle".into());

    // S6 h: closes without adding a segment of its own
    let it = run_bytes("0 0 m 4 0 l 4 4 l h S");
    let s = &it.log.as_ref().unwrap()[0][0];
    put("S6", s.closed && s.segs.len() == 2 && it.st.get("segs.close") == 1, "h marks the subpath closed; the closing edge is counted once as segs.close".into());

    // S7 compound path, opposite winding, even-odd fill
    let it = run_bytes("0 0 m 10 0 l 10 10 l 0 10 l h 2 2 m 2 8 l 8 8 l 8 2 l h f*");
    let ok = it.log.as_ref().unwrap()[0].len() == 2 && it.st.get("paths.compound") == 1 && it.st.get("compound.mixed_winding") == 1 && it.st.get("fill.evenodd") == 1;
    put("S7", ok, "two subpaths in one path, opposite winding detected, f* counted as even-odd".into());

    // S8 q / cm / Q folded through PVec's `then`
    let it = run_bytes("q 2 0 0 2 10 20 cm 1 0 0 1 3 0 cm 0 0 m 1 0 l S Q 0 0 m 1 0 l S");
    let log = it.log.as_ref().unwrap();
    let inner = log[0][0].pstart;
    let outer = log[1][0].pstart;
    let ok = near2(inner, [16.0, 20.0]) && near2(outer, [0.0, 0.0]) && it.st.get("q.max_depth") == 1 && it.st.get("q.unbalanced") == 0;
    put("S8", ok, format!("two cm under one q compose to (0,0) -> ({:.1},{:.1}); Q restores the identity", inner[0], inner[1]));

    // S9 clipping: exact rectangle, rotated rectangle, non-rectangular W*
    let it = run_bytes("0 0 100 100 re W n q 0.7 0.7 -0.7 0.7 0 0 cm 0 0 100 100 re W n Q 0 0 m 10 0 l 5 5 l h W* n");
    let ok = it.st.get("clip.exact") == 1 && it.st.get("clip.bounded") == 2 && it.st.get("ops.W*") == 1;
    put("S9", ok, "axis-aligned rectangle clip is exact; rotated rectangle and a triangle clip are bounded-only".into());

    // S10 degenerate segment is PVec's refusal, not ours
    let it = run_bytes("0 0 m 0 0 l S");
    let ok = it.st.get("pvec.refused") == 1;
    let rule: Vec<String> = it.st.0.keys().filter(|k| k.starts_with("pvec.refused.line.")).cloned().collect();
    put("S10", ok && !rule.is_empty(), format!("a zero-length l is refused by PVec ({})", rule.join(",").replace("pvec.refused.line.", "")));

    // S11 singular cm is classified, and the CTM becomes unusable
    let it = run_bytes("1 0 0 0 0 0 cm 0 0 m 1 1 l S");
    put("S11", it.st.get("cm.singular") == 1 && it.st.get("ctm.refused") == 1, "a singular cm is refused by PVec's NP-19 and counted, never silently applied".into());

    // S12 what lopdf hands over for an operand: single precision
    let c = Content::decode(b"0.1 0 m").expect("parses");
    let v = num(&c.operations[0].operands[0]).unwrap_or(0.0);
    let widened = v != 0.1_f64;
    put("S12", widened, format!("lopdf reads the operand 0.1 as {v:.17}: operands arrive as f32 and widen, so 'exact' means exact relative to what lopdf delivers"));
}

// ---------------------------------------------------------------------------------- real files

fn s(o: &Object) -> String {
    match o {
        Object::String(b, _) => String::from_utf8_lossy(b).into_owned(),
        Object::Name(b) => String::from_utf8_lossy(b).into_owned(),
        _ => String::new(),
    }
}

fn info(doc: &Document, key: &[u8]) -> String {
    let Some(inf) = doc.trailer.get(b"Info").ok().and_then(|o| doc.dereference(o).ok()).and_then(|(_, o)| o.as_dict().ok()) else {
        return "-".into();
    };
    inf.get(key).ok().map(s).filter(|v| !v.is_empty()).unwrap_or_else(|| "-".into()).replace('\t', " ")
}

fn measure(name: &str, path: &std::path::Path, rows: &mut Vec<(String, String, String)>) {
    let doc = Document::load(path).expect("pdf loads");
    let mut it = Interp::new(Some(&doc), false);
    let pages = doc.get_pages();
    let mut user_unit = 0u64;
    let mut vp = 0u64;
    for (_, pid) in &pages {
        let page = doc.get_dictionary(*pid).expect("page dict");
        if page.has(b"UserUnit") {
            user_unit += 1;
        }
        if page.has(b"VP") {
            vp += 1;
        }
        let bytes = doc.get_page_content(*pid).unwrap_or_default();
        let Ok(content) = Content::decode(&bytes) else {
            it.st.inc("pages.undecodable");
            continue;
        };
        it.ctm = Some(Transform2::IDENTITY);
        it.stack.clear();
        let (res, _) = doc.get_page_resources(*pid).unwrap_or((None, vec![]));
        it.run(&content, res);
    }
    let ocgs = doc
        .catalog()
        .ok()
        .and_then(|c| c.get(b"OCProperties").ok())
        .and_then(|o| doc.dereference(o).ok())
        .and_then(|(_, o)| o.as_dict().ok())
        .and_then(|d| d.get(b"OCGs").ok())
        .and_then(|o| doc.dereference(o).ok())
        .and_then(|(_, o)| o.as_array().ok())
        .map(|a| a.len())
        .unwrap_or(0);
    let mut put = |k: &str, v: String| rows.push((name.to_string(), k.to_string(), v));
    put("pages", pages.len().to_string());
    put("producer", info(&doc, b"Producer"));
    put("creator", info(&doc, b"Creator"));
    put("ocg", ocgs.to_string());
    put("user_unit_pages", user_unit.to_string());
    put("vp_pages", vp.to_string());
    put("forms.unique", it.unique_forms.len().to_string());
    for op in OPS {
        put(&format!("ops.{op}"), it.st.get(&format!("ops.{op}")).to_string());
    }
    for k in [
        "paths", "subpaths", "subpaths.closed", "paths.compound", "compound.mixed_winding", "segs.line", "segs.close", "segs.cubic", "segs.rect",
        "fill.nonzero", "fill.evenodd", "stroke", "clip.exact", "clip.bounded", "q.max_depth", "q.unbalanced", "q.unclosed_at_end",
        "ctm.refused", "forms.do", "forms.skipped_cycle_or_depth", "xobject.image_do", "bdc.oc", "pvec.ok", "pvec.refused",
    ] {
        put(k, it.st.get(k).to_string());
    }
    for c in CM_CLASSES {
        put(&format!("cm.{c}"), it.st.get(&format!("cm.{c}")).to_string());
    }
    let refused: Vec<(String, u64)> = it.st.0.iter().filter(|(k, _)| k.starts_with("pvec.refused.")).map(|(k, v)| (k.clone(), *v)).collect();
    for (k, v) in refused {
        put(&k, v.to_string());
    }
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: harness <dir-with-pdfs>");
    let mut rows = Vec::new();
    synthetic(&mut rows);
    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("dir").filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map_or(false, |x| x == "pdf")).collect();
    files.sort();
    for f in files {
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        measure(&name, &f, &mut rows);
    }
    println!("fixture\tkey\tvalue");
    for (f, k, v) in rows {
        println!("{f}\t{k}\t{v}");
    }
}
