//! The generators and the resolver behind the conversion benchmark corpus (N8.1, #1064).
//!
//! `tests/fixtures/pvec_bench_corpus.json` names six cells. The vendored `pvec-gold-fixtures`
//! revision has no drawing at the scale the cells ask for (the largest fixture is a few hundred
//! entities, and none has twenty viewports), so a cell that needs scale names a *synthetic
//! stand-in*: a drawing written by [`generate`] from the parameters in the manifest, with a
//! fixed seed, through acadrust's `DwgWriter`. The manifest says `synthetic`, so nobody mistakes
//! one for a field drawing. Nothing here runs a converter or needs one, and nothing writes outside
//! the directory a caller hands in.
//!
//! The suites `#[path]`-include this file, like `pvec_gold.rs`.

#![allow(dead_code)] // each suite uses its own part
#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use acadrust::entities::{Arc, Circle, Insert, Line, LwPolyline, Spline, Viewport};
use acadrust::tables::{BlockRecord, Layer};
use acadrust::types::{Vector2, Vector3};
use acadrust::{CadDocument, DwgWriter, EntityType};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The six cells, in the order the manifest and the reports use.
pub const CELLS: [&str; 6] = [
    "simple",
    "block_heavy",
    "spline_heavy",
    "paper_space",
    "large_industrial",
    "malformed",
];

/// The manifest, relative to the crate root.
pub const MANIFEST: &str = "tests/fixtures/pvec_bench_corpus.json";

/// The crate root.
pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// The parsed manifest.
pub fn manifest() -> Value {
    let text = std::fs::read_to_string(root().join(MANIFEST))
        .unwrap_or_else(|e| panic!("the benchmark corpus manifest {MANIFEST} is missing: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{MANIFEST} is not JSON: {e}"))
}

/// The lowercase hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// SplitMix64: a fixed, dependency-free stream, so a generated drawing is the same on every host.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A float in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// A float in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

fn param(p: &Value, key: &str) -> u64 {
    p[key]
        .as_u64()
        .unwrap_or_else(|| panic!("generator parameter {key:?} is missing or not an integer"))
}

fn line(rng: &mut Rng, x: f64, y: f64, span: f64) -> EntityType {
    EntityType::Line(Line::from_coords(
        x,
        y,
        0.0,
        x + rng.range(-span, span),
        y + rng.range(-span, span),
        0.0,
    ))
}

fn arc(rng: &mut Rng, x: f64, y: f64, r: f64) -> EntityType {
    let a = rng.range(0.0, 3.0);
    EntityType::Arc(Arc::from_coords(x, y, 0.0, r, a, a + rng.range(0.5, 3.0)))
}

fn circle(x: f64, y: f64, r: f64) -> EntityType {
    EntityType::Circle(Circle::from_coords(x, y, 0.0, r))
}

fn insert(name: &str, x: f64, y: f64, angle: f64) -> EntityType {
    let mut i = Insert::new(name, Vector3::new(x, y, 0.0));
    i.rotation = angle;
    EntityType::Insert(i)
}

/// A block definition `name`, and a handle to put entities into it with [`put`].
fn define_block(doc: &mut CadDocument, name: &str) -> acadrust::types::Handle {
    let mut rec = BlockRecord::new(name);
    rec.handle = doc.allocate_handle();
    let h = rec.handle;
    doc.block_records.add(rec).expect("block record");
    h
}

/// Puts `e` into the block whose record handle is `owner`.
fn put(doc: &mut CadDocument, owner: acadrust::types::Handle, mut e: EntityType) {
    e.common_mut().owner_handle = owner;
    doc.add_entity(e).expect("entity");
}

/// Adds the two layers a visibility mix hides entities on: one turned off, one frozen.
fn add_hidden_layers(doc: &mut CadDocument, percent: u64) {
    if percent == 0 {
        return;
    }
    let mut off = Layer::new("HID_OFF");
    off.turn_off();
    off.handle = doc.allocate_handle();
    doc.layers.add(off).expect("layer");
    let mut frozen = Layer::new("HID_FRZ");
    frozen.freeze();
    frozen.handle = doc.allocate_handle();
    doc.layers.add(frozen).expect("layer");
}

/// Puts leaf number `n` on a hidden layer when a fixed hash of `n` falls under `percent`, so the
/// share is about `percent` over any run of a few hundred leaves. Off and frozen alternate.
fn maybe_hide(e: &mut EntityType, n: u64, percent: u64) {
    if percent > 0 && (n.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 33) % 100 < percent {
        e.common_mut().layer = if n % 2 == 0 { "HID_OFF" } else { "HID_FRZ" }.to_string();
    }
}

/// `simple`: flat model space, lines, arcs and circles. Parameter: `entities`.
fn simple(p: &Value) -> CadDocument {
    let mut doc = CadDocument::new();
    let mut rng = Rng::new(0x51_4D_50_4C);
    for i in 0..param(p, "entities") {
        let (x, y) = (rng.range(0.0, 400.0), rng.range(0.0, 300.0));
        let e = match i % 3 {
            0 => line(&mut rng, x, y, 8.0),
            1 => {
                let r = rng.range(0.5, 5.0);
                arc(&mut rng, x, y, r)
            }
            _ => circle(x, y, rng.range(0.3, 3.0)),
        };
        doc.add_entity(e).expect("entity");
    }
    doc
}

/// `block_heavy`: `depth` nested block definitions, each holding two lines and `fanout` INSERTs of
/// the next level, instanced `roots` times from model space. One root places
/// `fanout^0 + ... + fanout^(depth-1)` instances.
fn block_heavy(p: &Value) -> CadDocument {
    let (depth, fanout, roots) = (param(p, "depth"), param(p, "fanout"), param(p, "roots"));
    let mut doc = CadDocument::new();
    let mut rng = Rng::new(0x42_4C_4B);
    let hidden = p["hidden"].as_u64().unwrap_or(0);
    add_hidden_layers(&mut doc, hidden);
    let owners: Vec<_> = (0..depth)
        .map(|d| define_block(&mut doc, &format!("B{d}")))
        .collect();
    // A block's leaves are drawn once per instance of the block, and the deepest blocks have the
    // most instances, so a hash would give a lumpy share. Hide leaves deepest first, whole levels
    // while they fit, until the hidden weight (instances drawn) reaches the share asked for.
    let weight = |d: u64| fanout.pow(d as u32);
    let total: u64 = (0..depth).map(|d| 2 * weight(d)).sum();
    let mut left = total * hidden / 100;
    let mut hide = vec![[false; 2]; depth as usize];
    for d in (0..depth).rev() {
        for slot in hide[d as usize].iter_mut() {
            if weight(d) <= left {
                *slot = true;
                left -= weight(d);
            }
        }
    }
    for d in 0..depth as usize {
        for j in 0..2 {
            let (x, y) = (rng.range(0.0, 4.0), rng.range(0.0, 4.0));
            let mut l = line(&mut rng, x, y, 2.0);
            if hide[d][j] {
                l.common_mut().layer = if d % 2 == 0 { "HID_OFF" } else { "HID_FRZ" }.to_string();
            }
            put(&mut doc, owners[d], l);
        }
        if d + 1 < depth as usize {
            for k in 0..fanout {
                let at = insert(
                    &format!("B{}", d + 1),
                    k as f64 * 5.0,
                    d as f64,
                    rng.range(0.0, 0.5),
                );
                put(&mut doc, owners[d], at);
            }
        }
    }
    let side = (roots as f64).sqrt().ceil() as u64;
    for r in 0..roots {
        let e = insert(
            "B0",
            (r % side) as f64 * 40.0,
            (r / side) as f64 * 40.0,
            rng.range(0.0, 6.0),
        );
        doc.add_entity(e).expect("root insert");
    }
    doc
}

/// `spline_heavy`: `splines` clamped B-splines of degree 2 to 5 with 6 to 21 control points.
fn spline_heavy(p: &Value) -> CadDocument {
    let mut doc = CadDocument::new();
    let mut rng = Rng::new(0x53_50_4C);
    for i in 0..param(p, "splines") {
        let n = 6 + (i % 16) as usize;
        let (x, y) = (rng.range(0.0, 600.0), rng.range(0.0, 400.0));
        let pts = (0..n)
            .map(|k| Vector3::new(x + k as f64 * 2.0, y + rng.range(-4.0, 4.0), 0.0))
            .collect();
        let s = Spline::from_control_points(2 + (i % 4) as i32, pts);
        doc.add_entity(EntityType::Spline(s)).expect("spline");
    }
    doc
}

/// `paper_space`: `model_lines` lines in model space and one layout, "Sheet A", with `viewports`
/// viewports in a grid.
fn paper_space(p: &Value) -> CadDocument {
    let mut doc = CadDocument::new();
    let mut rng = Rng::new(0x50_41_50);
    let hidden = p["hidden"].as_u64().unwrap_or(0);
    add_hidden_layers(&mut doc, hidden);
    for n in 0..param(p, "model_lines") {
        let (x, y) = (rng.range(0.0, 500.0), rng.range(0.0, 500.0));
        let mut l = line(&mut rng, x, y, 10.0);
        maybe_hide(&mut l, n, hidden);
        doc.add_entity(l).expect("entity");
    }
    doc.add_layout("Sheet A").expect("layout");
    let n = param(p, "viewports");
    let cols = 5;
    for v in 0..n {
        let mut vp = Viewport::with_size(
            Vector3::new(
                60.0 + (v % cols) as f64 * 110.0,
                60.0 + (v / cols) as f64 * 90.0,
                0.0,
            ),
            100.0,
            80.0,
        );
        vp.id = 2 + v as i16;
        doc.add_entity_to_layout(EntityType::Viewport(vp), "Sheet A")
            .expect("viewport");
    }
    doc
}

/// `large_industrial`: a plant. A grid of `equipment` INSERTs of four block types (each holding
/// `parts` entities), with pipe runs (lines and polylines), valves (circles) and bends (arcs)
/// between them, until model space holds `entities` records.
fn large_industrial(p: &Value) -> CadDocument {
    let (total, parts) = (param(p, "entities"), param(p, "parts"));
    let mut doc = CadDocument::new();
    let mut rng = Rng::new(0x50_4C_41_4E_54);
    doc.layers.add(Layer::new("P-PIPE")).expect("layer");
    doc.layers.add(Layer::new("E-EQUIP")).expect("layer");
    let hidden = p["hidden"].as_u64().unwrap_or(0);
    add_hidden_layers(&mut doc, hidden);
    let owners: Vec<_> = (0..4)
        .map(|t| define_block(&mut doc, &format!("EQ{t}")))
        .collect();
    for (t, o) in owners.iter().enumerate() {
        for k in 0..parts {
            let a = k as f64;
            let mut e = match (k + t as u64) % 3 {
                0 => line(&mut rng, a, a, 3.0),
                1 => circle(a, -a, 1.0 + (k % 4) as f64),
                _ => arc(&mut rng, a, a, 2.0),
            };
            maybe_hide(&mut e, k * 4 + t as u64, hidden);
            put(&mut doc, *o, e);
        }
    }
    let side = (total as f64).sqrt().ceil() as u64;
    for i in 0..total {
        let (x, y) = ((i % side) as f64 * 3.0, (i / side) as f64 * 3.0);
        let mut e = match i % 20 {
            0..=7 => line(&mut rng, x, y, 2.5),
            8..=11 => {
                let pts = (0..4)
                    .map(|k| Vector2::new(x + k as f64 * 0.7, y + rng.range(-0.5, 0.5)))
                    .collect();
                EntityType::LwPolyline(LwPolyline::from_points(pts))
            }
            12..=14 => {
                let r = rng.range(0.2, 1.2);
                arc(&mut rng, x, y, r)
            }
            15..=16 => circle(x, y, rng.range(0.1, 0.6)),
            _ => insert(&format!("EQ{}", i % 4), x, y, rng.range(0.0, 6.0)),
        };
        e.common_mut().layer = if i % 20 >= 17 { "E-EQUIP" } else { "P-PIPE" }.to_string();
        if i % 20 < 17 {
            maybe_hide(&mut e, i, hidden);
        }
        doc.add_entity(e).expect("entity");
    }
    doc
}

/// One drawing for a synthetic cell, written to DWG bytes.
///
/// # Panics
///
/// On a cell name that is not synthetic, or a parameter the generator needs and the manifest lacks.
pub fn generate(generator: &str, params: &Value) -> Vec<u8> {
    let doc = match generator {
        "simple" => simple(params),
        "block_heavy" => block_heavy(params),
        "spline_heavy" => spline_heavy(params),
        "paper_space" => paper_space(params),
        "large_industrial" => large_industrial(params),
        other => panic!("no generator named {other:?}"),
    };
    DwgWriter::write_to_vec(&doc).expect("the drawing writes")
}

/// A recursion bomb: `levels` block definitions, each inserting the next, instanced once.
fn nested_chain(levels: u64) -> CadDocument {
    let mut doc = CadDocument::new();
    let owners: Vec<_> = (0..levels)
        .map(|d| define_block(&mut doc, &format!("N{d}")))
        .collect();
    for d in 0..levels as usize {
        let e = if d + 1 < levels as usize {
            insert(&format!("N{}", d + 1), 0.0, 0.0, 0.0)
        } else {
            EntityType::Line(Line::from_coords(0.0, 0.0, 0.0, 1.0, 1.0, 0.0))
        };
        put(&mut doc, owners[d], e);
    }
    doc.add_entity(insert("N0", 0.0, 0.0, 0.0)).expect("root");
    doc
}

/// A spline with `points` control points and the knot vector that goes with them.
fn big_spline(points: u64) -> CadDocument {
    let mut doc = CadDocument::new();
    let pts = (0..points)
        .map(|k| Vector3::new(k as f64, (k % 7) as f64, 0.0))
        .collect();
    let s = Spline::from_control_points(3, pts);
    doc.add_entity(EntityType::Spline(s)).expect("spline");
    doc
}

/// One input of the `malformed` cell: the drawing's bytes.
///
/// `kind` is one of `block_recursion`, `entity_count`, `control_points`, `knots`, `name_length`
/// or `truncated`. The limit it must trip is the manifest's business, not the generator's.
pub fn generate_malformed(kind: &str, params: &Value) -> Vec<u8> {
    let doc = match kind {
        "block_recursion" => nested_chain(param(params, "levels")),
        "entity_count" => simple(params),
        "control_points" | "knots" => big_spline(param(params, "points")),
        "name_length" => {
            let mut doc = CadDocument::new();
            let name = "L".repeat(param(params, "bytes") as usize);
            doc.layers.add(Layer::new(&name)).expect("layer");
            let mut l = Line::from_coords(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
            l.common.layer = name;
            doc.add_entity(EntityType::Line(l)).expect("entity");
            doc
        }
        "truncated" => {
            let whole = generate("simple", params);
            return whole[..whole.len() * param(params, "keep_percent") as usize / 100].to_vec();
        }
        other => panic!("no malformed generator named {other:?}"),
    };
    DwgWriter::write_to_vec(&doc).expect("the drawing writes")
}

/// What a drawing holds, counted by converting it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Counts {
    /// Entities the conversion met: drawn leaves, excluded leaves and unexpanded INSERTs.
    pub encountered: u64,
    /// Entities written into the artifact.
    pub written: u64,
    /// Artifact bytes.
    pub artifact_bytes: u64,
}

/// Converts `bytes` into `dir` and counts. The one command behind every `entities` figure in the
/// manifest.
pub fn count(bytes: &[u8], dir: &Path) -> Result<Counts, libviprs::pvec_convert::PvecConvertError> {
    let dest = dir.join("cell.pvec");
    let out =
        libviprs::pvec_convert::PvecConversion::new(libviprs::cad::CadSource::Bytes(bytes), &dest)
            .unit(libviprs::pvec::SourceUnit::Meters)
            .convert()?;
    Ok(Counts {
        encountered: out.report.totals.encountered,
        written: out.written,
        artifact_bytes: std::fs::metadata(&out.artifact).map_or(0, |m| m.len()),
    })
}

/// Cell name to its manifest entry.
pub fn cells(m: &Value) -> BTreeMap<String, Value> {
    m["cells"]
        .as_object()
        .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

/// How much of the manifest [`problems`] re-derives from the generators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regen {
    /// Read the manifest only.
    No,
    /// Regenerate every cell but `large_industrial`, which takes minutes in a debug build.
    Small,
    /// Regenerate everything.
    All,
}

/// The evidence classes a cell may carry: EPIC O's four, and the stand-in label.
pub const EVIDENCE: [&str; 5] = [
    "KNOWN_TRUTH_EXACT",
    "DUAL_ORACLE_EQUIVALENT",
    "REAL_WORLD_DIFFERENTIAL",
    "QUARANTINED",
    "SYNTHETIC_STAND_IN",
];

/// The N3.6 limits the `malformed` cell must trip, by the name of their `SourceLimits` field.
pub const MALFORMED_LIMITS: [&str; 5] = [
    "max_block_depth",
    "max_entities",
    "max_spline_control_points",
    "max_spline_knots",
    "max_name_bytes",
];

/// The `SourceLimits` a malformed input is run under: the defaults with the manifest's overrides.
pub fn limits_for(input: &Value) -> libviprs::pvec::SourceLimits {
    let mut l = libviprs::pvec::SourceLimits::default();
    if let Some(o) = input["limits"].as_object() {
        for (k, v) in o {
            let n = v
                .as_u64()
                .unwrap_or_else(|| panic!("limit {k} is not an integer"));
            match k.as_str() {
                "max_block_depth" => l.max_block_depth = n as u32,
                "max_entities" => l.max_entities = n,
                "max_spline_control_points" => l.max_spline_control_points = n as usize,
                "max_spline_knots" => l.max_spline_knots = n as usize,
                "max_name_bytes" => l.max_name_bytes = n as usize,
                other => panic!("the manifest overrides a limit this suite does not know: {other}"),
            }
        }
    }
    l
}

/// Which limit refused a conversion, by field name, or `open` when the provider could not read it.
pub fn refused_by(e: &libviprs::pvec_convert::PvecConvertError) -> String {
    use libviprs::pvec::flatten::FlattenError;
    use libviprs::pvec_convert::PvecConvertError as E;
    match e {
        E::Flatten(FlattenError::Limit { limit, .. }) => limit.name().to_string(),
        E::Flatten(FlattenError::BlockRecursionLimit { .. }) => "max_block_depth".to_string(),
        E::Open(_) => "open".to_string(),
        other => format!("other: {other}"),
    }
}

/// Every way `m` fails to be a good benchmark corpus manifest. Empty means it is one.
pub fn problems(m: &Value, regen: Regen) -> Vec<String> {
    let mut out = Vec::new();
    let pin: Value = serde_json::from_str(
        &std::fs::read_to_string(root().join("tests/fixtures/pvec-gold/PIN.json"))
            .expect("the gold PIN is vendored"),
    )
    .expect("PIN.json is JSON");
    let revision = m["corpus_revision"].as_str().unwrap_or("");
    if revision.is_empty() || Some(revision) != pin["corpus_revision"].as_str() {
        out.push(format!(
            "the manifest names corpus revision {revision:?}, the vendored pin is {:?}",
            pin["corpus_revision"]
        ));
    }
    let cells = cells(m);
    for name in CELLS {
        if !cells.contains_key(name) {
            out.push(format!("cell {name} is missing"));
        }
    }
    let tmp = tempfile::tempdir().expect("tempdir");
    for (name, c) in &cells {
        let at = |what: &str| format!("cell {name}: {what}");
        if !CELLS.contains(&name.as_str()) {
            out.push(at("is not one of the six cells"));
        }
        if c["corpus_revision"].as_str() != Some(revision) {
            out.push(at(&format!(
                "names corpus revision {:?}, not {revision:?}",
                c["corpus_revision"]
            )));
        }
        if !EVIDENCE.contains(&c["evidence_class"].as_str().unwrap_or("")) {
            out.push(at("has no valid evidence class"));
        }
        if c["provider"].as_str() != Some("acadrust") {
            out.push(at("does not name the acadrust provider"));
        }
        if c["unit"].as_str().is_none() {
            out.push(at("records no source unit"));
        }
        if c["source"].as_str() != Some("synthetic") {
            out.push(at(
                "is not marked synthetic, and no vendored fixture reaches this scale",
            ));
        }
        if name == "malformed" {
            malformed(c, regen, tmp.path(), &mut out);
            continue;
        }
        if c["entities"]["encountered"].as_u64().is_none_or(|n| n == 0) {
            out.push(at("has no measured entity count"));
        }
        let big = name == "large_industrial";
        let rebuild = regen == Regen::All || (regen == Regen::Small && !big);
        if rebuild {
            let bytes = generate(c["generator"].as_str().unwrap_or(""), &c["params"]);
            if c["sha256"].as_str() != Some(&sha256_hex(&bytes)) {
                out.push(at("its checksum no longer matches the generator's output"));
            }
            if c["bytes"].as_u64() != Some(bytes.len() as u64) {
                out.push(at("its byte size no longer matches the generator's output"));
            }
            match count(&bytes, tmp.path()) {
                Ok(n) => {
                    if c["entities"]["encountered"].as_u64() != Some(n.encountered)
                        || c["entities"]["written"].as_u64() != Some(n.written)
                    {
                        out.push(at(&format!("its entity count moved: now {n:?}")));
                    }
                }
                Err(e) => out.push(at(&format!("no longer converts: {e}"))),
            }
        } else if c["sha256"].as_str().is_none_or(|s| s.len() != 64) {
            out.push(at("records no sha256"));
        }
    }
    out
}

fn malformed(c: &Value, regen: Regen, dir: &Path, out: &mut Vec<String>) {
    let inputs = c["inputs"].as_array().cloned().unwrap_or_default();
    for limit in MALFORMED_LIMITS {
        if !inputs.iter().any(|i| i["limit"].as_str() == Some(limit)) {
            out.push(format!("cell malformed: no input trips {limit}"));
        }
    }
    for i in &inputs {
        let id = i["id"].as_str().unwrap_or("?");
        let at = |what: &str| format!("cell malformed, input {id}: {what}");
        if regen == Regen::No {
            continue;
        }
        let bytes = generate_malformed(i["kind"].as_str().unwrap_or(""), &i["params"]);
        if i["sha256"].as_str() != Some(&sha256_hex(&bytes)) {
            out.push(at("its checksum no longer matches the generator's output"));
        }
        let want = i["limit"].as_str().unwrap_or("");
        let dest = dir.join(format!("{id}.pvec"));
        let run = libviprs::pvec_convert::PvecConversion::new(
            libviprs::cad::CadSource::Bytes(&bytes),
            &dest,
        )
        .unit(libviprs::pvec::SourceUnit::Meters)
        .limits(limits_for(i))
        .convert();
        match run {
            Ok(_) => out.push(at(&format!("was converted, it must be refused by {want}"))),
            Err(e) if refused_by(&e) == want => {}
            Err(e) => out.push(at(&format!(
                "was refused by {:?}, not {want}",
                refused_by(&e)
            ))),
        }
    }
}

// ---------------------------------------------------------------------------------- measuring

/// Three runs of one figure: the median, with the extremes printed beside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stat {
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl Stat {
    /// The statistics of `samples`.
    ///
    /// # Panics
    ///
    /// On no samples: a column nobody measured is a hole, never a zero.
    pub fn of(samples: &[f64]) -> Self {
        assert!(!samples.is_empty(), "a statistic of no samples");
        let mut v = samples.to_vec();
        v.sort_by(f64::total_cmp);
        Self {
            median: v[v.len() / 2],
            min: v[0],
            max: v[v.len() - 1],
        }
    }

    /// How far apart the extremes are, as a fraction of the median.
    pub fn spread(&self) -> f64 {
        (self.max - self.min) / self.median
    }

    /// A row whose spread is over 20% is marked unstable and never gates.
    pub fn unstable(&self) -> bool {
        self.spread() > 0.20
    }
}

/// The poll, counting from 1, at which the one-minute load had been under `ncpu` for three polls
/// running, or `None` when it never was.
pub fn quiet_after(loads: impl IntoIterator<Item = f64>, ncpu: usize) -> Option<usize> {
    let mut run = 0;
    for (n, load) in loads.into_iter().enumerate() {
        run = if load < ncpu as f64 { run + 1 } else { 0 };
        if run == 3 {
            return Some(n + 1);
        }
    }
    None
}

/// The columns of `row` that are missing or null. A benchmark row with any is refused, not printed.
pub fn blank_columns(row: &Value, columns: &[&str]) -> Vec<String> {
    columns
        .iter()
        .filter(|c| row.get(**c).is_none_or(Value::is_null))
        .map(|c| (*c).to_string())
        .collect()
}

/// How `cur` compares with the baseline row `base` for a provider cell: more than 10% loss of
/// entities per second, or more than 15% growth in tracked allocation peak, is a regression. An
/// unstable row on either side never gates.
pub fn provider_regressions(base: &Value, cur: &Value) -> Vec<String> {
    let flag = |r: &Value| r["unstable"].as_bool().unwrap_or(false);
    if flag(base) || flag(cur) {
        return Vec::new();
    }
    let mut out = Vec::new();
    let eps = |r: &Value| r["entities_per_s"]["median"].as_f64().unwrap_or(f64::NAN);
    if eps(cur) < eps(base) * 0.90 {
        out.push(format!(
            "entities per second fell from {:.0} to {:.0}, more than 10%",
            eps(base),
            eps(cur)
        ));
    }
    let peak = |r: &Value| r["tracked_peak"].as_f64().unwrap_or(f64::NAN);
    if peak(cur) > peak(base) * 1.15 {
        out.push(format!(
            "tracked allocation peak grew from {:.0} to {:.0}, more than 15%",
            peak(base),
            peak(cur)
        ));
    }
    out
}

// ------------------------------------------------------------------------------- the host

/// Logical cpus of the host. A container reads the host's `/proc`, so this is the machine's count
/// and not a cgroup quota.
pub fn host_cpus() -> usize {
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|t| t.lines().filter(|l| l.starts_with("processor")).count())
        .ok()
        .filter(|n| *n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// The one-minute load average, read from `/proc/loadavg`, which a container shares with its host.
pub fn host_load() -> Option<f64> {
    std::fs::read_to_string("/proc/loadavg")
        .ok()?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Waits until the one-minute load has been under the cpu count for three polls, `every` apart,
/// for at most `polls` polls. Returns the load at the last poll, or says why it gave up.
pub fn require_quiet(polls: usize, every: std::time::Duration) -> Result<f64, String> {
    let ncpu = host_cpus();
    let mut seen = Vec::new();
    for _ in 0..polls {
        let Some(load) = host_load() else {
            return Err("this host reports no load average, so a quiet run cannot be shown".into());
        };
        seen.push(load);
        if quiet_after(seen.iter().copied(), ncpu).is_some() {
            return Ok(load);
        }
        std::thread::sleep(every);
    }
    Err(format!(
        "the host never was quiet: one-minute load {seen:?} against {ncpu} cpus; no number was taken"
    ))
}

/// Where a row ran: architecture, OS and the host's name from `LIBVIPRS_BENCH_HOST`, which a run
/// must set (an x86_64 row comes from a native x64 host, never a local emulation).
pub fn platform() -> Result<String, String> {
    let host = std::env::var("LIBVIPRS_BENCH_HOST")
        .ok()
        .filter(|h| !h.is_empty())
        .ok_or("LIBVIPRS_BENCH_HOST is not set: every row names the host it ran on")?;
    Ok(format!(
        "{}-{}-{host}",
        std::env::consts::ARCH,
        std::env::consts::OS
    ))
}

/// Resets the kernel's peak-RSS mark to the current RSS, so one cell's peak is its own. `false`
/// where the kernel refuses; the row then says its RSS is the process's running high-water mark.
pub fn reset_peak_rss() -> bool {
    std::fs::write("/proc/self/clear_refs", "5").is_ok()
}

/// Process peak resident set, from `VmHWM`.
pub fn peak_rss() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kib: u64 = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    Some(kib * 1024)
}

/// The baseline file, relative to the crate root.
pub const BASELINE: &str = "tests/fixtures/pvec_bench_baseline.json";

/// The committed baseline rows, `{}` when none is committed yet.
pub fn baseline_rows() -> Value {
    std::fs::read_to_string(root().join(BASELINE))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .map_or_else(|| Value::Object(Default::default()), |v| v["rows"].clone())
}

/// How `cur` compares with the baseline row `base` for a normalization cell: any single stage
/// that grew by more than 10% in nanoseconds per entity is a regression. The rows carry
/// `stage_ns` (stage name to nanoseconds per entity). An unstable row on either side never gates.
pub fn stage_regressions(base: &Value, cur: &Value) -> Vec<String> {
    let flag = |r: &Value| r["unstable"].as_bool().unwrap_or(false);
    if flag(base) || flag(cur) {
        return Vec::new();
    }
    let mut out = Vec::new();
    let Some(stages) = cur["stage_ns"].as_object() else {
        return vec!["the row has no stage_ns".into()];
    };
    for (name, now) in stages {
        let (Some(now), Some(was)) = (now.as_f64(), base["stage_ns"][name].as_f64()) else {
            continue;
        };
        if now > was * 1.10 {
            out.push(format!(
                "stage {name} grew from {was:.1} to {now:.1} ns per entity, more than 10%"
            ));
        }
    }
    out
}

/// The stages' share of a pass: the sum of the stage times against the pass total, as a fraction
/// of the total that no stage accounts for (it can be negative if the timers overlap). Printed,
/// never spread over the stages.
pub fn unattributed(stage_secs: &[f64], total_secs: f64) -> f64 {
    (total_secs - stage_secs.iter().sum::<f64>()) / total_secs
}

/// More than 10% growth in process peak RSS or in wall time against the baseline at an unchanged
/// fixture, even below the ceiling. An unstable row on either side never gates the wall time.
pub fn memory_regressions(base: &Value, cur: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let (b, c) = (
        base["peak_rss_bytes"].as_f64(),
        cur["peak_rss_bytes"].as_f64(),
    );
    if let (Some(b), Some(c)) = (b, c)
        && c > b * 1.10
    {
        out.push(format!(
            "peak RSS grew from {b:.0} to {c:.0} bytes, more than 10%"
        ));
    }
    let flag = |r: &Value| r["unstable"].as_bool().unwrap_or(false);
    let (bw, cw) = (
        base["wall_secs"]["median"].as_f64(),
        cur["wall_secs"]["median"].as_f64(),
    );
    if let (Some(bw), Some(cw)) = (bw, cw)
        && !flag(base)
        && !flag(cur)
        && cw > bw * 1.10
    {
        out.push(format!(
            "wall time grew from {bw:.3}s to {cw:.3}s, more than 10%"
        ));
    }
    out
}
