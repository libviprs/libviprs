//! Executable acceptance checks for `docs/spikes/dwg-mapping.md` (issue #1042, N1.3).
//!
//! The spike is a document plus a measurement, so these tests read both as text and hold them to
//! the acceptance criteria of the issue, one test (or a few) per criterion. They are std-only on
//! purpose: the claim they guard is "libviprs invents no mapping rule, no threshold and no curve
//! mathematics", and a test that pulled in a curve crate to check it would be the wrong shape.
//!
//! The measurements come from `docs/spikes/dwg-mapping/harness`, which maps the decoded records of
//! the r0-open corpus (libviprs/pvec-gold-fixtures, tag `corpus-r0-open`) into PVec's public
//! semantic API and compares the result with each fixture's `expected.math.json`. It writes
//! `results-x64.tsv`. The document cites that file by fixture id, and these tests fail when a row
//! and its measurement stop agreeing.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn spike_dir() -> PathBuf {
    root().join("docs/spikes/dwg-mapping")
}

fn doc() -> String {
    let p = root().join("docs/spikes/dwg-mapping.md");
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// One measured record, keyed by id (a fixture id for a row, `P<n>` for an alternative probe).
#[derive(Debug, Clone)]
struct Rec {
    kind: String,
    fixture: String,
    dwg_class: String,
    target: String,
    result: String,
    reason: String,
    worst_dev: String,
    ratio: String,
    notes: BTreeMap<String, String>,
}

fn results() -> BTreeMap<String, Rec> {
    results_from("results-x64.tsv")
}

fn results_from(file: &str) -> BTreeMap<String, Rec> {
    let p = spike_dir().join(file);
    let text = fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
    let mut lines = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<&str> = lines.next().expect("empty results file").split('\t').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("results file has no column `{name}`"))
    };
    let (id, kd, fx, dc, tg, rs, rn, wd, ra, no) = (
        col("id"),
        col("kind"),
        col("fixture"),
        col("dwg_class"),
        col("target"),
        col("result"),
        col("reason"),
        col("worst_dev"),
        col("ratio"),
        col("notes"),
    );
    let mut out = BTreeMap::new();
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        let notes = f[no]
            .split(';')
            .filter_map(|kv| kv.split_once('='))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect();
        out.insert(
            f[id].to_string(),
            Rec {
                kind: f[kd].to_string(),
                fixture: f[fx].to_string(),
                dwg_class: f[dc].to_string(),
                target: f[tg].to_string(),
                result: f[rs].to_string(),
                reason: f[rn].to_string(),
                worst_dev: f[wd].to_string(),
                ratio: f[ra].to_string(),
                notes,
            },
        );
    }
    out
}

/// The text of the section whose heading line starts with `heading`, up to the next heading of the
/// same or a higher level.
fn section(doc: &str, heading: &str) -> String {
    let level = heading.chars().take_while(|c| *c == '#').count();
    let start = doc
        .find(&format!("\n{heading}"))
        .map(|i| i + 1)
        .or_else(|| doc.starts_with(heading).then_some(0))
        .unwrap_or_else(|| panic!("section `{heading}` is missing from the spike"));
    let body_from = doc[start..].find('\n').map_or(doc.len(), |n| start + n + 1);
    let rest = &doc[body_from..];
    let mut end = rest.len();
    let mut at = 0;
    for line in rest.split_inclusive('\n') {
        let hashes = line.chars().take_while(|c| *c == '#').count();
        if hashes > 0 && hashes <= level && line[hashes..].starts_with(' ') {
            end = at;
            break;
        }
        at += line.len();
    }
    rest[..end].to_string()
}

fn table_rows(sec: &str) -> Vec<Vec<String>> {
    sec.lines()
        .filter(|l| l.starts_with('|'))
        .map(|l| {
            l.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect::<Vec<_>>()
        })
        .filter(|r| !r.iter().all(|c| c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' ')))
        .collect()
}

fn flat(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every fixture of the r0-open corpus at tag `corpus-r0-open` (30 probes). Shear, general affine
/// and singular are excluded by the corpus itself (no single INSERT form) and are not rows.
const FIXTURES: &[&str] = &[
    "arc-seam-001",
    "bezier-d3-001",
    "bspline-d3-001",
    "bspline-d5-001",
    "bulge-arc-001",
    "bulge-arc-neg-001",
    "circle-001",
    "closed-by-endpoint-001",
    "closed-periodic-001",
    "ellipse-001",
    "elliptical-arc-001",
    "large-coords-001",
    "layer-state-001",
    "line-001",
    "lwpolyline-closed-001",
    "near-singular-001",
    "nested-blocks-001",
    "nonuniform-scale-circle-001",
    "nonuniform-scale-rotated-001",
    "nonunit-weight-001",
    "nurbs-d2-offconic-001",
    "nurbs-d3-001",
    "polyline-001",
    "polyline2d-001",
    "reflection-001",
    "repeated-knot-c0-001",
    "rotation-irrational-001",
    "tiny-radius-001",
    "translation-001",
    "uniform-scale-001",
];

/// The classes the issue lists, each with the fixture that carries it.
const ISSUE_CLASSES: &[(&str, &str)] = &[
    ("LINE", "line-001"),
    ("LWPOLYLINE open", "polyline-001"),
    ("LWPOLYLINE closed", "lwpolyline-closed-001"),
    ("POLYLINE", "polyline2d-001"),
    ("bulge positive", "bulge-arc-001"),
    ("bulge negative", "bulge-arc-neg-001"),
    ("CIRCLE", "circle-001"),
    ("ARC", "arc-seam-001"),
    ("ELLIPSE", "ellipse-001"),
    ("ELLIPSE arc", "elliptical-arc-001"),
    ("SPLINE B-spline", "bspline-d3-001"),
    ("SPLINE repeated knot", "repeated-knot-c0-001"),
    ("SPLINE periodic", "closed-periodic-001"),
    ("SPLINE rational", "nonunit-weight-001"),
    ("block nest", "nested-blocks-001"),
    ("layer state", "layer-state-001"),
];

const RESULTS: &[&str] = &["exact", "equivalent", "RequiresPVecCapability", "refused"];

struct Row {
    dwg_class: String,
    fixture: String,
    target: String,
    result: String,
    worst_dev: String,
    evidence: String,
}

fn result_rows(doc: &str) -> Vec<Row> {
    let sec = section(doc, "## Result table");
    table_rows(&sec)
        .into_iter()
        .skip(1)
        .map(|c| {
            assert!(c.len() >= 7, "result row has {} cells, want 7: {c:?}", c.len());
            Row {
                dwg_class: c[1].clone(),
                fixture: c[2].trim_matches('`').to_string(),
                target: c[3].clone(),
                result: c[4].clone(),
                worst_dev: c[5].clone(),
                evidence: c[6].trim_matches('`').to_string(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// AC1: a row for every listed class, each naming the fixture it was measured on.
// ---------------------------------------------------------------------------------------------

#[test]
fn ac1_every_corpus_fixture_has_exactly_one_row_that_names_it() {
    let rows = result_rows(&doc());
    for f in FIXTURES {
        let n = rows.iter().filter(|r| r.fixture == *f).count();
        assert_eq!(n, 1, "fixture `{f}` must have exactly one row in the result table, has {n}");
    }
    assert_eq!(rows.len(), FIXTURES.len(), "the table has rows that are not corpus fixtures");
}

#[test]
fn ac1_every_class_the_issue_lists_is_covered_on_a_named_fixture() {
    let rows = result_rows(&doc());
    for (class, fixture) in ISSUE_CLASSES {
        let r = rows
            .iter()
            .find(|r| r.fixture == *fixture)
            .unwrap_or_else(|| panic!("no row for {class} ({fixture})"));
        assert!(!r.dwg_class.is_empty() && !r.target.is_empty(), "row for {class} is incomplete");
    }
}

#[test]
fn ac1_every_row_is_one_of_the_four_outcomes_and_agrees_with_its_measurement() {
    let doc = doc();
    let data = results();
    for r in result_rows(&doc) {
        assert!(RESULTS.contains(&r.result.as_str()), "row {}: `{}` is not one of {RESULTS:?}", r.fixture, r.result);
        let m = data
            .get(&r.evidence)
            .unwrap_or_else(|| panic!("row {} cites evidence `{}` that is not in results-x64.tsv", r.fixture, r.evidence));
        assert_eq!(m.fixture, r.fixture, "evidence {} was measured on another fixture", r.evidence);
        assert_eq!(m.kind, "row", "evidence {} is a probe, not a row", r.evidence);
        assert_eq!(m.result, r.result, "row {}: document says {}, measurement says {}", r.fixture, r.result, m.result);
        assert_eq!(m.worst_dev, r.worst_dev, "row {}: worst deviation differs from the measurement", r.fixture);
        assert!(!m.dwg_class.is_empty());
    }
}

#[test]
fn ac1_every_measured_row_is_in_the_document() {
    let doc = doc();
    let rows = result_rows(&doc);
    for (id, m) in results() {
        if m.kind == "row" {
            assert!(rows.iter().any(|r| r.evidence == id), "measurement `{id}` is not in the result table");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// AC2: bulge is checked against the fixture's own expected arc, and a negative sweep is covered.
// ---------------------------------------------------------------------------------------------

#[test]
fn ac2_bulge_rows_compare_against_the_expected_arc_parameters() {
    let data = results();
    for id in ["bulge-arc-001", "bulge-arc-neg-001"] {
        let m = &data[id];
        assert_eq!(m.notes.get("compared_to").map(String::as_str), Some("expected.math.json"), "{id}");
        for k in ["centre", "radius", "start_angle", "sweep", "sense"] {
            assert!(m.notes.contains_key(&format!("expected_{k}")), "{id}: no expected_{k} in the record");
            assert!(m.notes.contains_key(&format!("candidate_{k}")), "{id}: no candidate_{k} in the record");
        }
        assert_eq!(
            m.notes.get("pvec_arc_from_expected").map(String::as_str),
            Some("ok"),
            "{id}: PVec must be shown to hold the expected arc itself"
        );
    }
}

#[test]
fn ac2_negative_bulge_sweep_sign_is_measured_and_stated() {
    let data = results();
    let m = &data["bulge-arc-neg-001"];
    assert_eq!(m.notes["expected_sense"], "CW");
    assert_eq!(m.notes["candidate_sense"], "CW", "the sign of the bulge must give a clockwise arc");
    assert_eq!(m.notes["pvec_sense"], "cw", "PVec must hold it as a reversed trim");
    let pos = &data["bulge-arc-001"];
    assert_eq!(pos.notes["expected_sense"], "CCW");
    assert_eq!(pos.notes["candidate_sense"], "CCW");
    let d = flat(&section(&doc(), "## Bulge"));
    assert!(d.contains("negative") && d.contains("clockwise"), "the Bulge section must say what the negative sign does");
    assert!(d.contains("expected.math.json"), "the Bulge section must name the truth it compared against");
    assert!(
        d.contains("never against a second") || d.contains("not against a second"),
        "the Bulge section must say the comparison is not against a second libviprs computation"
    );
}

// ---------------------------------------------------------------------------------------------
// AC3: a repeated interior knot is reported, with what the mapping did with it, citing P1.2.
// ---------------------------------------------------------------------------------------------

#[test]
fn ac3_repeated_interior_knot_is_reported_with_what_the_mapping_did() {
    let data = results();
    let m = &data["repeated-knot-c0-001"];
    assert_eq!(m.notes["interior_knot"], "0.5");
    assert_eq!(m.notes["interior_multiplicity"], "3");
    assert_eq!(m.notes["degree"], "3");
    assert_eq!(m.notes["knots_identical"], "true", "the knot vector must go through whole (NG-B3)");
    assert_eq!(m.notes["two_sided_d1_at_knot"], "refused", "a two-sided derivative at a C0 knot is refused");
    assert!(m.notes["one_sided_d1_left"] != m.notes["one_sided_d1_right"], "a C0 break has two different one-sided tangents");
    let d = flat(&section(&doc(), "## Splines"));
    for needle in ["repeated-knot-c0-001", "multiplicity 3", "NP-14", "NP-16", "P1.2"] {
        assert!(d.contains(needle), "the Splines section must mention `{needle}`");
    }
    assert!(
        d.contains("passed through whole") || d.contains("kept whole"),
        "the Splines section must say the knot vector is passed through, not rewritten"
    );
}

// ---------------------------------------------------------------------------------------------
// AC4: no exact target means RequiresPVecCapability with the capability named, never tessellated.
// ---------------------------------------------------------------------------------------------

#[test]
fn ac4_requires_pvec_capability_rows_name_the_missing_capability() {
    let doc = doc();
    let data = results();
    let mut seen = 0;
    for r in result_rows(&doc) {
        if r.result == "RequiresPVecCapability" {
            seen += 1;
            let m = &data[&r.evidence];
            assert!(m.reason.len() > 8 && m.reason != "-", "row {}: the measurement names no missing capability", r.fixture);
            assert!(
                r.target.contains("RequiresPVecCapability") && r.target.contains(&m.reason),
                "row {}: the target cell must carry the missing capability `{}`",
                r.fixture,
                m.reason
            );
        }
    }
    assert!(seen >= 3, "bulge (two rows) and a periodic spline are expected to need a PVec capability, found {seen}");
}

#[test]
fn ac4_nothing_is_tessellated_to_make_a_row_green() {
    let doc = flat(&doc());
    assert!(
        doc.contains("none of these rows is tessellated") || doc.contains("no row is tessellated"),
        "the document must say no row was tessellated"
    );
    for (id, m) in results() {
        assert!(!m.target.to_lowercase().contains("tessellat"), "{id}: a target named tessellation");
        assert!(!m.reason.to_lowercase().contains("tessellat") || m.reason.to_lowercase().contains("not"), "{id}");
    }
    let src = fs::read_to_string(spike_dir().join("harness/src/main.rs")).expect("harness source");
    for banned in ["flatten", "tessellate", "sample_to_polyline", "discretise", "discretize"] {
        let lower = src.to_lowercase();
        for line in lower.lines().filter(|l| !l.trim_start().starts_with("//")) {
            assert!(!line.contains(banned), "the harness calls `{banned}`: {line}");
        }
    }
}

#[test]
fn ac4_the_periodic_spline_is_refused_by_pvec_and_not_rewritten() {
    let data = results();
    let m = &data["closed-periodic-001"];
    assert_eq!(m.result, "RequiresPVecCapability");
    assert_eq!(m.notes["pvec_refusal_rule"], "NP-14");
    assert_eq!(m.notes["source_periodic"], "true");
    assert_eq!(m.notes["control_points"], "6");
    assert_eq!(m.notes["knots"], "7");
    let d = flat(&section(&doc(), "## Splines"));
    assert!(d.contains("periodic") && d.contains("knot insertion"), "the Splines section must say why libviprs cannot unwrap it");
}

// ---------------------------------------------------------------------------------------------
// AC5: the decision paragraph says what phase 4 may implement exactly and what is deferred.
// ---------------------------------------------------------------------------------------------

#[test]
fn ac5_decision_lists_exact_and_deferred_classes_with_measured_deviations() {
    let doc = doc();
    let dec = flat(&section(&doc, "## Decision"));
    let rows = result_rows(&doc);
    assert!(dec.contains("phase 4") || dec.contains("Phase 4"), "the decision must address phase 4");
    assert!(dec.contains("may implement") && dec.contains("deferred"), "the decision must separate what may be implemented from what is deferred");
    for r in &rows {
        if r.result == "RequiresPVecCapability" {
            assert!(dec.contains(&r.fixture) || dec.contains(r.dwg_class.as_str()), "deferred class {} is not in the decision", r.fixture);
        }
    }
    // The worst deviation of the exact rows is quoted.
    let data = results();
    let worst_exact = data
        .values()
        .filter(|m| m.kind == "row" && (m.result == "exact" || m.result == "equivalent"))
        .map(|m| m.worst_dev.clone())
        .max_by(|a, b| a.parse::<f64>().unwrap_or(0.0).total_cmp(&b.parse::<f64>().unwrap_or(0.0)))
        .expect("no exact rows");
    assert!(dec.contains(&worst_exact), "the decision must quote the worst measured deviation {worst_exact}");
}

#[test]
fn ac5_no_measured_row_exceeds_the_p12_tolerance() {
    for (id, m) in results() {
        if m.kind == "row" && (m.result == "exact" || m.result == "equivalent") {
            let ratio: f64 = m.ratio.parse().unwrap_or_else(|_| panic!("{id}: ratio `{}`", m.ratio));
            assert!(ratio <= 1.0, "{id}: worst deviation is {ratio} of the NP-04 tolerance");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Honesty about what the evidence is.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_evidence_scope_is_stated_and_not_inflated() {
    let d = flat(&doc());
    for needle in [
        "READER_MAPPING_ONLY",
        "oracles: none",
        "ADR 0008",
        "common-mode",
        "corpus-r0-open",
        "f94f7f2",
    ] {
        assert!(d.contains(needle), "the document must say `{needle}`");
    }
    assert!(!d.contains("DUAL_ORACLE_EQUIVALENT"), "no row may claim a dual-oracle class");
    assert!(
        d.contains("not a conformance result") || d.contains("is not conformance evidence"),
        "the document must say this is not conformance evidence"
    );
    let un = flat(&section(&doc(), "## What I could not verify"));
    for needle in ["3D POLYLINE", "polyface", "closed old-style POLYLINE", "closing segment", "shear"] {
        assert!(un.contains(needle), "the unverified list must name `{needle}` as not covered by the corpus");
    }
}

#[test]
fn the_pins_are_cited_and_match_the_run_script() {
    let d = doc();
    let run = fs::read_to_string(spike_dir().join("run.sh")).expect("run.sh");
    for key in ["corpus_tag=corpus-r0-open", "corpus_rev=f94f7f2", "pvec_rev="] {
        assert!(run.contains(key), "run.sh must pin `{key}`");
    }
    let pvec = run
        .lines()
        .find_map(|l| l.strip_prefix("pvec_rev="))
        .expect("pvec_rev line")
        .trim()
        .to_string();
    assert!(d.contains(&pvec), "the document must cite the PVec revision {pvec} that run.sh pins");
}

// ---------------------------------------------------------------------------------------------
// The harness invents no tolerance and carries no second curve library.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_harness_defines_no_tolerance_of_its_own() {
    let src = fs::read_to_string(spike_dir().join("harness/src/main.rs")).expect("harness source");
    for (n, line) in src.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        let lower = code.to_lowercase();
        assert!(
            !lower.contains("1e-") && !lower.contains("epsilon") && !lower.contains("f64::eps"),
            "harness line {} carries a tolerance-shaped literal: {line}",
            n + 1
        );
    }
    let toml = fs::read_to_string(spike_dir().join("harness/Cargo.toml")).expect("harness manifest");
    let deps: Vec<&str> = toml
        .lines()
        .skip_while(|l| !l.starts_with("[dependencies]"))
        .skip(1)
        .take_while(|l| !l.starts_with('['))
        .filter(|l| l.contains('='))
        .map(|l| l.split('=').next().unwrap().trim())
        .collect();
    for d in &deps {
        assert!(
            ["pvec-geometry", "pvec-numeric", "serde_json"].contains(d),
            "the harness depends on `{d}`: only PVec and a JSON reader are allowed"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The gaps: every real one is an issue in libviprs/pvec, referencing #1042, none closing it.
// ---------------------------------------------------------------------------------------------

#[test]
fn gaps_every_gap_is_listed_with_a_pvec_issue_and_the_adr_candidates_are_answered() {
    let doc = doc();
    let gaps = section(&doc, "## PVec gaps");
    let g = flat(&gaps);
    let rows = table_rows(&gaps);
    assert!(rows.len() >= 3, "the gap table is missing or empty");
    for row in rows.iter().skip(1) {
        assert!(
            row.iter().any(|c| c.contains("libviprs/pvec#")),
            "gap row has no libviprs/pvec issue reference: {row:?}"
        );
    }
    // The five candidates of NG-11 are each answered, whether or not they turned out to be real.
    for cand in [
        "line segment from two points",
        "polyline bulge",
        "ellipse from a major-axis vector",
        "bounds of an arc",
        "angle convention",
    ] {
        assert!(g.contains(cand), "the NG-11 candidate `{cand}` is not answered");
    }
}

#[test]
fn the_document_never_closes_the_parent_epic() {
    let d = doc();
    let lower = d.to_lowercase();
    for kw in ["closes #1039", "fixes #1039", "resolves #1039", "close #1039"] {
        assert!(!lower.contains(kw), "`{kw}` must not appear: #1039 is the parent epic");
    }
}

// ---------------------------------------------------------------------------------------------
// Both architectures were run, and the verdicts do not depend on which.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_arm64_run_reaches_the_same_verdicts_as_the_native_x64_run() {
    let (x, a) = (results(), results_from("results-arm64.tsv"));
    assert_eq!(x.keys().collect::<Vec<_>>(), a.keys().collect::<Vec<_>>(), "the two runs measured different records");
    for (id, m) in &x {
        assert_eq!(m.result, a[id].result, "{id}: x64 and arm64 disagree on the verdict");
    }
    let d = flat(&doc());
    assert!(d.contains("results-arm64.tsv") && d.contains("native x86_64"), "the document must say which results came from which machine");
}
