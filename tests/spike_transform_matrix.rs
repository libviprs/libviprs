//! Executable acceptance checks for `docs/spikes/transform-matrix.md` (issue #1044, N1.5).
//!
//! The spike is a document plus a measurement, so these tests read both as text and hold them
//! to the acceptance criteria of the issue, one test per criterion. They are std-only on
//! purpose: the claim they guard is "libviprs invents no transform rule and no threshold", and a
//! test that pulled in a curve crate to check it would be the wrong shape.
//!
//! The measurements come from `docs/spikes/transform-matrix/harness`, which pushes fixtures
//! through PVec's own transform entry point and writes `results-x64.tsv`. The document cites
//! that file by evidence id (`E1`, `E2`, ...), and the tests below fail when a cell and its
//! evidence stop agreeing.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn doc() -> String {
    let p = root().join("docs/spikes/transform-matrix.md");
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// One measured record, keyed by evidence id.
#[derive(Debug, Clone)]
struct Rec {
    fixture: String,
    transform: String,
    outcome: String,
    result: String,
    rule: String,
    worst_dev: String,
    ratio: String,
    notes: BTreeMap<String, String>,
}

fn results() -> BTreeMap<String, Rec> {
    let p = root().join("docs/spikes/transform-matrix/results-x64.tsv");
    let text = fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
    let mut lines = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<&str> = lines.next().expect("empty results file").split('\t').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .unwrap_or_else(|| panic!("results file has no column `{name}`"))
    };
    let (id, fx, tr, oc, re, ru, wd, ra, no) = (
        col("id"),
        col("fixture"),
        col("transform"),
        col("outcome"),
        col("result"),
        col("rule"),
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
                fixture: f[fx].to_string(),
                transform: f[tr].to_string(),
                outcome: f[oc].to_string(),
                result: f[re].to_string(),
                rule: f[ru].to_string(),
                worst_dev: f[wd].to_string(),
                ratio: f[ra].to_string(),
                notes,
            },
        );
    }
    out
}

/// The text of the section whose heading line starts with `heading` (for example `## The matrix`),
/// up to the next heading of the same or a higher level.
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

const COLUMNS: &[&str] = &[
    "translation",
    "rotation",
    "uniform scale",
    "non-uniform scale",
    "reflection",
    "shear / general affine",
    "singular",
];

/// The source shapes of the N1.2 ADR (NG-03). The matrix may add rows, never drop one.
const ROWS: &[&str] = &[
    "line",
    "polyline",
    "circular arc",
    "circle",
    "ellipse",
    "b-spline",
    "nurbs",
];

/// PVec semantic type names the cells may land on (ISO_PROFILE.md names, libviprs/pvec).
const PVEC_TYPES: &[&str] = &[
    "line",
    "polyline",
    "circle",
    "ellipse",
    "trimmed_curve over line",
    "trimmed_curve over circle",
    "trimmed_curve over ellipse",
    "trimmed_curve over b_spline_curve_with_knots",
    "composite_curve",
    "b_spline_curve_with_knots",
    "rational_b_spline_curve",
];

/// The six classifications of N3.5.
const N35: &[&str] = &[
    "SupportedExact",
    "SupportedEquivalent",
    "RequiresPVecCapability",
    "RequiresExtension",
    "Unsupported",
    "IgnoredByPolicy",
];

struct Cell {
    row: String,
    col: String,
    /// Backticked tokens, in order, with the backticks removed.
    tokens: Vec<String>,
    /// The evidence id the cell cites.
    evidence: String,
}

fn matrix_cells(doc: &str) -> Vec<Cell> {
    let sec = section(doc, "## The matrix");
    let rows = table_rows(&sec);
    assert!(!rows.is_empty(), "the matrix section holds no table");
    let header: Vec<String> = rows[0].iter().map(|c| c.to_lowercase()).collect();
    assert_eq!(
        header[1..].to_vec(),
        COLUMNS.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        "the matrix columns must be exactly the seven transform classes of the issue, in order"
    );
    let mut cells = Vec::new();
    for r in &rows[1..] {
        assert_eq!(r.len(), COLUMNS.len() + 1, "row `{}` has the wrong number of cells", r[0]);
        for (i, text) in r[1..].iter().enumerate() {
            let evidence = text.split_whitespace().last().unwrap_or("").to_string();
            let mut tokens = Vec::new();
            let mut rest = text.as_str();
            while let Some(a) = rest.find('`') {
                let Some(b) = rest[a + 1..].find('`') else { break };
                tokens.push(rest[a + 1..a + 1 + b].to_string());
                rest = &rest[a + b + 2..];
            }
            cells.push(Cell {
                row: r[0].to_lowercase(),
                col: COLUMNS[i].to_string(),
                tokens,
                evidence,
            });
        }
    }
    cells
}

fn is_evidence_id(s: &str) -> bool {
    s.len() > 1 && s.starts_with('E') && s[1..].chars().all(|c| c.is_ascii_digit())
}

// ---------------------------------------------------------------- AC 1

#[test]
fn ac1_matrix_has_every_row_and_every_cell() {
    let d = doc();
    let cells = matrix_cells(&d);
    for row in ROWS {
        for col in COLUMNS {
            let c = cells
                .iter()
                .find(|c| c.row == *row && c.col == *col)
                .unwrap_or_else(|| panic!("matrix has no cell for `{row}` x `{col}`"));
            assert!(!c.tokens.is_empty(), "cell `{row}` x `{col}` is empty or names nothing in backticks");
        }
    }
    // The rows are the source types of N1.2, so each must be a shape its mapping table names.
    let adr = fs::read_to_string(root().join("docs/adr/pvec-normalized-geometry.md")).expect("N1.2 ADR");
    for row in ROWS {
        assert!(
            adr.to_lowercase().contains(&format!("| {row} |")),
            "row `{row}` is not a source shape of the N1.2 mapping table"
        );
    }
}

#[test]
fn ac1_every_cell_names_a_pvec_type_or_an_n35_classification() {
    let d = doc();
    for c in matrix_cells(&d) {
        let head = &c.tokens[0];
        let ok = PVEC_TYPES.contains(&head.as_str()) || N35.contains(&head.as_str());
        assert!(
            ok,
            "cell `{}` x `{}` lands on `{head}`, which is neither a PVec semantic type nor an N3.5 classification",
            c.row, c.col
        );
        if matches!(head.as_str(), "Unsupported" | "RequiresPVecCapability") {
            assert!(
                c.tokens.len() >= 2 && c.tokens[1].chars().all(|ch| ch.is_ascii_uppercase() || ch == '_'),
                "cell `{}` x `{}` carries no stable reason code (N3.5 requires one)",
                c.row, c.col
            );
        }
        if head == "SupportedEquivalent" {
            assert!(
                c.tokens.len() >= 2,
                "cell `{}` x `{}` is SupportedEquivalent and must name the PVec entry point that makes it exact",
                c.row, c.col
            );
        }
    }
}

// ---------------------------------------------------------------- AC 2

#[test]
fn ac2_every_cell_cites_a_measurement_and_agrees_with_it() {
    let d = doc();
    let data = results();
    for c in matrix_cells(&d) {
        assert!(is_evidence_id(&c.evidence), "cell `{}` x `{}` cites no evidence id", c.row, c.col);
        let rec = data
            .get(&c.evidence)
            .unwrap_or_else(|| panic!("cell `{}` x `{}` cites {}, which is not in results-x64.tsv", c.row, c.col, c.evidence));
        if rec.outcome == "ok" {
            assert_eq!(
                c.tokens.join(" "),
                rec.result,
                "cell `{}` x `{}` says something other than what {} measured",
                c.row, c.col, c.evidence
            );
        } else {
            assert_eq!(rec.outcome, "refused", "{} has an outcome that is neither ok nor refused", c.evidence);
            assert_eq!(c.tokens[0], "Unsupported", "a refused measurement ({}) must be an Unsupported cell", c.evidence);
        }
    }
}

#[test]
fn ac2_evidence_table_cites_fixture_and_worst_deviation() {
    let d = doc();
    let data = results();
    let rows = table_rows(&section(&d, "## Evidence"));
    assert!(rows.len() > 1, "the Evidence section holds no table");
    let mut seen = Vec::new();
    for r in &rows[1..] {
        let rec = data
            .get(&r[0])
            .unwrap_or_else(|| panic!("evidence row {} is not in results-x64.tsv", r[0]));
        assert_eq!(r[1], rec.fixture, "{}: fixture differs from the measurement", r[0]);
        assert_eq!(r[2], rec.transform, "{}: transform differs from the measurement", r[0]);
        if rec.outcome == "ok" {
            assert_eq!(r[3], rec.worst_dev, "{}: worst deviation differs from the measurement", r[0]);
            assert_eq!(r[4], rec.ratio, "{}: deviation-to-tolerance ratio differs from the measurement", r[0]);
        }
        seen.push(r[0].clone());
    }
    for c in matrix_cells(&d) {
        assert!(seen.contains(&c.evidence), "{} is cited by a cell but missing from the Evidence table", c.evidence);
    }
}

#[test]
fn ac2_no_measured_cell_exceeds_the_p12_tolerance() {
    for (id, rec) in results() {
        if rec.outcome != "ok" {
            continue;
        }
        if rec.ratio == "-" {
            continue;
        }
        let ratio: f64 = rec.ratio.parse().unwrap_or_else(|_| panic!("{id}: ratio `{}` is not a number", rec.ratio));
        assert!(
            ratio <= 1.0,
            "{id} ({} under {}) deviates by {} of the NP-04 tolerance",
            rec.fixture, rec.transform, rec.ratio
        );
    }
}

#[test]
fn ac2_the_named_cases_are_answered_explicitly() {
    let d = doc();
    let data = results();
    let cells = matrix_cells(&d);
    let cell = |row: &str, col: &str| {
        cells.iter().find(|c| c.row == row && c.col == col).unwrap().clone_tokens()
    };
    // circle under non-uniform scale: an exact ellipse, not a capability gap.
    assert_eq!(cell("circle", "non-uniform scale").0, "ellipse");
    // arc under reflection keeps its end points in order and stays a trimmed circle.
    assert_eq!(cell("circular arc", "reflection").0, "trimmed_curve over circle");
    // ellipse under shear is an ellipse (PVec ADR 0012), not a rational B-spline.
    assert_eq!(cell("ellipse", "shear / general affine").0, "ellipse");
    // NURBS under a general affine: control points and weights move, knots do not.
    let (_, ev) = cell("nurbs", "shear / general affine");
    let rec = &data[&ev];
    assert_eq!(rec.result, "rational_b_spline_curve");
    assert_eq!(rec.notes.get("knots_identical").map(String::as_str), Some("true"), "{ev}: knot vector must be untouched");
    assert_eq!(rec.notes.get("weights_identical").map(String::as_str), Some("true"), "{ev}: weights must be preserved");
    assert!(rec.notes.contains_key("control_points_max_dev"), "{ev}: control points must be compared one by one");
    for heading in [
        "### Circle under non-uniform scale",
        "### Arc under reflection",
        "### Ellipse under shear",
        "### NURBS under a general affine",
        "### Singular and near-singular transforms",
    ] {
        assert!(d.contains(heading), "the spike has no section `{heading}`");
    }
}

trait CloneTokens {
    fn clone_tokens(&self) -> (String, String);
}
impl CloneTokens for Cell {
    fn clone_tokens(&self) -> (String, String) {
        (self.tokens.join(" "), self.evidence.clone())
    }
}

// ---------------------------------------------------------------- AC 3

#[test]
fn ac3_singular_column_is_a_refusal_with_a_named_diagnostic() {
    let d = doc();
    let data = results();
    let diag = section(&d, "## Diagnostics");
    let singular: Vec<Cell> = matrix_cells(&d).into_iter().filter(|c| c.col == "singular").collect();
    assert!(singular.len() >= ROWS.len(), "the singular column is not filled for every row");
    for c in &singular {
        assert_eq!(c.tokens[0], "Unsupported", "singular cell of `{}` is not a refusal", c.row);
        let code = &c.tokens[1];
        let rec = &data[&c.evidence];
        assert_eq!(rec.outcome, "refused", "{} measured an output for a singular transform", c.evidence);
        assert!(
            rec.notes.get("curve_emitted").map(String::as_str) == Some("false"),
            "{}: a refused transform must emit no curve",
            c.evidence
        );
        let line = diag
            .lines()
            .find(|l| l.contains(&format!("`{code}`")))
            .unwrap_or_else(|| panic!("diagnostic `{code}` is not defined in the Diagnostics section"));
        assert!(line.contains(&rec.rule), "diagnostic `{code}` does not cite {} as measured", rec.rule);
    }
    let sing = section(&d, "### Singular and near-singular transforms")
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    for needle in ["refus", "conversion report", "not a best-effort", "counted"] {
        assert!(sing.contains(needle), "the singular section never says `{needle}`");
    }
    assert!(sing.contains("`unsupported`") && sing.contains("`transform_singular`"));
}

// ---------------------------------------------------------------- AC 4

#[test]
fn ac4_near_singular_cites_the_conditioning_rule_and_defines_no_threshold() {
    let d = doc();
    let sec = section(&d, "### Singular and near-singular transforms");
    assert!(
        sec.contains("PVec numeric policy v1, NP-19"),
        "near-singular handling must cite `PVec numeric policy v1, NP-19`"
    );
    // A numeric literal in this section would be a local threshold. Names and rule ids only.
    for tok in sec.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_')) {
        let looks_float = tok.contains('.') && tok.chars().next().is_some_and(|c| c.is_ascii_digit());
        let exponent = tok.len() > 2
            && tok.chars().next().is_some_and(|c| c.is_ascii_digit())
            && tok.contains('e')
            && tok.chars().any(|c| c == '-');
        assert!(!looks_float && !exponent, "the near-singular section carries a numeric literal `{tok}`");
    }
    for word in ["epsilon", "our threshold", "we define", "i define"] {
        assert!(!sec.to_lowercase().contains(word), "the near-singular section says `{word}`");
    }
    // The measurement brackets the boundary using the policy's own constant, one side each.
    let data = results();
    let near: Vec<&Rec> = data.values().filter(|r| r.notes.get("band").map(String::as_str) == Some("near-singular")).collect();
    assert!(near.iter().any(|r| r.outcome == "refused" && r.rule == "NP-19"), "no near-singular map was refused under NP-19");
    assert!(near.iter().any(|r| r.outcome == "ok" || r.rule == "NP-07"), "no near-singular map was measured on the accepted side");
    assert!(
        near.iter().all(|r| r.notes.get("derived_from").map(String::as_str) == Some("Policy::V1.rel")),
        "near-singular fixtures must be derived from the policy constant, not typed in"
    );
}

#[test]
fn ac4_the_harness_defines_no_tolerance_of_its_own() {
    let dir = root().join("docs/spikes/transform-matrix/harness/src");
    let mut seen = 0;
    for entry in fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display())).flatten() {
        let p = entry.path();
        if p.extension().is_some_and(|e| e == "rs") {
            seen += 1;
            let text = fs::read_to_string(&p).unwrap();
            assert!(!text.contains("EPSILON"), "{} uses EPSILON", p.display());
            assert!(
                !text.contains("1e-") && !text.contains("1E-"),
                "{} carries an exponent-form literal, which is how a local tolerance starts",
                p.display()
            );
            assert!(text.contains("Policy::V1"), "{} never goes through PVec's policy", p.display());
        }
    }
    assert!(seen > 0, "the harness has no source file");
}

// ---------------------------------------------------------------- AC 5

#[test]
fn ac5_reflection_states_orientation_and_parameter_direction() {
    let d = doc();
    let sec = section(&d, "### Arc under reflection").to_lowercase();
    for needle in ["orientation", "parameter direction", "sweep", "endpoint order", "reversed"] {
        assert!(sec.contains(needle), "the reflection section never mentions `{needle}`");
    }
    let table = section(&d, "## Reflection rows");
    let rows = table_rows(&table);
    assert!(rows.len() > 1, "the Reflection rows section holds no table");
    let header: Vec<String> = rows[0].iter().map(|c| c.to_lowercase()).collect();
    for h in ["row", "orientation", "parameter direction"] {
        assert!(header.iter().any(|c| c == h), "the reflection table has no `{h}` column");
    }
    for row in ROWS {
        assert!(
            rows.iter().any(|r| r[0].to_lowercase() == *row && r.iter().all(|c| !c.is_empty())),
            "the reflection table has no complete row for `{row}`"
        );
    }
}

#[test]
fn ac5_a_mirrored_arc_is_reproducible_from_the_table_and_the_measurement() {
    let d = doc();
    let data = results();
    let cells = matrix_cells(&d);
    let c = cells.iter().find(|c| c.row == "circular arc" && c.col == "reflection").unwrap();
    let rec = &data[&c.evidence];
    for key in [
        "sweep_before",
        "sweep_after",
        "reversed_before",
        "reversed_after",
        "start_preserved",
        "end_preserved",
        "param_offset",
        "param_scale",
    ] {
        assert!(rec.notes.contains_key(key), "{}: the reflected arc records no `{key}`", c.evidence);
    }
    assert_eq!(rec.notes["start_preserved"], "true", "reflection must keep the arc's start point");
    assert_eq!(rec.notes["end_preserved"], "true", "reflection must keep the arc's end point");
    assert_ne!(rec.notes["sweep_before"], rec.notes["sweep_after"], "reflection must flip the sweep direction");
    let sec = section(&d, "### Arc under reflection");
    for key in ["sweep_before", "sweep_after", "reversed_before", "reversed_after"] {
        assert!(
            sec.contains(&format!("{}={}", key, rec.notes[key])),
            "the worked example does not quote the measured {key}={}",
            rec.notes[key]
        );
    }
}
