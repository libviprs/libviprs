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
    results_from("results-x64.tsv")
}

fn results_from(file: &str) -> BTreeMap<String, Rec> {
    let p = root().join("docs/spikes/transform-matrix").join(file);
    let text =
        fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
    let mut lines = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .expect("empty results file")
        .split('\t')
        .collect();
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
        .filter(|r| {
            !r.iter()
                .all(|c| c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' '))
        })
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
const ADR_ROWS: &[&str] = &[
    "line",
    "polyline",
    "circular arc",
    "circle",
    "ellipse",
    "b-spline",
    "nurbs",
];

/// Rows this spike adds because the ADR names the shapes in passing (a bulge polyline is a
/// composite, an elliptical arc is a trimmed ellipse). They are not dropped either.
const EXTRA_ROWS: &[&str] = &["polyline with bulge", "elliptical arc"];

/// Every row of the matrix, in the order of the document.
const ROWS: &[&str] = &[
    "line",
    "polyline",
    "polyline with bulge",
    "circular arc",
    "circle",
    "ellipse",
    "elliptical arc",
    "b-spline",
    "nurbs",
];

/// The fixture behind each row (the Fixtures table of the spike).
fn row_fixture(row: &str) -> &'static str {
    match row {
        "line" => "F1",
        "polyline" => "F2",
        "polyline with bulge" => "F3",
        "circular arc" => "F4",
        "circle" => "F5",
        "ellipse" => "F6",
        "elliptical arc" => "F7",
        "b-spline" => "F8",
        "nurbs" => "F9",
        other => panic!("no fixture is defined for row `{other}`"),
    }
}

/// The transform instance behind each column (the Transform instances table of the spike).
fn col_transform(col: &str) -> &'static str {
    match col {
        "translation" => "T1",
        "rotation" => "T2",
        "uniform scale" => "T3",
        "non-uniform scale" => "T4",
        "reflection" => "T5",
        "shear / general affine" => "T6",
        "singular" => "T7",
        other => panic!("no transform is defined for column `{other}`"),
    }
}

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
        assert_eq!(
            r.len(),
            COLUMNS.len() + 1,
            "row `{}` has the wrong number of cells",
            r[0]
        );
        for (i, text) in r[1..].iter().enumerate() {
            let evidence = text.split_whitespace().last().unwrap_or("").to_string();
            let mut tokens = Vec::new();
            let mut rest = text.as_str();
            while let Some(a) = rest.find('`') {
                let Some(b) = rest[a + 1..].find('`') else {
                    break;
                };
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
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac1_matrix_has_every_row_and_every_cell() {
    let d = doc();
    let cells = matrix_cells(&d);
    for row in ROWS {
        for col in COLUMNS {
            let c = cells
                .iter()
                .find(|c| c.row == *row && c.col == *col)
                .unwrap_or_else(|| panic!("matrix has no cell for `{row}` x `{col}`"));
            assert!(
                !c.tokens.is_empty(),
                "cell `{row}` x `{col}` is empty or names nothing in backticks"
            );
        }
    }
    // The rows are the source types of N1.2, so each must be a shape its mapping table names.
    // The two extra rows are named by the ADR in passing, and must be in the matrix anyway.
    let adr =
        fs::read_to_string(root().join("docs/adr/pvec-normalized-geometry.md")).expect("N1.2 ADR");
    assert!(
        ADR_ROWS.iter().chain(EXTRA_ROWS).all(|r| ROWS.contains(r)),
        "ROWS must hold every ADR row and every extra row"
    );
    for row in ADR_ROWS {
        assert!(
            adr.to_lowercase().contains(&format!("| {row} |")),
            "row `{row}` is not a source shape of the N1.2 mapping table"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
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
                c.tokens.len() >= 2
                    && c.tokens[1]
                        .chars()
                        .all(|ch| ch.is_ascii_uppercase() || ch == '_'),
                "cell `{}` x `{}` carries no stable reason code (N3.5 requires one)",
                c.row,
                c.col
            );
        }
        if head == "SupportedEquivalent" {
            assert!(
                c.tokens.len() >= 2,
                "cell `{}` x `{}` is SupportedEquivalent and must name the PVec entry point that makes it exact",
                c.row,
                c.col
            );
        }
    }
}

// ---------------------------------------------------------------- AC 2

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac2_every_cell_cites_a_measurement_and_agrees_with_it() {
    let d = doc();
    let data = results();
    for c in matrix_cells(&d) {
        assert!(
            is_evidence_id(&c.evidence),
            "cell `{}` x `{}` cites no evidence id",
            c.row,
            c.col
        );
        let rec = data.get(&c.evidence).unwrap_or_else(|| {
            panic!(
                "cell `{}` x `{}` cites {}, which is not in results-x64.tsv",
                c.row, c.col, c.evidence
            )
        });
        // The evidence must be the measurement of this very cell: the fixture of the row and the
        // transform of the column. A re-pointed id with the same result string must fail here.
        assert_eq!(
            rec.fixture,
            row_fixture(&c.row),
            "cell `{}` x `{}` cites {}, which measured fixture {} and not the fixture of its row",
            c.row,
            c.col,
            c.evidence,
            rec.fixture
        );
        assert_eq!(
            rec.transform,
            col_transform(&c.col),
            "cell `{}` x `{}` cites {}, which measured transform {} and not the transform of its column",
            c.row,
            c.col,
            c.evidence,
            rec.transform
        );
        if rec.outcome == "ok" {
            assert_eq!(
                c.tokens.join(" "),
                rec.result,
                "cell `{}` x `{}` says something other than what {} measured",
                c.row,
                c.col,
                c.evidence
            );
        } else {
            assert_eq!(
                rec.outcome, "refused",
                "{} has an outcome that is neither ok nor refused",
                c.evidence
            );
            assert_eq!(
                c.tokens[0], "Unsupported",
                "a refused measurement ({}) must be an Unsupported cell",
                c.evidence
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
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
        assert_eq!(
            r[1], rec.fixture,
            "{}: fixture differs from the measurement",
            r[0]
        );
        assert_eq!(
            r[2], rec.transform,
            "{}: transform differs from the measurement",
            r[0]
        );
        if rec.outcome == "ok" {
            assert_eq!(
                r[3], rec.worst_dev,
                "{}: worst deviation differs from the measurement",
                r[0]
            );
            assert_eq!(
                r[4], rec.ratio,
                "{}: deviation-to-tolerance ratio differs from the measurement",
                r[0]
            );
        }
        seen.push(r[0].clone());
    }
    for c in matrix_cells(&d) {
        assert!(
            seen.contains(&c.evidence),
            "{} is cited by a cell but missing from the Evidence table",
            c.evidence
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac2_no_measured_cell_exceeds_the_p12_tolerance() {
    for (id, rec) in results() {
        if rec.outcome != "ok" {
            continue;
        }
        if rec.ratio == "-" {
            continue;
        }
        let ratio: f64 = rec
            .ratio
            .parse()
            .unwrap_or_else(|_| panic!("{id}: ratio `{}` is not a number", rec.ratio));
        assert!(
            ratio <= 1.0,
            "{id} ({} under {}) deviates by {} of the NP-04 tolerance",
            rec.fixture,
            rec.transform,
            rec.ratio
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac2_the_named_cases_are_answered_explicitly() {
    let d = doc();
    let data = results();
    let cells = matrix_cells(&d);
    let cell = |row: &str, col: &str| {
        cells
            .iter()
            .find(|c| c.row == row && c.col == col)
            .unwrap()
            .clone_tokens()
    };
    // circle under non-uniform scale: an exact ellipse, not a capability gap.
    assert_eq!(cell("circle", "non-uniform scale").0, "ellipse");
    // arc under reflection keeps its end points in order and stays a trimmed circle.
    assert_eq!(
        cell("circular arc", "reflection").0,
        "trimmed_curve over circle"
    );
    // ellipse under shear is an ellipse (PVec ADR 0012), not a rational B-spline.
    assert_eq!(cell("ellipse", "shear / general affine").0, "ellipse");
    // NURBS under a general affine: control points and weights move, knots do not.
    let (_, ev) = cell("nurbs", "shear / general affine");
    let rec = &data[&ev];
    assert_eq!(rec.result, "rational_b_spline_curve");
    assert_eq!(
        rec.notes.get("knots_identical").map(String::as_str),
        Some("true"),
        "{ev}: knot vector must be untouched"
    );
    assert_eq!(
        rec.notes.get("weights_identical").map(String::as_str),
        Some("true"),
        "{ev}: weights must be preserved"
    );
    assert!(
        rec.notes.contains_key("control_points_max_dev"),
        "{ev}: control points must be compared one by one"
    );
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
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac3_singular_column_is_a_refusal_with_a_named_diagnostic() {
    let d = doc();
    let data = results();
    let diag = section(&d, "## Diagnostics");
    let singular: Vec<Cell> = matrix_cells(&d)
        .into_iter()
        .filter(|c| c.col == "singular")
        .collect();
    assert!(
        singular.len() >= ROWS.len(),
        "the singular column is not filled for every row"
    );
    for c in &singular {
        assert_eq!(
            c.tokens[0], "Unsupported",
            "singular cell of `{}` is not a refusal",
            c.row
        );
        let code = &c.tokens[1];
        let rec = &data[&c.evidence];
        assert_eq!(
            rec.outcome, "refused",
            "{} measured an output for a singular transform",
            c.evidence
        );
        assert!(
            rec.notes.get("curve_emitted").map(String::as_str) == Some("false"),
            "{}: a refused transform must emit no curve",
            c.evidence
        );
        let line = diag
            .lines()
            .find(|l| l.contains(&format!("`{code}`")))
            .unwrap_or_else(|| {
                panic!("diagnostic `{code}` is not defined in the Diagnostics section")
            });
        assert!(
            line.contains(&rec.rule),
            "diagnostic `{code}` does not cite {} as measured",
            rec.rule
        );
    }
    let sing = section(&d, "### Singular and near-singular transforms")
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    for needle in ["refus", "conversion report", "not a best-effort", "counted"] {
        assert!(
            sing.contains(needle),
            "the singular section never says `{needle}`"
        );
    }
    assert!(sing.contains("`unsupported`") && sing.contains("`transform_singular`"));
}

// ---------------------------------------------------------------- AC 4

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac4_near_singular_cites_the_conditioning_rule_and_defines_no_threshold() {
    let d = doc();
    let sec = section(&d, "### Singular and near-singular transforms");
    assert!(
        sec.contains("PVec numeric policy v1, NP-19"),
        "near-singular handling must cite `PVec numeric policy v1, NP-19`"
    );
    // A numeric literal in this section would be a local threshold. Names and rule ids only.
    for tok in sec.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_'))
    {
        let looks_float =
            tok.contains('.') && tok.chars().next().is_some_and(|c| c.is_ascii_digit());
        let exponent = tok.len() > 2
            && tok.chars().next().is_some_and(|c| c.is_ascii_digit())
            && tok.contains('e')
            && tok.chars().any(|c| c == '-');
        assert!(
            !looks_float && !exponent,
            "the near-singular section carries a numeric literal `{tok}`"
        );
    }
    for word in ["epsilon", "our threshold", "we define", "i define"] {
        assert!(
            !sec.to_lowercase().contains(word),
            "the near-singular section says `{word}`"
        );
    }
    // The measurement brackets the boundary using the policy's own constant, one side each.
    let data = results();
    let near: Vec<&Rec> = data
        .values()
        .filter(|r| r.notes.get("band").map(String::as_str) == Some("near-singular"))
        .collect();
    assert!(
        near.iter()
            .any(|r| r.outcome == "refused" && r.rule == "NP-19"),
        "no near-singular map was refused under NP-19"
    );
    assert!(
        near.iter().any(|r| r.outcome == "ok" || r.rule == "NP-07"),
        "no near-singular map was measured on the accepted side"
    );
    assert!(
        near.iter()
            .all(|r| r.notes.get("derived_from").map(String::as_str) == Some("Policy::V1.rel")),
        "near-singular fixtures must be derived from the policy constant, not typed in"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac4_the_harness_defines_no_tolerance_of_its_own() {
    let dir = root().join("docs/spikes/transform-matrix/harness/src");
    let mut seen = 0;
    for entry in fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .flatten()
    {
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
            assert!(
                text.contains("Policy::V1"),
                "{} never goes through PVec's policy",
                p.display()
            );
        }
    }
    assert!(seen > 0, "the harness has no source file");
}

// ---------------------------------------------------------------- AC 5

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac5_reflection_states_orientation_and_parameter_direction() {
    let d = doc();
    let sec = section(&d, "### Arc under reflection").to_lowercase();
    for needle in [
        "orientation",
        "parameter direction",
        "sweep",
        "endpoint order",
        "reversed",
    ] {
        assert!(
            sec.contains(needle),
            "the reflection section never mentions `{needle}`"
        );
    }
    let table = section(&d, "## Reflection rows");
    let rows = table_rows(&table);
    assert!(rows.len() > 1, "the Reflection rows section holds no table");
    let header: Vec<String> = rows[0].iter().map(|c| c.to_lowercase()).collect();
    for h in ["row", "orientation", "parameter direction"] {
        assert!(
            header.iter().any(|c| c == h),
            "the reflection table has no `{h}` column"
        );
    }
    for row in ROWS {
        assert!(
            rows.iter()
                .any(|r| r[0].to_lowercase() == *row && r.iter().all(|c| !c.is_empty())),
            "the reflection table has no complete row for `{row}`"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac5_a_mirrored_arc_is_reproducible_from_the_table_and_the_measurement() {
    let d = doc();
    let data = results();
    let cells = matrix_cells(&d);
    let c = cells
        .iter()
        .find(|c| c.row == "circular arc" && c.col == "reflection")
        .unwrap();
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
        assert!(
            rec.notes.contains_key(key),
            "{}: the reflected arc records no `{key}`",
            c.evidence
        );
    }
    assert_eq!(
        rec.notes["start_preserved"], "true",
        "reflection must keep the arc's start point"
    );
    assert_eq!(
        rec.notes["end_preserved"], "true",
        "reflection must keep the arc's end point"
    );
    assert_ne!(
        rec.notes["sweep_before"], rec.notes["sweep_after"],
        "reflection must flip the sweep direction"
    );
    let sec = section(&d, "### Arc under reflection");
    for key in [
        "sweep_before",
        "sweep_after",
        "reversed_before",
        "reversed_after",
    ] {
        assert!(
            sec.contains(&format!("{}={}", key, rec.notes[key])),
            "the worked example does not quote the measured {key}={}",
            rec.notes[key]
        );
    }
}

// ---------------------------------------------------------------- review follow-ups
//
// Everything below ties prose in the spike to the measurement file, so a sentence that drifts from
// the numbers goes red instead of staying quietly wrong.

/// Evidence ids cited in `text`, with "E10 to E12" expanded to the three ids it stands for.
fn cited_ids(text: &str) -> Vec<String> {
    let toks: Vec<&str> = text
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if is_evidence_id(toks[i]) {
            if i + 2 < toks.len() && toks[i + 1] == "to" && is_evidence_id(toks[i + 2]) {
                let a: usize = toks[i][1..].parse().unwrap();
                let b: usize = toks[i + 2][1..].parse().unwrap();
                for n in a..=b {
                    out.push(format!("E{n}"));
                }
                i += 3;
                continue;
            }
            out.push(toks[i].to_string());
        }
        i += 1;
    }
    out
}

fn flat(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn note<'a>(data: &'a BTreeMap<String, Rec>, id: &str, key: &str) -> &'a str {
    data.get(id)
        .unwrap_or_else(|| panic!("{id} is not in the results"))
        .notes
        .get(key)
        .unwrap_or_else(|| panic!("{id} records no `{key}`"))
}

fn find<'a>(
    data: &'a BTreeMap<String, Rec>,
    fixture: &str,
    transform: &str,
) -> (&'a String, &'a Rec) {
    data.iter()
        .find(|(_, r)| r.fixture == fixture && r.transform == transform)
        .unwrap_or_else(|| panic!("no record for {fixture} under {transform}"))
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac2_findings_prose_quotes_the_measured_numbers() {
    let d = doc();
    let data = results();

    let circle = flat(&section(&d, "### Circle under non-uniform scale"));
    for (id, keys) in [
        ("E32", vec!["semi_axes_after", "x_axis_after"]),
        ("E50", vec!["param_offset"]),
    ] {
        for k in keys {
            let v = note(&data, id, k);
            assert!(
                circle.contains(v),
                "circle section: {id} measured {k}={v}, which the prose does not quote"
            );
        }
    }
    let e32 = &data["E32"];
    assert!(
        circle.contains(&format!(
            "{} ({} of the NP-04 tolerance)",
            e32.worst_dev, e32.ratio
        )),
        "circle section: E32 deviation and ratio differ from the measurement"
    );
    assert_eq!(data["E31"].result, "trimmed_curve over ellipse");
    assert!(circle.contains("`trimmed_curve over ellipse` (E31)"));

    let arc = flat(&section(&d, "### Arc under reflection"));
    for k in [
        "centre_after",
        "radius_after",
        "x_axis_after",
        "trim_start_after",
        "trim_span_after",
    ] {
        let v = note(&data, "E40", k);
        assert!(
            arc.contains(v),
            "arc section: E40 measured {k}={v}, which the prose does not quote"
        );
    }
    // The by-hand derivation: trim start is 2 pi minus the larger old trim end, the span is unchanged.
    let by_hand = format!("{:.9}", std::f64::consts::TAU - 2.1);
    assert_eq!(
        by_hand,
        note(&data, "E40", "trim_start_after"),
        "2 pi minus 2.1 is no longer the measured trim start"
    );
    assert_eq!(
        note(&data, "E40", "trim_span_after"),
        note(&data, "E40", "trim_span_before")
    );

    let ell = flat(&section(&d, "### Ellipse under shear"));
    let e69 = &data["E69"];
    for k in ["semi_axes_after", "x_axis_after"] {
        let v = note(&data, "E69", k);
        assert!(
            ell.contains(v),
            "ellipse section: E69 measured {k}={v}, which the prose does not quote"
        );
    }
    assert!(
        ell.contains(&format!(
            "{} ({} of the tolerance)",
            e69.worst_dev, e69.ratio
        )),
        "ellipse section: E69 deviation and ratio differ from the measurement"
    );
    assert_eq!(data["E69"].result, "ellipse");
    for id in ["E102", "E103", "E104", "E105"] {
        assert_eq!(
            note(&data, id, "control_points"),
            "9",
            "{id}: a full conic's rational form is nine control points"
        );
    }
    assert!(ell.contains("nine control points"));

    let n = flat(&section(&d, "### NURBS under a general affine"));
    let e54 = &data["E54"];
    assert!(
        n.contains(&format!(
            "{} ({} of the tolerance)",
            e54.worst_dev, e54.ratio
        )),
        "NURBS section: E54 deviation and ratio differ from the measurement"
    );
    assert!(
        n.contains(&format!(
            "weights_identical={}",
            note(&data, "E54", "weights_identical")
        )),
        "NURBS section: weights_identical differs from the measurement"
    );
    assert!(
        n.contains(&format!(
            "knots_identical={}",
            note(&data, "E54", "knots_identical")
        )),
        "NURBS section: knots_identical differs from the measurement"
    );
    assert_eq!(note(&data, "E54", "knots_identical"), "true");
    assert!(
        n.contains(note(&data, "E54", "control_points_max_dev")),
        "NURBS section: control point difference differs from the measurement"
    );
    assert!(
        n.contains("not touched"),
        "NURBS section no longer says the knot vector is untouched"
    );
    assert!(
        !n.contains("rescaled"),
        "NURBS section says the knots are rescaled"
    );

    // The id ranges of the singular section are what they claim to be.
    let sing = flat(&section(&d, "### Singular and near-singular transforms"));
    for (range, tr) in [("E55 to E63", "T7"), ("E73 to E81", "T9")] {
        assert!(
            sing.contains(range),
            "singular section does not cite {range}"
        );
        for id in cited_ids(range) {
            let r = &data[&id];
            assert_eq!(
                (r.transform.as_str(), r.outcome.as_str(), r.rule.as_str()),
                (tr, "refused", "NP-19"),
                "{id} is not a {tr} refusal under NP-19"
            );
        }
    }
    assert!(sing.contains("E82 to E90"));
    for id in cited_ids("E82 to E90") {
        let r = &data[&id];
        assert_eq!(r.transform, "T10");
        assert_ne!(
            r.rule, "NP-19",
            "{id}: the accepted-side map was refused by NP-19"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac2_largest_deviation_and_ratio_sentences_match_the_results() {
    let d = flat(&section(&doc(), "## Evidence"));
    let data = results();
    let pick = |f: &dyn Fn(&Rec) -> &str| {
        data.iter()
            .filter(|(_, r)| r.outcome == "ok" && f(r) != "-")
            .map(|(id, r)| (f(r).parse::<f64>().unwrap(), id.clone(), f(r).to_string()))
            .fold(None::<(f64, String, String)>, |best, c| match best {
                Some(b) if b.0 >= c.0 => Some(b),
                _ => Some(c),
            })
            .unwrap()
    };
    let (_, id, text) = pick(&|r| r.worst_dev.as_str());
    assert!(
        d.contains(&format!(
            "largest deviation of any accepted case is {text} ({id})"
        )),
        "the largest deviation is {text} ({id})"
    );
    let (_, id, text) = pick(&|r| r.ratio.as_str());
    assert!(
        d.contains(&format!("largest ratio is {text} ({id})")),
        "the largest ratio is {text} ({id})"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac2_every_fixture_and_transform_in_the_results_is_defined_in_the_doc() {
    let d = doc();
    for r in results().values() {
        if r.fixture != "-" {
            assert!(
                d.contains(&format!("| {} |", r.fixture)),
                "fixture {} is measured but not defined in the Fixtures table",
                r.fixture
            );
        }
        if r.transform.starts_with('T') {
            assert!(
                d.contains(&format!("| {} |", r.transform)),
                "transform {} is measured but not defined in the Transform instances table",
                r.transform
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac2_bulge_row_shows_what_its_segments_became() {
    let d = doc();
    let data = results();
    let line = "trimmed_curve over line";
    for c in matrix_cells(&d)
        .into_iter()
        .filter(|c| c.row == "polyline with bulge" && c.col != "singular")
    {
        let after = note(&data, &c.evidence, "segments_after");
        let arc = if matches!(
            c.col.as_str(),
            "non-uniform scale" | "shear / general affine"
        ) {
            "trimmed_curve over ellipse"
        } else {
            "trimmed_curve over circle"
        };
        assert_eq!(
            after,
            format!("{line}+{arc}"),
            "{}: the segments of the bulge composite under `{}`",
            c.evidence,
            c.col
        );
        assert_eq!(
            note(&data, &c.evidence, "segments_before"),
            format!("{line}+trimmed_curve over circle")
        );
        assert_eq!(
            note(&data, &c.evidence, "joints_kept"),
            "true",
            "{}: the joints of the composite are not kept",
            c.evidence
        );
    }
    let sec = flat(&section(&d, "### Bulge composite under non-uniform scale"));
    for id in ["E30", "E48"] {
        assert!(
            sec.contains(&id.to_string()),
            "the bulge section does not cite {id}"
        );
        assert!(
            sec.contains(note(&data, id, "segments_after")),
            "the bulge section does not quote the measured segments of {id}"
        );
    }
    assert!(sec.contains("joints_kept=true"));
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac5_reflection_rows_agree_with_their_records() {
    let d = doc();
    let data = results();
    let rows = table_rows(&section(&d, "## Reflection rows"));
    let flips = |a: &str, b: &str| matches!((a, b), ("ccw", "cw") | ("cw", "ccw"));
    for r in &rows[1..] {
        let row = r[0].to_lowercase();
        let (orientation, param, evidence) = (r[1].as_str(), r[2].as_str(), r[3].as_str());
        let ids = cited_ids(evidence);
        assert!(!ids.is_empty(), "reflection row `{row}` cites no evidence");
        let mut by_fx: BTreeMap<&str, &String> = BTreeMap::new();
        for id in &ids {
            let rec = data.get(id).unwrap_or_else(|| {
                panic!("reflection row `{row}` cites {id}, which is not measured")
            });
            assert_eq!(
                rec.transform, "T5",
                "reflection row `{row}` cites {id}, which is not a reflection"
            );
            by_fx.entry(rec.fixture.as_str()).or_insert(id);
        }
        let own = row_fixture_or_ring(&row);
        let main = *by_fx
            .get(own)
            .unwrap_or_else(|| panic!("reflection row `{row}` does not cite a record of {own}"));
        let n = |k: &str| note(&data, main, k).to_string();
        match row.as_str() {
            "line" => {
                assert_eq!(
                    (n("sweep_before"), n("sweep_after")),
                    ("straight".into(), "straight".into())
                );
                assert_eq!(
                    (n("reversed_before"), n("reversed_after")),
                    ("false".into(), "false".into())
                );
                assert_eq!(n("param_scale"), "1.000000000");
                assert!(
                    orientation.contains("nothing turns")
                        && param.contains("reversed flag stays false")
                );
            }
            "polyline" => {
                assert!(
                    flips(&n("ring_winding_before"), &n("ring_winding_after")),
                    "{main}: the ring's winding does not flip"
                );
                assert!(
                    orientation.contains(&format!(
                        "{} to {}",
                        n("ring_winding_before"),
                        n("ring_winding_after")
                    )),
                    "polyline row does not quote the measured winding"
                );
                assert_eq!(n("param_scale"), "1.000000000");
                let open = by_fx.get("F2").unwrap_or_else(|| {
                    panic!("polyline row cites no record of the open polyline F2")
                });
                assert_eq!(note(&data, open, "param_scale"), "1.000000000");
                assert!(param.contains("unchanged"));
            }
            "polyline with bulge" => {
                let (b, a) = (n("segment_sweeps_before"), n("segment_sweeps_after"));
                let (b, a): (Vec<&str>, Vec<&str>) =
                    (b.split('+').collect(), a.split('+').collect());
                assert_eq!(b.len(), a.len(), "{main}: the segment count changed");
                assert!(
                    b.iter().zip(&a).any(|(x, y)| flips(x, y)),
                    "{main}: no bulge arc flips"
                );
                assert!(
                    b.iter().zip(&a).all(|(x, y)| flips(x, y) || x == y),
                    "{main}: a segment changed its turning without flipping"
                );
                assert_eq!(n("joints_kept"), "true");
                assert_eq!(n("param_scale"), "1.000000000");
                assert!(orientation.contains("other way") && param.contains("unchanged"));
            }
            "circular arc" | "elliptical arc" => {
                let (sb, sa) = (n("sweep_before"), n("sweep_after"));
                assert!(flips(&sb, &sa), "{main}: the sweep does not flip");
                assert!(
                    orientation.contains(&format!("{sb} to {sa}")),
                    "{row}: row does not quote the measured sweep {sb} to {sa}"
                );
                assert_eq!(
                    (n("reversed_before"), n("reversed_after")),
                    ("false".into(), "true".into())
                );
                assert_eq!(
                    (n("start_preserved"), n("end_preserved")),
                    ("true".into(), "true".into())
                );
                assert_eq!(
                    n("param_scale"),
                    "1.000000000",
                    "{main}: the trimmed parameter is not rescaled"
                );
                // The basis reversal is only visible on the untrimmed conic, so the row must cite it.
                let basis = if row == "circular arc" { "F5" } else { "F6" };
                let bid = by_fx.get(basis).unwrap_or_else(|| {
                    panic!("{row}: the basis parameter reversal needs a record of {basis}")
                });
                assert_eq!(note(&data, bid, "param_scale"), "-1.000000000");
                assert!(param.contains("basis"));
            }
            "circle" | "ellipse" => {
                assert_eq!(
                    n("frame_after"),
                    "right",
                    "{main}: the new frame is not right-handed"
                );
                assert_eq!(n("param_scale"), "-1.000000000");
                assert!(orientation.contains("right-handed") && param.contains("minus one"));
                // The new curve runs the same way in its own parameter, and the parameter map is
                // reversed, so in source parameter order the traversal turns the other way.
                let (sb, sa) = (n("sweep_before"), n("sweep_after"));
                assert_eq!(
                    sb, sa,
                    "{main}: the re-expressed conic should keep its own turning"
                );
                assert!(
                    flips(&sb, if sa == "ccw" { "cw" } else { "ccw" }),
                    "{main}: no turning to flip"
                );
                let in_source_order = if sa == "ccw" { "cw" } else { "ccw" };
                assert!(
                    orientation.contains(&format!("{sb} to {in_source_order} in source order")),
                    "{row}: row does not quote the measured turning {sb} to {in_source_order} in source order"
                );
                if row == "ellipse" {
                    let ax = n("semi_axes_after");
                    let v: Vec<f64> = ax
                        .trim_matches(|c| c == '(' || c == ')')
                        .split(',')
                        .map(|x| x.parse().unwrap())
                        .collect();
                    assert!(v[0] >= v[1], "{main}: the major axis is not first");
                    assert!(orientation.contains("major first"));
                }
            }
            "b-spline" | "nurbs" => {
                let (sb, sa) = (n("sweep_before"), n("sweep_after"));
                assert!(flips(&sb, &sa), "{main}: the turning does not flip");
                assert!(
                    orientation.contains(&format!("{sb} to {sa}")),
                    "{row}: row does not quote the measured turning {sb} to {sa}"
                );
                assert_eq!(n("knots_identical"), "true");
                assert_eq!(n("control_points_max_dev"), "0.000e0");
                assert_eq!(n("param_scale"), "1.000000000");
                if row == "nurbs" {
                    assert_eq!(n("weights_identical"), "true");
                }
                assert!(param.contains("unchanged"));
            }
            other => panic!("reflection row `{other}` has no check"),
        }
    }
}

/// The fixture a reflection row's own measurement comes from: the closed ring for the polyline.
fn row_fixture_or_ring(row: &str) -> &'static str {
    if row == "polyline" {
        "F10"
    } else {
        row_fixture(row)
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac3_every_diagnostic_rule_is_the_rule_of_the_records_it_cites() {
    let d = doc();
    let data = results();
    let rows = table_rows(&section(&d, "## Diagnostics"));
    let codes: Vec<&str> = rows[1..].iter().map(|r| r[0].trim_matches('`')).collect();
    assert_eq!(
        codes,
        [
            "TRANSFORM_SINGULAR",
            "TRANSFORM_CONIC_ILL_CONDITIONED",
            "TRANSFORM_NON_FINITE"
        ]
    );
    for r in &rows[1..] {
        let (code, rule, when) = (r[0].trim_matches('`'), r[2].as_str(), r[3].as_str());
        let ids = cited_ids(when);
        assert!(!ids.is_empty(), "diagnostic `{code}` cites no evidence");
        for id in ids {
            let rec = data
                .get(&id)
                .unwrap_or_else(|| panic!("diagnostic `{code}` cites {id}, which is not measured"));
            assert_eq!(
                rec.outcome, "refused",
                "diagnostic `{code}` cites {id}, which was not refused"
            );
            assert_eq!(
                rec.rule, rule,
                "diagnostic `{code}` says {rule}, but {id} was refused under {}",
                rec.rule
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac4_the_conic_conditioning_exit_is_bracketed_on_both_sides() {
    let d = doc();
    let data = results();
    let sec = flat(&section(&d, "### Conditioning exit"));
    assert!(
        !sec.contains("exact for every map NP-07 allows"),
        "the conditioning exit still claims exactness for every allowed map"
    );
    assert!(
        sec.contains("bracket"),
        "the conditioning exit does not say it brackets the bound"
    );
    for t in ["T12", "T13"] {
        assert!(sec.contains(t), "the conditioning exit does not name {t}");
    }
    let conic_or_bulge = ["F3", "F4", "F5", "F6", "F7"];
    for fx in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"] {
        let (idn, near) = find(&data, fx, "T12");
        assert_eq!(
            near.outcome, "ok",
            "{idn}: a conic just inside the NP-07 bound must be accepted"
        );
        assert_eq!(
            near.notes.get("derived_from").map(String::as_str),
            Some("Policy::V1.axis_ratio_max"),
            "{idn}"
        );
        let (idr, far) = find(&data, fx, "T13");
        if conic_or_bulge.contains(&fx) {
            assert_eq!(
                (far.outcome.as_str(), far.rule.as_str()),
                ("refused", "NP-07"),
                "{idr}: a conic just outside the bound must be refused under NP-07"
            );
        } else {
            assert_eq!(
                far.outcome, "ok",
                "{idr}: a hull that goes linear is a legal curve"
            );
        }
        assert_eq!(
            far.notes.get("derived_from").map(String::as_str),
            Some("Policy::V1.axis_ratio_max")
        );
        let cited = cited_ids(&sec);
        for id in [idn, idr] {
            if conic_or_bulge.contains(&fx) {
                assert!(
                    cited.contains(id),
                    "the conditioning exit does not cite {id}"
                );
            }
        }
    }
    let harness =
        fs::read_to_string(root().join("docs/spikes/transform-matrix/harness/src/main.rs"))
            .unwrap();
    assert!(
        harness.contains("axis_ratio_max"),
        "the harness does not derive the conditioning maps from the policy constant"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac4_cell_type_strings_are_said_to_come_from_the_harness_not_the_registry() {
    let d = flat(&doc());
    assert!(
        d.contains("kind_name()"),
        "the spike never says the cell type strings come from the harness's kind_name()"
    );
    assert!(
        d.contains("capability registry"),
        "the spike never says these are not PVec's capability registry names"
    );
    // And the mapping the doc gives has to be the harness's, so a renamed arm cannot drift.
    let h = fs::read_to_string(root().join("docs/spikes/transform-matrix/harness/src/main.rs"))
        .unwrap();
    for t in [
        "\"b_spline_curve_with_knots\"",
        "\"rational_b_spline_curve\"",
        "\"composite_curve\"",
    ] {
        assert!(h.contains(t), "kind_name() no longer returns {t}");
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the spike document and its results from disk, blocked by Miri isolation
fn ac5_arm64_results_are_recorded_and_the_verdicts_are_architecture_independent() {
    let x = results();
    let a = results_from("results-arm64.tsv");
    assert_eq!(
        x.keys().collect::<Vec<_>>(),
        a.keys().collect::<Vec<_>>(),
        "arm64 and x64 measured different records"
    );
    let mut differing = 0;
    for (id, rx) in &x {
        let ra = &a[id];
        assert_eq!(
            (
                &rx.fixture,
                &rx.transform,
                &rx.outcome,
                &rx.result,
                &rx.rule
            ),
            (
                &ra.fixture,
                &ra.transform,
                &ra.outcome,
                &ra.result,
                &ra.rule
            ),
            "{id}: the verdict differs between x64 and arm64"
        );
        assert_eq!(
            rx.notes.len(),
            ra.notes.len(),
            "{id}: different notes on arm64"
        );
        for (k, v) in &rx.notes {
            if k == "control_points_max_dev" {
                continue;
            }
            assert_eq!(
                Some(v),
                ra.notes.get(k),
                "{id}: note `{k}` differs between x64 and arm64"
            );
        }
        if ra.ratio != "-" {
            let r: f64 = ra.ratio.parse().unwrap();
            assert!(
                r <= 1.0,
                "{id}: arm64 deviates by {} of the tolerance",
                ra.ratio
            );
        }
        if rx.worst_dev != ra.worst_dev || rx.ratio != ra.ratio {
            differing += 1;
        }
    }
    let d = flat(&doc());
    assert!(
        !d.contains("arm64 not run")
            && !d.contains("I did not run arm64")
            && !d.contains("Only native x86_64 was run"),
        "the arm64 caveat is still in the spike"
    );
    assert!(
        d.contains("architecture independent"),
        "the spike does not say the verdicts are architecture independent"
    );
    assert!(
        d.contains(&format!(
            "last digits of the deviation and the ratio differ in {differing} of the {} records",
            x.len()
        )),
        "the spike does not say the last digits differ in {differing} of the {} records",
        x.len()
    );
    let max = |m: &BTreeMap<String, Rec>, f: &dyn Fn(&Rec) -> String| {
        m.iter()
            .filter(|(_, r)| r.outcome == "ok" && f(r) != "-")
            .map(|(id, r)| (f(r).parse::<f64>().unwrap(), id.clone(), f(r)))
            .fold(None::<(f64, String, String)>, |b, c| match b {
                Some(b) if b.0 >= c.0 => Some(b),
                _ => Some(c),
            })
            .unwrap()
    };
    let (_, id, text) = max(&a, &|r| r.ratio.clone());
    assert!(
        d.contains(&format!("largest arm64 ratio is {text} ({id})")),
        "the largest arm64 ratio is {text} ({id})"
    );
    let (_, id, text) = max(&a, &|r| r.worst_dev.clone());
    assert!(
        d.contains(&format!("largest arm64 deviation is {text} ({id})")),
        "the largest arm64 deviation is {text} ({id})"
    );
}
