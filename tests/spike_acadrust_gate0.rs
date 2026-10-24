//! Gate 0 for the acadrust provider (N4.1, #1072), checked as text.
//!
//! The decision on #1239 is that the first provider reads through acadrust, on one
//! condition: an open failure on an AutoCAD-authored file flips it back to the
//! existing decoder, with the traits unchanged. This file does not run acadrust. It
//! reads the table the run wrote (`docs/spikes/acadrust-gate0/results-x64.tsv`) and
//! the document that quotes it, and fails when
//!
//! * a corpus file has no row, or has two;
//! * a row's columns are malformed;
//! * an AutoCAD-authored file is `OPEN_FAILED`, in which case the message says STOP
//!   and the PR does not leave draft;
//! * the document's table and the TSV disagree, or its verdict does not match.
//!
//! Std only, no feature, no network.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

fn read(rel: &str) -> String {
    fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel))
        .unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

/// The AutoCAD-authored Tier A corpus of ACAD-ENGINE-DECISION-PLAN.md, from ACadSharp
/// `samples/` at d7dc111023477d8a9fffc2153139459c95b4f345 (MIT). The plan says
/// "eleven" dynamic-block files and 23 in all; that commit holds ten and 22.
const TIER_A: [&str; 22] = [
    "sample_AC1014.dwg",
    "sample_AC1015.dwg",
    "sample_AC1018.dwg",
    "sample_AC1021.dwg",
    "sample_AC1024.dwg",
    "sample_AC1027.dwg",
    "sample_AC1032.dwg",
    "sample_base.dwg",
    "empty.dwg",
    "BLOCKALIGNMENTPARAMETER.dwg",
    "BLOCKBASEPOINTPARAMETER.dwg",
    "BLOCKFLIPPARAMETER.dwg",
    "BLOCKLINEARPARAMETER.dwg",
    "BLOCKLOOKUPPARAMETER.dwg",
    "BLOCKPOINTPARAMETER.dwg",
    "BLOCKPOLARPARAMETER.dwg",
    "BLOCKROTATIONPARAMETER.dwg",
    "BLOCKVISIBILITYPARAMETER.dwg",
    "BLOCKXYPARAMETER.dwg",
    "AecObjects.dwg",
    "geoloc.dwg",
    "export_sample.dwg",
];

const COLUMNS: [&str; 15] = [
    "file",
    "tier",
    "source",
    "sha256",
    "version",
    "open",
    "layers",
    "layouts",
    "blocks",
    "model_entities",
    "entities_all_scopes",
    "unmapped",
    "unmapped_classes",
    "reread_identical",
    "error",
];

struct Row {
    cols: BTreeMap<&'static str, String>,
}

fn rows() -> (String, Vec<Row>) {
    let text = read("docs/spikes/acadrust-gate0/results-x64.tsv");
    let mut header_rev = String::new();
    let mut out = Vec::new();
    let mut seen_header = false;
    for line in text.lines() {
        if let Some(c) = line.strip_prefix("# acadrust ") {
            header_rev = c.split_whitespace().next().unwrap_or("").to_string();
            continue;
        }
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split('\t').collect();
        if !seen_header {
            assert_eq!(cells, COLUMNS, "header row");
            seen_header = true;
            continue;
        }
        assert_eq!(cells.len(), COLUMNS.len(), "malformed row: {line}");
        let mut cols = BTreeMap::new();
        for (name, v) in COLUMNS.iter().zip(&cells) {
            cols.insert(*name, (*v).to_string());
        }
        out.push(Row { cols });
    }
    assert!(seen_header, "no header row");
    (header_rev, out)
}

impl Row {
    fn get(&self, k: &str) -> &str {
        self.cols.get(k).unwrap_or_else(|| panic!("column {k}"))
    }
}

#[test]
fn every_tier_a_file_has_exactly_one_row() {
    let (_, rows) = rows();
    for f in TIER_A {
        let n = rows
            .iter()
            .filter(|r| r.get("file") == f && r.get("tier") == "A")
            .count();
        assert_eq!(n, 1, "{f} has {n} Tier A rows");
    }
    let tier_a = rows.iter().filter(|r| r.get("tier") == "A").count();
    assert_eq!(tier_a, TIER_A.len(), "unexpected extra Tier A rows");
}

#[test]
fn rows_are_well_formed() {
    let (rev, rows) = rows();
    assert_eq!(
        rev.len(),
        40,
        "the header names the acadrust revision it ran against"
    );
    for r in &rows {
        let f = r.get("file");
        assert!(["A", "B", "S"].contains(&r.get("tier")), "{f}: tier");
        let sha = r.get("sha256");
        assert!(
            sha.len() == 64 && sha.chars().all(|c| c.is_ascii_hexdigit()),
            "{f}: sha256"
        );
        let open = r.get("open");
        assert!(
            ["OPEN_OK", "OPEN_FAILED"].contains(&open),
            "{f}: open = {open}"
        );
        if open == "OPEN_OK" {
            for c in [
                "layers",
                "layouts",
                "blocks",
                "model_entities",
                "entities_all_scopes",
                "unmapped",
            ] {
                assert!(r.get(c).parse::<u64>().is_ok(), "{f}: {c} = {}", r.get(c));
            }
            assert!(r.get("version").starts_with("AC"), "{f}: version");
            assert_eq!(
                r.get("reread_identical"),
                "yes",
                "{f}: two traversals must agree"
            );
            assert_eq!(r.get("error"), "-", "{f}: an OPEN_OK row has no error");
        } else {
            assert_ne!(r.get("error"), "-", "{f}: a failure names its error");
        }
    }
}

#[test]
fn an_open_failure_on_an_autocad_file_stops_the_work() {
    let (_, rows) = rows();
    let failed: Vec<String> = rows
        .iter()
        .filter(|r| r.get("tier") == "A" && r.get("open") == "OPEN_FAILED")
        .map(|r| format!("{}: {}", r.get("file"), r.get("error")))
        .collect();
    assert!(
        failed.is_empty(),
        "STOP. Gate 0 failed: acadrust cannot open an AutoCAD-authored file, so the first \
         provider flips back to the existing decoder with the traits unchanged (#1239). \
         Report instead of shipping:\n{}",
        failed.join("\n")
    );
}

#[test]
fn the_document_quotes_the_table_and_agrees_with_it() {
    let doc = read("docs/spikes/acadrust-gate0.md");
    let (rev, rows) = rows();
    assert!(
        doc.contains(&rev),
        "the document must name the revision {rev}"
    );
    assert!(
        doc.contains("OPEN_FAILED"),
        "the document defines the failure term"
    );
    // Column order of the document's Tier A table.
    let doc_cols = [
        ("version", 2),
        ("open", 3),
        ("layers", 4),
        ("layouts", 5),
        ("blocks", 6),
        ("model_entities", 7),
        ("entities_all_scopes", 8),
        ("unmapped", 9),
    ];
    for r in rows.iter().filter(|r| r.get("tier") == "A") {
        let cells: Vec<String> = doc
            .lines()
            .filter(|l| l.trim_start().starts_with('|'))
            .map(|l| {
                l.split('|')
                    .map(|c| c.trim().to_string())
                    .collect::<Vec<_>>()
            })
            .find(|c| c.get(1).map(String::as_str) == Some(r.get("file")))
            .unwrap_or_else(|| panic!("{} is not in the document's table", r.get("file")));
        for (col, idx) in doc_cols {
            assert_eq!(
                cells[idx],
                r.get(col),
                "{}: the document says {} = {}, the table says {}",
                r.get("file"),
                col,
                cells[idx],
                r.get(col)
            );
        }
    }
    let all_ok = rows
        .iter()
        .filter(|r| r.get("tier") == "A")
        .all(|r| r.get("open") == "OPEN_OK");
    let verdict = doc
        .lines()
        .find(|l| l.starts_with("Gate 0:"))
        .expect("a `Gate 0:` verdict line");
    assert_eq!(
        verdict.contains("PASS"),
        all_ok,
        "verdict `{verdict}` against the table"
    );
    assert_eq!(
        verdict.contains("STOP"),
        !all_ok,
        "verdict `{verdict}` against the table"
    );
}

#[test]
fn the_document_says_what_it_could_not_check() {
    let doc = read("docs/spikes/acadrust-gate0.md");
    for needle in [
        "native x86_64",
        "not known truth",
        "What this does not show",
    ] {
        assert!(
            doc.contains(needle),
            "the document should contain `{needle}`"
        );
    }
}

/// AutoCAD wrote `sample_AC10xx_ascii.dxf` in the same SAVEAS session as each DWG, so
/// its census of layers, block records and top-level model-space entities is AutoCAD's
/// own statement, with no DWG reader in the loop (`dxf_census.py` reads the text). A
/// reader that reports a different count is wrong about count, which is where
/// "faster" hides "skipped". Versions with no DWG twin (AC1009) are not compared.
#[test]
fn acadrust_exposes_the_counts_autocads_own_dxf_states() {
    let (_, rows) = rows();
    let census = read("docs/spikes/acadrust-gate0/dxf-census.tsv");
    let mut compared = 0;
    for line in census.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let c: Vec<&str> = line.split('\t').collect();
        assert_eq!(c.len(), 4, "census row: {line}");
        let Some(r) = rows
            .iter()
            .find(|r| r.get("file") == c[0] && r.get("tier") == "A")
        else {
            continue;
        };
        assert_eq!(
            r.get("layers"),
            c[1],
            "{}: layers against AutoCAD's DXF",
            c[0]
        );
        assert_eq!(
            r.get("blocks"),
            c[2],
            "{}: block records against AutoCAD's DXF",
            c[0]
        );
        assert_eq!(
            r.get("model_entities"),
            c[3],
            "{}: model-space entities against AutoCAD's DXF",
            c[0]
        );
        compared += 1;
    }
    assert!(compared >= 6, "only {compared} saves compared");
}
