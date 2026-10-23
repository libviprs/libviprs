//! The acceptance criteria of #1043 (N1.4), applied to `docs/spikes/view-context.md`.
//!
//! The spike is a document, so the only thing a test can hold it to is its
//! shape: every case the issue lists has a row, every row has one of the four
//! v1 behaviours and a fixture that exists, no row says "TBD", the xref
//! decision is one of the three allowed and names its diagnostic, paper space
//! is called in or out, and a provider limitation names the provider version
//! it was seen on. Whether the numbers in the document are right is what the
//! probe under `docs/spikes/view-context/` is for; this file only stops the
//! document from drifting back into a shape that cannot be checked.

use std::collections::BTreeMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn doc() -> String {
    let path = root().join("docs/spikes/view-context.md");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} is not readable: {e}", path.display()))
}

/// The body of the `## heading` section, up to the next `## ` heading.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let marker = format!("## {heading}");
    let start = doc
        .lines()
        .scan(0usize, |off, l| {
            let here = *off;
            *off += l.len() + 1;
            Some((here, l))
        })
        .find(|(_, l)| l.trim_end() == marker || l.starts_with(&format!("{marker} ")))
        .unwrap_or_else(|| panic!("no `{marker}` section"))
        .0;
    let rest = &doc[start..];
    let body_start = rest.find('\n').map_or(rest.len(), |i| i + 1);
    let body = &rest[body_start..];
    let end = body
        .match_indices("\n## ")
        .next()
        .map_or(body.len(), |(i, _)| i + 1);
    &body[..end]
}

/// Rows of the first pipe table in `body`, as trimmed cells, header and rule dropped.
fn table(body: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut seen_rule = false;
    for line in body.lines() {
        let t = line.trim();
        if !t.starts_with('|') {
            if seen_rule {
                break;
            }
            continue;
        }
        let cells: Vec<String> = t
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim().to_owned())
            .collect();
        if cells.iter().all(|c| c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' ')) {
            seen_rule = true;
            continue;
        }
        if seen_rule {
            rows.push(cells);
        }
    }
    rows
}

const CASES: [&str; 13] = [
    "model-space",
    "paper-space-layout",
    "viewport-clip-exact",
    "viewport-clip-bounded",
    "layer-off",
    "layer-frozen",
    "layer-frozen-in-viewport",
    "insert-layer-override",
    "nested-insert",
    "block-twice-two-scales",
    "anonymous-block",
    "dynamic-block",
    "xref",
];

const BEHAVIOURS: [&str; 4] = ["included", "excluded", "IgnoredByPolicy", "Unsupported"];

fn strip_ticks(s: &str) -> String {
    s.trim_matches('`').to_owned()
}

#[test]
fn n14_every_listed_case_has_a_row_with_a_v1_behaviour() {
    let d = doc();
    let rows = table(section(&d, "Decision table"));
    let by_case: BTreeMap<String, Vec<String>> = rows
        .into_iter()
        .map(|r| (strip_ticks(&r[0]), r))
        .collect();
    for case in CASES {
        let row = by_case
            .get(case)
            .unwrap_or_else(|| panic!("the decision table has no row for `{case}`"));
        assert_eq!(row.len(), 5, "`{case}` row must be case | v1 behaviour | diagnostic | fixture | basis");
        let behaviour = strip_ticks(&row[1]);
        assert!(
            BEHAVIOURS.contains(&behaviour.as_str()),
            "`{case}`: `{behaviour}` is not one of {BEHAVIOURS:?}"
        );
        assert!(!row[2].is_empty(), "`{case}`: the diagnostic cell is empty (write `none` if none is emitted)");
    }
}

#[test]
fn n14_no_row_reads_tbd() {
    let d = doc().to_lowercase();
    for banned in ["tbd", "to be decided", "todo", "fixme"] {
        assert!(!d.contains(banned), "the document contains `{banned}`");
    }
}

#[test]
fn n14_every_row_names_a_fixture_that_is_listed_and_exists() {
    let d = doc();
    let fixtures: BTreeMap<String, Vec<String>> = table(section(&d, "Fixtures"))
        .into_iter()
        .map(|r| (strip_ticks(&r[0]), r))
        .collect();
    assert!(!fixtures.is_empty(), "the Fixtures table is empty");
    for (id, row) in &fixtures {
        assert!(row.len() >= 4, "fixture `{id}` needs id | path | origin | evidence class");
        let path = strip_ticks(&row[1]);
        if path.starts_with("external:") {
            continue;
        }
        assert!(root().join(&path).is_file(), "fixture `{id}`: `{path}` does not exist in this repository");
    }
    for r in table(section(&d, "Decision table")) {
        for name in r[3].split(',').map(strip_ticks) {
            let name = name.trim().to_owned();
            assert!(
                fixtures.contains_key(&name),
                "row `{}` names fixture `{name}`, which the Fixtures table does not list",
                r[0]
            );
        }
    }
}

#[test]
fn n14_the_v1_visible_drawing_is_an_applicable_rule_with_two_worked_fixtures() {
    let d = doc();
    let rule = section(&d, "The v1 visible flattened drawing");
    for n in 1..=5 {
        assert!(rule.contains(&format!("**V1-{n}**")), "rule clause V1-{n} is missing");
    }
    let worked = d.lines().filter(|l| l.starts_with("### Worked example: ")).count();
    assert!(worked >= 2, "{worked} worked example(s), the issue asks for two");
}

#[test]
fn n14_the_xref_decision_is_one_of_three_and_names_its_diagnostic() {
    let d = doc();
    let x = section(&d, "Xrefs");
    let decision = x
        .lines()
        .find_map(|l| l.strip_prefix("Decision: "))
        .expect("the Xrefs section has no `Decision:` line")
        .trim()
        .trim_end_matches('.')
        .to_owned();
    assert!(
        ["resolve", "reject", "report-as-external"].contains(&decision.as_str()),
        "xref decision `{decision}` is not resolve, reject or report-as-external"
    );
    assert!(
        x.lines().any(|l| l.starts_with("Diagnostic: ") && l.contains("UNRESOLVED_BLOCK")),
        "the Xrefs section does not name the diagnostic on a `Diagnostic:` line"
    );
    assert!(
        x.to_lowercase().contains("omit silently") && x.to_lowercase().contains("rejected"),
        "the Xrefs section does not record `omit silently` as rejected"
    );
}

#[test]
fn n14_paper_space_is_called_in_or_out_with_viewport_evidence() {
    let d = doc();
    let p = section(&d, "Paper space");
    let call = p
        .lines()
        .find_map(|l| l.strip_prefix("Paper space in v1: "))
        .expect("the Paper space section has no `Paper space in v1:` line")
        .trim()
        .trim_end_matches('.')
        .to_owned();
    assert!(["in", "out"].contains(&call.as_str()), "`{call}` is not in or out");
    assert!(p.contains("### Viewport transform evidence"), "no viewport transform evidence subsection");
}

#[test]
fn n14_a_provider_limitation_names_the_provider_version() {
    let d = doc();
    let lim = section(&d, "Provider limitations");
    assert!(lim.contains("ACadSharp 3.7.1"), "no ACadSharp version in Provider limitations");
    assert!(lim.contains("3.7.1-viprs.2"), "no shim version in Provider limitations");
    for case in ["layer-off", "layer-frozen", "layer-frozen-in-viewport"] {
        assert!(lim.contains(case), "Provider limitations does not mention `{case}`");
    }
}

#[test]
fn n14_the_decoder_exposure_of_layers_and_block_names_is_recorded() {
    let d = doc();
    let e = section(&d, "What the decoder exposes");
    for needle in ["layer visibility", "layer name", "block name", "#1233"] {
        assert!(e.to_lowercase().contains(needle), "`What the decoder exposes` does not cover `{needle}`");
    }
}
