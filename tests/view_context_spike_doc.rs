//! The acceptance criteria of #1043 (N1.4), applied to `docs/spikes/view-context.md`.
//!
//! The spike is a document, so the only thing a test can hold it to is its
//! shape: every case the issue lists has a row, every row has one of the four
//! v1 behaviours and a fixture that exists, no row says "TBD", the xref
//! decision is one of the three allowed and names its diagnostic, paper space
//! is called in or out, and a provider limitation names the provider version
//! it was seen on. Whether the numbers in the document are right is what the
//! probe under `docs/spikes/view-context/` is for. This file also holds the
//! figures the document cites to the committed probe output
//! (`results-x64.txt`) and the diagnostic names to the sources, so an edited
//! figure or a name that does not exist fails here.

use std::collections::BTreeMap;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn doc() -> String {
    let path = root().join("docs/spikes/view-context.md");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} is not readable: {e}", path.display()))
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
        if cells
            .iter()
            .all(|c| c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' '))
        {
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
#[cfg_attr(miri, ignore)]
fn n14_every_listed_case_has_a_row_with_a_v1_behaviour() {
    let d = doc();
    let rows = table(section(&d, "Decision table"));
    let by_case: BTreeMap<String, Vec<String>> =
        rows.into_iter().map(|r| (strip_ticks(&r[0]), r)).collect();
    for case in CASES {
        let row = by_case
            .get(case)
            .unwrap_or_else(|| panic!("the decision table has no row for `{case}`"));
        assert_eq!(
            row.len(),
            5,
            "`{case}` row must be case | v1 behaviour | diagnostic | fixture | basis"
        );
        let behaviour = strip_ticks(&row[1]);
        assert!(
            BEHAVIOURS.contains(&behaviour.as_str()),
            "`{case}`: `{behaviour}` is not one of {BEHAVIOURS:?}"
        );
        assert!(
            !row[2].is_empty(),
            "`{case}`: the diagnostic cell is empty (write `none` if none is emitted)"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn n14_no_row_reads_tbd() {
    let d = doc().to_lowercase();
    for banned in ["tbd", "to be decided", "todo", "fixme"] {
        assert!(!d.contains(banned), "the document contains `{banned}`");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn n14_every_row_names_a_fixture_that_is_listed_and_exists() {
    let d = doc();
    let fixtures: BTreeMap<String, Vec<String>> = table(section(&d, "Fixtures"))
        .into_iter()
        .map(|r| (strip_ticks(&r[0]), r))
        .collect();
    assert!(!fixtures.is_empty(), "the Fixtures table is empty");
    for (id, row) in &fixtures {
        assert!(
            row.len() >= 4,
            "fixture `{id}` needs id | path | origin | evidence class"
        );
        let path = strip_ticks(&row[1]);
        if path.starts_with("external:") {
            continue;
        }
        assert!(
            root().join(&path).is_file(),
            "fixture `{id}`: `{path}` does not exist in this repository"
        );
    }
    for r in table(section(&d, "Decision table")) {
        for name in r[3].split(',').map(|n| strip_ticks(n.trim())) {
            assert!(
                fixtures.contains_key(&name),
                "row `{}` names fixture `{name}`, which the Fixtures table does not list",
                r[0]
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn n14_the_v1_visible_drawing_is_an_applicable_rule_with_two_worked_fixtures() {
    let d = doc();
    let rule = section(&d, "The v1 visible flattened drawing");
    for n in 1..=5 {
        assert!(
            rule.contains(&format!("**V1-{n} ")),
            "rule clause V1-{n} is missing"
        );
    }
    let worked = d
        .lines()
        .filter(|l| l.starts_with("### Worked example: "))
        .count();
    assert!(
        worked >= 2,
        "{worked} worked example(s), the issue asks for two"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
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
        x.lines()
            .any(|l| l.starts_with("Diagnostic: ") && l.contains("UNRESOLVED_BLOCK")),
        "the Xrefs section does not name the diagnostic on a `Diagnostic:` line"
    );
    assert!(
        x.to_lowercase().contains("omit silently") && x.to_lowercase().contains("rejected"),
        "the Xrefs section does not record `omit silently` as rejected"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
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
    assert!(
        ["in", "out"].contains(&call.as_str()),
        "`{call}` is not in or out"
    );
    assert!(
        p.contains("### Viewport transform evidence"),
        "no viewport transform evidence subsection"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn n14_a_provider_limitation_names_the_provider_version() {
    let d = doc();
    let lim = section(&d, "Provider limitations");
    assert!(
        lim.contains("ACadSharp 3.7.1"),
        "no ACadSharp version in Provider limitations"
    );
    assert!(
        lim.contains("3.7.1-viprs.2"),
        "no shim version in Provider limitations"
    );
    for case in ["layer-off", "layer-frozen", "layer-frozen-in-viewport"] {
        assert!(
            lim.contains(case),
            "Provider limitations does not mention `{case}`"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn n14_the_decoder_exposure_of_layers_and_block_names_is_recorded() {
    let d = doc();
    let e = section(&d, "What the decoder exposes");
    for needle in ["layer visibility", "layer name", "block name", "#1233"] {
        assert!(
            e.to_lowercase().contains(needle),
            "`What the decoder exposes` does not cover `{needle}`"
        );
    }
}

fn read(rel: &str) -> String {
    let path = root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} is not readable: {e}", path.display()))
}

fn results() -> String {
    read("docs/spikes/view-context/results-x64.txt")
}

/// The text of one probe run in `results-x64.txt`: from the line that starts
/// with `header` to the next blank line.
fn run<'a>(results: &'a str, header: &str) -> &'a str {
    let start = results
        .lines()
        .scan(0usize, |off, l| {
            let here = *off;
            *off += l.len() + 1;
            Some((here, l))
        })
        .find(|(_, l)| l.starts_with(header))
        .unwrap_or_else(|| panic!("results-x64.txt has no `{header}` run"))
        .0;
    let rest = &results[start..];
    let end = rest.find("\n\n").unwrap_or(rest.len());
    &rest[..end]
}

/// Every figure the document states, tied to the run that produced it:
/// (a phrase that must be in the document, the run header, what that run must hold).
const FIGURES: &[(&str, &str, &[&str])] = &[
    (
        "`rule_visible_leaves 3` and `rule_hidden_entities 6`",
        "== rule /work/out/vc_layers.dwg",
        &["rule_visible_leaves 3", "rule_hidden_entities 6"],
    ),
    (
        "decoder emitted **13** records",
        "== rule /work/out/vc_layers.dwg",
        &["shim_model_space_geometry_records 13"],
    ),
    (
        "Ten of them are excluded",
        "== rule /work/out/vc_layers.dwg",
        &["shim_emitted_but_hidden_by_rule 10"],
    ),
    (
        "rule counts 146 leaves and the decoder emitted 134, and the 12 missing",
        "== rule /work/in/sample_AC1032.dwg",
        &[
            "rule_visible_leaves 146",
            "shim_model_space_geometry_records 134",
            "shim_visible_by_rule_but_not_emitted 12",
        ],
    ),
    (
        "20 records, rule 20",
        "== rule /work/in/BLOCKVISIBILITYPARAMETER.dwg",
        &[
            "rule_visible_leaves 20",
            "shim_model_space_geometry_records 20",
        ],
    ),
    (
        "Four leaves, and the decoder emitted exactly these four",
        "== rule /work/out/vc_nested.dwg",
        &[
            "rule_visible_leaves 4",
            "shim_model_space_geometry_records 4",
        ],
    ),
    (
        "`rule_external 1`",
        "== rule /work/in/g13_xref.dwg",
        &["rule_external 1"],
    ),
    (
        "0.693488",
        "== describe /work/in/sample_AC1032.dwg",
        &[
            "representsPaper=False",
            "viewHeight=9.228705",
            "scale=0.693488",
        ],
    ),
    (
        "0.696757",
        "== describe /work/in/sample_AC1032.dwg",
        &["viewHeight=9.108871", "scale=0.696757"],
    ),
    (
        "`ViewDirection = (-1,-1,1)`",
        "-- /work/in/AecObjects.dwg",
        &[
            "layer=Defpoints",
            "viewDir=(-1,-1,1)",
            "representsPaper=False",
        ],
    ),
    (
        "`ViewHeight 9.228705`",
        "-- /work/in/sample_AC1015.dwg",
        &["viewHeight=9.228705", "representsPaper=False"],
    ),
    (
        "`ViewHeight 0`",
        "-- /work/in/sample_AC1014.dwg",
        &["viewCenter=(0,0) viewHeight=0", "representsPaper=False"],
    ),
];

#[test]
#[cfg_attr(miri, ignore)]
fn n14_every_cited_figure_is_in_the_committed_probe_output() {
    let d = doc();
    let r = results();
    for (phrase, header, needles) in FIGURES.iter().copied() {
        assert!(
            d.contains(phrase),
            "the document no longer says `{phrase}`; update FIGURES with it"
        );
        let block = run(&r, header);
        for n in needles {
            assert!(
                block.contains(n),
                "the document cites `{phrase}` but `{header}` in results-x64.txt has no `{n}`"
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn n14_every_long_decimal_in_the_document_is_in_the_probe_output() {
    let d = doc();
    let r = results();
    let mut cur = String::new();
    let mut found = Vec::new();
    for ch in d.chars().chain(std::iter::once(' ')) {
        if ch.is_ascii_digit() || ch == '.' {
            cur.push(ch);
        } else {
            let t = cur.trim_matches('.');
            if let Some((_, frac)) = t.split_once('.')
                && frac.len() >= 4
                && !frac.contains('.')
            {
                found.push(t.to_owned());
            }
            cur.clear();
        }
    }
    for n in found {
        assert!(
            r.contains(&n),
            "the document cites {n}, which is not in results-x64.txt"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn n14_diagnostic_names_exist_unless_marked_proposed() {
    let d = doc();
    let report = read("src/cad/report.rs");
    let tiler = read("src/vector/tiler.rs");
    let existing = |name: &str| report.contains(&format!("pub const {name}: Self = Self("));
    let code_of = |name: &str| -> String {
        let tail = report
            .split(&format!("pub const {name}: Self = Self("))
            .nth(1)
            .unwrap();
        tail.split(')').next().unwrap().to_owned()
    };
    let bytes: Vec<&str> = d.split('`').collect();
    // Odd-indexed pieces are inside backticks; the piece after it is the prose that follows.
    let mut proposed = std::collections::BTreeSet::new();
    let mut named = std::collections::BTreeSet::new();
    for i in (1..bytes.len()).step_by(2) {
        let tok = bytes[i];
        let after = bytes.get(i + 1).copied().unwrap_or("");
        if let Some(v) = tok.strip_prefix("TilerError::") {
            assert!(
                tiler.contains(&format!("    {v} {{")) || tiler.contains(&format!("    {v},")),
                "`{tok}` is not a variant in src/vector/tiler.rs"
            );
            continue;
        }
        let is_code = tok.len() >= 6
            && tok.contains('_')
            && tok
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
        if !is_code {
            continue;
        }
        if after.trim_start().starts_with("(proposed)") {
            proposed.insert(tok.to_owned());
            continue;
        }
        let num = after
            .trim_start()
            .strip_prefix('(')
            .and_then(|a| a.split(')').next())
            .filter(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
        // Other upper-case words (an evidence class such as KNOWN_TRUTH_EXACT)
        // are not diagnostics; a code is one when it carries its number, is
        // marked proposed, or sits in the table's diagnostic column below.
        if let Some(num) = num {
            named.insert(tok.to_owned());
            if existing(tok) {
                assert_eq!(
                    code_of(tok),
                    num,
                    "`{tok}` is ({num}) in the document but {} in src/cad/report.rs",
                    code_of(tok)
                );
            }
        }
    }
    for row in table(section(&d, "Decision table")) {
        for piece in row[2].split('`').skip(1).step_by(2) {
            let is_code = piece.contains('_')
                && piece
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
            if is_code {
                named.insert(piece.to_owned());
            }
        }
    }
    for p in &proposed {
        assert!(
            !existing(p),
            "`{p}` is marked proposed but already exists in src/cad/report.rs"
        );
    }
    for n in &named {
        assert!(
            existing(n) || proposed.contains(n),
            "`{n}` is named as a diagnostic but is not in src/cad/report.rs and is not marked (proposed)"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn n14_the_paper_space_call_agrees_with_the_layout_row() {
    let d = doc();
    let call = section(&d, "Paper space")
        .lines()
        .find_map(|l| l.strip_prefix("Paper space in v1: "))
        .unwrap()
        .trim()
        .trim_end_matches('.')
        .to_owned();
    let rows = table(section(&d, "Decision table"));
    let behaviour = rows
        .iter()
        .find(|r| strip_ticks(&r[0]) == "paper-space-layout")
        .map(|r| strip_ticks(&r[1]))
        .unwrap();
    let want = if call == "out" {
        "Unsupported"
    } else {
        "included"
    };
    assert_eq!(
        behaviour, want,
        "Paper space in v1 is `{call}` but the paper-space-layout row says `{behaviour}`"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn n14_every_no_geometry_sentence_in_paper_space_names_its_fixture() {
    let d = doc();
    let fixtures: Vec<String> = table(section(&d, "Fixtures"))
        .into_iter()
        .map(|r| strip_ticks(&r[0]))
        .collect();
    let p = section(&d, "Paper space")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let mut checked = 0;
    for sentence in p.split(". ") {
        let low = sentence.to_lowercase();
        if low.contains("no geometry")
            || low.contains("emits no")
            || low.contains("holds no drawing")
        {
            checked += 1;
            assert!(
                fixtures
                    .iter()
                    .any(|f| sentence.contains(&format!("`{f}`"))),
                "this paper-space sentence does not name the fixture it is about: {sentence}"
            );
        }
    }
    assert!(checked > 0, "no `no geometry` sentence found to check");
}

#[test]
#[cfg_attr(miri, ignore)]
fn n14_viewport_numbers_say_what_they_prove_and_what_they_are() {
    let d = doc();
    let p = section(&d, "Paper space");
    let ev = p.split("### Viewport transform evidence").nth(1).unwrap();
    assert!(
        ev.contains("by construction"),
        "the vc_paper worked points are ViewCenter mapping to Center; the evidence section must say that is true by construction"
    );
    assert!(
        ev.contains("representsPaper"),
        "the 1.0 ratios are paper-background viewports; the observations must say representsPaper"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn n14_v1_3_says_what_an_entitys_own_invisible_flag_does() {
    let d = doc();
    let rule = section(&d, "The v1 visible flattened drawing");
    let v13 = rule
        .split("- **V1-3 ")
        .nth(1)
        .and_then(|t| t.split("- **V1-4 ").next())
        .expect("V1-3 clause");
    assert!(
        v13.to_lowercase().contains("invisible"),
        "V1-3 does not say what happens with an entity's own invisible flag"
    );
}
