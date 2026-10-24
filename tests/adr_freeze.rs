//! Executable acceptance checks for the frozen integration ADRs (issue #1047, N1.8).
//!
//! Same shape as `adr_normalized_geometry.rs`: the ADRs are specifications, so these tests read
//! them as text at compile time (no filesystem call, fine under Miri) and hold them to the
//! acceptance criteria of the issue, one test per criterion. std-only on purpose.
//!
//! The streaming ADR is deliberately not read here. The pre-push gate pins it as a path nothing
//! reads (`INERT_IN_LIBVIPRS` in libviprs-tests), and pulling it in would make that pin false. Its
//! freeze was checked by hand against the N1.7 spike instead, see the pull request.

const VIEW_PROFILE: &str = include_str!("../docs/adr/pvec-view-profile.md");
const SEMANTIC_BOUNDARY: &str = include_str!("../docs/adr/pvec-semantic-boundary.md");
const TRANSFORM_MATRIX: &str = include_str!("../docs/adr/pvec-transform-matrix.md");
const UNSUPPORTED_POLICY: &str = include_str!("../docs/adr/pvec-unsupported-policy.md");
const PROVENANCE: &str = include_str!("../docs/adr/pvec-provenance.md");
const PDF_POLICY: &str = include_str!("../docs/adr/pvec-pdf-policy.md");
/// The N1.2 ADR predates the freeze and keeps its clause text; it is frozen in place.
const NORMALIZED_GEOMETRY: &str = include_str!("../docs/adr/pvec-normalized-geometry.md");

const SIX: [(&str, &str); 6] = [
    ("pvec-view-profile", VIEW_PROFILE),
    ("pvec-semantic-boundary", SEMANTIC_BOUNDARY),
    ("pvec-transform-matrix", TRANSFORM_MATRIX),
    ("pvec-unsupported-policy", UNSUPPORTED_POLICY),
    ("pvec-provenance", PROVENANCE),
    ("pvec-pdf-policy", PDF_POLICY),
];

fn lower(s: &str) -> String {
    s.to_lowercase()
}

/// The text under the heading that starts with `## {name}`, up to the next level-two heading.
fn section(doc: &str, name: &str) -> String {
    let head = format!("\n## {name}");
    let start = doc
        .find(&head)
        .unwrap_or_else(|| panic!("section `{name}` is missing"));
    let rest = &doc[start + head.len()..];
    let end = rest.find("\n## ").unwrap_or(rest.len());
    rest[..end].to_string()
}

#[test]
fn ac1_seven_files_each_with_a_status_line_and_a_spike_link() {
    for (name, doc) in SIX {
        let status = doc
            .lines()
            .find(|l| l.starts_with("Status:"))
            .unwrap_or_else(|| panic!("{name} has no status line"));
        assert!(
            status.starts_with("Status: Frozen"),
            "{name}: status must be a freeze, got `{status}`"
        );
        assert!(
            doc.contains("spikes/") && doc.contains(".md"),
            "{name} must link the spike output it rests on"
        );
        assert!(
            doc.contains("#1047"),
            "{name} must say which issue froze it"
        );
    }
}

#[test]
fn ac1_the_n1_2_adr_is_frozen_in_place() {
    let status = NORMALIZED_GEOMETRY
        .lines()
        .find(|l| l.starts_with("Status:"))
        .expect("status line");
    assert!(status.starts_with("Status: Frozen"), "got `{status}`");
    assert!(NORMALIZED_GEOMETRY.contains("## Freeze record"));
}

#[test]
fn ac2_every_adr_names_what_is_still_open_and_the_missing_measurement() {
    for (name, doc) in SIX {
        let open = section(doc, "Open items");
        assert!(
            open.trim().len() > 80,
            "{name}: the open items section is empty or a placeholder"
        );
        assert!(
            lower(&open).contains("measure")
                || lower(&open).contains("not measured")
                || lower(&open).contains("unmeasured"),
            "{name}: open items must name the missing measurement"
        );
    }
}

/// A float with an exponent, or one below one, reads as a tolerance. The ADRs quote measured
/// counts and peak RSS (which have a fraction but are not below one), and nothing else.
fn tolerance_shaped(doc: &str) -> Vec<String> {
    doc.split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+')))
        .map(|t| t.trim_matches(|c| c == '.' || c == '-' || c == '+'))
        .filter(|t| t.chars().any(|c| c.is_ascii_digit()))
        .filter(|t| t.parse::<f64>().is_ok())
        .filter(|t| t.contains(['e', 'E']) || t.starts_with("0."))
        .map(str::to_string)
        .collect()
}

#[test]
fn ac3_no_adr_defines_a_tolerance_and_each_cites_the_numeric_policy() {
    for (name, doc) in SIX {
        let bad = tolerance_shaped(doc);
        assert!(
            bad.is_empty(),
            "{name} defines tolerance-shaped numbers: {bad:?}"
        );
        assert!(
            doc.contains("P1.2"),
            "{name} must cite the numeric policy P1.2"
        );
        assert!(
            lower(doc).contains("libviprs defines no epsilon")
                || lower(doc).contains("defines no tolerance")
                || lower(doc).contains("no epsilon of its own"),
            "{name} must say libviprs defines no tolerance of its own"
        );
    }
}

#[test]
fn ac4_unsupported_policy_lists_six_classes_and_the_tessellation_defect() {
    let doc = UNSUPPORTED_POLICY;
    for c in [
        "SupportedExact",
        "SupportedEquivalent",
        "RequiresPVecCapability",
        "RequiresExtension",
        "Unsupported",
        "IgnoredByPolicy",
    ] {
        assert!(doc.contains(c), "class {c} missing");
    }
    let l = lower(doc);
    assert!(
        l.contains("silently tessellating an unsupported curve into canonical pvec is a defect")
    );
    assert!(l.contains("refusal rule"));
}

#[test]
fn ac6_pdf_policy_does_not_assume_cad_semantics_survive_the_export() {
    let l = lower(PDF_POLICY);
    assert!(l.contains("not assumed to preserve cad semantics"));
    assert!(l.contains("display fidelity"));
    assert!(l.contains("epic p"));
    assert!(!l.contains("mathematically exact") || l.contains("never mathematically exact"));
}

#[test]
fn cross_cutting_conformance_classes_are_epic_ps_and_cited_once() {
    for (name, doc) in [("unsupported", UNSUPPORTED_POLICY), ("pdf", PDF_POLICY)] {
        assert!(
            lower(doc).contains("epic p"),
            "{name} must cite EPIC P's classes"
        );
    }
}

#[test]
fn transform_matrix_freezes_the_three_codes_and_the_check_the_returned_kind_rule() {
    let doc = TRANSFORM_MATRIX;
    for c in [
        "TRANSFORM_SINGULAR",
        "TRANSFORM_CONIC_ILL_CONDITIONED",
        "TRANSFORM_NON_FINITE",
    ] {
        assert!(doc.contains(c), "{c} missing");
    }
    assert!(lower(doc).contains("checks pvec's returned kind"));
}

#[test]
fn view_profile_states_the_v1_drawing() {
    let l = lower(VIEW_PROFILE);
    for needle in ["model space only", "layer", "xref", "paper space", "#1239"] {
        assert!(l.contains(needle), "view profile must mention `{needle}`");
    }
}

#[test]
fn provenance_is_optional_component_data_outside_the_hot_path() {
    let l = lower(PROVENANCE);
    assert!(l.contains("optional component data"));
    assert!(l.contains("hot path"));
}

#[test]
fn prose_hygiene() {
    for (name, doc) in SIX {
        assert!(!doc.contains('\u{2014}'), "{name}: no em dashes");
        assert!(!doc.contains(" -- "), "{name}: no `--` as an em dash");
        let l = lower(doc);
        for w in [
            "claude",
            "co-authored",
            "generated by",
            "generated with",
            "anthropic",
        ] {
            assert!(!l.contains(w), "{name} mentions `{w}`");
        }
    }
}
