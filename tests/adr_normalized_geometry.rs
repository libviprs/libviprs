//! Executable acceptance checks for `docs/adr/pvec-normalized-geometry.md` (issue #1041, N1.2).
//!
//! The ADR is a specification, so these tests read it as text and hold it to the
//! acceptance criteria of the issue, one test per criterion. They are std-only on
//! purpose: the claim they guard is "libviprs grows no geometry library", and a
//! test that pulled in a curve crate to check it would be the wrong shape.
//!
//! Clause ids are stable. A reviewer who wants the clause that forbids libviprs
//! from tessellating a curve itself looks for `NG-V1`; the tests below fail if
//! that clause, or the prohibition it points at, goes missing.

/// The ADR, pulled in at compile time rather than read at runtime, so these tests make no
/// filesystem call, run under Miri and stay off `tests/miri_fs_test_inventory.txt`.
const ADR: &str = include_str!("../docs/adr/pvec-normalized-geometry.md");

fn adr() -> String {
    ADR.to_string()
}

/// The text of the clause whose heading starts with `### {id} `, up to the next
/// heading of level two or three.
fn clause(doc: &str, id: &str) -> String {
    let head = format!("### {id} ");
    let start = doc
        .find(&head)
        .unwrap_or_else(|| panic!("clause {id} is missing from the ADR"));
    let rest = &doc[start + head.len()..];
    let end = rest
        .find("\n## ")
        .into_iter()
        .chain(rest.find("\n### "))
        .min()
        .unwrap_or(rest.len());
    rest[..end].to_string()
}

/// The pvec revision the ADR cites (`libviprs/pvec` main after P3.7, #83). It moves only together
/// with `NP_HEADINGS` below and with a re-check of every citation in NG-03 and NG-09.
const PIN: &str = "fdf87de";

/// The `### NP-nn Title` headings of `docs/NUMERIC_POLICY.md` at `PIN`, lowercased. The numbers are
/// the index plus one. A citation is only right when the heading it lands on is about the question.
const NP_HEADINGS: [&str; 20] = [
    "finite and normal values",
    "coordinate magnitude",
    "negative zero",
    "the tolerance triple",
    "equality",
    "degenerate length",
    "zero radius and conditioning",
    "coincident control points and degenerate hulls",
    "orientation",
    "direction",
    "periodic parameter normalisation",
    "trim domains",
    "parameter domain and unit map",
    "knots",
    "weights",
    "continuity and one-sided derivatives",
    "composite joints",
    "maximum counts and dimensions",
    "affine transforms",
    "closed flag",
];

/// (a word from the question cell of the NG-09 row, the rules that row must cite, each with a word
/// its heading at `PIN` must contain)
type Ng09Row = (&'static str, &'static [(usize, &'static str)]);
const NG09_ROWS: &[Ng09Row] = &[
    ("are two values equal", &[(5, "equality")]),
    ("finite and normal", &[(1, "finite"), (2, "magnitude")]),
    ("degenerate", &[(6, "length"), (7, "radius")]),
    ("direction", &[(10, "direction")]),
    ("full period", &[(11, "periodic"), (12, "trim")]),
    (
        "knots, weights",
        &[(14, "knots"), (15, "weights"), (18, "counts")],
    ),
    ("composite joint", &[(17, "joints"), (20, "closed")]),
    ("affine map", &[(19, "affine")]),
];

/// The rule numbers cited in a piece of text, in order.
fn np_numbers(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("NP-") {
        let tail = &rest[i + 3..];
        let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        out.push(digits.parse().unwrap_or(0));
        rest = &tail[digits.len()..];
    }
    out
}

/// Holds one NG-09 row to the headings of the pinned policy: it must cite exactly the expected
/// rules, and each cited rule's heading must carry the expected word. Err names what is wrong.
fn check_ng09_row(row: &str, expected: &[(usize, &str)]) -> Result<(), String> {
    let cells: Vec<&str> = row.split('|').map(str::trim).collect();
    let cited = np_numbers(cells.get(2).copied().unwrap_or(""));
    let want: Vec<usize> = expected.iter().map(|(n, _)| *n).collect();
    if cited != want {
        return Err(format!(
            "cites NP-{cited:?}, the pinned policy says {want:?}: {row}"
        ));
    }
    for (n, word) in expected {
        let heading = NP_HEADINGS[n - 1];
        if !heading.contains(word) {
            return Err(format!(
                "NP-{n:02} is `{heading}` at {PIN}, not about `{word}`: {row}"
            ));
        }
    }
    Ok(())
}

fn lower(s: &str) -> String {
    s.to_lowercase()
}

// ---------------------------------------------------------------- AC 1

/// (source shape, ISO construct names the row must carry, ISO_PROFILE sections it must cite)
const MAPPING: &[(&str, &[&str], &[&str])] = &[
    (
        "line",
        &["`trimmed_curve`", "`line`"],
        &["section 4", "section 8"],
    ),
    ("polyline", &["`polyline`"], &["section 5"]),
    (
        "circular arc",
        &["`trimmed_curve`", "`circle`"],
        &["section 6", "section 8"],
    ),
    ("circle", &["`circle`"], &["section 6"]),
    ("ellipse", &["`ellipse`"], &["section 7"]),
    (
        "b-spline",
        &["`b_spline_curve_with_knots`"],
        &["section 11"],
    ),
    (
        "nurbs",
        &["`rational_b_spline_curve`", "`b_spline_curve_with_knots`"],
        &["section 12"],
    ),
];

#[test]
fn ac1_each_curve_kind_names_the_pvec_semantic_type() {
    let doc = adr();
    let table = clause(&doc, "NG-03");
    for (shape, names, sections) in MAPPING {
        let row = table
            .lines()
            .find(|l| {
                l.starts_with('|') && lower(l.split('|').nth(1).unwrap_or("")).trim() == *shape
            })
            .unwrap_or_else(|| panic!("NG-03 has no mapping row for `{shape}`"));
        for n in *names {
            assert!(
                row.contains(n),
                "row for `{shape}` does not name {n}: {row}"
            );
        }
        for s in *sections {
            assert!(
                row.contains(s),
                "row for `{shape}` does not cite ISO_PROFILE {s}: {row}"
            );
        }
    }
}

#[test]
fn ac1_the_mapping_cites_a_pinned_profile_revision_not_a_branch() {
    let doc = adr();
    let t = clause(&doc, "NG-03");
    assert!(
        t.contains("ISO_PROFILE.md"),
        "NG-03 must name the profile it maps to"
    );
    assert!(
        t.contains(&format!("libviprs/pvec@{PIN}")),
        "NG-03 must pin the PVec revision it cites, so the names cannot drift silently"
    );
    assert!(
        doc.contains(&format!("revision `{PIN}`")),
        "the Context section must name the same pinned revision as NG-03"
    );
    assert!(
        !doc.contains("e64b1a4"),
        "the old pin is still mentioned, so some citation was not re-checked against the new one"
    );
}

// ---------------------------------------------------------------- AC 2

#[test]
fn ac2_thin_adapter_statement_is_present_verbatim() {
    let doc = adr();
    let stmt = "`Normalized2DGeometry` is a thin adapter over PVec's public semantic concepts and is not a second geometry library.";
    assert!(
        doc.contains(stmt),
        "the thin-adapter statement is missing or reworded"
    );
    assert!(
        clause(&doc, "NG-01").contains(stmt),
        "the statement must sit in NG-01"
    );
}

#[test]
fn ac2_banned_constructs_are_listed_each_with_a_reason() {
    let doc = adr();
    let banned = [
        ("NG-B1", "curve evaluator"),
        ("NG-B2", "tessellator"),
        ("NG-B3", "knot insertion"),
        ("NG-B4", "arc-to-nurbs"),
        ("NG-B5", "epsilon"),
        ("NG-B6", "coordinate equality"),
    ];
    let list = clause(&doc, "NG-10");
    for (id, what) in banned {
        let row = list
            .lines()
            .find(|l| l.starts_with('|') && l.contains(id))
            .unwrap_or_else(|| panic!("NG-10 has no row for {id}"));
        assert!(
            lower(row).contains(what),
            "{id} row does not name `{what}`: {row}"
        );
        let cells: Vec<&str> = row.split('|').map(str::trim).collect();
        // | id | construct | reason | what to do instead |
        assert!(
            cells.len() >= 5 && cells[3].split_whitespace().count() >= 6,
            "{id} needs a reason of its own, not a bare name: {row}"
        );
        assert!(
            cells.len() >= 5 && !cells[4].is_empty(),
            "{id} must say what the adapter does instead"
        );
    }
}

// ---------------------------------------------------------------- AC 3

/// A token that parses as a float with a fraction or an exponent is a threshold
/// by another name, and the ADR may not contain one.
fn tolerance_shaped_tokens(doc: &str) -> Vec<String> {
    doc.split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+')))
        .map(|t| t.trim_matches(|c| c == '.' || c == '-' || c == '+'))
        .filter(|t| t.chars().any(|c| c.is_ascii_digit()))
        .filter(|t| t.parse::<f64>().is_ok() && t.chars().any(|c| matches!(c, '.' | 'e' | 'E')))
        .map(str::to_string)
        .collect()
}

#[test]
fn ac3_no_tolerance_literal_anywhere() {
    let doc = adr();
    let bad = tolerance_shaped_tokens(&doc);
    assert!(
        bad.is_empty(),
        "the ADR defines numbers that look like tolerances: {bad:?}"
    );
}

#[test]
fn ac3_numeric_questions_are_answered_by_citation() {
    let doc = adr();
    let t = clause(&doc, "NG-09");
    let cite = "PVec numeric policy v1, NP-";
    let n = t.matches(cite).count();
    assert!(
        n >= 6,
        "NG-09 must answer its numeric questions by citation, found {n} citations"
    );
    // every NP-nn in the whole document is a real rule number (01..=20) in the citation form
    let mut rest = doc.as_str();
    let mut seen = 0;
    while let Some(i) = rest.find("NP-") {
        let tail = &rest[i + 3..];
        let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        let k: u32 = digits
            .parse()
            .unwrap_or_else(|_| panic!("`NP-` not followed by a rule number"));
        assert!(
            (1..=20).contains(&k),
            "NP-{k} is not a rule in PVec numeric policy v1"
        );
        seen += 1;
        rest = &tail[digits.len()..];
    }
    assert!(seen >= n);
}

#[test]
fn ac3_each_row_cites_the_rules_whose_headings_are_about_its_question() {
    let doc = adr();
    let t = clause(&doc, "NG-09");
    for (key, expected) in NG09_ROWS {
        let row = t
            .lines()
            .find(|l| l.starts_with('|') && lower(l.split('|').nth(1).unwrap_or("")).contains(key))
            .unwrap_or_else(|| panic!("NG-09 has no row for `{key}`"));
        if let Err(e) = check_ng09_row(row, expected) {
            panic!("NG-09 row `{key}` {e}");
        }
    }
    let rows = t
        .lines()
        .filter(|l| l.starts_with("| ") && !l.contains("question"))
        .count();
    assert_eq!(
        rows,
        NG09_ROWS.len(),
        "NG-09 has a row this check does not know"
    );
}

#[test]
fn ac3_the_row_check_rejects_a_swapped_rule_number() {
    let good = "| are two values equal, are two points equal | PVec numeric policy v1, NP-05 |";
    let swapped = "| are two values equal, are two points equal | PVec numeric policy v1, NP-09 |";
    assert!(check_ng09_row(good, &[(5, "equality")]).is_ok());
    assert!(check_ng09_row(swapped, &[(5, "equality")]).is_err());
    // right set of numbers, wrong order of meaning: the heading word guards a stale table too
    assert!(check_ng09_row(good, &[(5, "orientation")]).is_err());
}

#[test]
fn every_np_citation_in_the_document_resolves_to_a_heading_at_the_pin() {
    let doc = adr();
    for n in np_numbers(&doc) {
        assert!(
            (1..=NP_HEADINGS.len()).contains(&n),
            "NP-{n} is not a rule in PVec numeric policy v1 at {PIN}"
        );
    }
}

// ------------------------------------------------ clauses re-checked at the new pin

#[test]
fn ng05_says_pvecs_bound_and_not_exact_and_agrees_with_ng02() {
    let doc = adr();
    let t = lower(&clause(&doc, "NG-05"));
    assert!(
        !t.contains("exact"),
        "NG-05 calls the bound exact, PVec does not"
    );
    for w in ["conservative", "not tight", "spline", "control hull"] {
        assert!(t.contains(w), "NG-05 does not say `{w}`");
    }
    assert!(
        lower(&clause(&doc, "NG-02")).contains("conservative"),
        "NG-02 must describe source_bounds the way NG-05 does"
    );
}

#[test]
fn ng_b4_points_at_pvecs_conversion_and_status_has_no_promotion_table_here() {
    let doc = adr();
    let row = clause(&doc, "NG-10")
        .lines()
        .find(|l| l.contains("NG-B4"))
        .expect("NG-B4 row")
        .to_string();
    assert!(row.contains("Curve::to_rational_b_spline"), "NG-B4: {row}");
    assert!(
        row.contains("ISO_PROFILE.md"),
        "NG-B4 must name the table's home: {row}"
    );
    let st = doc
        .split("## Status and dependencies")
        .nth(1)
        .expect("status");
    assert!(!st.contains("owns the transform promotion table"));
    assert!(st.contains("ISO_PROFILE.md") && st.contains("P3.8"));
    assert!(st.contains("ellipse") && st.contains("shear"));
}

#[test]
fn ng11_no_longer_lists_bounds_or_calls_p3_1_a_prototype() {
    let doc = adr();
    let t = lower(&clause(&doc, "NG-11"));
    assert!(
        !t.contains("bounds of an arc"),
        "bounds() exists for every kind at the pin"
    );
    assert!(!t.contains("prototype"), "P3.1 has landed");
    let st = lower(doc.split("## Status and dependencies").nth(1).unwrap());
    assert!(!st.contains("not final"), "P3.1 has landed");
}

#[test]
fn ac3_a_numeric_question_the_adapter_must_not_answer_is_routed_to_pvec() {
    let doc = adr();
    let t = lower(&clause(&doc, "NG-09"));
    for q in [
        "equal",
        "degenerate",
        "closed",
        "singular",
        "periodic",
        "magnitude",
    ] {
        assert!(
            t.contains(q),
            "NG-09 does not route the `{q}` question to the policy"
        );
    }
}

// ---------------------------------------------------------------- AC 4

#[test]
fn ac4_provenance_is_optional_component_data_and_the_record_works_without_it() {
    let doc = adr();
    let t = lower(&clause(&doc, "NG-06"));
    for needle in ["optional", "absent", "handle", "never inside", "component"] {
        assert!(t.contains(needle), "NG-06 does not say `{needle}`");
    }
    let rec = clause(&doc, "NG-02");
    assert!(
        rec.contains("Option<ProvenanceHandle>"),
        "the record must declare provenance as an Option of a handle"
    );
    assert!(
        lower(&rec).contains("provenance absent"),
        "NG-02 must state that the record is complete with provenance absent"
    );
}

// ---------------------------------------------------------------- AC 5

#[test]
fn ac5_tessellating_a_curve_in_libviprs_is_a_named_specification_violation() {
    let doc = adr();
    let v = lower(&clause(&doc, "NG-V1"));
    assert!(v.contains("specification violation"));
    assert!(v.contains("tessellat"));
    assert!(
        v.contains("ng-b2"),
        "NG-V1 must point at the prohibition it enforces"
    );
    assert!(
        v.contains("requirespvec") || v.contains("unsupported"),
        "NG-V1 must say what happens instead"
    );
}

// ------------------------------------------------- scope items the issue fixes

#[test]
fn record_has_the_five_fields_and_no_provider_type() {
    let doc = adr();
    let rec = clause(&doc, "NG-02");
    for f in [
        "geometry",
        "source_bounds",
        "provenance",
        "fidelity",
        "entity",
    ] {
        assert!(rec.contains(f), "NG-02 record lacks `{f}`");
    }
    let b = lower(&clause(&doc, "NG-08"));
    assert!(b.contains("provider type") && b.contains("decoder primitive"));
    assert!(
        b.contains("cad::primitive"),
        "NG-08 must name the decoder primitive type it keeps out"
    );
    assert!(b.contains("public signature"));
}

#[test]
fn coordinates_are_unconverted_source_space() {
    let doc = adr();
    let t = lower(&clause(&doc, "NG-04"));
    for n in ["f64", "projection", "unit", "source-space"] {
        assert!(t.contains(n), "NG-04 does not mention `{n}`");
    }
}

#[test]
fn fidelity_field_names_all_six_classes_and_leaves_the_enum_to_n3_5() {
    let doc = adr();
    let t = clause(&doc, "NG-07");
    for c in [
        "SupportedExact",
        "SupportedEquivalent",
        "RequiresPVecCapability",
        "RequiresExtension",
        "Unsupported",
        "IgnoredByPolicy",
    ] {
        assert!(t.contains(c), "NG-07 does not name {c}");
    }
    assert!(t.contains("N3.5") && lower(&t).contains("not redefined"));
}

#[test]
fn capability_gap_rule_files_against_p_and_lists_the_known_candidates() {
    let doc = adr();
    let t = clause(&doc, "NG-11");
    assert!(t.contains("RequiresPVecCapability"));
    assert!(lower(&t).contains("file") && t.contains("libviprs/pvec"));
    for cand in ["segment", "bulge", "ellipse"] {
        assert!(
            lower(&t).contains(cand),
            "NG-11 does not list the `{cand}` entry-point candidate"
        );
    }
}

// ------------------------------------- the place for classes and regions (pvec P3.9)

#[test]
fn semantic_class_and_regions_have_a_slot_outside_the_geometry_record() {
    let doc = adr();
    let t = clause(&doc, "NG-12");
    for n in [
        "ElementClass",
        "RegionRecord",
        "EntityKey",
        "libviprs/pvec#84",
        "N4.9",
        "N6.9",
        "Wall",
        "Column",
        "Room",
        "Corridor",
    ] {
        assert!(t.contains(n), "NG-12 does not mention {n}");
    }
    let tl = lower(&t);
    assert!(tl.contains("side table") && tl.contains("byte-identical"));
    assert!(
        tl.contains("must not share"),
        "NG-12 must keep the class enum apart from the N3.5 classification"
    );
    // the geometry record itself must carry nothing of the class stream
    let rec = lower(&clause(&doc, "NG-02"));
    let code: String = rec
        .split("```")
        .enumerate()
        .filter(|(i, _)| i % 2 == 1)
        .map(|(_, s)| s)
        .collect();
    for w in [
        "wall",
        "column",
        "room",
        "corridor",
        "region",
        "semantic",
        "element_class",
    ] {
        assert!(
            !code.contains(w),
            "the NG-02 record declaration mentions `{w}`; classes live beside it, not in it"
        );
    }
}

// ------------------------------------------------------------------ hygiene

#[test]
fn prose_hygiene() {
    let doc = adr();
    assert!(!doc.contains('\u{2014}'), "no em dashes");
    assert!(!doc.contains(" -- "), "no `--` standing in for an em dash");
    let l = lower(&doc);
    for w in [
        "claude",
        "co-authored",
        "generated by",
        "generated with",
        "anthropic",
    ] {
        assert!(!l.contains(w), "the ADR mentions `{w}`");
    }
    assert!(
        !l.contains("conforms to iso")
            && !l.contains("iso-compliant")
            && !l.contains("compliant with iso")
    );
}

#[test]
fn dependencies_on_unfinished_work_are_declared() {
    let doc = adr();
    let t = lower(&doc);
    assert!(t.contains("## status and dependencies"));
    for n in ["n3.1", "n3.5", "n1.3", "p3.9", "libviprs/pvec#84"] {
        assert!(t.contains(n), "dependency `{n}` not declared");
    }
}
