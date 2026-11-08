//! Slice 1 of the layouts epic (#1557, issue #1559): the print profile is written down and pinned
//! before any code depends on it.
//!
//! `ViewProfile::print_profile(layout)` is pinned field by field beside `default_profile()`, which
//! keeps its V1 values. The ADR `docs/adr/pvec-print-profile.md` holds P-1 to P-9, and
//! `docs/adr/pvec-unsupported-policy.md` no longer refuses paper space as a whole. A print rule
//! whose implementation has not landed is refused, never ignored.

use libviprs::pvec::{
    InsertLayerRule, LayerStateRule, LayerVisibility, NestedInsertPolicy, SourceLimits,
    SpaceSelection, ViewProfile, ViewportClipping, ViewportFreezeRule, XrefPolicy,
};

fn doc(path: &str) -> String {
    std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_else(|e| panic!("{path}: {e}"))
}

#[test]
fn issue_1559_print_profile_is_pinned_field_by_field() {
    let expected = ViewProfile {
        space: SpaceSelection::PaperLayout("A-101".into()),
        viewport_clipping: ViewportClipping::Exact,
        layers: LayerVisibility {
            off: LayerStateRule::Exclude,
            frozen: LayerStateRule::Exclude,
            no_plot: LayerStateRule::Exclude,
            frozen_in_viewport: ViewportFreezeRule::Apply,
            insert_override: InsertLayerRule::InheritOnLayerZero,
        },
        nested_inserts: NestedInsertPolicy { max_depth: 64 },
        xrefs: XrefPolicy::ReportExternal,
        limits: SourceLimits::default(),
    };
    assert_eq!(ViewProfile::print_profile("A-101"), expected);
}

#[test]
fn issue_1559_the_default_profile_keeps_no_plot_layers_visible() {
    let p = ViewProfile::default_profile();
    assert_eq!(p.space, SpaceSelection::Model);
    assert_eq!(p.viewport_clipping, ViewportClipping::Unsupported);
    assert_eq!(p.layers.no_plot, LayerStateRule::Ignore);
    assert_eq!(
        p.layers.frozen_in_viewport,
        ViewportFreezeRule::IgnoreAndReport
    );
    assert!(p.to_string().contains("layers.no_plot = ignore"), "{p}");
}

#[test]
fn issue_1559_print_profile_prints_every_knob() {
    let text = ViewProfile::print_profile("A-101").to_string();
    for needle in [
        "space = paper:A-101",
        "viewport_clipping = exact",
        "layers.off = exclude",
        "layers.frozen = exclude",
        "layers.no_plot = exclude",
        "layers.frozen_in_viewport = apply",
        "layers.insert_override = inherit-on-layer-zero",
        "nested_inserts.max_depth = 64",
        "xrefs = report-external",
    ] {
        assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1559_every_print_knob_names_a_row_of_a_decision_table() {
    let tables = doc("docs/spikes/view-context.md") + &doc("docs/adr/pvec-print-profile.md");
    for k in ViewProfile::print_profile("A-101").knobs() {
        assert!(
            tables.contains(&format!("| `{}` |", k.row)),
            "{} names row {:?}, which no decision table has",
            k.field,
            k.row
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1559_the_adr_holds_p1_to_p9_and_what_is_still_unmeasured() {
    let adr = doc("docs/adr/pvec-print-profile.md");
    let status = adr
        .lines()
        .find(|l| l.starts_with("Status:"))
        .expect("status line");
    assert!(status.starts_with("Status: Frozen"), "{status}");
    for n in 1..=9 {
        assert!(adr.contains(&format!("**P-{n} ")), "P-{n} is missing");
    }
    for needle in [
        "print_profile",
        "default_profile",
        "spikes/view-context.md",
        "P1.2",
        "no epsilon of its own",
        "paper-space-multi-viewport-001",
        "viewport-clipped-001",
        "viewport-clipped-bounded-001",
        "pvec-gold-fixtures",
        "twist",
        "## Open items",
    ] {
        assert!(adr.contains(needle), "the ADR does not say {needle:?}");
    }
    let open = &adr[adr.find("## Open items").unwrap()..];
    assert!(open.contains("measure"), "open items name no measurement");
    assert!(!adr.contains('\u{2014}') && !adr.contains(" -- "));
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1559_the_unsupported_policy_no_longer_refuses_paper_space_as_a_whole() {
    let policy = doc("docs/adr/pvec-unsupported-policy.md");
    assert!(
        !policy
            .contains("- Paper space and viewport clips: `Unsupported`, `VIEW_NOT_ENABLED` (508)."),
        "line 47 still refuses paper space as a whole"
    );
    for needle in [
        "pvec-print-profile.md",
        "per viewport",
        "per entity",
        "VIEW_NOT_ENABLED",
    ] {
        assert!(
            policy.contains(needle),
            "the amendment does not say {needle:?}"
        );
    }
}

#[cfg(feature = "dwg-acadrust")]
mod with_a_drawing {
    use libviprs::cad::CadSource;
    use libviprs::pvec::dwg_acadrust::AcadrustProvider;
    use libviprs::pvec::{DwgProvider, LayerStateRule, LayoutKind, ViewProfile, ViewProfileError};

    fn sample() -> AcadrustProvider {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/cad/sample_AC1018.dwg");
        AcadrustProvider::open(CadSource::Path(&path)).unwrap()
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn issue_1559_the_print_profile_is_refused_until_the_sheet_lands() {
        let provider = sample();
        let layout = provider
            .layouts()
            .unwrap()
            .into_iter()
            .find(|l| l.kind == LayoutKind::Paper)
            .expect("sample_AC1018 has a paper layout")
            .name;
        match ViewProfile::print_profile(layout.clone()).apply(&provider) {
            Err(ViewProfileError::PaperSpaceUnsupported { layout: l }) => assert_eq!(l, layout),
            other => panic!("a print must be refused until slice 3, got {other:?}"),
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn issue_1559_excluding_no_plot_layers_is_refused_not_ignored() {
        let mut p = ViewProfile::default_profile();
        p.layers.no_plot = LayerStateRule::Exclude;
        match p.apply(&sample()) {
            Err(ViewProfileError::RuleNotEnabled { field }) => assert_eq!(field, "layers.no_plot"),
            other => {
                panic!("a no-plot exclusion must be refused until it is applied, got {other:?}")
            }
        }
    }
}
