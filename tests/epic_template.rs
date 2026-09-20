//! The EPIC issue form still asks for all four things a feature has to deliver
//! (libviprs-org#85, which absorbed libviprs#1117).
//!
//! An epic is where the four live, because three of them are not things one PR
//! does: the benchmark capture happens on the measurement host, the libviprs.org
//! page is a different repository, and the end-to-end suite is a third. So the
//! checklist rides on the issue that owns all four, and this file is what stops
//! it quietly becoming three.
//!
//! # Why a test rather than trust
//!
//! The four items were not invented here. They are what EPIC F ended up
//! delivering, and it delivered them because the campaign ran long enough for
//! somebody to notice each one was missing, not because anything asked. A
//! checklist written down and then edited down is worse than no checklist,
//! because it reads as the standard while no longer being it, and nothing about
//! a shortened list looks wrong. That is the same shape as `merge-gate.yml`
//! claiming the crate had no `unsafe` of its own when it had ten (#897): a
//! load-bearing claim in prose with nothing verifying it.
//!
//! The canonical wording lives in libviprs-org (`EPIC_CHECKLIST.md`,
//! libviprs-org#85) and the form here is the mechanism that puts it in front of
//! whoever opens the epic. This test checks the mechanism still carries the four,
//! not that the two copies are word for word identical: the form is deliberately
//! terse and the doc is deliberately not, so a text comparison would either
//! force the doc into checkbox labels or go red on every edit to it.
//!
//! # What a naive version of this test would miss
//!
//! Matching each item's phrase against the whole file passes on a template that
//! moved the four into the description prose, where they render as a paragraph
//! nobody ticks and GitHub tracks as nothing. The point of the form is the
//! checkbox: an unticked box is visible on the issue, in the sub-issue rollup and
//! to anybody skimming, and a sentence is not. So the search is scoped to the
//! `checkboxes` block's own labels.
//!
//! The other half is worse. A matcher that cannot fail passes a template with
//! every item deleted just as happily as a complete one, and it looks exactly
//! like a test that works. The PMTiles goldens are the worked example: of the 134
//! non-control reader mutations, `leaves-z0z7` is blind to 60, meaning not one of
//! its 21912 probes moves, and it was the only one of the three original goldens
//! with leaves. So the fixture every leaf cell ran on could not have caught a
//! wrong answer on 60 of them, and nothing about that looked wrong from the
//! outside. Every assertion below therefore has a control that deletes what it is
//! looking for and requires the same matcher to go red.

/// The form itself, read at compile time so a deleted file is a build failure
/// rather than a test that quietly stops having anything to check.
const TEMPLATE: &str = include_str!("../.github/ISSUE_TEMPLATE/epic.yml");

/// The four deliverables, each as the fragments its label has to contain.
///
/// Fragments rather than a whole sentence, so rewording a label for clarity does
/// not red the build, while dropping the subject of the item does. The second
/// item carries three: a benchmark checkbox that forgot either comparison is the
/// specific way this one gets hollowed out, because running a benchmark at all
/// feels like having done it.
const REQUIRED_ITEMS: &[(&str, &[&str])] = &[
    ("end-to-end tests", &["libviprs-tests", "fixture"]),
    (
        "benchmarks with both comparisons",
        &["benchmark", "previous", "librar"],
    ),
    ("CLI commands", &["CLI", "command"]),
    ("documentation", &["libviprs.org", "document"]),
];

/// The `checkboxes` block's labels, lowercased, in file order.
///
/// A deliberately small parser rather than a YAML dependency: `CONTRIBUTING.md`
/// sets a high bar for one and a hygiene test is nowhere near clearing it. It
/// reads the `options:` list that follows the `type: checkboxes` line and stops
/// at the next element, which is all the structure the assertions need.
fn checkbox_labels(template: &str) -> Vec<String> {
    let mut labels = Vec::new();
    let mut inside = false;

    for line in template.lines() {
        let trimmed = line.trim();

        if trimmed == "- type: checkboxes" || trimmed == "type: checkboxes" {
            inside = true;
            continue;
        }
        // A new element at the top of the list ends the block. `- type:` is the
        // only thing that can start one, so a `- label:` further in is safe.
        if inside && trimmed.starts_with("- type:") {
            inside = false;
            continue;
        }
        if !inside {
            continue;
        }

        if let Some(label) = trimmed
            .strip_prefix("- label:")
            .or_else(|| trimmed.strip_prefix("label:"))
        {
            labels.push(label.trim().trim_matches('"').to_lowercase());
        }
    }

    labels
}

/// Whether every required item has a checkbox label carrying all its fragments.
///
/// The predicate the real assertion and its control both run, so the control
/// tests the thing that actually guards the file rather than a second
/// implementation of it that could differ.
fn every_item_has_a_checkbox(template: &str) -> bool {
    let labels = checkbox_labels(template);
    REQUIRED_ITEMS.iter().all(|(_, fragments)| {
        labels.iter().any(|label| {
            fragments
                .iter()
                .all(|fragment| label.contains(&fragment.to_lowercase()))
        })
    })
}

#[test]
fn the_form_asks_for_all_four_deliverables() {
    let labels = checkbox_labels(TEMPLATE);

    assert!(
        !labels.is_empty(),
        "the epic form has no checkboxes block, so the four deliverables are \
         prose at best and nobody ticks prose"
    );

    for (name, fragments) in REQUIRED_ITEMS {
        let found = labels.iter().any(|label| {
            fragments
                .iter()
                .all(|fragment| label.contains(&fragment.to_lowercase()))
        });
        assert!(
            found,
            "no checkbox in the epic form covers {name}. Its label has to carry \
             all of {fragments:?}, and the labels present are {labels:#?}"
        );
    }
}

#[test]
fn deleting_an_item_reddens_the_check() {
    // The control. Without it, a parser that returned an empty label list would
    // make `every_item_has_a_checkbox` pass vacuously on any input, including a
    // template with nothing in it, and the test above would look like it was
    // working for as long as nobody checked.
    assert!(
        every_item_has_a_checkbox(TEMPLATE),
        "the template as committed has to satisfy the predicate for this control \
         to mean anything"
    );

    for (name, fragments) in REQUIRED_ITEMS {
        let without = TEMPLATE
            .lines()
            .filter(|line| {
                let lowered = line.to_lowercase();
                !(lowered.contains("label:")
                    && fragments
                        .iter()
                        .all(|fragment| lowered.contains(&fragment.to_lowercase())))
            })
            .collect::<Vec<_>>()
            .join("\n");

        assert_ne!(
            without, TEMPLATE,
            "the control for {name} removed nothing, so it proves nothing"
        );
        assert!(
            !every_item_has_a_checkbox(&without),
            "dropping the {name} checkbox left the check green, so it is not \
             guarding that item"
        );
    }
}

#[test]
fn the_form_states_the_epic_size_limits() {
    let lowered = TEMPLATE.to_lowercase();

    // The other thing every epic re-derives. Two phases, five issues a phase, so
    // ten is the ceiling, and an epic that cannot fit is really two epics.
    assert!(
        lowered.contains("2 phases") || lowered.contains("two phases"),
        "the epic form does not state the phase cap, so the next epic invents one"
    );
    assert!(
        lowered.contains("5 issues") || lowered.contains("five issues"),
        "the epic form does not state the per-phase issue cap"
    );
}

#[test]
fn the_form_points_at_the_canonical_checklist() {
    // Terse labels are the right call on a form and they are not the whole
    // standard: each item has a trap behind it that a one-line label cannot
    // carry, and those live in libviprs-org. A form that does not link out
    // leaves whoever opens the epic with four sentences and no reasons.
    assert!(
        TEMPLATE.contains("EPIC_CHECKLIST.md"),
        "the epic form does not link to EPIC_CHECKLIST.md in libviprs-org, so the \
         reasoning behind each item is unreachable from where it is asked for"
    );
}
