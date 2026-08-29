//! Every issue number the `Unreleased` preamble names has to exist as a
//! `### Breaking` entry (issue #636).
//!
//! # One direction, on purpose
//!
//! #636 gathered the breaking changes into one section and gave `Unreleased` a
//! preamble grouping them, because they had been spread across `Breaking`,
//! `Changed` and `Fixed` with nothing saying so. A summary like that decays,
//! and the obvious guard is set equality against the entries. This file
//! deliberately checks only the **subset** direction.
//!
//! Equality would pin a prose-derived set in the file every lane also edits,
//! and `CHANGELOG.md` is the single most reliable source of merge conflict in
//! this repository, resolved by union every time. A lane adding a breaking
//! entry would have to move a second file in lockstep or go red, which is the
//! shared-count hazard that has put `main` red twice here, spent on a summary
//! paragraph rather than a contract.
//!
//! The subset direction has none of that. Adding an entry can never break it,
//! because the preamble is a subset by construction. What it catches is the
//! preamble naming an issue that has been removed, renamed or mistyped:
//!
//! * a summary that is **incomplete** is stale, and a reader loses nothing,
//! * a summary that names something **absent** is wrong, and sends a reader
//!   looking for an entry that is not there.
//!
//! Only the second is worth a test.
//!
//! # Read the file as it is actually written
//!
//! The check that produced #636's evidence got this wrong first, and its
//! mistake is pinned below rather than described. Three shapes in this file
//! defeat the obvious parse:
//!
//! 1. **An attribution splits across a line break.** `Extend::White` is filed
//!    `(issue\n  #667)`, so a search for `issue #` on one line misses it and
//!    falls through to a secondary `#694` twenty lines further down. That is
//!    the same family as reading a comment as code: the scanner answers
//!    confidently about the wrong text. Everything here normalises whitespace
//!    before it reads an attribution.
//! 2. **The preamble names its own section in prose.** It says "gathered in
//!    `### Breaking` below", so a substring search for the heading stops
//!    inside the preamble. Headings are matched at the start of a line.
//! 3. **An entry cites issues it is not filed under.** The attribution is the
//!    *first* `issue #N` run in the entry, not every `#N` in it.

use std::collections::BTreeSet;

/// The changelog is embedded at compile time rather than read at run time, so
/// this test reaches no filesystem: it needs no `#[cfg_attr(miri, ignore)]`,
/// no row in `tests/miri_fs_test_inventory.txt`, and it does not move
/// `EXPECTED_FS_TOUCHING_TESTS`.
const CHANGELOG: &str = include_str!("../CHANGELOG.md");

/// A number no issue in this repository will reach, used by the negative
/// controls so a mutation cannot collide with a real attribution.
const FABRICATED: u32 = 999_001;

/// The `## [Unreleased]` block, from its heading to the next released version.
///
/// Headings are matched at the start of a line because the preamble quotes
/// `### Breaking` inside backticks, and a released heading could be quoted the
/// same way.
fn unreleased(changelog: &str) -> &str {
    let start = changelog
        .find("## [Unreleased]")
        .expect("CHANGELOG.md must have an `## [Unreleased]` block");
    let rest = &changelog[start + "## [Unreleased]".len()..];
    let end = rest
        .match_indices("\n## ")
        .next()
        .map(|(i, _)| i)
        .unwrap_or(rest.len());
    &rest[..end]
}

/// The prose between `## [Unreleased]` and the first `###` section.
fn preamble(unreleased: &str) -> &str {
    let end = unreleased
        .match_indices("\n### ")
        .next()
        .map(|(i, _)| i)
        .unwrap_or(unreleased.len());
    &unreleased[..end]
}

/// The raw text of each top-level bullet under `### Breaking`.
///
/// A bullet runs from a line starting `- ` to the next such line, the next
/// `###` heading, or the end of the block, so a nested list or an indented
/// code block inside an entry stays with it.
fn breaking_entries(unreleased: &str) -> Vec<&str> {
    let Some(head) = unreleased.find("\n### Breaking\n") else {
        return Vec::new();
    };
    let section = &unreleased[head + "\n### Breaking\n".len()..];
    let section = match section.match_indices("\n### ").next() {
        Some((i, _)) => &section[..i],
        None => section,
    };

    let mut starts: Vec<usize> = Vec::new();
    if section.starts_with("- ") {
        starts.push(0);
    }
    starts.extend(section.match_indices("\n- ").map(|(i, _)| i + 1));

    let mut out = Vec::new();
    for (n, &s) in starts.iter().enumerate() {
        let e = starts.get(n + 1).copied().unwrap_or(section.len());
        out.push(section[s..e].trim_end());
    }
    out
}

/// Every `#<digits>` in `text`, whatever it is doing there.
fn issue_numbers(text: &str) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'#' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > i + 1 {
                if let Ok(n) = text[i + 1..j].parse::<u32>() {
                    out.insert(n);
                }
                i = j;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// The issues an entry is *filed under*: the first `issue #N` or
/// `issues #A, #B` run in it.
///
/// The entry is whitespace-normalised first, so an attribution written across
/// a line break reads the same as one written on a single line. Numbers cited
/// later in the entry are not attributions and are not returned.
fn attribution(entry: &str) -> BTreeSet<u32> {
    let flat = entry.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = BTreeSet::new();

    let singular = flat.find("issue #");
    let plural = flat.find("issues #");
    let kw = match (singular, plural) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => return out,
    };
    let mut rest = &flat[kw..];
    // Step over the keyword itself so the scan starts on the first `#`.
    let hash = rest.find('#').expect("the keyword match contains a `#`");
    rest = &rest[hash..];

    loop {
        if !rest.starts_with('#') {
            break;
        }
        let digits: String = rest[1..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if digits.is_empty() {
            break;
        }
        if let Ok(n) = digits.parse::<u32>() {
            out.insert(n);
        }
        rest = &rest[1 + digits.len()..];
        // Only a separator that joins a list continues it. Anything else, a
        // comma followed by prose included, ends the attribution.
        let sep = [", and #", " and #", ", #"]
            .iter()
            .copied()
            .find(|s| rest.starts_with(s));
        match sep {
            Some(s) => rest = &rest[s.len() - 1..],
            None => break,
        }
    }
    out
}

/// The preamble issue numbers that no `Breaking` entry is filed under.
fn unbacked(changelog: &str) -> BTreeSet<u32> {
    let block = unreleased(changelog);
    let named = issue_numbers(preamble(block));
    let filed: BTreeSet<u32> = breaking_entries(block)
        .iter()
        .flat_map(|e| attribution(e))
        .collect();
    named.difference(&filed).copied().collect()
}

/// The parse finds a preamble and entries at all.
///
/// Without this the real assertion below is vacuous in the one direction that
/// matters: a preamble parse that silently returns nothing has an empty set to
/// compare, and an empty set is a subset of everything. A zero has two
/// explanations, and this is the positive control that separates them.
#[test]
fn the_parse_finds_a_preamble_and_breaking_entries() {
    let block = unreleased(CHANGELOG);

    let named = issue_numbers(preamble(block));
    assert!(
        named.len() >= 5,
        "the `Unreleased` preamble should name the issues behind the breaking \
         changes, and the parse found {}. Either the preamble was removed, or \
         this file no longer knows how to find it.",
        named.len()
    );

    let entries = breaking_entries(block);
    assert!(
        entries.len() >= 5,
        "expected the `### Breaking` section to hold entries, found {}",
        entries.len()
    );

    let filed: BTreeSet<u32> = entries.iter().flat_map(|e| attribution(e)).collect();
    assert!(
        filed.len() >= 5,
        "expected the `### Breaking` entries to be filed under issues, and the \
         parse read {} attribution(s) from {} entries",
        filed.len(),
        entries.len()
    );
}

/// Every issue the preamble names exists as a `Breaking` entry.
///
/// This is the contract. It says nothing about entries the preamble omits,
/// for the reason in the module doc.
#[test]
fn every_issue_the_preamble_names_has_a_breaking_entry() {
    let missing = unbacked(CHANGELOG);
    assert!(
        missing.is_empty(),
        "the `Unreleased` preamble names {} issue(s) that no `### Breaking` \
         entry is filed under: {}\n\n\
         Either the entry was removed or renamed and the preamble still points \
         at it, or the number is a typo. Fix whichever is wrong. Adding a \
         breaking entry the preamble does not mention is fine and this test \
         does not ask for it.",
        missing.len(),
        missing
            .iter()
            .map(|n| format!("#{n}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// Dropping an issue from an entry's attribution reports exactly that issue.
///
/// The negative control for the real direction of decay: an entry stops being
/// filed under the number the preamble points at. The mutation is asserted
/// before it is believed, because a mutation that fails to apply reports green
/// and is indistinguishable from a test that cannot fail.
#[test]
fn dropping_an_issue_from_an_entry_leaves_exactly_that_issue_unbacked() {
    let block = unreleased(CHANGELOG);
    let named = issue_numbers(preamble(block));
    let entries = breaking_entries(block);

    // A number the preamble names that exactly one entry is filed under, so
    // removing it from that entry cannot be masked by a second entry.
    let (victim, entry) = entries
        .iter()
        .filter_map(|e| {
            let a = attribution(e);
            let only: Vec<u32> = a.intersection(&named).copied().collect();
            match only.as_slice() {
                [n] => Some((*n, *e)),
                _ => None,
            }
        })
        .find(|(n, _)| {
            entries
                .iter()
                .filter(|e| attribution(e).contains(n))
                .count()
                == 1
        })
        .expect("expected at least one preamble issue backed by exactly one entry");

    let needle = format!("#{victim}");
    let at = entry
        .find("issue")
        .and_then(|k| entry[k..].find(&needle).map(|o| k + o))
        .expect("the attribution follows the word `issue` in the entry");
    let mutated_entry = format!(
        "{}#{}{}",
        &entry[..at],
        FABRICATED,
        &entry[at + needle.len()..]
    );

    // Assert what the mutation did, not merely that an edit happened.
    let after = attribution(&mutated_entry);
    assert!(
        !after.contains(&victim) && after.contains(&FABRICATED),
        "the mutation did not move the attribution off #{victim}: {after:?}"
    );
    assert_eq!(
        mutated_entry.matches(&needle).count(),
        entry.matches(&needle).count() - 1,
        "the mutation should remove exactly one `{needle}` from the entry"
    );

    let mutated = CHANGELOG.replacen(entry, &mutated_entry, 1);
    assert_ne!(
        mutated, CHANGELOG,
        "the mutation did not reach the document"
    );

    let missing = unbacked(&mutated);
    assert_eq!(
        missing,
        BTreeSet::from([victim]),
        "removing #{victim} from the only entry filed under it should leave \
         exactly #{victim} unbacked"
    );
}

/// An issue number in the preamble that no entry carries is named.
///
/// The control in the new direction: the preamble grows a reference to
/// something that is not there.
#[test]
fn a_fabricated_issue_number_in_the_preamble_is_named() {
    assert!(
        unbacked(CHANGELOG).is_empty(),
        "this control starts from a clean document"
    );

    let block = unreleased(CHANGELOG);
    let pre = preamble(block);
    let grown = format!("{pre}\nA sentence naming issue #{FABRICATED}.\n");
    let mutated = CHANGELOG.replacen(pre, &grown, 1);
    assert_ne!(
        mutated, CHANGELOG,
        "the mutation did not reach the document"
    );

    let missing = unbacked(&mutated);
    assert_eq!(
        missing,
        BTreeSet::from([FABRICATED]),
        "a preamble naming #{FABRICATED}, which no entry is filed under, must \
         be reported"
    );
}

/// An attribution split across a line break is still read, and a number cited
/// deeper in the same entry is not mistaken for it.
///
/// This is the failure the check behind #636 actually hit, kept as a test
/// rather than a warning. The fixture carries both halves: the attribution
/// wraps after the word `issue`, and a second number appears further down, so
/// a parse that reads one line at a time returns the wrong answer instead of
/// no answer.
#[test]
fn an_attribution_split_across_a_line_break_is_read_not_skipped() {
    let entry = "- `Extend::White` inks its fill from the raster's interpretation instead of\n  \
                 from its sample depth, so a float raster tagged `ScRgb` fills with `1.0` (issue\n  \
                 #667). That covers `embed` and `gravity` in `extract`.\n\n  \
                 Whether the interpolating resamplers follow is `resample`'s alone for now\n  \
                 (issue #694).";

    assert_eq!(
        attribution(entry),
        BTreeSet::from([667]),
        "the attribution wraps after `issue`, and normalising whitespace first \
         is what finds it"
    );

    // The control that makes the assertion above capable of failing: the wrong
    // answer is present in the entry and reachable, so returning it would have
    // been a plausible result rather than an obvious error.
    assert!(
        issue_numbers(entry).contains(&694),
        "the fixture must carry a second number the parse could wrongly pick"
    );
    assert!(
        !attribution(entry).contains(&694),
        "#694 is cited by this entry, not the issue it is filed under"
    );
}

/// The attribution reads a list, and stops at prose.
///
/// `(issues #516, #759)` is two attributions; `(issue #664, found while
/// measuring the above)` is one, and the comma after it does not open a list.
#[test]
fn an_attribution_reads_a_list_and_stops_at_prose() {
    assert_eq!(
        attribution("- Something changed (issues #516, #759). Body."),
        BTreeSet::from([516, 759])
    );
    assert_eq!(
        attribution("- Something changed (issues #748 and #607). Body."),
        BTreeSet::from([748, 607])
    );
    assert_eq!(
        attribution("- Something changed (issue #664, found while measuring). Body #999."),
        BTreeSet::from([664])
    );
    assert_eq!(
        attribution("- Something changed (issue #339's class). Body."),
        BTreeSet::from([339])
    );
    assert!(
        attribution("- Something changed with no attribution at all.").is_empty(),
        "an entry filed under nothing contributes nothing, which can only make \
         the subset check stricter"
    );
}

/// A heading quoted in the preamble's own prose does not end the preamble.
///
/// The second shape from the module doc, kept as a fixture rather than a
/// warning. The live preamble says "gathered in `### Breaking` below", so a
/// parse that searches for the heading as a bare substring stops inside the
/// preamble, reads no entries, and then passes the subset check vacuously
/// because an empty set is a subset of everything. That is the worst kind of
/// green: the guard reports success precisely when it has stopped working.
#[test]
fn a_heading_quoted_in_prose_does_not_end_the_preamble() {
    let doc = "# Changelog\n\n\
               ## [Unreleased]\n\n\
               Everything that breaks is gathered in `### Breaking` below, and \
               issue #4242 is one of them.\n\n\
               ### Breaking\n\n\
               - A thing broke (issue #4242). Body of the entry.\n\n\
               ### Added\n\n\
               - Something new (issue #7).\n\n\
               ## [0.4.0] - 2026-07-20\n\n\
               ### Breaking\n\n\
               - An older break (issue #1).\n";

    let block = unreleased(doc);
    assert!(
        preamble(block).contains("`### Breaking`"),
        "the inline mention belongs to the preamble"
    );

    let entries = breaking_entries(block);
    assert_eq!(
        entries.len(),
        1,
        "the real heading, at the start of a line, is what ends the preamble \
         and opens the section: {entries:?}"
    );
    assert_eq!(attribution(entries[0]), BTreeSet::from([4242]));

    // The whole point: with the section found, the subset check is answerable.
    assert!(unbacked(doc).is_empty());

    // And it fails when it should, on the same shaped document.
    let broken = doc.replace("(issue #4242). Body", "(issue #4243). Body");
    assert_eq!(unbacked(&broken), BTreeSet::from([4242]));
}

/// The `Unreleased` block stops at the next released version.
///
/// A released section further down also has a `### Breaking` heading, and
/// reading its entries would let a stale preamble reference resolve against a
/// shipped release rather than this one.
#[test]
fn the_unreleased_block_stops_at_the_next_release() {
    let doc = "# Changelog\n\n\
               ## [Unreleased]\n\n\
               Naming issue #10.\n\n\
               ### Breaking\n\n\
               - A thing broke (issue #10).\n\n\
               ## [0.4.0] - 2026-07-20\n\n\
               ### Breaking\n\n\
               - An older break (issue #11).\n";

    let block = unreleased(doc);
    assert!(
        !block.contains("0.4.0"),
        "the block ran into the release below"
    );
    let filed: BTreeSet<u32> = breaking_entries(block)
        .iter()
        .flat_map(|e| attribution(e))
        .collect();
    assert_eq!(filed, BTreeSet::from([10]), "#11 belongs to 0.4.0");
}
