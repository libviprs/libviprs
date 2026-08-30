//! Every claim the CHANGELOG makes about what shipped is checked against the
//! tags (issue #947).
//!
//! The `### Breaking` entry for `ConvolutionError::TimesOutOfRange` closed with
//! "`ZeroTimes` has never been in a release, and the enum is
//! `#[non_exhaustive]`". `ZeroTimes` shipped in **v0.4.0**: it is at
//! `v0.4.0:src/convolution.rs:182`, constructed at `:839` and documented at
//! `:821`, introduced on 2026-07-11 in `2b9f9caf`, nine days before the tag on
//! 2026-07-20.
//!
//! That is the sentence a 0.4.0 user matching that variant reads to decide
//! whether the removal affects them, and it told them it could not. Same class
//! as `merge-gate.yml` claiming the crate had no `unsafe` of its own when it
//! had ten (#897): a load-bearing factual claim in prose with nothing
//! verifying it.
//!
//! # The claim is decidable, which is the whole point
//!
//! "Has this identifier ever been in a release" is not a judgement call. It is
//! `git tag --contains`, or here `git grep <ident> <tag>`, which also catches
//! an identifier that arrived and left inside one release. So the fix is not
//! only correcting the sentence: it is that the class of sentence now has a
//! checker, and the next one written is either true or red.
//!
//! # Retracted text
//!
//! A claim wrapped in `~~ ~~` is struck through, which is markdown for
//! retracted, and the scan strips those spans before reading. That is how the
//! CHANGELOG keeps the wrong sentence visible in the record (the convention the
//! #501 and #920 entries set) without this reading it as live. The stripper has
//! its own control below, because a stripper that removed everything would make
//! every claim disappear and this pass on nothing.
//!
//! # `MIGRATION.md` makes the same kind of claim
//!
//! Issue #961's 0.4.0-to-0.5.0 section says
//! "`ConvolutionError::ZeroTimes` shipped in `v0.4.0`" and
//! "`ConversionError::UnsupportedSampleKind` has never been in a release",
//! the exact two phrasings this file already checks, about the exact two
//! identifiers this file already carries as controls for `CHANGELOG.md`. A
//! migration guide making an unchecked claim about what shipped is the same
//! failure this file exists to catch, so the scan runs over both documents
//! rather than staying CHANGELOG-only and leaving the second copy to drift on
//! its own.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

const CHANGELOG: &str = include_str!("../CHANGELOG.md");
const MIGRATION: &str = include_str!("../MIGRATION.md");

/// The phrase that says an identifier was never released.
const NEVER: &str = "has never been in a release";

/// The phrase that says which tag an identifier first shipped in.
///
/// The tag after it may be backticked or bare. Requiring the backticks left a
/// real hole: a true claim written `shipped in v0.4.0` was skipped, so a false
/// one would have been too. Measured, as row C7 of this PR's mutation table
/// before this widened.
const SHIPPED: &str = " shipped in ";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// The document with every `~~struck~~` span removed.
fn without_retractions(doc: &str) -> String {
    let mut out = String::with_capacity(doc.len());
    let mut rest = doc;
    while let Some(i) = rest.find("~~") {
        out.push_str(&rest[..i]);
        rest = &rest[i + 2..];
        match rest.find("~~") {
            Some(j) => rest = &rest[j + 2..],
            // An unterminated `~~` retracts nothing; keep the tail so a typo
            // cannot silently swallow the rest of the file.
            None => break,
        }
    }
    out.push_str(rest);
    out
}

/// The backtick-quoted token immediately before `at` on the same line.
fn identifier_before(doc: &str, at: usize) -> &str {
    let line_start = doc[..at].rfind('\n').map_or(0, |i| i + 1);
    let head = &doc[line_start..at];
    let close = head
        .rfind('`')
        .unwrap_or_else(|| panic!("no backticked identifier before {:?}", &doc[at..at + 40]));
    let open = head[..close]
        .rfind('`')
        .unwrap_or_else(|| panic!("unbalanced backticks before {:?}", &doc[at..at + 40]));
    &head[open + 1..close]
}

/// Every tag in the repository.
fn tags() -> Vec<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo_root())
        .arg("tag")
        .output()
        .expect("failed to spawn git tag");
    assert!(
        out.status.success(),
        "git tag failed, so this guard has nothing to check:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Whether `tag`'s `src/` contains `needle` anywhere.
fn tag_src_contains(tag: &str, needle: &str) -> bool {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo_root())
        .args(["grep", "-F", "-q", "--", needle, tag, "--", "src"])
        .output()
        .expect("failed to spawn git grep");
    // 0 is a match, 1 is no match; anything else is a broken invocation and
    // must not read as "no match".
    match out.status.code() {
        Some(0) => true,
        Some(1) => false,
        other => panic!(
            "git grep for {needle:?} in {tag} exited {other:?}:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ),
    }
}

/**
 * Tests that every release claim in `CHANGELOG.md` and `MIGRATION.md` is true
 * of the tags (issues #947, #961).
 *
 * Two phrasings are claims: "`X` has never been in a release", which requires
 * `X` to be absent from every tag, and "`X` shipped in `vN`", which requires it
 * to be present in that one. Both are answered by `git grep` over the tag's
 * `src/`, which catches an identifier that arrived and left inside a single
 * release as well as one that survived. `MIGRATION.md` picked up the same two
 * phrasings once its 0.4.0-to-0.5.0 section existed, about the same two
 * identifiers this file already carries controls for, so the scan runs over
 * both documents rather than leaving the second copy unchecked.
 *
 * Four controls, because every step here has a state where it agrees with
 * everything:
 *
 * * the tag listing has to hold the tags this repository is known to have, so
 *   a clone without them fails loudly instead of finding no counterexamples.
 *   That is the case that would otherwise make this pass in CI for a reason
 *   unrelated to the CHANGELOG;
 * * the grep is checked in both directions on known answers, because an
 *   invocation that always returns "no match" satisfies every "never" claim;
 * * at least one live claim has to be found **in each document**, because a
 *   scan that has stopped parsing one of them agrees with any claim written
 *   there;
 * * the retraction stripper is run on a planted string, because one that
 *   removed everything would empty the document.
 */
#[test]
#[cfg_attr(miri, ignore)] // spawns git, blocked by Miri isolation
fn every_release_claim_in_the_changelog_is_true_of_the_tags() {
    // Control 1: the tags are here.
    let tags = tags();
    let have: BTreeSet<&str> = tags.iter().map(String::as_str).collect();
    for anchor in ["v0.1.1", "v0.2.0", "v0.4.0"] {
        assert!(
            have.contains(anchor),
            "git lists {tags:?} and {anchor} is not among them, so this is not \
             the tag set the guard means to read. A shallow clone without tags \
             would otherwise satisfy every \"never released\" claim by finding \
             nothing"
        );
    }

    // Control 2: the grep answers both ways on known cases.
    assert!(
        tag_src_contains("v0.4.0", "ZeroTimes"),
        "positive control: `ZeroTimes` is in v0.4.0's src/, at \
         src/convolution.rs:182"
    );
    assert!(
        !tag_src_contains("v0.4.0", "TimesOutOfRange"),
        "negative control: `TimesOutOfRange` replaced it after the tag"
    );

    // Control 4: the stripper strips the span and nothing else.
    let stripped = without_retractions("keep ~~drop~~ keep2 ~~drop2~~ tail");
    assert_eq!(
        stripped, "keep  keep2  tail",
        "the retraction stripper is wrong"
    );

    let mut claims = 0usize;

    for (file, doc) in [("CHANGELOG.md", CHANGELOG), ("MIGRATION.md", MIGRATION)] {
        let live = without_retractions(doc);
        let mut claims_here = 0usize;

        // "`X` has never been in a release"
        let mut from = 0usize;
        while let Some(i) = live[from..].find(NEVER) {
            let at = from + i;
            let ident = identifier_before(&live, at);
            claims += 1;
            claims_here += 1;
            let released: Vec<&String> =
                tags.iter().filter(|t| tag_src_contains(t, ident)).collect();
            assert!(
                released.is_empty(),
                "{file} says `{ident}` has never been in a release, and it is \
                 in {released:?}. That sentence is what a user of those \
                 releases reads to decide whether a removal affects them \
                 (issue #947)"
            );
            from = at + NEVER.len();
        }

        // "`X` shipped in vN", with or without backticks round the tag.
        let mut from = 0usize;
        while let Some(i) = live[from..].find(SHIPPED) {
            let at = from + i;
            let tail = &live[at + SHIPPED.len()..];
            let tail = tail.strip_prefix('`').unwrap_or(tail);
            let end = tail
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '.'))
                .unwrap_or(tail.len());
            let tag = &tail[..end];
            from = at + SHIPPED.len();
            // Only version tags are claims about a release; "shipped in
            // `#547`" or any other backticked thing is prose.
            if !tag.starts_with('v') || !have.contains(tag) {
                continue;
            }
            let ident = identifier_before(&live, at);
            claims += 1;
            claims_here += 1;
            assert!(
                tag_src_contains(tag, ident),
                "{file} says `{ident}` shipped in {tag}, and it is not in \
                 that tag's src/"
            );
        }

        // Control 3, per document: something was actually examined in each
        // one, because an empty MIGRATION.md-side scan would agree with any
        // claim added there later, exactly the hole a CHANGELOG-only version
        // of this control has.
        assert!(
            claims_here >= 1,
            "the claim scan found nothing in {file}, so it has stopped \
             parsing and would agree with any release claim written there"
        );
    }

    assert!(
        claims >= 2,
        "the claim scan found {claims} claim(s) total across both documents, \
         fewer than the two per-document positive controls above require"
    );
}
