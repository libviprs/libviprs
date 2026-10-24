//! The DWG versions `docs/vector-providers.md` claims (issue #1294): the sentence
//! has to agree with `SUPPORTED_DWG_VERSIONS`, and it has to say what the claim
//! rests on, so a reader can check it without leaving the repository's pointers.

use std::path::Path;

fn doc() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/vector-providers.md"))
        .unwrap()
}

/// The sentence that states the range.
fn versions_sentence(doc: &str) -> String {
    let start = doc
        .find("Supported DWG versions")
        .expect("the provider note no longer states the supported DWG versions");
    doc[start..]
        .split("\n\n")
        .next()
        .unwrap()
        .replace('\n', " ")
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_note_states_the_range_the_probe_accepts() {
    let s = versions_sentence(&doc());
    assert!(s.contains("AC1014 through AC1032"), "{s}");
    assert!(s.contains("R14") && s.contains("2018"), "{s}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_note_says_what_the_claim_rests_on() {
    let s = versions_sentence(&doc());
    // The probe side is checkable in this repository. The decode side lives in
    // the fixtures of acadsharp-rs-tests, one drawing per stamp, and a claim
    // nobody can trace is the one the reviewer could not verify.
    assert!(s.contains("SUPPORTED_DWG_VERSIONS"), "{s}");
    assert!(s.contains("acadsharp-rs-tests"), "{s}");
    assert!(s.contains("one drawing per version"), "{s}");
}

#[cfg(feature = "dwg-acadsharp")]
#[test]
fn the_note_and_the_list_in_code_name_the_same_ends() {
    use libviprs::cad::acadsharp::{SUPPORTED_DWG_VERSIONS, is_supported_version};
    let s = versions_sentence(&doc());
    let first = std::str::from_utf8(SUPPORTED_DWG_VERSIONS[0]).unwrap();
    let last =
        std::str::from_utf8(SUPPORTED_DWG_VERSIONS[SUPPORTED_DWG_VERSIONS.len() - 1]).unwrap();
    assert!(
        s.contains(&format!("{first} through {last}")),
        "the doc says one range and the code accepts {first} to {last}: {s}"
    );
    assert_eq!(SUPPORTED_DWG_VERSIONS.len(), 7);
    assert!(SUPPORTED_DWG_VERSIONS.windows(2).all(|w| w[0] < w[1]));
    for stamp in SUPPORTED_DWG_VERSIONS {
        assert!(is_supported_version(stamp.as_slice()));
    }
    // One past each end is not accepted, so the range is not wider than said.
    assert!(!is_supported_version(b"AC1012"));
    assert!(!is_supported_version(b"AC1100"));
}
