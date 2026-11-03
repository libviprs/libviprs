//! Issue #1366 (RL2.1 follow-up 5/6): the docs say which codecs decode, behind which feature and
//! with which reason keys, and none of them says pdfium decodes images.

use libviprs::raster_pvec::rules::RASTER_PRODUCER_RULES;

const PRODUCER: &str = include_str!("../docs/pvec-raster-producer.md");
const RASTER: &str = include_str!("../docs/pvec-raster.md");
const ADR: &str = include_str!("../docs/adr/pvec-pdf-policy.md");

fn rule(id: &str) -> &'static str {
    RASTER_PRODUCER_RULES
        .iter()
        .find(|r| r.id == id)
        .unwrap_or_else(|| panic!("no rule named {id}"))
        .rule
}

#[test]
fn the_bake_rule_no_longer_calls_the_three_codecs_refused() {
    let text = rule("pdf-colour-and-mask-bake");
    assert!(
        !text.contains("`JPXDecode`, `JBIG2Decode`, `CCITTFaxDecode`, `/Separation`"),
        "the rule still lists the three codecs as refused by name: {text}"
    );
    for key in [
        "jbig2",
        "filter-not-enabled",
        "unsupported-filter",
        "decode-failed",
    ] {
        assert!(text.contains(key), "the rule does not name `{key}`: {text}");
    }
}

#[test]
fn the_producer_page_tells_the_per_codec_story() {
    for stale in [
        "is still to land",
        "is not decided yet",
        "through pdfium, or through small",
    ] {
        assert!(
            !PRODUCER.contains(stale),
            "the producer page still says `{stale}`"
        );
    }
    for want in [
        "`jbig2` feature",
        "`jp2k` feature",
        "hayro-jbig2",
        "hayro-ccitt",
        "option E",
    ] {
        assert!(
            PRODUCER.contains(want),
            "the producer page does not mention {want}"
        );
    }
}

#[test]
fn the_producer_page_has_the_corpus_table() {
    for want in [
        "| Filter | Files | Images | Size class |",
        "| `JBIG2Decode` | 0 |",
        "| `CCITTFaxDecode` | 1 | 7 |",
        "| `JPXDecode` | 2 | 27 |",
        "| `DCTDecode` | 12 | 135 |",
    ] {
        assert!(PRODUCER.contains(want), "the corpus table lacks `{want}`");
    }
}

#[test]
fn the_rules_page_does_not_list_the_codecs_as_refused() {
    assert!(
        !RASTER.contains("`JPXDecode`, `JBIG2Decode` and `CCITTFaxDecode`, `/Separation`"),
        "docs/pvec-raster.md still lists the three codecs as refused by name"
    );
}

#[test]
fn the_adr_answers_open_question_5() {
    for want in [
        "## Image codecs (2026-11-01, issue #1366)",
        "open question 5",
        "pdfium is not required",
        "no raw-stream decode call",
        "wasm32",
        "chosen by feature",
    ] {
        assert!(ADR.contains(want), "the ADR lacks `{want}`");
    }
}

#[test]
fn no_page_says_pdfium_decodes_images() {
    for (name, doc) in [("producer", PRODUCER), ("rules", RASTER), ("adr", ADR)] {
        for line in doc.lines() {
            let l = line.to_lowercase();
            assert!(
                !(l.contains("through pdfium") || l.contains("pdfium decodes")),
                "the {name} page says pdfium decodes images: {line}"
            );
        }
    }
}
