//! Issue #1333 (RL2.2): `docs/pvec-raster.md` is the producer rules page, and the table in it is
//! rendered from `RASTER_PRODUCER_RULES`, so a rule that changes in the code changes the page or
//! fails here. Regenerate with `PVEC_RASTER_RULES_BLESS=1 cargo test --test pvec_1333_raster_rules_doc`.

use libviprs::raster_pvec::rules::{RASTER_PRODUCER_RULES, render_rules_table};

const DOC: &str = include_str!("../docs/pvec-raster.md");
const BEGIN: &str = "<!-- producer-rules:begin -->";
const END: &str = "<!-- producer-rules:end -->";

/// The block between the markers, without them.
fn block(doc: &str) -> &str {
    let a = doc.find(BEGIN).expect("begin marker") + BEGIN.len();
    let b = doc.find(END).expect("end marker");
    doc[a..b].trim_matches('\n')
}

#[test]
fn the_table_in_the_page_is_the_rendered_constant_byte_for_byte() {
    if std::env::var_os("PVEC_RASTER_RULES_BLESS").is_some() {
        let a = DOC.find(BEGIN).unwrap() + BEGIN.len();
        let b = DOC.find(END).unwrap();
        let blessed = format!("{}\n{}\n{}", &DOC[..a], render_rules_table(), &DOC[b..]);
        std::fs::write(
            concat!(env!("CARGO_MANIFEST_DIR"), "/docs/pvec-raster.md"),
            blessed,
        )
        .unwrap();
        return;
    }
    assert_eq!(
        block(DOC),
        render_rules_table().trim_matches('\n'),
        "docs/pvec-raster.md is generated between its markers: regenerate it with \
         `PVEC_RASTER_RULES_BLESS=1 cargo test --test pvec_1333_raster_rules_doc`"
    );
}

#[test]
fn a_hand_edit_of_the_table_is_caught() {
    let edited = block(DOC).replacen("Google", "Googol", 1);
    assert_ne!(edited, render_rules_table().trim_matches('\n'));
}

#[test]
fn every_rule_the_adr_puts_on_libviprs_is_a_row() {
    let ids: Vec<&str> = RASTER_PRODUCER_RULES.iter().map(|r| r.id).collect();
    for want in [
        "centring-off",
        "overlap-0",
        "google-tiles-past-the-grid-dropped",
        "skip-blanks-not-stored",
        "placeholder-refused",
        "raw-refused",
        "bit-depth-16-converted",
        "palette-converted",
        "legal-codec-and-format-pairs",
        "tile-edge",
        "limit-15",
        "no-interlace",
        "raster-id-draw-order",
        "negative-zero-normalised",
        "attribution-plain-text",
        "source-digest-sha-256",
        "placement-np-19",
        "geo-transform-permutation",
        "pixel-origin-corner",
        "composition-precondition",
        "geo-transform-in-pmtiles-metadata",
        "clip-baked-into-alpha",
        "clipped-image-lossless-tiles",
        "pdf-colour-and-mask-bake",
        "band-from-paint-order",
        "dwg-image-is-a-frame",
        "nothing-dropped-silently",
    ] {
        assert!(ids.contains(&want), "no rule named {want}");
    }
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "rule ids are unique");
}

#[test]
fn every_rule_names_the_adr_item_or_the_code_it_comes_from() {
    assert!(!RASTER_PRODUCER_RULES.is_empty());
    for r in RASTER_PRODUCER_RULES {
        let named = ["ADR 0018", "RASTER-0", "LIMIT-15", "NP-19", "NP-03"]
            .iter()
            .any(|needle| r.source.contains(needle));
        assert!(named, "{} names no ADR item or code: {:?}", r.id, r.source);
        assert!(!r.rule.is_empty());
        assert!(!r.rule.contains('|') && !r.source.contains('|'), "{}", r.id);
    }
}

#[test]
fn the_page_names_the_three_commands() {
    for cmd in [
        "viprs pyramid --storage pvec",
        "viprs pmtiles to-pvec",
        "viprs pvec rasters",
    ] {
        assert!(DOC.contains(cmd), "the page does not name `{cmd}`");
    }
}

#[test]
fn the_page_points_at_the_adr_and_every_relative_link_resolves() {
    assert!(DOC.contains("docs/adr/0018-raster-in-pvec.md"));
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs");
    let mut checked = 0;
    for part in DOC.split("](").skip(1) {
        let target = part.split(')').next().unwrap();
        if target.starts_with("http") || target.starts_with('#') {
            continue;
        }
        let path = target.split('#').next().unwrap();
        assert!(
            root.join(path).exists(),
            "the page links to {path}, which is not there"
        );
        checked += 1;
    }
    assert!(checked >= 2, "only {checked} relative links checked");
}
