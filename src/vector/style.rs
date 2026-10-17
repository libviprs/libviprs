//! The fixed black-on-white MapLibre style for a libviprs vector archive.
//!
//! One style, three layers: a white paper background, the `lines` layer drawn
//! in black, and the `text` layer drawn in black. There is no colour and no
//! theme, because the point of the product is a drawing you can read.
//!
//! # Only properties every MapLibre client implements
//!
//! The style uses layout and paint properties that MapLibre GL JS, MapLibre
//! Native for Android and MapLibre Native for iOS all support, and nothing
//! else. In particular text overlap is `text-allow-overlap: true`. The newer
//! `text-overlap: always` is a GL JS property the Native clients do not
//! implement, so a style that used it would draw on the web and drop labels on
//! a phone, and would have to be rewritten in L2.4.
//! `tests/vector_1033_style.rs` holds the allowlist and fails on anything
//! outside it.
//!
//! # Text size follows the drawing
//!
//! A label is `height` drawing units tall, so it has to be `height` units tall
//! in pixels at every zoom. The tiler writes `h0`, the label's height in
//! pixels at zoom 0 for 512 px tiles, and the style interpolates it with an
//! exponential base 2 between zoom 0 and zoom 24, which is exactly
//! `h0 * 2^zoom` (the test pins the identity).

use serde_json::{Value, json};

const TEMPLATE: &str = include_str!("black_on_white.style.json");

/// The style with the archive and the glyph server filled in.
///
/// `archive_url` is the URL of the `.pmtiles` file, without the `pmtiles://`
/// scheme: the style adds it. `glyphs_url` is a MapLibre glyph template and
/// must contain `{fontstack}` and `{range}`; the style asks for the font stack
/// `Noto Sans Regular`.
///
/// # Panics
///
/// Never for a valid template, which the tests check. The template is a
/// compile-time constant.
#[must_use]
pub fn black_on_white_style(archive_url: &str, glyphs_url: &str) -> Value {
    let mut style: Value = serde_json::from_str(TEMPLATE).expect("the bundled style is valid JSON");
    style["sources"]["drawing"]["url"] = json!(format!("pmtiles://{archive_url}"));
    style["glyphs"] = json!(glyphs_url);
    style
}

/// The template as shipped, with its two placeholders still in it.
#[must_use]
pub const fn black_on_white_template() -> &'static str {
    TEMPLATE
}

/// The code point ranges the viewer's glyph files cover, inclusive: Latin
/// (0 to 511), general punctuation, letterlike symbols and mathematical
/// operators (8192 to 8959, which holds the diameter sign U+2205).
///
/// `web/vector-viewer/public/fonts/` holds one 256-code-point file per slice of
/// these, and `tests/vector_1034_fidelity.rs` checks that the two agree.
pub const SERVED_GLYPH_RANGES: [(u32, u32); 2] = [(0, 511), (8192, 8959)];

/// Whether the served glyph files can draw `c`.
#[must_use]
pub fn glyph_available(c: char) -> bool {
    let n = u32::from(c);
    SERVED_GLYPH_RANGES
        .iter()
        .any(|&(lo, hi)| (lo..=hi).contains(&n))
}
